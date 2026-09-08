/**
@file sbd/chemistry/gdb/pt2.h
@brief Epstein-Nesbet second-order perturbative correction on top of a converged
       GDB diagonalization.

WHAT THIS IS AND WHY IT IS SEPARATE
-----------------------------------
The variational energy of a selected-CI calculation is an upper bound: it is above
the true energy by whatever the omitted determinants are worth. For a determinant
space that comes from a quantum circuit the omission is not a choice -- the space is
fixed by sampling -- so the gap is not something we can close by selecting better.
The standard estimate of it is second-order perturbation theory,

    E_PT2 = sum_a |sum_i H_ai c_i|^2 / (E_0 - H_aa)                          (1)

with `i` running over the variational determinants and `a` over determinants outside
that space which are connected to it. `E_var + E_PT2` is the number selected-CI
codes report.

This lives in its own header, and is driven by its own app
(`apps/chemistry_gdb_pt2`), reading a wavefunction that a previous run wrote. That
is deliberate: the variational solver is cluster-validated and has just been through
a long bug hunt, so the correction is additive by construction and cannot
destabilise it. It also means `epsilon2` can be re-swept without redoing the solve,
which is the normal way this quantity is used.

TWO VARIANTS
------------
`Variant::Determinant` evaluates (1) as written, over determinant perturbers. It is
directly comparable to published SHCI numbers, and that is its purpose -- it is the
only part of this file with an external oracle.

`Variant::SpinPure` replaces the perturber index by a target-S configuration state
function. This matters more than it first appears. Even a perfectly spin-pure
reference gives a spin-contaminated first-order wavefunction in a determinant basis,
because determinants within one spin-coupled configuration have *different* diagonal
energies and therefore different denominators in (1); the first-order coefficients
then no longer respect the spin coupling. The contamination is intrinsic to
Epstein-Nesbet PT2 in a determinant basis and does NOT vanish as epsilon2 -> 0.
Giving each CSF a single denominator <CSF|H|CSF> removes it structurally, so only
this variant may be added to a single-spin variational energy and quoted as one.

RELATION TO OTHER CODE
----------------------
Nothing here modifies the solver. It reuses, and does not reimplement:
`Hij`/`ZeroExcite` (chemistry/basic/determinants.h) for matrix elements, and
`build_config_projector` (chemistry/gdb/single_spin.h) for the CSF blocks of the
spin-pure variant -- that function groups an arbitrary determinant set by spatial
configuration on its own, so the perturber CSFs are built by the same code that
builds the variational ones and cannot drift from it.

The algorithm follows Sharma, Holmes, Jeanmairet, Alavi, Umrigar,
JCTC 13, 1595 (2017). Their implementation (Dice) is GPL-3.0 and was read as a
specification only; no code is derived from it.
*/
#ifndef SBD_CHEMISTRY_GDB_PT2_H
#define SBD_CHEMISTRY_GDB_PT2_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace sbd {

  namespace gdb {

    /// Which perturber space to sum over. See the header comment: these are not
    /// two approximations of one quantity, they are two different quantities, and
    /// only SpinPure is a target-S energy.
    enum class PT2Variant { Determinant, SpinPure };

    inline const char * pt2_variant_name(PT2Variant v) {
      return v == PT2Variant::SpinPure ? "spin-pure (CSF perturbers)"
                                       : "determinant";
    }

    /// Everything the PT2 app is told on the command line.
    struct PT2Options {
      std::string fcidumpfile;          ///< integrals; same file the solve used
      std::vector<std::string> detfiles;///< variational determinant list
      std::string loadname;             ///< wavefunction: flat binary, or shard prefix
      double epsilon2   = 1.0e-8;       ///< screening: keep |H_ai c_i| > epsilon2
      double e0         = 0.0;          ///< variational energy E_0 (electronic)
      bool   e0_given   = false;        ///< false => must be read from the wavefunction file
      PT2Variant variant = PT2Variant::Determinant;
      int    multiplicity = -1;         ///< 2S+1 for the spin-pure variant; <1 = off
      size_t bit_length = 20;           ///< MUST match the solve that wrote the wavefunction
      int    b_comm_size = 1;
      int    t_comm_size = 1;
      size_t batch_size = 200000;       ///< references per batch; bounds perturber memory
      /// Floor on |E_0 - H_aa|. Dice applies none, so a perturber that happens to be
      /// near-degenerate with E_0 produces an unbounded contribution. The floor keeps
      /// the sign (flipping it would flip the sign of that term's contribution) and
      /// the number of times it triggers is reported: a nonzero count means the
      /// result is being held up by regularization and should not be trusted quietly.
      double den_floor  = 1.0e-8;
    };

    /// Parse the PT2 command line. Unknown arguments are ignored, matching the
    /// convention of the other apps in this tree.
    inline PT2Options parse_pt2_options(int argc, char * argv[]) {
      PT2Options o;
      for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--fcidump"   && i + 1 < argc) { o.fcidumpfile = argv[++i]; }
        else if (a == "--loadname" && i + 1 < argc) { o.loadname = argv[++i]; }
        else if (a == "--epsilon2" && i + 1 < argc) { o.epsilon2 = std::atof(argv[++i]); }
        else if (a == "--e0" && i + 1 < argc) {
          o.e0 = std::atof(argv[++i]);
          o.e0_given = true;
        }
        else if (a == "--single_spin" && i + 1 < argc) { o.multiplicity = std::atoi(argv[++i]); }
        else if (a == "--bit_length"  && i + 1 < argc) { o.bit_length = std::atoi(argv[++i]); }
        else if (a == "--b_comm_size" && i + 1 < argc) { o.b_comm_size = std::atoi(argv[++i]); }
        else if (a == "--t_comm_size" && i + 1 < argc) { o.t_comm_size = std::atoi(argv[++i]); }
        else if (a == "--batch_size"  && i + 1 < argc) { o.batch_size = std::atoll(argv[++i]); }
        else if (a == "--den_floor"   && i + 1 < argc) { o.den_floor = std::atof(argv[++i]); }
        else if (a == "--variant" && i + 1 < argc) {
          const std::string v(argv[++i]);
          if (v == "a" || v == "det" || v == "determinant") {
            o.variant = PT2Variant::Determinant;
          } else if (v == "c" || v == "csf" || v == "spinpure" || v == "spin-pure") {
            o.variant = PT2Variant::SpinPure;
          } else {
            std::cerr << " sbd: ERROR --variant expects a|det|determinant or"
                      << " c|csf|spin-pure, got '" << v << "'." << std::endl;
            std::exit(2);
          }
        }
        else if (a == "--detfiles" && i + 1 < argc) {
          const std::string list(argv[++i]);
          size_t pos = 0;
          while (pos <= list.size()) {
            const size_t comma = list.find(',', pos);
            const size_t end = (comma == std::string::npos) ? list.size() : comma;
            if (end > pos) o.detfiles.push_back(list.substr(pos, end - pos));
            if (comma == std::string::npos) break;
            pos = comma + 1;
          }
        }
      }
      return o;
    }

    /// Reject a command line that cannot produce a meaningful number, before any
    /// work is done. Returns false after explaining; the caller aborts.
    inline bool validate_pt2_options(const PT2Options & o, int mpi_size, int mpi_rank) {
      const char * why = nullptr;
      if (o.fcidumpfile.empty())      why = "--fcidump is required";
      else if (o.detfiles.empty())    why = "--detfiles is required (the variational space)";
      else if (o.loadname.empty())    why = "--loadname is required (the converged wavefunction)";
      else if (!(o.epsilon2 > 0.0))   why = "--epsilon2 must be positive";
      else if (o.variant == PT2Variant::SpinPure && o.multiplicity < 1)
        why = "--variant c requires --single_spin <2S+1>: the CSF perturber space is"
              " defined by a target spin, so there is no default";
      else if (o.b_comm_size < 1 || o.t_comm_size < 1)
        why = "--b_comm_size and --t_comm_size must be >= 1";
      else if (o.b_comm_size * o.t_comm_size > mpi_size)
        why = "b_comm_size * t_comm_size exceeds the rank count";
      else if (mpi_size % (o.b_comm_size * o.t_comm_size) != 0)
        why = "b_comm_size * t_comm_size must divide the rank count";
      else if (o.t_comm_size > o.b_comm_size)
        /// Same constraint the solver enforces: t_comm partitions exactly
        /// b_comm_size ring-rotation tasks, so surplus t ranks get an empty range.
        why = "t_comm_size must not exceed b_comm_size";
      if (why == nullptr) return true;
      if (mpi_rank == 0) {
        std::cerr << " sbd: ERROR pt2: " << why << "." << std::endl;
      }
      return false;
    }

    inline void cout_pt2_options(const PT2Options & o) {
      std::cout << "# --- pt2 -------------------------------------------------\n"
                << "# variant:       " << pt2_variant_name(o.variant) << "\n"
                << "# epsilon2:      " << o.epsilon2 << "\n"
                << "# den_floor:     " << o.den_floor << "\n"
                << "# multiplicity:  " << o.multiplicity << "\n"
                << "# bit_length:    " << o.bit_length << "\n"
                << "# b_comm_size:   " << o.b_comm_size << "\n"
                << "# t_comm_size:   " << o.t_comm_size << "\n"
                << "# batch_size:    " << o.batch_size << "\n"
                << "# loadname:      " << o.loadname << "\n"
                << "# fcidump:       " << o.fcidumpfile << "\n"
                << "# detfiles:      ";
      for (size_t i = 0; i < o.detfiles.size(); ++i) {
        std::cout << (i ? "," : "") << o.detfiles[i];
      }
      std::cout << "\n";
      if (o.e0_given) std::cout << "# E_0 (given):   " << o.e0 << "\n";
      else            std::cout << "# E_0:           from the wavefunction file\n";
      std::cout << "# --------------------------------------------------------"
                << std::endl;
    }


    // ======================================================================
    // Reading the converged wavefunction
    // ======================================================================
    //
    // Two input formats, both self-describing, and NEITHER goes through
    // LoadWavefunction (caop/basic/restart.h:52). That function binary-searches a
    // caller-supplied basis and SILENTLY ZEROES any amplitude it cannot match, then
    // normalizes -- so a determinant-list mismatch leaves no trace at all. For a
    // post-processing step whose whole job is to add a small correction to someone
    // else's converged energy, a silently truncated wavefunction is the worst
    // possible failure mode: the PT2 number would look entirely reasonable. Both
    // readers below therefore count what they matched and the caller aborts on a
    // shortfall.
    //
    // Format 1, PREFERRED -- flat binary, converted from the merged .npz that the
    // Python wrapper writes (`experimental_SCIState`). This is the format to rely on
    // in production, because the per-rank shards are deleted once the merge verifies
    // (`delete_shards=True`), precisely to keep the footprint down; depending on
    // them would make PT2 unusable in the configuration actually used. The .npz is
    // also read back and compared before those shards are removed, so a truncated
    // file cannot pass unnoticed.
    //
    //   magic   uint64  0x5342445057463031  ("SBDPWF01")
    //   ndet    uint64
    //   nword   uint64  words per determinant (must equal det_vector row size)
    //   e0      float64 the variational energy this wavefunction converged to
    //   dets    uint64[ndet][nword]
    //   amps    float64[ndet]
    //
    // Format 2, FALLBACK -- the raw per-b_comm-rank shards written by
    // SaveWavefunction (caop/basic/restart.h:24), for when PT2 runs straight after a
    // solve that has not been merged. Per file: two uint64 (basis_size,
    // basis_length), then the determinants, then the amplitudes. No E_0 is stored,
    // so --e0 must be supplied.

    /// Magic for the flat format. Spells "SBDPWF01" so a wrong file is named as such
    /// rather than misparsed into plausible garbage.
    inline constexpr uint64_t pt2_wf_magic() { return 0x5342445057463031ULL; }

    /// One determinant plus its amplitude, as read from disk.
    template <typename ElemT>
    struct PT2Wavefunction {
      std::vector<std::vector<size_t>> dets;  ///< determinant words, as stored
      std::vector<ElemT> amps;                ///< matching amplitudes
      double e0 = 0.0;                        ///< variational energy, if the file carried one
      bool   e0_from_file = false;
    };

    /// Read the flat binary. Returns false (with a diagnostic) if the file is
    /// missing, has the wrong magic, or is truncated -- never a partial result.
    template <typename ElemT>
    bool read_pt2_wavefunction_flat(const std::string & path,
                                    size_t nword_expected,
                                    PT2Wavefunction<ElemT> & wf) {
      std::ifstream ifs(path, std::ios::binary);
      if (!ifs.is_open()) return false;

      uint64_t magic = 0, ndet = 0, nword = 0;
      double e0 = 0.0;
      ifs.read(reinterpret_cast<char *>(&magic), sizeof(uint64_t));
      ifs.read(reinterpret_cast<char *>(&ndet),  sizeof(uint64_t));
      ifs.read(reinterpret_cast<char *>(&nword), sizeof(uint64_t));
      ifs.read(reinterpret_cast<char *>(&e0),    sizeof(double));
      if (!ifs) {
        std::cerr << " sbd: ERROR pt2: " << path << " is shorter than its header."
                  << std::endl;
        return false;
      }
      if (magic != pt2_wf_magic()) {
        std::cerr << " sbd: ERROR pt2: " << path << " is not an SBD PT2 wavefunction"
                  << " (magic 0x" << std::hex << magic << std::dec << ")."
                  << std::endl;
        return false;
      }
      if (nword != nword_expected) {
        // A bit_length mismatch is the likely cause, and it would otherwise parse
        // into determinants that are wrong but well-formed.
        std::cerr << " sbd: ERROR pt2: " << path << " stores " << nword
                  << " words per determinant but this run expects " << nword_expected
                  << ". Pass the same --bit_length the solve used." << std::endl;
        return false;
      }

      wf.dets.assign(static_cast<size_t>(ndet), std::vector<size_t>(nword, 0));
      wf.amps.assign(static_cast<size_t>(ndet), ElemT(0.0));
      for (size_t i = 0; i < ndet; ++i) {
        ifs.read(reinterpret_cast<char *>(wf.dets[i].data()),
                 sizeof(size_t) * nword);
      }
      ifs.read(reinterpret_cast<char *>(wf.amps.data()), sizeof(ElemT) * ndet);
      if (!ifs) {
        std::cerr << " sbd: ERROR pt2: " << path << " is truncated: expected "
                  << ndet << " determinants and amplitudes." << std::endl;
        return false;
      }
      wf.e0 = e0;
      wf.e0_from_file = true;
      return true;
    }

    /// Read the per-rank shards `prefix000000.bin`, `prefix000001.bin`, ... until one
    /// is missing. Concatenates them; the shard set is a partition of the space, so
    /// no merging is needed. `nshard` reports how many were consumed.
    template <typename ElemT>
    bool read_pt2_wavefunction_shards(const std::string & prefix,
                                      size_t nword_expected,
                                      PT2Wavefunction<ElemT> & wf,
                                      int & nshard) {
      nshard = 0;
      for (int rank = 0;; ++rank) {
        const std::string path = sbd::statefilename(prefix, rank);
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) break;

        size_t basis_size = 0, basis_length = 0;
        ifs.read(reinterpret_cast<char *>(&basis_size),   sizeof(size_t));
        ifs.read(reinterpret_cast<char *>(&basis_length), sizeof(size_t));
        if (!ifs) {
          std::cerr << " sbd: ERROR pt2: shard " << path << " has no header."
                    << std::endl;
          return false;
        }
        if (basis_length != nword_expected) {
          std::cerr << " sbd: ERROR pt2: shard " << path << " stores "
                    << basis_length << " words per determinant, expected "
                    << nword_expected << ". Check --bit_length." << std::endl;
          return false;
        }
        const size_t base = wf.dets.size();
        wf.dets.resize(base + basis_size, std::vector<size_t>(basis_length, 0));
        for (size_t i = 0; i < basis_size; ++i) {
          ifs.read(reinterpret_cast<char *>(wf.dets[base + i].data()),
                   sizeof(size_t) * basis_length);
        }
        std::vector<ElemT> shard_amps(basis_size, ElemT(0.0));
        ifs.read(reinterpret_cast<char *>(shard_amps.data()),
                 sizeof(ElemT) * basis_size);
        if (!ifs) {
          std::cerr << " sbd: ERROR pt2: shard " << path << " is truncated."
                    << std::endl;
          return false;
        }
        wf.amps.insert(wf.amps.end(), shard_amps.begin(), shard_amps.end());
        ++nshard;
      }
      if (nshard == 0) {
        std::cerr << " sbd: ERROR pt2: no shard found at "
                  << sbd::statefilename(prefix, 0) << "." << std::endl;
        return false;
      }
      wf.e0_from_file = false;   // shards carry no energy
      return true;
    }

    /// Align a wavefunction read from disk onto this rank's slice of the variational
    /// determinant list, returning the amplitude vector indexed like `det`.
    ///
    /// `n_matched` counts determinants placed on THIS rank; the caller must reduce it
    /// over the whole run and compare against the file's determinant count. That
    /// check is the point of this function: an unmatched determinant means the
    /// wavefunction and the --detfiles list disagree, and PT2 would otherwise
    /// proceed with a quietly incomplete reference state.
    template <typename ElemT, typename DetsContainer>
    void align_pt2_wavefunction(const PT2Wavefunction<ElemT> & wf,
                                const DetsContainer & det,
                                std::vector<ElemT> & c,
                                size_t & n_matched) {
      c.assign(det.size(), ElemT(0.0));
      n_matched = 0;
      // `det` is in canonical less_from_back order (sort_bitarray), so binary search
      // is valid. The file's order is whatever the writer used and is not assumed.
      auto cmp = [](const auto & x, const auto & y) {
        return sbd::less_from_back(x, y);
      };
      for (size_t i = 0; i < wf.dets.size(); ++i) {
        auto it = std::lower_bound(det.begin(), det.end(), wf.dets[i], cmp);
        if (it == det.end()) continue;
        const std::vector<size_t> found = *it;
        if (found != wf.dets[i]) continue;
        c[static_cast<size_t>(it - det.begin())] = wf.amps[i];
        ++n_matched;
      }
    }


    // ======================================================================
    // Perturber generation
    // ======================================================================
    //
    // For each reference determinant |D_i> with coefficient c_i, emit every
    // determinant |D_a> reachable by a single or double excitation whose
    // contribution passes the screening test
    //
    //     |H_ai * c_i| > epsilon2 .
    //
    // Following Dice, the threshold is divided by |c_i| ONCE per reference and the
    // test is then applied to the matrix element alone. That is not a
    // micro-optimization: it is what allows a screening decision to be made before
    // the perturber determinant is constructed, and it means a large-weight
    // reference automatically gets a looser per-element threshold.
    //
    // Sz is preserved by construction: an excitation moves an electron from an
    // occupied spin-orbital to an empty one OF THE SAME SPIN, so the alpha and beta
    // counts are individually conserved. Nothing here needs to check that.
    //
    // WHAT THIS DOES NOT DO. Dice screens its doubles against a table of two-electron
    // integrals pre-sorted in descending magnitude, so its inner loop breaks at the
    // first element below threshold and the cost of generation is proportional to the
    // number of perturbers PRODUCED rather than to the size of the orbital space. SBD
    // has no such table (`makeHeatbathLookup` in expansion.h is exhaustive
    // enumeration, not magnitude-sorted screening). This implementation therefore
    // evaluates H_ai for every candidate and tests it, which is correct but costs
    // O(n_occ^2 * n_virt^2) per reference regardless of how many perturbers survive.
    // Building the sorted table is the obvious optimization if generation turns out
    // to dominate; it is deliberately not done first, because a correct slow number
    // is worth more than a fast unverified one.

    /// One emitted perturber: the determinant, its numerator contribution
    /// H_ai * c_i, and its diagonal energy H_aa.
    ///
    /// `num` is a CONTRIBUTION, not the numerator: the same determinant reached from
    /// several references contributes several of these, and they must be summed before
    /// squaring. That summation happens in the merge step, not here.
    template <typename ElemT>
    struct PT2Perturber {
      std::vector<size_t> det;
      ElemT  num = ElemT(0.0);
      double haa = 0.0;
    };

    /// Scratch buffers reused across references, so the inner loops allocate nothing.
    /// `Hij`'s convenient overload takes std::vector<size_t> and det_vector::row
    /// converts implicitly -- which allocates two vectors per call. In a loop that
    /// calls Hij O(n_occ^2 n_virt^2) times per reference that allocation dominates, so
    /// the scratch-buffer overload (determinants.h:568) is used with these instead.
    struct PT2Scratch {
      std::vector<int> open;      ///< empty spin-orbitals of the reference
      std::vector<int> closed;    ///< occupied spin-orbitals of the reference
      std::vector<int> c;         ///< Hij scratch
      std::vector<int> d;         ///< Hij scratch
      std::vector<size_t> cand;   ///< candidate determinant being built
    };

    /// Generate the perturbers of one reference determinant, appending to `out`.
    ///
    /// @param ref      the reference determinant, as stored (interleaved, 2L bits)
    /// @param ci       its coefficient
    /// @param eps_over_ci  epsilon2 / |c_i|; the test applied to |H_ai|
    /// @param L        spatial orbitals (so 2L spin-orbitals)
    template <typename ElemT>
    void generate_perturbers_from(const std::vector<size_t> & ref,
                                  const ElemT & ci,
                                  double eps_over_ci,
                                  size_t bit_length,
                                  size_t L,
                                  const ElemT & I0,
                                  const oneInt<ElemT> & I1,
                                  const twoInt<ElemT> & I2,
                                  PT2Scratch & s,
                                  std::vector<PT2Perturber<ElemT>> & out) {
      const int nso = static_cast<int>(2 * L);   // spin-orbitals
      const int nocc = sbd::bitcount(ref, bit_length, nso);
      const int nvir = nso - nocc;
      if (nocc == 0 || nvir == 0) return;

      // getOpenClosed writes through .at() and does NOT resize (its resize calls are
      // commented out at determinants.h:216-217), so pre-sizing is mandatory.
      s.closed.assign(static_cast<size_t>(nocc), 0);
      s.open.assign(static_cast<size_t>(nvir), 0);
      sbd::getOpenClosed(ref, bit_length, nso, s.open, s.closed);

      // Hij's scratch overload (determinants.h:568) writes c[nc] / d[nd] with RAW
      // operator[], no resize and no bounds check -- an empty scratch vector is a
      // segfault, which is exactly what an untested first version produced here. A
      // single or double excitation differs in at most 2 spin-orbitals per side, but
      // size to nso: the cost is nothing and it cannot be too small for any
      // determinant pair, including ones a future caller might pass.
      if (s.c.size() < static_cast<size_t>(nso)) s.c.assign(nso, 0);
      if (s.d.size() < static_cast<size_t>(nso)) s.d.assign(nso, 0);

      size_t orbDiff = 0;

      // ---- singles: i -> a, same spin (parity of the spin-orbital index) --------
      for (int ii = 0; ii < nocc; ++ii) {
        const int i = s.closed[ii];
        for (int aa = 0; aa < nvir; ++aa) {
          const int a = s.open[aa];
          if ((i & 1) != (a & 1)) continue;      // spin-flip: Sz would change
          s.cand = ref;
          sbd::setocc(s.cand, bit_length, i, false);
          sbd::setocc(s.cand, bit_length, a, true);
          const ElemT h = sbd::Hij(ref, s.cand, bit_length, L, s.c, s.d,
                                   I0, I1, I2, orbDiff);
          if (std::abs(h) <= eps_over_ci) continue;
          PT2Perturber<ElemT> p;
          p.det = s.cand;
          p.num = h * ci;
          p.haa = static_cast<double>(std::real(
              std::complex<double>(sbd::ZeroExcite(s.cand, bit_length, L, I0, I1, I2))));
          out.push_back(std::move(p));
        }
      }

      // ---- doubles: (i,j) -> (a,b), each replacement spin-preserving ------------
      for (int ii = 0; ii < nocc; ++ii) {
        const int i = s.closed[ii];
        for (int jj = ii + 1; jj < nocc; ++jj) {
          const int j = s.closed[jj];
          for (int aa = 0; aa < nvir; ++aa) {
            const int a = s.open[aa];
            for (int bb = aa + 1; bb < nvir; ++bb) {
              const int b = s.open[bb];
              // The pair of created spins must match the pair of annihilated spins,
              // in one assignment or the other. Checking the multiset rather than
              // (i,a) and (j,b) individually is what admits the ia-jb and ib-ja
              // pairings that a same-spin double legitimately has.
              const int spin_in  = (i & 1) + (j & 1);
              const int spin_out = (a & 1) + (b & 1);
              if (spin_in != spin_out) continue;
              s.cand = ref;
              sbd::setocc(s.cand, bit_length, i, false);
              sbd::setocc(s.cand, bit_length, j, false);
              sbd::setocc(s.cand, bit_length, a, true);
              sbd::setocc(s.cand, bit_length, b, true);
              const ElemT h = sbd::Hij(ref, s.cand, bit_length, L, s.c, s.d,
                                       I0, I1, I2, orbDiff);
              if (std::abs(h) <= eps_over_ci) continue;
              PT2Perturber<ElemT> p;
              p.det = s.cand;
              p.num = h * ci;
              p.haa = static_cast<double>(std::real(
                  std::complex<double>(sbd::ZeroExcite(s.cand, bit_length, L, I0, I1, I2))));
              out.push_back(std::move(p));
            }
          }
        }
      }
    }


    // ======================================================================
    // Merge, set-difference, and the energy
    // ======================================================================
    //
    // THE ONE THING THIS FILE MUST GET RIGHT. A perturber |D_a> reachable from
    // several references contributes
    //
    //     |sum_i H_ai c_i|^2       NOT       sum_i |H_ai c_i|^2 .
    //
    // Summing the squares instead of squaring the sum is a plausible-looking error
    // that changes the answer, does not crash, and does not obviously show up in an
    // epsilon2 sweep. So the contributions are accumulated LINEARLY here and squared
    // exactly once, per unique perturber, in the energy loop -- and there is a
    // dedicated test for it (tests/pt2/merge_properties.cc).
    //
    // Order of operations also matters. The set-difference against the variational
    // space assumes its input is already deduplicated: it advances past an equal
    // entry unconditionally, so a perturber duplicated in the list would have only
    // its first copy removed. Dedup FIRST, subtract second.

    /// A perturber after merging: one entry per distinct determinant, with the
    /// numerator contributions from every reference already summed.
    template <typename ElemT>
    struct PT2Merged {
      std::vector<size_t> det;
      ElemT  num = ElemT(0.0);   ///< sum_i H_ai c_i  -- squared later, once
      double haa = 0.0;
      int    n_parents = 0;      ///< how many contributions were summed (diagnostic)
    };

    /// Sort by determinant and combine duplicates, SUMMING their numerators.
    ///
    /// `haa` is a property of the perturber alone, so all copies must agree; a
    /// disagreement means two different determinants compared equal, which would be a
    /// defect in the ordering rather than a numerical issue. It is checked, not
    /// assumed.
    template <typename ElemT>
    void merge_pt2_perturbers(std::vector<PT2Perturber<ElemT>> & in,
                              std::vector<PT2Merged<ElemT>> & out,
                              double haa_tol = 1.0e-8) {
      out.clear();
      if (in.empty()) return;

      std::sort(in.begin(), in.end(),
                [](const PT2Perturber<ElemT> & x, const PT2Perturber<ElemT> & y) {
                  return sbd::less_from_back(x.det, y.det);
                });

      out.reserve(in.size());
      for (size_t i = 0; i < in.size(); ++i) {
        if (!out.empty() && out.back().det == in[i].det) {
          // THE accumulation. Linear, never squared here.
          out.back().num += in[i].num;
          out.back().n_parents += 1;
          if (std::abs(out.back().haa - in[i].haa) >
              haa_tol * std::max(1.0, std::abs(in[i].haa))) {
            std::cerr << " sbd: ERROR pt2: two perturbers compared equal but carry"
                      << " different diagonal energies (" << out.back().haa
                      << " vs " << in[i].haa << "). H_aa depends only on the"
                      << " determinant, so this indicates an ordering defect."
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 4);
          }
        } else {
          PT2Merged<ElemT> m;
          m.det = in[i].det;
          m.num = in[i].num;
          m.haa = in[i].haa;
          m.n_parents = 1;
          out.push_back(std::move(m));
        }
      }
    }

    /// Remove from `pert` every determinant present in `det`, in place.
    ///
    /// Those are not perturbers: their contribution is already in the variational
    /// energy, and including them would double-count it. `pert` must be deduplicated
    /// (see the note above) and `det` must be in canonical less_from_back order.
    /// Returns how many were removed.
    template <typename ElemT, typename DetsContainer>
    size_t remove_variational(std::vector<PT2Merged<ElemT>> & pert,
                              const DetsContainer & det) {
      if (pert.empty() || det.size() == 0) return 0;
      auto cmp = [](const auto & x, const auto & y) {
        return sbd::less_from_back(x, y);
      };
      size_t keep = 0, removed = 0;
      for (size_t i = 0; i < pert.size(); ++i) {
        auto it = std::lower_bound(det.begin(), det.end(), pert[i].det, cmp);
        bool present = false;
        if (it != det.end()) {
          const std::vector<size_t> found = *it;
          present = (found == pert[i].det);
        }
        if (present) { ++removed; continue; }
        if (keep != i) pert[keep] = std::move(pert[i]);
        ++keep;
      }
      pert.resize(keep);
      return removed;
    }

    /// Accumulated PT2 result, so the caller can report more than one number.
    struct PT2Result {
      double energy = 0.0;      ///< the correction; negative for a ground-state root
      double psi1_norm2 = 0.0;  ///< |Psi_1|^2, a measure of how perturbative this is
      size_t n_perturbers = 0;
      size_t n_floored = 0;     ///< how many denominators hit the floor
      double worst_ratio = 0.0; ///< largest |num/(E0-haa)|; a big value means trouble
    };

    /// Sum |num|^2 / (E_0 - H_aa) over the merged perturbers.
    ///
    /// The denominator is floored in magnitude WITH ITS SIGN PRESERVED. Dice applies
    /// no floor at all, so a perturber that happens to be near-degenerate with E_0
    /// contributes an unbounded term; flipping the sign instead of preserving it would
    /// turn a downward correction into an upward one. `n_floored` is reported because
    /// a nonzero count means the result is being held up by regularization, and that
    /// should be visible rather than silent.
    template <typename ElemT>
    PT2Result accumulate_pt2(const std::vector<PT2Merged<ElemT>> & pert,
                             double e0,
                             double den_floor) {
      PT2Result r;
      r.n_perturbers = pert.size();
      for (const auto & p : pert) {
        double den = e0 - p.haa;
        const double ad = std::abs(den);
        if (ad < den_floor) {
          den = (den < 0.0) ? -den_floor : den_floor;
          r.n_floored += 1;
        }
        const double n2 = std::norm(std::complex<double>(p.num));
        r.energy += n2 / den;
        const double coef = std::sqrt(n2) / std::abs(den);
        r.psi1_norm2 += coef * coef;
        r.worst_ratio = std::max(r.worst_ratio, coef);
      }
      return r;
    }


    // ======================================================================
    // Variant (c): configuration-driven spin-pure PT2
    // ======================================================================
    //
    // WHY THIS IS NOT A COSMETIC VARIANT OF (a)
    // -----------------------------------------
    // Write a perturbing configuration's Sz-orbit as determinants r = 1..d. Variant
    // (a) treats each r as an independent perturber:
    //
    //     E_a = sum_r |x_r|^2 / (E_0 - H_rr),     x_r = sum_i H_ri c_i .        (2)
    //
    // The d determinants of one spin-coupled configuration have DIFFERENT H_rr, so
    // (2) weights them by d different denominators. The resulting first-order vector
    // is therefore NOT a combination of target-S CSFs even when Psi_0 is perfectly
    // spin-pure: the relative weights inside the orbit are set by the denominators,
    // not by the spin coupling. That contamination is a property of Epstein-Nesbet
    // PT2 in a determinant basis and does not shrink as epsilon2 -> 0.
    //
    // Variant (c) resolves the orbit into its target-S CSFs first,
    //
    //     E_c = sum_c |num_c|^2 / (E_0 - h_cc),                                 (3)
    //     num_c = sum_r V_rc x_r ,     h_cc = sum_{r,r'} V_rc V_r'c H_{r r'} ,
    //
    // one denominator per CSF, so every term of (3) is a target-S state and the sum
    // may be added to a single-spin variational energy. Two consequences worth
    // stating because each is a way to get (3) wrong:
    //
    //   * h_cc needs the FULL intra-block double sum, r != r' included. Using only
    //     the diagonal r == r' is the identical mistake that once made the projected
    //     Davidson preconditioner drop intra-configuration exchange and converge to
    //     the right spin at a too-high energy.
    //
    //   * num_c needs x_r for EVERY row r of the completed orbit, not just the rows
    //     the generator emitted. A row whose |H_ri c_i| fell under the threshold, or
    //     which no reference reaches at all, still enters num_c through its V_rc, and
    //     silently treating it as zero would truncate the projection. So the orbit is
    //     completed and the missing rows' x_r are computed explicitly against the
    //     reference list. This is the perturber-space analogue of the variational
    //     requirement that a configuration be spin-complete, and it is why (c) is
    //     driven by configurations rather than by determinants.
    //
    // Configurations that carry no target-S CSF (n_csf == 0, e.g. a closed-shell
    // config when the target is a triplet) drop out of (3) entirely -- correctly:
    // they have no target-S component to perturb into. Their weight is what (a)
    // counts and (c) does not, and it is reported as `n_configs_no_target_s` so the
    // difference between the two variants is accountable rather than mysterious.

    /// A perturbing spatial configuration: its per-orbital occupation pattern, the
    /// completed Sz-orbit determinants, and the numerators x_r accumulated so far.
    template <typename ElemT>
    struct PT2Config {
      std::vector<int> config;                       ///< per-orbital 0/1/2, the key
      std::vector<std::vector<size_t>> rows;         ///< the block_dim orbit determinants
      std::vector<ElemT> x;                          ///< x_r = sum_i H_ri c_i, per row
      std::vector<bool> x_from_generator;            ///< row was emitted (else x_r computed later)
      int n_open = 0;
    };

    /// Report for the spin-pure accumulation, so the (a)-vs-(c) difference is
    /// accountable and the completion cost is visible instead of inferred.
    struct PT2SpinPureStats {
      size_t n_configs = 0;              ///< distinct perturbing configurations
      size_t n_configs_no_target_s = 0;  ///< dropped: no target-S CSF at all
      size_t n_rows_total = 0;           ///< completed orbit determinants
      size_t n_rows_from_generator = 0;  ///< of those, ones the generator emitted
      size_t n_rows_completed = 0;       ///< of those, ones added by completion
      size_t n_rows_variational = 0;     ///< orbit rows that are in the variational space
      size_t n_configs_partly_variational = 0; ///< configs with >=1 such row
      size_t n_csf = 0;                  ///< target-S CSFs summed over configs
      double t_complete = 0.0;           ///< seconds spent completing orbits
      double t_numerators = 0.0;         ///< seconds spent on the missing-row x_r
      double t_denominators = 0.0;       ///< seconds spent on the h_cc double sums
    };

    /// Rebuild the determinant of one spin arrangement of a configuration.
    ///
    /// `config` is per-orbital 0/1/2 and `mask` says, for each singly-occupied
    /// orbital in ascending order, whether it holds alpha (bit set) or beta. This is
    /// the exact inverse of _det_config(), which is what makes the row order here
    /// agree with the row order of the canonical S^2 eigenvectors: both index
    /// arrangements by the same ascending-orbital slot convention.
    inline void _det_from_config(const std::vector<int> & config,
                                 unsigned long long mask,
                                 size_t bit_length, size_t nword,
                                 std::vector<size_t> & out) {
      out.assign(nword, 0);
      const int norb = static_cast<int>(config.size());
      int slot = 0;
      for (int p = 0; p < norb; ++p) {
        if (config[p] == 2) {
          sbd::setocc(out, bit_length, 2 * p,     true);
          sbd::setocc(out, bit_length, 2 * p + 1, true);
        } else if (config[p] == 1) {
          const bool alpha = ((mask >> slot) & 1ULL) != 0ULL;
          sbd::setocc(out, bit_length, alpha ? (2 * p) : (2 * p + 1), true);
          ++slot;
        }
      }
    }

    /// Group emitted perturbers by spatial configuration and complete each orbit.
    ///
    /// Input `pert` is the merged determinant-level list (numerators already summed
    /// over references, variational determinants NOT yet removed -- see below).
    /// Output is one PT2Config per distinct configuration, with `rows` holding ALL
    /// C(n_open, n_up) arrangements in canonical mask order and `x` holding the
    /// generator's numerator where there was one and 0 where the row must still be
    /// filled in.
    ///
    /// Variational rows are kept in the orbit and flagged, not dropped: a CSF spans
    /// its whole configuration, so if part of an orbit is variational the CSF is not
    /// a legitimate perturber at all and the config must be excluded as a unit. That
    /// is decided in the accumulator, which is why the set-difference is NOT applied
    /// to the input here.
    template <typename ElemT, typename DetsContainer>
    void group_pt2_configs(const std::vector<PT2Merged<ElemT>> & pert,
                           size_t bit_length, int norb, size_t nword,
                           int Sz2,
                           const DetsContainer & det,
                           std::vector<PT2Config<ElemT>> & out,
                           PT2SpinPureStats & st) {
      out.clear();
      if (pert.empty()) return;

      // Bucket the emitted perturbers by configuration, remembering each one's
      // arrangement mask so it can be placed on the right orbit row.
      std::map<std::vector<int>, std::vector<std::pair<unsigned long long, ElemT>>> by_config;
      {
        std::vector<int> config, open_slots;
        for (const auto & p : pert) {
          const unsigned long long m =
              _det_config(p.det, bit_length, norb, config, open_slots);
          by_config[config].push_back({m, p.num});
        }
      }

      const double t0 = MPI_Wtime();
      out.reserve(by_config.size());
      for (const auto & kv : by_config) {
        const std::vector<int> & config = kv.first;
        int n_open = 0;
        for (int cc : config) if (cc == 1) ++n_open;
        const int n_up = (n_open + Sz2) / 2;
        if (((n_open + Sz2) % 2) != 0 || n_up < 0 || n_up > n_open) {
          // Wrong open-shell parity for this Sz. The generator preserves Sz exactly
          // (every excitation copies the annihilated spin), so this cannot happen for
          // perturbers of a single-Sz reference list; it is checked rather than
          // assumed because a truncating division would make n_up silently wrong.
          std::cerr << " sbd: ERROR pt2: a perturbing configuration has " << n_open
                    << " open shells, incompatible with 2*Sz = " << Sz2
                    << ". The generator preserves Sz, so this indicates the reference"
                    << " list is not a single Sz sector." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 6);
        }

        const auto arrangements = _open_shell_arrangements(n_open, n_up);
        const size_t d = arrangements.size();
        std::map<unsigned long long, size_t> mask_to_row;
        for (size_t r = 0; r < d; ++r) mask_to_row[arrangements[r]] = r;

        PT2Config<ElemT> pc;
        pc.config = config;
        pc.n_open = n_open;
        pc.rows.resize(d);
        pc.x.assign(d, ElemT(0.0));
        pc.x_from_generator.assign(d, false);

        for (size_t r = 0; r < d; ++r)
          _det_from_config(config, arrangements[r], bit_length, nword, pc.rows[r]);

        for (const auto & mn : kv.second) {
          auto it = mask_to_row.find(mn.first);
          if (it == mask_to_row.end()) {
            std::cerr << " sbd: ERROR pt2: a perturber's spin arrangement is not in"
                      << " its own configuration's Sz orbit. _det_config and"
                      << " _open_shell_arrangements disagree on slot order."
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 6);
          }
          // Emitted perturbers were already merged, so one row gets at most one entry.
          pc.x[it->second] = mn.second;
          pc.x_from_generator[it->second] = true;
        }

        st.n_rows_total += d;
        for (size_t r = 0; r < d; ++r) {
          if (pc.x_from_generator[r]) ++st.n_rows_from_generator;
          else                        ++st.n_rows_completed;
        }
        out.push_back(std::move(pc));
      }
      st.n_configs = out.size();
      st.t_complete += MPI_Wtime() - t0;
    }

    /// Fill in x_r for the orbit rows the generator did not emit, and mark which
    /// rows are variational.
    ///
    /// A completed row is a determinant nobody reached above threshold, so its x_r is
    /// small -- but it is not zero, and it enters num_c with weight V_rc. Computing it
    /// means H_ri against every LOCAL reference i; the reference list is this b rank's
    /// slice, so the result is this rank's share of x_r, consistent with how (a)
    /// accumulates. Rows that ARE in the variational space are flagged here so the
    /// accumulator can drop their whole configuration.
    template <typename ElemT, typename DetsContainer>
    void complete_pt2_numerators(std::vector<PT2Config<ElemT>> & cfg,
                                 const DetsContainer & det,
                                 const std::vector<ElemT> & c,
                                 size_t bit_length, size_t L, size_t nword,
                                 const ElemT & I0,
                                 const oneInt<ElemT> & I1,
                                 const twoInt<ElemT> & I2,
                                 std::vector<bool> & row_variational_flat,
                                 std::vector<size_t> & row_offset,
                                 PT2SpinPureStats & st) {
      const double t0 = MPI_Wtime();
      const int nso = static_cast<int>(2 * L);

      row_offset.assign(cfg.size() + 1, 0);
      for (size_t k = 0; k < cfg.size(); ++k)
        row_offset[k + 1] = row_offset[k] + cfg[k].rows.size();
      row_variational_flat.assign(row_offset.back(), false);

      auto cmp = [](const auto & x, const auto & y) { return sbd::less_from_back(x, y); };

      // Rows already in the variational space: a binary search per row against the
      // canonically-ordered `det`.
      for (size_t k = 0; k < cfg.size(); ++k) {
        bool any = false;
        for (size_t r = 0; r < cfg[k].rows.size(); ++r) {
          auto it = std::lower_bound(det.begin(), det.end(), cfg[k].rows[r], cmp);
          bool present = false;
          if (it != det.end()) {
            const std::vector<size_t> found = *it;
            present = (found == cfg[k].rows[r]);
          }
          if (present) {
            row_variational_flat[row_offset[k] + r] = true;
            ++st.n_rows_variational;
            any = true;
          }
        }
        if (any) ++st.n_configs_partly_variational;
      }

      // x_r for the rows the generator skipped. This is the dominant cost of variant
      // (c) and the only phase that scales worse than the space: it is
      // (completed rows) x (references), so on N2 it went from 0.02 s at top100 to
      // 7.62 s at top1000 -- 381x for a 45x larger space, while every other phase
      // grew 6-59x. Two things make that affordable without changing the algorithm:
      //
      //   * References are copied into one flat buffer ONCE. `det[i]` on a det_vector
      //     materialises a std::vector per access, so reading it in the innermost
      //     loop meant ~1.4e9 heap allocations at top1000 -- more expensive than the
      //     matrix element it was fetching an argument for.
      //
      //   * A popcount prefilter before Hij. H_ri vanishes unless the two
      //     determinants differ by at most 2 spin-orbitals (4 differing bits), so
      //     nearly every pair in that product is a guaranteed zero. Testing it with
      //     an XOR-popcount over `nword` words rejects those without entering Hij,
      //     which would otherwise walk every bit position to build its c/d lists
      //     before reaching the same conclusion. This is exact, not a screening
      //     threshold: it drops only pairs whose matrix element is identically zero.
      //
      // Threaded over configurations. The loop body writes only into its own cfg[k],
      // so there is no reduction and no order-dependent summation -- the failure mode
      // that produced the b_comm bug in the solver.
      std::vector<size_t> refw(det.size() * nword, 0);
      std::vector<size_t> refi;             // references with a nonzero coefficient
      refi.reserve(det.size());
      for (size_t i = 0; i < det.size(); ++i) {
        if (std::abs(std::complex<double>(c[i])) == 0.0) continue;
        const std::vector<size_t> ri = det[i];
        for (size_t w = 0; w < nword; ++w) refw[i * nword + w] = ri[w];
        refi.push_back(i);
      }

      #pragma omp parallel
      {
        std::vector<int> sc(nso, 0), sd(nso, 0);
        std::vector<size_t> rbuf(nword, 0);
        size_t orbDiff = 0;
        #pragma omp for schedule(dynamic, 1)
        for (size_t k = 0; k < cfg.size(); ++k) {
          for (size_t r = 0; r < cfg[k].rows.size(); ++r) {
            if (cfg[k].x_from_generator[r]) continue;
            if (row_variational_flat[row_offset[k] + r]) continue;  // config will be dropped
            const std::vector<size_t> & row = cfg[k].rows[r];
            ElemT acc(0.0);
            for (size_t q = 0; q < refi.size(); ++q) {
              const size_t i = refi[q];
              const size_t * rp = &refw[i * nword];
              int nd = 0;
              for (size_t w = 0; w < nword; ++w)
                nd += __builtin_popcountll(rp[w] ^ row[w]);
              if (nd > 4) continue;   // more than a double excitation: H_ri == 0
              for (size_t w = 0; w < nword; ++w) rbuf[w] = rp[w];
              const ElemT h = sbd::Hij(rbuf, row, bit_length, L, sc, sd,
                                       I0, I1, I2, orbDiff);
              if (h == ElemT(0.0)) continue;
              acc += h * c[i];
            }
            cfg[k].x[r] = acc;
          }
        }
      }
      st.t_numerators += MPI_Wtime() - t0;
    }

    /// Sum (3) over the target-S CSFs of every perturbing configuration.
    ///
    /// A configuration with any variational row is skipped WHOLE: its CSFs overlap
    /// the variational space, so they are not perturbers, and keeping the rest of the
    /// orbit would give a CSF that is not an S^2 eigenvector. This is stricter than
    /// (a)'s per-determinant set-difference, and necessarily so -- the perturber index
    /// is the CSF, not the determinant.
    template <typename ElemT>
    PT2Result accumulate_pt2_spinpure(const std::vector<PT2Config<ElemT>> & cfg,
                                      const std::vector<bool> & row_variational_flat,
                                      const std::vector<size_t> & row_offset,
                                      double e0, double den_floor,
                                      int multiplicity, int Sz2,
                                      size_t bit_length, size_t L,
                                      const ElemT & I0,
                                      const oneInt<ElemT> & I1,
                                      const twoInt<ElemT> & I2,
                                      PT2SpinPureStats & st) {
      PT2Result r;
      const double s2_target = 0.25 * (static_cast<double>(multiplicity) * multiplicity - 1.0);
      const int nso = static_cast<int>(2 * L);
      const double t0 = MPI_Wtime();

      // The S^2 eigenvectors depend only on (n_open, n_up), not on which orbitals are
      // open -- the spin-coupling matrix is orbital-independent. So one cache serves
      // every configuration with the same open-shell count, which is what makes
      // completing many perturber configurations affordable. Same reasoning, and same
      // helper, as the variational projector: _canonical_csf_coeffs is called here
      // too, so the two cannot drift apart.
      std::map<int, std::vector<double>> coeff_cache;
      std::map<int, std::pair<int, int>> dim_cache;   // n_open -> (block_dim, n_csf)

      std::vector<int> sc(nso, 0), sd(nso, 0);
      size_t orbDiff = 0;

      for (size_t k = 0; k < cfg.size(); ++k) {
        const auto & pc = cfg[k];
        bool skip = false;
        for (size_t rr = 0; rr < pc.rows.size(); ++rr)
          if (row_variational_flat[row_offset[k] + rr]) { skip = true; break; }
        if (skip) continue;

        const int n_open = pc.n_open;
        const int n_up = (n_open + Sz2) / 2;
        if (coeff_cache.find(n_open) == coeff_cache.end()) {
          int bd = 0, nc = 0;
          coeff_cache[n_open] = _canonical_csf_coeffs(n_open, n_up, s2_target, bd, nc);
          dim_cache[n_open] = {bd, nc};
        }
        const int block_dim = dim_cache[n_open].first;
        const int n_csf     = dim_cache[n_open].second;
        if (n_csf == 0) {
          // No target-S component: this configuration cannot be perturbed into by a
          // target-S wavefunction, so it contributes nothing. Variant (a) DOES count
          // its determinants, and that is a real part of the (a)-(c) difference.
          ++st.n_configs_no_target_s;
          continue;
        }
        if (block_dim != static_cast<int>(pc.rows.size())) {
          std::cerr << " sbd: ERROR pt2: orbit size " << pc.rows.size()
                    << " does not match the canonical block dimension " << block_dim
                    << " for n_open = " << n_open << "." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 6);
        }
        const std::vector<double> & V = coeff_cache[n_open];   // (block_dim x n_csf)

        // H over the orbit. Symmetric, so only the upper triangle is evaluated; the
        // diagonal comes from ZeroExcite and the off-diagonals from Hij. This is the
        // dominant cost of (c) at large n_open, and it is why the block is built once
        // per configuration and reused for all its CSFs.
        std::vector<double> Hblk(static_cast<size_t>(block_dim) * block_dim, 0.0);
        for (int a = 0; a < block_dim; ++a) {
          Hblk[static_cast<size_t>(a) * block_dim + a] = static_cast<double>(std::real(
              std::complex<double>(sbd::ZeroExcite(pc.rows[a], bit_length, L, I0, I1, I2))));
          for (int b = a + 1; b < block_dim; ++b) {
            const double h = static_cast<double>(std::real(std::complex<double>(
                sbd::Hij(pc.rows[a], pc.rows[b], bit_length, L, sc, sd,
                         I0, I1, I2, orbDiff))));
            Hblk[static_cast<size_t>(a) * block_dim + b] = h;
            Hblk[static_cast<size_t>(b) * block_dim + a] = h;
          }
        }

        for (int cc = 0; cc < n_csf; ++cc) {
          // num_c = sum_r V_rc x_r
          ElemT num(0.0);
          for (int a = 0; a < block_dim; ++a)
            num += static_cast<ElemT>(V[static_cast<size_t>(a) * n_csf + cc]) * pc.x[a];

          // h_cc = sum_{a,b} V_ac V_bc H_ab -- the FULL double sum. Dropping b != a
          // is the same error that made the projected preconditioner lose
          // intra-configuration exchange.
          double hcc = 0.0;
          for (int a = 0; a < block_dim; ++a) {
            const double va = V[static_cast<size_t>(a) * n_csf + cc];
            if (va == 0.0) continue;
            for (int b = 0; b < block_dim; ++b) {
              const double vb = V[static_cast<size_t>(b) * n_csf + cc];
              if (vb == 0.0) continue;
              hcc += va * vb * Hblk[static_cast<size_t>(a) * block_dim + b];
            }
          }

          double den = e0 - hcc;
          const double ad = std::abs(den);
          if (ad < den_floor) {
            den = (den < 0.0) ? -den_floor : den_floor;
            r.n_floored += 1;
          }
          const double n2 = std::norm(std::complex<double>(num));
          r.energy += n2 / den;
          const double coef = std::sqrt(n2) / std::abs(den);
          r.psi1_norm2 += coef * coef;
          r.worst_ratio = std::max(r.worst_ratio, coef);
          r.n_perturbers += 1;
          st.n_csf += 1;
        }
      }
      st.t_denominators += MPI_Wtime() - t0;
      return r;
    }

  } // namespace gdb

} // namespace sbd

#endif // SBD_CHEMISTRY_GDB_PT2_H
