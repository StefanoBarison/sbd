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

  } // namespace gdb

} // namespace sbd

#endif // SBD_CHEMISTRY_GDB_PT2_H
