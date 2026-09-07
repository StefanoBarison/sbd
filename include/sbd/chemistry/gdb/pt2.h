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

  } // namespace gdb

} // namespace sbd

#endif // SBD_CHEMISTRY_GDB_PT2_H
