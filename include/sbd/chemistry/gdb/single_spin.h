/**
@file sbd/chemistry/gdb/single_spin.h
@brief Option-2 single-spin projection for the general determinant basis.

Projects the (Anderson-completed) determinant subspace onto a single target
total spin S BEFORE diagonalizing, then runs the block Davidson-Liu solver in
the reduced target-S CSF space. Single-spin by construction (no post-hoc <S^2>
filtering) and a smaller solve.

Distribution. V is block diagonal by spatial configuration, so it is rank-local
provided each configuration's determinants all live on one rank. SBD's default
sort (by the full interleaved bitstring) scatters them, which
`--do_redist_config 1` repairs and `require_complete_config_blocks()` verifies
rather than assumes. Each b_comm rank then owns the contiguous CSF range
[csf_base, csf_base+local_csf) (MPI_Exscan; see SpinProjector), and the CSF-space
algebra in `_davidson_projected_core` is b_comm-collective, so b_comm_size > 1 is
supported: inner products, norms, the Rayleigh matrix, the seed and both
Gram-Schmidt passes all reduce over b_comm.

Two rules for anyone editing the solver:
  * Every CSF-space reduction must be collective on b_comm. A rank-local one does
    not crash -- it converges to a plausible wrong energy (see the history note in
    require_complete_config_blocks).
  * Every b rank must execute the same sequence of collectives. Control flow must
    therefore branch on `Kg`/`global_csf`, never on the local `K`, which differs
    between ranks.
`mult` needs no change: it already reduces over t_comm/h_comm and slides over
b_comm, so it is a distributed operator on b-local determinant vectors.

Conventions: determinant bit 2p = alpha orbital p, 2p+1 = beta orbital p
(matches occupation.h getocc). Spatial config code per orbital: 0 empty,
1 singly occupied, 2 doubly occupied.
*/
#ifndef SBD_CHEMISTRY_GDB_SINGLE_SPIN_H
#define SBD_CHEMISTRY_GDB_SINGLE_SPIN_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <omp.h>

namespace sbd {
  namespace gdb {

    /// One configuration's contribution to V: the local determinant indices of
    /// this config's Sz-orbit and the target-S CSF coefficient columns.
    struct ConfigBlock {
      std::vector<size_t> det_indices;   // local indices into `det` (size = block dim)
      std::vector<double> coeffs;        // (block_dim x n_csf) row-major: coeffs[r*n_csf + c]
      int block_dim = 0;
      int n_csf = 0;
    };

    /// Per-rank block-completeness report, filled while V is built.
    ///
    /// A block is COMPLETE when all block_dim arrangement rows of its
    /// configuration are present in the local determinant list. V is
    /// block-diagonal by spatial configuration, so an incomplete block gives a
    /// CSF column built from a strict subset of its spin-coupling set: not
    /// normalized, not an S^2 eigenvector, and -- before this audit -- silently
    /// accepted, because every consumer skips the (size_t)-1 sentinel rows while
    /// `coeffs` still assumes every row is present.
    ///
    /// Two ways a block ends up incomplete:
    ///   1. the determinant list is not spin-complete (missing Sz-orbit members);
    ///   2. b_comm_size > 1 without config-aligned redistribution, so a
    ///      configuration's determinants are scattered across ranks.
    struct ProjectorAudit {
      size_t n_blocks       = 0;   ///< blocks emitted on this rank
      size_t n_incomplete   = 0;   ///< blocks missing >= 1 arrangement row
      size_t n_missing_rows = 0;   ///< total missing rows (surviving sentinels)
      size_t n_dets_covered = 0;   ///< local dets placed in an emitted block
      /// Dets whose configuration has the wrong open-shell parity for the
      /// requested Sz. Fatal: the list is not a single Sz sector.
      size_t n_dets_bad_parity = 0;
      /// Dets in a configuration with no target-S CSF at all. NOT an error --
      /// those determinants genuinely have no target-S component (e.g. a
      /// closed-shell configuration has no triplet CSF) -- but reported so the
      /// count is visible rather than silently dropped.
      size_t n_dets_no_target_s = 0;
      /// One offending configuration, for the diagnostic message.
      std::vector<int> worst_config;
      int worst_block_dim = 0, worst_present = 0;
    };

    /// V: rank-local block-sparse det<->CSF map.
    ///
    /// CSF INDEXING -- two distinct spaces, do not mix them:
    ///
    ///   csf_offset[b]  RANK-LOCAL first CSF column of block b, in [0, local_csf).
    ///                  Every vector this rank stores has length `local_csf`, so
    ///                  this is the index to use for element access (project_up,
    ///                  project_down, ndiag, the Krylov vectors).
    ///   csf_base       This rank's first CSF in the GLOBAL numbering
    ///                  (MPI_Exscan of local_csf over b_comm; 0 when b_comm_size
    ///                  == 1). Block b's global column is csf_base+csf_offset[b].
    ///                  Needed only when a global index must be named (e.g. a
    ///                  MINLOC seed) -- never for indexing a local array.
    ///   local_csf      Number of CSFs this rank owns == length of its slice.
    ///   global_csf     Sum over b_comm. The dimension of the eigenproblem, so
    ///                  the value every control-flow clamp (nb, nkeep, nroot) and
    ///                  every user-facing "CSF dim" print must use.
    ///
    /// The former single `total_csf` was removed deliberately: it meant "local"
    /// at some call sites and "global" at others, which is exactly the confusion
    /// that made b_comm_size > 1 converge to a plausible wrong energy.
    struct SpinProjector {
      std::vector<ConfigBlock> blocks;
      std::vector<int> csf_offset;
      int csf_base = 0;
      int local_csf = 0;
      int global_csf = 0;
      /// Largest block_dim over all blocks. Number of probe matvecs needed to
      /// extract the exact block-diagonal of the projected Hamiltonian.
      int max_block_dim = 0;
      /// Completeness report for this rank's blocks; see ProjectorAudit.
      ProjectorAudit audit;
    };

    // ---- canonical per-n_open S^2 block (orbital-independent), cached ----------

    /// Enumerate the Sz-preserving spin arrangements of n_open open shells with
    /// n_up alpha among them, as bitmasks over the open-shell slots (bit s set =>
    /// slot s is alpha, else beta). Lexicographic, matching a fixed slot order.
    inline std::vector<unsigned long long> _open_shell_arrangements(int n_open, int n_up) {
      std::vector<unsigned long long> out;
      if (n_up < 0 || n_up > n_open) return out;
      // all masks of n_open bits with popcount n_up, ascending
      for (unsigned long long m = 0; m < (1ULL << n_open); ++m) {
        if (__builtin_popcountll(m) == n_up) out.push_back(m);
      }
      return out;
    }

    /// S^2 matrix element between two open-shell spin arrangements (masks a, b)
    /// on n_open singly-occupied slots. Doubly-occupied / empty orbitals are
    /// spectators and drop out. S^2 = Sz^2 + Sz + S_ S+.
    ///   Sz = (n_up - n_dn)/2  (diagonal),
    ///   S_ S+ : sum over slot pairs of a single alpha(slot i)->beta,
    ///           beta(slot j)->alpha transfer (i != j) plus the diagonal count of
    ///           beta-occupied slots.
    inline double _s2_element(unsigned long long a, unsigned long long b,
                              int n_open, int n_up) {
      int n_dn = n_open - n_up;
      double sz = 0.5 * (n_up - n_dn);
      if (a == b) {
        // diagonal: Sz^2 + Sz + (number of beta slots) from S_S+ diagonal part
        int n_beta_slots = n_open - __builtin_popcountll(a); // slots that are beta
        return sz * sz + sz + static_cast<double>(n_beta_slots);
      }
      // off-diagonal S_ S+: a and b must differ by moving one alpha from slot i
      // (alpha in a, beta in b) to a beta->alpha at slot j (beta in a, alpha in b).
      unsigned long long diff = a ^ b;
      if (__builtin_popcountll(diff) != 2) return 0.0;
      // the two differing slots: one is alpha-in-a/beta-in-b, other beta-in-a/alpha-in-b
      // S_S+ connects them with amplitude 1 (spin-1/2 raising/lowering on distinct sites)
      // identify slots
      int i = __builtin_ctzll(diff);
      unsigned long long rest = diff & ~(1ULL << i);
      int j = __builtin_ctzll(rest);
      bool ai = (a >> i) & 1, aj = (a >> j) & 1;
      bool bi = (b >> i) & 1, bj = (b >> j) & 1;
      // need exactly one slot alpha->beta and the other beta->alpha
      bool ok = ((ai && !bi) && (!aj && bj)) || ((!ai && bi) && (aj && !bj));
      return ok ? 1.0 : 0.0;
    }

    /// Build and diagonalize the canonical S^2 block for a given (n_open, n_up),
    /// returning the eigenvectors whose eigenvalue == S_target(S_target+1) as a
    /// (block_dim x n_csf) row-major coefficient matrix. Cached by n_open (n_up is
    /// fixed by Sz across all configs).
    inline std::vector<double> _canonical_csf_coeffs(int n_open, int n_up,
                                                     double s2_target,
                                                     int & block_dim, int & n_csf) {
      auto masks = _open_shell_arrangements(n_open, n_up);
      int d = static_cast<int>(masks.size());
      block_dim = d;
      if (d == 0) { n_csf = 0; return {}; }
      std::vector<double> S2(static_cast<size_t>(d) * d, 0.0);
      for (int r = 0; r < d; ++r)
        for (int c = 0; c < d; ++c)
          S2[static_cast<size_t>(r) * d + c] = _s2_element(masks[r], masks[c], n_open, n_up);
      // diagonalize (column-major for LAPACK; S2 is symmetric so layout is fine)
      std::vector<double> U(S2);           // will hold eigenvectors
      std::vector<double> E(d);
      hp_numeric::MatHeev('V', 'U', d, U.data(), d, E.data());
      // collect columns with eigenvalue ~ s2_target
      std::vector<int> sel;
      for (int c = 0; c < d; ++c)
        if (std::abs(E[c] - s2_target) < 1e-6) sel.push_back(c);
      n_csf = static_cast<int>(sel.size());
      // U is column-major (lda=d): eigenvector c is U[row + d*c]
      std::vector<double> coeffs(static_cast<size_t>(d) * n_csf, 0.0);
      for (int r = 0; r < d; ++r)
        for (int cc = 0; cc < n_csf; ++cc)
          coeffs[static_cast<size_t>(r) * n_csf + cc] = U[static_cast<size_t>(r) + static_cast<size_t>(d) * sel[cc]];
      return coeffs;
    }

    // ---- build V from the local determinant list ------------------------------

    /// Spatial-config key + open-shell slot order for a determinant. Fills
    /// `config` (per-orbital 0/1/2) and `open_slots` (orbital indices of singly
    /// occupied, ascending) and returns whether each open shell is alpha (bit).
    template <typename DetT>
    inline unsigned long long _det_config(const DetT & det,
                                          size_t bit_length, int norb,
                                          std::vector<int> & config,
                                          std::vector<int> & open_slots) {
      config.assign(norb, 0);
      open_slots.clear();
      for (int p = 0; p < norb; ++p) {
        bool a = getocc(det, bit_length, 2 * p);
        bool b = getocc(det, bit_length, 2 * p + 1);
        config[p] = (a ? 1 : 0) + (b ? 1 : 0);
      }
      unsigned long long spin_mask = 0;
      int slot = 0;
      for (int p = 0; p < norb; ++p) {
        if (config[p] == 1) {
          open_slots.push_back(p);
          bool a = getocc(det, bit_length, 2 * p);
          if (a) spin_mask |= (1ULL << slot);   // this open shell is alpha
          ++slot;
        }
      }
      return spin_mask;   // alpha pattern over open slots, in ascending-orbital slot order
    }

    /// Derive the common spin projection 2*Sz of the determinant list, verified to
    /// be the same for every determinant on every b rank.
    ///
    /// COLLECTIVE on b_comm. Replaces reading `det[0]` on each rank, which had
    /// three failure modes, all silent and all reachable once b_comm_size > 1:
    ///   1. `det[0]` is rank-LOCAL, so ranks could derive different Sz2 and build
    ///      projectors for different Sz sectors. Their block dims and CSF counts
    ///      would then disagree and the MPI_Exscan numbering would be nonsense.
    ///   2. An empty local list silently defaulted to Sz2 = 0, which happens to be
    ///      right only for an Sz=0 sector. At high rank counts a rank can own no
    ///      determinants.
    ///   3. Nothing checked that all determinants actually share one Sz. A mixed-Sz
    ///      input is not a single spin sector at all; the projector is meaningless
    ///      for it. This is the same class of silent-garbage hole as an incomplete
    ///      configuration block, so it aborts rather than guesses.
    ///
    /// Cost is one O(ndet * norb) bit scan, done once outside the Davidson loop,
    /// plus two 4-byte allreduces. Returns 2*Sz; aborts on a mixed-Sz list.
    template <typename DetsContainer>
    inline int derive_common_sz2(const DetsContainer & det, size_t bit_length,
                                 int norb, MPI_Comm b_comm, MPI_Comm comm) {
      // Sentinels chosen so a rank owning nothing cannot influence the min/max.
      int loc_min = std::numeric_limits<int>::max();
      int loc_max = std::numeric_limits<int>::min();
      const long long nd = static_cast<long long>(det.size());
#pragma omp parallel for schedule(static) reduction(min:loc_min) reduction(max:loc_max) if(nd > 4096)
      for (long long i = 0; i < nd; ++i) {
        int na = 0, nbe = 0;
        for (int p = 0; p < norb; ++p) {
          if (getocc(det[i], bit_length, 2 * p))     ++na;
          if (getocc(det[i], bit_length, 2 * p + 1)) ++nbe;
        }
        const int s = na - nbe;
        if (s < loc_min) loc_min = s;
        if (s > loc_max) loc_max = s;
      }

      int glb_min = loc_min, glb_max = loc_max;
      int b_size = 1;
      MPI_Comm_size(b_comm, &b_size);
      if (b_size > 1) {
        MPI_Allreduce(&loc_min, &glb_min, 1, MPI_INT, MPI_MIN, b_comm);
        MPI_Allreduce(&loc_max, &glb_max, 1, MPI_INT, MPI_MAX, b_comm);
      }

      int mpi_rank_w = 0;
      MPI_Comm_rank(comm, &mpi_rank_w);

      // Every rank empty => no determinants at all globally.
      if (glb_min == std::numeric_limits<int>::max()) {
        if (mpi_rank_w == 0)
          std::cerr << " sbd: ERROR --single_spin: the determinant list is empty,"
                       " so the spin projection Sz cannot be determined."
                    << std::endl;
        MPI_Abort(comm, 1);
      }
      if (glb_min != glb_max) {
        if (mpi_rank_w == 0)
          std::cerr << " sbd: ERROR --single_spin: the determinant list mixes spin"
                       " projections (found 2*Sz from " << glb_min << " to "
                    << glb_max << ").\n"
                       "   The single-spin projector is defined for ONE Sz sector:"
                       " S^2 is block diagonal over a configuration's Sz-orbit only\n"
                       "   within a fixed Sz. Select a single-Sz determinant list,"
                       " or run without --single_spin." << std::endl;
        MPI_Abort(comm, 1);
      }
      return glb_max;
    }

    /// Build the rank-local single-spin projector V for target spin multiplicity
    /// (2S+1: 1=singlet, 2=doublet, 3=triplet, ...) and spin projection Sz
    /// (n_up open-shell alphas is fixed per config by Sz).
    ///
    /// COLLECTIVE on b_comm (for the global CSF numbering at the end), so it must
    /// be called on every b rank. Each rank's blocks come from its own slice of a
    /// canonically-sorted determinant list, so block emission order is
    /// deterministic and the Exscan is reproducible. That the blocks partition
    /// configurations cleanly across ranks is a precondition, checked separately
    /// by require_complete_config_blocks().
    template <typename ElemT, typename DetsContainer>
    SpinProjector build_config_projector(const DetsContainer & det,
                                         size_t bit_length, int norb,
                                         int multiplicity, int Sz2 /* 2*Sz */,
                                         MPI_Comm b_comm) {
      // multiplicity m = 2S+1 -> S = (m-1)/2 -> S(S+1) = (m^2 - 1)/4
      double s2_target = 0.25 * (static_cast<double>(multiplicity) * multiplicity - 1.0);
      // group local det indices by (config pattern, open-slot arrangement mask)
      // key = config vector serialized; within a config, remember the arrangement.
      std::map<std::vector<int>, std::vector<std::pair<size_t, unsigned long long>>> by_config;
      for (size_t i = 0; i < det.size(); ++i) {
        std::vector<int> config;
        std::vector<int> open_slots;
        unsigned long long m = _det_config(det[i], bit_length, norb, config, open_slots);
        by_config[config].push_back({i, m});
      }

      // cache canonical CSF coeffs by n_open
      std::map<int, std::vector<double>> coeff_cache;
      std::map<int, int> dim_cache, ncsf_cache;

      SpinProjector V;
      int running = 0;
      for (auto & kv : by_config) {
        const std::vector<int> & config = kv.first;
        auto & members = kv.second;   // (det_index, arrangement_mask)
        int n_open = 0;
        for (int c : config) if (c == 1) ++n_open;
        int n_up = (n_open + Sz2) / 2;   // alpha among open shells; Sz2 = 2*Sz
        // Sz2 comes from derive_common_sz2(), which has already verified it is the
        // same for every determinant on every rank. A config whose open-shell count
        // has the wrong parity for that Sz would still make n_up silently wrong
        // (truncating division), so it is counted and skipped, not built.
        if (((n_open + Sz2) % 2) != 0 || n_up < 0 || n_up > n_open) {
          V.audit.n_dets_bad_parity += members.size();
          continue;
        }

        if (coeff_cache.find(n_open) == coeff_cache.end()) {
          int bd, nc;
          coeff_cache[n_open] = _canonical_csf_coeffs(n_open, n_up, s2_target, bd, nc);
          dim_cache[n_open] = bd; ncsf_cache[n_open] = nc;
        }
        int block_dim = dim_cache[n_open];
        int n_csf = ncsf_cache[n_open];
        if (n_csf == 0) {
          // Legitimate: this configuration has no component of the target spin
          // (e.g. a closed-shell config has no triplet CSF). Counted, not fatal.
          V.audit.n_dets_no_target_s += members.size();
          continue;
        }

        // members must be exactly the block_dim arrangements of this config; order
        // them by arrangement mask to match the canonical basis ordering
        // (_open_shell_arrangements enumerates masks ascending).
        auto arrangements = _open_shell_arrangements(n_open, n_up);
        std::map<unsigned long long, int> mask_to_row;
        for (int r = 0; r < static_cast<int>(arrangements.size()); ++r)
          mask_to_row[arrangements[r]] = r;

        ConfigBlock blk;
        blk.block_dim = block_dim;
        blk.n_csf = n_csf;
        blk.det_indices.resize(block_dim, static_cast<size_t>(-1));
        for (auto & pr : members) {
          auto it = mask_to_row.find(pr.second);
          if (it != mask_to_row.end()) blk.det_indices[it->second] = pr.first;
        }
        blk.coeffs = coeff_cache[n_open];   // (block_dim x n_csf) row-major

        // Audit this block's completeness. `coeffs` is the canonical S^2
        // eigenvector for the FULL block_dim orbit, so any row still holding the
        // (size_t)-1 sentinel means the CSF column is a truncation of that
        // eigenvector: neither normalized nor spin-pure. Cheap -- one pass over
        // rows we have just written.
        {
          int present = 0;
          for (int r = 0; r < block_dim; ++r)
            if (blk.det_indices[r] != static_cast<size_t>(-1)) ++present;
          V.audit.n_blocks++;
          V.audit.n_dets_covered += static_cast<size_t>(present);
          if (present != block_dim) {
            V.audit.n_incomplete++;
            V.audit.n_missing_rows += static_cast<size_t>(block_dim - present);
            // Keep the most-truncated block as the example to report.
            if (V.audit.worst_config.empty() ||
                (block_dim - present) > (V.audit.worst_block_dim - V.audit.worst_present)) {
              V.audit.worst_config = config;
              V.audit.worst_block_dim = block_dim;
              V.audit.worst_present = present;
            }
          }
        }

        if (block_dim > V.max_block_dim) V.max_block_dim = block_dim;
        V.blocks.push_back(std::move(blk));
        V.csf_offset.push_back(running);
        running += n_csf;
      }
      V.local_csf = running;

      // Global CSF numbering over b_comm.
      //
      // `running` restarts at 0 on every rank, so csf_offset alone is only
      // meaningful rank-locally. MPI_Exscan gives each rank the sum of the
      // strictly-lower ranks' counts, i.e. the global index of its first CSF.
      // Collective on b_comm: every b rank must reach this, including one that
      // owns no determinants (it contributes 0 and still needs the barrier).
      //
      // MPI_Exscan leaves the recvbuf UNTOUCHED on rank 0, so csf_base is
      // pre-zeroed rather than read back blind. csf_offset itself is left
      // rank-local on purpose -- see the SpinProjector comment.
      {
        int b_size = 1, b_rank = 0;
        MPI_Comm_size(b_comm, &b_size);
        MPI_Comm_rank(b_comm, &b_rank);
        if (b_size > 1) {
          int base = 0;
          MPI_Exscan(&V.local_csf, &base, 1, MPI_INT, MPI_SUM, b_comm);
          V.csf_base = (b_rank == 0) ? 0 : base;
          MPI_Allreduce(&V.local_csf, &V.global_csf, 1, MPI_INT, MPI_SUM, b_comm);
        } else {
          V.csf_base   = 0;
          V.global_csf = V.local_csf;
        }
        // SBD_SS_CSFMAP=1 dumps the per-rank CSF partition. The invariant to check
        // is that the [csf_base, csf_base+local_csf) ranges tile [0, global_csf)
        // with no gap and no overlap: a gap means a rank's blocks were dropped, an
        // overlap means the Exscan input disagreed with what was actually built.
        if (const char* e = std::getenv("SBD_SS_CSFMAP")) {
          if (e[0] == '1') {
            for (int r = 0; r < b_size; ++r) {
              MPI_Barrier(b_comm);
              if (r == b_rank) {
                std::cout << " sbd: single_spin CSF map: b_rank " << b_rank
                          << " blocks=" << V.blocks.size()
                          << " local_csf=" << V.local_csf
                          << " range=[" << V.csf_base << ","
                          << (V.csf_base + V.local_csf) << ")"
                          << " global_csf=" << V.global_csf << std::endl;
                std::cout.flush();
              }
            }
            MPI_Barrier(b_comm);
          }
        }
      }
      return V;
    }

    /**
       Abort unless every rank's configuration blocks are complete.

       Collective on b_comm: must be called on ALL b ranks. Replaces the previous
       blanket `b_comm_size > 1` abort with a checked precondition, so the
       projected solver runs exactly when its assumption actually holds.

       This also closes a pre-existing silent failure at b_comm_size == 1: an
       input determinant list that is not spin-complete produced truncated,
       non-spin-pure CSF columns and a plausible-looking wrong energy, with no
       warning of any kind.
    */
    inline void require_complete_config_blocks(const ProjectorAudit & audit,
                                               int norb,
                                               int b_comm_size,
                                               MPI_Comm b_comm,
                                               MPI_Comm comm) {
      int mpi_rank_b; MPI_Comm_rank(b_comm, &mpi_rank_b);
      int mpi_rank_w; MPI_Comm_rank(comm, &mpi_rank_w);

      // Fatal defects: truncated blocks and wrong-Sz-parity configurations.
      // n_dets_no_target_s is deliberately NOT fatal (see ProjectorAudit).
      unsigned long long loc[4] = {
        static_cast<unsigned long long>(audit.n_incomplete),
        static_cast<unsigned long long>(audit.n_missing_rows),
        static_cast<unsigned long long>(audit.n_dets_bad_parity),
        static_cast<unsigned long long>(audit.n_dets_no_target_s)
      };
      unsigned long long tot[4] = {0,0,0,0};
      MPI_Allreduce(loc, tot, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, b_comm);

      const bool fatal = (tot[0] != 0) || (tot[2] != 0);
      if (!fatal) {
        if (mpi_rank_w == 0) {
          std::cout << " sbd: single_spin: all configuration blocks complete";
          if (tot[3] != 0)
            std::cout << " (" << tot[3] << " determinant(s) in configurations with"
                      << " no target-S component, correctly excluded)";
          std::cout << std::endl;
        }
        // Complete blocks are now SUFFICIENT as well as necessary: the CSF-space
        // reductions in _davidson_projected_core (_local_inner, _local_normalize,
        // the Rayleigh build, the MINLOC seed, the correction MGS and the
        // thick-restart MGS) are b_comm-collective, and the CSF numbering is
        // global (csf_base/global_csf). So b_comm_size > 1 is allowed here.
        //
        // History, since the failure mode was silent rather than loud: with the
        // reductions local-only, b_comm=2 on N2 top50 reported "projected CSF
        // dim = 27" (rank-local, vs 56 global) and converged to -111.07 with the
        // residual stuck near 4 instead of -108.6675. Each rank was solving its
        // own slice as if it were the whole space. If a future change reintroduces
        // a rank-local reduction in that solver, expect exactly this shape:
        // converged-looking, wrong, and no error.
        (void)b_comm_size;
        return;
      }

      // Every rank with a defect reports its own, so the offending rank is named.
      if (audit.n_incomplete != 0 || audit.n_dets_bad_parity != 0) {
        std::ostringstream os;
        os << " sbd: ERROR --single_spin: b_comm rank " << mpi_rank_b
           << " has " << audit.n_incomplete << " INCOMPLETE configuration block(s)"
           << " (" << audit.n_missing_rows << " arrangement row(s) missing)";
        if (audit.n_dets_bad_parity != 0)
          os << " and " << audit.n_dets_bad_parity
             << " determinant(s) whose configuration has the wrong open-shell"
                " parity for the requested Sz";
        os << "\n";
        if (!audit.worst_config.empty()) {
          os << "       worst block: block_dim=" << audit.worst_block_dim
             << " present=" << audit.worst_present << "  config(0=empty,1=single,2=double) =";
          for (int p = 0; p < norb && p < static_cast<int>(audit.worst_config.size()); ++p)
            os << " " << audit.worst_config[p];
          os << "\n";
        }
        std::cerr << os.str() << std::flush;
      }

      if (mpi_rank_w == 0) {
        std::cerr
          << " sbd: V is block-diagonal by spatial configuration. A truncated block\n"
          << "      yields a CSF column that is neither normalized nor an S^2\n"
          << "      eigenvector, so the reported energies would be silently wrong.\n"
          << "      Two possible causes:\n"
          << "        (1) b_comm_size = " << b_comm_size << " > 1 without config-aligned\n"
          << "            redistribution, so a configuration's determinants are split\n"
          << "            across b ranks. Add --do_redist_config 1.\n"
          << "        (2) the determinant list is not spin-complete: some configuration\n"
          << "            is missing Sz-orbit members, or the list spans several Sz\n"
          << "            sectors. Run the Anderson spin completion on the determinant\n"
          << "            set before diagonalizing.\n"
          << std::flush;
      }
      MPI_Barrier(b_comm);      // let every rank's message land before aborting
      MPI_Abort(comm, 1);
    }

    // ---- optional per-phase timing (env SBD_SS_TIMING=1) ----------------------
    // Zero overhead when off: one getenv at construction, a branch per phase.
    struct SSTimers {
      bool on = false;
      double t_matvec=0, t_proj_up=0, t_mult=0, t_proj_down=0;
      double t_subbuild=0, t_ritz=0, t_correct=0, t_restart=0;
      long   n_matvec=0, n_inner=0;
      // --- finer breakdown, added to close the "unaccounted time" gap -----------
      // The coarse timers above covered only ~23% of the measured davidson wall
      // time at K=113394 (45 s of 194 s), so the dominant cost was invisible.
      // These split the two loops that were previously lumped or untimed:
      //   ritz block   : t_ritz_build (Ritz+res assembly) + t_ritz_norm
      //                  (_local_normalize of Ritz[p] and res)
      //   correct block: t_corr_ritz (the DUPLICATE Ritz/res rebuild per
      //                  unconverged root) + t_corr_norm + t_corr_prec
      //                  (preconditioner divide) + t_corr_mgs (two-pass MGS)
      //   t_mv_outer   : wall time of the matvec() callback as seen by the
      //                  driver, counted for BOTH method 0 and method 1 (the
      //                  per-phase project_up/mult/project_down split is only
      //                  filled in by the stored-matrix path).
      //   t_loop_total : whole inner-cycle loop, so unaccounted =
      //                  t_loop_total - (sum of the parts).
      double t_ritz_build=0, t_ritz_norm=0;
      double t_corr_ritz=0, t_corr_norm=0, t_corr_prec=0, t_corr_mgs=0;
      double t_mv_outer=0, t_loop_total=0, t_seed=0, t_ndiag=0;
      long   n_corr_ritz=0;   // duplicate-rebuild count (nroot-weighted)
      SSTimers() { const char* e = std::getenv("SBD_SS_TIMING"); on = (e && e[0]=='1'); }
    };
    static inline double _wtime() { return MPI_Wtime(); }

    // ---- exact projected diagonal (preconditioner) ----------------------------

    /**
       Exact diagonal of the projected Hamiltonian, H^csf_cc = sum_{r,r'} V_rc
       V_r'c H_rr', for use as the Davidson preconditioner.

       Why this must be exact: the earlier version used only the r == r' terms,
       i.e. the diagonal of V^T diag(H) V. The dropped r != r' terms are exactly
       the determinant pairs WITHIN one spatial configuration -- dets differing
       only by a spin flip among the open shells. Their Hamiltonian elements are
       the exchange integrals, which are large and predominantly negative, so
       omitting them biases every preconditioner denominator (E[p] - ndiag[i])
       in the same direction. The projected solve stays single-S regardless (V is
       exact and fixed), but Davidson stalls on a plateau and reports a converged
       Ritz value ABOVE the true root -- i.e. correct spin, wrong (too high)
       energy, worsening with K. This is the "spin-averaged preconditioner"
       requirement of Fales/Hohenstein/Levine, JCTC 13, 4162 (2017), Sec. 2.6:
       the preconditioner must average the exchange integrals WITHIN a
       spin-coupling set, and a ConfigBlock is precisely one such set.

       How: H^csf restricted to one block is V_b^T H_bb V_b, where H_bb is the
       block's det-det submatrix. We build H_bb directly with the general
       Hij(DetA,DetB,...) primitive from chemistry/basic/determinants.h, which
       carries SBD's own parity/phase conventions (the same OneExcite/TwoExcite
       used by mult()), so no phase logic is duplicated here.

       Cost: sum over blocks of block_dim^2 Hij evaluations, once, before the
       iteration. block_dim = C(n_open, n_up) is small (open shells per
       configuration), and sum(block_dim) == ndet, so this is O(max_block_dim *
       ndet) element evaluations -- negligible next to the solve.

       (An earlier attempt batched this as max_block_dim probe matvecs, exploiting
       that blocks own disjoint determinant sets. That is WRONG: H couples
       determinants of DIFFERENT configurations, so probing row r of every block
       at once contaminates each block's readback with cross-block elements. A
       numerical check showed errors of order the matrix elements themselves.
       Probing one block at a time would be exact but costs sum(block_dim)
       matvecs; direct Hij is both exact and far cheaper.)

       Diagonal elements are taken from the precomputed hii to stay bit-identical
       with the determinant-space solver's diagonal.
    */
    template <typename ElemT, typename RealT, typename DetsContainer>
    void build_projected_diagonal(const SpinProjector & V,
                                  const std::vector<ElemT> & hii,
                                  const DetsContainer & det,
                                  size_t bit_length,
                                  size_t norb,
                                  const ElemT & I0,
                                  const oneInt<ElemT> & I1,
                                  const twoInt<ElemT> & I2,
                                  std::vector<RealT> & ndiag) {
      // Local: ndiag is indexed by csf_offset[b]+c, which is rank-local.
      const int K = V.local_csf;
      ndiag.assign(K, RealT(0));
      const long long nblk = static_cast<long long>(V.blocks.size());

#pragma omp parallel for schedule(dynamic) if(nblk > 64)
      for (long long b = 0; b < nblk; ++b) {
        const ConfigBlock & blk = V.blocks[b];
        const int d = blk.block_dim;
        const int off = V.csf_offset[b];

        // dense intra-block det-det submatrix H_bb (d x d, symmetric)
        std::vector<double> Hbb(static_cast<size_t>(d) * d, 0.0);
        for (int r = 0; r < d; ++r) {
          const size_t di = blk.det_indices[r];
          if (di == static_cast<size_t>(-1)) continue;
          Hbb[static_cast<size_t>(r) * d + r] = GetReal(hii[di]);
          for (int rp = r + 1; rp < d; ++rp) {
            const size_t dj = blk.det_indices[rp];
            if (dj == static_cast<size_t>(-1)) continue;
            size_t orbDiff = 0;
            ElemT h = Hij(det[di], det[dj], bit_length, norb, I0, I1, I2, orbDiff);
            const double hv = GetReal(h);
            Hbb[static_cast<size_t>(r) * d + rp] = hv;
            Hbb[static_cast<size_t>(rp) * d + r] = hv;   // H is symmetric
          }
        }

        // ndiag[off+c] = sum_{r,r'} V_rc V_r'c H_bb[r,r']
        for (int c = 0; c < blk.n_csf; ++c) {
          double acc = 0.0;
          for (int r = 0; r < d; ++r) {
            if (blk.det_indices[r] == static_cast<size_t>(-1)) continue;
            const double vr = blk.coeffs[static_cast<size_t>(r) * blk.n_csf + c];
            if (vr == 0.0) continue;
            for (int rp = 0; rp < d; ++rp) {
              if (blk.det_indices[rp] == static_cast<size_t>(-1)) continue;
              acc += vr * blk.coeffs[static_cast<size_t>(rp) * blk.n_csf + c]
                        * Hbb[static_cast<size_t>(r) * d + rp];
            }
          }
          ndiag[off + c] = static_cast<RealT>(acc);
        }
      }
    }

    // ---- projected matvec y_csf = V^T (mult) (V x_csf) ------------------------

    /// Expand a CSF-space vector to determinant space: x_det = V x_csf (local).
    template <typename ElemT>
    void project_up(const SpinProjector & V, const std::vector<ElemT> & x_csf,
                    std::vector<ElemT> & x_det, size_t ndet) {
      x_det.assign(ndet, ElemT(0.0));
      // Blocks own disjoint det indices (grouped by exact spatial config), so
      // writing x_det[di] across blocks is race-free -> parallelize over blocks.
      const long long nblk = static_cast<long long>(V.blocks.size());
#pragma omp parallel for schedule(dynamic) if(nblk > 256)
      for (long long b = 0; b < nblk; ++b) {
        const ConfigBlock & blk = V.blocks[b];
        int off = V.csf_offset[b];
        for (int r = 0; r < blk.block_dim; ++r) {
          size_t di = blk.det_indices[r];
          if (di == static_cast<size_t>(-1)) continue;
          ElemT acc = ElemT(0.0);
          for (int c = 0; c < blk.n_csf; ++c)
            acc += ElemT(blk.coeffs[static_cast<size_t>(r) * blk.n_csf + c]) * x_csf[off + c];
          x_det[di] = acc;
        }
      }
    }

    /// Contract a determinant-space vector to CSF space: y_csf = V^T y_det (local).
    template <typename ElemT>
    void project_down(const SpinProjector & V, const std::vector<ElemT> & y_det,
                      std::vector<ElemT> & y_csf) {
      y_csf.assign(V.local_csf, ElemT(0.0));
      // Each block writes a disjoint y_csf[off .. off+n_csf) range (csf_offset is
      // unique per block), so parallelizing over blocks is race-free.
      const long long nblk = static_cast<long long>(V.blocks.size());
#pragma omp parallel for schedule(dynamic) if(nblk > 256)
      for (long long b = 0; b < nblk; ++b) {
        const ConfigBlock & blk = V.blocks[b];
        int off = V.csf_offset[b];
        for (int r = 0; r < blk.block_dim; ++r) {
          size_t di = blk.det_indices[r];
          if (di == static_cast<size_t>(-1)) continue;
          ElemT val = y_det[di];
          for (int c = 0; c < blk.n_csf; ++c)
            y_csf[off + c] += ElemT(blk.coeffs[static_cast<size_t>(r) * blk.n_csf + c]) * val;
        }
      }
    }

    // ---- distributed CSF-space vector helpers ---------------------------------
    // Each b_comm rank owns a contiguous slice of the CSF space (see
    // SpinProjector: [csf_base, csf_base+local_csf)), so an inner product or a
    // norm over the GLOBAL vector is the sum of the per-rank partials. Both
    // helpers therefore take b_comm and allreduce; they are the primitives the
    // rest of the solver is built from, so making them collective makes the Ritz
    // build, the correction MGS and the thick restart distributed for free.
    //
    // At b_comm_size == 1 the allreduce is skipped entirely, so the arithmetic
    // is bit-identical to the previous local-only version.
    //
    // COLLECTIVE: every b rank must call these the same number of times in the
    // same order. That is why the parallel-vs-serial branch below keys off a
    // communicator-independent threshold and never off the local slice length --
    // ranks with different local_csf must still agree on the call sequence.
    //
    // ElemT may be complex, so we cannot use an OpenMP reduction clause on it
    // directly. Accumulate per-thread partial sums into a scratch array and
    // combine serially (thread count is small vs the K-length loop).

    /// Allreduce-sum `n` ElemT values in place over comm. No-op when size == 1,
    /// so b_comm_size == 1 stays bit-identical to the old local-only code.
    /// Uses SBD's own GetMpiType trait (framework/type_def.h) rather than
    /// reinterpreting the buffer, so float and complex ElemT work unchanged.
    /// Number of _bcomm_sum calls issued by this rank, and the total element count.
    /// Diagnostic for MPI_ERR_TRUNCATE: a count mismatch means ranks disagreed on
    /// the SEQUENCE of reductions, which this makes visible.
    inline long long & _bcomm_call_count() { static long long c = 0; return c; }
    inline long long & _bcomm_elem_count() { static long long c = 0; return c; }

    template <typename ElemT>
    inline void _bcomm_sum(ElemT * x, int n, MPI_Comm comm) {
      int csize = 1; MPI_Comm_size(comm, &csize);
      if (csize == 1 || n == 0) return;
      _bcomm_call_count()++;
      _bcomm_elem_count() += n;
      // SBD_SS_TRACE_RED=1: before each reduction, confirm every rank agrees on
      // WHICH reduction this is (call ordinal) and on its element count. A
      // disagreement is the cause of MPI_ERR_TRUNCATE and is otherwise invisible.
      if (const char* _e = std::getenv("SBD_SS_TRACE_RED")) {
        if (_e[0] == '1') {
          long long mine[2] = { _bcomm_call_count(), static_cast<long long>(n) };
          long long lo[2] = { mine[0], mine[1] }, hi[2] = { mine[0], mine[1] };
          MPI_Allreduce(MPI_IN_PLACE, lo, 2, MPI_LONG_LONG, MPI_MIN, comm);
          MPI_Allreduce(MPI_IN_PLACE, hi, 2, MPI_LONG_LONG, MPI_MAX, comm);
          if (lo[0] != hi[0] || lo[1] != hi[1]) {
            int cr = 0, wr = 0;
            MPI_Comm_rank(comm, &cr); MPI_Comm_rank(MPI_COMM_WORLD, &wr);
            std::cerr << " sbd: ERROR b_comm reduction DESYNC: this rank is at call "
                      << mine[0] << " with n=" << n << "; across b_comm the call"
                      << " ordinal spans " << lo[0] << ".." << hi[0]
                      << " and n spans " << lo[1] << ".." << hi[1]
                      << " (b rank " << cr << ", world rank " << wr << ")"
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, x, n, GetMpiType<ElemT>::MpiT, MPI_SUM, comm);
    }

    template <typename ElemT>
    ElemT _local_inner(const std::vector<ElemT> & a, const std::vector<ElemT> & b,
                       MPI_Comm b_comm) {
      const size_t n = a.size();
      ElemT s = ElemT(0.0);
      if (n <= 4096) {   // small: parallel overhead not worth it
        for (size_t i = 0; i < n; ++i) s += Conjugate(a[i]) * b[i];
      } else {
        int nth = omp_get_max_threads();
        std::vector<ElemT> partial(nth, ElemT(0.0));
#pragma omp parallel
        {
          int tid = omp_get_thread_num();
          ElemT loc = ElemT(0.0);
#pragma omp for
          for (size_t i = 0; i < n; ++i) loc += Conjugate(a[i]) * b[i];
          partial[tid] = loc;
        }
        for (int t = 0; t < nth; ++t) s += partial[t];
      }
      _bcomm_sum(&s, 1, b_comm);
      return s;
    }

    template <typename ElemT, typename RealT>
    RealT _local_normalize(std::vector<ElemT> & a, MPI_Comm b_comm) {
      const size_t n = a.size();
      RealT n2 = RealT(0.0);
      if (n <= 4096) {
        for (size_t i = 0; i < n; ++i) n2 += GetReal(Conjugate(a[i]) * a[i]);
      } else {
#pragma omp parallel for reduction(+:n2)
        for (size_t i = 0; i < n; ++i) n2 += GetReal(Conjugate(a[i]) * a[i]);
      }
      // The norm is global: sum the squared partials BEFORE the square root, or
      // each rank would scale its slice by its own local norm and the assembled
      // global vector would not be normalized at all.
      _bcomm_sum(&n2, 1, b_comm);
      RealT nrm = std::sqrt(n2);
      if (nrm > RealT(0)) {
#pragma omp parallel for if(n > 4096)
        for (size_t i = 0; i < n; ++i) a[i] /= nrm;
      }
      return nrm;
    }

    /**
       Multi-root block Davidson-Liu in the target-S CSF space (b_comm==1).
       The Hamiltonian matvec is the projected operator y = V^T H (V x): expand
       CSF -> det via V, apply the existing gdb mult, contract det -> CSF via V^T.
       Converged roots are single-S by construction. Mirrors DavidsonMultiRoot;
       the det-basis solver is left untouched.

       @param[out] Wcsf  nroot CSF-space eigenvectors (each length V.local_csf,
                         i.e. this rank's slice of the global CSF space)
       @param[out] Eout  nroot lowest energies (ascending)
    */
    /**
       Core block Davidson-Liu in the target-S CSF space (b_comm==1). The H*v
       matvec is supplied as a callable so the same core serves both the
       matrix-free (method 0) and stored-matrix (method 1) projected solvers.
       `matvec(x_csf, y_csf)` must implement y = V^T H (V x).
    */
    /// SBD_SS_CHECK_MGS=1 verifies each fused-MGS overlap against a serial recompute.
    inline bool _ss_check_mgs() {
      static const bool on = [](){
        const char* e = std::getenv("SBD_SS_CHECK_MGS");
        return e && e[0] == '1';
      }();
      return on;
    }

    /// Threshold above which CSF-space loops go parallel; see SBD_SS_PAR_THRESHOLD.
    inline int _csf_par_thr_value() {
      static const int v = [](){
        if (const char* e = std::getenv("SBD_SS_PAR_THRESHOLD")) {
          int x = std::atoi(e);
          if (x >= 0) return x;
        }
        return 4096;
      }();
      return v;
    }

    template <typename ElemT, typename RealT>
    void _davidson_projected_core(const std::vector<ElemT> & hii,
                                  const SpinProjector & V,
                                  const std::function<void(const std::vector<ElemT>&,
                                                           std::vector<ElemT>&)> & matvec,
                                  std::vector<std::vector<ElemT>> & Wcsf,
                                  std::vector<RealT> & Eout,
                                  MPI_Comm h_comm,
                                  MPI_Comm b_comm,
                                  MPI_Comm t_comm,
                                  int max_iteration,
                                  int num_block,
                                  int nroot,
                                  RealT eps,
                                  int nkeep,
                                  SSTimers * tmr = nullptr,
                                  const std::vector<RealT> * ndiag_exact = nullptr) {
      RealT eps_reg = 1.0e-12;
      RealT tau_drop = 1.0e-6;
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);

      // K  = this rank's slice length -> every vector allocation and element loop.
      // Kg = the eigenproblem dimension -> every control-flow clamp below.
      // They are equal iff b_comm_size == 1. Using K for a clamp would let each
      // rank pick a different nb/nkeep and desynchronize the collectives.
      const int K  = V.local_csf;
      const int Kg = V.global_csf;

      int b_comm_size = 1; MPI_Comm_size(b_comm, &b_comm_size);

      // Threshold above which the K-length loops go parallel. Normally 4096 (below
      // that, OpenMP fork/join costs more than the work). SBD_SS_PAR_THRESHOLD
      // overrides it so the parallel/fused-MGS paths can be exercised on small
      // test cases: K is the LOCAL slice length, so at b_comm>1 a problem big
      // enough to trip 4096 on one rank may not trip it on any rank, leaving the
      // fused-MGS path (with its collectives) untested. Set it to 0 in tests.
      // SBD_SS_CHECK_IB=1 asserts every b rank is on the same inner step before
      // each Rayleigh reduction. Diagnostic for collective-divergence bugs.
      const bool _ss_check_ib = [](){
        const char* e = std::getenv("SBD_SS_CHECK_IB");
        return e && e[0] == '1';
      }();
      // SBD_SS_SERIAL=<n> forces CSF-loop region n to run SERIAL while the others
      // stay parallel, for bisecting a threading defect. Regions are numbered in
      // source order; 0 = none. Diagnostic only.
      const int _ss_serial_region = [](){
        const char* e = std::getenv("SBD_SS_SERIAL");
        return e ? std::atoi(e) : 0;
      }();
      auto _ss_par = [&](int region, int k) {
        return (region != _ss_serial_region) && (k > _csf_par_thr_value());
      };
      const int csf_par_threshold = [](){
        if (const char* e = std::getenv("SBD_SS_PAR_THRESHOLD")) {
          int v = std::atoi(e);
          if (v >= 0) return v;
        }
        return 4096;
      }();

      // Subspace (Krylov) cap before collapse. Modest size is fine — with a
      // correct residual the block Davidson-Liu converges in tens of iterations
      // at small nb (verified in Python: nroot=2, nb=12 -> 57 iters at K=3906).
      // Keep RAM low: CSF vectors are held per rank (b_comm==1), so 2*nb*K.
      int nb = num_block;
      int nb_min = nroot + std::max(nroot, 10);
      if (nb < nb_min) nb = nb_min;
      if (nb > Kg) nb = Kg;
      if (nb < nroot) nb = nroot;

      // Thick-restart carryover: keep the lowest `nkeep` Ritz vectors at each
      // subspace collapse (was: reseed only the nroot targets). Must keep at
      // least nroot.
      if (nkeep < nroot) nkeep = nroot;
      if (nkeep < 1) nkeep = 1;

      // Auto-size nb from nkeep to preserve per-cycle Krylov growth.
      //
      // The correction loop appends new directions until (ib+1) reaches nb, so
      // each restart grows the subspace by exactly (nb - nseed) = (nb - nkeep)
      // vectors before the next collapse. If nkeep is large relative to nb the
      // growth room (nb - nkeep) shrinks and the solve stalls at a wrong
      // eigenvalue (the K=113394 failure: keep=15/nb=30 -> 15 growth slots
      // stalled; keep=1/nb=30 -> 29 slots converged to -108.8435).
      //
      // Rather than clamp nkeep down (which throws away the matvec-count win of
      // a large carry), grow nb up so the growth room stays fixed at the value
      // the user's num_block implied with a minimal carry: growth = nb - nroot.
      // Then nb_effective = nkeep + growth, so (nb - nkeep) == growth for any
      // nkeep. keep=nroot reproduces the old nb exactly; larger keep just widens
      // the subspace. Cost is RAM only (2*nb*K per rank), bounded by K.
      {
        int growth = nb - nroot;               // directions/cycle user intended
        if (growth < 1) growth = 1;
        int nb_need = nkeep + growth;
        if (nb < nb_need) nb = nb_need;
        if (nb > Kg) nb = Kg;
        // nkeep must still leave at least one growth slot after the Kg clamp.
        if (nkeep > nb - 1) nkeep = nb - 1;
        if (nkeep < nroot) nkeep = std::min(nroot, nb - 1);
        if (nkeep < 1) nkeep = 1;
      }

      if (mpi_rank_h == 0 && mpi_rank_t == 0 && mpi_rank_b == 0) {
        std::cout << "sbd(ss): nb=" << nb << " keep=" << nkeep
                  << " growth=" << (nb - nkeep) << " (K=" << Kg << ")"
                  << std::endl;
      }

      // Projected diagonal of H in CSF space, for the preconditioner.
      //
      // Prefer the EXACT block diagonal supplied by build_projected_diagonal():
      //   ndiag[off+c] = sum_{r,r'} V_rc V_r'c H_rr'
      // The r != r' terms are the intra-configuration (spin-flip) exchange
      // elements; dropping them biases every denominator (E[p] - ndiag[i]) the
      // same way and stalls Davidson above the true root -- correct spin, too
      // high energy, worse with K. See build_projected_diagonal().
      //
      // Fallback (only if no exact diagonal was supplied): the old approximation
      // V^T diag(H) V, kept so the core still runs standalone.
      std::vector<RealT> ndiag;
      if (ndiag_exact != nullptr &&
          static_cast<int>(ndiag_exact->size()) == K) {
        ndiag = *ndiag_exact;
      } else {
        ndiag.assign(K, RealT(0));
        const long long nblk_diag = static_cast<long long>(V.blocks.size());
#pragma omp parallel for schedule(dynamic) if(nblk_diag > 256)
        for (long long bidx = 0; bidx < nblk_diag; ++bidx) {
          const ConfigBlock & blk = V.blocks[bidx];
          int off = V.csf_offset[bidx];
          for (int c = 0; c < blk.n_csf; ++c) {
            RealT acc = RealT(0);
            for (int r = 0; r < blk.block_dim; ++r) {
              size_t di = blk.det_indices[r];
              if (di == static_cast<size_t>(-1)) continue;
              RealT cr = static_cast<RealT>(blk.coeffs[static_cast<size_t>(r)*blk.n_csf + c]);
              acc += cr * cr * GetReal(hii[di]);
            }
            ndiag[off + c] = acc;
          }
        }
        if (mpi_rank_h == 0 && mpi_rank_t == 0 && mpi_rank_b == 0)
          std::cout << "sbd(ss): WARNING using approximate preconditioner "
                    << "(V^T diag(H) V); energies may converge above the true root"
                    << std::endl;
      }

      std::vector<std::vector<ElemT>> v(nb, std::vector<ElemT>(K));
      std::vector<std::vector<ElemT>> Hv(nb, std::vector<ElemT>(K));
      std::vector<std::vector<ElemT>> Ritz(nkeep, std::vector<ElemT>(K));
      std::vector<ElemT> res(K);
      std::vector<RealT> norm_r(nroot, RealT(0));

      ElemT * H = (ElemT *) calloc((size_t)nb*nb, sizeof(ElemT));
      ElemT * U = (ElemT *) calloc((size_t)nb*nb, sizeof(ElemT));
      RealT * E = (RealT *) malloc((size_t)nb * sizeof(RealT));
      char jobz = 'V', uplo = 'U';

      // Seed: unit CSF vectors on the nroot LOWEST-ndiag CSFs.
      //
      // Was: e_0..e_{nroot-1}, i.e. the first nroot CSFs in whatever order
      // std::map<std::vector<int>> happened to emit configurations. That is an
      // arbitrary (often high-energy) corner of the space with no relation to the
      // target roots, so the solve wastes iterations climbing down and, combined
      // with a weak preconditioner, can plateau. Fales/Hohenstein/Levine build
      // their guess from the lowest-energy determinants for exactly this reason
      // (JCTC 13, 4162 (2017), Sec. 2.6 / Algorithm 3).
      //
      // Cheapest faithful analogue here: pick the nroot smallest projected
      // diagonals. Unit vectors are trivially orthonormal, so no extra
      // orthogonalization is needed, and with the exact ndiag these are the
      // variationally best single-CSF starting guesses.
      int nseed = nroot;
      // Number of leading seed vectors whose Hv is already valid (thick restart
      // carries Hv for the kept Ritz vectors, so they need no fresh matvec). 0 at
      // the initial unit-vector seed.
      int nhv_valid = 0;
      {
        double _tsd = (tmr && tmr->on) ? _wtime() : 0.0;
        const int ntake = std::min(nroot, Kg);
        for (int p = 0; p < nroot; p++) std::fill(v[p].begin(), v[p].end(), ElemT(0.0));

        if (b_comm_size == 1) {
          std::vector<int> order(K);
          for (int i = 0; i < K; ++i) order[i] = i;
          std::partial_sort(order.begin(), order.begin() + ntake, order.end(),
                            [&](int a, int b) { return ndiag[a] < ndiag[b]; });
          for (int p = 0; p < ntake; p++) v[p][order[p]] = ElemT(1.0);
          if (mpi_rank_h == 0 && mpi_rank_t == 0 && mpi_rank_b == 0 && ntake > 0)
            std::cout << "sbd(ss): seed CSFs (lowest ndiag) first="
                      << order[0] << " diag=" << ndiag[order[0]] << std::endl;
        } else {
          // Distributed: the nroot lowest diagonals are GLOBAL minima, so picking
          // them rank-locally would seed nroot vectors per rank (and put
          // b_comm_size ones into each assembled global vector). Take them one at
          // a time with MPI_MINLOC over (value, global index), masking each winner
          // so the next pass finds a distinct CSF. nroot is small (<= ~10), so
          // nroot tiny collectives is cheaper than any global sort, and using the
          // global index as the MINLOC tiebreak makes the choice deterministic
          // when diagonals tie exactly (symmetry-equivalent CSFs do tie).
          // MPI_DOUBLE_INT is defined by the standard to match exactly this
          // layout -- a double followed by an int, tail padding included -- so no
          // packing pragma is wanted here (adding one would BREAK the match). The
          // assertion catches an ABI where that stops holding.
          struct MinLocDI { double val; int idx; };
          static_assert(sizeof(MinLocDI) >= sizeof(double) + sizeof(int),
                        "MPI_DOUBLE_INT layout assumption");
          std::vector<char> taken(static_cast<size_t>(K), 0);
          for (int p = 0; p < ntake; p++) {
            MinLocDI loc, glb;
            // A rank owning no CSF (or none left untaken) must not win: seed it
            // with +inf so MPI_MINLOC always prefers a real candidate.
            loc.val = std::numeric_limits<double>::max();
            loc.idx = std::numeric_limits<int>::max();
            for (int i = 0; i < K; ++i) {
              if (taken[i]) continue;
              const double d = static_cast<double>(ndiag[i]);
              const int gi = V.csf_base + i;
              if (d < loc.val || (d == loc.val && gi < loc.idx)) { loc.val = d; loc.idx = gi; }
            }
            MPI_Allreduce(&loc, &glb, 1, MPI_DOUBLE_INT, MPI_MINLOC, b_comm);
            // Only the owner of the winning global index writes the 1.0.
            const int li = glb.idx - V.csf_base;
            if (li >= 0 && li < K) { v[p][li] = ElemT(1.0); taken[li] = 1; }
            if (p == 0 && mpi_rank_h == 0 && mpi_rank_t == 0 && mpi_rank_b == 0)
              std::cout << "sbd(ss): seed CSFs (lowest ndiag) first="
                        << glb.idx << " diag=" << glb.val << std::endl;
          }
        }
        if (tmr && tmr->on) tmr->t_seed += _wtime() - _tsd;
      }

      bool do_continue = true;
      double _tloop0 = (tmr && tmr->on) ? _wtime() : 0.0;
      for (int it = 0; it < max_iteration && do_continue; it++) {
        int ib = 0, m = nhv_valid, ncur = nseed;
        while (true) {
          {
            double _tmv = (tmr && tmr->on) ? _wtime() : 0.0;
            for (int jb = m; jb < ncur; jb++) matvec(v[jb], Hv[jb]);
            if (tmr && tmr->on) tmr->t_mv_outer += _wtime() - _tmv;
          }
          m = ncur; ib = ncur - 1;
          nhv_valid = 0;  // consumed; only the first cycle after a restart carries Hv

          double _ts = (tmr && tmr->on) ? _wtime() : 0.0;
          // Subspace (Rayleigh) matrix H[jb,kb] = <v_jb, Hv_kb>. Parallelize ONCE
          // over the flattened (jb,kb) pair index — each thread does full serial
          // inner products for its pairs. The old code called _local_inner per
          // pair, each spawning its own OMP region: ~nb^2 fork/join barriers per
          // inner step over tiny K-length reductions -> catastrophic at many
          // threads (measured subspace-build 0.18s serial -> 4.3s at 8 threads,
          // K=3906; the 1120->3008s regression at K=113k/192 threads). One region
          // amortizes the barrier over all pairs.
          const int nv2 = (ib + 1) * (ib + 1);
          // H is reduced as a whole nb x nb buffer (see below), so every element
          // OUTSIDE the live (ib+1)^2 block must be zero -- it is written only
          // inside that block, and a leftover value from a previous inner step
          // would otherwise be summed across ranks and land in U. Zeroing the
          // whole buffer costs nb^2 <= ~1600 writes per inner step, against
          // nb^2 K-length dot products.
          if (b_comm_size > 1)
            std::fill(H, H + static_cast<size_t>(nb)*nb, ElemT(0.0));
#pragma omp parallel for schedule(static) if(_ss_par(1,K))
          for (int idx = 0; idx < nv2; ++idx) {
            int jb = idx / (ib + 1);
            int kb = idx % (ib + 1);
            const std::vector<ElemT> & a = v[jb];
            const std::vector<ElemT> & b = Hv[kb];
            ElemT s = ElemT(0.0);
            for (int i = 0; i < K; ++i) s += Conjugate(a[i]) * b[i];
            H[jb + nb*kb] = s;
          }
          // SBD_SS_CHECK_H=1: recompute the Rayleigh block serially and compare.
          // Isolates a threading defect in the loop above from anything downstream.
          if (const char* _e = std::getenv("SBD_SS_CHECK_H")) {
            if (_e[0] == '1') {
              double worst = 0.0;
              for (int jb = 0; jb <= ib; ++jb)
                for (int kb = 0; kb <= ib; ++kb) {
                  ElemT s = ElemT(0.0);
                  for (int i = 0; i < K; ++i) s += Conjugate(v[jb][i]) * Hv[kb][i];
                  const double d = std::abs(GetReal(s - H[jb + nb*kb]));
                  if (d > worst) worst = d;
                }
              if (worst > 1.0e-9) {
                int wr = 0; MPI_Comm_rank(MPI_COMM_WORLD, &wr);
                std::cerr << " sbd: ERROR Rayleigh block wrong by " << worst
                          << " on world rank " << wr << " (ib=" << ib
                          << ", nv2=" << nv2 << ", K=" << K << ", threads="
                          << omp_get_max_threads() << ")" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
            }
          }
          // Each rank has only its slice's contribution to every <v_jb, Hv_kb>.
          // Reduce the whole (ib+1)^2 block in ONE allreduce -- per-pair
          // allreduces would be nb^2 collectives per inner step, the MPI analogue
          // of the OpenMP fork/join storm this loop was restructured to avoid.
          // H is nb x nb column-major with the live block in the leading
          // (ib+1) x (ib+1) corner, so the columns are strided: pack, reduce, unpack.
          if (b_comm_size > 1) {
            // Reduce the FULL nb x nb buffer, not the live (ib+1)^2 block.
            //
            // The count must not depend on anything rank-local. `ib` is derived
            // from `appended`, which is derived from per-root residual norms --
            // all globally reduced, so in principle every rank agrees. In
            // practice a divergence there produced MPI_ERR_TRUNCATE ("message
            // truncated") deep in a production run at K=540707, because two
            // ranks called MPI_Allreduce with different (ib+1)^2 counts. Using
            // nb*nb -- a scalar fixed before the loop and identical on every
            // rank by construction -- makes that failure mode unreachable
            // instead of merely unlikely. The extra elements are zeros outside
            // the live block, so the reduction result is unchanged; nb <= ~40,
            // so nb*nb doubles is a few KB.
            //
            // If ib really can diverge, that is a separate correctness bug and
            // this reduction would silently combine different subspaces -- hence
            // the SBD_SS_CHECK_IB guard below, which turns it into a loud error.
            _bcomm_sum(H, nb*nb, b_comm);
          }
          if (b_comm_size > 1 && _ss_check_ib) {
            // Verify every b rank is on the same inner step. Cheap (one int) and
            // only on when asked, but it converts a truncated-message abort or a
            // silently-mismatched Rayleigh matrix into a named diagnosis.
            int ib_min = ib, ib_max = ib;
            MPI_Allreduce(MPI_IN_PLACE, &ib_min, 1, MPI_INT, MPI_MIN, b_comm);
            MPI_Allreduce(MPI_IN_PLACE, &ib_max, 1, MPI_INT, MPI_MAX, b_comm);
            if (ib_min != ib_max) {
              if (mpi_rank_b == 0)
                std::cerr << " sbd: ERROR --single_spin: b ranks disagree on the"
                             " inner step (ib ranges " << ib_min << ".." << ib_max
                          << " at outer iteration " << it << ").\n"
                             "      The Rayleigh matrix would combine different"
                             " subspaces. This is a solver bug, not an input"
                             " problem -- please report the log." << std::endl;
              MPI_Abort(b_comm, 1);
            }
          }
          for (int kb = 0; kb <= ib; ++kb)
            for (int jb = 0; jb <= ib; ++jb)
              U[jb + nb*kb] = H[jb + nb*kb];
          if (tmr && tmr->on) tmr->n_inner += nv2;
          // Eigen-decompose on b-rank 0 and broadcast, rather than letting every
          // rank call LAPACK on (bitwise identical) input and trusting the results
          // to agree. If they diverged even slightly, ranks could order or sign
          // near-degenerate eigenvectors differently and would then build DIFFERENT
          // Ritz vectors from the same basis -- a silent wrong answer, not a crash.
          // Cost is nb^2 + nb doubles (~6 KB at nb=31), negligible against the
          // K-length allreduces already in this loop.
          if (b_comm_size > 1) {
            if (mpi_rank_b == 0) hp_numeric::MatHeev(jobz, uplo, ib+1, U, nb, E);
            MPI_Bcast(U, nb*nb, GetMpiType<ElemT>::MpiT, 0, b_comm);
            MPI_Bcast(E, nb, GetMpiType<RealT>::MpiT, 0, b_comm);
          } else {
            hp_numeric::MatHeev(jobz, uplo, ib+1, U, nb, E);
          }
          // CANONICALIZE EIGENVECTOR SIGNS. Broadcasting over b_comm is not
          // sufficient when h_comm > 1 or t_comm > 1: there is then one b-rank-0
          // PER h/t replica, each running its own LAPACK on its own copy. The
          // input matrix is bitwise identical, but an eigenvector is only defined
          // up to sign, and nothing obliges two independent LAPACK invocations to
          // pick the same one for a near-degenerate root. Each replica would then
          // build a Ritz vector of opposite sign from the same basis, and since
          // mult() allreduces over t_comm and h_comm (mult.h:207-208, :264-265),
          // those opposite vectors get SUMMED -- cancelling to noise.
          //
          // Observed exactly that way at b_comm=12 with h_comm=2 or 4: converging
          // normally to a residual of 2.8e-4, then the residual GROWING as the
          // roots tighten and the sign choice becomes ambiguous, diverging to -36
          // and finally aborting with MPI_ERR_TRUNCATE. b_comm == mpi_size (one
          // replica) always worked, which is what made this look like an h_comm
          // bug rather than a sign-convention bug.
          //
          // Fixing it by sign convention rather than a wider broadcast keeps this
          // purely local: no new communicator, no extra collective, and it also
          // covers any future caller that replicates the solve differently. The
          // convention -- make the first entry of largest magnitude positive --
          // is deterministic and depends only on the eigenvector's own values.
          for (int c = 0; c <= ib; ++c) {
            ElemT * col = U + static_cast<size_t>(nb) * c;
            int piv = 0;
            RealT best = RealT(0);
            for (int r = 0; r <= ib; ++r) {
              const RealT a = std::abs(GetReal(col[r]));
              if (a > best + RealT(1.0e-12)) { best = a; piv = r; }
            }
            if (GetReal(col[piv]) < RealT(0))
              for (int r = 0; r <= ib; ++r) col[r] = -col[r];
          }
          if (tmr && tmr->on) tmr->t_subbuild += _wtime() - _ts;

          double _tr = (tmr && tmr->on) ? _wtime() : 0.0;
          bool all_converged = true;
          std::vector<int> unconverged;
          for (int p = 0; p < nroot; p++) {
            // Ritz[p] = sum_kb U[kb,p] v[kb];  res = sum_kb U[kb,p] Hv[kb].
            // Parallelize over the CSF index i (independent); loop order swapped
            // so each i accumulates its full kb-sum in one thread iteration.
            std::vector<ElemT> & Rp = Ritz[p];
            double _tb = (tmr && tmr->on) ? _wtime() : 0.0;
#pragma omp parallel for if(_ss_par(2,K))
            for (int i = 0; i < K; i++) {
              ElemT ri = ElemT(0.0), si = ElemT(0.0);
              for (int kb = 0; kb <= ib; kb++) {
                ElemT x = U[kb + nb*p];
                ri += v[kb][i]*x;
                si += Hv[kb][i]*x;
              }
              Rp[i] = ri;
              res[i] = si - E[p]*ri;
            }
            if (tmr && tmr->on) { tmr->t_ritz_build += _wtime() - _tb; _tb = _wtime(); }
            RealT nrmw = _local_normalize<ElemT,RealT>(Ritz[p], b_comm);
            (void)nrmw;
            norm_r[p] = _local_normalize<ElemT,RealT>(res, b_comm);
            if (tmr && tmr->on) tmr->t_ritz_norm += _wtime() - _tb;
            if (norm_r[p] >= eps) { all_converged = false; unconverged.push_back(p); }
          }

          if (mpi_rank_h==0 && mpi_rank_t==0 && mpi_rank_b==0) {
            RealT maxr = RealT(0);
            for (int p=0;p<nroot;p++) if (norm_r[p]>maxr) maxr=norm_r[p];
            std::cout << " DavidsonSS iteration " << it << "." << ib
                      << " (max_res=" << maxr << "):";
            for (int p = 0; p < nroot; p++) std::cout << " " << E[p];
            std::cout << std::endl;
          }

          if (tmr && tmr->on) tmr->t_ritz += _wtime() - _tr;

          if (all_converged) { do_continue = false; break; }

          double _tc = (tmr && tmr->on) ? _wtime() : 0.0;
          int appended = 0;
          for (size_t u = 0; u < unconverged.size(); u++) {
            if (ib + 1 + appended >= nb) break;
            int p = unconverged[u];
            // rebuild residual for root p. residual r = H*ritz - E*ritz; form it
            // with the same (un-normalized) ritz scaling on both terms (the old
            // bug normalized ritz between the two terms -> inconsistent residual
            // -> stalled/divergent Davidson at large K). Parallelize over i.
            std::vector<ElemT> & Rp = Ritz[p];
            double _td = (tmr && tmr->on) ? _wtime() : 0.0;
#pragma omp parallel for if(_ss_par(3,K))
            for (int i = 0; i < K; i++) {
              ElemT ri = ElemT(0.0), si = ElemT(0.0);
              for (int kb = 0; kb <= ib; kb++) {
                ElemT x = U[kb + nb*p];
                ri += v[kb][i]*x;
                si += Hv[kb][i]*x;
              }
              Rp[i] = ri;
              res[i] = si - E[p]*ri;
            }
            if (tmr && tmr->on) {
              tmr->t_corr_ritz += _wtime() - _td; tmr->n_corr_ritz++; _td = _wtime();
            }
            _local_normalize<ElemT,RealT>(res, b_comm);
            if (tmr && tmr->on) { tmr->t_corr_norm += _wtime() - _td; _td = _wtime(); }
            int slot = ib + 1 + appended;
            std::vector<ElemT> & vslot = v[slot];
            // Diagonal (Jacobi) preconditioner, with a MEANINGFUL floor on the
            // denominator.
            //
            // eps_reg = 1e-12 only ever caught exact division by zero, which is
            // not the failure that occurs in practice. When E[p] lies INSIDE the
            // spread of ndiag rather than below it, some denominators are merely
            // small -- 1e-3, say -- and the correction picks up a ~1000x
            // amplification on those components. MGS cannot repair that (the
            // direction is genuinely in the space).
            //
            // HARDENING, NOT A DIAGNOSED FIX. This was written while chasing a
            // reported divergence at K=113394, b_comm=12 (residual 0.029 -> 5.13
            // on the first appended correction, then a slow drift to -37 over 100
            // outer iterations). That report is NOT reproduced here: the same
            // settings at K=93331 b_comm=12, and with the core shift moved outside
            // the FCIDUMP to match the reporter's integrals, both converge
            // normally and agree with b_comm=1 to 1e-13. So this floor removes a
            // real fragility but is not confirmed to be that bug's cause -- do not
            // treat the report as closed on the strength of it.
            //
            // Floor the magnitude at a scale-aware value and PRESERVE THE SIGN:
            // flipping the sign of a denominator flips the correction direction,
            // which turns a descent step into an ascent step. The scale follows
            // |E[p]| so it is meaningful for both core-shifted and bare integrals.
            const RealT den_floor =
                std::max(static_cast<RealT>(1.0e-8),
                         static_cast<RealT>(1.0e-10) * std::abs(E[p]));
            #pragma omp parallel for if(_ss_par(4,K))
            for (int i = 0; i < K; i++) {
              RealT den = E[p] - ndiag[i];
              const RealT ad = std::abs(den);
              if (ad < den_floor) den = (den < RealT(0) ? -den_floor : den_floor);
              vslot[i] = res[i]/den;
            }
            if (tmr && tmr->on) { tmr->t_corr_prec += _wtime() - _td; _td = _wtime(); }
            // MGS (two passes) against all current basis + accepted corrections.
            // Classical MGS keeps the kb loop sequential (each subtraction depends
            // on the previous), but the two K-length loops per kb (inner product +
            // axpy) were each spawning their own OpenMP region: ~4*slot fork/join
            // barriers per correction over tiny K-length work. At many threads
            // across NUMA sockets that dwarfs the work (correction+MGS regressed
            // 29.7s -> 73.7s from 12 to 48 threads, K=113k). Fix: ONE persistent
            // parallel region for the whole MGS; the K-loops run as omp-for inside
            // it and the per-kb inner product reduces through a per-thread partial
            // array (same pattern as the subspace build). Barriers remain (2*slot)
            // but there is no fork/join or first-touch storm. Numerics identical:
            // same order, same two passes, same Conjugate(v_kb).vslot dot.
            // NOTE ON COLLECTIVE SAFETY. `K` is the LOCAL slice length, so ranks
            // can take DIFFERENT sides of this branch (verified: at b_comm=2 with
            // a rank-dependent threshold, rank 0 took the parallel path and rank 1
            // the serial one, and the run still gave the right energy). That is
            // safe only because the two paths issue the SAME sequence of b_comm
            // collectives: one reduction per (pass, kb), in the same order --
            // _bcomm_sum inside `omp master` on this side, the _bcomm_sum at the
            // end of _local_inner on the other. If you change either path's
            // collective count or order, they must be changed together, or ranks
            // on opposite sides of the branch will deadlock.
            if (_ss_par(5,K)) {
              const int nth = omp_get_max_threads();
              std::vector<ElemT> mgs_partial(nth, ElemT(0.0));
              ElemT ol_shared = ElemT(0.0);
#pragma omp parallel
              {
                const int tid = omp_get_thread_num();
                for (int pass = 0; pass < 2; pass++) {
                  for (int kb = 0; kb < slot; kb++) {
                    const std::vector<ElemT> & vkb = v[kb];
                    ElemT loc = ElemT(0.0);
#pragma omp for
                    for (int i = 0; i < K; i++) loc += Conjugate(vkb[i]) * vslot[i];
                    mgs_partial[tid] = loc;
#pragma omp barrier
                    // `omp master`, NOT `omp single`: this region contains an MPI
                    // call, and under MPI_THREAD_FUNNELED only the thread that
                    // initialized MPI (the master) may call it. `single` guarantees
                    // one thread, not the master thread, so it is not FUNNELED-safe.
                    // `master` has no implicit barrier (`single` does), hence the
                    // explicit barrier below -- without it the other threads would
                    // read ol_shared before the master has written it.
#pragma omp master
                    {
                      ElemT s = ElemT(0.0);
                      for (int t = 0; t < nth; ++t) s += mgs_partial[t];
                      // Global overlap: this rank holds only its CSF slice.
                      _bcomm_sum(&s, 1, b_comm);
                      ol_shared = s;
                    }
#pragma omp barrier
                    const ElemT ol = ol_shared;
                    // SBD_SS_CHECK_MGS=1: recompute this overlap serially on the
                    // master and compare against the threaded reduction, BEFORE it
                    // is used. Catches a wrong reduction at the exact (pass,kb)
                    // where it first happens, rather than as a wrong energy later.
                    if (_ss_check_mgs()) {
#pragma omp master
                      {
                        ElemT ref = ElemT(0.0);
                        for (int i = 0; i < K; i++) ref += Conjugate(vkb[i]) * vslot[i];
                        _bcomm_sum(&ref, 1, b_comm);
                        const double d = std::abs(GetReal(ref - ol));
                        const double scale = std::max(1.0, std::abs(GetReal(ref)));
                        if (d / scale > 1.0e-10) {
                          int wr = 0; MPI_Comm_rank(MPI_COMM_WORLD, &wr);
                          std::cerr << " sbd: ERROR fused MGS overlap wrong: pass="
                                    << pass << " kb=" << kb << " slot=" << slot
                                    << " got " << GetReal(ol) << " want " << GetReal(ref)
                                    << " rel " << (d/scale) << " (K=" << K
                                    << ", threads=" << omp_get_num_threads()
                                    << ", nth=" << nth << ", world rank " << wr << ")"
                                    << std::endl;
                          MPI_Abort(MPI_COMM_WORLD, 1);
                        }
                      }
#pragma omp barrier
                    }
#pragma omp for
                    for (int i = 0; i < K; i++) vslot[i] -= vkb[i]*ol;
                  }
                }
              }
            } else {
              for (int pass = 0; pass < 2; pass++)
                for (int kb = 0; kb < slot; kb++) {
                  ElemT ol = _local_inner(v[kb], vslot, b_comm);
                  const std::vector<ElemT> & vkb = v[kb];
                  for (int i = 0; i < K; i++) vslot[i] -= vkb[i]*ol;
                }
            }
            // NOTE: this normalization is load-bearing -- an unnormalized
            // correction vector makes the energy never converge (the historical
            // pathology). It stays exactly as-is; only timing is added around it.
            RealT nv = _local_normalize<ElemT,RealT>(v[slot], b_comm);
            if (tmr && tmr->on) tmr->t_corr_mgs += _wtime() - _td;
            // `nv` is safe to branch on across ranks: _local_normalize allreduces
            // the squared norm BEFORE taking the square root, so every rank gets
            // the same bits and makes the same drop/keep decision -- and therefore
            // the same `appended`, the same loop trip count, and the same
            // collective sequence. The same argument covers norm_r[p] and E[p]
            // (broadcast from b-rank 0). Any NEW branch in this loop must rest on
            // a globally reduced quantity for the same reason.
            if (nv < tau_drop) continue;
            appended++;
          }
          if (tmr && tmr->on) tmr->t_correct += _wtime() - _tc;
          if (appended == 0) break;
          ncur = ib + 1 + appended;
          if (ncur >= nb) break;
        }
        if (!do_continue) break;
        double _trs = (tmr && tmr->on) ? _wtime() : 0.0;
        // Thick restart: reseed with the lowest `keep` Ritz vectors (not just the
        // nroot targets), so the collapsed subspace restarts from `keep` instead
        // of climbing from nroot again -> fewer, shorter re-climbs.
        //
        // The key to a NET matvec saving is that we carry the Ritz vectors' Hv
        // too, as the SAME linear combination of the existing basis Hv:
        //   Ritz[p]  = sum_kb U[kb,p] v[kb],   H*Ritz[p] = sum_kb U[kb,p] Hv[kb].
        // By linearity these are consistent with NO fresh matvec. nhv_valid then
        // tells the next inner cycle to skip re-matvec-ing them. Without the Hv
        // carry, reseeding `keep` vectors would cost `keep` matvecs per restart and
        // make thick restart slower, not faster.
        //
        // The kept Ritz vectors are eigenvectors of the symmetric Rayleigh matrix,
        // so orthonormal in exact arithmetic -- but at large K rounding lets that
        // drift, and the inner loop's Rayleigh-Ritz uses a STANDARD eigensolve that
        // assumes an orthonormal basis (no overlap matrix). A non-orthonormal seed
        // then yields spurious eigenvalues (observed: correct energy at K=3906, but
        // a nonsense -19.47 at K=113394). Fix: re-orthonormalize the carried block
        // with modified Gram-Schmidt, applying the SAME scale/subtract operations
        // to Hv as to v -- since Hv[p] = H v[p] and H is linear, an identical linear
        // combination of the v's is mirrored by that combination of the Hv's, so
        // the v<->Hv correspondence is preserved exactly and still needs no matvec.
        int keep = std::min(nkeep, ib + 1);
        std::vector<std::vector<ElemT>> Vnew(keep, std::vector<ElemT>(K));
        std::vector<std::vector<ElemT>> HVnew(keep, std::vector<ElemT>(K));
        for (int p = 0; p < keep; p++) {
          std::vector<ElemT> & vp = Vnew[p];
          std::vector<ElemT> & hp = HVnew[p];
#pragma omp parallel for if(_ss_par(6,K))
          for (int i = 0; i < K; i++) {
            ElemT ri = ElemT(0.0), hi = ElemT(0.0);
            for (int kb = 0; kb <= ib; kb++) {
              ElemT x = U[kb + nb*p];
              ri += v[kb][i]  * x;
              hi += Hv[kb][i] * x;
            }
            vp[i] = ri;
            hp[i] = hi;
          }
        }
        // Re-orthonormalize {Vnew} by MGS, mirroring every op onto {HVnew}. TWO
        // passes (DGKS): a single MGS pass leaves residual non-orthogonality that,
        // for the many near-degenerate kept Ritz vectors at large K, is enough to
        // make the standard (no-overlap) Rayleigh-Ritz return a spurious eigenvalue
        // -- the exact failure seen at K=113394 (keep=15 wrong, keep=1 fine because
        // one vector has nothing to orthogonalize against). Both K sizes run the
        // same code; only the accumulated rounding over the larger vectors differs,
        // so the second pass is what makes it robust. Every subtraction/scale is
        // applied identically to HVnew, so Hv stays = H v with no fresh matvec.
        for (int pass = 0; pass < 2; pass++)
        for (int p = 0; p < keep; p++) {
          for (int q = 0; q < p; q++) {
            ElemT ol = _local_inner(Vnew[q], Vnew[p], b_comm);   // <v_q, v_p>
#pragma omp parallel for if(_ss_par(7,K))
            for (int i = 0; i < K; i++) {
              Vnew[p][i]  -= Vnew[q][i]  * ol;
              HVnew[p][i] -= HVnew[q][i] * ol;   // same subtraction => Hv stays = H v
            }
          }
          RealT nrm = _local_normalize<ElemT,RealT>(Vnew[p], b_comm);   // v_p /= |v_p|
          if (nrm > RealT(0)) {
            RealT inv = RealT(1) / nrm;
#pragma omp parallel for if(_ss_par(8,K))
            for (int i = 0; i < K; i++) HVnew[p][i] *= inv;     // scale Hv identically
          }
        }
        for (int p = 0; p < keep; p++) { v[p] = Vnew[p]; Hv[p] = HVnew[p]; }
        nseed = keep;
        nhv_valid = keep;   // Hv[0..keep) already valid -> next cycle skips their matvec
        if (tmr && tmr->on) tmr->t_restart += _wtime() - _trs;
      }

      if (tmr && tmr->on) tmr->t_loop_total += _wtime() - _tloop0;

      // Exhausting max_iteration is NOT convergence: the loop simply stops and
      // the current Ritz values are returned. Without this warning that is
      // indistinguishable from a converged answer in the output -- the caller
      // sees a plausible energy and no indication the residual never met eps.
      // (do_continue is still true only if the convergence test never passed.)
      if (do_continue && mpi_rank_h == 0 && mpi_rank_t == 0 && mpi_rank_b == 0) {
        RealT worst = RealT(0);
        for (int p = 0; p < nroot; p++) if (norm_r[p] > worst) worst = norm_r[p];
        std::cerr << " sbd: WARNING --single_spin did NOT converge in "
                  << max_iteration << " outer iteration(s): max residual "
                  << worst << " > tolerance " << eps << ".\n"
                  << "      The reported energies are the current Ritz values and"
                     " are NOT variationally converged.\n"
                  << "      Raise --iteration (it counts thick-restart CYCLES, not"
                     " matvecs) or loosen --tolerance." << std::endl;
      }

      if (tmr && tmr->on && mpi_rank_h==0 && mpi_rank_t==0 && mpi_rank_b==0) {
        // Sum of the individually timed regions inside the inner-cycle loop.
        // Anything left over is work in the loop that is still not instrumented
        // (vector copies, allocations, the Ritz->Wcsf handoff, MPI waits).
        const double acc = tmr->t_mv_outer + tmr->t_subbuild
                         + tmr->t_ritz_build + tmr->t_ritz_norm
                         + tmr->t_corr_ritz + tmr->t_corr_norm
                         + tmr->t_corr_prec + tmr->t_corr_mgs
                         + tmr->t_restart;
        const double tot = tmr->t_loop_total;
        auto pct = [&](double t) { return (tot > 0.0) ? (100.0 * t / tot) : 0.0; };
        std::cout << "sbd: SS-TIMING (K=" << Kg << ", threads=" << omp_get_max_threads()
                  << ", nroot=" << nroot << ", nb=" << nb << ", keep=" << nkeep << ")\n"
                  << "  setup: ndiag         = " << tmr->t_ndiag << " s, seed = " << tmr->t_seed << " s\n"
                  << "  LOOP TOTAL           = " << tot << " s\n"
                  << "  matvec (outer)       = " << tmr->t_mv_outer << " s (" << pct(tmr->t_mv_outer) << "%)\n"
                  << "    [method1 split] project_up = " << tmr->t_proj_up
                  << " s, mult = " << tmr->t_mult
                  << " s, project_down = " << tmr->t_proj_down << " s ("
                  << tmr->n_matvec << " calls)\n"
                  << "  subspace build       = " << tmr->t_subbuild << " s (" << pct(tmr->t_subbuild)
                  << "%, " << tmr->n_inner << " inner products)\n"
                  << "  ritz build           = " << tmr->t_ritz_build << " s (" << pct(tmr->t_ritz_build) << "%)\n"
                  << "  ritz normalize       = " << tmr->t_ritz_norm  << " s (" << pct(tmr->t_ritz_norm) << "%)\n"
                  << "  corr: ritz REBUILD   = " << tmr->t_corr_ritz  << " s (" << pct(tmr->t_corr_ritz)
                  << "%, " << tmr->n_corr_ritz << " rebuilds)  <-- duplicate of 'ritz build'\n"
                  << "  corr: normalize      = " << tmr->t_corr_norm  << " s (" << pct(tmr->t_corr_norm) << "%)\n"
                  << "  corr: precondition   = " << tmr->t_corr_prec  << " s (" << pct(tmr->t_corr_prec) << "%)\n"
                  << "  corr: MGS + norm     = " << tmr->t_corr_mgs   << " s (" << pct(tmr->t_corr_mgs) << "%)\n"
                  << "  restart              = " << tmr->t_restart    << " s (" << pct(tmr->t_restart) << "%)\n"
                  << "  ---------------------\n"
                  << "  accounted            = " << acc << " s (" << pct(acc) << "%)\n"
                  << "  UNACCOUNTED          = " << (tot - acc) << " s (" << pct(tot - acc) << "%)\n";
      }

      Wcsf.resize(nroot);
      Eout.resize(nroot);
      for (int p = 0; p < nroot; p++) { Wcsf[p] = Ritz[p]; Eout[p] = E[p]; }
      free(H); free(U); free(E);
    }

    /**
       Matrix-free (method 0) projected multi-root solver. matvec = V^T H (V x)
       with H applied on the fly via the integral-driven mult overload.
    */
    template <typename ElemT, typename RealT, typename DetsContainer>
    void DavidsonMultiRootProjected(const std::vector<ElemT> & hii,
                                    const SpinProjector & V,
                                    std::vector<std::vector<ElemT>> & Wcsf,
                                    std::vector<RealT> & Eout,
                                    const DetsContainer & det,
                                    const size_t bit_length,
                                    const size_t norb,
                                    const DetIndexMap & idxmap,
                                    const std::vector<ExcitationLookup> & exidx,
                                    const ElemT & I0,
                                    const oneInt<ElemT> & I1,
                                    const twoInt<ElemT> & I2,
                                    MPI_Comm h_comm,
                                    MPI_Comm b_comm,
                                    MPI_Comm t_comm,
                                    int max_iteration,
                                    int num_block,
                                    int nroot,
                                    RealT eps,
                                    int nkeep) {
      const size_t ndet = det.size();
      SSTimers tm;
      auto matvec = [&](const std::vector<ElemT> & xc, std::vector<ElemT> & yc) {
        std::vector<ElemT> xdet, ydet(ndet, ElemT(0.0));
        if (!tm.on) {
          project_up(V, xc, xdet, ndet);
          Zero(ydet);
          mult(hii, xdet, ydet, bit_length, norb, det, idxmap, exidx, I0, I1, I2,
               h_comm, b_comm, t_comm);
          project_down(V, ydet, yc);
          return;
        }
        double t0 = _wtime();
        project_up(V, xc, xdet, ndet);
        double t1 = _wtime(); tm.t_proj_up += t1 - t0;
        Zero(ydet);
        mult(hii, xdet, ydet, bit_length, norb, det, idxmap, exidx, I0, I1, I2,
             h_comm, b_comm, t_comm);
        double t2 = _wtime(); tm.t_mult += t2 - t1;
        project_down(V, ydet, yc);
        tm.t_proj_down += _wtime() - t2;
        tm.t_matvec += _wtime() - t0; tm.n_matvec++;
      };
      // hii is only filled for k % mpi_size_h == mpi_rank_h (qcham.h:397-400), so
      // it is an h_comm-SHARDED diagonal, not a complete one. mult() tolerates
      // that because it allreduces its output over h_comm, but we read the
      // diagonal element-wise, so we need the completed vector. This mirrors
      // davidson.h:77 / :356 / :693, which all call GetTotalD before using hii.
      // Without it, h_comm_size > 1 silently produced a badly wrong energy
      // (measured: -164.079118447 instead of -108.729420078 on N2 top100 at
      // -np 2 --b_comm_size 1 --t_comm_size 1).
      std::vector<ElemT> dii;
      GetTotalD(hii, dii, h_comm);
      // Exact projected diagonal for the preconditioner (intra-block H via Hij).
      std::vector<RealT> ndiag_exact;
      {
        double _t0 = tm.on ? _wtime() : 0.0;
        build_projected_diagonal<ElemT,RealT>(V, dii, det, bit_length, norb,
                                              I0, I1, I2, ndiag_exact);
        if (tm.on) tm.t_ndiag += _wtime() - _t0;
      }
      _davidson_projected_core<ElemT,RealT>(dii, V, matvec, Wcsf, Eout,
                                            h_comm, b_comm, t_comm,
                                            max_iteration, num_block, nroot, eps,
                                            nkeep, &tm, &ndiag_exact);
    }

    /**
       Stored-matrix (method 1) projected multi-root solver. matvec = V^T H (V x)
       with H applied via the pre-built sparse Hamiltonian (ih/jh/hij/len/slide).
       Faster than matrix-free when the matrix fits; single-spin's basis is small
       (b_comm==1) so it always fits.
    */
    template <typename ElemT, typename RealT, typename DetsContainer>
    void DavidsonMultiRootProjectedStored(const std::vector<ElemT> & hii,
                                          const SpinProjector & V,
                                          std::vector<std::vector<ElemT>> & Wcsf,
                                          std::vector<RealT> & Eout,
                                          size_t ndet,
                                          const DetsContainer & det,
                                          const size_t bit_length,
                                          const size_t norb,
                                          const ElemT & I0,
                                          const oneInt<ElemT> & I1,
                                          const twoInt<ElemT> & I2,
                                          const std::vector<std::vector<size_t*>> & ih,
                                          const std::vector<std::vector<size_t*>> & jh,
                                          const std::vector<std::vector<ElemT*>> & hij,
                                          const std::vector<std::vector<size_t>> & len,
                                          const std::vector<int> & slide,
                                          MPI_Comm h_comm,
                                          MPI_Comm b_comm,
                                          MPI_Comm t_comm,
                                          int max_iteration,
                                          int num_block,
                                          int nroot,
                                          RealT eps,
                                          int nkeep) {
      SSTimers tm;
      auto matvec = [&](const std::vector<ElemT> & xc, std::vector<ElemT> & yc) {
        std::vector<ElemT> xdet, ydet(ndet, ElemT(0.0));
        if (!tm.on) {
          project_up(V, xc, xdet, ndet);
          Zero(ydet);
          mult(hii, ih, jh, hij, len, slide, xdet, ydet, h_comm, b_comm, t_comm);
          project_down(V, ydet, yc);
          return;
        }
        double t0 = _wtime();
        project_up(V, xc, xdet, ndet);
        double t1 = _wtime(); tm.t_proj_up += t1 - t0;
        Zero(ydet);
        mult(hii, ih, jh, hij, len, slide, xdet, ydet, h_comm, b_comm, t_comm);
        double t2 = _wtime(); tm.t_mult += t2 - t1;
        project_down(V, ydet, yc);
        tm.t_proj_down += _wtime() - t2;
        tm.t_matvec += _wtime() - t0; tm.n_matvec++;
      };
      // See the matrix-free overload above: hii is h_comm-sharded, so complete it
      // with GetTotalD before reading diagonal elements directly. mult() keeps
      // using the sharded hii, since it allreduces its own output over h_comm.
      std::vector<ElemT> dii;
      GetTotalD(hii, dii, h_comm);
      // Exact projected diagonal for the preconditioner (intra-block H via Hij).
      std::vector<RealT> ndiag_exact;
      {
        double _t0 = tm.on ? _wtime() : 0.0;
        build_projected_diagonal<ElemT,RealT>(V, dii, det, bit_length, norb,
                                              I0, I1, I2, ndiag_exact);
        if (tm.on) tm.t_ndiag += _wtime() - _t0;
      }
      _davidson_projected_core<ElemT,RealT>(dii, V, matvec, Wcsf, Eout,
                                            h_comm, b_comm, t_comm,
                                            max_iteration, num_block, nroot, eps,
                                            nkeep, &tm, &ndiag_exact);
    }

  } // namespace gdb
} // namespace sbd

#endif
