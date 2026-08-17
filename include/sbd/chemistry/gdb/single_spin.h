/**
@file sbd/chemistry/gdb/single_spin.h
@brief Option-2 single-spin projection for the general determinant basis.

Projects the (Anderson-completed) determinant subspace onto a single target
total spin S BEFORE diagonalizing, then runs the block Davidson-Liu solver in
the reduced target-S CSF space. Single-spin by construction (no post-hoc <S^2>
filtering) and a smaller solve.

Restricted to b_comm_size == 1: the whole determinant vector is local to each
task, so the projector V (block-diagonal by spatial configuration) is entirely
rank-local. With b_comm_size > 1 a configuration's determinants scatter across
ranks (SBD sorts by the full interleaved bitstring); the caller must guard
against that case.

Conventions: determinant bit 2p = alpha orbital p, 2p+1 = beta orbital p
(matches occupation.h getocc). Spatial config code per orbital: 0 empty,
1 singly occupied, 2 doubly occupied.
*/
#ifndef SBD_CHEMISTRY_GDB_SINGLE_SPIN_H
#define SBD_CHEMISTRY_GDB_SINGLE_SPIN_H

#include <algorithm>
#include <functional>
#include <map>
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

    /// V: rank-local block-sparse det<->CSF map. csf_offset[b] = first CSF column
    /// index of block b in the global CSF ordering; total_csf = k.
    struct SpinProjector {
      std::vector<ConfigBlock> blocks;
      std::vector<int> csf_offset;
      int total_csf = 0;
      /// Largest block_dim over all blocks. Number of probe matvecs needed to
      /// extract the exact block-diagonal of the projected Hamiltonian.
      int max_block_dim = 0;
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

    /// Build the rank-local single-spin projector V for target spin multiplicity
    /// (2S+1: 1=singlet, 2=doublet, 3=triplet, ...) and spin projection Sz
    /// (n_up open-shell alphas is fixed per config by Sz).
    /// Requires b_comm_size == 1 (caller guards).
    template <typename ElemT, typename DetsContainer>
    SpinProjector build_config_projector(const DetsContainer & det,
                                         size_t bit_length, int norb,
                                         int multiplicity, int Sz2 /* 2*Sz */) {
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
        // Sz2 is taken from det[0] by the caller and assumed common to all dets.
        // If a config's open-shell count has the wrong parity for that Sz, the
        // determinant list is not a single Sz sector and n_up would be silently
        // wrong (truncating division) -- skip rather than build a bad block.
        if (((n_open + Sz2) % 2) != 0 || n_up < 0 || n_up > n_open) continue;

        if (coeff_cache.find(n_open) == coeff_cache.end()) {
          int bd, nc;
          coeff_cache[n_open] = _canonical_csf_coeffs(n_open, n_up, s2_target, bd, nc);
          dim_cache[n_open] = bd; ncsf_cache[n_open] = nc;
        }
        int block_dim = dim_cache[n_open];
        int n_csf = ncsf_cache[n_open];
        if (n_csf == 0) continue;    // no target-S CSF for this config

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
        if (block_dim > V.max_block_dim) V.max_block_dim = block_dim;
        V.blocks.push_back(std::move(blk));
        V.csf_offset.push_back(running);
        running += n_csf;
      }
      V.total_csf = running;
      return V;
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
      const int K = V.total_csf;
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
      y_csf.assign(V.total_csf, ElemT(0.0));
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

    // ---- local (b_comm==1) CSF-space vector helpers ---------------------------
    // CSF vectors are not distributed (single b rank), so inner products and
    // norms are plain local reductions. mult() inside the matvec still uses the
    // t/h communicators for its own parallelism.

    // ElemT may be complex, so we cannot use an OpenMP reduction clause on it
    // directly. Accumulate per-thread partial sums into a scratch array and
    // combine serially (thread count is small vs the K-length loop).
    template <typename ElemT>
    ElemT _local_inner(const std::vector<ElemT> & a, const std::vector<ElemT> & b) {
      const size_t n = a.size();
      if (n <= 4096) {   // small: parallel overhead not worth it
        ElemT s = ElemT(0.0);
        for (size_t i = 0; i < n; ++i) s += Conjugate(a[i]) * b[i];
        return s;
      }
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
      ElemT s = ElemT(0.0);
      for (int t = 0; t < nth; ++t) s += partial[t];
      return s;
    }

    template <typename ElemT, typename RealT>
    RealT _local_normalize(std::vector<ElemT> & a) {
      const size_t n = a.size();
      RealT n2 = RealT(0.0);
      if (n <= 4096) {
        for (size_t i = 0; i < n; ++i) n2 += GetReal(Conjugate(a[i]) * a[i]);
      } else {
#pragma omp parallel for reduction(+:n2)
        for (size_t i = 0; i < n; ++i) n2 += GetReal(Conjugate(a[i]) * a[i]);
      }
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

       @param[out] Wcsf  nroot CSF-space eigenvectors (each length V.total_csf)
       @param[out] Eout  nroot lowest energies (ascending)
    */
    /**
       Core block Davidson-Liu in the target-S CSF space (b_comm==1). The H*v
       matvec is supplied as a callable so the same core serves both the
       matrix-free (method 0) and stored-matrix (method 1) projected solvers.
       `matvec(x_csf, y_csf)` must implement y = V^T H (V x).
    */
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

      const int K = V.total_csf;

      // Subspace (Krylov) cap before collapse. Modest size is fine — with a
      // correct residual the block Davidson-Liu converges in tens of iterations
      // at small nb (verified in Python: nroot=2, nb=12 -> 57 iters at K=3906).
      // Keep RAM low: CSF vectors are held per rank (b_comm==1), so 2*nb*K.
      int nb = num_block;
      int nb_min = nroot + std::max(nroot, 10);
      if (nb < nb_min) nb = nb_min;
      if (nb > K) nb = K;
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
        if (nb > K) nb = K;
        // nkeep must still leave at least one growth slot after the K clamp.
        if (nkeep > nb - 1) nkeep = nb - 1;
        if (nkeep < nroot) nkeep = std::min(nroot, nb - 1);
        if (nkeep < 1) nkeep = 1;
      }

      if (mpi_rank_h == 0 && mpi_rank_t == 0 && mpi_rank_b == 0) {
        std::cout << "sbd(ss): nb=" << nb << " keep=" << nkeep
                  << " growth=" << (nb - nkeep) << " (K=" << K << ")"
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
        std::vector<int> order(K);
        for (int i = 0; i < K; ++i) order[i] = i;
        const int ntake = std::min(nroot, K);
        std::partial_sort(order.begin(), order.begin() + ntake, order.end(),
                          [&](int a, int b) { return ndiag[a] < ndiag[b]; });
        for (int p = 0; p < nroot; p++) {
          std::fill(v[p].begin(), v[p].end(), ElemT(0.0));
          if (p < K) v[p][order[p]] = ElemT(1.0);
        }
        if (mpi_rank_h == 0 && mpi_rank_t == 0 && mpi_rank_b == 0 && ntake > 0)
          std::cout << "sbd(ss): seed CSFs (lowest ndiag) first="
                    << order[0] << " diag=" << ndiag[order[0]] << std::endl;
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
#pragma omp parallel for schedule(static) if(K > 4096)
          for (int idx = 0; idx < nv2; ++idx) {
            int jb = idx / (ib + 1);
            int kb = idx % (ib + 1);
            const std::vector<ElemT> & a = v[jb];
            const std::vector<ElemT> & b = Hv[kb];
            ElemT s = ElemT(0.0);
            for (int i = 0; i < K; ++i) s += Conjugate(a[i]) * b[i];
            H[jb + nb*kb] = s;
            U[jb + nb*kb] = s;
          }
          if (tmr && tmr->on) tmr->n_inner += nv2;
          hp_numeric::MatHeev(jobz, uplo, ib+1, U, nb, E);
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
#pragma omp parallel for if(K > 4096)
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
            RealT nrmw = _local_normalize<ElemT,RealT>(Ritz[p]);
            (void)nrmw;
            norm_r[p] = _local_normalize<ElemT,RealT>(res);
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
#pragma omp parallel for if(K > 4096)
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
            _local_normalize<ElemT,RealT>(res);
            if (tmr && tmr->on) { tmr->t_corr_norm += _wtime() - _td; _td = _wtime(); }
            int slot = ib + 1 + appended;
            std::vector<ElemT> & vslot = v[slot];
#pragma omp parallel for if(K > 4096)
            for (int i = 0; i < K; i++) {
              RealT den = E[p] - ndiag[i];
              if (std::abs(den) > eps_reg) vslot[i] = res[i]/den;
              else                          vslot[i] = res[i]/(den - eps_reg);
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
            if (K > 4096) {
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
#pragma omp single
                    {
                      ElemT s = ElemT(0.0);
                      for (int t = 0; t < nth; ++t) s += mgs_partial[t];
                      ol_shared = s;
                    }
                    // implicit barrier after single -> ol_shared visible to all
                    const ElemT ol = ol_shared;
#pragma omp for
                    for (int i = 0; i < K; i++) vslot[i] -= vkb[i]*ol;
                  }
                }
              }
            } else {
              for (int pass = 0; pass < 2; pass++)
                for (int kb = 0; kb < slot; kb++) {
                  ElemT ol = _local_inner(v[kb], vslot);
                  const std::vector<ElemT> & vkb = v[kb];
                  for (int i = 0; i < K; i++) vslot[i] -= vkb[i]*ol;
                }
            }
            // NOTE: this normalization is load-bearing -- an unnormalized
            // correction vector makes the energy never converge (the historical
            // pathology). It stays exactly as-is; only timing is added around it.
            RealT nv = _local_normalize<ElemT,RealT>(v[slot]);
            if (tmr && tmr->on) tmr->t_corr_mgs += _wtime() - _td;
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
#pragma omp parallel for if(K > 4096)
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
            ElemT ol = _local_inner(Vnew[q], Vnew[p]);   // <v_q, v_p>
#pragma omp parallel for if(K > 4096)
            for (int i = 0; i < K; i++) {
              Vnew[p][i]  -= Vnew[q][i]  * ol;
              HVnew[p][i] -= HVnew[q][i] * ol;   // same subtraction => Hv stays = H v
            }
          }
          RealT nrm = _local_normalize<ElemT,RealT>(Vnew[p]);   // v_p /= |v_p|
          if (nrm > RealT(0)) {
            RealT inv = RealT(1) / nrm;
#pragma omp parallel for if(K > 4096)
            for (int i = 0; i < K; i++) HVnew[p][i] *= inv;     // scale Hv identically
          }
        }
        for (int p = 0; p < keep; p++) { v[p] = Vnew[p]; Hv[p] = HVnew[p]; }
        nseed = keep;
        nhv_valid = keep;   // Hv[0..keep) already valid -> next cycle skips their matvec
        if (tmr && tmr->on) tmr->t_restart += _wtime() - _trs;
      }

      if (tmr && tmr->on) tmr->t_loop_total += _wtime() - _tloop0;

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
        std::cout << "sbd: SS-TIMING (K=" << K << ", threads=" << omp_get_max_threads()
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
