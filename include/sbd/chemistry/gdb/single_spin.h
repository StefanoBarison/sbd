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

#include <functional>
#include <map>
#include <vector>

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
        V.blocks.push_back(std::move(blk));
        V.csf_offset.push_back(running);
        running += n_csf;
      }
      V.total_csf = running;
      return V;
    }

    // ---- projected matvec y_csf = V^T (mult) (V x_csf) ------------------------

    /// Expand a CSF-space vector to determinant space: x_det = V x_csf (local).
    template <typename ElemT>
    void project_up(const SpinProjector & V, const std::vector<ElemT> & x_csf,
                    std::vector<ElemT> & x_det, size_t ndet) {
      x_det.assign(ndet, ElemT(0.0));
      for (size_t b = 0; b < V.blocks.size(); ++b) {
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
      for (size_t b = 0; b < V.blocks.size(); ++b) {
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

    template <typename ElemT>
    ElemT _local_inner(const std::vector<ElemT> & a, const std::vector<ElemT> & b) {
      ElemT s = ElemT(0.0);
      for (size_t i = 0; i < a.size(); ++i) s += Conjugate(a[i]) * b[i];
      return s;
    }

    template <typename ElemT, typename RealT>
    RealT _local_normalize(std::vector<ElemT> & a) {
      RealT n2 = RealT(0.0);
      for (size_t i = 0; i < a.size(); ++i) n2 += GetReal(Conjugate(a[i]) * a[i]);
      RealT nrm = std::sqrt(n2);
      if (nrm > RealT(0)) for (size_t i = 0; i < a.size(); ++i) a[i] /= nrm;
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
                                  RealT eps) {
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

      // projected diagonal of H in CSF space (approx: V^T diag(H) V, diagonal
      // part) for the preconditioner. Build once: for each CSF column, its
      // diagonal is sum_r coeffs[r,c]^2 * hii[det_index[r]] (rank-local).
      std::vector<RealT> ndiag(K, RealT(0));
      for (size_t bidx = 0; bidx < V.blocks.size(); ++bidx) {
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

      std::vector<std::vector<ElemT>> v(nb, std::vector<ElemT>(K));
      std::vector<std::vector<ElemT>> Hv(nb, std::vector<ElemT>(K));
      std::vector<std::vector<ElemT>> Ritz(nroot, std::vector<ElemT>(K));
      std::vector<ElemT> res(K);
      std::vector<RealT> norm_r(nroot, RealT(0));

      ElemT * H = (ElemT *) calloc((size_t)nb*nb, sizeof(ElemT));
      ElemT * U = (ElemT *) calloc((size_t)nb*nb, sizeof(ElemT));
      RealT * E = (RealT *) malloc((size_t)nb * sizeof(RealT));
      char jobz = 'V', uplo = 'U';

      // seed: unit CSF vectors e_0..e_{nroot-1}, orthonormalized (they already are)
      int nseed = nroot;
      for (int p = 0; p < nroot; p++) {
        std::fill(v[p].begin(), v[p].end(), ElemT(0.0));
        if (p < K) v[p][p] = ElemT(1.0);
      }

      bool do_continue = true;
      for (int it = 0; it < max_iteration && do_continue; it++) {
        int ib = 0, m = 0, ncur = nseed;
        while (true) {
          for (int jb = m; jb < ncur; jb++) matvec(v[jb], Hv[jb]);
          m = ncur; ib = ncur - 1;

          for (int jb = 0; jb <= ib; jb++)
            for (int kb = 0; kb <= ib; kb++)
              H[jb + nb*kb] = _local_inner(v[jb], Hv[kb]);
          for (int jb = 0; jb <= ib; jb++)
            for (int kb = 0; kb <= ib; kb++)
              U[jb + nb*kb] = H[jb + nb*kb];
          hp_numeric::MatHeev(jobz, uplo, ib+1, U, nb, E);

          bool all_converged = true;
          std::vector<int> unconverged;
          for (int p = 0; p < nroot; p++) {
            for (int i = 0; i < K; i++) { Ritz[p][i] = ElemT(0.0); res[i] = ElemT(0.0); }
            for (int kb = 0; kb <= ib; kb++) {
              ElemT x = U[kb + nb*p];
              for (int i = 0; i < K; i++) Ritz[p][i] += v[kb][i]*x;
              for (int i = 0; i < K; i++) res[i]     += Hv[kb][i]*x;
            }
            for (int i = 0; i < K; i++) res[i] -= E[p]*Ritz[p][i];
            RealT nrmw = _local_normalize<ElemT,RealT>(Ritz[p]);
            (void)nrmw;
            norm_r[p] = _local_normalize<ElemT,RealT>(res);
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

          if (all_converged) { do_continue = false; break; }

          int appended = 0;
          for (size_t u = 0; u < unconverged.size(); u++) {
            if (ib + 1 + appended >= nb) break;
            int p = unconverged[u];
            // rebuild residual for root p
            for (int i = 0; i < K; i++) { Ritz[p][i] = ElemT(0.0); res[i] = ElemT(0.0); }
            for (int kb = 0; kb <= ib; kb++) {
              ElemT x = U[kb + nb*p];
              for (int i = 0; i < K; i++) Ritz[p][i] += v[kb][i]*x;
              for (int i = 0; i < K; i++) res[i]     += Hv[kb][i]*x;
            }
            // residual r = H*ritz - E*ritz; form it BEFORE normalizing ritz so
            // both terms use the same (un-normalized) ritz scaling. (Ritz here is
            // already unit-norm from the Rayleigh eigenvector, so this is just the
            // clean residual; the earlier bug normalized ritz between the two
            // terms, producing an inconsistently-scaled residual -> bad correction
            // -> stalled/divergent Davidson at large K.)
            for (int i = 0; i < K; i++) res[i] -= E[p]*Ritz[p][i];
            _local_normalize<ElemT,RealT>(res);
            int slot = ib + 1 + appended;
            for (int i = 0; i < K; i++) {
              RealT den = E[p] - ndiag[i];
              if (std::abs(den) > eps_reg) v[slot][i] = res[i]/den;
              else                          v[slot][i] = res[i]/(den - eps_reg);
            }
            // MGS (two passes) against all current basis + accepted corrections
            for (int pass = 0; pass < 2; pass++)
              for (int kb = 0; kb < slot; kb++) {
                ElemT ol = _local_inner(v[kb], v[slot]);
                for (int i = 0; i < K; i++) v[slot][i] -= v[kb][i]*ol;
              }
            RealT nv = _local_normalize<ElemT,RealT>(v[slot]);
            if (nv < tau_drop) continue;
            appended++;
          }
          if (appended == 0) break;
          ncur = ib + 1 + appended;
          if (ncur >= nb) break;
        }
        if (!do_continue) break;
        // Restart: reseed with the nroot Ritz vectors, re-orthonormalize.
        for (int p = 0; p < nroot; p++) v[p] = Ritz[p];
        for (int p = 0; p < nroot; p++) {
          for (int q = 0; q < p; q++) {
            ElemT ol = _local_inner(v[q], v[p]);
            for (int i = 0; i < K; i++) v[p][i] -= v[q][i]*ol;
          }
          _local_normalize<ElemT,RealT>(v[p]);
        }
        nseed = nroot;
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
                                    RealT eps) {
      const size_t ndet = det.size();
      auto matvec = [&](const std::vector<ElemT> & xc, std::vector<ElemT> & yc) {
        std::vector<ElemT> xdet, ydet(ndet, ElemT(0.0));
        project_up(V, xc, xdet, ndet);
        Zero(ydet);
        mult(hii, xdet, ydet, bit_length, norb, det, idxmap, exidx, I0, I1, I2,
             h_comm, b_comm, t_comm);
        project_down(V, ydet, yc);
      };
      _davidson_projected_core<ElemT,RealT>(hii, V, matvec, Wcsf, Eout,
                                            h_comm, b_comm, t_comm,
                                            max_iteration, num_block, nroot, eps);
    }

    /**
       Stored-matrix (method 1) projected multi-root solver. matvec = V^T H (V x)
       with H applied via the pre-built sparse Hamiltonian (ih/jh/hij/len/slide).
       Faster than matrix-free when the matrix fits; single-spin's basis is small
       (b_comm==1) so it always fits.
    */
    template <typename ElemT, typename RealT>
    void DavidsonMultiRootProjectedStored(const std::vector<ElemT> & hii,
                                          const SpinProjector & V,
                                          std::vector<std::vector<ElemT>> & Wcsf,
                                          std::vector<RealT> & Eout,
                                          size_t ndet,
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
                                          RealT eps) {
      auto matvec = [&](const std::vector<ElemT> & xc, std::vector<ElemT> & yc) {
        std::vector<ElemT> xdet, ydet(ndet, ElemT(0.0));
        project_up(V, xc, xdet, ndet);
        Zero(ydet);
        mult(hii, ih, jh, hij, len, slide, xdet, ydet, h_comm, b_comm, t_comm);
        project_down(V, ydet, yc);
      };
      _davidson_projected_core<ElemT,RealT>(hii, V, matvec, Wcsf, Eout,
                                            h_comm, b_comm, t_comm,
                                            max_iteration, num_block, nroot, eps);
    }

  } // namespace gdb
} // namespace sbd

#endif
