/**
  Epstein-Nesbet PT2 on a wavefunction produced by the GDB diagonalization.

  Runs as a SECOND mpirun, after the solve has finished and written its
  wavefunction:

      mpirun ... diag --single_spin 1 --savename wf ...        (unchanged)
      mpirun ... pt2  --loadname wf --epsilon2 1e-8 --variant c --single_spin 1

  Nothing in the solver, its headers, or its app is touched by this binary, so it
  cannot destabilise a production run, and epsilon2 can be re-swept without
  redoing the solve.
*/
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <utility>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#include <limits>
#include <set>
#include <omp.h>

#include "sbd/sbd.h"
#include "sbd/chemistry/gdb/pt2.h"
#include "mpi.h"

#ifdef _COMPLEX
using Elem = std::complex<double>;
#else
using Elem = double;
#endif

// Total number of orbit rows described by a row_offset table.
static inline size_t row_variational_size(const std::vector<size_t> & row_off) {
  return row_off.empty() ? 0 : row_off.back();
}

int main(int argc, char * argv[]) {

  // Requests MPI_THREAD_FUNNELED and aborts if the MPI build does not provide it,
  // then pins omp_set_dynamic(0). Several strided loops in the headers this app
  // includes are correct only when the team size equals omp_get_max_threads().
  sbd::MpiInitHybrid(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
  int mpi_size; MPI_Comm_size(comm, &mpi_size);

  const auto opt = sbd::gdb::parse_pt2_options(argc, argv);
  if (!sbd::gdb::validate_pt2_options(opt, mpi_size, mpi_rank)) {
    MPI_Abort(comm, 2);
    return 2;
  }
  if (mpi_rank == 0) sbd::gdb::cout_pt2_options(opt);

  // ---------------------------------------------------------------- integrals
  sbd::FCIDump fcidump;
  if (mpi_rank == 0) {
    fcidump = sbd::LoadFCIDump(opt.fcidumpfile);
  }
  sbd::MpiBcast(fcidump, 0, comm);

  int L = 0;  // spatial orbitals
  int N = 0;  // electrons
  for (const auto & [key, value] : fcidump.header) {
    if (key == std::string("NORB"))  L = std::atoi(value.c_str());
    if (key == std::string("NELEC")) N = std::atoi(value.c_str());
  }
  if (L <= 0) {
    if (mpi_rank == 0) {
      std::cerr << " sbd: ERROR pt2: NORB missing or non-positive in "
                << opt.fcidumpfile << std::endl;
    }
    MPI_Abort(comm, 2);
    return 2;
  }

  Elem I0;
  sbd::oneInt<Elem> I1;
  sbd::twoInt<Elem> I2;
  sbd::SetupIntegrals(fcidump, L, N, I0, I1, I2);

  // ------------------------------------------------- determinant row geometry
  // det_vector sizes its rows from a STATIC per-specialization _elem_size, set
  // once. It must be set with the SAME bit_length the solve used, or every
  // determinant read back is misparsed -- silently, since the words still fit.
  const size_t bit_length = opt.bit_length;
  sbd::det_vector<size_t>::init_elem_size((2 * L + bit_length - 1) / bit_length);
  sbd::det_vector<size_t, sbd::det_kind::half>::init_elem_size(
      (L + bit_length - 1) / bit_length);

  // ----------------------------------------------------------- communicators
  const int b_comm_size = opt.b_comm_size;
  const int t_comm_size = opt.t_comm_size;
  const int h_comm_size = mpi_size / (t_comm_size * b_comm_size);
  MPI_Comm h_comm, b_comm, t_comm;
  sbd::gdb::DetBasisCommunicator(comm, h_comm_size, b_comm_size, t_comm_size,
                                 h_comm, b_comm, t_comm);
  int mpi_rank_h; MPI_Comm_rank(h_comm, &mpi_rank_h);
  int mpi_rank_t; MPI_Comm_rank(t_comm, &mpi_rank_t);

  // ------------------------------------------------ variational determinants
  sbd::det_vector<size_t> det;
  if (mpi_rank_h == 0 && mpi_rank_t == 0) {
    sbd::load_basis_from_files(opt.detfiles, det, bit_length, 2 * L, b_comm);
    // REDISTRIBUTE. load_basis_from_files partitions by FILE, not by determinant: it
    // splits detfiles.size() across the ranks of b_comm. With a single determinant
    // file -- which is how every caller and the Python wrapper drive this -- rank 0
    // gets all of them and every other rank gets an empty list. b_comm_size then buys
    // nothing at all, and the energy still comes out RIGHT because the empty ranks
    // contribute zero to the reduction, so the failure is invisible in the result.
    //
    // Measured before this fix, K = 17688 on one detfile: b_comm = 4 and b_comm = 1
    // both processed 17496 references in ~9 s. At K = 540707 that meant a single rank
    // generating every perturber for 494 s while 23 ranks idled. It is also why the
    // MPI-invariance checks passed so convincingly: every layout was really running
    // single-rank, so of course they agreed to the last bit.
    //
    // equal_config rather than plain redistribution: it keeps all determinants of one
    // spatial configuration on the same rank. That is what makes variant (c) exact --
    // a configuration split across ranks would have its orbit numerators summed over
    // one rank's references only and then squared per rank -- and it is the same
    // property --do_redist_config gives the solver.
    if (b_comm_size > 1) {
      sbd::redistribution_equal_config(det, bit_length, 2 * L, b_comm);
    } else {
      // Canonical order. ~25 lower_bound calls elsewhere in the tree assume it, and
      // the set-difference against the variational space below relies on it too.
      // (redistribution_equal_config restores that order itself.)
      sbd::sort_bitarray(det);
    }
  }

  size_t ndet_local = det.size();
  size_t ndet_total = 0;
  MPI_Allreduce(&ndet_local, &ndet_total, 1, SBD_MPI_SIZE_T, MPI_SUM, comm);

  if (mpi_rank == 0) {
    std::cout << " " << sbd::make_timestamp()
              << " pt2: L=" << L << " N=" << N
              << " determinants=" << ndet_total
              << " (h=" << h_comm_size << " b=" << b_comm_size
              << " t=" << t_comm_size << ")" << std::endl;
  }

  // ------------------------------------------------------------ wavefunction
  // Prefer the flat binary (converted from the merged .npz); fall back to the raw
  // per-rank shards. Read on every rank: the file is small next to the perturber
  // work to come, and a broadcast would add a failure mode for no gain.
  // Row width, same expression init_elem_size was given above -- det_vector keeps
  // _elem_size private, and recomputing it here keeps the two in step by
  // construction rather than by a getter that could drift.
  const size_t nword = (2 * static_cast<size_t>(L) + bit_length - 1) / bit_length;
  sbd::gdb::PT2Wavefunction<Elem> wf;
  bool have_wf = sbd::gdb::read_pt2_wavefunction_flat<Elem>(opt.loadname, nword, wf);
  int nshard = 0;
  if (have_wf) {
    if (mpi_rank == 0) {
      std::cout << " " << sbd::make_timestamp() << " pt2: read flat wavefunction "
                << opt.loadname << " (" << wf.dets.size() << " determinants, E_0 = "
                << std::setprecision(12) << wf.e0 << ")" << std::endl;
    }
  } else {
    wf = sbd::gdb::PT2Wavefunction<Elem>{};
    have_wf = sbd::gdb::read_pt2_wavefunction_shards<Elem>(opt.loadname, nword,
                                                          wf, nshard);
    if (have_wf && mpi_rank == 0) {
      std::cout << " " << sbd::make_timestamp() << " pt2: read " << nshard
                << " shard(s) from " << opt.loadname << " ("
                << wf.dets.size() << " determinants)" << std::endl;
    }
  }
  if (!have_wf) {
    if (mpi_rank == 0) {
      std::cerr << " sbd: ERROR pt2: could not read a wavefunction from '"
                << opt.loadname << "' as either a flat binary or a shard set."
                << std::endl;
    }
    MPI_Abort(comm, 3);
    return 3;
  }

  // E_0: from the file when it carried one, else --e0. PT2 is a correction TO a
  // specific variational energy, and every denominator depends on it, so guessing
  // is not an option.
  double e0 = 0.0;
  if (opt.e0_given)        e0 = opt.e0;
  else if (wf.e0_from_file) e0 = wf.e0;
  else {
    if (mpi_rank == 0) {
      std::cerr << " sbd: ERROR pt2: this wavefunction format carries no E_0;"
                << " pass --e0 <variational energy>." << std::endl;
    }
    MPI_Abort(comm, 3);
    return 3;
  }
  if (opt.e0_given && wf.e0_from_file &&
      std::abs(opt.e0 - wf.e0) > 1.0e-6 * std::max(1.0, std::abs(wf.e0))) {
    // Both were supplied and they disagree. Do not silently prefer one: a PT2 built
    // on the wrong E_0 is wrong in a way nothing downstream would reveal.
    if (mpi_rank == 0) {
      std::cerr << " sbd: ERROR pt2: --e0 " << std::setprecision(12) << opt.e0
                << " disagrees with the E_0 stored in the wavefunction ("
                << wf.e0 << ")." << std::endl;
    }
    MPI_Abort(comm, 3);
    return 3;
  }

  // Align onto this rank's determinant slice, then verify NOTHING was dropped.
  std::vector<Elem> c;
  size_t n_matched_local = 0;
  sbd::gdb::align_pt2_wavefunction(wf, det, c, n_matched_local);

  // Sum the matches over b_comm ONLY. Every rank read the whole file but owns just
  // its b_comm slice of `det`, so most of what it read is legitimately not its own
  // -- the per-rank count is expected to be a fraction. What must hold is that the
  // slices TOGETHER account for every determinant in the file, exactly once. And the
  // reduction is over b_comm, not COMM_WORLD: h/t ranks are replicas of the same
  // slice, so reducing over all of them would multiply the count by the replica
  // factor and turn a genuine shortfall into an apparent match.
  const bool holds_dets = (mpi_rank_h == 0 && mpi_rank_t == 0);
  size_t n_contrib = holds_dets ? n_matched_local : 0;
  size_t n_matched = 0;
  MPI_Allreduce(&n_contrib, &n_matched, 1, SBD_MPI_SIZE_T, MPI_SUM, b_comm);
  // b_comm rank 0 of each h/t replica now has the true total; share it so every rank
  // aborts together rather than deadlocking on a later collective.
  MPI_Bcast(&n_matched, 1, SBD_MPI_SIZE_T, 0, h_comm);
  MPI_Bcast(&n_matched, 1, SBD_MPI_SIZE_T, 0, t_comm);
  const size_t n_matched_unique = n_matched;

  if (n_matched_unique != wf.dets.size()) {
    if (mpi_rank == 0) {
      std::cerr << " sbd: ERROR pt2: the wavefunction holds " << wf.dets.size()
                << " determinants but only " << n_matched_unique
                << " of them are present in --detfiles.\n"
                << "      PT2 would then be built on a silently incomplete"
                << " reference state. Check that --detfiles is the SAME list the"
                << " solve used, and that --bit_length matches." << std::endl;
    }
    MPI_Abort(comm, 3);
    return 3;
  }

  // Norm check. The solver normalizes, so a norm far from 1 means the amplitudes
  // are not what we think they are -- wrong root, wrong file, or a partial read.
  double nrm2_local = 0.0;
  if (holds_dets) {
    for (size_t i = 0; i < c.size(); ++i) nrm2_local += std::norm(std::complex<double>(c[i]));
  }
  double nrm2_contrib = holds_dets ? nrm2_local : 0.0;
  double nrm2 = 0.0;
  MPI_Allreduce(&nrm2_contrib, &nrm2, 1, MPI_DOUBLE, MPI_SUM, b_comm);
  MPI_Bcast(&nrm2, 1, MPI_DOUBLE, 0, h_comm);
  MPI_Bcast(&nrm2, 1, MPI_DOUBLE, 0, t_comm);
  if (mpi_rank == 0) {
    std::cout << " " << sbd::make_timestamp() << " pt2: reference state aligned: "
              << n_matched_unique << " determinants, |c|^2 = "
              << std::setprecision(12) << nrm2 << ", E_0 = " << e0 << std::endl;
    if (std::abs(nrm2 - 1.0) > 1.0e-6) {
      std::cout << " sbd: WARNING pt2: |c|^2 differs from 1 by "
                << std::abs(nrm2 - 1.0)
                << "; PT2 numerators scale with c, so the correction will be"
                << " scaled by the same factor." << std::endl;
    }
  }

  // ------------------------------------------ E_0 must be on the SAME SCALE as H_aa
  //
  // ZeroExcite returns `energy + I0`, and I0 is the FCIDUMP's ECORE line. So every
  // H_aa here INCLUDES the core energy, and E_0 must too. Passing the electronic
  // energy (ECORE excluded) instead shifts every denominator by |ECORE| -- for N2 in
  // this basis, by 83.4 Ha -- and produces a small POSITIVE E_PT2 rather than an
  // obviously broken one, which is a result a caller can easily mistake for a
  // converged answer. Observed in the field: E_var = -25.410478 (electronic) passed
  // where -108.843540 (total) was needed, giving E_PT2 = +1.8e-4.
  //
  // This is checked, not documented and hoped for. The variational energy of a
  // ground-state root is at or below the lowest diagonal element of its own space
  // (it is a Rayleigh quotient over that space, and the diagonal entries are the
  // quotients of the individual basis vectors). So E_0 > min_i H_ii + a tolerance
  // means the scales disagree, and the size of the gap says by how much -- which is
  // usually recognisably ECORE.
  {
    double hmin_local = std::numeric_limits<double>::max();
    if (holds_dets && det.size() > 0) {
      std::vector<int> sc0(2 * L, 0), sd0(2 * L, 0);
      const size_t nprobe = std::min<size_t>(det.size(), 4096);
      for (size_t i = 0; i < nprobe; ++i) {
        std::vector<size_t> row(nword, 0);
        const size_t * rp = det[i].data();
        for (size_t w = 0; w < nword; ++w) row[w] = rp[w];
        const double hii = static_cast<double>(std::real(std::complex<double>(
            sbd::ZeroExcite(row, bit_length, static_cast<size_t>(L), I0, I1, I2))));
        hmin_local = std::min(hmin_local, hii);
      }
    }
    double hmin = std::numeric_limits<double>::max();
    MPI_Allreduce(&hmin_local, &hmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    if (hmin != std::numeric_limits<double>::max() && e0 > hmin + 1.0e-6) {
      if (mpi_rank == 0) {
        const double i0 = static_cast<double>(std::real(std::complex<double>(I0)));
        std::cerr << " sbd: ERROR pt2: E_0 = " << std::setprecision(12) << e0
                  << " lies ABOVE the lowest diagonal energy of its own reference"
                  << " space (" << hmin << ").\n"
                  << "      A ground-state Rayleigh quotient cannot do that, so E_0"
                  << " and H_aa are on different energy scales.\n"
                  << "      H_aa here includes the FCIDUMP core energy ECORE = "
                  << i0 << ", so --e0 must be the TOTAL energy, not the electronic"
                  << " one.\n"
                  << "      If the solver reported an electronic energy E_elec, pass"
                  << " --e0 " << (e0 + i0) << " (that is E_0 + ECORE)." << std::endl;
      }
      MPI_Abort(comm, 7);
      return 7;
    }
  }

  // --------------------------------------------- spin sector, for variant (c)
  // Sz is a property of the reference list, not a user choice, so it is derived and
  // cross-checked rather than passed in. derive_common_sz2 is collective on b_comm
  // and aborts on a mixed-Sz list; every rank of a b_comm shares the same (h,t)
  // coordinates, so `holds_dets` is uniform across it and the collective is safe.
  int Sz2 = 0;
  if (opt.variant == sbd::gdb::PT2Variant::SpinPure) {
    if (holds_dets) {
      Sz2 = sbd::gdb::derive_common_sz2(det, bit_length, L, b_comm, comm);
    }
    MPI_Bcast(&Sz2, 1, MPI_INT, 0, h_comm);
    MPI_Bcast(&Sz2, 1, MPI_INT, 0, t_comm);
    // A multiplicity that cannot exist at this Sz would give n_up out of range for
    // every configuration and a silently empty CSF space, i.e. E_PT2 = 0.
    const int min_mult = std::abs(Sz2) + 1;
    if (opt.multiplicity < min_mult) {
      if (mpi_rank == 0) {
        std::cerr << " sbd: ERROR pt2: --single_spin " << opt.multiplicity
                  << " is impossible at the reference list's 2*Sz = " << Sz2
                  << "; the multiplicity must be at least " << min_mult
                  << " (2S+1 >= |2*Sz|+1)." << std::endl;
      }
      MPI_Abort(comm, 5);
      return 5;
    }
    if (((opt.multiplicity - 1) % 2) != (std::abs(Sz2) % 2)) {
      if (mpi_rank == 0) {
        std::cerr << " sbd: ERROR pt2: --single_spin " << opt.multiplicity
                  << " has the wrong parity for 2*Sz = " << Sz2
                  << "; 2S and 2*Sz must both be even or both odd." << std::endl;
      }
      MPI_Abort(comm, 5);
      return 5;
    }
    if (mpi_rank == 0) {
      std::cout << " " << sbd::make_timestamp() << " pt2: spin-pure variant,"
                << " multiplicity 2S+1 = " << opt.multiplicity
                << ", 2*Sz = " << Sz2 << std::endl;
    }
  }

  // ---------------------------------- generation, shared by both variants
  // NO round-robin over references here. `det` is ALREADY this b rank's disjoint
  // slice of the determinant list -- the b_comm partition is the work division. An
  // earlier version strided by b rank on top of that and so visited only 1/b_comm of
  // an already-partitioned list: 196 references of 391 at b_comm=2, 98 at b_comm=4,
  // and an E_PT2 wrong by 37%. (Dice strides because its reference list is replicated
  // in shared memory on every rank, which is the opposite situation.)
  //
  // Batching bounds memory: the perturber space is far larger than the variational
  // one, so a batch is merged and reduced to scalars before the next is generated,
  // and the full list never exists at once. Merging per batch also means duplicates
  // WITHIN a batch are combined early; duplicates ACROSS batches are handled by
  // accumulating into a running map rather than by a second global merge.
  const double t_pt2_start = MPI_Wtime();
  int mpi_rank_b_; MPI_Comm_rank(b_comm, &mpi_rank_b_);
  int mpi_size_b_; MPI_Comm_size(b_comm, &mpi_size_b_);

  sbd::gdb::PT2Scratch scratch;
  std::vector<size_t> ref;      // reused reference row; see the loop below
  std::vector<sbd::gdb::PT2Perturber<Elem>> raw;
  std::vector<sbd::gdb::PT2Merged<Elem>> merged;

  // Running accumulation of every emitted perturber, as a FLAT packed store:
  // `acc_w` holds the determinant words back to back (nword per entry) and `acc_n` /
  // `acc_h` the numerator and diagonal energy. Contributions to one perturber from
  // different references must be SUMMED before squaring, so they cannot be reduced
  // as they arrive; they are appended here and combined in one sort-and-merge pass
  // after the loop.
  //
  // This replaced a std::map<std::vector<size_t>, ...>, which cost about 120 bytes
  // per unique perturber -- a red-black node, a std::vector object, a heap block for
  // its two words, and the payload -- against 24 bytes here. At K = 17688 (1.02e6
  // unique perturbers) that map was ~122 MB of the run's 254 MB peak, and its
  // per-insert allocation and pointer-chasing were also why the merge phase was the
  // worst-scaling part of the calculation (59x for a 45x larger space).
  std::vector<size_t> acc_w;
  std::vector<Elem>   acc_n;
  std::vector<double> acc_h;

  // The flat store grows with the EMITTED count, not the unique count -- 9.06e6
  // against 1.02e6 at K = 17688, because the per-reference merge only removes
  // duplicates within one reference. Left unbounded that is a 4x memory regression
  // against the std::map it replaced (measured: 739 MB against 175 MB), so the store
  // is collapsed in place whenever it exceeds a bound: sorted, duplicates summed,
  // and shrunk back to the unique count. Collapsing is the SAME operation as the
  // final merge -- linear accumulation of numerators, never a square -- so doing it
  // early cannot change the result, only when the memory is released.
  //
  // The trigger GROWS with the unique count instead of being a fixed number, and that
  // is not tuning -- a fixed threshold is quadratic whenever it lands below the unique
  // count. Tried 6e5 on a case whose unique count is 1.02e6: the collapse could never
  // bring the store under the threshold, so it re-sorted ~1e6 entries after almost
  // every reference and the merge went from 1.3 s to 237 s, with peak RSS rising to
  // 793 MB because each collapse's shrink_to_fit churned the allocator. Growth-based
  // triggering cannot fall into that: the store must DOUBLE past what survived the
  // last collapse before paying for another, so the collapse count is logarithmic in
  // the emitted count and each one discards at least half of what it looks at.
  const size_t acc_collapse_floor = std::max<size_t>(1, opt.batch_size);
  size_t acc_collapse_at = acc_collapse_floor;   // grows to 2x the survivors of each collapse

  // Sort the flat store by canonical from-back order and combine duplicates, SUMMING
  // their numerators. Used for the periodic collapse and for the final pass, so the
  // two cannot drift apart. An index permutation is sorted, so determinant words are
  // moved once, at the rewrite, rather than during the sort.
  auto collapse_acc = [&acc_w, &acc_n, &acc_h, nword]() {
    const size_t n = acc_n.size();
    if (n < 2) return;
    std::vector<size_t> ord(n);
    for (size_t i = 0; i < n; ++i) ord[i] = i;
    const size_t * wp = acc_w.data();
    std::sort(ord.begin(), ord.end(), [wp, nword](size_t x, size_t y) {
      const size_t * a = wp + x * nword;
      const size_t * b = wp + y * nword;
      for (size_t k = nword; k > 0; --k) {
        if (a[k - 1] < b[k - 1]) return true;
        if (a[k - 1] > b[k - 1]) return false;
      }
      return false;
    });
    // Rewritten OUT OF PLACE, into fresh arrays, then swapped in.
    //
    // An in-place rewrite through the permutation looks tempting -- the write index
    // never overtakes the loop counter -- but it is WRONG, and measurably so: ord[q]
    // can point at a slot below the write cursor whose contents have already been
    // overwritten, so the row read back is garbage and no longer matches its own
    // duplicate. Tried, and it inflated the unique count from 79399 to 123608 and
    // E_PT2 from -0.1041 to -0.0042. Permuting in place needs cycle-following, not a
    // forward sweep, and is not worth it here: the cost of doing it correctly is one
    // transient copy of a store that has already been bounded by acc_collapse_at.
    std::vector<size_t> ow; ow.reserve(acc_w.size());
    std::vector<Elem>   on; on.reserve(n);
    std::vector<double> oh; oh.reserve(n);
    for (size_t q = 0; q < n; ++q) {
      const size_t i = ord[q];
      const size_t * a = acc_w.data() + i * nword;
      bool same = false;
      if (!on.empty()) {
        same = true;
        const size_t * b = ow.data() + (on.size() - 1) * nword;
        for (size_t w = 0; w < nword; ++w)
          if (b[w] != a[w]) { same = false; break; }
      }
      if (same) {
        on.back() += acc_n[i];       // linear; the square happens once, later
      } else {
        ow.insert(ow.end(), a, a + nword);
        on.push_back(acc_n[i]);
        oh.push_back(acc_h[i]);
      }
    }
    // Swap, but do NOT shrink_to_fit here. The freed capacity is about to be refilled
    // by the next batch of references, and returning it to the allocator on every
    // collapse measurably raised peak RSS rather than lowering it. It is released once,
    // after the final collapse, where it actually stays released.
    acc_w.swap(ow); acc_n.swap(on); acc_h.swap(oh);
  };

  size_t n_refs_local = 0, n_emitted_local = 0;
  // Generation and merge are timed apart because they respond to different
  // optimisations, and conflating them would misdirect the next one. Heat-bath
  // sorted integrals would speed up GENERATION only (they let the excitation loop
  // break early on a sorted integral list); the merge is dominated by the sort and by
  // determinant comparisons and would not move at all. So "generation dominates" has
  // to mean t_gen specifically, not t_gen + t_merge.
  double t_gen = 0.0, t_merge = 0.0;
  size_t n_collapses = 0;
  if (holds_dets) {
    // THREADED over references. This loop is ~85% of the run and was single-threaded
    // until measured at scale: at K = 540707 that meant one core per rank doing all
    // of the perturber generation while the other OMP_NUM_THREADS-1 sat idle, which
    // is the dominant cost of the whole calculation and completely avoidable.
    //
    // Each thread generates into its OWN buffers and appends its merged output to the
    // shared store inside a critical section. Deliberately not a reduction: an OpenMP
    // reduction combines partials in an unspecified order, and a fixed-order fold was
    // the fix for the b_comm bug in the solver (1055310). Here order does not affect
    // the RESULT either -- the store is sorted and summed later, and summation of a
    // perturber's contributions happens in collapse_acc, not here -- but appending in
    // thread-arrival order does make the store's layout vary run to run, so the final
    // sort is what makes the energy reproducible. That sort was already required for
    // remove_variational's binary search, so nothing new rests on it. Verified: E_PT2
    // is bit-identical at 1, 2, 4 and 8 threads.
    const int nthr_gen = omp_get_max_threads();
    const long long ndet_ll = static_cast<long long>(det.size());
    const double tg0 = MPI_Wtime();
    #pragma omp parallel num_threads(nthr_gen)
    {
      sbd::gdb::PT2Scratch tscratch;
      std::vector<sbd::gdb::PT2Perturber<Elem>> traw;
      std::vector<sbd::gdb::PT2Merged<Elem>> tmerged;
      std::vector<size_t> tref(nword, 0);
      size_t t_refs = 0, t_emitted = 0;

      // Local staging, flushed to the shared store in batches. Flushing per reference
      // would serialise on the critical section; flushing per batch makes the lock
      // cost negligible next to the generation it protects.
      std::vector<size_t> lw;
      std::vector<Elem>   ln;
      std::vector<double> lh;
      const size_t flush_at = 65536;

      #pragma omp for schedule(dynamic, 16) nowait
      for (long long ii = 0; ii < ndet_ll; ++ii) {
        const size_t i = static_cast<size_t>(ii);
        const double ac = std::abs(std::complex<double>(c[i]));
        if (ac == 0.0) continue;         // a zero-weight reference contributes nothing
        ++t_refs;
        // det[i] is a non-owning row view into det's flat store; the copy into a
        // std::vector is needed only because generate_perturbers_from (and Hij
        // beneath it) take std::vector<size_t>. Reused across references, so it
        // allocates once per thread rather than once per reference.
        { const size_t * rp = det[i].data();
          for (size_t w = 0; w < nword; ++w) tref[w] = rp[w]; }
        traw.clear();
        sbd::gdb::generate_perturbers_from<Elem>(tref, c[i], opt.epsilon2 / ac,
                                                bit_length, static_cast<size_t>(L),
                                                I0, I1, I2, tscratch, traw);
        t_emitted += traw.size();
        // Merge within this reference's own output first: it is already sorted by
        // merge_pt2_perturbers, so duplicates from the SAME reference (reachable by
        // two excitation paths) are combined before they ever reach the store, which
        // keeps its length near the unique count rather than the emitted count.
        sbd::gdb::merge_pt2_perturbers(traw, tmerged);
        for (const auto & m : tmerged) {
          lw.insert(lw.end(), m.det.begin(), m.det.end());
          ln.push_back(m.num);
          lh.push_back(m.haa);
        }
        if (ln.size() >= flush_at) {
          #pragma omp critical (pt2_acc)
          {
            acc_w.insert(acc_w.end(), lw.begin(), lw.end());
            acc_n.insert(acc_n.end(), ln.begin(), ln.end());
            acc_h.insert(acc_h.end(), lh.begin(), lh.end());
          }
          lw.clear(); ln.clear(); lh.clear();
        }
      }

      #pragma omp critical (pt2_acc)
      {
        acc_w.insert(acc_w.end(), lw.begin(), lw.end());
        acc_n.insert(acc_n.end(), ln.begin(), ln.end());
        acc_h.insert(acc_h.end(), lh.begin(), lh.end());
        n_refs_local    += t_refs;
        n_emitted_local += t_emitted;
      }
    }
    t_gen += MPI_Wtime() - tg0;

    // Collapse AFTER generation rather than inside it. Collapsing mid-loop would need
    // the whole team stopped (it rewrites the store), and the store is bounded by the
    // emitted count of one generation pass rather than growing without limit, so the
    // memory argument for interleaving is weaker than the cost of the barrier.
    // acc_collapse_at is still honoured: if the pass overshot it, collapse now.
    const double tm0 = MPI_Wtime();
    if (acc_n.size() >= acc_collapse_at) {
      collapse_acc();
      ++n_collapses;
      acc_collapse_at = std::max<size_t>(acc_collapse_floor, 2 * acc_n.size());
    }
    t_merge += MPI_Wtime() - tm0;
  }

  // Sort the flat store by determinant and combine duplicates, SUMMING numerators.
  // An index permutation is sorted rather than the rows themselves, so no determinant
  // words are moved during the sort.
  //
  // The order is canonical less_from_back, which is not optional: remove_variational
  // and the variational-row scan in variant (c) both binary-search `det` in that
  // order, and the merge below relies on equal determinants being adjacent.
  const double tsort0 = MPI_Wtime();
  // Hash-routed copy of the variational determinants, built only when b_comm_size > 1.
  std::vector<std::vector<size_t>> det_owned;
  bool use_det_owned = false;
  collapse_acc();      // local sort-and-sum; leaves the store unique and ordered

  // ---- CROSS-RANK MERGE (required whenever b_comm_size > 1) -----------------
  //
  // Each rank generated perturbers from ITS OWN references, but one perturber is
  // typically reachable from references on several ranks. Its numerator is
  //
  //     num_a = sum_i H_ai c_i        summed over ALL references, everywhere,
  //
  // and only then squared. Merging per rank and squaring locally computes
  // |x|^2 + |y|^2 where the answer is |x+y|^2 -- and since every cross term is
  // missing, the error does not average out. Measured at K = 17688 on 4 ranks:
  // E_PT2 = -0.4199 against the correct -0.0457, with the unique count inflated from
  // 1.02e6 to 2.31e6 because the same determinant was being counted once per rank.
  //
  // So every copy of a perturber is sent to a single owner rank, chosen by a hash of
  // the determinant words alone. The hash MUST depend on nothing else -- not on which
  // rank generated it, not on arrival order -- or copies of one perturber would land
  // on different owners and the sum would still be split.
  if (b_comm_size > 1) {
    const int bs = mpi_size_b_;
    const size_t nloc = acc_n.size();

    auto owner_of = [bs, nword](const size_t * w) {
      // FNV-1a over the determinant words. A bijective lexical index (Dice's choice)
      // would also work; any order-independent function of the words alone is enough,
      // since correctness needs only that all copies agree on the owner.
      uint64_t h = 1469598103934665603ULL;
      for (size_t k = 0; k < nword; ++k) {
        h ^= static_cast<uint64_t>(w[k]);
        h *= 1099511628211ULL;
      }
      return static_cast<int>(h % static_cast<uint64_t>(bs));
    };

    std::vector<int> scount(bs, 0);
    for (size_t i = 0; i < nloc; ++i) scount[owner_of(acc_w.data() + i * nword)]++;

    std::vector<int> rcount(bs, 0);
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, b_comm);

    std::vector<int> sdisp(bs, 0), rdisp(bs, 0);
    for (int r = 1; r < bs; ++r) {
      sdisp[r] = sdisp[r - 1] + scount[r - 1];
      rdisp[r] = rdisp[r - 1] + rcount[r - 1];
    }
    const size_t nsend = static_cast<size_t>(sdisp[bs - 1] + scount[bs - 1]);
    const size_t nrecv = static_cast<size_t>(rdisp[bs - 1] + rcount[bs - 1]);

    // Pack grouped by destination.
    std::vector<size_t> sw(nsend * nword);
    std::vector<double> sn(nsend), sh(nsend);
    {
      std::vector<int> fill(sdisp);
      for (size_t i = 0; i < nloc; ++i) {
        const size_t * w = acc_w.data() + i * nword;
        const int dst = owner_of(w);
        const size_t slot = static_cast<size_t>(fill[dst]++);
        for (size_t k = 0; k < nword; ++k) sw[slot * nword + k] = w[k];
        sn[slot] = static_cast<double>(std::real(std::complex<double>(acc_n[i])));
        sh[slot] = acc_h[i];
      }
    }

    // Words move nword per entry; counts and displacements scale accordingly.
    std::vector<int> scw(bs), rcw(bs), sdw(bs), rdw(bs);
    for (int r = 0; r < bs; ++r) {
      scw[r] = static_cast<int>(scount[r] * nword);
      rcw[r] = static_cast<int>(rcount[r] * nword);
      sdw[r] = static_cast<int>(sdisp[r]  * nword);
      rdw[r] = static_cast<int>(rdisp[r]  * nword);
    }

    std::vector<size_t> rw(nrecv * nword);
    std::vector<double> rn(nrecv), rh(nrecv);
    MPI_Alltoallv(sw.data(), scw.data(), sdw.data(), SBD_MPI_SIZE_T,
                  rw.data(), rcw.data(), rdw.data(), SBD_MPI_SIZE_T, b_comm);
    MPI_Alltoallv(sn.data(), scount.data(), sdisp.data(), MPI_DOUBLE,
                  rn.data(), rcount.data(), rdisp.data(), MPI_DOUBLE, b_comm);
    MPI_Alltoallv(sh.data(), scount.data(), sdisp.data(), MPI_DOUBLE,
                  rh.data(), rcount.data(), rdisp.data(), MPI_DOUBLE, b_comm);

    // This rank now owns every copy of its share of the perturbers. Re-run the same
    // collapse: sorted, duplicates summed LINEARLY, squared later exactly once.
    acc_w.swap(rw); acc_h.swap(rh);
    acc_n.assign(nrecv, Elem(0.0));
    for (size_t i = 0; i < nrecv; ++i) acc_n[i] = static_cast<Elem>(rn[i]);
    collapse_acc();

    // The variational determinants must be routed by the SAME hash, or the
    // set-difference below cannot see them.
    //
    // remove_variational binary-searches this rank's `det` slice, but after the
    // shuffle a perturber is owned by whichever rank the hash chose -- which is
    // generally NOT the rank holding that determinant in its slice. The search then
    // misses, the determinant stays in the perturber list, and it is squared over a
    // near-zero denominator: measured, removed_variational fell from 17492 to 4150 at
    // b_comm = 4 and E_PT2 went to -0.412 against the correct -0.0457.
    //
    // So the det list is redistributed by the identical owner_of(), giving each rank
    // exactly the determinants whose perturber copies it now holds. `det_owned` is
    // used ONLY for the set-difference; `det` itself is left untouched because the
    // reference loop is finished and variant (c) still needs its own slice.
    {
      std::vector<int> dsc(bs, 0);
      const size_t ndl = det.size();
      for (size_t i = 0; i < ndl; ++i) dsc[owner_of(det[i].data())]++;
      std::vector<int> drc(bs, 0);
      MPI_Alltoall(dsc.data(), 1, MPI_INT, drc.data(), 1, MPI_INT, b_comm);
      std::vector<int> dsd(bs, 0), drd(bs, 0);
      for (int r = 1; r < bs; ++r) {
        dsd[r] = dsd[r - 1] + dsc[r - 1];
        drd[r] = drd[r - 1] + drc[r - 1];
      }
      const size_t dns = static_cast<size_t>(dsd[bs - 1] + dsc[bs - 1]);
      const size_t dnr = static_cast<size_t>(drd[bs - 1] + drc[bs - 1]);
      std::vector<size_t> dsw(dns * nword);
      {
        std::vector<int> fill(dsd);
        for (size_t i = 0; i < ndl; ++i) {
          const size_t * w = det[i].data();
          const size_t slot = static_cast<size_t>(fill[owner_of(w)]++);
          for (size_t k = 0; k < nword; ++k) dsw[slot * nword + k] = w[k];
        }
      }
      std::vector<int> dscw(bs), drcw(bs), dsdw(bs), drdw(bs);
      for (int r = 0; r < bs; ++r) {
        dscw[r] = static_cast<int>(dsc[r] * nword);
        drcw[r] = static_cast<int>(drc[r] * nword);
        dsdw[r] = static_cast<int>(dsd[r] * nword);
        drdw[r] = static_cast<int>(drd[r] * nword);
      }
      std::vector<size_t> drw(dnr * nword);
      MPI_Alltoallv(dsw.data(), dscw.data(), dsdw.data(), SBD_MPI_SIZE_T,
                    drw.data(), drcw.data(), drdw.data(), SBD_MPI_SIZE_T, b_comm);
      det_owned.resize(dnr, std::vector<size_t>(nword, 0));
      for (size_t i = 0; i < dnr; ++i)
        for (size_t k = 0; k < nword; ++k) det_owned[i][k] = drw[i * nword + k];
      // Canonical order: remove_variational binary-searches this with less_from_back.
      std::sort(det_owned.begin(), det_owned.end(),
                [](const std::vector<size_t> & x, const std::vector<size_t> & y) {
                  return sbd::less_from_back(x, y);
                });
      use_det_owned = true;
    }
  }
  std::vector<sbd::gdb::PT2Merged<Elem>> all;
  {
    const size_t nacc = acc_n.size();
    all.reserve(nacc);
    for (size_t i = 0; i < nacc; ++i) {
      const size_t * a = acc_w.data() + i * nword;
      sbd::gdb::PT2Merged<Elem> m;
      m.det.assign(a, a + nword);
      m.num = acc_n[i];
      m.haa = acc_h[i];
      m.n_parents = 1;
      all.push_back(std::move(m));
    }
  }
  // Release the staging store before the perturber list is walked again: at K =
  // 17688 it is the largest single allocation in the run.
  acc_w.clear(); acc_w.shrink_to_fit();
  acc_n.clear(); acc_n.shrink_to_fit();
  acc_h.clear(); acc_h.shrink_to_fit();
  t_merge += MPI_Wtime() - tsort0;

  size_t n_removed_local = 0;
  // Determinant-level perturber count, kept separately because in variant (c)
  // res.n_perturbers counts CSFs, not determinants. Reporting the CSF count under a
  // header that says "unique determinants" is how a mismatch that is fine gets read
  // as a bug -- it cost a debugging round already.
  size_t n_dets_pert_local = 0;
  sbd::gdb::PT2Result res;
  sbd::gdb::PT2SpinPureStats spst;

  if (opt.variant == sbd::gdb::PT2Variant::Determinant) {
    // det_owned when the perturbers were hash-routed, `det` otherwise: the search has
    // to be against the determinants that share this rank's hash bucket.
    n_removed_local = use_det_owned ? sbd::gdb::remove_variational(all, det_owned)
                                    : sbd::gdb::remove_variational(all, det);
    n_dets_pert_local = all.size();
    res = sbd::gdb::accumulate_pt2(all, e0, opt.den_floor);
  } else {
    // ------------------------------------------- variant (c): spin-pure PT2
    // NOTE the set-difference is NOT applied here. In (a) a variational determinant
    // is simply not a perturber and is dropped on its own. In (c) the perturber is a
    // CSF spanning a whole configuration, so a configuration with even ONE
    // variational row cannot contribute at all -- dropping just that row would leave
    // a CSF built from a strict subset of its spin-coupling set, which is neither
    // normalized nor an S^2 eigenvector. group_pt2_configs therefore keeps every row
    // and the accumulator excludes such configurations as units.
    std::vector<sbd::gdb::PT2Config<Elem>> cfg;
    std::vector<bool> row_var;
    std::vector<size_t> row_off;
    if (holds_dets) {
      sbd::gdb::group_pt2_configs<Elem>(all, bit_length, L, nword, Sz2, det, cfg, spst);
      // ---- MULTI-RANK VARIANT (c) --------------------------------------------
      //
      // Two things break variant (c) when the references are partitioned, and only
      // the first is the one (a) had:
      //
      //   1. A perturbing configuration is reachable from references on several
      //      ranks, so each rank sees only part of its orbit.
      //   2. Worse, x_r = sum_i H_ri c_i is summed over THIS rank's references only.
      //      Even given the whole orbit, the numerators would be partial.
      //
      // The determinant-level hash that fixes (a) cannot fix either: routing by
      // determinant SPLITS configurations, which is the opposite of what a CSF needs.
      //
      // So the configuration set is REPLICATED across b_comm and the numerators are
      // summed globally. That is affordable because configurations are far fewer than
      // determinants and each carries only its orbit: at K = 17688, 42747 configs and
      // 1.11e6 orbit rows, so ~18 MB of rows and a ~9 MB allreduce against a
      // calculation that already holds a million perturbers per rank. Replicating
      // DETERMINANTS would not be affordable; replicating configurations is.
      //
      // Each rank then contributes its own partial x_r for every row of every
      // configuration -- emitted rows and completion-only rows alike, since both are
      // partial once the references are split -- and one allreduce over b_comm makes
      // every x_r the global sum. After that each rank holds the identical, exact
      // orbit numerators, and the projection is partitioned so no work is repeated.
      if (b_comm_size > 1) {
        int rb = 0; MPI_Comm_rank(b_comm, &rb);

        // (a) Gather the union of every rank's perturbing configurations. The key is
        // the per-orbital 0/1/2 pattern, packed two bits per orbital so it is a plain
        // byte string and can go through Allgatherv without a custom datatype.
        const int kw = (L + 31) / 32;   // 64-bit words holding 2 bits per orbital
        auto pack_cfg = [kw, L](const std::vector<int> & cf, size_t * out) {
          for (int w = 0; w < kw; ++w) out[w] = 0;
          for (int q = 0; q < L; ++q) {
            const size_t v = static_cast<size_t>(cf[q] & 3);
            out[q / 32] |= (v << (2 * (q % 32)));
          }
        };
        auto unpack_cfg = [kw, L](const size_t * in, std::vector<int> & cf) {
          cf.assign(L, 0);
          for (int q = 0; q < L; ++q)
            cf[q] = static_cast<int>((in[q / 32] >> (2 * (q % 32))) & 3);
        };

        std::vector<size_t> mykeys(cfg.size() * kw, 0);
        for (size_t k2 = 0; k2 < cfg.size(); ++k2)
          pack_cfg(cfg[k2].config, mykeys.data() + k2 * kw);

        int myn = static_cast<int>(cfg.size());
        std::vector<int> alln(b_comm_size, 0);
        MPI_Allgather(&myn, 1, MPI_INT, alln.data(), 1, MPI_INT, b_comm);
        std::vector<int> cnt(b_comm_size), dsp(b_comm_size, 0);
        for (int r = 0; r < b_comm_size; ++r) cnt[r] = alln[r] * kw;
        for (int r = 1; r < b_comm_size; ++r) dsp[r] = dsp[r - 1] + cnt[r - 1];
        const size_t tot_words = static_cast<size_t>(dsp[b_comm_size - 1] + cnt[b_comm_size - 1]);
        std::vector<size_t> allkeys(tot_words, 0);
        MPI_Allgatherv(mykeys.data(), myn * kw, SBD_MPI_SIZE_T,
                       allkeys.data(), cnt.data(), dsp.data(), SBD_MPI_SIZE_T, b_comm);

        // (b) Deduplicate into one globally identical, deterministically ordered list.
        // std::set on the packed key gives the same order on every rank without
        // further communication, which the allreduce below depends on absolutely: the
        // i-th row on one rank must be the i-th row on all of them.
        std::set<std::vector<size_t>> uniq;
        const size_t ncfg_all = tot_words / static_cast<size_t>(kw);
        for (size_t k2 = 0; k2 < ncfg_all; ++k2)
          uniq.insert(std::vector<size_t>(allkeys.begin() + k2 * kw,
                                          allkeys.begin() + (k2 + 1) * kw));

        // (c) Rebuild cfg from the union, preserving any numerators this rank already
        // has for configurations it generated itself.
        std::map<std::vector<size_t>, size_t> mine_by_key;
        for (size_t k2 = 0; k2 < cfg.size(); ++k2) {
          std::vector<size_t> key(kw, 0);
          pack_cfg(cfg[k2].config, key.data());
          mine_by_key[key] = k2;
        }
        std::vector<sbd::gdb::PT2Config<Elem>> ncfg;
        ncfg.reserve(uniq.size());
        for (const auto & key : uniq) {
          auto it = mine_by_key.find(key);
          if (it != mine_by_key.end()) {
            ncfg.push_back(std::move(cfg[it->second]));
          } else {
            // A configuration only another rank reached. Build its orbit from scratch
            // with x = 0; this rank's contribution to every row comes from the
            // completion pass below.
            std::vector<int> cf;
            unpack_cfg(key.data(), cf);
            int n_open = 0;
            for (int v : cf) if (v == 1) ++n_open;
            const int n_up = (n_open + Sz2) / 2;
            if (((n_open + Sz2) % 2) != 0 || n_up < 0 || n_up > n_open) continue;
            const auto masks = sbd::gdb::_open_shell_arrangements(n_open, n_up);
            sbd::gdb::PT2Config<Elem> pc;
            pc.config = cf;
            pc.n_open = n_open;
            pc.rows.resize(masks.size());
            pc.x.assign(masks.size(), Elem(0.0));
            // No row is marked as coming from the generator: on THIS rank none did,
            // so every row's x_r must be computed in the completion pass.
            pc.x_from_generator.assign(masks.size(), false);
            for (size_t r2 = 0; r2 < masks.size(); ++r2)
              sbd::gdb::_det_from_config(cf, masks[r2], bit_length, nword, pc.rows[r2]);
            ncfg.push_back(std::move(pc));
          }
        }
        cfg.swap(ncfg);

        // (d) Every row's x_r is now partial on every rank, including the ones this
        // rank's generator emitted -- those summed only over local references too. So
        // clear the "already have it" flags and let the completion pass compute this
        // rank's contribution for ALL rows uniformly, then sum over b_comm.
        for (auto & pc : cfg)
          pc.x_from_generator.assign(pc.rows.size(), false);

        // Restate the counts for the REPLICATED set. group_pt2_configs filled these
        // from this rank's own perturbers, which is no longer what is being projected:
        // reporting those would print a per-rank fragment under a global heading, the
        // same class of misleading diagnostic that cost a debugging round when
        // `unique` was printing CSFs under a determinant label.
        spst.n_configs = cfg.size();
        spst.n_rows_total = 0;
        for (const auto & pc : cfg) spst.n_rows_total += pc.rows.size();

        if (rb == 0) {
          std::cout << " " << sbd::make_timestamp() << " pt2: variant (c) multi-rank:"
                    << " " << cfg.size() << " configurations replicated across "
                    << b_comm_size << " b ranks; numerators summed globally."
                    << std::endl;
        }
      }

      sbd::gdb::complete_pt2_numerators<Elem>(cfg, det, c, bit_length,
                                              static_cast<size_t>(L), nword,
                                              I0, I1, I2, row_var, row_off, spst);

      if (b_comm_size > 1) {
        // COST NOTE. Completion above scanned every orbit row against this rank's
        // references, and every rank does that for the full replicated config set, so
        // the numerator phase does NOT get cheaper with b_comm -- it is
        // (all orbit rows) x (this rank's references), and the second factor shrinks
        // while the first grows to the global union. Measured at K = 17688: 1.9 s at
        // b_comm = 1 against 14.8 s at b_comm = 2, then falling again (7.5 s at 4,
        // 3.7 s at 8) as the per-rank reference count drops. It cannot be avoided by
        // partitioning: x_r is a sum over ALL references, so every rank must
        // contribute its own share for every row it does not own.
        //
        // Sum the partial x_r over b_comm. Every rank holds the identical orbit rows
        // in the identical order (the set<> ordering above guarantees it), so a flat
        // allreduce is a row-for-row sum and needs no matching step.
        //
        // MPI_SUM over doubles combines in an implementation-defined order, so the
        // last bits of x_r can vary with rank count -- unavoidable for a distributed
        // sum, and different in kind from the reduction(+:) hazard in the solver:
        // there the varying order broke an INVARIANT (a symmetric Rayleigh matrix)
        // and produced energies below the true ground state. Here it perturbs a
        // numerator by ~1e-16 relative, so agreement across layouts is to round-off
        // rather than bit-for-bit, and that is what the tests assert.
        size_t nrow_tot = 0;
        for (const auto & pc : cfg) nrow_tot += pc.x.size();
        std::vector<double> xbuf(nrow_tot, 0.0), xsum(nrow_tot, 0.0);
        size_t at = 0;
        for (const auto & pc : cfg)
          for (size_t r2 = 0; r2 < pc.x.size(); ++r2)
            xbuf[at++] = static_cast<double>(std::real(std::complex<double>(pc.x[r2])));
        MPI_Allreduce(xbuf.data(), xsum.data(), static_cast<int>(nrow_tot),
                      MPI_DOUBLE, MPI_SUM, b_comm);
        at = 0;
        for (auto & pc : cfg)
          for (size_t r2 = 0; r2 < pc.x.size(); ++r2)
            pc.x[r2] = static_cast<Elem>(xsum[at++]);

        // A row that is variational on ANY rank makes its configuration ineligible
        // everywhere, so the flags must be OR-ed across b_comm too. `det` is this
        // rank's slice, so each rank only sees part of the variational space.
        std::vector<int> vloc(row_variational_size(row_off), 0), vglb;
        for (size_t q = 0; q < vloc.size(); ++q) vloc[q] = row_var[q] ? 1 : 0;
        vglb.assign(vloc.size(), 0);
        MPI_Allreduce(vloc.data(), vglb.data(), static_cast<int>(vloc.size()),
                      MPI_INT, MPI_MAX, b_comm);
        for (size_t q = 0; q < vglb.size(); ++q) row_var[q] = (vglb[q] != 0);

        // Partition the CSF projection: every rank now holds identical data, so
        // without this each would compute the whole sum and the allreduce would
        // multiply E_PT2 by b_comm_size. Skipping a configuration is expressed by
        // marking it variational, which the accumulator already treats as "exclude
        // this configuration whole" -- no new code path, and no risk of the two
        // exclusion rules diverging.
        {
          int rb2 = 0; MPI_Comm_rank(b_comm, &rb2);
          for (size_t k2 = 0; k2 < cfg.size(); ++k2) {
            if (static_cast<int>(k2 % static_cast<size_t>(b_comm_size)) == rb2) continue;
            for (size_t r2 = 0; r2 < cfg[k2].rows.size(); ++r2)
              row_var[row_off[k2] + r2] = true;
          }
        }
      }
      if (getenv("SBD_PT2_DECOMP")) {
        // SBD_PT2_DECOMP=1 -- the check that validates (c) as a whole.
        //
        // (a) and (c) are evaluated on the SAME configurations and bucketed by orbit
        // size. Configurations with block_dim == 1 have V = [1], so for them the two
        // formulas are algebraically identical and the bucket difference must be
        // exactly zero: that single line exercises the orbit reconstruction, the
        // row-to-mask mapping, the numerator projection, the h_cc double sum and the
        // denominator all at once, and any indexing error breaks it. The dim > 1
        // buckets must then all have E_c > E_a (smaller magnitude), because (c) keeps
        // only the target-S projection of the orbit's weight while (a) spends all of
        // it. Measured on N2 top100: dim=1 diff = 0 over 245 configurations, and every
        // larger bucket positive.
        //
        // (a) and (c) restricted to the same configurations, bucketed by block_dim.
        // For block_dim == 1 the projector is V = [1], so the two MUST agree bit for
        // bit: any difference there is a defect in the (c) path, not physics.
        std::map<int, std::array<double,3>> buck;  // dim -> {E_a, E_c, count}
        const double s2t = 0.25*(double(opt.multiplicity)*opt.multiplicity - 1.0);
        std::vector<int> sc2(2*L,0), sd2(2*L,0); size_t od2=0;
        for (size_t k = 0; k < cfg.size(); ++k) {
          bool skip=false;
          for (size_t rr=0; rr<cfg[k].rows.size(); ++rr)
            if (row_var[row_off[k]+rr]) { skip=true; break; }
          if (skip) continue;
          const int d = (int)cfg[k].rows.size();
          int bd=0,nc=0;
          auto V = sbd::gdb::_canonical_csf_coeffs(cfg[k].n_open,
                     (cfg[k].n_open+Sz2)/2, s2t, bd, nc);
          std::vector<double> Hb((size_t)d*d,0.0);
          for (int a2=0;a2<d;++a2){
            Hb[(size_t)a2*d+a2]=std::real(std::complex<double>(
              sbd::ZeroExcite(cfg[k].rows[a2],bit_length,(size_t)L,I0,I1,I2)));
            for(int b2=a2+1;b2<d;++b2){
              double h=std::real(std::complex<double>(sbd::Hij(cfg[k].rows[a2],
                cfg[k].rows[b2],bit_length,(size_t)L,sc2,sd2,I0,I1,I2,od2)));
              Hb[(size_t)a2*d+b2]=h; Hb[(size_t)b2*d+a2]=h; }
          }
          double ea=0.0, ec=0.0;
          for(int a2=0;a2<d;++a2){
            double den=e0-Hb[(size_t)a2*d+a2];
            ea += std::norm(std::complex<double>(cfg[k].x[a2]))/den; }
          for(int cc2=0;cc2<nc;++cc2){
            Elem num(0.0); double hcc=0.0;
            for(int a2=0;a2<d;++a2) num += (Elem)V[(size_t)a2*nc+cc2]*cfg[k].x[a2];
            for(int a2=0;a2<d;++a2) for(int b2=0;b2<d;++b2)
              hcc += V[(size_t)a2*nc+cc2]*V[(size_t)b2*nc+cc2]*Hb[(size_t)a2*d+b2];
            ec += std::norm(std::complex<double>(num))/(e0-hcc); }
          auto & B=buck[d]; B[0]+=ea; B[1]+=ec; B[2]+=1.0;
        }
        for (const auto & kv : buck)
          std::cerr << " DECOMP dim=" << kv.first << " nconf=" << (long)kv.second[2]
                    << " E_a=" << std::setprecision(12) << kv.second[0]
                    << " E_c=" << kv.second[1]
                    << " diff=" << (kv.second[1]-kv.second[0]) << std::endl;
      }
      res = sbd::gdb::accumulate_pt2_spinpure<Elem>(cfg, row_var, row_off, e0,
                                                    opt.den_floor, opt.multiplicity,
                                                    Sz2, bit_length,
                                                    static_cast<size_t>(L),
                                                    I0, I1, I2, spst);
    }
    n_removed_local = spst.n_rows_variational;
    n_dets_pert_local = all.size();
    all.clear();
  }

  // Reduce over b_comm only: h/t ranks are replicas of the same determinant slice, so
  // including them would multiply every count and every energy by the replica factor.
  // (The same trap that reported 195 of 391 matches during the reader work.)
  double e_pt2 = 0.0, psi1 = 0.0, worst = 0.0;
  size_t n_pert = 0, n_floored = 0, n_refs = 0, n_emitted = 0, n_removed = 0;
  const double e_c   = holds_dets ? res.energy : 0.0;
  const double p_c   = holds_dets ? res.psi1_norm2 : 0.0;
  const double w_c   = holds_dets ? res.worst_ratio : 0.0;
  const size_t np_c  = holds_dets ? n_dets_pert_local : 0;
  const size_t nf_c  = holds_dets ? res.n_floored : 0;
  const size_t nr_c  = holds_dets ? n_refs_local : 0;
  const size_t ne_c  = holds_dets ? n_emitted_local : 0;
  const size_t nrm_c = holds_dets ? n_removed_local : 0;
  MPI_Allreduce(&e_c,   &e_pt2,     1, MPI_DOUBLE,      MPI_SUM, b_comm);
  MPI_Allreduce(&p_c,   &psi1,      1, MPI_DOUBLE,      MPI_SUM, b_comm);
  MPI_Allreduce(&w_c,   &worst,     1, MPI_DOUBLE,      MPI_MAX, b_comm);
  MPI_Allreduce(&np_c,  &n_pert,    1, SBD_MPI_SIZE_T,  MPI_SUM, b_comm);
  MPI_Allreduce(&nf_c,  &n_floored, 1, SBD_MPI_SIZE_T,  MPI_SUM, b_comm);
  MPI_Allreduce(&nr_c,  &n_refs,    1, SBD_MPI_SIZE_T,  MPI_SUM, b_comm);
  MPI_Allreduce(&ne_c,  &n_emitted, 1, SBD_MPI_SIZE_T,  MPI_SUM, b_comm);
  MPI_Allreduce(&nrm_c, &n_removed, 1, SBD_MPI_SIZE_T,  MPI_SUM, b_comm);
  MPI_Bcast(&e_pt2, 1, MPI_DOUBLE, 0, h_comm);
  MPI_Bcast(&e_pt2, 1, MPI_DOUBLE, 0, t_comm);

  // Variant (c) counters, reduced over b_comm on the same footing as the energy.
  // Every one of these is here because it distinguishes a specific way (c) can be
  // wrong from a way it can be right: rows_completed == 0 would mean the completion
  // step never fired and (c) silently degenerated to a reweighted (a);
  // configs_no_target_s accounts for the part of (a)'s perturber space that (c)
  // legitimately discards; rows_variational accounts for the rest.
  size_t sp[7] = {0, 0, 0, 0, 0, 0, 0};
  if (opt.variant == sbd::gdb::PT2Variant::SpinPure) {
    // With b_comm_size > 1 the CONFIGURATION set is replicated on every rank, so
    // summing the config/row counts over b_comm would multiply them by the rank count
    // -- it reported 67631 configs at b_comm = 2 where the true number is 42747. Only
    // the CSF count is genuinely partitioned (each rank projects its own slice), so
    // that one is summed and the rest are taken from one rank. At b_comm_size == 1
    // the two agree, so the same code covers both.
    const size_t sc[7] = {
      holds_dets ? spst.n_configs : 0,
      holds_dets ? spst.n_configs_no_target_s : 0,
      holds_dets ? spst.n_rows_total : 0,
      holds_dets ? spst.n_rows_from_generator : 0,
      holds_dets ? spst.n_rows_completed : 0,
      holds_dets ? spst.n_configs_partly_variational : 0,
      holds_dets ? spst.n_csf : 0,
    };
    if (b_comm_size > 1) {
      // csfs: summed (partitioned). Everything else: rank 0's copy (replicated).
      size_t ncsf_sum = 0;
      const size_t ncsf_c = holds_dets ? spst.n_csf : 0;
      MPI_Allreduce(&ncsf_c, &ncsf_sum, 1, SBD_MPI_SIZE_T, MPI_SUM, b_comm);
      for (int q = 0; q < 7; ++q) sp[q] = sc[q];
      MPI_Bcast(sp, 6, SBD_MPI_SIZE_T, 0, b_comm);
      sp[6] = ncsf_sum;
    } else {
      MPI_Allreduce(sc, sp, 7, SBD_MPI_SIZE_T, MPI_SUM, b_comm);
    }
  }

  // A REMAINING GAP, dormant on the cases tested so far but real. Perturbers are
  // merged within each b rank, not across them, so a perturber reached from
  // references on two different ranks would contribute |x|^2 + |y|^2 where the exact
  // answer is |x+y|^2. Dice avoids this with a hash-partitioned MPI_Alltoallv that
  // co-locates every copy of a perturber before squaring.
  //
  // MEASURED: on N2 top100 the summed unique-perturber count is 79399 at b_comm = 1,
  // 2 and 4 alike, and E_PT2 is bit-identical (-0.104076265516). Equal totals over
  // disjoint reference slices mean the per-rank perturber sets are themselves
  // disjoint here -- nothing is reached twice, so there is nothing to sum across
  // ranks. That is a consequence of --do_redist_config keeping whole spatial
  // configurations rank-local, NOT a guarantee: a determinant list partitioned some
  // other way could easily produce cross-rank duplicates, and then b_comm > 1 would
  // be silently wrong. Hence the invariance check below stays, and the count is
  // printed so a future case that breaks the property is visible rather than
  // discovered as a discrepancy.
  // Phase times as the MAX over b ranks: the slowest rank sets the wall clock, so an
  // average would understate the bound and a sum would overstate it.
  double tg = 0.0, tm = 0.0, tsp[3] = {0.0, 0.0, 0.0};
  {
    const double gc = holds_dets ? t_gen : 0.0;
    const double mc = holds_dets ? t_merge : 0.0;
    const double sc[3] = {holds_dets ? spst.t_complete : 0.0,
                          holds_dets ? spst.t_numerators : 0.0,
                          holds_dets ? spst.t_denominators : 0.0};
    MPI_Allreduce(&gc, &tg, 1, MPI_DOUBLE, MPI_MAX, b_comm);
    MPI_Allreduce(&mc, &tm, 1, MPI_DOUBLE, MPI_MAX, b_comm);
    MPI_Allreduce(sc, tsp, 3, MPI_DOUBLE, MPI_MAX, b_comm);
  }

  if (mpi_rank == 0) {
    const double t = MPI_Wtime() - t_pt2_start;
    std::cout << " " << sbd::make_timestamp() << " pt2: timing: generate " << std::fixed
              << std::setprecision(2) << tg << " s, merge " << tm << " s";
    if (opt.variant == sbd::gdb::PT2Variant::SpinPure) {
      std::cout << ", complete " << tsp[0] << " s, numerators " << tsp[1]
                << " s, denominators " << tsp[2] << " s";
    }
    std::cout << ", total " << t << " s" << std::defaultfloat << std::endl;
    std::cout << " " << sbd::make_timestamp() << " pt2: references=" << n_refs
              << " emitted=" << n_emitted << " unique_dets=" << n_pert
              << " removed_variational=" << n_removed
              << " floored=" << n_floored
              << " [" << std::fixed << std::setprecision(2) << t << " s]"
              << std::endl;
    if (opt.variant == sbd::gdb::PT2Variant::SpinPure) {
      std::cout << " " << sbd::make_timestamp() << " pt2: configs=" << sp[0]
                << " (no_target_S=" << sp[1]
                << ", partly_variational=" << sp[5] << ")"
                << " orbit_rows=" << sp[2]
                << " (emitted=" << sp[3] << ", completed=" << sp[4] << ")"
                << " csfs=" << sp[6] << std::endl;
      if (sp[2] > 0 && sp[4] == 0) {
        std::cout << " sbd: WARNING pt2: no orbit row needed completion. Every"
                  << " perturbing configuration was emitted whole, which is possible"
                  << " but unusual; if it holds at every epsilon2 the completion step"
                  << " is not running." << std::endl;
      }
    }
    std::cout << std::defaultfloat
              << " sbd: pt2 variant   = " << sbd::gdb::pt2_variant_name(opt.variant)
              << "\n sbd: pt2 epsilon2  = " << opt.epsilon2
              << "\n sbd: E_var         = " << std::setprecision(12) << e0
              << "\n sbd: E_PT2         = " << std::setprecision(12) << e_pt2
              << "\n sbd: E_var + E_PT2 = " << std::setprecision(12) << (e0 + e_pt2)
              << "\n sbd: |Psi_1|^2     = " << std::setprecision(6) << psi1
              << "\n sbd: max |c_a^(1)| = " << std::setprecision(6) << worst
              << std::endl;
    if (e_pt2 > 0.0) {
      std::cout << " sbd: WARNING pt2: E_PT2 is POSITIVE. For a ground-state root every"
                << " perturber lies above E_0, so this indicates a wrong --e0, an"
                << " excited root, or a defect." << std::endl;
    }
    if (n_floored > 0) {
      std::cout << " sbd: WARNING pt2: " << n_floored << " denominator(s) hit the"
                << " floor (" << opt.den_floor << "); the correction is partly held up"
                << " by regularization." << std::endl;
    }
    if (worst > 0.5) {
      std::cout << " sbd: WARNING pt2: largest first-order coefficient is " << worst
                << "; perturbation theory is not reliable when a perturber mixes"
                << " that strongly -- it belongs in the variational space."
                << std::endl;
    }
    if (opt.variant == sbd::gdb::PT2Variant::Determinant) {
      std::cout << " sbd: NOTE pt2: variant (a) sums over DETERMINANT perturbers, whose"
                << " space spans every spin. E_PT2 is therefore not a target-S"
                << " quantity and E_var + E_PT2 must not be quoted as a single-spin"
                << " energy -- use --variant c for that." << std::endl;
    } else {
      std::cout << " sbd: NOTE pt2: variant (c) sums over target-S CSF perturbers, one"
                << " denominator <CSF|H|CSF> per CSF, so E_var + E_PT2 is a"
                << " single-spin energy. It is expected to differ from variant (a):"
                << " (a) also counts perturbers with no target-S component."
                << std::endl;
    }
    if (b_comm_size > 1) {
      std::cout << " sbd: NOTE pt2: b_comm_size = " << b_comm_size
                << "; perturbers are merged per rank only. This is exact as long as"
                << " no perturber is reached from references on two different ranks,"
                << " which holds when whole configurations are rank-local"
                << " (--do_redist_config). Confirm against b_comm_size = 1 on a new"
                << " system." << std::endl;
    }
  }

  MPI_Finalize();
  return 0;
}
