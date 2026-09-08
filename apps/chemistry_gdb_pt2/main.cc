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

#include "sbd/sbd.h"
#include "sbd/chemistry/gdb/pt2.h"
#include "mpi.h"

#ifdef _COMPLEX
using Elem = std::complex<double>;
#else
using Elem = double;
#endif

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
    // Canonical order. ~25 lower_bound calls elsewhere in the tree assume it, and
    // the set-difference against the variational space below relies on it too.
    sbd::sort_bitarray(det);
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
  // break early on a sorted integral list); the merge is dominated by std::map
  // lookups on determinant keys and would not move at all. So "generation
  // dominates" has to mean t_gen specifically, not t_gen + t_merge.
  double t_gen = 0.0, t_merge = 0.0;
  size_t n_collapses = 0;
  if (holds_dets) {
    for (size_t i = 0; i < det.size(); ++i) {
      const double ac = std::abs(std::complex<double>(c[i]));
      if (ac == 0.0) continue;            // a zero-weight reference contributes nothing
      ++n_refs_local;
      // det[i] is a non-owning row view into det's flat store; the copy into a
      // std::vector is needed only because generate_perturbers_from (and Hij beneath
      // it) take std::vector<size_t>. Reused across references so it allocates once,
      // not once per reference.
      ref.assign(nword, 0);
      { const size_t * rp = det[i].data();
        for (size_t w = 0; w < nword; ++w) ref[w] = rp[w]; }
      raw.clear();
      const double tg0 = MPI_Wtime();
      sbd::gdb::generate_perturbers_from<Elem>(ref, c[i], opt.epsilon2 / ac,
                                              bit_length, static_cast<size_t>(L),
                                              I0, I1, I2, scratch, raw);
      t_gen += MPI_Wtime() - tg0;
      n_emitted_local += raw.size();
      const double tm0 = MPI_Wtime();
      // Merge within this reference's own output first: it is already sorted by
      // merge_pt2_perturbers, so duplicates from the SAME reference (reachable by two
      // excitation paths) are combined before they ever reach the flat store, which
      // keeps its length near the unique count rather than the emitted count.
      sbd::gdb::merge_pt2_perturbers(raw, merged);
      for (const auto & m : merged) {
        acc_w.insert(acc_w.end(), m.det.begin(), m.det.end());
        acc_n.push_back(m.num);
        acc_h.push_back(m.haa);
      }
      if (acc_n.size() >= acc_collapse_at) {
        collapse_acc();
        ++n_collapses;
        // Next collapse only after the store has doubled past what survived this one.
        acc_collapse_at = std::max<size_t>(acc_collapse_floor, 2 * acc_n.size());
      }
      t_merge += MPI_Wtime() - tm0;
    }
  }

  // Sort the flat store by determinant and combine duplicates, SUMMING numerators.
  // An index permutation is sorted rather than the rows themselves, so no determinant
  // words are moved during the sort.
  //
  // The order is canonical less_from_back, which is not optional: remove_variational
  // and the variational-row scan in variant (c) both binary-search `det` in that
  // order, and the merge below relies on equal determinants being adjacent.
  const double tsort0 = MPI_Wtime();
  collapse_acc();      // final sort-and-sum; leaves the store unique and ordered
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
    n_removed_local = sbd::gdb::remove_variational(all, det);
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
      // Cross-rank exposure specific to (c), checked rather than assumed. A completed
      // orbit row's x_r is summed over THIS rank's references only, so if one
      // perturbing configuration is reachable from references on two b ranks, each
      // rank squares its own partial x_r and the CSF energy is wrong. Unlike (a)'s
      // per-rank merge (dormant because perturber SETS turn out disjoint), this one
      // depends on configurations being rank-local, which is what --do_redist_config
      // provides. Count how many of this rank's perturbing configurations are also
      // seen by another rank: zero means the property holds for this input.
      if (b_comm_size > 1) {
        // A cheap order-independent fingerprint of the configuration set. Summing
        // per-config hashes over b_comm and comparing against the count of distinct
        // configs would not localize a collision, so instead each rank hashes its
        // config list into a fixed number of buckets and the buckets are summed: a
        // config present on two ranks lands in the same bucket twice.
        const size_t NB = 1u << 16;
        std::vector<int> mine(NB, 0);
        for (const auto & pc : cfg) {
          size_t hh = 1469598103934665603ULL;
          for (int v : pc.config) { hh ^= (size_t)(v + 1); hh *= 1099511628211ULL; }
          mine[hh % NB] = 1;
        }
        std::vector<int> tot(NB, 0);
        MPI_Allreduce(mine.data(), tot.data(), (int)NB, MPI_INT, MPI_SUM, b_comm);
        size_t shared_buckets = 0;
        for (size_t k2 = 0; k2 < NB; ++k2) if (tot[k2] > 1) ++shared_buckets;
        int rb = 0; MPI_Comm_rank(b_comm, &rb);
        if (rb == 0 && shared_buckets > 0) {
          std::cout << " sbd: WARNING pt2: " << shared_buckets << " configuration"
                    << " hash bucket(s) are populated on more than one b rank. If"
                    << " those are genuinely the same configurations (not hash"
                    << " collisions), variant (c) is INEXACT at b_comm_size > 1:"
                    << " a completed orbit row's numerator is summed over one rank's"
                    << " references only, then squared per rank. Re-run with"
                    << " --b_comm_size 1 to get the exact value." << std::endl;
        } else if (rb == 0) {
          std::cout << " " << sbd::make_timestamp() << " pt2: configuration sets are"
                    << " disjoint across all " << b_comm_size << " b ranks; the"
                    << " spin-pure numerators are exact." << std::endl;
        }
      }
      sbd::gdb::complete_pt2_numerators<Elem>(cfg, det, c, bit_length,
                                              static_cast<size_t>(L), nword,
                                              I0, I1, I2, row_var, row_off, spst);
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
    const size_t sc[7] = {
      holds_dets ? spst.n_configs : 0,
      holds_dets ? spst.n_configs_no_target_s : 0,
      holds_dets ? spst.n_rows_total : 0,
      holds_dets ? spst.n_rows_from_generator : 0,
      holds_dets ? spst.n_rows_completed : 0,
      holds_dets ? spst.n_configs_partly_variational : 0,
      holds_dets ? spst.n_csf : 0,
    };
    MPI_Allreduce(sc, sp, 7, SBD_MPI_SIZE_T, MPI_SUM, b_comm);
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
