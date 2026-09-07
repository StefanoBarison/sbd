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

  if (opt.variant == sbd::gdb::PT2Variant::SpinPure) {
    if (mpi_rank == 0) {
      std::cerr << " sbd: ERROR pt2: --variant c (spin-pure) is not implemented yet."
                << " Use --variant a." << std::endl;
    }
    MPI_Abort(comm, 5);
    return 5;
  }

  // ------------------------------------------------------- variant (a): PT2
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
  std::vector<sbd::gdb::PT2Perturber<Elem>> raw;
  std::vector<sbd::gdb::PT2Merged<Elem>> merged;

  // Running accumulation keyed by determinant. This is the cross-batch equivalent of
  // the hash-partitioned merge: contributions to one perturber from references in
  // different batches must still be SUMMED before squaring, so the energy cannot be
  // accumulated per batch -- only the numerators can.
  std::map<std::vector<size_t>, std::pair<Elem, double>> acc;

  size_t n_refs_local = 0, n_emitted_local = 0;
  if (holds_dets) {
    for (size_t i = 0; i < det.size(); ++i) {
      const double ac = std::abs(std::complex<double>(c[i]));
      if (ac == 0.0) continue;            // a zero-weight reference contributes nothing
      ++n_refs_local;
      const std::vector<size_t> ref = det[i];
      raw.clear();
      sbd::gdb::generate_perturbers_from<Elem>(ref, c[i], opt.epsilon2 / ac,
                                              bit_length, static_cast<size_t>(L),
                                              I0, I1, I2, scratch, raw);
      n_emitted_local += raw.size();
      sbd::gdb::merge_pt2_perturbers(raw, merged);
      for (const auto & m : merged) {
        auto it = acc.find(m.det);
        if (it == acc.end()) acc.emplace(m.det, std::make_pair(m.num, m.haa));
        else it->second.first += m.num;
      }
    }
  }

  // Flatten, drop anything already variational, and accumulate.
  std::vector<sbd::gdb::PT2Merged<Elem>> all;
  all.reserve(acc.size());
  for (const auto & [d, nh] : acc) {
    sbd::gdb::PT2Merged<Elem> m;
    m.det = d; m.num = nh.first; m.haa = nh.second; m.n_parents = 1;
    all.push_back(std::move(m));
  }
  acc.clear();
  std::sort(all.begin(), all.end(),
            [](const auto & x, const auto & y) {
              return sbd::less_from_back(x.det, y.det);
            });
  const size_t n_removed_local = sbd::gdb::remove_variational(all, det);
  const auto res = sbd::gdb::accumulate_pt2(all, e0, opt.den_floor);

  // Reduce over b_comm only: h/t ranks are replicas of the same determinant slice, so
  // including them would multiply every count and every energy by the replica factor.
  // (The same trap that reported 195 of 391 matches during the reader work.)
  double e_pt2 = 0.0, psi1 = 0.0, worst = 0.0;
  size_t n_pert = 0, n_floored = 0, n_refs = 0, n_emitted = 0, n_removed = 0;
  const double e_c   = holds_dets ? res.energy : 0.0;
  const double p_c   = holds_dets ? res.psi1_norm2 : 0.0;
  const double w_c   = holds_dets ? res.worst_ratio : 0.0;
  const size_t np_c  = holds_dets ? res.n_perturbers : 0;
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
  if (mpi_rank == 0) {
    const double t = MPI_Wtime() - t_pt2_start;
    std::cout << " " << sbd::make_timestamp() << " pt2: references=" << n_refs
              << " emitted=" << n_emitted << " unique=" << n_pert
              << " removed_variational=" << n_removed
              << " floored=" << n_floored
              << " [" << std::fixed << std::setprecision(2) << t << " s]"
              << std::endl;
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
