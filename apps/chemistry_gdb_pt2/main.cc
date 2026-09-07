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
#include <complex>
#include <iomanip>
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

  // Perturber generation and the energy accumulation follow. Deliberately not
  // stubbed with a fake number: an app that prints a plausible-looking energy
  // before the physics exists is how a wrong result gets trusted.
  if (mpi_rank == 0) {
    std::cout << " " << sbd::make_timestamp()
              << " pt2: input verified; correction not yet implemented."
              << std::endl;
  }

  MPI_Finalize();
  return 0;
}
