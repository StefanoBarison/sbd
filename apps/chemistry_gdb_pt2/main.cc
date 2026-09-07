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

  // The wavefunction reader, perturber generation and the energy accumulation
  // follow. Deliberately not stubbed with a fake number: an app that prints a
  // plausible-looking energy before the physics exists is how a wrong result gets
  // trusted.
  if (mpi_rank == 0) {
    std::cout << " " << sbd::make_timestamp()
              << " pt2: setup complete; correction not yet implemented."
              << std::endl;
  }

  MPI_Finalize();
  return 0;
}
