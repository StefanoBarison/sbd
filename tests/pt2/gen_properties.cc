// Property tests for the PT2 perturber generator (chemistry/gdb/pt2.h).
//
// These check invariants that hold for ANY correct generator, independent of any
// reference PT2 energy -- so they can be trusted before a single energy number
// exists, which is the point. Run as:
//
//   mpicxx -std=c++17 -O2 -I<sbd>/include -o gen_properties gen_properties.cc <blas/lapack>
//   ./gen_properties <FCIDUMP>
//
// Every line must report the "want" value. A nonzero count is a real defect.
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include "sbd/sbd.h"
#include "sbd/chemistry/gdb/pt2.h"
#include "mpi.h"
using Elem = double;

int main(int argc, char ** argv) {
  sbd::MpiInitHybrid(&argc, &argv);
  std::string fci = argv[1];
  auto fcidump = sbd::LoadFCIDump(fci);
  int L = 0, N = 0;
  for (const auto & [k, v] : fcidump.header) {
    if (k == "NORB") L = std::atoi(v.c_str());
    if (k == "NELEC") N = std::atoi(v.c_str());
  }
  Elem I0; sbd::oneInt<Elem> I1; sbd::twoInt<Elem> I2;
  sbd::SetupIntegrals(fcidump, L, N, I0, I1, I2);
  const size_t bl = 20;
  const size_t nword = (2*L + bl - 1) / bl;
  sbd::det_vector<size_t>::init_elem_size(nword);
  sbd::det_vector<size_t, sbd::det_kind::half>::init_elem_size((L + bl - 1)/bl);

  // A reference determinant: lowest N spin-orbitals, alternating spins => Sz=0.
  std::vector<size_t> ref(nword, 0);
  for (int e = 0; e < N; ++e) sbd::setocc(ref, bl, e, true);

  sbd::gdb::PT2Scratch s;
  std::vector<sbd::gdb::PT2Perturber<Elem>> out;
  const double eps = 1e-10;
  sbd::gdb::generate_perturbers_from<Elem>(ref, 1.0, eps, bl, L, I0, I1, I2, s, out);

  std::cout << "reference: N=" << N << " electrons in 2L=" << 2*L << " spin-orbitals\n";
  std::cout << "perturbers generated: " << out.size() << "\n";

  // --- property 1: particle number conserved
  int bad_n = 0;
  for (auto & p : out) if (sbd::bitcount(p.det, bl, 2*L) != N) ++bad_n;
  std::cout << "  particle number wrong: " << bad_n << " (want 0)\n";

  // --- property 2: Sz conserved
  auto sz2 = [&](const std::vector<size_t> & d) {
    int na = 0, nb = 0;
    for (int x = 0; x < 2*L; ++x) if (sbd::getocc(d, bl, x)) { (x % 2 == 0 ? na : nb)++; }
    return na - nb;
  };
  const int sz_ref = sz2(ref);
  int bad_sz = 0;
  for (auto & p : out) if (sz2(p.det) != sz_ref) ++bad_sz;
  std::cout << "  Sz wrong (ref Sz2=" << sz_ref << "): " << bad_sz << " (want 0)\n";

  // --- property 3: every emitted numerator really is H_ai (c_i = 1 here)
  size_t od = 0; std::vector<int> c(2*L, 0), d(2*L, 0);  // Hij writes c[]/d[] raw
  double worst = 0.0;
  for (auto & p : out) {
    Elem h = sbd::Hij(ref, p.det, bl, L, c, d, I0, I1, I2, od);
    worst = std::max(worst, std::abs(h - p.num));
  }
  std::cout << "  max |num - Hij|: " << std::scientific << worst << " (want 0)\n";

  // --- property 4: every |num| is above threshold, and none is the reference
  int below = 0, is_ref = 0;
  for (auto & p : out) { if (std::abs(p.num) <= eps) ++below; if (p.det == ref) ++is_ref; }
  std::cout << "  below threshold: " << below << " (want 0);  equals reference: "
            << is_ref << " (want 0)\n";

  // --- property 5: excitation rank is 1 or 2 only
  std::map<int,int> rank_hist;
  for (auto & p : out) {
    int diff = 0;
    for (int x = 0; x < 2*L; ++x)
      if (sbd::getocc(p.det, bl, x) != sbd::getocc(ref, bl, x)) ++diff;
    rank_hist[diff/2]++;
  }
  std::cout << "  excitation ranks:";
  for (auto & [r, n] : rank_hist) std::cout << " rank" << r << "=" << n;
  std::cout << " (want only 1 and 2)\n";

  // --- property 6: haa really is the diagonal
  double worst_d = 0.0;
  for (auto & p : out) {
    double e = sbd::ZeroExcite(p.det, bl, (size_t)L, I0, I1, I2);
    worst_d = std::max(worst_d, std::abs(e - p.haa));
  }
  std::cout << "  max |haa - ZeroExcite|: " << worst_d << " (want 0)\n";

  // --- property 7: raising the threshold can only remove perturbers, and the
  //     survivors must be exactly those above it
  std::vector<sbd::gdb::PT2Perturber<Elem>> out2;
  sbd::gdb::generate_perturbers_from<Elem>(ref, 1.0, 1e-3, bl, L, I0, I1, I2, s, out2);
  size_t expect = 0;
  for (auto & p : out) if (std::abs(p.num) > 1e-3) ++expect;
  std::cout << "  eps=1e-3: got " << out2.size() << ", expected " << expect
            << (out2.size() == expect ? "  OK" : "  MISMATCH") << "\n";

  MPI_Finalize();
  return 0;
}
