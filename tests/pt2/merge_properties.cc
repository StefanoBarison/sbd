// Tests for the PT2 merge / set-difference / accumulation step
// (chemistry/gdb/pt2.h).
//
// The first test is the reason this file exists. A perturber reachable from several
// references contributes |sum_i H_ai c_i|^2, not sum_i |H_ai c_i|^2. Getting that
// wrong changes the answer, does not crash, and does not show up as a non-monotonic
// epsilon2 sweep -- so it would survive every other check in this project. It is
// tested here on hand-built input with a known answer, before any real energy is
// computed.
//
//   mpicxx -std=c++17 -O2 -I<sbd>/include -o merge_properties merge_properties.cc <blas/lapack>
//   ./merge_properties
//
// Exit 0 = all pass. Any FAIL line is a real defect.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "sbd/sbd.h"
#include "sbd/chemistry/gdb/pt2.h"
#include "mpi.h"

using Elem = double;
static int failures = 0;

static void check(const char * what, double got, double want, double tol = 1e-12) {
  const bool ok = std::abs(got - want) <= tol * std::max(1.0, std::abs(want));
  if (!ok) ++failures;
  std::cout << (ok ? "  ok   " : "  FAIL ") << what
            << ": got " << std::setprecision(15) << got
            << ", want " << want << std::endl;
}

static void check_eq(const char * what, long got, long want) {
  const bool ok = (got == want);
  if (!ok) ++failures;
  std::cout << (ok ? "  ok   " : "  FAIL ") << what
            << ": got " << got << ", want " << want << std::endl;
}

/// A determinant with a single word, so the tests are readable.
static std::vector<size_t> D(size_t bits) { return std::vector<size_t>{bits}; }

int main(int argc, char ** argv) {
  sbd::MpiInitHybrid(&argc, &argv);
  sbd::det_vector<size_t>::init_elem_size(1);

  // ------------------------------------------------------------------------
  // 1. SUM BEFORE SQUARE. Two references reach the same perturber with
  //    H_a1 c_1 = 0.3 and H_a2 c_2 = 0.4. E_0 - H_aa = -1.
  //      correct:   |0.3 + 0.4|^2 / -1 = -0.49
  //      the error: (0.09 + 0.16) / -1 = -0.25
  //    The two differ by a factor of ~2, so this is not a subtle discrepancy --
  //    but nothing except this test would reveal which one the code computes.
  // ------------------------------------------------------------------------
  std::cout << "1. numerators summed before squaring" << std::endl;
  {
    std::vector<sbd::gdb::PT2Perturber<Elem>> in(2);
    in[0].det = D(0b1011); in[0].num = 0.3; in[0].haa = 1.0;
    in[1].det = D(0b1011); in[1].num = 0.4; in[1].haa = 1.0;
    std::vector<sbd::gdb::PT2Merged<Elem>> merged;
    sbd::gdb::merge_pt2_perturbers(in, merged);
    check_eq("distinct perturbers after merge", (long)merged.size(), 1);
    check_eq("parents recorded", merged.empty() ? -1 : merged[0].n_parents, 2);
    check("merged numerator", merged.empty() ? 0.0 : merged[0].num, 0.7);
    const auto r = sbd::gdb::accumulate_pt2(merged, /*e0=*/0.0, /*floor=*/1e-12);
    check("energy = |0.3+0.4|^2 / (0-1)", r.energy, -0.49);
    // And state the wrong answer explicitly, so a future regression is recognisable.
    if (std::abs(r.energy - (-0.25)) < 1e-12) {
      std::cout << "  FAIL this is sum-of-squares (-0.25), not square-of-sum"
                << std::endl;
      ++failures;
    }
  }

  // ------------------------------------------------------------------------
  // 2. Cancellation must survive. Opposite-sign contributions to one perturber
  //    cancel exactly; summing squares would instead give a spurious 2x.
  // ------------------------------------------------------------------------
  std::cout << "2. opposite-sign contributions cancel" << std::endl;
  {
    std::vector<sbd::gdb::PT2Perturber<Elem>> in(2);
    in[0].det = D(0b0111); in[0].num =  0.5; in[0].haa = 2.0;
    in[1].det = D(0b0111); in[1].num = -0.5; in[1].haa = 2.0;
    std::vector<sbd::gdb::PT2Merged<Elem>> merged;
    sbd::gdb::merge_pt2_perturbers(in, merged);
    check("merged numerator is zero", merged.empty() ? 1.0 : merged[0].num, 0.0);
    const auto r = sbd::gdb::accumulate_pt2(merged, 0.0, 1e-12);
    check("energy contribution vanishes", r.energy, 0.0);
  }

  // ------------------------------------------------------------------------
  // 3. Distinct perturbers stay distinct, and their energies add.
  // ------------------------------------------------------------------------
  std::cout << "3. distinct perturbers are not merged" << std::endl;
  {
    std::vector<sbd::gdb::PT2Perturber<Elem>> in(3);
    in[0].det = D(0b0011); in[0].num = 0.1; in[0].haa = 1.0;
    in[1].det = D(0b0101); in[1].num = 0.2; in[1].haa = 3.0;
    in[2].det = D(0b0011); in[2].num = 0.1; in[2].haa = 1.0;
    std::vector<sbd::gdb::PT2Merged<Elem>> merged;
    sbd::gdb::merge_pt2_perturbers(in, merged);
    check_eq("distinct perturbers", (long)merged.size(), 2);
    const auto r = sbd::gdb::accumulate_pt2(merged, 0.0, 1e-12);
    // (0.1+0.1)^2/(0-1) + (0.2)^2/(0-3) = -0.04 - 0.0133333...
    check("energy sums over perturbers", r.energy, -0.04 - 0.04/3.0);
  }

  // ------------------------------------------------------------------------
  // 4. Set-difference removes variational determinants -- INCLUDING one that was
  //    duplicated before the merge. This is why dedup must run first: a
  //    single-pass set-difference over a list with duplicates removes only the
  //    first copy, leaving the rest to double-count the variational energy.
  // ------------------------------------------------------------------------
  std::cout << "4. variational determinants are removed" << std::endl;
  {
    sbd::det_vector<size_t> var(2);
    var[0] = D(0b0011);
    var[1] = D(0b1100);
    sbd::sort_bitarray(var);

    std::vector<sbd::gdb::PT2Perturber<Elem>> in(4);
    in[0].det = D(0b0011); in[0].num = 0.5; in[0].haa = 1.0;  // in var, twice
    in[1].det = D(0b0011); in[1].num = 0.5; in[1].haa = 1.0;
    in[2].det = D(0b1100); in[2].num = 0.7; in[2].haa = 1.0;  // in var
    in[3].det = D(0b0101); in[3].num = 0.9; in[3].haa = 4.0;  // a real perturber
    std::vector<sbd::gdb::PT2Merged<Elem>> merged;
    sbd::gdb::merge_pt2_perturbers(in, merged);
    check_eq("merged to distinct", (long)merged.size(), 3);
    const size_t removed = sbd::gdb::remove_variational(merged, var);
    check_eq("removed", (long)removed, 2);
    check_eq("survivors", (long)merged.size(), 1);
    if (merged.size() == 1) {
      check("survivor numerator", merged[0].num, 0.9);
      const auto r = sbd::gdb::accumulate_pt2(merged, 0.0, 1e-12);
      check("energy from survivor only", r.energy, -0.81/4.0);
    } else { ++failures; }
  }

  // ------------------------------------------------------------------------
  // 5. The denominator floor keeps the SIGN. Flipping it would turn a downward
  //    correction into an upward one, which is worse than a large term.
  // ------------------------------------------------------------------------
  std::cout << "5. denominator floor preserves sign" << std::endl;
  {
    std::vector<sbd::gdb::PT2Merged<Elem>> merged(1);
    merged[0].det = D(0b0110); merged[0].num = 1.0;
    merged[0].haa = 1.0;   // e0 - haa = -1e-15, i.e. below any sane floor
    const auto r = sbd::gdb::accumulate_pt2(merged, 1.0 - 1e-15, 1e-8);
    check_eq("floor triggered", (long)r.n_floored, 1);
    check("still negative (sign kept)", r.energy < 0.0 ? -1.0 : 1.0, -1.0);
    check("magnitude is 1/floor", std::abs(r.energy), 1.0e8, 1e-6);
  }

  // ------------------------------------------------------------------------
  // 6. A ground-state correction is negative: every H_aa above E_0 gives a
  //    negative term, which is the sign convention the app relies on.
  // ------------------------------------------------------------------------
  std::cout << "6. sign convention" << std::endl;
  {
    std::vector<sbd::gdb::PT2Merged<Elem>> merged(3);
    for (int i = 0; i < 3; ++i) {
      merged[i].det = D(0b1u << (i + 4));
      merged[i].num = 0.1 * (i + 1);
      merged[i].haa = -100.0 + (i + 1);      // all above e0
    }
    const auto r = sbd::gdb::accumulate_pt2(merged, -108.0, 1e-8);
    check("E_PT2 < 0", r.energy < 0.0 ? -1.0 : 1.0, -1.0);
    check_eq("nothing floored", (long)r.n_floored, 0);
    check("psi1_norm2 > 0", r.psi1_norm2 > 0.0 ? 1.0 : -1.0, 1.0);
  }

  std::cout << (failures == 0 ? "\nALL PASS" : "\nFAILURES: ")
            << (failures ? std::to_string(failures) : "") << std::endl;
  MPI_Finalize();
  return failures == 0 ? 0 : 1;
}
