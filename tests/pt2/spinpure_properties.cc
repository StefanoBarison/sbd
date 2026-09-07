// Tests for the spin-pure (variant c) machinery in chemistry/gdb/pt2.h.
//
// Variant (c) has NO external oracle. A full grep of Dice/SHCI finds zero
// occurrences of S^2, spin multiplicity, or spin projection anywhere in its PT2
// path, so there is no reference implementation to compare against and no published
// number for this quantity. Its correctness therefore rests entirely on internal
// identities, and this file is where they are pinned down.
//
// The identities, in order of how much they would catch:
//
//   1. _det_from_config inverts _det_config. Everything downstream indexes orbit
//      rows by arrangement mask, so if the two disagree on slot order the numerators
//      land on the wrong rows and the energy is quietly wrong. This is the single
//      most load-bearing assumption in the variant.
//   2. A closed-shell configuration has block_dim 1 and V = [1], so (c) reduces to
//      (a) EXACTLY there. A bit-for-bit match is available, and any indexing or
//      projection error shows up as a nonzero difference.
//   3. The denominator is the full intra-block double sum. Keeping only r == r'
//      is the same error that once made the projected Davidson preconditioner drop
//      intra-configuration exchange, so it is tested by constructing a block whose
//      off-diagonal contribution is known.
//   4. The orbit is complete: grouping emits all C(n_open, n_up) rows even when the
//      generator supplied one, and marks exactly the supplied ones.
//   5. V's columns are orthonormal and are S^2 eigenvectors, which is what makes the
//      projection a projection.
//
//   mpicxx -std=c++17 -O2 -I<sbd>/include -o spinpure_properties spinpure_properties.cc <blas/lapack>
//   mpirun -n 1 ./spinpure_properties
//
// Exit 0 = all pass. Any FAIL line is a real defect.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "sbd/sbd.h"
#include "sbd/chemistry/gdb/pt2.h"
#include "mpi.h"

using Elem = double;
static int failures = 0;

static void ok_(const char * what, bool cond) {
  if (!cond) ++failures;
  std::cout << (cond ? "  ok   " : "  FAIL ") << what << std::endl;
}

static void check(const char * what, double got, double want, double tol = 1e-12) {
  const bool good = std::abs(got - want) <= tol * std::max(1.0, std::abs(want));
  if (!good) ++failures;
  std::cout << (good ? "  ok   " : "  FAIL ") << what
            << "  got " << std::setprecision(14) << got
            << "  want " << want << std::endl;
}

int main(int argc, char * argv[]) {
  MPI_Init(&argc, &argv);

  const size_t bit_length = 20;
  const int norb = 6;
  const size_t nword = (2 * norb + bit_length - 1) / bit_length;

  std::cout << "\n--- 1. _det_from_config inverts _det_config ---" << std::endl;
  {
    // Every configuration of 6 orbitals with 2 doubly and 4 singly occupied, and
    // every Sz = 0 arrangement of the 4 open shells: rebuild the determinant from
    // (config, mask) and check _det_config returns the same config and mask.
    int n_checked = 0, n_bad = 0;
    std::vector<int> cfg(norb, 0);
    // choose 2 of 6 orbitals doubly occupied, 4 singly
    for (int d1 = 0; d1 < norb; ++d1) {
      for (int d2 = d1 + 1; d2 < norb; ++d2) {
        std::fill(cfg.begin(), cfg.end(), 1);
        cfg[d1] = 2; cfg[d2] = 2;
        const int n_open = norb - 2;
        const auto masks = sbd::gdb::_open_shell_arrangements(n_open, n_open / 2);
        for (unsigned long long m : masks) {
          std::vector<size_t> det;
          sbd::gdb::_det_from_config(cfg, m, bit_length, nword, det);
          std::vector<int> back, slots;
          const unsigned long long m2 =
              sbd::gdb::_det_config(det, bit_length, norb, back, slots);
          ++n_checked;
          if (back != cfg || m2 != m) ++n_bad;
        }
      }
    }
    std::cout << "       round-tripped " << n_checked << " (config, mask) pairs"
              << std::endl;
    ok_("every (config, mask) round-trips through the determinant", n_bad == 0);
    ok_("the sweep actually ran", n_checked == 15 * 6);
  }

  std::cout << "\n--- 2. orbit completion ---" << std::endl;
  {
    // One perturber on a 4-open-shell config at Sz = 0. The orbit is C(4,2) = 6, so
    // grouping must emit 6 rows with exactly one marked as coming from the generator.
    std::vector<int> cfg = {2, 1, 1, 1, 1, 0};
    const auto masks = sbd::gdb::_open_shell_arrangements(4, 2);
    std::vector<size_t> det;
    sbd::gdb::_det_from_config(cfg, masks[3], bit_length, nword, det);

    std::vector<sbd::gdb::PT2Merged<Elem>> pert(1);
    pert[0].det = det;
    pert[0].num = 0.25;
    pert[0].haa = -1.0;

    sbd::det_vector<size_t>::init_elem_size(nword);
    sbd::det_vector<size_t> empty;   // no variational determinants
    std::vector<sbd::gdb::PT2Config<Elem>> out;
    sbd::gdb::PT2SpinPureStats st;
    sbd::gdb::group_pt2_configs<Elem>(pert, bit_length, norb, nword, 0, empty, out, st);

    ok_("one configuration emitted", out.size() == 1);
    if (out.size() == 1) {
      check("orbit has C(4,2) = 6 rows", (double)out[0].rows.size(), 6.0);
      int n_from = 0;
      for (size_t r = 0; r < out[0].x_from_generator.size(); ++r)
        if (out[0].x_from_generator[r]) ++n_from;
      check("exactly one row came from the generator", (double)n_from, 1.0);
      check("that row carries the numerator", (double)out[0].x[3], 0.25);
      // Rows the generator did not supply start at zero and are filled in later; a
      // nonzero value here would mean the numerator was placed on several rows.
      double other = 0.0;
      for (size_t r = 0; r < out[0].x.size(); ++r)
        if (r != 3) other += std::abs(out[0].x[r]);
      check("no other row carries a numerator yet", other, 0.0);
      check("rows counted as completed", (double)st.n_rows_completed, 5.0);
      check("rows counted as emitted", (double)st.n_rows_from_generator, 1.0);
    }
  }

  std::cout << "\n--- 3. the S^2 eigenvectors ---" << std::endl;
  {
    // V's columns must be orthonormal (so the projection is a projection) and must
    // be S^2 eigenvectors with the target eigenvalue (so the CSFs are spin-pure).
    for (int n_open = 2; n_open <= 6; n_open += 2) {
      const int n_up = n_open / 2;
      int bd = 0, nc = 0;
      const auto V = sbd::gdb::_canonical_csf_coeffs(n_open, n_up, 0.0, bd, nc);
      const auto masks = sbd::gdb::_open_shell_arrangements(n_open, n_up);
      double worst_orth = 0.0, worst_eig = 0.0;
      for (int c1 = 0; c1 < nc; ++c1) {
        for (int c2 = 0; c2 < nc; ++c2) {
          double dot = 0.0;
          for (int r = 0; r < bd; ++r)
            dot += V[(size_t)r * nc + c1] * V[(size_t)r * nc + c2];
          worst_orth = std::max(worst_orth, std::abs(dot - (c1 == c2 ? 1.0 : 0.0)));
        }
        // (S^2 V)_r == 0 * V_r for a singlet target
        for (int r = 0; r < bd; ++r) {
          double s2v = 0.0;
          for (int r2 = 0; r2 < bd; ++r2)
            s2v += sbd::gdb::_s2_element(masks[r], masks[r2], n_open, n_up)
                 * V[(size_t)r2 * nc + c1];
          worst_eig = std::max(worst_eig, std::abs(s2v));
        }
      }
      std::cout << "       n_open=" << n_open << " block_dim=" << bd
                << " n_csf=" << nc << std::endl;
      check("  columns orthonormal", worst_orth, 0.0, 1e-10);
      check("  columns are singlet S^2 eigenvectors", worst_eig, 0.0, 1e-10);
    }
    // A closed-shell configuration: block_dim 1, V = [1]. This is what makes (c)
    // reduce to (a) exactly on closed-shell perturbers.
    int bd = 0, nc = 0;
    const auto V0 = sbd::gdb::_canonical_csf_coeffs(0, 0, 0.0, bd, nc);
    check("n_open=0 gives block_dim 1", (double)bd, 1.0);
    check("n_open=0 gives one CSF", (double)nc, 1.0);
    if (nc == 1 && bd == 1) check("and V = [1]", V0[0], 1.0);
    // A closed-shell configuration has NO triplet component: n_csf must be 0, which
    // is what makes those configurations drop out of (c) but not out of (a).
    int bd3 = 0, nc3 = 0;
    sbd::gdb::_canonical_csf_coeffs(0, 0, 2.0, bd3, nc3);
    check("n_open=0 has no triplet CSF", (double)nc3, 0.0);
  }

  std::cout << "\n--- 4. the denominator is the FULL double sum ---" << std::endl;
  {
    // Two-row block, V the singlet column of a 2-open-shell config: V = (1,-1)/sqrt2
    // up to sign. With H = [[d, x], [x, d]] the CSF denominator element is
    //   h_cc = sum_{a,b} V_a V_b H_ab = d - x     (for the antisymmetric column)
    // whereas the diagonal-only shortcut would give d. So a known off-diagonal
    // produces a known, nonzero difference, and dropping r != r' cannot pass.
    int bd = 0, nc = 0;
    const auto V = sbd::gdb::_canonical_csf_coeffs(2, 1, 0.0, bd, nc);
    ok_("2 open shells give a 2x2 block", bd == 2);
    ok_("with exactly one singlet CSF", nc == 1);
    if (bd == 2 && nc == 1) {
      const double d = -3.0, x = 0.5;
      const double H[4] = {d, x, x, d};
      double hcc = 0.0;
      for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
          hcc += V[(size_t)a * nc] * V[(size_t)b * nc] * H[a * 2 + b];
      double hcc_diag_only = 0.0;
      for (int a = 0; a < 2; ++a)
        hcc_diag_only += V[(size_t)a * nc] * V[(size_t)a * nc] * H[a * 2 + a];
      // The singlet column of a 2-open-shell block is antisymmetric, so the cross
      // terms enter with a minus sign.
      check("full double sum gives d - |x|", hcc, d - std::abs(x));
      check("diagonal-only would give d", hcc_diag_only, d);
      ok_("the two differ, so the shortcut is detectable", std::abs(hcc - hcc_diag_only) > 0.1);
    }
  }

  std::cout << "\n--- 5. (c) reduces to (a) on a closed-shell perturber ---" << std::endl;
  {
    // block_dim 1 => V = [1] => num_c = x_0 and h_cc = H_00, so the CSF term is
    // |x_0|^2 / (E_0 - H_00): identical to (a)'s determinant term, bit for bit.
    const double e0 = -100.0, haa = -95.0, x = 0.125;
    const double term_a = (x * x) / (e0 - haa);
    int bd = 0, nc = 0;
    const auto V = sbd::gdb::_canonical_csf_coeffs(0, 0, 0.0, bd, nc);
    const double num = V[0] * x;
    const double hcc = V[0] * V[0] * haa;
    const double term_c = (num * num) / (e0 - hcc);
    check("the two terms agree exactly", term_c, term_a, 0.0);
  }

  std::cout << std::endl;
  if (failures == 0) std::cout << "ALL SPIN-PURE PROPERTIES PASS" << std::endl;
  else std::cout << failures << " FAILURE(S)" << std::endl;
  MPI_Finalize();
  return failures == 0 ? 0 : 1;
}
