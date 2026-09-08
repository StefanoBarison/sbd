"""Example: Epstein-Nesbet PT2 on top of a finished GDB diagonalization.

Shows the two things worth knowing:

  1. PT2 is a SECOND process, driven by SBDPT2Corrector, reading the wavefunction the
     solve wrote. The solve is not re-run, so `epsilon2` can be swept cheaply -- which
     is the normal way this quantity is used.

  2. The two variants answer different questions. Variant (a) sums over determinant
     perturbers and is comparable to published SHCI numbers, but its perturber space
     spans every spin, so `E_var + E_PT2` is NOT a single-spin energy. Variant (c)
     sums over target-S CSF perturbers with one denominator per CSF, and is the one
     you may quote alongside a single-spin variational energy.

Run:  python example_pt2.py
      SBD_EXAMPLE_FCIDUMP=... SBD_EXAMPLE_DETFILE=... SBD_EXAMPLE_WF=... python example_pt2.py

Needs the `pt2` binary built (`cd apps/chemistry_gdb_pt2 && make`) and mpirun on PATH.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbd_PT2_solver import SBDPT2Corrector

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PT2 = os.path.join(REPO, "apps/chemistry_gdb_pt2/pt2")

# Inputs are not in the repo (FCIDUMPs, determinant lists and wavefunctions are large
# and molecule-specific). The three MUST come from the same solve: PT2 aborts if the
# wavefunction is not fully covered by the determinant list, but it cannot detect a
# FCIDUMP from a different calculation.
FCIDUMP = os.environ.get(
    "SBD_EXAMPLE_FCIDUMP",
    os.path.expanduser("~/Downloads/for_claude_n2/N2_R2.0_6-31g_10e_16o.dat"))
DETFILE = os.environ.get("SBD_EXAMPLE_DETFILE", "/tmp/n2reg/det_top100.txt")
# Either a merged .npz (preferred) or a shard prefix. A shard prefix keeps any root
# tag but drops the rank tag: wf_root0000000.bin -> "wf_root0".
WAVEFUNCTION = os.environ.get("SBD_EXAMPLE_WF", "/tmp/pt2test/w_top100_root0")

NORB = 16
E_VAR = float(os.environ.get("SBD_EXAMPLE_E0", "-108.729420075"))
MULTIPLICITY = 1   # 2S+1: 1=singlet, 3=triplet

for _label, _path in (("FCIDUMP", FCIDUMP), ("DETFILE", DETFILE)):
    if not os.path.isfile(_path):
        raise SystemExit(
            f"{_label} not found: {_path}\n"
            f"Set SBD_EXAMPLE_{_label} to a readable path.")
if not os.path.isfile(PT2):
    raise SystemExit(f"pt2 binary not found: {PT2}\nBuild it: cd apps/chemistry_gdb_pt2 && make")

is_npz = str(WAVEFUNCTION).endswith(".npz")


def run(variant, multiplicity=None, epsilon2=1e-8, **kw):
    corr = SBDPT2Corrector(
        pt2_binary=PT2,
        fcidump_path=FCIDUMP,
        variant=variant,
        single_spin=multiplicity,
        epsilon2=epsilon2,
        bit_length=20,          # MUST match the solve
        **kw,
    )
    # Warnings here are the binary's own diagnostics re-raised (positive E_PT2, floored
    # denominators, a perturber mixing too strongly). Do not filter them away in real
    # use: each marks a way the number can be misleading while still looking fine.
    if is_npz:
        return corr.correct_from_npz(WAVEFUNCTION, e0=E_VAR, norb=NORB, detfile=DETFILE)
    return corr.correct_from_shards(WAVEFUNCTION, e0=E_VAR, norb=NORB, detfile=DETFILE)


print("=" * 74)
print(" 1. Both variants, same wavefunction")
print("=" * 74)
res = {}
for variant, mult in (("a", None), ("c", MULTIPLICITY)):
    out = run(variant, mult)
    res[variant] = out
    tag = "single-spin energy" if out.is_spin_pure else "NOT a single-spin energy"
    print(f"\n variant {variant}:")
    print(f"   E_var         = {out.e_var:.9f}")
    print(f"   E_PT2         = {out.e_pt2:.9f}")
    print(f"   E_var + E_PT2 = {out.e_total:.9f}   <- {tag}")
    print(f"   |Psi_1|^2 = {out.psi1_norm2}   max |c_a^(1)| = {out.max_c1}")
    print(f"   references={out.n_references} emitted={out.n_emitted} "
          f"unique_dets={out.n_unique_dets} floored={out.n_floored}")
    if out.variant == "c":
        print(f"   configs={out.n_configs} (no_target_S={out.n_configs_no_target_s}, "
              f"partly_variational={out.n_configs_partly_variational})")
        print(f"   orbit_rows={out.n_orbit_rows} completed={out.n_rows_completed} "
              f"csfs={out.n_csf}")
    print(f"   timing: {out.timing}")

gap = res["c"].e_pt2 - res["a"].e_pt2
print(f"\n (a) - (c) gap = {gap*1000:+.3f} mH")
print(" Expected, and not an error: (a) also counts perturbers with no target-S")
print(" component. The gap shrinks as the variational space grows.")

print()
print("=" * 74)
print(" 2. epsilon2 sweep -- cheap, because the solve is not repeated")
print("=" * 74)
print(f"{'epsilon2':>10}  {'E_PT2':>16}  {'E_var+E_PT2':>16}  {'csfs':>8}")
for eps in (1e-6, 1e-7, 1e-8, 1e-9):
    out = run("c", MULTIPLICITY, epsilon2=eps, clean_temp_dir=True)
    print(f"{eps:>10g}  {out.e_pt2:>16.12f}  {out.e_total:>16.9f}  {out.n_csf or '-':>8}")
print("\n Variant (c) barely moves with epsilon2: rows the screening drops are")
print(" recovered by the orbit-completion step rather than lost.")

print()
print("=" * 74)
print(" 3. What to check before quoting the number")
print("=" * 74)
out = res["c"]
print(f"   E_PT2 < 0                     : {out.e_pt2 < 0}   (must be, for a ground root)")
print(f"   no floored denominators       : {out.n_floored == 0}")
print(f"   max |c_a^(1)| well below 0.5  : {out.max_c1}")
print(f"   completion actually ran       : {bool(out.n_rows_completed)}")
print(f"   is a single-spin energy       : {out.is_spin_pure}")
print(f"\n   workdir kept for inspection: {out.workdir}")
print("   (clean_temp_dir defaults to False here: it holds the stdout that is the")
print("    only provenance this number has.)")
