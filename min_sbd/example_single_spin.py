"""Example: drive the GDB single-spin (Option 2) solver through the Python wrapper.

Loads the N2 (R=2.0, 6-31g, 10e/16o) FCIDUMP + a completed determinant file,
then asks SBDGdbSolver for the 4 lowest TRIPLET roots (single_spin=3, i.e.
spin multiplicity 2S+1=3) using the
stored-matrix path (method=1). Prints per-root energies + <S^2>.

Run:  python example_single_spin.py
(needs the gdb `diag` binary built, mpirun on PATH, conda env `sqd`.)
"""
import glob
import os
import shutil
import sys

import numpy as np
from pyscf.tools import fcidump

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbd_GDB_solver import SBDGdbSolver

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DIAG = os.path.join(REPO, "apps/chemistry_gdb_selected_basis_diagonalization/diag")
# Inputs are not in the repo (FCIDUMPs and determinant lists are large and
# molecule-specific), so point at them with SBD_EXAMPLE_FCIDUMP / SBD_EXAMPLE_DETFILE.
# The fallbacks are $HOME-relative rather than absolute so the script at least fails
# with a readable message on a machine that is not the author's.
FCIDUMP = os.environ.get(
    "SBD_EXAMPLE_FCIDUMP",
    os.path.expanduser("~/Downloads/for_claude_n2/N2_R2.0_6-31g_10e_16o.dat"))
DETFILE = os.environ.get(
    "SBD_EXAMPLE_DETFILE",
    os.path.expanduser("~/sbd_dets_top50.txt"))
for _label, _path in (("FCIDUMP", FCIDUMP), ("DETFILE", DETFILE)):
    if not os.path.isfile(_path):
        raise SystemExit(
            f"{_label} not found: {_path}\n"
            f"Set SBD_EXAMPLE_{_label} to a readable path, e.g.\n"
            f"  SBD_EXAMPLE_{_label}=/path/to/file python {os.path.basename(__file__)}")

NORB = 16
MULTIPLICITY = 3   # spin multiplicity 2S+1: 1=singlet, 2=doublet, 3=triplet, 4=quartet, ...
NROOTS = 4


def deinterleave(bitstring: str, norb: int):
    """SBD interleaved bitstring -> (alpha_int, beta_int).
    Bit 2p (from the right) = alpha orbital p, bit 2p+1 = beta orbital p.
    """
    L = 2 * norb
    a = b = 0
    for p in range(norb):
        if bitstring[L - 1 - (2 * p)] == "1":
            a |= (1 << p)
        if bitstring[L - 1 - (2 * p + 1)] == "1":
            b |= (1 << p)
    return a, b


def main():
    # integrals from the FCIDUMP
    m = fcidump.read(FCIDUMP)
    hcore = m["H1"]
    eri = m["H2"]
    nelec = m["NELEC"]
    na = nb = nelec // 2

    # completed determinants -> paired (strs_a, strs_b) integer arrays
    strs_a, strs_b = [], []
    for line in open(DETFILE):
        bs = line.strip()
        if not bs:
            continue
        a, b = deinterleave(bs, NORB)
        strs_a.append(a)
        strs_b.append(b)
    strs_a = np.array(strs_a, dtype=np.int64)
    strs_b = np.array(strs_b, dtype=np.int64)
    print(f"loaded {len(strs_a)} determinants, mult={MULTIPLICITY}, nroots={NROOTS}")

    solver = SBDGdbSolver(
        sbd_binary=DIAG,
        mpi_np=1,
        method=1,                 # stored matrix (fast for this size)
        nroots=NROOTS,
        single_spin=MULTIPLICITY,  # Option 2: target spin multiplicity 2S+1
        davidson_block=20,
        davidson_iterations=80,
        davidson_tolerance=1e-8,
        b_comm_size=1,            # single_spin requires b_comm==1
        t_comm_size=1,
        bit_length=64,
        rdm=True,                 # write per-root RDMs -> enables <S^2>
        do_redist_det=True,
        temp_dir=os.path.join(HERE, "_example_work"),  # known location for RDM files
        clean_temp_dir=False,     # keep workdir so we can read the RDM files
        verbose=True,
    )
    # fresh work dir so we read this run's RDM files
    shutil.rmtree(os.path.join(HERE, "_example_work"), ignore_errors=True)

    results = solver.sci_solver_callable(
        ci_strs_batches=[(strs_a, strs_b)],
        hcore=hcore,
        eri=eri,
        norb=NORB,
        nelec=(na, nb),
    )
    res = results[0]
    ecore = m["ECORE"]
    print(f"\nroot-0 electronic energy: {res.energy:.8f}   (+ECORE {ecore:.6f} = {res.energy + ecore:.6f})")

    # per-root <S^2> from the RDM files this run wrote in its (kept) workdir
    work = sorted(glob.glob(os.path.join(HERE, "_example_work", "sbd_gdb_solver_*")))
    workdir = work[-1] if work else "."
    spins = SBDGdbSolver.parse_per_root_spin(workdir, norb=NORB, nroots=NROOTS)
    print("\nper-root spin:")
    for s in spins:
        S = round(s["S"]) if s["spin_square"] is not None else None
        sq = f"{s['spin_square']:.4f}" if s["spin_square"] is not None else "n/a"
        print(f"  root {s['root']}: <S^2> = {sq}  ->  S = {S}")


if __name__ == "__main__":
    main()
