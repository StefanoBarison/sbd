"""
SBD GDB (General Determinant Basis) sci_solver implementation.

Shells out to RIKEN's ``sbd`` binary — specifically the
``chemistry_gdb_selected_basis_diagonalization`` app — for each batch's
diagonalization.  Unlike the TPB wrapper (``sbd_solver.py``), the GDB app
diagonalizes over an **explicit, non-cartesian** set of full determinants:
each determinant encodes both alpha- and beta-spin occupancies in a single
interleaved bitstring, and the wavefunction is a 1-D vector of length M
(one amplitude per determinant).

Non-cartesian semantics
-----------------------
The TPB app constructs the Hilbert space as the tensor product of an alpha
list and a beta list, ``{α₁,…,αM} ⊗ {β₁,…,βN}``, giving M·N determinants
and a 2-D M×N amplitude matrix (qiskit's ``SCIState``).

The GDB app takes *explicit* determinants ``{(α₁,β₁),…,(αM,βM)}`` — exactly
those listed in the input files, no expansion — and its wavefunction is a
**1-D** vector of length M.  This wrapper returns ``experimental_SCIResult``
objects whose ``sci_state`` is an ``experimental_SCIState`` with 1-D
``amplitudes`` (paired non-cartesian mode), usable in configuration-recovery
loops that never form a cartesian product.

Interleaved bit layout
-----------------------
A GDB full-determinant bitstring of length ``2·norb`` has:

  bit ``2·io`` = alpha-spin occupancy of spatial orbital ``io`` (0-indexed)
  bit ``2·io+1`` = beta-spin occupancy of spatial orbital ``io``

The string is printed MSB-first (leftmost character = highest bit).  This
wrapper interleaves pairs of qiskit-style integers ``(alpha_int, beta_int)``
into this format when writing ``dets.txt``, and un-interleaves the binary
shard contents when recovering amplitudes.

Prerequisites
-------------
- A compiled ``diag`` binary from
  ``sbd/apps/chemistry_gdb_selected_basis_diagonalization/``.
- ``mpirun`` reachable via PATH, or passed explicitly via ``mpirun=...``.
- The ``pyscf`` package for FCIDUMP generation.

Energy convention
-----------------
Each ``experimental_SCIResult.energy`` is the **electronic energy only**.
The FCIDUMP written to disk always has ``ECORE=0``.  Callers add ``e_nuc``
themselves (same convention as qiskit-addon-sqd and the TPB wrapper).

Spin handling
-------------
Two ways to get single-spin roots (this branch): (1) diagonalize the (spin-mixed)
basis with ``nroots`` and filter by ``parse_per_root_spin``; or (2) the projected
Option-2 solver via ``single_spin`` = target multiplicity 2S+1 (1=singlet,
2=doublet, 3=triplet, ...), which returns only that spin.  The ``spin_sq`` argument
of ``sci_solver_callable`` is unused and must stay ``None``.

Thick restart (``restart_keep``)
-------------------------------
Number of lowest Ritz vectors carried across each Davidson subspace collapse.
``None`` (default) lets the binary pick ``max(nroots, davidson_block // 2)``,
which is the tuned heuristic -- prefer that unless you are deliberately
scanning the parameter.

Set it too small and the solver discards most of the subspace at every collapse
and re-climbs it: with ``restart_keep=7``, ``nroots=4`` and
``davidson_block=20`` the subspace reaches ~19 vectors, collapses to 7, and
rebuilds -- ~63% thrown away per cycle.  Raising it to 15 at K=113394 cut the
matvec count 127 -> 104.

Expect the *matvec count* to improve, not necessarily wall-clock.  In the same
K=113394 test the total ``end davidson`` time went 184 s -> 193 s despite the
lower matvec count, because with ``nroots=4`` and ``rdm=True`` most of that
figure is post-solve work (one full ``H*v`` plus a ``Correlation`` call per
root, outside the Davidson loop), not the iteration itself.  Per-phase
instrumentation of the loop at that size showed the loop is only ~52 s of the
~205 s, is 87% matvec, and is >99.9% accounted -- so there is little left to
tune inside it.  Set ``SBD_SS_TIMING=1`` to print that breakdown.

The rule of thumb from PERFORMANCE_LOG.md is ``restart_keep ~ nb/2``; the binary
auto-grows ``nb`` to ``restart_keep + (davidson_block - nroots)`` so a larger
keep does not steal growth room, it only costs RAM (``2 * nb * n_csf`` doubles).

Carryover
---------
``carryover_type`` (0=none, 1=weight truncation, 2/3=HCI heatbath expansion),
``carryover_root`` (which converged root to carry over in multiroot/single-spin),
and ``carryover_options`` (dict of extra ``--flag value`` pairs, e.g.
``{"carryover_ratio": "0.1"}`` or ``{"heatbath_cutoff": "1e-4",
"carryovername": "carryover"}``) are passed through to the binary.  Supply
``carryovername`` in ``carryover_options`` if you want the carried determinants
written to disk.

Seed caveat
-----------
The GDB binary seeds Davidson with ``w[0]=1`` on the first *sorted*
determinant.  If that determinant has no single- or double-excitation
neighbours inside the sampled subspace — common in sample-based subspaces —
Davidson returns the seed's diagonal element as the "ground state" energy,
silently wrong.  No ordering workaround is available from Python because the
binary re-sorts determinants internally.  Verify results against PySCF on a
representative small case before trusting production runs.

Debugging
---------
Each batch's stdout, stderr, and command line are written to
``sbd_stdout.txt`` / ``sbd_stderr.txt`` / ``sbd_command.txt`` inside the
batch's workdir.  On failure the workdir is preserved regardless of
``clean_temp_dir``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import warnings
import math
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pyscf.tools import fcidump as _pyscf_fcidump

from collections.abc import Sequence

from dataclasses import dataclass

from qiskit_addon_sqd.counts import bitstring_matrix_to_integers

# ---------------------------------------------------------------------------
# Import experimental_SCIState / experimental_SCIResult from the sbd root.
# These live two directories above this file (sbd/experimental_SCIState.py),
# outside the physics package hierarchy, so we load them by file path.
# ---------------------------------------------------------------------------
#_exp_path = Path(__file__).resolve().parents[2] / "experimental_SCIState.py"
#_exp_path = "/scratch/sbarison/sbd/apps/chemistry_gdb_selected_basis_diagonalization/experimental_SCIState.py"
#import sys as _sys
#_spec = importlib.util.spec_from_file_location("sbd_experimental_sci_state", _exp_path)
#_mod = importlib.util.module_from_spec(_spec)
#_sys.modules["sbd_experimental_sci_state"] = _mod  # required for @dataclass to resolve __module__
#_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
#experimental_SCIState = _mod.experimental_SCIState
#experimental_SCIResult = _mod.experimental_SCIResult
#experimental_bitstring_matrix_to_ci_strs = _mod.experimental_bitstring_matrix_to_ci_strs


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class SBDGdbSolver:
    """
    Standalone solver wrapping RIKEN's GDB ``diag`` binary.

    Each ``sci_solver_callable`` invocation spawns one subprocess per batch.
    Input CI strings must be **paired**: ``ci_strings[0][i]`` and
    ``ci_strings[1][i]`` are the alpha and beta parts of the i-th explicit
    determinant.  The Hilbert space is the union of those M determinants —
    no cartesian product is formed.

    See module docstring for prerequisites, bit layout, energy convention,
    spin handling, and the seed caveat.
    """

    def __init__(
        self,
        *,
        sbd_binary: str | Path | None = None,
        fcidump_path: str | Path | None = None,
        verbose: bool = False,
        # MPI
        mpirun: str = "mpirun",
        mpi_np: int = 1,
        mpirun_args: list[str] | None = None,
        # Davidson tunables
        davidson_block: int = 10,
        davidson_iterations: int = 4,
        davidson_tolerance: float = 1e-4,
        nroots: int = 1,
        single_spin: int = -1,
        restart_keep: int | None = None,
        carryover_type: int = 0,
        carryover_root: int = 0,
        carryover_options: dict | None = None,
        method: int = 0,
        b_comm_size: int = 1,
        t_comm_size: int = 1,
        bit_length: int = 20,
        do_redist_det: bool = True,
        # Workdir / bookkeeping
        temp_dir: str | Path | None = None,
        clean_temp_dir: bool = True,
        timeout_s: float | None = None,
        extra_cli_args: list[str] | None = None,
        # FCIDUMP handling
        canonicalize_fcidump: bool = True,
        # RDM handling
        rdm: bool = False,
        rdm_root: int | None = None,
    ):
        """
        Args:
            sbd_binary: Path to the GDB ``diag`` executable.  Defaults to
                ``<repo_root>/build/…/chemistry_gdb_selected_basis_diagonalization/diag``.
            fcidump_path: Pre-computed FCIDUMP file.  If not given, a fresh
                one is written from ``hcore``/``eri`` at each call.
            verbose: Print command and workdir for each batch.
            mpirun: MPI launcher (default ``"mpirun"``).
            mpi_np: Total MPI rank count.  Must equal
                ``h_comm_size × b_comm_size × t_comm_size`` where
                ``h_comm_size = mpi_np / (b_comm_size × t_comm_size)``.
            mpirun_args: Extra args after ``-np``.
            davidson_block: Davidson subspace size (``--block``).
            davidson_iterations: Davidson restart cycles (``--iteration``).
            davidson_tolerance: Convergence threshold (``--tolerance``).
            method: 0=matrix-free, 1=matrix-stored (``--method``).
            b_comm_size: Basis communicator size (``--b_comm_size``).
                One wavefunction shard file is written per b_comm rank.
            t_comm_size: Task communicator size (``--t_comm_size``).
            bit_length: Bits per ``size_t`` word in the binary wavefunction
                format (``--bit_length``, default 20).
            do_redist_det: Redistribute determinants for load-balance
                (``--do_redist_det``).  Recommended for multi-rank runs.
            temp_dir: Parent for per-batch workdirs.
            clean_temp_dir: Delete workdir on success.
            timeout_s: Subprocess timeout in seconds.
            extra_cli_args: Appended verbatim to every invocation.
            canonicalize_fcidump: When ``fcidump_path`` is set, roundtrip it
                through pyscf to force ECORE=0 before use.
            rdm: Whether to compute the 1/2-RDM (``--rdm True``) or not (``--rdm False``, default).
            rdm_root: Which root gets a per-root RDM when ``rdm=True``.
                ``None`` (default) computes one for every root, matching the
                binary's default.  An integer restricts it to that root index.

                This is usually the largest single wall-clock lever in a run:
                each per-root RDM is a full ``Correlation()`` pass over the
                determinant space, comparable in cost to the whole Davidson
                solve.  At K=113394 with ``nroots=4`` the per-root RDMs took
                ~150 s of a ~295 s run.  If you only need energies plus the
                carryover determinants, set ``rdm=False``; if you need one
                RDM, set ``rdm_root`` to that root.  Choosing ``rdm_root``
                equal to ``carryover_root`` is cheapest, because the binary
                then reuses that RDM instead of recomputing it for the
                carryover wavefunction (worth a further ~38 s at that size).
                ``rdm_root`` is independent of ``carryover_root``, so
                carryover can still be taken from any root.
        """
        if sbd_binary is None:
            repo_root = Path(__file__).resolve().parents[5]
            sbd_binary = (
                repo_root
                / "build"
                / "energy_solvers"
                / "sbd_gpu_workdir"
                / "sbd"
                / "apps"
                / "chemistry_gdb_selected_basis_diagonalization"
                / "diag"
            )

        sbd_binary_path = Path(sbd_binary).resolve()
        if not sbd_binary_path.is_file() or not os.access(sbd_binary_path, os.X_OK):
            raise FileNotFoundError(
                f"sbd_binary not found or not executable: {sbd_binary_path}"
            )
        self._sbd_binary = str(sbd_binary_path)

        if fcidump_path is not None:
            fp = Path(fcidump_path).resolve()
            if not fp.is_file():
                raise FileNotFoundError(f"fcidump_path not found: {fp}")
            self._fcidump_path: str | None = str(fp)
        else:
            self._fcidump_path = None

        self.verbose = bool(verbose)
        self.mpirun = mpirun
        self.mpi_np = int(mpi_np)
        self.mpirun_args = (
            list(mpirun_args) if mpirun_args is not None else ["-x", "OMP_NUM_THREADS=1"]
        )
        self.davidson_block = int(davidson_block)
        self.davidson_iterations = int(davidson_iterations)
        self.davidson_tolerance = float(davidson_tolerance)
        self.nroots = int(nroots)
        self.single_spin = int(single_spin)
        self.restart_keep = None if restart_keep is None else int(restart_keep)
        # A restart_keep well below nb/2 makes the solver discard most of the
        # subspace at every collapse and re-climb it, which costs matvecs
        # superlinearly in the number of roots. See the "Thick restart" section
        # of the module docstring for the measured K=113394 case.
        # Warn only when restart_keep leaves less history than the binary's own
        # default would, i.e. below max(nroots, davidson_block // 2). Values at
        # or above that default are legitimate tuning choices and stay silent.
        if self.restart_keep is not None:
            _default_keep = max(self.nroots, self.davidson_block // 2)
            if self.restart_keep < _default_keep:
                _growth = max(1, self.davidson_block - self.nroots)
                _nb_eff = self.restart_keep + _growth
                warnings.warn(
                    f"restart_keep={self.restart_keep} is below the tuned "
                    f"default max(nroots, davidson_block//2)={_default_keep} "
                    f"(nroots={self.nroots}, davidson_block="
                    f"{self.davidson_block}). The solver will collapse to "
                    f"{self.restart_keep} of ~{_nb_eff} subspace vectors at each "
                    f"restart and rebuild the rest, costing extra matvecs per "
                    f"cycle. Pass restart_keep=None for the default, or a value "
                    f">= {_default_keep}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.carryover_type = int(carryover_type)
        self.carryover_root = int(carryover_root)
        self.carryover_options = dict(carryover_options or {})
        self.method = int(method)
        self.b_comm_size = int(b_comm_size)
        self.t_comm_size = int(t_comm_size)
        self.bit_length = int(bit_length)
        self.do_redist_det = bool(do_redist_det)
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.clean_temp_dir = bool(clean_temp_dir)
        self.timeout_s = timeout_s
        self.extra_cli_args = list(extra_cli_args or [])
        self.canonicalize_fcidump = bool(canonicalize_fcidump)
        self.rdm = bool(rdm)
        self.rdm_root = None if rdm_root is None else int(rdm_root)

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def sci_solver_callable(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        spin_sq: float | None = None,
        max_cycle: int | None = None,
    ) -> List[experimental_SCIResult]:
        """
        Diagonalize each batch via a fresh GDB sbd subprocess.

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` paired int64 tuples.
                **Paired** means ``len(strs_a) == len(strs_b)`` and determinant
                ``i`` is ``(strs_a[i], strs_b[i])``.  No cartesian product.
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron integrals; any shape accepted by
                ``pyscf.tools.fcidump.from_integrals``.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``.
            single_spin (solver ctor): target spin multiplicity 2S+1 (1=singlet,2=doublet,3=triplet,...); -1=off. spin_sq arg here must stay None.
            max_cycle: Accepted for interface compatibility but ignored; the
                GDB binary has no ``--max_cycle`` flag.

        Returns:
            One entry per batch. With ``nroots == 1`` (default) each entry is a
            single :class:`experimental_SCIResult` (unchanged contract). With
            ``nroots > 1`` each entry is a **list** of ``nroots``
            ``experimental_SCIResult`` (one per converged root, index-ordered by
            ascending energy), each carrying that root's energy, ``spin_square``
            (from ``1pRDM.<p>.txt`` when ``rdm=True``), and ``amplitudes`` (from
            the per-root ``wf_root<p>`` wavefunction shards).
            ``energy`` is the electronic energy (ECORE=0).
            ``sci_state.amplitudes`` is a 1-D array of length M, aligned to
            the input ``(strs_a, strs_b)`` order.
        """
        if spin_sq is not None:
            raise ValueError(
                "SBDGdbSolver does not support spin projection; spin_sq must be None."
            )
        results: List[experimental_SCIResult] = []
        for ci_strings in ci_strs_batches:
            results.append(
                self._solve_one_batch(
                    ci_strings=ci_strings,
                    hcore=hcore,
                    eri=eri,
                    norb=int(norb),
                    nelec=(int(nelec[0]), int(nelec[1])),
                )
            )
        return results

    def solve_bitstring_batches(
        self,
        bitstring_matrix_batches: List[np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
    ) -> List[experimental_SCIResult]:
        """
        Convenience wrapper: convert bitstring matrices to paired CI strings.

        Uses :func:`experimental_bitstring_matrix_to_ci_strs`, which splits
        each row in half (left half = beta, right half = alpha) without
        sorting, deduplicating, or merging spins — preserving non-cartesian
        structure.

        Args:
            bitstring_matrix_batches: List of 2-D bool arrays; each row is a
                bitstring of length ``2*norb``.
        """
        ci_strs_batches = [
            experimental_bitstring_matrix_to_ci_strs(bm)
            for bm in bitstring_matrix_batches
        ]
        return self.sci_solver_callable(ci_strs_batches, hcore, eri, norb, nelec)

    def __repr__(self) -> str:
        return (
            f"SBDGdbSolver(sbd_binary={self._sbd_binary!r}, "
            f"mpi_np={self.mpi_np}, b_comm_size={self.b_comm_size}, "
            f"t_comm_size={self.t_comm_size})"
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _solve_one_batch(
        self,
        *,
        ci_strings: Tuple[np.ndarray, np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
    ) -> experimental_SCIResult:
        strs_a = np.asarray(ci_strings[0], dtype=np.int64)
        strs_b = np.asarray(ci_strings[1], dtype=np.int64)
        if len(strs_a) != len(strs_b):
            raise ValueError(
                f"ci_strings must be paired (len(strs_a)={len(strs_a)} != "
                f"len(strs_b)={len(strs_b)}).  SBDGdbSolver is non-cartesian; "
                "both arrays must have the same length."
            )
        M = len(strs_a)

        succeeded = False
        workdir, _ = self._make_workdir()
        fcidump_path, fcidump_is_temp = self._resolve_fcidump(
            workdir, hcore=hcore, eri=eri, norb=norb,
            nelec_a=nelec[0], nelec_b=nelec[1],
        )

        try:
            det_file = workdir / "dets.txt"
            _write_gdb_dets(det_file, strs_a, strs_b, norb)

            cmd = self._build_cmd(fcidump_path, det_file)

            if self.verbose:
                print(f"  Work dir: {workdir}")
                print(f"  Command:  {' '.join(cmd)}")

            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )

            try:
                (workdir / "sbd_stdout.txt").write_text(proc.stdout or "")
                (workdir / "sbd_stderr.txt").write_text(proc.stderr or "")
                (workdir / "sbd_command.txt").write_text(" ".join(cmd) + "\n")
            except Exception as exc:
                warnings.warn(
                    f"Failed to persist sbd stdout/stderr to {workdir}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )

            if proc.returncode != 0:
                tail = "\n".join(proc.stdout.splitlines()[-40:])
                raise RuntimeError(
                    f"sbd GDB invocation failed (rc={proc.returncode}).\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Work dir: {workdir}\n"
                    f"--- stderr ---\n{proc.stderr}\n"
                    f"--- stdout tail ---\n{tail}"
                )

            def _spin_from_rdm(one_name: str, two_name: str):
                """Compute <S^2> from a pair of RDM files in workdir, or None."""
                if not self.rdm:
                    return None
                rdm1_file = workdir / one_name
                rdm2_file = workdir / two_name
                # With rdm_root set, only that root's RDM files exist. Absent
                # files are then expected, not a failure -- return None quietly
                # rather than warning once per skipped root.
                if self.rdm_root is not None and not rdm1_file.exists():
                    return None
                try:
                    rdm1 = np.zeros((norb, norb), dtype=np.float64)
                    with open(rdm1_file, "r") as f:
                        for line in f:
                            i, j, value = map(float, line.split())
                            rdm1[int(i), int(j)] = value
                    rdm2 = np.zeros((norb, norb, norb, norb), dtype=np.float64)
                    with open(rdm2_file, "r") as f:
                        for line in f:
                            i, j, k, l, value = map(float, line.split())
                            rdm2[int(i), int(j), int(k), int(l)] = value
                    return compute_spin_square(rdm1, rdm2)
                except Exception:
                    warnings.warn(
                        f"Failed to parse RDM files {rdm1_file}/{rdm2_file} for "
                        "spin square; setting spin_square=None.",
                        UserWarning, stacklevel=2,
                    )
                    return None

            def _make_result(energy, savename, one_rdm, two_rdm):
                amplitudes = self._recover_amplitudes(
                    workdir, strs_a, strs_b, norb, M, savename=savename
                )
                occ_a, occ_b = _compute_occupancies(strs_a, strs_b, amplitudes, norb)
                _check_occupancy_consistency(occ_a, occ_b, proc.stdout, norb)
                return experimental_SCIResult(
                    energy=energy,
                    sci_state=experimental_SCIState(
                        amplitudes=amplitudes,
                        ci_strs_a=strs_a, ci_strs_b=strs_b,
                        norb=norb, nelec=nelec,
                    ),
                    orbital_occupancies=(occ_a, occ_b),
                    spin_square=_spin_from_rdm(one_rdm, two_rdm),
                )

            succeeded = True
            if self.nroots > 1:
                # Multi-root / single-spin: one result per root, each with its own
                # energy (stdout), spin (1pRDM.<p>.txt), and amplitudes (savename_root<p>).
                energies = _parse_gdb_multiroot_energies(proc.stdout)
                results = []
                for p in range(self.nroots):
                    e_p = energies[p] if p < len(energies) else float("nan")
                    results.append(_make_result(
                        e_p, f"wf_root{p}", f"1pRDM.{p}.txt", f"2pRDM.{p}.txt"
                    ))
                return results
            else:
                # single root (root 0): unchanged contract
                return _make_result(
                    _parse_gdb_energy(proc.stdout), "wf", "1pRDM.txt", "2pRDM.txt"
                )

        finally:
            should_clean = succeeded and self.clean_temp_dir
            try:
                if fcidump_is_temp and should_clean:
                    Path(fcidump_path).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                if should_clean:
                    shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass
            if not succeeded:
                print(f"[SBDGdbSolver] Preserving workdir for inspection: {workdir}")

    @staticmethod
    def parse_per_root_spin(workdir, norb: int, nroots: int) -> list[dict]:
        """Read per-root RDM files ``1pRDM.<p>.txt`` / ``2pRDM.<p>.txt`` written by
        the multi-root solver and return one dict per root with its ``<S^2>`` and
        the implied total spin ``S`` (from S(S+1)).

        Returns a list of ``{"root": p, "spin_square": <S^2>, "S": S}``.
        Use to filter roots to a target spin sector.
        """
        workdir = Path(workdir)
        out = []
        for p in range(nroots):
            f1 = workdir / f"1pRDM.{p}.txt"
            f2 = workdir / f"2pRDM.{p}.txt"
            if not (f1.exists() and f2.exists()):
                out.append({"root": p, "spin_square": None, "S": None})
                continue
            rdm1 = np.zeros((norb, norb), dtype=np.float64)
            for line in open(f1):
                i, j, v = line.split()
                rdm1[int(i), int(j)] = float(v)
            rdm2 = np.zeros((norb, norb, norb, norb), dtype=np.float64)
            for line in open(f2):
                i, j, k, l, v = line.split()
                rdm2[int(i), int(j), int(k), int(l)] = float(v)
            s2 = compute_spin_square(rdm1, rdm2)
            S = (-1.0 + np.sqrt(1.0 + 4.0 * max(s2, 0.0))) / 2.0
            out.append({"root": p, "spin_square": float(s2), "S": float(S)})
        return out

    def _make_workdir(self) -> tuple[Path, bool]:
        parent = self.temp_dir if self.temp_dir else Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="sbd_gdb_solver_", dir=str(parent))), True

    def _resolve_fcidump(
        self,
        workdir: Path,
        *,
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec_a: int,
        nelec_b: int,
    ) -> tuple[Path, bool]:
        if self._fcidump_path is None or self.canonicalize_fcidump:
            out = workdir / "FCIDUMP.generated"
            if self._fcidump_path is None:
                _write_fcidump_from_integrals(
                    out, hcore=hcore, eri=eri, norb=norb,
                    nelec_a=nelec_a, nelec_b=nelec_b,
                )
            else:
                _roundtrip_fcidump_via_pyscf(Path(self._fcidump_path), out)
            return out, True
        return Path(self._fcidump_path), False

    def _build_cmd(self, fcidump_path: Path, det_file: Path) -> list[str]:
        return [
            self.mpirun, "-np", str(self.mpi_np),
            *self.mpirun_args,
            self._sbd_binary,
            "--fcidump", str(fcidump_path),
            "--detfiles", det_file.name,  # relative to cwd=workdir
            "--method", str(self.method),
            "--block", str(self.davidson_block),
            "--iteration", str(self.davidson_iterations),
            "--tolerance", f"{self.davidson_tolerance:.6e}",
            "--nroots", str(self.nroots),
            *(["--single_spin", str(self.single_spin)] if self.single_spin >= 0 else []),
            *(["--restart_keep", str(self.restart_keep)] if self.restart_keep is not None else []),
            *(["--carryover_type", str(self.carryover_type),
               "--carryover_root", str(self.carryover_root)]
              if self.carryover_type > 0 else []),
            *([tok for k, v in self.carryover_options.items()
               for tok in (f"--{k}", str(v))] if self.carryover_type > 0 else []),
            "--b_comm_size", str(self.b_comm_size),
            "--t_comm_size", str(self.t_comm_size),
            "--bit_length", str(self.bit_length),
            "--do_redist_det", "1" if self.do_redist_det else "0",
            "--rdm", "1" if self.rdm else "0",
            *(["--rdm_root", str(self.rdm_root)]
              if (self.rdm and self.rdm_root is not None) else []),
            "--savename", "wf",
            *self.extra_cli_args,
        ]

    def _recover_amplitudes(
        self,
        workdir: Path,
        strs_a: np.ndarray,
        strs_b: np.ndarray,
        norb: int,
        M: int,
        savename: str = "wf",
    ) -> np.ndarray:
        amp_map = _read_gdb_wavefunction(
            workdir, savename, self.b_comm_size, norb, self.bit_length
        )
        if amp_map is None:
            warnings.warn(
                f"Failed to read GDB wavefunction shards from {workdir}/{savename}*.bin. "
                "Returning a stub amplitudes array (1.0 at index 0). Downstream "
                "carryover selection will behave as if the state is a single determinant.",
                UserWarning,
                stacklevel=3,
            )
            return _stub_amplitudes_1d(M)

        amplitudes = np.zeros(M, dtype=np.float64)
        for i, (a, b) in enumerate(zip(strs_a.tolist(), strs_b.tolist())):
            amplitudes[i] = amp_map.get((int(a), int(b)), 0.0)

        norm = float(np.linalg.norm(amplitudes))
        if norm > 0.0:
            amplitudes /= norm
        else:
            warnings.warn(
                "All recovered amplitudes are zero after matching against input "
                "determinant pairs.  Returning stub.  Check that norb and "
                "bit_length match the binary.",
                UserWarning,
                stacklevel=3,
            )
            return _stub_amplitudes_1d(M)
        return amplitudes


# ---------------------------------------------------------------------------
# Determinant I/O
# ---------------------------------------------------------------------------


def _interleave_det(alpha_int: int, beta_int: int, norb: int) -> int:
    """
    Interleave ``(alpha_int, beta_int)`` into a single ``2·norb``-bit integer.

    Bit layout: ``bit 2·io`` = alpha orbital ``io``, ``bit 2·io+1`` = beta.
    Uses Python big-int arithmetic — safe for ``norb > 31``.
    """
    full = 0
    for io in range(norb):
        if (alpha_int >> io) & 1:
            full |= 1 << (2 * io)
        if (beta_int >> io) & 1:
            full |= 1 << (2 * io + 1)
    return full


def _deinterleave_det(full_int: int, norb: int) -> tuple[int, int]:
    """Recover ``(alpha_int, beta_int)`` from an interleaved full-determinant int."""
    alpha = 0
    beta = 0
    for io in range(norb):
        if (full_int >> (2 * io)) & 1:
            alpha |= 1 << io
        if (full_int >> (2 * io + 1)) & 1:
            beta |= 1 << io
    return alpha, beta


def _write_gdb_dets(
    path: Path, strs_a: np.ndarray, strs_b: np.ndarray, norb: int
) -> None:
    """
    Write a GDB-format determinant file (one bitstring per line).

    Each line is a ``2·norb``-character string, MSB-first, matching
    ``sbd::makestring``: character 0 = bit ``2·norb-1``, last character = bit 0.
    """
    total_bits = 2 * norb
    lines = []
    for a, b in zip(strs_a.tolist(), strs_b.tolist()):
        full = _interleave_det(int(a), int(b), norb)
        if full.bit_length() > total_bits:
            raise ValueError(
                f"Determinant has a bit set above position {total_bits - 1}; "
                f"norb={norb} is too small."
            )
        lines.append(format(full, "b").zfill(total_bits))
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Wavefunction binary reader
# ---------------------------------------------------------------------------


def _deinterleave_det_from_words(
    words: np.ndarray, norb: int, bit_length: int
) -> tuple[int, int]:
    """
    Decode a determinant stored as an array of ``basis_length`` uint64 words.

    The binary packs ``bit_length`` bits per word.  Bit ``k`` of the full
    determinant lives in ``words[k // bit_length]`` at position
    ``k % bit_length``.
    """
    alpha = 0
    beta = 0
    for io in range(norb):
        k_a = 2 * io
        k_b = 2 * io + 1
        if int(words[k_a // bit_length]) & (1 << (k_a % bit_length)):
            alpha |= 1 << io
        if int(words[k_b // bit_length]) & (1 << (k_b % bit_length)):
            beta |= 1 << io
    return alpha, beta


def _read_gdb_wavefunction(
    workdir: Path,
    savename: str,
    b_comm_size: int,
    norb: int,
    bit_length: int,
) -> dict[tuple[int, int], float] | None:
    """
    Read all GDB wavefunction shards; return ``{(alpha_int, beta_int): amplitude}``.

    Shard files are ``{savename}{rank:06d}.bin`` for ``rank`` in
    ``0..b_comm_size-1`` (format from ``sbd/caop/basic/restart.h``).

    Binary format per file (sizes in bytes):
      uint64 basis_size         (8)
      uint64 basis_length       (8)   — words per determinant
      basis_size × basis_length × uint64   (determinants)
      basis_size × float64               (amplitudes)

    Returns ``None`` if any shard is missing or unreadable.
    """
    sz = 8  # sizeof(size_t) = sizeof(double) = 8 bytes on 64-bit platforms
    amp_map: dict[tuple[int, int], float] = {}

    for rank in range(b_comm_size):
        path = workdir / f"{savename}{rank:06d}.bin"
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                header = np.frombuffer(fh.read(2 * sz), dtype=np.uint64)
                if header.size != 2:
                    return None
                basis_size = int(header[0])
                basis_length = int(header[1])

                det_bytes = basis_size * basis_length * sz
                det_buf = fh.read(det_bytes)
                if len(det_buf) != det_bytes:
                    return None
                det_words = np.frombuffer(det_buf, dtype=np.uint64).reshape(
                    basis_size, basis_length
                )

                amp_bytes = basis_size * sz
                amp_buf = fh.read(amp_bytes)
                if len(amp_buf) != amp_bytes:
                    return None
                amps = np.frombuffer(amp_buf, dtype=np.float64)
        except (OSError, ValueError):
            return None

        for i in range(basis_size):
            alpha, beta = _deinterleave_det_from_words(det_words[i], norb, bit_length)
            amp_map[(alpha, beta)] = float(amps[i])

    return amp_map


# ---------------------------------------------------------------------------
# Occupancy and parsing helpers
# ---------------------------------------------------------------------------


def _compute_occupancies(
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    amplitudes: np.ndarray,
    norb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-orbital occupancies from the 1-D amplitude vector.

    ``occ_a[io] = Σ_i |amp_i|² · bit(strs_a[i], io)``
    ``occ_b[io] = Σ_i |amp_i|² · bit(strs_b[i], io)``
    """
    weights = amplitudes.astype(np.float64) ** 2
    occ_a = np.zeros(norb, dtype=np.float64)
    occ_b = np.zeros(norb, dtype=np.float64)
    for io in range(norb):
        mask_a = ((strs_a >> io) & np.int64(1)).astype(np.float64)
        mask_b = ((strs_b >> io) & np.int64(1)).astype(np.float64)
        occ_a[io] = float(np.dot(weights, mask_a))
        occ_b[io] = float(np.dot(weights, mask_b))
    return occ_a, occ_b


def _check_occupancy_consistency(
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    stdout: str,
    norb: int,
) -> None:
    density = _parse_gdb_density(stdout)
    if density is None or density.shape[0] != norb:
        return
    ref = float(np.sum(density))
    python_sum = float(np.sum(occ_a + occ_b))
    if ref > 0.0 and abs(python_sum - ref) / abs(ref) > 0.01:
        warnings.warn(
            f"Occupancy cross-check: sbd stdout spin-summed sum={ref:.6f} but "
            f"Python-computed sum={python_sum:.6f} (>1% relative deviation). "
            f"norb={norb}.",
            UserWarning,
            stacklevel=4,
        )


_GDB_ENERGY_RE = re.compile(
    r"sbd:\s+Energy\s*=\s*(?P<e>-?\d+\.\d+(?:[eE][+-]?\d+)?)"
)
# per-root energies for multi-root / single-spin runs:
#   "sbd: MultiRoot E[<p>] = <e>"  or  "sbd: SingleSpin E[<p>] = <e>"
_GDB_MULTIROOT_ENERGY_RE = re.compile(
    r"sbd:\s+(?:MultiRoot|SingleSpin)\s+E\[(?P<p>\d+)\]\s*=\s*"
    r"(?P<e>-?\d+\.\d+(?:[eE][+-]?\d+)?)"
)


def _parse_gdb_multiroot_energies(stdout: str) -> list[float]:
    """Return per-root energies (index-ordered) from MultiRoot/SingleSpin lines."""
    hits = {}
    for m in _GDB_MULTIROOT_ENERGY_RE.finditer(stdout):
        hits[int(m.group("p"))] = float(m.group("e"))
    return [hits[p] for p in sorted(hits)]
_GDB_DENSITY_RE = re.compile(
    r"sbd:\s+density\s*=\s*\[(?P<d>[^\n]*)"
)


def _parse_gdb_energy(stdout: str) -> float:
    m = _GDB_ENERGY_RE.search(stdout)
    if m:
        return float(m.group("e"))
    tail = "\n".join(stdout.splitlines()[-40:])
    raise RuntimeError(
        "Could not find 'sbd: Energy = ...' in GDB stdout.\n"
        f"Tail of output:\n{tail}"
    )


def _parse_gdb_density(stdout: str) -> np.ndarray | None:
    m = _GDB_DENSITY_RE.search(stdout)
    if not m:
        return None
    raw = m.group("d").rstrip().rstrip("]").strip()
    if not raw:
        return None
    try:
        return np.array([float(x) for x in raw.split(",") if x.strip()])
    except ValueError:
        return None


def _stub_amplitudes_1d(M: int) -> np.ndarray:
    a = np.zeros(M, dtype=np.float64)
    if M > 0:
        a[0] = 1.0
    return a


# ---------------------------------------------------------------------------
# FCIDUMP helpers (mirror sbd_solver.py; ECORE always 0)
# ---------------------------------------------------------------------------


def _write_fcidump_from_integrals(
    path: Path,
    *,
    hcore: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec_a: int,
    nelec_b: int,
) -> None:
    _pyscf_fcidump.from_integrals(
        str(path),
        h1e=hcore,
        h2e=eri,
        nmo=norb,
        nelec=(nelec_a, nelec_b),
        nuc=0.0,
        ms=nelec_a - nelec_b,
        orbsym=None,
        tol=1e-15,
    )


def _roundtrip_fcidump_via_pyscf(src: Path, dst: Path) -> None:
    data = _pyscf_fcidump.read(str(src))
    _pyscf_fcidump.from_integrals(
        str(dst),
        h1e=data["H1"],
        h2e=data["H2"],
        nmo=int(data["NORB"]),
        nelec=int(data["NELEC"]),
        nuc=0.0,
        ms=int(data.get("MS2", 0)),
        orbsym=data.get("ORBSYM"),
        tol=1e-15,
    )


# ---------------------------------------------------------------------------
# Spin helpers
# ---------------------------------------------------------------------------


def compute_spin_square(rdm1: np.ndarray, rdm2: np.ndarray) -> float:
    """
    Compute spin square <S²> from 1-RDM and 2-RDM.
    
    Formula: S² = 0.75 * Tr(rdm1) - 0.5 * Σᵢⱼ rdm2[i,j,j,i] - 0.25 * Σᵢⱼ rdm2[i,j,i,j]
    
    The RDM files contain spin-summed density matrices.
    rdm1[i,j] is indexed as (i, j)
    rdm2[io,jo,ia,ja] is indexed as (io, jo, ia, ja)
    
    Parameters:
        rdm1: 1-particle reduced density matrix (norb, norb)
        rdm2: 2-particle reduced density matrix (norb, norb, norb, norb)
    
    Returns:
        spin_square: <S²> value
    
    Note: For singlet states, S² should be exactly 0. Small deviations (~1e-4)
    are due to numerical precision in the RDM calculation.
    """
    norb = rdm1.shape[0]
    
    # First term: 0.75 * Tr(rdm1) = 0.75 * sum_i rdm1[i,i]
    term1 = 0.75 * np.trace(rdm1)
    
    # Second term: -0.5 * sum_{i,j} rdm2[i,j,j,i]
    # Use symmetry: rdm2[i,j,j,i] ≈ rdm2[j,i,i,j] for better numerical stability
    term2 = 0.0
    for i in range(norb):
        for j in range(norb):
            # Average symmetric terms for better numerical stability
            term2 += 0.5 * (rdm2[i, j, j, i] + rdm2[j, i, i, j])
    term2 *= -0.5
    
    # Third term: -0.25 * sum_{i,j} rdm2[i,j,i,j]
    # Use symmetry: rdm2[i,j,i,j] ≈ rdm2[j,i,j,i] for better numerical stability
    term3 = 0.0
    for i in range(norb):
        for j in range(norb):
            # Average symmetric terms for better numerical stability
            term3 += 0.5 * (rdm2[i, j, i, j] + rdm2[j, i, j, i])
    term3 *= -0.25
    
    spin_square = term1 + term2 + term3
    
    return spin_square

# ---------------------------------------------------------------------------
# Classes to store results without cartesian product
# ---------------------------------------------------------------------------

def experimental_bitstring_matrix_to_ci_strs(
    bitstring_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert bitstrings (rows) in a ``bitstring_matrix`` into integer representations of determinants.

    This function separates each bitstring in ``bitstring_matrix`` in half, translates them into
    integer representations, and finally appends them to their respective (spin-up or spin-down) lists.
    Those lists are not sorted not reduced, to keep the non-cartesian formulation.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.

    Returns:
        A length-2 tuple of determinant lists representing the right (spin-up) and left (spin-down)
        halves of the bitstrings, respectively.

    """
    norb = bitstring_matrix.shape[1] // 2

    ci_strs_left  = bitstring_matrix_to_integers(bitstring_matrix[:, :norb])
    ci_strs_right = bitstring_matrix_to_integers(bitstring_matrix[:, norb:])


    return ci_strs_right, ci_strs_left


@dataclass(frozen=True)
class experimental_SCIState:
    """The amplitudes and determinants describing a quantum state."""

    amplitudes: np.ndarray
    """Either an :math:`M \\times N` array where :math:`M =` len(``ci_strs_a``)
    and :math:`N` = len(``ci_strs_b``). ``amplitudes[i][j]`` is the
    amplitude of the determinant pair (``ci_strs_a[i]``, ``ci_strs_b[j]``).

    Or an array of length :math:`M` where :math:`M =` len(``ci_strs_a``) = len(``ci_strs_a``).
    ``amplitudes[i]`` is the amplitude of the determinant pair (``ci_strs_a[i]``, ``ci_strs_b[i]``).
    """

    ci_strs_a: np.ndarray
    """The alpha determinants."""

    ci_strs_b: np.ndarray
    """The beta determinants."""

    norb: int
    """The number of spatial orbitals."""

    nelec: tuple[int, int]
    """The numbers of alpha and beta electrons."""

    def __post_init__(self):
        """Validate dimensions of inputs."""
        object.__setattr__(
            self, "amplitudes", np.asarray(self.amplitudes)
        )  # Convert to ndarray if not already
        if len(self.amplitudes.shape) == 2 and self.amplitudes.shape != (
            len(self.ci_strs_a),
            len(self.ci_strs_b),
        ):
            raise ValueError(
                f"'amplitudes' shape must be ({len(self.ci_strs_a)}, {len(self.ci_strs_b)}) "
                f"but got {self.amplitudes.shape}"
            )

        if len(self.amplitudes.shape) == 1 and len(self.amplitudes) != len(
            self.ci_strs_a
        ):
            raise ValueError(
                f"'amplitudes' length must be ({len(self.ci_strs_a)}, ) in non Cartesian product mode"
                f"but got {len(self.amplitudes)}"
            )

        if len(self.amplitudes.shape) == 1 and len(self.ci_strs_b) != len(
            self.ci_strs_a
        ):
            raise ValueError(
                f"In non Cartesian product mode len(ci_strs_a) must be equal to len(ci_strs_b) "
                f"but got {len(self.ci_strs_a)} and {len(self.ci_strs_b)}"
            )

    @property
    def subspace_dimension(self):
        """Returns the size of the subspace where the state is supported."""
        return self.amplitudes.size

    @property
    def cartesian_product_structure(self):
        """Whether the state is specified by the cartesian product of two lists of alpha and beta determinants."""
        return len(self.amplitudes.shape) == 2

    def save(self, filename):
        """Save the SCIState object to an .npz file."""
        np.savez(
            filename,
            amplitudes=self.amplitudes,
            ci_strs_a=self.ci_strs_a,
            ci_strs_b=self.ci_strs_b,
            norb=self.norb,
            nelec=self.nelec,
        )

    @classmethod
    def load(cls, filename):
        """Load an SCIState object from an .npz file."""
        with np.load(filename) as data:
            return cls(
                data["amplitudes"],
                data["ci_strs_a"],
                data["ci_strs_b"],
                norb=data["norb"],
                nelec=tuple(data["nelec"]),
            )

    def orbital_occupancies(self) -> tuple[np.ndarray, np.ndarray]:
        """Average orbital occupancies."""
        raise NotImplementedError()

    def rdm(self, rank: int = 1, spin_summed: bool = False) -> np.ndarray:
        """Compute reduced density matrix."""
        # Reason for type: ignore: mypy can't tell the return type of the
        # PySCF functions
        raise NotImplementedError()

    def spin_square(self) -> float:
        """Return spin squared."""
        raise NotImplementedError()


@dataclass(frozen=True)
class experimental_SCIResult:
    """Result of an SCI calculation."""

    energy: float
    """The SCI energy."""

    sci_state: experimental_SCIState
    """The SCI state."""

    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    """The average orbital occupancies."""

    rdm1: np.ndarray | None = None
    """Spin-summed 1-particle reduced density matrix."""

    rdm2: np.ndarray | None = None
    """Spin-summed 2-particle reduced density matrix."""

    spin_square: float | None = None
    """The spin square of the state."""