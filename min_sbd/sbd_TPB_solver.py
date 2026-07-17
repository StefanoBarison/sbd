"""
SBD (RIKEN Sample-Based Davidson) sci_solver implementation.

Standalone module for the ``sqdrift`` pipeline.
Wraps the ``chemistry_tpb_selected_basis_diagonalization`` app (TPB
= Tensor-Product-Basis, i.e. cartesian alpha⊗beta subspace).

Prerequisites
-------------
- A compiled ``diag`` binary from
  ``sbd-main/apps/chemistry_tpb_selected_basis_diagonalization/``.
  For GPU, enable ``-DSBD_THRUST`` and build with nvc++.
  For authoritative per-spin orbital occupancies on disk, enable ``-DSBD_PREFECT``.
- ``mpirun`` reachable via PATH, or passed explicitly via ``mpirun=...``.
- The ``pyscf`` package for FCIDUMP generation.

Spin handling
-------------
Vanilla sbd has NO spin-projection mechanism.  ``spin_sq`` must be ``None``.

Alpha/beta pool merging: if ``strs_a == strs_b`` (closed-shell case from
``bitstring_matrix_to_ci_strs(open_shell=False)``), only ``AlphaDets.txt`` is
written; sbd reuses it as the beta pool.

Returned energy
---------------
Each ``SCIResult.energy`` is the ELECTRONIC energy only (FCIDUMP with ECORE=0).
Callers add ``e_nuc`` themselves.

Debugging
---------
stdout, stderr, and the command line are written to
``sbd_stdout.txt`` / ``sbd_stderr.txt`` / ``sbd_command.txt`` inside each
batch's workdir.  On failure the workdir is preserved regardless of
``clean_temp_dir``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pyscf.tools import fcidump
from qiskit_addon_sqd.fermion import SCIResult, SCIState


class SBDTpbSolver:
    """
    GPU-capable sci_solver wrapping RIKEN's TPB ``sbd`` binary.

    Drop-in replacement for qiskit-addon-sqd's ``solve_sci_batch``:
    ``solver.sci_solver_callable`` accepts the same positional call
    ``(ci_strings, one_body_tensor, two_body_tensor, norb, nelec)``.

    See module docstring for prerequisites, spin handling, FCIDUMP
    conventions, and debugging.
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
        method: int = 0,
        task_comm_size: int = 1,
        adet_comm_size: int = 1,
        bdet_comm_size: int = 1,
        # Build-flag dependent
        have_sbd_prefect: bool = False,
        # Workdir / bookkeeping
        temp_dir: str | Path | None = None,
        clean_temp_dir: bool = True,
        timeout_s: float | None = None,
        extra_cli_args: list[str] | None = None,
        # FCIDUMP handling
        canonicalize_fcidump: bool = True,
        #RDM handling
        rdm: bool = False,
    ):
        self.verbose = bool(verbose)

        if sbd_binary is None:
            repo_root = Path(__file__).resolve().parents[1]
            sbd_binary = (
                repo_root
                / "build"
                / "energy_solvers"
                / "sbd_gpu_workdir"
                / "sbd"
                / "apps"
                / "chemistry_tpb_selected_basis_diagonalization"
                / "diag"
            )

        sbd_binary_path = Path(sbd_binary).resolve()
        if not sbd_binary_path.is_file() or not os.access(sbd_binary_path, os.X_OK):
            raise FileNotFoundError(
                f"sbd_binary not found or not executable: {sbd_binary_path}\n"
                "Build it with: cd sbd-main/apps/chemistry_tpb_selected_basis_diagonalization && make"
            )
        self._sbd_binary = str(sbd_binary_path)

        if fcidump_path is not None:
            fp = Path(fcidump_path).resolve()
            if not fp.is_file():
                raise FileNotFoundError(f"fcidump_path not found: {fp}")
            self._fcidump_path: str | None = str(fp)
        else:
            self._fcidump_path = None

        self.mpirun = mpirun
        self.mpi_np = int(mpi_np)
        self.mpirun_args = (
            list(mpirun_args)
            if mpirun_args is not None
            else ["-x", "OMP_NUM_THREADS=1"]
        )

        self.davidson_block = int(davidson_block)
        self.davidson_iterations = int(davidson_iterations)
        self.davidson_tolerance = float(davidson_tolerance)
        self.method = int(method)
        self.task_comm_size = int(task_comm_size)
        self.adet_comm_size = int(adet_comm_size)
        self.bdet_comm_size = int(bdet_comm_size)
        self.have_sbd_prefect = bool(have_sbd_prefect)

        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.clean_temp_dir = bool(clean_temp_dir)
        self.timeout_s = timeout_s
        self.extra_cli_args = list(extra_cli_args or [])
        self.canonicalize_fcidump = bool(canonicalize_fcidump)
        self.rdm = bool(rdm)

    def sci_solver_callable(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        spin_sq: float | None = None,
        max_cycle: int | None = None,
    ) -> List[SCIResult]:
        """
        Diagonalize each batch via a fresh sbd subprocess.

        Signature matches ``qiskit_addon_sqd.fermion.solve_sci_batch``.

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` int64-array tuples.
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron integrals (full 4-D or any PySCF packed form).
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``.
            spin_sq: Must be None; sbd has no spin-projection mechanism.
            max_cycle: Optional ``--max_cycle`` override.

        Returns:
            One :class:`SCIResult` per batch. Energies are electronic only.
        """
        assert (
            spin_sq is None
        ), "SBDSolver does not support spin projection; spin_sq must be None"

        results: List[SCIResult] = []
        for ci_strings in ci_strs_batches:
            results.append(
                self._solve_one_batch(
                    ci_strings=ci_strings,
                    hcore=hcore,
                    eri=eri,
                    norb=int(norb),
                    nelec=(int(nelec[0]), int(nelec[1])),
                    max_cycle=max_cycle,
                )
            )
        return results

    def _solve_one_batch(
        self,
        *,
        ci_strings: Tuple[np.ndarray, np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        max_cycle: int | None = None,
    ) -> SCIResult:

        strs_a = np.asarray(ci_strings[0], dtype=np.int64)
        strs_b = np.asarray(ci_strings[1], dtype=np.int64)
        n_a = len(strs_a)
        n_b = len(strs_b)

        succeeded = False
        workdir, workdir_is_temp = self._make_workdir()
        fcidump_path, fcidump_is_temp = self._resolve_fcidump(
            workdir,
            hcore=hcore,
            eri=eri,
            norb=norb,
            nelec_a=nelec[0],
            nelec_b=nelec[1],
        )

        try:
            same_pool = np.array_equal(strs_a, strs_b)
            strs_a = _reorder_for_connected_seed(strs_a)
            strs_b = strs_a if same_pool else _reorder_for_connected_seed(strs_b)

            adet_file = workdir / "AlphaDets.txt"
            _write_dets(adet_file, strs_a, norb)

            if not same_pool:
                bdet_file: Path | None = workdir / "BetaDets.txt"
                _write_dets(bdet_file, strs_b, norb)
            else:
                bdet_file = None

            cmd = self._build_cmd(
                fcidump_path, adet_file, bdet_file, max_cycle=max_cycle
            )

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
            except Exception as e:
                warnings.warn(
                    f"Failed to persist sbd stdout/stderr to {workdir}: {e}",
                    UserWarning,
                    stacklevel=2,
                )

            if proc.returncode != 0:
                tail = "\n".join(proc.stdout.splitlines()[-40:])
                raise RuntimeError(
                    f"sbd invocation failed (rc={proc.returncode}).\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Work dir: {workdir}\n"
                    f"--- stderr ---\n{proc.stderr}\n"
                    f"--- stdout tail ---\n{tail}"
                )

            electronic_energy = _parse_sbd_energy(proc.stdout)
            occ_a, occ_b = self._parse_occupancies(workdir, proc.stdout, norb, nelec)
            amplitudes   = self._parse_amplitudes(workdir, n_a, n_b)

            succeeded = True

            sci_final = SCIState(
                    amplitudes=amplitudes,
                    ci_strs_a=np.asarray(ci_strings[0]),
                    ci_strs_b=np.asarray(ci_strings[1]),
                    norb=norb,
                    nelec=nelec,
                )

            return SCIResult(
                energy=electronic_energy,
                sci_state=sci_final,
                #spin_square = sci_final.spin_square(), #The structure of SCIState let us commpute spin square on the fly using pyscf backbone  
                orbital_occupancies=(occ_a, occ_b),
            )

        finally:
            should_clean = succeeded and self.clean_temp_dir
            try:
                if fcidump_is_temp and should_clean:
                    Path(fcidump_path).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                if workdir_is_temp and should_clean:
                    shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass
            if not succeeded and workdir_is_temp:
                print(f"[SBDSolver] Preserving workdir for inspection: {workdir}")

    def _make_workdir(self) -> tuple[Path, bool]:
        parent = self.temp_dir if self.temp_dir else Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="sbd_solver_", dir=str(parent))), True

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
                    out,
                    hcore=hcore,
                    eri=eri,
                    norb=norb,
                    nelec_a=nelec_a,
                    nelec_b=nelec_b,
                )
            else:
                _roundtrip_fcidump_via_pyscf(Path(self._fcidump_path), out)
            return out, True
        return Path(self._fcidump_path), False

    def _build_cmd(
        self,
        fcidump_path: Path,
        adet_file: Path,
        bdet_file: Path | None,
        max_cycle: int | None = None,
    ) -> list[str]:
        cmd = [
            self.mpirun,
            "-np",
            str(self.mpi_np),
            *self.mpirun_args,
            self._sbd_binary,
            "--fcidump",
            str(fcidump_path),
            "--adetfile",
            adet_file.name,
        ]
        if bdet_file is not None:
            cmd.extend(["--bdetfile", bdet_file.name])
        cmd.extend(
            [
                "--method",
                str(self.method),
                "--block",
                str(self.davidson_block),
                "--iteration",
                str(self.davidson_iterations),
                "--tolerance",
                f"{self.davidson_tolerance:.6e}",
                "--task_comm_size",
                str(self.task_comm_size),
                "--adet_comm_size",
                str(self.adet_comm_size),
                "--bdet_comm_size",
                str(self.bdet_comm_size),
                "--init",
                "0",
                "--shuffle",
                "0",
                "--carryover_type",
                "0",
                "--rdm",
                "1" if self.rdm else "0",
                "--savename",
                "wf",
            ]
        )
        if max_cycle is not None:
            cmd.extend(["--max_cycle", str(max_cycle)])
        cmd.extend(self.extra_cli_args)
        return cmd

    def _parse_occupancies(
        self, workdir: Path, stdout: str, norb: int, nelec: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.have_sbd_prefect:
            pair = _read_occ_files(workdir, norb)
            if pair is not None:
                return pair
            warnings.warn(
                "have_sbd_prefect=True was set but occ_a.txt / occ_b.txt "
                "were not found. Falling back to stdout density parsing.",
                UserWarning,
                stacklevel=3,
            )

        density = _parse_sbd_density(stdout)
        if density is None:
            tail = "\n".join(stdout.splitlines()[-40:])
            raise RuntimeError(
                "Could not recover orbital occupancies: occ_a.txt / occ_b.txt "
                "not present (build sbd with -DSBD_PREFECT to enable them) "
                f"and no 'density = [...]' line in stdout. Tail:\n{tail}"
            )
        if density.shape != (norb,):
            raise RuntimeError(
                f"sbd density line has {density.shape[0]} entries, expected {norb}."
            )
        if nelec[0] != nelec[1]:
            warnings.warn(
                "Splitting spin-summed density 50/50 into occ_a/occ_b, but "
                "the system is open-shell. The returned per-spin "
                "occupancies are WRONG. Rebuild sbd with -DSBD_PREFECT to "
                "get accurate per-spin data.",
                UserWarning,
                stacklevel=3,
            )
        return density / 2.0, density / 2.0

    def _parse_amplitudes(self, workdir: Path, n_a: int, n_b: int) -> np.ndarray:
        amps = _read_sbd_wavefunction(
            workdir,
            savename="wf",
            n_a=n_a,
            n_b=n_b,
            adet_comm_size=self.adet_comm_size,
            bdet_comm_size=self.bdet_comm_size,
        )
        if amps is not None:
            return amps
        warnings.warn(
            f"Failed to read wavefunction shards from {workdir}/wf*. "
            "Returning a stub amplitudes matrix.",
            UserWarning,
            stacklevel=3,
        )
        return _stub_amplitudes(n_a, n_b)

    def __repr__(self) -> str:
        return f"SBDSolver(sbd_binary={self._sbd_binary!r}, " f"mpi_np={self.mpi_np})"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_ENERGY_PATTERNS = [
    re.compile(r"^Davidson energy:\s+(?P<e>-?\d+\.\d+(?:[eE][+-]?\d+)?)\s*$", re.M),
    re.compile(
        r"^\s*Sample-based diagonalization: Energy\s*=\s*"
        r"(?P<e>-?\d+\.\d+(?:[eE][+-]?\d+)?)\s*$",
        re.M,
    ),
]

_DENSITY_RE = re.compile(
    r"^\s*Sample-based diagonalization: density\s*=\s*\[(?P<d>[^\n]*)",
    re.M,
)


def _parse_sbd_energy(stdout: str) -> float:
    for pat in _ENERGY_PATTERNS:
        m = pat.search(stdout)
        if m:
            return float(m.group("e"))
    tail = "\n".join(stdout.splitlines()[-40:])
    raise RuntimeError(f"Could not find energy in sbd stdout. Tail of output:\n{tail}")


def _parse_sbd_density(stdout: str) -> np.ndarray | None:
    m = _DENSITY_RE.search(stdout)
    if not m:
        return None
    raw = m.group("d").rstrip().rstrip("]").strip()
    if not raw:
        return None
    try:
        return np.array([float(x) for x in raw.split(",") if x.strip()])
    except ValueError as e:
        raise RuntimeError(
            f"Density line found but couldn't be parsed as floats: {e}. Raw: {raw!r}"
        ) from e


def _read_occ_files(workdir: Path, norb: int) -> tuple[np.ndarray, np.ndarray] | None:
    occ_a_path = workdir / "occ_a.txt"
    occ_b_path = workdir / "occ_b.txt"
    if not (occ_a_path.exists() and occ_b_path.exists()):
        return None
    occ_a = np.loadtxt(occ_a_path)
    occ_b = np.loadtxt(occ_b_path)
    if occ_a.shape != (norb,) or occ_b.shape != (norb,):
        raise RuntimeError(
            f"occ_a/occ_b shapes {occ_a.shape}/{occ_b.shape}, expected ({norb},)"
        )
    return occ_a, occ_b


def _write_dets(path: Path, strs: np.ndarray, norb: int) -> None:
    lines = []
    for s in strs:
        s_int = int(s)
        bits = format(s_int, "b")
        if len(bits) > norb:
            raise ValueError(
                f"CI string {s_int} has a bit set above position {norb - 1}; "
                f"norb={norb} is too small."
            )
        lines.append(bits.zfill(norb))
    path.write_text("\n".join(lines) + "\n")


def _reorder_for_connected_seed(strs: np.ndarray) -> np.ndarray:
    """Move the most-connected determinant to index 0 for a better Davidson seed."""
    n = len(strs)
    if n <= 1:
        return strs

    strs_int = strs.astype(np.int64, copy=False)
    n_close = np.zeros(n, dtype=np.int64)
    for i in range(n):
        xor = strs_int ^ strs_int[i]
        d = _popcount64(xor)
        close_mask = (d == 2) | (d == 4)
        n_close[i] = int(np.sum(close_mask))

    best_idx = int(np.argmax(n_close))
    if n_close[best_idx] == 0 or best_idx == 0:
        return strs

    return np.concatenate(
        [
            strs[best_idx : best_idx + 1],
            strs[:best_idx],
            strs[best_idx + 1 :],
        ]
    )


def _popcount64(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.int64, copy=True)
    m1 = np.int64(0x5555555555555555)
    m2 = np.int64(0x3333333333333333)
    m4 = np.int64(0x0F0F0F0F0F0F0F0F)
    h01 = np.int64(0x0101010101010101)
    x = x - ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    return ((x * h01) >> 56) & np.int64(0x7F)


def _stub_amplitudes(n_a: int, n_b: int) -> np.ndarray:
    a = np.zeros((n_a, n_b), dtype=float)
    a[0, 0] = 1.0
    return a


def _read_sbd_wavefunction(
    workdir: Path,
    savename: str,
    n_a: int,
    n_b: int,
    adet_comm_size: int,
    bdet_comm_size: int,
) -> np.ndarray | None:
    sz = np.dtype(np.int64).itemsize
    n_shards = adet_comm_size * bdet_comm_size
    amplitudes = np.zeros((n_a, n_b), dtype=float)
    slabs: dict[tuple[int, int], np.ndarray] = {}
    rows_by_a: dict[int, int] = {}
    cols_by_b: dict[int, int] = {}

    for rank in range(n_shards):
        a_rank = rank // bdet_comm_size
        b_rank = rank % bdet_comm_size
        path = workdir / f"{savename}{rank:06d}"
        if not path.exists():
            return None
        with Path(path).open("rb") as f:
            header = np.frombuffer(f.read(3 * sz), dtype=np.int64)
            if header.size != 3:
                return None
            adet_range = int(header[0])
            bdet_range = int(header[1])
            det_length = int(header[2])
            f.seek((adet_range + bdet_range) * det_length * sz, 1)
            n_amp = adet_range * bdet_range
            buf = f.read(n_amp * sz)
            if len(buf) != n_amp * sz:
                return None
            wave_array = np.frombuffer(buf, dtype=np.float64).reshape(
                adet_range, bdet_range
            )

        if a_rank in rows_by_a and rows_by_a[a_rank] != adet_range:
            return None
        if b_rank in cols_by_b and cols_by_b[b_rank] != bdet_range:
            return None
        rows_by_a[a_rank] = adet_range
        cols_by_b[b_rank] = bdet_range
        slabs[(a_rank, b_rank)] = wave_array

    total_rows = sum(rows_by_a.get(a, 0) for a in range(adet_comm_size))
    total_cols = sum(cols_by_b.get(b, 0) for b in range(bdet_comm_size))
    if total_rows != n_a or total_cols != n_b:
        return None

    row_offsets = [0]
    for a in range(adet_comm_size):
        row_offsets.append(row_offsets[-1] + rows_by_a[a])
    col_offsets = [0]
    for b in range(bdet_comm_size):
        col_offsets.append(col_offsets[-1] + cols_by_b[b])

    for (a_rank, b_rank), wave_array in slabs.items():
        r0, r1 = row_offsets[a_rank], row_offsets[a_rank + 1]
        c0, c1 = col_offsets[b_rank], col_offsets[b_rank + 1]
        amplitudes[r0:r1, c0:c1] = wave_array
    return amplitudes


# ---------------------------------------------------------------------------
# FCIDUMP helpers
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
    fcidump.from_integrals(
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
    data = fcidump.read(str(src))
    fcidump.from_integrals(
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
