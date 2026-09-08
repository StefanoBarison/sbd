"""
Epstein-Nesbet PT2 on top of a finished GDB diagonalization.

WHY THIS IS A SEPARATE WRAPPER AND A SEPARATE PROCESS
-----------------------------------------------------
The variational energy of a selected-CI calculation is an upper bound: it sits above
the true energy by whatever the omitted determinants are worth. For a determinant
space that comes from a quantum circuit that omission is not a choice -- the space is
fixed by sampling -- so the standard estimate of the gap is second-order perturbation
theory, and ``E_var + E_PT2`` is the number selected-CI codes report.

PT2 runs as a SECOND ``mpirun``, after the solve has finished and written its
wavefunction:

    solver = SBDGdbSolver(...); res = solver.solve_bitstring_batches(...)   # unchanged
    pt2 = SBDPT2Corrector(...);  out = pt2.correct(...)                     # new

This split is deliberate, not an accident of implementation:

* The variational solver is cluster-validated. PT2 is additive by construction and
  its binary cannot destabilise a production solve.
* ``epsilon2`` can be re-swept without redoing the diagonalization, which is the
  normal way this quantity is used and is otherwise very expensive.
* If PT2 runs out of memory (the perturber space is much larger than the variational
  one) the variational result is already on disk.

THE TWO VARIANTS, AND WHICH NUMBER YOU MAY QUOTE
------------------------------------------------
``variant="a"`` sums over DETERMINANT perturbers. Its perturber space spans every
spin, so ``E_PT2`` is not a target-S quantity and ``E_var + E_PT2`` is NOT a
single-spin energy. It is the variant that is directly comparable to published SHCI
numbers, which is why it exists.

``variant="c"`` sums over target-S CSF perturbers, one denominator per CSF. Use this
one with ``single_spin``. It matters more than it first looks: determinants within one
spin-coupled configuration have DIFFERENT diagonal energies, so variant (a) gives them
different denominators and its first-order wavefunction is spin-contaminated even when
the reference is perfectly spin-pure. That contamination is intrinsic to
Epstein-Nesbet PT2 in a determinant basis and does not vanish as ``epsilon2 -> 0``.

The two are expected to differ, and measured on N2 the gap shrinks as the variational
space grows (6.7 -> 3.8 -> 1.7 mH over top50/top100/top1000) because it is the weight
of the non-target-S perturbers that (c) correctly discards.

WHAT IT NEEDS
-------------
1. The ``pt2`` binary (``apps/chemistry_gdb_pt2/pt2``).
2. The FCIDUMP -- the SAME one the solve used.
3. The determinant list -- the SAME list the solve used. PT2 aborts rather than
   proceed if the wavefunction it reads is not fully covered by this list, because a
   silently incomplete reference state gives a plausible wrong number.
4. The wavefunction, as either the merged ``.npz`` (preferred) or the raw shards.
5. ``E_0``, the variational energy. Every denominator depends on it. The flat format
   carries it; the shard format does not, so ``e0`` must be passed with shards.

The ``.npz`` is the better input: it is already merged across ``b_comm`` ranks, it is
read back and verified before ``delete_shards=True`` removes the shards, and it is
what survives in a low-footprint production run.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sbd_GDB_solver import (
    _write_gdb_dets,
    npz_to_pt2_flat,
    _roundtrip_fcidump_via_pyscf,
)

__all__ = ["SBDPT2Corrector", "PT2Output"]


@dataclass
class PT2Output:
    """Result of one PT2 run.

    ``e_total`` is ``e_var + e_pt2``. Whether that may be quoted as a single-spin
    energy depends on the variant -- see ``is_spin_pure``, which is not a restatement
    of ``variant`` but the actual condition: variant (c) run with a multiplicity.
    """

    e_var: float
    e_pt2: float
    e_total: float
    variant: str
    epsilon2: float
    multiplicity: int | None = None
    psi1_norm2: float | None = None
    max_c1: float | None = None
    n_references: int | None = None
    n_emitted: int | None = None
    n_unique_dets: int | None = None
    n_removed_variational: int | None = None
    n_floored: int | None = None
    # variant (c) only
    n_configs: int | None = None
    n_configs_no_target_s: int | None = None
    n_configs_partly_variational: int | None = None
    n_orbit_rows: int | None = None
    n_rows_completed: int | None = None
    n_csf: int | None = None
    timing: dict = field(default_factory=dict)
    workdir: str | None = None
    stdout: str = ""

    @property
    def is_spin_pure(self) -> bool:
        """True only for variant (c) with a target multiplicity set."""
        return self.variant == "c" and self.multiplicity is not None and self.multiplicity >= 1

    def __repr__(self) -> str:
        tag = "spin-pure" if self.is_spin_pure else "determinant-space"
        return (
            f"PT2Output(E_var={self.e_var:.9f}, E_PT2={self.e_pt2:.9f}, "
            f"E_total={self.e_total:.9f}, variant={self.variant!r} [{tag}], "
            f"epsilon2={self.epsilon2:g})"
        )


class SBDPT2Corrector:
    """
    Wraps the ``pt2`` binary.

    One ``correct()`` call spawns one ``mpirun``. Nothing here writes into the
    variational solver's state, and the binary this drives does not link the solver.
    """

    def __init__(
        self,
        *,
        pt2_binary: str | Path | None = None,
        fcidump_path: str | Path | None = None,
        verbose: bool = False,
        # MPI
        mpirun: str = "mpirun",
        mpi_np: int = 1,
        mpirun_args: list[str] | None = None,
        omp_num_threads: int | None = None,
        b_comm_size: int = 1,
        t_comm_size: int = 1,
        # PT2 knobs
        variant: str = "a",
        epsilon2: float = 1e-8,
        single_spin: int | None = None,
        den_floor: float = 1e-8,
        collapse_at: int | None = None,
        bit_length: int = 20,
        # bookkeeping
        temp_dir: str | Path | None = None,
        clean_temp_dir: bool = False,
        timeout_s: float | None = None,
        extra_cli_args: list[str] | None = None,
        canonicalize_fcidump: bool = False,
    ):
        """
        Args:
            pt2_binary: path to the ``pt2`` executable. Defaults to
                ``<repo_root>/apps/chemistry_gdb_pt2/pt2``.
            fcidump_path: the FCIDUMP. MUST be the one the solve used: PT2 evaluates
                ``H_ai`` with these integrals against amplitudes obtained from
                different ones otherwise, and nothing downstream would reveal it.
            mpi_np: total rank count. Must equal
                ``h_comm_size x b_comm_size x t_comm_size``.
            b_comm_size: reference-partition communicator, and the main parallel
                axis of variant (a): the determinant list is redistributed across
                these ranks and perturbers are merged back across them by a
                hash-partitioned Alltoallv, so E_PT2 is exact and bit-identical for
                any b_comm_size. Variant (c) does NOT support b_comm_size > 1 and
                this constructor rejects it -- its perturber index is a CSF spanning
                a whole configuration, which a determinant-level partition
                splits. Variant (c) supports it too, by a different route: the
                configuration set is replicated across b_comm and the orbit
                numerators are summed globally, so its result agrees across layouts
                to round-off (~1e-9 relative) rather than bit-for-bit, since a
                distributed MPI_SUM combines in an implementation-defined order.
                Note that (c)'s numerator phase does not speed up monotonically
                with b_comm: each rank scans every replicated orbit row against its
                own references, so the row count grows to the global union as the
                per-rank reference count falls.
            t_comm_size: task communicator. Must not exceed ``b_comm_size``; the
                binary aborts otherwise rather than returning a plausible number.
            variant: ``"a"`` (determinant perturbers) or ``"c"`` (target-S CSF
                perturbers). ``"c"`` requires ``single_spin``.
            epsilon2: screening threshold on ``|H_ai c_i|``. Smaller is more
                complete and more expensive. Variant (c) is far less sensitive to it
                than (a), because rows dropped by screening are recovered by the
                orbit-completion step rather than lost.
            single_spin: target spin multiplicity ``2S+1`` (1=singlet, 3=triplet).
                Required for ``variant="c"``, ignored by ``variant="a"``.
            den_floor: floor on ``|E_0 - H_aa|``. A perturber near-degenerate with
                ``E_0`` would otherwise contribute an unbounded term. The count of
                times it triggers is reported in ``PT2Output.n_floored``; a nonzero
                count means the result is partly held up by regularization.
            collapse_at: floor on the perturber count held before the accumulation
                store is collapsed (``--batch_size``). This is a SAFETY floor, not a
                memory dial: the trigger only grows from it, so peak memory is set by
                the unique perturber count. Measured on N2 K=17688, three settings
                spanning 40x moved peak RSS by 8%. Leave it alone unless a run dies
                on memory, and see the note in ``chemistry/gdb/pt2.h`` first.
            bit_length: MUST match the solve that wrote the wavefunction. Every
                determinant is misparsed otherwise, silently, because the words still
                fit.
            clean_temp_dir: default False, unlike the solver's. The PT2 workdir holds
                the flat wavefunction and the stdout that carries the diagnostics
                worth reading; deleting it by default would throw away the evidence
                for a number that has no other provenance.
            canonicalize_fcidump: default False. The solve has already canonicalized
                its FCIDUMP; round-tripping it again here would risk a different
                integral file than the amplitudes were computed with, which is the
                one inconsistency this wrapper cannot detect.
        """
        variant = str(variant).lower()
        if variant in ("a", "det", "determinant"):
            variant = "a"
        elif variant in ("c", "csf", "spinpure", "spin-pure"):
            variant = "c"
        else:
            raise ValueError(
                f"variant must be 'a' (determinant) or 'c' (spin-pure), got {variant!r}"
            )
        if variant == "c" and (single_spin is None or single_spin < 1):
            raise ValueError(
                "variant='c' needs single_spin=<2S+1> (1=singlet, 3=triplet): the CSF "
                "perturber space is defined by a target spin, so there is no default"
            )
        if variant == "a" and single_spin is not None:
            warnings.warn(
                "single_spin is ignored by variant='a': its perturber space spans "
                "every spin regardless, so E_var + E_PT2 is not a single-spin energy. "
                "Use variant='c' if you need one.",
                UserWarning,
                stacklevel=2,
            )
        if not (epsilon2 > 0.0):
            raise ValueError(f"epsilon2 must be positive, got {epsilon2}")
        if b_comm_size < 1 or t_comm_size < 1:
            raise ValueError("b_comm_size and t_comm_size must be >= 1")
        if t_comm_size > b_comm_size:
            raise ValueError(
                f"t_comm_size ({t_comm_size}) must not exceed b_comm_size "
                f"({b_comm_size}); t_comm partitions exactly b_comm_size "
                f"ring-rotation tasks, so surplus t ranks would get an empty range"
            )
        if mpi_np % (b_comm_size * t_comm_size) != 0:
            raise ValueError(
                f"b_comm_size x t_comm_size ({b_comm_size * t_comm_size}) must divide "
                f"mpi_np ({mpi_np})"
            )

        if pt2_binary is None:
            here = Path(__file__).resolve().parent
            pt2_binary = here.parent / "apps" / "chemistry_gdb_pt2" / "pt2"
        # Absolute, always: the subprocess runs with cwd=workdir, so a relative path
        # given here would resolve against the temp directory and fail with a launcher
        # error rather than a readable one.
        self._pt2_binary = str(Path(pt2_binary).resolve())
        if not os.path.isfile(self._pt2_binary):
            raise FileNotFoundError(
                f"pt2 binary not found at {self._pt2_binary}. Build it with "
                f"`cd apps/chemistry_gdb_pt2 && make`, or pass pt2_binary=..."
            )

        self._fcidump_path = (
            Path(fcidump_path).resolve() if fcidump_path is not None else None
        )
        self.verbose = bool(verbose)
        self.mpirun = mpirun
        self.mpi_np = int(mpi_np)
        if mpirun_args is None:
            n_omp = 1 if omp_num_threads is None else int(omp_num_threads)
            self.mpirun_args = ["-x", f"OMP_NUM_THREADS={n_omp}"]
        else:
            self.mpirun_args = list(mpirun_args)
        self.b_comm_size = int(b_comm_size)
        self.t_comm_size = int(t_comm_size)
        self.variant = variant
        self.epsilon2 = float(epsilon2)
        self.single_spin = None if single_spin is None else int(single_spin)
        self.den_floor = float(den_floor)
        self.collapse_at = None if collapse_at is None else int(collapse_at)
        self.bit_length = int(bit_length)
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.clean_temp_dir = bool(clean_temp_dir)
        self.timeout_s = timeout_s
        self.extra_cli_args = list(extra_cli_args or [])
        self.canonicalize_fcidump = bool(canonicalize_fcidump)

    # -- the two entry points ------------------------------------------------

    def correct_from_npz(
        self,
        npz_path: str | Path,
        *,
        e0: float,
        norb: int,
        ci_strings: tuple[np.ndarray, np.ndarray] | None = None,
        detfile: str | Path | None = None,
        fcidump_path: str | Path | None = None,
    ) -> PT2Output:
        """
        Correct a merged ``experimental_SCIState`` ``.npz``.

        This is the preferred path. The ``.npz`` is already merged across ``b_comm``
        ranks and is read back and verified before ``delete_shards=True`` deletes the
        shards, so it is the object that exists in a low-footprint production run.

        Exactly one of ``ci_strings`` or ``detfile`` must identify the variational
        space. Passing ``ci_strings`` is the safer of the two: the determinant file is
        then written from the same arrays, so it cannot be a stale list from a
        different solve -- the one error PT2 detects but cannot repair.

        Args:
            npz_path: the merged archive.
            e0: variational energy of the root, INCLUDING the FCIDUMP core energy.
                This is the number that matches the diagonal elements PT2 computes,
                which carry ECORE. `SBDGdbSolver` reports an ELECTRONIC energy, so
                what goes here is ``result.energy + core_energy``, not
                ``result.energy``. Getting it wrong shifts every denominator by
                |ECORE| and gives a small positive E_PT2 -- the binary detects that
                and aborts with the corrected value rather than returning it.
            norb: spatial orbitals.
            ci_strings: ``(strs_a, strs_b)``, paired, as handed to the solver.
            detfile: an existing determinant file, if you prefer.
            fcidump_path: overrides the constructor's.
        """
        return self._run(
            wf_source=("npz", Path(npz_path)),
            e0=e0,
            norb=norb,
            ci_strings=ci_strings,
            detfile=detfile,
            fcidump_path=fcidump_path,
        )

    def correct_from_shards(
        self,
        shard_prefix: str | Path,
        *,
        e0: float,
        norb: int,
        ci_strings: tuple[np.ndarray, np.ndarray] | None = None,
        detfile: str | Path | None = None,
        fcidump_path: str | Path | None = None,
    ) -> PT2Output:
        """
        Correct a raw per-rank shard set, for the case where PT2 runs immediately
        after a solve that has not merged.

        ``shard_prefix`` is the path with the rank tag REMOVED but any root tag kept:
        files are named ``<prefix>000000.bin``, ``<prefix>000001.bin``, ... so a
        multi-root solve writing ``wf_root0000000.bin`` has prefix ``wf_root0``.
        Getting this wrong produces "no shard found", not a wrong answer.

        ``e0`` includes the core energy here too -- see :meth:`correct_from_npz`.

        Shards carry no energy, so ``e0`` is required here and is not cross-checked
        against anything -- unlike the ``.npz`` path, where the flat file stores
        ``E_0`` and the binary refuses to run if the two disagree. Prefer the ``.npz``
        for that reason alone.
        """
        return self._run(
            wf_source=("shards", Path(shard_prefix)),
            e0=e0,
            norb=norb,
            ci_strings=ci_strings,
            detfile=detfile,
            fcidump_path=fcidump_path,
        )

    def correct(self, *args, **kwargs) -> PT2Output:
        """Alias for :meth:`correct_from_npz`."""
        return self.correct_from_npz(*args, **kwargs)

    # -- internals -----------------------------------------------------------

    def _run(
        self,
        *,
        wf_source: tuple[str, Path],
        e0: float,
        norb: int,
        ci_strings,
        detfile,
        fcidump_path,
    ) -> PT2Output:
        kind, wf_path = wf_source
        norb = int(norb)

        if (ci_strings is None) == (detfile is None):
            raise ValueError(
                "pass exactly one of ci_strings=(strs_a, strs_b) or detfile=<path>: "
                "PT2 needs the variational determinant list, and guessing which of "
                "two disagreeing sources is authoritative is how a silently wrong "
                "reference state happens"
            )

        fci = (
            Path(fcidump_path).resolve() if fcidump_path is not None
            else self._fcidump_path
        )
        if fci is None:
            raise ValueError(
                "fcidump_path is required: PT2 evaluates H_ai from the integrals, and "
                "they must be the ones the solve used"
            )
        if not fci.is_file():
            raise FileNotFoundError(f"FCIDUMP not found: {fci}")

        parent = self.temp_dir if self.temp_dir else Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="sbd_pt2_", dir=str(parent)))

        succeeded = False
        try:
            # -- FCIDUMP
            if self.canonicalize_fcidump:
                fci_use = workdir / "FCIDUMP.generated"
                _roundtrip_fcidump_via_pyscf(fci, fci_use)
            else:
                fci_use = fci

            # -- determinant list
            if ci_strings is not None:
                strs_a = np.asarray(ci_strings[0], dtype=np.int64)
                strs_b = np.asarray(ci_strings[1], dtype=np.int64)
                if len(strs_a) != len(strs_b):
                    raise ValueError(
                        f"ci_strings must be paired: {len(strs_a)} alpha vs "
                        f"{len(strs_b)} beta"
                    )
                det_use = workdir / "dets.txt"
                _write_gdb_dets(det_use, strs_a, strs_b, norb)
            else:
                det_use = Path(detfile).resolve()
                if not det_use.is_file():
                    raise FileNotFoundError(f"detfile not found: {det_use}")

            # -- wavefunction
            if kind == "npz":
                flat = workdir / "wf.flat"
                ndet = npz_to_pt2_flat(
                    wf_path, flat, e0=float(e0), norb=norb,
                    bit_length=self.bit_length,
                )
                loadname = str(flat)
                if self.verbose:
                    print(f"  converted {wf_path} -> {flat} ({ndet} determinants)")
            else:
                # Shard prefix is a path prefix, not a file, so .resolve() would be
                # wrong on the last component; make it absolute without resolving.
                loadname = str(wf_path if wf_path.is_absolute()
                               else Path.cwd() / wf_path)

            cmd = self._build_cmd(fci_use, det_use, loadname, e0)
            if self.verbose:
                print(f"  Work dir: {workdir}")
                print(f"  Command:  {' '.join(cmd)}")

            proc = subprocess.run(
                cmd, cwd=str(workdir), capture_output=True, text=True,
                timeout=self.timeout_s, check=False,
            )
            try:
                (workdir / "pt2_stdout.txt").write_text(proc.stdout or "")
                (workdir / "pt2_stderr.txt").write_text(proc.stderr or "")
                (workdir / "pt2_command.txt").write_text(" ".join(cmd) + "\n")
            except Exception as exc:
                warnings.warn(
                    f"Failed to persist pt2 stdout/stderr to {workdir}: {exc}",
                    UserWarning, stacklevel=3,
                )

            if proc.returncode != 0:
                tail = "\n".join((proc.stdout or "").splitlines()[-40:])
                raise RuntimeError(
                    f"pt2 invocation failed (rc={proc.returncode}).\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Work dir: {workdir}\n"
                    f"--- stderr ---\n{proc.stderr}\n"
                    f"--- stdout tail ---\n{tail}"
                )

            out = _parse_pt2_stdout(proc.stdout or "")
            out.variant = self.variant
            out.epsilon2 = self.epsilon2
            out.multiplicity = self.single_spin
            out.workdir = str(workdir)
            out.stdout = proc.stdout or ""
            _warn_on_pt2_diagnostics(out, proc.stdout or "", workdir)
            succeeded = True
            return out
        finally:
            if self.clean_temp_dir and succeeded:
                import shutil
                shutil.rmtree(workdir, ignore_errors=True)

    def _build_cmd(
        self, fcidump_path: Path, det_file: Path, loadname: str, e0: float
    ) -> list[str]:
        return [
            self.mpirun, "-np", str(self.mpi_np),
            *self.mpirun_args,
            self._pt2_binary,
            "--fcidump", str(fcidump_path),
            "--detfiles", str(det_file),
            "--loadname", loadname,
            "--e0", f"{float(e0):.12f}",
            "--epsilon2", f"{self.epsilon2:.6e}",
            "--variant", self.variant,
            *(["--single_spin", str(self.single_spin)]
              if self.single_spin is not None else []),
            "--den_floor", f"{self.den_floor:.6e}",
            *(["--batch_size", str(self.collapse_at)]
              if self.collapse_at is not None else []),
            "--bit_length", str(self.bit_length),
            "--b_comm_size", str(self.b_comm_size),
            "--t_comm_size", str(self.t_comm_size),
            *self.extra_cli_args,
        ]

    def __repr__(self) -> str:
        return (
            f"SBDPT2Corrector(variant={self.variant!r}, epsilon2={self.epsilon2:g}, "
            f"single_spin={self.single_spin}, mpi_np={self.mpi_np}, "
            f"b_comm_size={self.b_comm_size}, t_comm_size={self.t_comm_size})"
        )


# ---------------------------------------------------------------------------
# stdout parsing
# ---------------------------------------------------------------------------

def _f(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _i(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _parse_pt2_stdout(text: str) -> PT2Output:
    """
    Pull the numbers out of the binary's stdout.

    The energies are REQUIRED: a run that exited 0 without printing them did not
    produce a result, and inventing a default would turn that into a plausible
    number. Everything else is a diagnostic and is allowed to be absent.
    """
    e_var = _f(text, r"E_var\s*=\s*(-?[\d.eE+-]+)")
    e_pt2 = _f(text, r"E_PT2\s*=\s*(-?[\d.eE+-]+)")
    e_tot = _f(text, r"E_var \+ E_PT2\s*=\s*(-?[\d.eE+-]+)")
    if e_var is None or e_pt2 is None or e_tot is None:
        raise RuntimeError(
            "pt2 exited 0 but printed no energies; its stdout was:\n" + text
        )

    timing = {}
    tm = re.search(r"pt2: timing:\s*(.+?),\s*total\s*([\d.]+)\s*s", text)
    if tm:
        for part in tm.group(1).split(","):
            part = part.strip()
            m = re.match(r"([a-z_ ]+?)\s+([\d.]+)\s*s", part)
            if m:
                timing[m.group(1).strip().replace(" ", "_")] = float(m.group(2))
        timing["total"] = float(tm.group(2))

    out = PT2Output(
        e_var=e_var, e_pt2=e_pt2, e_total=e_tot,
        variant="?", epsilon2=float("nan"),
        psi1_norm2=_f(text, r"\|Psi_1\|\^2\s*=\s*([\d.eE+-]+)"),
        max_c1=_f(text, r"max \|c_a\^\(1\)\|\s*=\s*([\d.eE+-]+)"),
        n_references=_i(text, r"references=(\d+)"),
        n_emitted=_i(text, r"emitted=(\d+)"),
        n_unique_dets=_i(text, r"unique_dets=(\d+)"),
        n_removed_variational=_i(text, r"removed_variational=(\d+)"),
        n_floored=_i(text, r"floored=(\d+)"),
        n_configs=_i(text, r"configs=(\d+)"),
        n_configs_no_target_s=_i(text, r"no_target_S=(\d+)"),
        n_configs_partly_variational=_i(text, r"partly_variational=(\d+)"),
        n_orbit_rows=_i(text, r"orbit_rows=(\d+)"),
        n_rows_completed=_i(text, r"completed=(\d+)"),
        n_csf=_i(text, r"csfs=(\d+)"),
        timing=timing,
    )
    return out


def _warn_on_pt2_diagnostics(out: PT2Output, text: str, workdir: Path) -> None:
    """
    Re-raise the binary's own warnings as Python warnings.

    The binary prints these to stdout, where they end up in a file nobody reads. Each
    one marks a way the number can be misleading while still looking fine, so they are
    surfaced rather than archived.
    """
    if out.e_pt2 > 0.0:
        warnings.warn(
            f"E_PT2 is POSITIVE ({out.e_pt2:.9f}). Every perturber lies above E_0 for "
            f"a ground-state root, so this points at a wrong e0, an excited root, or a "
            f"defect. Work dir: {workdir}",
            RuntimeWarning, stacklevel=4,
        )
    if out.n_floored:
        warnings.warn(
            f"{out.n_floored} PT2 denominator(s) hit the floor: the correction is "
            f"partly held up by regularization, meaning some perturber is nearly "
            f"degenerate with E_0 and belongs in the variational space instead.",
            RuntimeWarning, stacklevel=4,
        )
    if out.max_c1 is not None and out.max_c1 > 0.5:
        warnings.warn(
            f"largest first-order coefficient is {out.max_c1:.4f}; perturbation theory "
            f"is not reliable when a perturber mixes that strongly -- it belongs in the "
            f"variational space.",
            RuntimeWarning, stacklevel=4,
        )
    if "are populated on more than one b rank" in text:
        warnings.warn(
            "pt2 reports configurations shared across b_comm ranks: variant (c) is "
            "INEXACT in that case, because a completed orbit row's numerator is summed "
            "over one rank's references only and then squared per rank. Re-run with "
            "b_comm_size=1 to get the exact value.",
            RuntimeWarning, stacklevel=4,
        )
    if out.variant == "c" and out.n_orbit_rows and not out.n_rows_completed:
        warnings.warn(
            "no orbit row needed completion. Possible but unusual; if it holds at "
            "every epsilon2 the completion step is not running, and variant (c) has "
            "silently degenerated into a reweighted variant (a).",
            RuntimeWarning, stacklevel=4,
        )
