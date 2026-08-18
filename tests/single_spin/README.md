# Phase C verification: single-spin solver at b_comm_size > 1

## Run it

```bash
# 1. generate the determinant files (from single_spin_workflow/)
python make_n2_dets.py ~/Downloads/for_claude_n2/counts_target_0-1487_50exc_N2R2_631G_FB.json \
       16 50,100,1000 /tmp/n2reg

# 2. generate the deliberately-invalid files for the negative tests
python tests/single_spin/make_negative_dets.py /tmp/n2reg/det_top50.txt 16 /tmp/n2reg

# 3. run the suite (exit 0 = all passed)
tests/single_spin/run_phase_c.sh <fcidump> /tmp/n2reg
```

64 checks. The property under test is that **converged energies do not depend on
rank count**; the rest either supports that or guards a failure mode that is
silent rather than loud.

References in `oracle_refs.txt` come from `verify_n2_sbd.py`, which builds H and
S^2 over the same determinant list and diagonalizes densely — different code,
different language, no shared Slater-Condon routines. So section 1 compares SBD
against exact diagonalization, not against SBD.

## Two things that will bite you

**`SBD_SS_PAR_THRESHOLD=0` is mandatory for multi-rank testing.** The parallel and
fused-MGS branches are gated on `K > 4096` where `K` is the **local** slice
length. Distributing a problem shrinks `K`, so those branches — the ones holding
the MPI-inside-OpenMP code — become unreachable in exactly the runs that need to
exercise them. Section 3 forces them. This was verified to actually reach the code
by deleting the fused-MGS allreduce: the answer breaks (−333.18) at threshold 0
and stays correct at the default.

**Tolerance, not bitwise.** Comparisons use 1e-8 Ha. The solver's own tolerance is
1e-4, and codegen differences move the last digits by ~1e-13 (see the Phase B2
commit). Demanding bitwise equality here would produce false failures.

## Cluster scaling: where B2 actually helps

Measured single-root optimum was `np=48, OMP_NUM_THREADS=4`. Note what that means
for the decomposition, because it is easy to get no benefit from B2:

    h_comm_size = mpi_size / (b_comm_size * t_comm_size)

So `np=48` with the default `--b_comm_size 1` gives **h_comm=48**, and `h_comm`
*replicates* the determinant list across ranks rather than splitting it
(`main.cc:162-164`). B2 distributes the CSF/Krylov space over **b_comm** only, so
at `b_comm=1` it changes nothing at all.

To use it, move ranks into `b_comm` and pass `--do_redist_config 1`:

| np | `--b_comm_size` | h_comm | CSF space per rank | det list |
|---|---|---|---|---|
| 48 | 1 (old default) | 48 | full (no benefit) | replicated 48× |
| 48 | 4 | 12 | 1/4 | replicated 12× |
| 48 | 12 | 4 | 1/12 | replicated 4× |
| 48 | 48 | 1 | 1/48 | not replicated |

Larger `b_comm` shrinks both the replicated determinant list and the Krylov basis,
which is what the 1e8-determinant target needs (a replicated Krylov basis there is
7.6–10.2 GiB/rank and does **not** shrink with node count). But `b_comm` also
carries the matvec's ket rotation, so the wall-clock optimum is an empirical
trade-off — sweep `--b_comm_size 1/4/12/48` at fixed `np=48` rather than assuming
the single-rank optimum transfers.

`--do_redist_config 1` is required whenever `b_comm_size > 1`; without it the run
aborts (correctly) because configuration orbits get split across ranks.

## Running the sweep

```bash
tests/single_spin/cluster_bcomm_sweep.sh <fcidump> <detfile> [np] [omp] [nroots]
# e.g. at the measured single-root optimum:
tests/single_spin/cluster_bcomm_sweep.sh $FCI dets.txt 48 4 1
```

It runs every valid `--b_comm_size` divisor of `np`, prints energy alongside
`LOOP TOTAL` (the solve) and wall-clock, and adds `--do_redist_config 1`
automatically whenever `b_comm > 1`. **Every energy in the table must agree to
~1e-8 Ha** -- a row that differs is wrong, not just slower.

Measured locally at np=4, K=3906 (N2 top1000), solve time:

| b_comm | h_comm | solve (s) |
|---|---|---|
| 1 | 4 | 0.64 |
| 2 | 2 | 0.83 |
| 4 | 1 | 0.99 |

So at small K, distributing is a net LOSS -- the CSF-space allreduces cost more
than the shrunken slices save. That is the expected shape and the reason to sweep
rather than assume: b_comm pays off when the Krylov basis is large enough that
memory, not latency, is the binding constraint (at 1e8 determinants a replicated
basis is 7.6-10.2 GiB/rank and does not shrink with node count at all). Do not
read the small-K result as an argument against B2; read it as the reason the
optimum must be measured at production K.
