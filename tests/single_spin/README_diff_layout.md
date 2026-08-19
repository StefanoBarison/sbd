# Differential layout debugging

Tooling for the open bug: `--single_spin --method 1` returns non-deterministic
wrong energies when `b_comm>1` **and** `h_comm>1` **and** `OMP_NUM_THREADS>=3`.
Remove any one condition and it is correct.

Ten mechanism hypotheses were falsified one at a time before this. The method here
is different: compare a failing decomposition against a working one step by step,
and let the measurement name the step. No hypothesis is required.

## Usage

```sh
./diff_layout.sh [DETFILE] [NP] [NB] [OMP] [REPEATS]
CURVE=--curve ./diff_layout.sh /tmp/n2reg/det_top8000.txt 4 2 3 5
```

Reference layout is `b_comm=NP` (so `h_comm=1`), which is verified correct at any
thread count. Suspect layout is `b_comm=NB`, giving `h_comm=NP/NB`. Failing runs
are kept as `/tmp/fp_bad.log` next to `/tmp/fp_ref.log`.

Then read the growth curve:

```sh
python3 fp_compare.py --curve /tmp/fp_ref.log /tmp/fp_bad.log
```

## Two traps this tooling exists to avoid

**Most of the fingerprint output is not comparable across layouts.** The Krylov
basis legitimately differs between decompositions, so `v`, `Hv` and `H` disagree in
the 4th digit while `E0` agrees to 1 ulp at every step. The global CSF numbering is
also layout-dependent, so `sw` and `aidx` name different CSFs at `b_comm=1` and
`2`. Only `E`, `Ritz` and `norm_r` are physical. Diffing the rest produces nothing
but false positives — 48 of 68 lines in calibration, with nothing wrong.

**A threshold crossing is not the defect's origin.** The error ramps geometrically
once it starts, so it crosses any fixed threshold several steps after it began.
Reporting the crossing points at an innocent step. `--curve` shows the whole
trajectory; look for the order-of-magnitude jump away from the floor.

## What has been measured

Noise floor between two *correct* layouts, `top100` (K=117) and `top8000`
(K=36947), over `b_comm=1/2/4`:

| quantity | floor | threshold used |
|---|---|---|
| `E` | 9.2e-16 .. 1.0e-15 | 1e-10 |
| `Ritz` | 1.5e-12 .. 2.5e-12 | 1e-8 |
| `norm_r` | 2.2e-9 .. 3.5e-9 | 1e-5 (weak, never decides) |

The floor barely moves with problem size, so the thresholds transfer.

Controls, all clean: identical layout (exactly 0.00e+00, bitwise reproducible);
different layout at OMP=1; same layout at OMP=3. The harness does not fire on
layout changes, thread changes, or repetition.

Failing case, `top8000`, ref `b=4/h=1` OMP=1 vs suspect `b=2/h=2` OMP=3, 4/4
diverged:

```
1.6    Ritz=7.02e-13            <- floor, 17 straight steps
1.7    Ritz=4.95e-10   x705     <- origin
1.8    Ritz=2.37e-08   x47.9
1.11   Ritz=1.63e-02   x795
1.12   Ritz=6.56e-01            <- fully wrong
```

Origin step across repeats: `1.7`, `0.11`, `0.10`, `0.11`. Not fixed, but always
`ib >= 10` — a large Krylov basis late in an inner cycle. Two are in outer
iteration 0, so thick restart is not involved.

The reference converges in 26 steps; the failing runs take ~3000, reproducing the
reference trajectory correctly and then failing to stop. `fp_compare.py` returns 3
for that case rather than 0, so it cannot be misread as a pass.

## Next

Instrument the operations active at `ib >= 10` and bisect within them. Do not
propose a mechanism first.
