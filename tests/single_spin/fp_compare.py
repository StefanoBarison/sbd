#!/usr/bin/env python3
"""Compare two SBD_SS_FP logs and report the FIRST inner step where the
physically-comparable quantities diverge.

Chunk 2 of the differential hunt for the b_comm>1 + h_comm>1 + OMP>=3 wrong-energy
bug. The point of this script is to name the first operation that goes wrong, not
to guess a mechanism -- ten mechanism hypotheses have already been falsified.

WHAT IS COMPARED, and why only this. Calibration on three known-good layouts
(see the SBD_SS_FP block in include/sbd/chemistry/gdb/single_spin.h) showed that
the Krylov basis itself legitimately differs between MPI decompositions: E0 agreed
to 1 ulp at every step while 48 of 68 basis-vector fingerprints differed in the
4th digit. So v, Hv, H, sw and aidx are NOT compared here -- doing so yields pure
false positives. Only E, Ritz and norm_r are basis-independent.

Thresholds are the calibrated worst-case good-layout disagreement, with headroom.
Measured on two problem sizes (top100 K=117, top8000 K=36947) across b_comm=1/2/4:
  E       9.2e-16 .. 1.0e-15  -> 1e-10
  Ritz    1.5e-12 .. 2.5e-12  -> 1e-8
  norm_r  2.2e-09 .. 3.5e-09  -> 1e-5   (weak; reported, never decides)
The floor barely moves with problem size, so these transfer.

REPORTING THE GROWTH CURVE, NOT JUST THE FIRST CROSSING. A threshold crossing is
not the same as the defect's origin: a corruption that starts at the 1e-12 noise
floor and is amplified geometrically will cross 1e-8 several steps after it began,
and reporting only the crossing would point at an innocent step. Earlier work on
this bug already measured such an amplifier (residual growth x0.75..x2.04 for
seven steps, then x73.5 in one). So --curve prints the per-step relative error for
every common step, which distinguishes:
  * a step where the error JUMPS from the floor by orders of magnitude
        -> that step is where the defect acts
  * a smooth geometric ramp from the floor
        -> the defect is upstream of the first ramping step; the crossing is
           a consequence, not the cause
Read the curve before concluding anything about a step.

Step ordering: "it=I.B" is outer iteration I, inner step B. Sorted numerically by
(I, B), NOT lexically -- 0.10 must come after 0.9.
"""
import re
import sys

TOL = {'E': 1.0e-10, 'Ritz': 1.0e-8, 'norm_r': 1.0e-5}


def parse(path):
    """-> {(it,ib): {'E':[..], 'norm_r':[..], 'Ritz':{root:(s1,s2,amax)}}}"""
    steps = {}

    def slot(st):
        return steps.setdefault(st, {'E': [], 'norm_r': [], 'Ritz': {}})

    for line in open(path, errors='replace'):
        if '[fp] ' not in line:
            continue
        m = re.search(r'it=(\d+)\.(\d+)', line)
        if not m:
            continue
        st = (int(m.group(1)), int(m.group(2)))
        if '[fp] H ' in line:
            s = slot(st)
            s['E'] = [float(x) for x in re.findall(r'E\d+=(\S+)', line)]
            s['norm_r'] = [float(x) for x in re.findall(r'nr\d+=(\S+)', line)]
        elif '[fp] Rz' in line:
            r = int(re.search(r'r=(\d+)', line).group(1))
            g = lambda k: float(re.search(k + r'=(\S+)', line).group(1))
            # |s1| : eigenvectors are sign-arbitrary between layouts.
            slot(st)['Ritz'][r] = (abs(g('s1')), g('s2'), g('amax'))
    return steps


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def compare(ref, tst):
    """-> (first_bad_step, quantity, relerr, detail) or (None, ...) if clean."""
    common = sorted(set(ref) & set(tst))
    if not common:
        return ('NO-COMMON-STEPS', None, None,
                'the two runs share no inner step; one probably aborted early')
    worst = {k: 0.0 for k in TOL}
    for st in common:
        r, t = ref[st], tst[st]
        # E first: sharpest, and a wrong E is the actual symptom.
        for i, (a, b) in enumerate(zip(r['E'], t['E'])):
            e = rel(a, b)
            worst['E'] = max(worst['E'], e)
            if e > TOL['E']:
                return (st, 'E', e, f'root {i}: ref={a!r} tst={b!r}')
        for i in sorted(set(r['Ritz']) & set(t['Ritz'])):
            for nm, a, b in zip(('s1', 's2', 'amax'), r['Ritz'][i], t['Ritz'][i]):
                e = rel(a, b)
                worst['Ritz'] = max(worst['Ritz'], e)
                if e > TOL['Ritz']:
                    return (st, 'Ritz', e,
                            f'root {i} {nm}: ref={a!r} tst={b!r}')
        for i, (a, b) in enumerate(zip(r['norm_r'], t['norm_r'])):
            worst['norm_r'] = max(worst['norm_r'], rel(a, b))
    return (None, None, worst, f'{len(common)} common steps')


def curve(ref, tst):
    """Per-step relative error for every comparable quantity, all steps.

    This is the output to read when deciding WHERE a defect acts. The `first`
    column flags the step at which each quantity first exceeds its threshold, but
    the numbers matter more than the flag -- look for an order-of-magnitude jump
    away from the ~1e-12 floor, not for the flag.
    """
    common = sorted(set(ref) & set(tst))
    print(f'{"step":>8}  {"E":>10}  {"Ritz":>10}  {"norm_r":>10}   ratio(Ritz)')
    prev = None
    for st in common:
        r, t = ref[st], tst[st]
        eE = max((rel(a, b) for a, b in zip(r['E'], t['E'])), default=0.0)
        eR = 0.0
        for i in sorted(set(r['Ritz']) & set(t['Ritz'])):
            for a, b in zip(r['Ritz'][i], t['Ritz'][i]):
                eR = max(eR, rel(a, b))
        eN = max((rel(a, b) for a, b in zip(r['norm_r'], t['norm_r'])), default=0.0)
        ratio = '' if prev in (None, 0.0) else f'x{eR / prev:.1f}'
        flag = ''
        if eE > TOL['E']:
            flag += ' E!'
        if eR > TOL['Ritz']:
            flag += ' Ritz!'
        print(f'{st[0]}.{st[1]:<6}  {eE:10.2e}  {eR:10.2e}  {eN:10.2e}   '
              f'{ratio:>7}{flag}')
        prev = eR
    extra = sorted(set(tst) - set(ref))
    if extra:
        print(f'  ... plus {len(extra)} step(s) present only in the test run '
              f'(it did not converge where the reference did)')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    want_curve = '--curve' in sys.argv
    if len(args) != 2:
        sys.exit('usage: fp_compare.py [--curve] REF.log TST.log')
    ref, tst = parse(args[0]), parse(args[1])
    if not ref:
        sys.exit('FAIL: no [fp] lines in the reference log -- was SBD_SS_FP=1 set?')
    if not tst:
        print(f'DIVERGE at step -       : quantity=NO-FP-OUTPUT relerr=n/a')
        print('  the test run produced no [fp] lines (aborted before the first '
              'inner step, or SBD_SS_FP was unset)')
        return 2
    if want_curve:
        curve(ref, tst)
        print()
    st, q, e, detail = compare(ref, tst)
    if st is None:
        # "Clean over the common steps" is NOT the same as "the run was correct".
        # The reference converges and stops; a failing run can reproduce every one
        # of those steps and then continue for thousands more. Report the step
        # counts so that case cannot be mistaken for a pass.
        extra = len(set(tst) - set(ref))
        print(f'CLEAN  ({detail}): worst rel  E={e["E"]:.2e}  '
              f'Ritz={e["Ritz"]:.2e}  norm_r={e["norm_r"]:.2e}')
        if extra:
            print(f'  NOTE: the test run took {extra} step(s) BEYOND the '
                  f'{len(ref)} the reference needed. It reproduces the '
                  f'reference trajectory and then fails to stop -- the defect '
                  f'is after this window, not inside it.')
            return 3
        return 0
    if q is None:
        print(f'DIVERGE at step {st}: {detail}')
        return 2
    print(f'DIVERGE at step it={st[0]}.{st[1]:<4} quantity={q:<7} relerr={e:.3e}')
    print(f'  {detail}')
    print(f'  ref steps={len(ref)}  tst steps={len(tst)}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
