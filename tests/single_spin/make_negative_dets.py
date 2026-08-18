"""
Build deliberately-invalid determinant files for the Phase C negative tests.

Two failure modes the projected solver must REFUSE rather than silently mangle:

  det_incomplete.txt  a spin-incomplete list: determinants deleted from one
                      configuration's Sz-orbit. V is block diagonal by
                      configuration, so a truncated block yields a CSF column
                      that is neither normalized nor an S^2 eigenvector. Before
                      the Phase B audit this returned -108.667380612 instead of
                      -108.667515671 with no warning at all.

  det_mixedsz.txt     a list spanning two Sz sectors. The projector is defined
                      for ONE Sz; mixing them is not a spin sector at all.

Usage: python make_negative_dets.py <good-detfile> <norb> <outdir>
"""
import sys
from collections import defaultdict
from pathlib import Path


def sz2(bits, norb):
    g = lambda k: bits[2 * norb - 1 - k]
    return sum(g(2 * p) for p in range(norb)) - sum(g(2 * p + 1) for p in range(norb))


def config_key(bits, norb):
    """Spatial configuration: per-orbital occupancy 0/1/2."""
    g = lambda k: bits[2 * norb - 1 - k]
    return tuple(g(2 * p) + g(2 * p + 1) for p in range(norb))


def main():
    good, norb_s, outdir_s = sys.argv[1:4]
    norb = int(norb_s)
    outdir = Path(outdir_s)
    outdir.mkdir(parents=True, exist_ok=True)

    dets = [l.strip() for l in open(good) if l.strip()]
    bits = {d: [int(c) for c in d] for d in dets}
    assert len({sz2(bits[d], norb) for d in dets}) == 1, "input is not a single Sz sector"

    # --- incomplete: drop 3 members of the LARGEST orbit -------------------
    orbits = defaultdict(list)
    for d in dets:
        orbits[config_key(bits[d], norb)].append(d)
    biggest = max(orbits.values(), key=len)
    assert len(biggest) > 3, f"largest orbit has only {len(biggest)} members"
    drop = set(biggest[:3])
    out = [d for d in dets if d not in drop]
    (outdir / "det_incomplete.txt").write_text("\n".join(out) + "\n")
    print(f"det_incomplete.txt: {len(out)} dets "
          f"(dropped 3 of an orbit of {len(biggest)})")

    # --- mixed Sz: flip one beta->alpha, requiring a NEW determinant --------
    # A flip that lands on an existing determinant is a silent no-op, because
    # the file is a sorted set -- so check membership and Sz explicitly.
    have = set(dets)
    for i, d in enumerate(dets):
        b = bits[d]
        for p in range(norb):
            ib_beta, ib_alpha = 2 * norb - 1 - (2 * p + 1), 2 * norb - 1 - (2 * p)
            if b[ib_beta] == 1 and b[ib_alpha] == 0:
                nb = list(b)
                nb[ib_beta], nb[ib_alpha] = 0, 1
                new = "".join(map(str, nb))
                if new not in have and sz2(nb, norb) != sz2(b, norb):
                    out = list(dets); out[i] = new
                    (outdir / "det_mixedsz.txt").write_text("\n".join(out) + "\n")
                    print(f"det_mixedsz.txt: {len(out)} dets, Sz2 values "
                          f"{sorted({sz2([int(c) for c in x], norb) for x in out})}")
                    return
    raise SystemExit("no valid Sz-changing flip found")


if __name__ == "__main__":
    main()
