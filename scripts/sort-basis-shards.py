#!/usr/bin/env python3
"""Sort SBD basis/determinant shard files into the order load_basis_from_files requires.

Since caop/basis.h v12 (upstream 8187b8d) load_basis_from_files no longer sorts
its input: it *checks* that each shard is sorted and calls MPI_Abort if not, so a
file that was fine before now aborts the run with

    load_basis_from_files rank=0: ERROR file not sorted: <file>

The error message in basis.h points at this script, which was not in the repo.

Order: `less_from_back` (framework/bit_manipulation.h) compares the LAST word
first, i.e. the most significant. The reader packs character j of a record at
bit index (total_bit_length-1-j), so word 0 holds the *rightmost* bit_length
characters and the highest word the leftmost. Comparing from the back is
therefore comparing the bitstring as one big binary integer -- which for
fixed-width records of '0'/'1' is exactly ASCII lexicographic order.

So the sort key is the line itself. bit_length never enters: the packing is
uniform, so no word boundary can reorder anything. --verify-int rechecks that
claim by also sorting on the parsed integer and comparing.

Usage:
    python3 scripts/sort-basis-shards.py FILE...          # in place
    python3 scripts/sort-basis-shards.py --check FILE...   # report only
    python3 scripts/sort-basis-shards.py --verify-int FILE...
"""
import argparse
import os
import sys


def read_records(path):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return [], None
    width = len(lines[0])
    for i, ln in enumerate(lines):
        if len(ln) != width:
            raise SystemExit(
                f"{path}:{i+1}: record width {len(ln)} != {width}; "
                "load_basis_from_file assumes fixed-width records")
        bad = set(ln) - {"0", "1"}
        if bad:
            raise SystemExit(
                f"{path}:{i+1}: unexpected character(s) {sorted(bad)}; "
                "expected '0' or '1'")
    return lines, width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true",
                    help="report sortedness/duplicates, do not modify")
    ap.add_argument("--verify-int", action="store_true",
                    help="cross-check the string order against integer order")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="do not deduplicate (the k-way merge dedups anyway)")
    args = ap.parse_args()

    status = 0
    for path in args.files:
        lines, width = read_records(path)
        if not lines:
            print(f"{path}: empty, nothing to do")
            continue

        ordered = sorted(lines)
        if args.verify_int:
            by_int = sorted(lines, key=lambda s: int(s, 2))
            if by_int != ordered:
                raise SystemExit(f"{path}: string order != integer order (bug)")

        already = lines == ordered
        ndup = len(lines) - len(set(lines))

        if args.check:
            print(f"{path}: {len(lines)} records, width {width}, "
                  f"sorted={'yes' if already else 'NO'}, duplicates={ndup}")
            if not already:
                status = 1
            continue

        out = ordered
        if not args.keep_duplicates and ndup:
            deduped, prev = [], None
            for ln in ordered:
                if ln != prev:
                    deduped.append(ln)
                prev = ln
            out = deduped

        if already and len(out) == len(lines):
            print(f"{path}: already sorted ({len(lines)} records), unchanged")
            continue

        tmp = path + ".sorting.tmp"
        with open(tmp, "w") as fh:
            fh.write("\n".join(out) + "\n")
        os.replace(tmp, path)
        print(f"{path}: sorted {len(lines)} -> {len(out)} records"
              + (f" ({ndup} duplicate(s) removed)" if len(out) != len(lines) else ""))

    return status


if __name__ == "__main__":
    sys.exit(main())
