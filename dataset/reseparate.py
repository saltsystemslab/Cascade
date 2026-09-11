#!/usr/bin/env python3
"""Convert a token array from TRAILING to LEADING document separators.

    trailing (what this pipeline used to emit):  d1 SEP d2 SEP d3 SEP
    leading  (infini-gram's convention):         SEP d1 SEP d2 SEP d3

    python3 reseparate.py --in-place arxiv.tokenized.bin

The separator prepended is exactly the one dropped from the end, so the
transform is a right-rotation by one uint16: **the file size does not change,
the separator count does not change, and the content-token count does not
change.** That makes the correctness argument short -- if the output is the
same size as the input and output[2:] is byte-identical to input[:-2], then the
element multiset is unchanged and every derived count is preserved. No semantic
re-scan is needed, and this script proves exactly that before touching anything.

The transform is also its own inverse in cost (drop the leading separator,
append one), so converting is not a one-way door.

What is NOT free: "prepend to a file" moves every byte, so this is a full
read + write, not an edit. It is done copy-then-verify-then-swap, never in
place -- an interrupted in-place right-shift would leave an array that is
neither convention and cannot be recovered.

Note what the leading convention costs: a trailing-separator array had a free
truncation check, since its last element MUST be 65535. Under the leading
convention the last element is legitimately a real token id, so truncation is
no longer visible from the tail. The document count, the byte count and the
sha256 are what detect it instead.
"""

import argparse
import hashlib
import os
import sys

SEP = 0xFFFF
SEP_BYTES = SEP.to_bytes(2, "little")
BLOCK = 16 * 1024 * 1024


def read_ends(path):
    """First and last uint16 of the array, without mapping the whole file."""
    size = os.path.getsize(path)
    if size < 4 or size % 2:
        return None, None, size
    with open(path, "rb") as f:
        first = int.from_bytes(f.read(2), "little")
        f.seek(-2, os.SEEK_END)
        last = int.from_bytes(f.read(2), "little")
    return first, last, size


def detect(first, last):
    """Which separator convention the array is in, or None if undecidable."""
    if last == SEP and first != SEP:
        return "last"
    if first == SEP and last != SEP:
        return "first"
    if first == SEP and last == SEP:
        # Ambiguous: a trailing-convention array whose FIRST document is empty
        # also starts with a separator. Refusing is the only safe answer -- the
        # two readings differ by one document at each end.
        return None
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--output", help="write here (default: <path>.sepfirst)")
    ap.add_argument("--in-place", action="store_true",
                    help="os.replace the verified output onto the input")
    ap.add_argument("--force", choices=["last", "first"],
                    help="assert the input's convention when autodetection is "
                         "ambiguous (both ends are separators)")
    ap.add_argument("--keep", action="store_true",
                    help="with --in-place, keep the original as <path>.seplast")
    args = ap.parse_args()

    first, last, size = read_ends(args.path)
    if first is None:
        print(f"{args.path}: {size} bytes is not a uint16 array (odd or too short)",
              file=sys.stderr)
        return 2

    conv = args.force or detect(first, last)
    print(f"{args.path}\n  {size:,} bytes ({size / 1e9:.1f} GB)"
          f"   first={first} last={last}")

    if conv is None:
        print(f"  cannot tell which convention this is: both ends are {SEP}. A "
              f"trailing array whose first document is empty looks identical to a "
              f"leading one. Pass --force last|first.", file=sys.stderr)
        return 2

    if conv == "first":
        print("  already LEADING-separator; nothing to do.")
        return 0

    out = args.output or args.path + ".sepfirst"
    if os.path.exists(out):
        print(f"  {out} already exists; remove it or pass --output", file=sys.stderr)
        return 2

    print(f"  converting TRAILING -> LEADING  ->  {out}")

    # Write SEP, then all input bytes except the final separator. Hash the
    # payload as we go: this is sha256(input[:-2]).
    src_h = hashlib.sha256()
    remaining = size - 2
    try:
        with open(args.path, "rb") as fi, open(out, "wb") as fo:
            fo.write(SEP_BYTES)
            while remaining:
                block = fi.read(min(BLOCK, remaining))
                if not block:
                    raise IOError(f"input ended {remaining} bytes early")
                src_h.update(block)
                fo.write(block)
                remaining -= len(block)
            fo.flush()
            os.fsync(fo.fileno())
    except BaseException:
        # A partial output is worse than none: it is a valid-looking array.
        if os.path.exists(out):
            os.unlink(out)
        raise

    # --- verify before swapping -------------------------------------------
    new_size = os.path.getsize(out)
    if new_size != size:
        print(f"  FAIL size: {new_size:,} != {size:,} (the transform is "
              f"length-preserving, so this must match exactly)", file=sys.stderr)
        return 1

    nf, nl, _ = read_ends(out)
    if nf != SEP:
        print(f"  FAIL first element: {nf} != {SEP}", file=sys.stderr)
        return 1
    if nl == SEP:
        print(f"  FAIL last element is still {SEP}; the final separator was not "
              f"dropped", file=sys.stderr)
        return 1

    # Re-read the output and hash everything after the prepended separator. It
    # must equal the hash of the input minus its final separator -- which proves
    # the copy is byte-exact, hence that the element multiset (and so every
    # document and token count) is unchanged.
    dst_h = hashlib.sha256()
    with open(out, "rb") as f:
        f.seek(2)
        for block in iter(lambda: f.read(BLOCK), b""):
            dst_h.update(block)
    if dst_h.hexdigest() != src_h.hexdigest():
        print(f"  FAIL payload hash mismatch:\n    input[:-2]  {src_h.hexdigest()}"
              f"\n    output[2:]  {dst_h.hexdigest()}", file=sys.stderr)
        return 1

    print(f"  [PASS] size preserved: {new_size:,} bytes")
    print(f"  [PASS] first={nf}, last={nl}")
    print(f"  [PASS] output[2:] == input[:-2]  (sha256 {src_h.hexdigest()[:16]}...)")
    print("         -> element multiset unchanged, so document and token counts "
          "are provably preserved")

    if args.in_place:
        if args.keep:
            backup = args.path + ".seplast"
            os.link(args.path, backup)
            print(f"  original hard-linked to {backup}")
        os.replace(out, args.path)      # atomic; no window with a missing file
        print(f"  swapped into place: {args.path}")
        print("  NOTE: re-run verifier.py (it now expects LEADING by default) and "
              "refresh the corpora.tsv row -- the file's sha256 has changed.")
    else:
        print(f"  left at {out}; re-run with --in-place, or move it yourself")

    return 0


if __name__ == "__main__":
    sys.exit(main())
