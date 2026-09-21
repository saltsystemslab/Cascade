
# Corner-case design to reproduce missing-terminator bug.
# Emits a small token-array that reproduces the missing-terminator bug: a text
# drawn from a tiny alphabet excluding token 0, followed by `ell` tokens of
# value `0`. The `ell` trailing suffixes share the all-zero `ell`-prefix key the
# tail-padding synthesizes, so they tie at initial ranking, regardless of how
# the text gets partitioned, and none of them has a sender at generation 1. As
# 0 occurs nowhere else, they are also the `ell` lexicographically smallest
# suffixes, shortest first. The tiny alphabet keeps ties alive for several
# doubling generations rather than retiring everything at generation 0.


import argparse
import array
import random
import sys

ap = argparse.ArgumentParser(description="Generate the no-terminator corner-case corpus.")
ap.add_argument("out", help="output path for the LE-uint16 token array")
ap.add_argument("--n", type=int, default=1 << 24, help="body length in tokens (default 16Mi)")
ap.add_argument("--ell", type=int, default=8, help="trailing zero-run length; the build's `ell` (default 8)")
ap.add_argument("--sigma", type=int, default=4, help="body alphabet size; ids are 1..sigma (default 4)")
ap.add_argument("--seed", type=int, default=0x9E3779B9, help="RNG seed")
a = ap.parse_args()

rng = random.Random(a.seed)
T = array.array("H", (rng.randrange(1, a.sigma + 1) for _ in range(a.n)))
T.extend(array.array("H", [0]) * a.ell)
if sys.byteorder == "big":
    T.byteswap()

with open(a.out, "wb") as f:
    T.tofile(f)

n = a.n + a.ell
print(f"{a.out}: {n:,} tokens ({2 * n / 1e6:.1f} MB); body ids 1..{a.sigma}, {a.ell} trailing zeros")
print(f"expected SA[0, {a.ell}) = {list(range(n - 1, n - a.ell - 1, -1))}")
