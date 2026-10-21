#!/usr/bin/env bash
# Append one token array onto another, in place, with a rollback.
#
#   BASE=.../c4.tokenized.bin TAIL=.../c4-tail-olmo.tokenized.bin \
#   EXPECT_BASE=341350676994 EXPECT_TAIL=8188738312 ./append-array.sh
#
# This completes a partially-built corpus without re-tokenizing what already
# exists -- C4's OLMo array covers shards 00000-00999, and the 24-shard tail is
# ~8 GB against 341 GB, so appending is ~40x cheaper than a full rebuild.
#
# WHY THE APPEND IS VALID. Under the leading-separator convention BASE ends in a
# real token id and TAIL begins with 0xFFFF, so BASE+TAIL is exactly the byte
# stream one run over all shards would have emitted. (It held under the old
# trailing convention too, for the mirror-image reason: BASE ended in a
# separator and TAIL began with a real token.) It also relies on sorted file
# order putting 01000-01023 strictly after 00999, which 5-digit zero-padded
# shard numbers guarantee.
#
# WHY IT IS RISKY, AND WHAT THAT BUYS. `cat >>` mutates the only copy of an
# expensive, already-verified artifact. An interrupted append leaves a file that
# is neither the old one nor the new one, and -- this is the part that bites --
# is indistinguishable from a complete one unless the original size was recorded
# BEFORE starting. So EXPECT_BASE and EXPECT_TAIL are mandatory: they are
# asserted first, the arithmetic result is asserted after, and any mismatch
# truncates straight back to EXPECT_BASE. Verify the result separately
# afterwards; size arithmetic is necessary, not sufficient.
set -euo pipefail

: "${BASE:?set BASE to the array to append onto (mutated in place)}"
: "${TAIL:?set TAIL to the array to append}"
: "${EXPECT_BASE:?set EXPECT_BASE to BASE's exact current size in bytes}"
: "${EXPECT_TAIL:?set EXPECT_TAIL to TAIL's exact size in bytes}"

[ -f "$BASE" ] || { echo "no such file: $BASE" >&2; exit 2; }
[ -f "$TAIL" ] || { echo "no such file: $TAIL" >&2; exit 2; }

sz() { stat -c %s "$1"; }

base_now=$(sz "$BASE")
tail_now=$(sz "$TAIL")
want=$((EXPECT_BASE + EXPECT_TAIL))

echo "base   $BASE"
echo "       $base_now bytes (expect $EXPECT_BASE)"
echo "tail   $TAIL"
echo "       $tail_now bytes (expect $EXPECT_TAIL)"
echo "result should be $want bytes"

if [ "$base_now" != "$EXPECT_BASE" ]; then
    if [ "$base_now" = "$want" ]; then
        echo "BASE is already $want bytes -- the append appears to have been done." >&2
        echo "Refusing to append twice. Verify it instead." >&2
        exit 3
    fi
    echo "BASE is $base_now, expected $EXPECT_BASE. Refusing: either the wrong" >&2
    echo "file, or a previous append left it in an unknown state." >&2
    exit 1
fi
[ "$tail_now" = "$EXPECT_TAIL" ] || {
    echo "TAIL is $tail_now, expected $EXPECT_TAIL. Refusing." >&2; exit 1; }

# Both are uint16 arrays; an odd size anywhere means a truncated write upstream
# and the concatenation would silently shift every element after the seam.
for f in "$BASE" "$TAIL"; do
    [ $(( $(sz "$f") % 2 )) -eq 0 ] || { echo "$f is not 2-byte aligned" >&2; exit 1; }
done

echo "appending..."
cat "$TAIL" >> "$BASE"

got=$(sz "$BASE")
if [ "$got" != "$want" ]; then
    echo "FAILED: result is $got bytes, wanted $want. Rolling back to $EXPECT_BASE." >&2
    truncate -s "$EXPECT_BASE" "$BASE"
    echo "rolled back; $BASE is $(sz "$BASE") bytes" >&2
    exit 1
fi

echo "[PASS] $BASE is now $got bytes ($((got / 2)) uint16 values)"
echo
echo "Size arithmetic only proves the bytes landed. Re-verify the array now --"
echo "for C4 the document count must be exactly 364,868,892."
