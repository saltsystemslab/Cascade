#!/usr/bin/env bash
# Tests for presplit.sh. Builds its own tiny fixtures; runs in about a second.
#
#   bash dataset/test-presplit.sh
#
# MUST run on Linux: presplit.sh needs `split -n l/N`, `--additional-suffix`,
# `find -printf` and `stat -c`, none of which exist on BSD/macOS. On a dev Mac
# this script is expected to fail at the first split.
#
# The property that actually matters is `cat parts/*.jsonl == source`, because
# it proves two things at once: that no line was lost or duplicated, and that
# the glob's lexicographic order matches the numeric order that
# build_file_list()'s sorted() assumes. Get that order wrong and the array is
# silently permuted -- a corruption no other check in the pipeline would catch.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESPLIT_SH="$HERE/presplit.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
ok()   { pass=$((pass + 1)); printf '  ok    %s\n' "$1"; }
bad()  { fail=$((fail + 1)); printf '  FAIL  %s\n' "$1"; [ $# -gt 1 ] && printf '        %s\n' "$2"; }
want() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "got [$2] want [$3]"; fi; }

# mono <dir> <nlines> -- a raw/-shaped dir holding one monolithic .jsonl
mono() {
    mkdir -p "$1"
    seq 1 "$2" | sed 's/.*/{"text":"document number & with some filler"}/' > "$1/wiki.jsonl"
    echo "https://example.invalid/wiki.jsonl" > "$1/urls.txt"   # must NOT be split
}

echo "presplit.sh tests"

# --- 1. the normal case ------------------------------------------------------
D="$WORK/a/raw"; mono "$D" 1000
SRCB=$(stat -c %s "$D/wiki.jsonl")
out=$( TOK_INPUT="$D" TOK_GLOB='*.jsonl*' SPLITS=8 bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1; echo "$TOK_INPUT|$TOK_GLOB"' )
want "input dir + glob rewritten to the parts dir" "$out" "$D/parts|*.jsonl"
want "8 parts produced" "$(find "$D/parts" -type f -name '*.jsonl' | wc -l)" "8"
cat "$D"/parts/*.jsonl > "$WORK/rejoined"
if cmp -s "$WORK/rejoined" "$D/wiki.jsonl"; then
    ok "cat parts/*.jsonl is byte-identical to the source (order + completeness)"
else
    bad "rejoined parts differ from the source"
fi
SUMB=$(find "$D/parts" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
want "part byte sum equals the source" "$SUMB" "$SRCB"
want "the monolith is kept by default" "$([ -f "$D/wiki.jsonl" ] && echo yes || echo no)" "yes"
want "urls.txt was not treated as corpus input" "$([ -f "$D/urls.txt" ] && echo yes || echo no)" "yes"
want "no parts.tmp left behind" "$([ -d "$D/parts.tmp" ] && echo yes || echo no)" "no"

# --- 2. idempotency: a resubmitted download job must not re-split ------------
before=$(find "$D/parts" -type f | wc -l)
out=$( TOK_INPUT="$D" TOK_GLOB='*.jsonl*' SPLITS=8 bash -c '. '"$PRESPLIT_SH"' 2>&1' )
case "$out" in *reusing*) ok "a re-run reports reusing the existing split" ;;
               *) bad "a re-run did not detect the existing parts/" "$out" ;; esac
want "re-run changed nothing" "$(find "$D/parts" -type f | wc -l)" "$before"

# --- 3. suffix width: the latent bug when SPLITS > 99 -----------------------
D2="$WORK/b/raw"; mono "$D2" 5000
TOK_INPUT="$D2" TOK_GLOB='*.jsonl*' SPLITS=128 bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1'
want "128 parts with a 3-wide suffix" "$(find "$D2/parts" -type f -name '*.jsonl' | wc -l)" "128"
want "first part is _000 (not _00)" "$([ -f "$D2/parts/wiki_part_000.jsonl" ] && echo yes || echo no)" "yes"
cat "$D2"/parts/*.jsonl > "$WORK/rejoined2"
if cmp -s "$WORK/rejoined2" "$D2/wiki.jsonl"; then
    ok "128-part glob order still matches numeric order"
else
    bad "128-part rejoin differs -- lexicographic vs numeric order mismatch"
fi

# --- 4. fewer lines than SPLITS: empty parts must not survive ---------------
D3="$WORK/c/raw"; mono "$D3" 3
TOK_INPUT="$D3" TOK_GLOB='*.jsonl*' SPLITS=64 bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1'
n3=$(find "$D3/parts" -type f | wc -l)
if [ "$n3" -le 3 ] && [ "$n3" -ge 1 ]; then
    ok "3 lines / 64 splits -> $n3 non-empty parts, empties dropped"
else
    bad "expected 1-3 non-empty parts, got $n3"
fi
want "no zero-byte part survived" "$(find "$D3/parts" -type f -size 0 | wc -l)" "0"

# --- 5. multi-file input is left alone -------------------------------------
D4="$WORK/d/raw"; mkdir -p "$D4"
for i in 1 2 3; do echo '{"text":"x"}' > "$D4/shard_$i.jsonl"; done
out=$( TOK_INPUT="$D4" TOK_GLOB='*.jsonl*' bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1; echo "$TOK_INPUT|$TOK_GLOB"' )
want "3 files: input dir untouched" "$out" "$D4|*.jsonl*"
want "no parts dir created" "$([ -d "$D4/parts" ] && echo yes || echo no)" "no"

( TOK_INPUT="$D4" TOK_GLOB='*.jsonl*' PRESPLIT=on bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1' )
want "PRESPLIT=on with 3 files is an error" "$?" "1"

# --- 6. compressed single file cannot be line-split ------------------------
D5="$WORK/e/raw"; mkdir -p "$D5"
echo '{"text":"x"}' | gzip > "$D5/wiki.jsonl.gz"
out=$( TOK_INPUT="$D5" TOK_GLOB='*.jsonl*' bash -c '. '"$PRESPLIT_SH"' 2>&1' )
case "$out" in *COMPRESSED*) ok "compressed input warns instead of splitting" ;;
               *) bad "no warning for a compressed single file" "$out" ;; esac
want "compressed input: no parts dir" "$([ -d "$D5/parts" ] && echo yes || echo no)" "no"

( TOK_INPUT="$D5" TOK_GLOB='*.jsonl*' PRESPLIT=on bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1' )
want "PRESPLIT=on with a compressed file is an error" "$?" "1"

# --- 7. opt-outs and input validation --------------------------------------
D6="$WORK/f/raw"; mono "$D6" 100
out=$( TOK_INPUT="$D6" TOK_GLOB='*.jsonl*' PRESPLIT=off bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1; echo "$TOK_INPUT"' )
want "PRESPLIT=off leaves the input dir alone" "$out" "$D6"

D7="$WORK/g/raw"; mono "$D7" 100
TOK_INPUT="$D7" TOK_GLOB='*.jsonl*' SPLITS=4 DROP_MONOLITH=1 bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1'
want "DROP_MONOLITH=1 removes the source" "$([ -f "$D7/wiki.jsonl" ] && echo yes || echo no)" "no"
want "DROP_MONOLITH=1 still produced parts" "$(find "$D7/parts" -type f | wc -l)" "4"

D8="$WORK/h/raw"; mono "$D8" 10
( TOK_INPUT="$D8" TOK_GLOB='*.jsonl*' SPLITS=notanumber bash -c '. '"$PRESPLIT_SH"' >/dev/null 2>&1' )
want "a non-numeric SPLITS is rejected" "$?" "1"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
