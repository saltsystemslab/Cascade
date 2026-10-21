#!/usr/bin/env bash
# Tests for reseparate.py (trailing -> leading document separators).
#
#   PYTHON=/path/to/python3 bash dataset/test-reseparate.sh
#
# reseparate.py is pure stdlib, so unlike presplit.sh this runs anywhere.
#
# The transform is a right-rotation by one uint16 -- the separator prepended is
# exactly the one dropped -- so size, separator count and token count are all
# preserved. The tests assert that directly, plus the two refusals that matter:
# an already-converted array must be a no-op (double-converting would eat a real
# token off the end), and an array whose both ends are separators is ambiguous
# and must not be guessed at.
set -uo pipefail

PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEP="$HERE/reseparate.py"
FIX="$(mktemp -d)"
trap 'rm -rf "$FIX"' EXIT

"$PYTHON" - "$FIX" <<'PY'
import os, sys, struct
d = sys.argv[1]
SEP = 0xFFFF

def w(name, vals):
    with open(os.path.join(d, name), "wb") as f:
        f.write(struct.pack("<%dH" % len(vals), *vals))

docs = [list(range(100 + 10 * i, 110 + 10 * i)) for i in range(5)]

trail = []
for doc in docs:
    trail += doc + [SEP]
w("trail.bin", trail)

lead = []
for doc in docs:
    lead += [SEP] + doc
w("lead.bin", lead)

# Ambiguous: trailing convention whose FIRST document is empty. Reads equally
# well as a leading-convention array, and the two differ by a document.
w("ambiguous.bin", [SEP] + docs[0] + [SEP])

w("odd.bin", [])
with open(os.path.join(d, "odd.bin"), "wb") as f:
    f.write(b"\x01\x02\x03")
PY

pass=0
fail=0
ok()   { pass=$((pass + 1)); printf '  ok    %s\n' "$1"; }
bad()  { fail=$((fail + 1)); printf '  FAIL  %s\n' "$1"; [ $# -gt 1 ] && printf '        %s\n' "$2"; }
want() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "got [$2] want [$3]"; fi; }

echo "reseparate.py tests  (python: $PYTHON)"

# --- the conversion ---------------------------------------------------------
cp "$FIX/trail.bin" "$FIX/work.bin"
before=$(wc -c < "$FIX/work.bin" | tr -d ' ')
"$PYTHON" "$RESEP" --in-place "$FIX/work.bin" > "$FIX/log" 2>&1
want "conversion exits 0" "$?" "0"
after=$(wc -c < "$FIX/work.bin" | tr -d ' ')
want "size is preserved exactly" "$after" "$before"
if cmp -s "$FIX/work.bin" "$FIX/lead.bin"; then
    ok "output is byte-identical to a natively leading array"
else
    bad "output differs from the expected leading array"
fi
want "no .sepfirst temp left behind" "$([ -f "$FIX/work.bin.sepfirst" ] && echo yes || echo no)" "no"

"$PYTHON" - "$FIX/work.bin" "$FIX/trail.bin" <<'PY'
import struct, sys
SEP = 0xFFFF
def rd(p):
    b = open(p, "rb").read()
    return list(struct.unpack("<%dH" % (len(b) // 2), b))
new, old = rd(sys.argv[1]), rd(sys.argv[2])
assert new[0] == SEP,  "new does not start with a separator"
assert new[-1] != SEP, "new still ends with a separator"
assert new.count(SEP) == old.count(SEP), "separator count changed"
assert sorted(new) == sorted(old), "element multiset changed"
print("  ok    separator count and element multiset unchanged")
PY

# --- idempotency: double-converting would eat a real token -----------------
out=$( "$PYTHON" "$RESEP" --in-place "$FIX/work.bin" 2>&1 )
want "a second conversion exits 0" "$?" "0"
case "$out" in *"already LEADING"*) ok "a second conversion is a no-op" ;;
               *) bad "already-converted array not detected" "$out" ;; esac
if cmp -s "$FIX/work.bin" "$FIX/lead.bin"; then
    ok "the no-op really changed nothing"
else
    bad "the second run modified the array"
fi

# --- refusals ---------------------------------------------------------------
out=$( "$PYTHON" "$RESEP" --in-place "$FIX/ambiguous.bin" 2>&1 )
want "both-ends-separator is refused" "$?" "2"
case "$out" in *"cannot tell"*) ok "the refusal explains the ambiguity" ;;
               *) bad "unexpected message" "$out" ;; esac

cp "$FIX/ambiguous.bin" "$FIX/forced.bin"
"$PYTHON" "$RESEP" --in-place --force last "$FIX/forced.bin" > /dev/null 2>&1
want "--force last converts the ambiguous case" "$?" "0"

"$PYTHON" "$RESEP" "$FIX/odd.bin" > /dev/null 2>&1
want "an odd-sized file is refused" "$?" "2"

# --- non-destructive by default --------------------------------------------
cp "$FIX/trail.bin" "$FIX/keep.bin"
"$PYTHON" "$RESEP" "$FIX/keep.bin" > /dev/null 2>&1
want "without --in-place the input is untouched" \
    "$(cmp -s "$FIX/keep.bin" "$FIX/trail.bin" && echo same || echo changed)" "same"
want "without --in-place the output is left alongside" \
    "$([ -f "$FIX/keep.bin.sepfirst" ] && echo yes || echo no)" "yes"
out=$( "$PYTHON" "$RESEP" "$FIX/keep.bin" 2>&1 )
want "an existing output file is refused, not clobbered" "$?" "2"

cp "$FIX/trail.bin" "$FIX/backup.bin"
"$PYTHON" "$RESEP" --in-place --keep "$FIX/backup.bin" > /dev/null 2>&1
want "--keep preserves the original as .seplast" \
    "$(cmp -s "$FIX/backup.bin.seplast" "$FIX/trail.bin" && echo same || echo differs)" "same"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
