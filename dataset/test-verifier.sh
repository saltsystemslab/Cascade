#!/usr/bin/env bash
# Self-contained negative tests for verifier.py. Generates its own tiny fixtures,
# so it needs nothing but numpy and runs in about a second.
#
#   PYTHON=/path/to/python3 bash dataset/test-verifier.sh
#
# Every case asserts an EXIT STATUS, because that is what invariant 7's
# "verify before deleting raw" gate depends on. A verifier that prints "FAIL"
# and exits 0 is worse than no verifier: it makes an automated gate silently
# permissive. 0 = all checks passed, 1 = a check failed, 2 = usage/config error.
set -uo pipefail

PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFIER="$HERE/verifier.py"
FIX="$(mktemp -d)"
trap 'rm -rf "$FIX"' EXIT

"$PYTHON" - "$FIX" <<'PY'
import numpy as np, os, sys
d = sys.argv[1]
SEP = 0xFFFF
docs = [np.arange(100 + 10 * i, 110 + 10 * i, dtype="<u2") for i in range(5)]

trailing = []                      # this pipeline's convention: SEP after each doc
for doc in docs:
    trailing.extend(doc.tolist())
    trailing.append(SEP)
np.asarray(trailing, dtype="<u2").tofile(os.path.join(d, "good-llama2.tokenized.bin"))
np.asarray(trailing, dtype="<u2").tofile(os.path.join(d, "untagged.tokenized.bin"))

prepend = []                       # infini-gram's dolmasample convention
for doc in docs:
    prepend.append(SEP)
    prepend.extend(doc.tolist())
np.asarray(prepend, dtype="<u2").tofile(os.path.join(d, "prepend-olmo.tokenized.bin"))

# 32000 is >= the Llama-2 vocab but < 65535: the exact shape of real corruption,
# and invisible to every check except the max-non-separator-id one.
np.asarray([100, 200, SEP, 32000, 300, SEP],
           dtype="<u2").tofile(os.path.join(d, "overvocab-llama2.tokenized.bin"))

open(os.path.join(d, "empty-llama2.tokenized.bin"), "wb").close()
with open(os.path.join(d, "odd-llama2.tokenized.bin"), "wb") as f:
    f.write(b"\x01\x02\x03")       # 3 bytes: not 2-byte aligned
PY

pass=0
fail=0

check() {
    local want="$1" name="$2"
    shift 2
    local out got
    out="$("$PYTHON" "$VERIFIER" "$@" 2>&1)"
    got=$?
    if [ "$got" = "$want" ]; then
        pass=$((pass + 1))
        printf '  ok    exit %s  %s\n' "$got" "$name"
    else
        fail=$((fail + 1))
        printf '  FAIL  exit %s (wanted %s)  %s\n' "$got" "$want" "$name"
        echo "$out" | sed 's/^/          /'
    fi
}

echo "verifier.py negative tests  (python: $PYTHON)"

# --- the happy paths --------------------------------------------------------
check 0 "valid array, doc + token counts asserted" \
    --tokenizer llama2 --expect-docs 5 --expect-tokens 50 "$FIX/good-llama2.tokenized.bin"
check 0 "vocab bound inferred from the -llama2 filename tag" \
    --expect-docs 5 "$FIX/good-llama2.tokenized.bin"
check 0 "--quick skips the document count" \
    --quick --tokenizer llama2 "$FIX/good-llama2.tokenized.bin"

# --- the checks that must actually fire ------------------------------------
check 1 "document count off by one reports a delta" \
    --expect-docs 4 "$FIX/good-llama2.tokenized.bin"
check 1 "token count off by one" \
    --expect-tokens 49 "$FIX/good-llama2.tokenized.bin"
check 1 "id at the Llama-2 vocab bound is corruption" \
    "$FIX/overvocab-llama2.tokenized.bin"
check 0 "...but the same array is clean under the OLMo bound" \
    --vocab 50280 "$FIX/overvocab-llama2.tokenized.bin"
check 1 "odd byte count fails cleanly, no numpy traceback" \
    "$FIX/odd-llama2.tokenized.bin"
check 1 "empty file fails cleanly" \
    "$FIX/empty-llama2.tokenized.bin"
check 1 "avg tokens/doc outside an explicit band" \
    --tokenizer llama2 --avg-range 20:30 "$FIX/good-llama2.tokenized.bin"

# --- separator conventions are not interchangeable -------------------------
check 0 "prepend convention accepted with --sep-first" \
    --sep-first "$FIX/prepend-olmo.tokenized.bin"
check 1 "prepend convention rejected without it" \
    "$FIX/prepend-olmo.tokenized.bin"
check 1 "trailing-separator array rejected under --sep-first" \
    --sep-first "$FIX/good-llama2.tokenized.bin"

# --- config errors stay distinguishable from bad arrays (exit 2, not 1) ----
check 2 "untagged filename with no --vocab/--tokenizer" \
    "$FIX/untagged.tokenized.bin"
check 2 "nonexistent path" \
    --tokenizer llama2 "$FIX/does-not-exist.bin"

# --- the durable record ----------------------------------------------------
check 0 "--tsv appends a row on pass" \
    --tokenizer llama2 --tsv "$FIX/corpora.tsv" --label smoke "$FIX/good-llama2.tokenized.bin"
if [ "$(grep -c '^smoke' "$FIX/corpora.tsv" 2>/dev/null || echo 0)" = 1 ]; then
    pass=$((pass + 1)); echo "  ok            corpora.tsv row written with a header"
else
    fail=$((fail + 1)); echo "  FAIL          corpora.tsv row missing"
fi
check 1 "--tsv writes nothing when the array fails" \
    --expect-docs 4 --tsv "$FIX/corpora.tsv" --label bad "$FIX/good-llama2.tokenized.bin"
if grep -q '^bad' "$FIX/corpora.tsv" 2>/dev/null; then
    fail=$((fail + 1)); echo "  FAIL          a FAILing array got recorded"
else
    pass=$((pass + 1)); echo "  ok            no row for the failing array"
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
