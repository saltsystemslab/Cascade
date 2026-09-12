#!/usr/bin/env bash
# Tests for tokenize-corpus.py's iter_records(): the JSONL and Parquet read
# paths, exercised without a tokenizer so they run anywhere pyarrow does.
#
#   PYTHON=/path/to/python3 bash dataset/test-records.sh
#
# The case that matters most is a missing or wrongly-typed text column. It must
# raise EAGERLY -- before tokenize_file() opens its output -- because a part
# file that exists is a part the resume logic treats as done, and concat accepts
# it. A wrong --text-field would otherwise produce a full set of empty parts and
# an array silently missing every document, with all the other checks passing.
set -uo pipefail

PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON" - "$HERE" <<'PY'
import gzip, importlib.util, json, os, sys, tempfile, types

here = sys.argv[1]
spec = importlib.util.spec_from_file_location(
    "tokcorpus", os.path.join(here, "tokenize-corpus.py"))
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

d = tempfile.mkdtemp()
npass = nfail = 0

def ok(msg):
    global npass; npass += 1; print(f"  ok    {msg}")

def bad(msg, detail=""):
    global nfail; nfail += 1
    print(f"  FAIL  {msg}")
    if detail:
        print(f"        {detail}")

def want(msg, got, exp):
    ok(msg) if got == exp else bad(msg, f"got {got!r} want {exp!r}")

def collect(path, field="text", bs=2):
    return list(tc.iter_records(path, field, bs))

print("iter_records() tests")

# ---------------------------------------------------------------- JSONL ----
p = os.path.join(d, "a.jsonl")
with open(p, "w") as f:
    f.write('{"text":"one"}\n')
    f.write('\n')                                  # blank line -> skipped
    f.write('   \n')                               # whitespace -> skipped
    f.write('{"text":"two"}\n')
    f.write('not json at all\n')                   # malformed -> skipped
    f.write('{"other":"x"}\n')                     # missing field -> None
    f.write('{"text":""}\n')                       # empty -> falsy
    f.write('{"text":"three"}\n')
got = collect(p)
want("jsonl: blank and malformed lines skipped, missing field -> None",
     got, ["one", "two", None, "", "three"])
want("jsonl: the caller's falsy filter leaves only real documents",
     [t for t in got if t], ["one", "two", "three"])

pgz = os.path.join(d, "a.jsonl.gz")
with gzip.open(pgz, "wt") as f:
    f.write('{"text":"gz one"}\n{"text":"gz two"}\n')
want("jsonl.gz: transparently decompressed", collect(pgz), ["gz one", "gz two"])

# 8 lines in, but 3 never reach json.loads (two blank, one malformed), so a
# different field yields 5 values -- one per PARSED record, not per line.
want("jsonl: --text-field selects another column",
     collect(p, field="other"), [None, None, "x", None, None])

# -------------------------------------------------------------- parquet ----
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("  SKIP  parquet cases (pyarrow not installed)")
    print(f"\n{npass} passed, {nfail} failed")
    raise SystemExit(1 if nfail else 0)

q = os.path.join(d, "b.parquet")
pq.write_table(pa.table({"text": ["p one", "p two", None, "", "p three"],
                         "id":   [1, 2, 3, 4, 5]}), q)
got = collect(q)
want("parquet: text column read, SQL null -> None",
     got, ["p one", "p two", None, "", "p three"])
want("parquet: the caller's falsy filter leaves only real documents",
     [t for t in got if t], ["p one", "p two", "p three"])

# Many row groups + a small batch size: the streaming path must not drop or
# duplicate rows across group boundaries.
q2 = os.path.join(d, "c.parquet")
pq.write_table(pa.table({"text": [f"doc{i}" for i in range(1000)]}),
               q2, row_group_size=37)
got = collect(q2, bs=13)
want("parquet: 1000 rows over 27 row groups, batch 13 -> exact round trip",
     got, [f"doc{i}" for i in range(1000)])

# --- the eager failures ----------------------------------------------------
try:
    tc.iter_records(q, "content", 2)
    bad("parquet: a missing column must raise")
except KeyError as e:
    ok("parquet: a missing column raises KeyError")
    if "resume logic" in str(e):
        ok("...and the message explains why an empty part is dangerous")
    else:
        bad("message does not explain the empty-part hazard", str(e))
except Exception as e:
    bad("parquet: missing column raised the wrong type", repr(e))

q3 = os.path.join(d, "d.parquet")
pq.write_table(pa.table({"text": [b"\x00\x01", b"\x02"]}), q3)
try:
    tc.iter_records(q3, "text", 2)
    bad("parquet: a binary text column must raise")
except TypeError:
    ok("parquet: a non-string text column raises TypeError")
except Exception as e:
    bad("parquet: binary column raised the wrong type", repr(e))

# Eagerness itself: the error must arrive on the call, not on first next().
raised_on_call = False
try:
    tc.iter_records(q, "nope", 2)
except KeyError:
    raised_on_call = True
want("parquet: validation is EAGER, not deferred to the first next()",
     raised_on_call, True)

q4 = os.path.join(d, "e.parquet")
with open(q4, "wb") as f:
    f.write(b"PAR1" + b"\x00" * 64)              # header only, no footer
try:
    tc.iter_records(q4, "text", 2)
    bad("parquet: a footerless file must raise")
except Exception:
    ok("parquet: a truncated file raises instead of yielding nothing")

print(f"\n{npass} passed, {nfail} failed")
raise SystemExit(1 if nfail else 0)
PY
