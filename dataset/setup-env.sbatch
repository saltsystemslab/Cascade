#!/usr/bin/env bash
#SBATCH --job-name=setup-env
#SBATCH --output=slurm-%x-%j.out
#
# Build the pipeline's Python environment and warm the HF tokenizer cache.
# Not run directly -- submit via submit-setup-env.sh.
#
# This is a JOB rather than a login-node command because the login nodes kill
# anything substantial: `python3 -m venv` was SIGKILLed there, as was
# `import transformers`. Compute nodes on the `short` partition do have network
# (that is where the download jobs run), so pip and the HF fetch work fine here.
#
# The warm step is what makes the tokenize array work at all: it runs with
# HF_HUB_OFFLINE=1 and loads purely from $HF_HOME, so a cold cache fails every
# task in seconds -- and that failure looks exactly like the wrong-Python
# failure, which is why this job prints both the interpreter and the cache path.
#
# Env: VENV, BASE_PYTHON (required); HF_HOME, FRESH, TOKENIZERS (optional).
set -euo pipefail
: "${VENV:?VENV not set (submit via submit-setup-env.sh)}"
: "${BASE_PYTHON:?BASE_PYTHON not set (submit via submit-setup-env.sh)}"

export HF_HOME="${HF_HOME:-/scratch/${USER:-$(id -un)}/hf}"
TOKENIZERS="${TOKENIZERS:-allenai/OLMo-7B-hf meta-llama/Llama-2-7b-hf}"

echo "[setup] node      $(hostname)"
echo "[setup] base      $BASE_PYTHON ($("$BASE_PYTHON" -V 2>&1))"
echo "[setup] venv      $VENV"
echo "[setup] HF_HOME   $HF_HOME"

if [ -n "${FRESH:-}" ] && [ -d "$VENV" ]; then
    echo "[setup] FRESH set -- removing the existing venv"
    rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "[setup] creating venv"
    "$BASE_PYTHON" -m venv "$VENV"
fi

PY="$VENV/bin/python"
"$PY" -m pip install -q -U pip setuptools wheel

# A deliberately clean venv rather than --system-site-packages: the module's
# numpy/tokenizers could shift under a long campaign, and every array in it must
# be built by the same stack to stay comparable.
#
# transformers does NOT pull torch, and must not -- the tokenizers are all that
# is needed here and torch would add gigabytes for nothing.
echo "[setup] installing packages"
"$PY" -m pip install -q -U \
    numpy \
    "huggingface_hub[hf_xet]" \
    transformers \
    tokenizers \
    sentencepiece \
    pyarrow \
    zstandard

echo "[setup] installed:"
"$PY" - <<'PYEOF'
import importlib
for m in ("numpy", "huggingface_hub", "transformers", "tokenizers",
          "sentencepiece", "pyarrow", "zstandard"):
    try:
        mod = importlib.import_module(m)
        print(f"    {m:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"    {m:18s} FAILED: {e}")
try:
    import torch  # noqa: F401
    print("    WARNING: torch got pulled in; it is not needed and is huge")
except ImportError:
    print("    torch              absent (correct)")
PYEOF

# --- warm the tokenizer cache ------------------------------------------------
# Each tokenizer is fetched separately and its vocab size printed, because that
# number is the verifier's corruption bound: an array is garbage if any
# non-separator id reaches it. Better to read it off the real tokenizer than to
# trust a constant.
echo "[setup] warming tokenizers into $HF_HOME"
rc=0
for t in $TOKENIZERS; do
    if "$PY" - "$t" <<'PYEOF'
import sys
from transformers import AutoTokenizer
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name, use_fast=True)
n = len(tok)
print(f"    {name}: vocab {n}", flush=True)
if n > 0xFFFF:
    raise SystemExit(f"vocab {n} does not fit below the 0xFFFF separator")
PYEOF
    then :; else
        rc=1
        echo "    $t: FAILED -- if it is gated (Llama-2), accept the licence at" >&2
        echo "        hf.co/$t and put a token in \$HF_HOME/token or HF_TOKEN" >&2
    fi
done

echo
echo "[setup] done. Pass this to every job:"
echo "    PYTHON=$VENV/bin/python"
echo "    HF_HOME=$HF_HOME"
exit $rc
