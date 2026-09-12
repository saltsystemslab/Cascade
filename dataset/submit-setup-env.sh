#!/usr/bin/env bash
# Build the pipeline's Python venv and warm the HF tokenizer cache, as a job.
#
#   module load python/3.13.5
#   ./submit-setup-env.sh
#
# Why a job: the login nodes SIGKILL anything substantial -- `python3 -m venv`
# and `import transformers` were both killed there. Compute nodes on `short`
# have network, so pip and the HF fetch work.
#
# Optional env:
#   VENV        venv location      (default /scratch/$USER/venv-dataset)
#   HF_HOME     tokenizer cache    (default /scratch/$USER/hf)
#   FRESH       set to rebuild the venv from scratch
#   TOKENIZERS  space-separated    (default: OLMo + Llama-2)
#   MEM/TIME/PART/EXCLUDE/DEPEND   as elsewhere
#
# Llama-2 is GATED: accept the licence at hf.co/meta-llama/Llama-2-7b-hf and
# leave a token in $HF_HOME/token (`hf auth login`) or export HF_TOKEN before
# submitting. Without it that tokenizer fails and the job exits non-zero, having
# still installed everything and warmed the ungated ones.
set -euo pipefail

VENV="${VENV:-/scratch/$USER/venv-dataset}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
MEM="${MEM:-8G}"
TIME="${TIME:-01:00:00}"
PART="${PART:-short}"
EXCLUDE="${EXCLUDE:-d4067}"

# The venv is built from whichever python3 is active now, so `module load` before
# submitting -- exactly the same pinning discipline as submit-tokenize.sh, and
# for the same reason: a batch node resetting PATH must not pick a system python.
BASE_PYTHON="${BASE_PYTHON:-$(command -v python3 || true)}"
[ -n "$BASE_PYTHON" ] || { echo "no python3 on PATH -- 'module load python/3.13.5' first" >&2; exit 1; }

DEPEND="${DEPEND:-}"
DEPFLAG=""
[ -n "$DEPEND" ] && DEPFLAG="--dependency=$DEPEND"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="$SCRIPT_DIR/setup-env.sbatch"

EXPORTS="VENV=$VENV,BASE_PYTHON=$BASE_PYTHON,HF_HOME=$HF_HOME"
[ -n "${FRESH:-}" ]      && EXPORTS="$EXPORTS,FRESH=$FRESH"
[ -n "${TOKENIZERS:-}" ] && EXPORTS="$EXPORTS,TOKENIZERS=$TOKENIZERS"

job=$(sbatch --parsable $DEPFLAG \
    --partition="$PART" --job-name=setup-env \
    --exclude="$EXCLUDE" \
    --cpus-per-task=2 --mem="$MEM" --time="$TIME" \
    --export="ALL,$EXPORTS" "$SB")

echo "setup-env: job $job"
echo "base:      $BASE_PYTHON"
echo "venv:      $VENV"
echo "log:       slurm-setup-env-$job.out"
echo
echo "when it succeeds, every later job takes:  PYTHON=$VENV/bin/python"
