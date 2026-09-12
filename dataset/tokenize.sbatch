#!/usr/bin/env bash
#SBATCH --job-name=tokenize
#SBATCH --output=slurm-%x-%A_%a.out
#
# Two-phase tokenize as scheduler-friendly 1-CPU jobs. Not run directly --
# submit it via submit-tokenize.sh, which sets the env vars below and wires the
# array + dependent concat.
#
#   Phase 1 (array):  each task tokenizes its stride of files into $OUT.parts/
#                     (MODE unset; strides by SLURM_ARRAY_TASK_ID over N tasks,
#                      N passed in by submit-tokenize.sh)
#   Phase 2 (concat): MODE=concat, no array; stitches parts into $OUT
#
# Env: INPUT_DIR, OUT (required); TOKENIZER, GLOB (optional).
set -euo pipefail
: "${INPUT_DIR:?INPUT_DIR not set}"
: "${OUT:?OUT not set}"
TOKENIZER="${TOKENIZER:-allenai/OLMo-7B-hf}"
GLOB="${GLOB:-*.jsonl*}"

# Interpreter is pinned by submit-tokenize.sh (absolute path); fall back to python3
# only for a hand-run. Load the tokenizer from the on-scratch cache and stay off the
# network -- compute nodes are typically offline and the cache is warmed at setup.
PYTHON="${PYTHON:-python3}"
export HF_HOME="${HF_HOME:-/scratch/${USER:-$(id -un)}/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# sbatch copies this script to the node spool, so BASH_SOURCE can't locate the
# repo. submit-tokenize.sh (not spooled) passes the real dir as TOK_DIR.
TOK_DIR="${TOK_DIR:?TOK_DIR not set (submit via submit-tokenize.sh)}"
TOK="$TOK_DIR/tokenize-corpus.py"

if [ "${MODE:-shard}" = concat ]; then
    echo "[concat] stitching $OUT.parts -> $OUT"
    "$PYTHON" "$TOK" --concat --input-dir "$INPUT_DIR" --output "$OUT" --glob "$GLOB"
else
    : "${N:?N not set (submit via submit-tokenize.sh)}"
    : "${SLURM_ARRAY_TASK_ID:?run this as a job array (submit via submit-tokenize.sh)}"
    echo "[shard ${SLURM_ARRAY_TASK_ID}/${N}] $INPUT_DIR -> $OUT.parts  (python: $PYTHON)"
    # BATCH bounds the Python-object peak inside flush(): a batch holds every
    # document's ids AND attention_mask as individual Python ints. At the default
    # 1000 that is fine for short documents, but arxiv averages ~18,000 tokens per
    # document, so 1000 of them is ~40M int objects -- enough to SIGKILL a task at
    # --mem=8G, which is exactly how two arxiv shards died. Lower it for
    # long-document corpora; throughput barely moves, because the batch exists to
    # amortise per-call overhead and that is already negligible at 18k tokens.
    "$PYTHON" "$TOK" \
        --shard "${SLURM_ARRAY_TASK_ID}/${N}" \
        --input-dir "$INPUT_DIR" --output "$OUT" \
        --tokenizer "$TOKENIZER" --glob "$GLOB" \
        --batch-size "${BATCH:-1000}" \
        --workers "${SLURM_CPUS_PER_TASK:-1}"
fi
