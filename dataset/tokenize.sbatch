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

# sbatch copies this script to the node spool, so BASH_SOURCE can't locate the
# repo. submit-tokenize.sh (not spooled) passes the real dir as TOK_DIR.
TOK_DIR="${TOK_DIR:?TOK_DIR not set (submit via submit-tokenize.sh)}"
TOK="$TOK_DIR/tokenize-corpus.py"

if [ "${MODE:-shard}" = concat ]; then
    echo "[concat] stitching $OUT.parts -> $OUT"
    python3 "$TOK" --concat --input-dir "$INPUT_DIR" --output "$OUT" --glob "$GLOB"
else
    : "${N:?N not set (submit via submit-tokenize.sh)}"
    : "${SLURM_ARRAY_TASK_ID:?run this as a job array (submit via submit-tokenize.sh)}"
    echo "[shard ${SLURM_ARRAY_TASK_ID}/${N}] $INPUT_DIR -> $OUT.parts"
    python3 "$TOK" \
        --shard "${SLURM_ARRAY_TASK_ID}/${N}" \
        --input-dir "$INPUT_DIR" --output "$OUT" \
        --tokenizer "$TOKENIZER" --glob "$GLOB" \
        --workers "${SLURM_CPUS_PER_TASK:-1}"
fi
