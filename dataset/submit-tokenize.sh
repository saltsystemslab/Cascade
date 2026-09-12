#!/usr/bin/env bash
# Submit the two-phase tokenize as scheduler-friendly 1-CPU jobs:
#   Phase 1: a job array of N tasks; each strides through the sorted file list
#            and tokenizes its share into $OUT.parts/  (backfills fast: 1 CPU each)
#   Phase 2: a single concat job, dependent on the whole array succeeding
#
# Usage:
#   INPUT_DIR=/scratch/$USER/corpora/redpajama-arxiv/raw \
#   OUT=/scratch/$USER/corpora/redpajama-arxiv/redpajama-arxiv.tokenized.bin \
#   ./submit-tokenize.sh
#
# Optional env:
#   N         array size / number of tasks   (default 32)
#   TOKENIZER HF tokenizer                    (default allenai/OLMo-7B-hf)
#   GLOB      input file pattern              (default *.jsonl*)
#   MEM       memory per 1-CPU task           (default 4G)
#   TIME      walltime per task               (default 04:00:00)
#   PART      Slurm partition                 (default short)
#   PYTHON    interpreter for the tasks       (default: the python3 active at submit)
#   BATCH     documents per tokenizer call    (default 1000; use ~100 for corpora
#                                              with very long documents, e.g. arxiv)
#   DEPEND    afterok:<jobid> to chain on     (unset)
#
# N need not match the file count: with striding, each task just handles
# ceil(files/N) files. Pick N for how much parallelism you want, not per file.
set -euo pipefail
: "${INPUT_DIR:?set INPUT_DIR to the raw JSONL dir}"
: "${OUT:?set OUT to the target token-array path}"
N="${N:-32}"
TOKENIZER="${TOKENIZER:-allenai/OLMo-7B-hf}"
GLOB="${GLOB:-*.jsonl*}"
MEM="${MEM:-4G}"
TIME="${TIME:-04:00:00}"
PART="${PART:-short}"

# Pin the interpreter: capture the absolute path of the python3 active *now* (after
# your `module load`) and pass it to every task, so a batch node that resets PATH to
# a system python can't run the tasks under the wrong version + stale site-packages.
PYTHON="${PYTHON:-$(command -v python3 || true)}"
[ -n "$PYTHON" ] || { echo "no python3 on PATH -- 'module load python/3.13.5' (or set PYTHON=/abs/path/python3) before submitting" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="$SCRIPT_DIR/tokenize.sbatch"
EXPORTS="INPUT_DIR=$INPUT_DIR,OUT=$OUT,TOKENIZER=$TOKENIZER,GLOB=$GLOB,N=$N,TOK_DIR=$SCRIPT_DIR,PYTHON=$PYTHON"
# BATCH: documents per tokenizer call. Lower it for long-document corpora --
# arxiv at ~18,000 tokens/doc SIGKILLed two shards at the default 1000 with
# --mem=8G, because a batch holds every id AND attention_mask as a Python int.
# BATCH=100 cuts that peak ~10x at negligible throughput cost.
[ -n "${BATCH:-}" ] && EXPORTS="$EXPORTS,BATCH=$BATCH"
# DEPEND lets a whole corpus chain unattended (see submit-verify.sh).
DEPEND="${DEPEND:-}"
DEPFLAG=""
[ -n "$DEPEND" ] && DEPFLAG="--dependency=$DEPEND"

phase1=$(sbatch --parsable $DEPFLAG \
    --partition="$PART" --array="0-$((N - 1))" \
    --cpus-per-task=1 --mem="$MEM" --time="$TIME" \
    --export="ALL,$EXPORTS" "$SB")
echo "Phase 1 (tokenize): array job $phase1  ($N tasks x 1 CPU)"

phase2=$(sbatch --parsable \
    --partition="$PART" \
    --cpus-per-task=1 --mem="$MEM" --time="$TIME" \
    --dependency="afterok:$phase1" \
    --export="ALL,$EXPORTS,MODE=concat" "$SB")
echo "Phase 2 (concat):   job $phase2  (runs after $phase1 succeeds)"
echo
echo "python: $PYTHON"
echo "watch:  squeue -j $phase1,$phase2"
echo "output: $OUT   (parts in $OUT.parts/ until concat cleans them up)"
