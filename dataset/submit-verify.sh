#!/usr/bin/env bash
# Submit a token-array verification as a 1-CPU batch job.
#
# A full scan reads the whole array (up to 341 GB) at multiple GB/s, which does
# not belong on a shared login node -- and the login nodes SIGKILL it anyway.
# The job's exit status is the gate: 0 all-pass, 1 a check failed, 2 config
# error. That is what makes `rm -rf raw` safe to hang off afterok.
#
# Usage:
#   ARRAY=/scratch/$USER/corpora/arxiv/arxiv.tokenized.bin \
#   TOKENIZER=olmo EXPECT_DOCS=1558306 ./submit-verify.sh
#
# Optional env:
#   TOKENIZER     olmo|llama2|gpt2       (default: the tag in the filename)
#   VOCAB         explicit max id + 1    (overrides TOKENIZER; for untagged arrays)
#   EXPECT_DOCS   exact document count   -- the tokenizer-invariant check
#   EXPECT_TOKENS exact content tokens   (rarely a known-good figure)
#   SEP_FIRST     set for prepend-convention arrays (dolmasample)
#   QUICK         set to skip the full scan (structure only, seconds)
#   SHA256        set to also hash the file (a second full read)
#   TSV           corpora.tsv to append to on PASS
#                 (default $ROOT/corpora.tsv when ROOT is set)
#   LABEL         label for the tsv row   (default: the filename stem)
#   MEM           job memory             (default 8G; peak is ~1 GB/chunk)
#   TIME          walltime               (default 02:00:00; 341 GB took minutes)
#   PART          partition              (default short)
#   EXCLUDE       nodes to avoid         (default d4067 -- see below)
#   PYTHON        interpreter, pinned by absolute path
set -euo pipefail
: "${ARRAY:?set ARRAY to the token-array path}"

MEM="${MEM:-8G}"
TIME="${TIME:-02:00:00}"
PART="${PART:-short}"

# d4067 is the pinned benchmark node. Its cache policy is a recorded measurement
# constant with only ~118 GiB of page cache, so streaming a 341 GB array through
# it would silently corrupt a timing. Dataset work never runs there.
EXCLUDE="${EXCLUDE:-d4067}"

if [ -z "${TSV:-}" ] && [ -n "${ROOT:-}" ]; then TSV="$ROOT/corpora.tsv"; fi

# Pin the interpreter: capture the absolute path of the python3 active *now*
# (after your `module load`) so a batch node that resets PATH cannot fall back
# to a system python with different site-packages.
PYTHON="${PYTHON:-$(command -v python3 || true)}"
[ -n "$PYTHON" ] || { echo "no python3 on PATH -- 'module load python/3.13.5' (or set PYTHON=/abs/path/python3) before submitting" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="$SCRIPT_DIR/verify.sbatch"

EXPORTS="ARRAY=$ARRAY,VER_DIR=$SCRIPT_DIR,PYTHON=$PYTHON"
for v in TOKENIZER VOCAB EXPECT_DOCS EXPECT_TOKENS SEP_FIRST QUICK SHA256 TSV LABEL; do
    eval "val=\${$v:-}"
    [ -n "$val" ] && EXPORTS="$EXPORTS,$v=$val"
done

# Chain on another job so the whole sequence runs unattended. This is what makes
# the pipeline independent of any local machine staying awake: once submitted,
# Slurm advances it. afterok specifically means a FAILED predecessor leaves this
# pending on DependencyNeverSatisfied rather than running on a bad artifact.
#   DEPEND=afterok:12345 ./<this script>
DEPEND="${DEPEND:-}"
DEPFLAG=""
[ -n "$DEPEND" ] && DEPFLAG="--dependency=$DEPEND"

NAME="verify-$(basename "$ARRAY" .tokenized.bin)"
job=$(sbatch --parsable $DEPFLAG \
    --partition="$PART" --job-name="$NAME" \
    --exclude="$EXCLUDE" \
    --cpus-per-task=1 --mem="$MEM" --time="$TIME" \
    --export="ALL,$EXPORTS" "$SB")

echo "verify: job $job  ($(basename "$ARRAY"))"
echo "python: $PYTHON"
echo "log:    slurm-$NAME-$job.out"
echo "gate:   sacct -j $job -X -o JobID,State,ExitCode,Elapsed  (exit 0 = every check passed)"
[ -n "$DEPEND" ] && echo "depends:    $DEPEND (runs only if the predecessor succeeds)"
