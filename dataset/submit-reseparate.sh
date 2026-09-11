#!/usr/bin/env bash
# Submit a separator-convention conversion (trailing -> leading) as a 1-CPU job.
#
#   ARRAY=/scratch/$USER/corpora/arxiv/arxiv.tokenized.bin ./submit-reseparate.sh
#
# "Prepend a separator" moves every byte, so this is a full read + write, not an
# edit: ~2x the array in I/O. That belongs in a job, not on a login node, which
# SIGKILLs large processes anyway.
#
# The conversion is copy-then-verify-then-swap. It proves output[2:] is
# byte-identical to input[:-2] and that the size is unchanged BEFORE replacing
# anything, which is what makes the swap safe -- and since the transform is a
# right-rotation by one uint16, that proof also establishes that the document
# and token counts are preserved.
#
# Optional env:
#   KEEP    set to hard-link the original to <ARRAY>.seplast first
#   DRYRUN  set to write <ARRAY>.sepfirst and stop, without swapping
#   MEM     job memory      (default 4G; the script streams in 16 MiB blocks)
#   TIME    walltime        (default 04:00:00; 341 GB is ~2x341 GB of I/O)
#   PART    partition       (default short)
#   EXCLUDE nodes to avoid  (default d4067, the pinned benchmark node)
#   PYTHON  interpreter, pinned by absolute path
set -euo pipefail
: "${ARRAY:?set ARRAY to the token-array path}"

MEM="${MEM:-4G}"
TIME="${TIME:-04:00:00}"
PART="${PART:-short}"
EXCLUDE="${EXCLUDE:-d4067}"

PYTHON="${PYTHON:-$(command -v python3 || true)}"
[ -n "$PYTHON" ] || { echo "no python3 on PATH -- 'module load python/3.13.5' first" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="resep-$(basename "$ARRAY" .tokenized.bin)"

# reseparate.py is pure stdlib, so --wrap is enough; there is no sbatch body to
# spool and therefore no need for the TOK_DIR/DL_DIR self-location dance.
FLAGS=""
[ -n "${DRYRUN:-}" ] || FLAGS="$FLAGS --in-place"
[ -n "${KEEP:-}" ]   && FLAGS="$FLAGS --keep"

job=$(sbatch --parsable \
    --partition="$PART" --job-name="$NAME" \
    --exclude="$EXCLUDE" \
    --cpus-per-task=1 --mem="$MEM" --time="$TIME" \
    --output="slurm-%x-%j.out" \
    --wrap="$PYTHON $SCRIPT_DIR/reseparate.py$FLAGS $ARRAY")

echo "reseparate: job $job  ($(basename "$ARRAY"))"
echo "python:     $PYTHON"
echo "log:        slurm-$NAME-$job.out"
echo "gate:       sacct -j $job -X -o JobID,State,ExitCode,Elapsed"
echo
echo "after it completes, re-verify (the default is now the LEADING convention)"
echo "and refresh the corpora.tsv row -- the file's sha256 has changed."
