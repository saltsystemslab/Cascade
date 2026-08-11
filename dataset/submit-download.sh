#!/usr/bin/env bash
# Submit the download-only step as a detached 1-CPU batch job, so a large pull
# survives logout and login-node time/CPU limits. Downloading is network/IO-bound,
# so 1 CPU + many parallel streams (DL_PROC) is the right shape -- and a 1-CPU job
# backfills fast even when wide allocations don't.
#
# Usage:
#   ROOT=/scratch/$USER/corpora SUBSET=github ./submit-download.sh redpajama
#   ROOT=/scratch/$USER/corpora INCLUDE="en/c4-train.00*-of-01024.json.gz" ./submit-download.sh c4
#
# Optional env:
#   SUBSET    RedPajama subset            (default arxiv; ignored by other datasets)
#   DL_PROC   parallel download streams   (default 8; 16 for a fat uplink)
#   INCLUDE   hf download glob            (c4/pile)
#   NFILES    cap on URL-list files       (redpajama/dolma)
#   VER       dolma version               (default v1_7)
#   MEM       job memory                  (default 16G; lower for wget subsets, raise for Xet/c4)
#   TIME      walltime                    (default 08:00:00)
#   PART      Slurm partition             (default short)
#
# The download resumes on resubmit (wget -c / hf Xet), so if the job hits the
# walltime just run this again -- completed files are skipped.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: ROOT=<dir> [SUBSET=..] ./submit-download.sh {c4|pile|redpajama|dolma}" >&2
    exit 1
fi
DATASET="$1"
ROOT="${ROOT:?set ROOT to a working dir for raw data (no default)}"
SUBSET="${SUBSET:-arxiv}"
DL_PROC="${DL_PROC:-8}"
MEM="${MEM:-16G}"
TIME="${TIME:-08:00:00}"
PART="${PART:-short}"

# Display/job label: redpajama is split by subset, others are not.
LABEL="$DATASET"
if [ "$DATASET" = redpajama ]; then LABEL="$DATASET-$SUBSET"; fi

# This launcher is not spooled, so it knows the real dataset/ dir; pass it to the
# batch job as DL_DIR (sbatch runs a spooled copy that can't self-locate).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="$SCRIPT_DIR/download.sbatch"

EXPORTS="DL_DIR=$SCRIPT_DIR,DATASET=$DATASET,ROOT=$ROOT,SUBSET=$SUBSET,DL_PROC=$DL_PROC"
if [ -n "${INCLUDE:-}" ]; then EXPORTS="$EXPORTS,INCLUDE=$INCLUDE"; fi
if [ -n "${NFILES:-}"  ]; then EXPORTS="$EXPORTS,NFILES=$NFILES";  fi
if [ -n "${VER:-}"     ]; then EXPORTS="$EXPORTS,VER=$VER";        fi

job=$(sbatch --parsable \
    --partition="$PART" --job-name="dl-$LABEL" \
    --cpus-per-task=1 --mem="$MEM" --time="$TIME" \
    --export="ALL,$EXPORTS" "$SB")

echo "download: job $job  ($LABEL, 1 CPU, $DL_PROC streams, up to $TIME)"
echo "watch:    squeue -j $job"
echo "log:      slurm-dl-$LABEL-$job.out"
echo
echo "when it finishes, the exact tokenize command is printed in the log:"
echo "    grep 'next:' slurm-dl-$LABEL-$job.out"
