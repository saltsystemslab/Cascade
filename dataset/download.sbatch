#!/usr/bin/env bash
#SBATCH --job-name=download
#SBATCH --output=slurm-%x-%j.out
#
# Download-only step as a detached 1-CPU batch job (survives logout / login-node
# limits). Not run directly -- submit via submit-download.sh, which sets DL_DIR
# (the real dataset/ dir) and the DATASET/ROOT/... env below.
#
# Network/IO-bound: 1 CPU + many streams. The wrapped download-and-tokenize.sh
# reads ROOT/SUBSET/DL_PROC/INCLUDE/NFILES/VER from the environment (propagated
# via --export) and stops after fetching because SKIP_TOKENIZE is set here.
set -euo pipefail
: "${DATASET:?DATASET not set (submit via submit-download.sh)}"
: "${ROOT:?ROOT not set}"

# sbatch copies this script to the node spool, so BASH_SOURCE can't locate the
# repo. submit-download.sh (not spooled) passes the real dir as DL_DIR.
DL_DIR="${DL_DIR:?DL_DIR not set (submit via submit-download.sh)}"
DL="$DL_DIR/download-and-tokenize.sh"
[ -f "$DL" ] || { echo "download-and-tokenize.sh not found at $DL" >&2; exit 1; }

echo "[download] $DATASET  ROOT=$ROOT  SUBSET=${SUBSET:-} DL_PROC=${DL_PROC:-8}"
SKIP_TOKENIZE=1 bash "$DL" "$DATASET"
