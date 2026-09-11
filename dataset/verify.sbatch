#!/usr/bin/env bash
#SBATCH --job-name=verify
#SBATCH --output=slurm-%x-%j.out
#
# Verify one token array. Not run directly -- submit it via submit-verify.sh,
# which sets the env vars below and pins the interpreter.
#
# This exists as a batch job rather than a login-node command for two reasons:
# a full scan is a multi-GB/s read of up to 341 GB (antisocial on a shared login
# node), and the login nodes kill large processes -- a 56 GB array's verify was
# SIGKILLed there while a 341 GB one submitted as a job was fine. Its non-zero
# exit is also what lets a later `rm -rf raw` hang off --dependency=afterok.
#
# Env: ARRAY (required); TOKENIZER, EXPECT_DOCS, EXPECT_TOKENS, SEP_FIRST,
#      QUICK, SHA256, TSV, LABEL, VOCAB (all optional).
set -euo pipefail
: "${ARRAY:?ARRAY not set (path to the token array)}"

PYTHON="${PYTHON:-python3}"

# sbatch runs a spooled copy from /var/spool/slurmd, so BASH_SOURCE cannot find
# the repo; submit-verify.sh (not spooled) passes the real dir as VER_DIR.
VER_DIR="${VER_DIR:?VER_DIR not set (submit via submit-verify.sh)}"
VERIFIER="$VER_DIR/verifier.py"
[ -f "$VERIFIER" ] || { echo "no verifier.py at $VERIFIER" >&2; exit 2; }

args=("$ARRAY")
[ -n "${VOCAB:-}" ]         && args+=(--vocab "$VOCAB")
[ -n "${TOKENIZER:-}" ]     && args+=(--tokenizer "$TOKENIZER")
[ -n "${EXPECT_DOCS:-}" ]   && args+=(--expect-docs "$EXPECT_DOCS")
[ -n "${EXPECT_TOKENS:-}" ] && args+=(--expect-tokens "$EXPECT_TOKENS")
[ -n "${SEP_FIRST:-}" ]     && args+=(--sep-first)
[ -n "${QUICK:-}" ]         && args+=(--quick)
[ -n "${SHA256:-}" ]        && args+=(--sha256)
[ -n "${TSV:-}" ]           && args+=(--tsv "$TSV")
[ -n "${LABEL:-}" ]         && args+=(--label "$LABEL")

echo "[verify] $ARRAY  (python: $PYTHON)"
echo "[verify] $VERIFIER ${args[*]}"
"$PYTHON" "$VERIFIER" "${args[@]}"
