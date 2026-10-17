#!/usr/bin/env bash
# Snapshot dataset/ into an immutable, commit-named directory, and print its
# path on stdout so a launcher can be invoked from it:
#
#   D=$(bash ~/GPU-SA-dataset/dataset/deploy.sh) && ROOT=... bash "$D/submit-download.sh" c4
#
# WHY THIS EXISTS -- a job silently truncated, and this is the fix.
#
# download.sbatch and tokenize.sbatch are spooled by Slurm, but the scripts they
# invoke (download-and-tokenize.sh, tokenize-corpus.py) are read from DL_DIR /
# TOK_DIR *at runtime*, i.e. out of the live git checkout. Bash reads a script
# incrementally from an open descriptor rather than slurping it, so a checkout
# that changes under a running job can leave it reading from a stale offset. A
# 33-minute wikipedia fetch completed its download, then stopped before the
# manifest and the pre-split, and still exited 0 -- a silent half-run, which is
# the worst failure mode available. The re-run, identical except that nothing
# pulled underneath it, worked.
#
# So jobs must never reference a directory that `git pull` can touch. A
# per-commit snapshot is immutable by construction and names the commit behind
# every artifact, which is the provenance the campaign wants anyway.
#
# The snapshot is built with `git archive` from a REF, never from the working
# tree, so this is safe to run while jobs are executing out of that checkout --
# it reads the object store and does not write to the worktree at all. Pair it
# with `git fetch` (also worktree-safe) instead of `git pull`:
#
#   git -C ~/GPU-SA-dataset fetch -q origin
#   D=$(REF=origin/dataset-pipeline bash ~/GPU-SA-dataset/dataset/deploy.sh)
#
#   REF           commit-ish to snapshot   (default HEAD)
#   DEPLOY_ROOT   where snapshots live     (default /scratch/$USER/deploy)
#   REPO          repo to read from        (default: this script's repo)
#
# Informational output goes to stderr; ONLY the path goes to stdout.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REF="${REF:-HEAD}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/scratch/${USER:-$(id -un)}/deploy}"

git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || {
    echo "deploy: $REPO is not a git checkout" >&2; exit 1; }

SHA="$(git -C "$REPO" rev-parse --short "$REF")" || {
    echo "deploy: no such ref: $REF" >&2; exit 1; }

# Only a warning: the snapshot is built from the REF, so uncommitted work is
# simply absent rather than misrepresented. Worth saying out loud, though --
# "I edited that" is the likeliest reason a snapshot behaves unexpectedly.
if [ "$REF" = HEAD ] && [ -n "$(git -C "$REPO" status --porcelain -- dataset/)" ]; then
    echo "deploy: NOTE dataset/ has uncommitted changes; the snapshot is built" >&2
    echo "        from the commit and will NOT contain them:" >&2
    git -C "$REPO" status --short -- dataset/ >&2
fi

DEST="$DEPLOY_ROOT/$SHA"

if [ -d "$DEST" ]; then
    echo "deploy: reusing $DEST" >&2
else
    # tmp + atomic rename, so $DEST exists only when complete -- the same
    # discipline the part files and the pre-split use, for the same reason.
    rm -rf "$DEST.tmp"
    mkdir -p "$DEST.tmp"
    git -C "$REPO" archive "$REF" dataset/ | tar -x --strip-components=1 -C "$DEST.tmp"
    {
        echo "commit:      $(git -C "$REPO" rev-parse "$REF")"
        echo "short:       $SHA"
        echo "ref:         $REF"
        echo "subject:     $(git -C "$REPO" log -1 --pretty=%s "$REF")"
        echo "deployed:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "deployed_by: ${USER:-$(id -un)}@$(hostname)"
    } > "$DEST.tmp/DEPLOY-INFO"
    chmod -R a-w "$DEST.tmp" 2>/dev/null || true
    mv "$DEST.tmp" "$DEST"
    echo "deploy: created $DEST ($(git -C "$REPO" log -1 --pretty=%s "$REF"))" >&2
fi

echo "$DEST"
