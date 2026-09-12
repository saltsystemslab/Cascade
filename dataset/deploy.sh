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
# TOK_DIR *at runtime*, i.e. from the live git checkout. Bash reads a script
# incrementally from an open descriptor rather than slurping it, so a checkout
# that changes under a running job can leave it reading from a stale offset.
# A 33-minute wikipedia fetch completed its download, then stopped before the
# manifest and the pre-split, and still exited 0 -- a silent half-run, which is
# the worst possible failure mode. The only difference on the successful re-run
# was that nothing pulled underneath it.
#
# So: jobs must never reference a directory that `git pull` can touch. A
# per-commit snapshot is immutable by construction, and names the commit that
# produced every artifact, which is the provenance the campaign wants anyway.
#
#   DEPLOY_ROOT   where snapshots live   (default /scratch/$USER/deploy)
#
# Informational output goes to stderr; ONLY the path goes to stdout.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEPLOY_ROOT="${DEPLOY_ROOT:-/scratch/${USER:-$(id -un)}/deploy}"

git rev-parse --git-dir >/dev/null 2>&1 || {
    echo "deploy: $REPO is not a git checkout" >&2; exit 1; }

# A dirty dataset/ would make the snapshot name a lie: the directory would claim
# a commit whose contents it does not have.
if [ -n "$(git status --porcelain -- dataset/)" ]; then
    echo "deploy: dataset/ has uncommitted changes, so a commit-named snapshot" >&2
    echo "would misdescribe its own contents. Commit or stash first:" >&2
    git status --short -- dataset/ >&2
    exit 1
fi

SHA="$(git rev-parse --short HEAD)"
DEST="$DEPLOY_ROOT/$SHA"

if [ -d "$DEST" ]; then
    echo "deploy: reusing $DEST" >&2
else
    # tmp + atomic rename, so $DEST exists only when it is complete -- the same
    # discipline the part files and the pre-split use, for the same reason.
    rm -rf "$DEST.tmp"
    mkdir -p "$DEST.tmp"
    cp dataset/* "$DEST.tmp/" 2>/dev/null || true
    {
        echo "commit:     $(git rev-parse HEAD)"
        echo "short:      $SHA"
        echo "branch:     $(git rev-parse --abbrev-ref HEAD)"
        echo "subject:    $(git log -1 --pretty=%s)"
        echo "deployed:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "deployed_by: ${USER:-$(id -un)}@$(hostname)"
    } > "$DEST.tmp/DEPLOY-INFO"
    chmod -R a-w "$DEST.tmp" 2>/dev/null || true
    mv "$DEST.tmp" "$DEST"
    echo "deploy: created $DEST ($(git log -1 --pretty=%s))" >&2
fi

echo "$DEST"
