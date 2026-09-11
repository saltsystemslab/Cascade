#!/usr/bin/env bash
# Pre-split a monolithic JSONL corpus into line-aligned parts.
#
# SOURCED, not executed -- it rewrites TOK_INPUT and TOK_GLOB in the caller:
#     . "$SCRIPT_DIR/presplit.sh"
#
# Some RedPajama subsets ship as ONE plain .jsonl (wiki.jsonl is 120.1 GB,
# stackexchange.jsonl is 80.0 GB). One file means one worker in the tokenize
# array, i.e. a single-core pass over a 27B-token corpus. Splitting on line
# boundaries gives the array something to spread across.
#
# It is a separate file so it can be tested directly (see test-presplit.sh) --
# every GNU-specific thing here (`split -n l/N`, `--additional-suffix`,
# `find -printf`, `stat -c`) is absent on BSD/macOS, so this cannot be
# exercised on a dev Mac at all.
#
# Reads:  TOK_INPUT, TOK_GLOB (required); PRESPLIT, SPLITS, DROP_MONOLITH,
#         ROOT, LABEL, OUT_BIN (optional)
# Writes: TOK_INPUT, TOK_GLOB  -- pointed at the parts dir when a split happens
#
# CALL THIS ONLY BEFORE TOKENIZE PHASE 1. Parts in $OUT_BIN.parts/ are keyed on
# the global index over the sorted file list, so changing that list under a
# half-finished part set makes concat stitch a silently wrong array. Running
# inside the download job -- which finishes before tokenize is submitted -- is
# what makes that impossible rather than merely discouraged.

PRESPLIT="${PRESPLIT:-auto}"          # auto | on | off
SPLITS="${SPLITS:-64}"

case "$SPLITS" in
    ''|*[!0-9]*) echo "presplit: SPLITS must be a positive integer (got '$SPLITS')" >&2; return 1 2>/dev/null || exit 1 ;;
esac

if [ "$PRESPLIT" = off ]; then
    echo ">> pre-split: disabled (PRESPLIT=off)"
elif [ -d "$TOK_INPUT/parts" ]; then
    # parts/ only ever appears via an atomic rename of a byte-verified
    # parts.tmp/, so its existence means a COMPLETE split. Re-runs are no-ops,
    # and a killed split can never be mistaken for a finished one.
    _np=$(find "$TOK_INPUT/parts" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
    echo ">> pre-split: reusing $TOK_INPUT/parts ($_np parts, already split)"
    TOK_INPUT="$TOK_INPUT/parts"; TOK_GLOB="*.jsonl"
else
    _n=$(find "$TOK_INPUT" -maxdepth 1 -type f -name "$TOK_GLOB" | wc -l)
    _src=$(find "$TOK_INPUT" -maxdepth 1 -type f -name "$TOK_GLOB" | sort | head -1)

    if [ "$_n" -ne 1 ]; then
        if [ "$PRESPLIT" = on ]; then
            echo "presplit: PRESPLIT=on, but $_n files match '$TOK_GLOB' in $TOK_INPUT." >&2
            echo "Pre-splitting applies only to a single monolithic file." >&2
            return 1 2>/dev/null || exit 1
        fi
        echo ">> pre-split: not needed ($_n files match '$TOK_GLOB')"
    else
        case "$_src" in
        *.jsonl)
            _base="${_src##*/}"
            _prefix="${_base%%.*}_part_"
            _suf=${#SPLITS}     # uniform width => lexicographic order == numeric
            echo ">> pre-split: $_base -> $SPLITS line-aligned parts (-a $_suf)"
            rm -rf "$TOK_INPUT/parts.tmp"
            mkdir -p "$TOK_INPUT/parts.tmp"

            # --additional-suffix so the parts match submit-tokenize.sh's default
            # GLOB="*.jsonl*". Extensionless parts match nothing, which is exactly
            # how the manual recipe in the runbook failed.
            split -d -a "$_suf" -n "l/$SPLITS" --additional-suffix=.jsonl \
                  "$_src" "$TOK_INPUT/parts.tmp/$_prefix"

            # `split -n l/N` emits empty files when the input has fewer than N
            # lines. An empty part is worse than no part: skip-if-exists counts
            # it as done and concat accepts it.
            find "$TOK_INPUT/parts.tmp" -maxdepth 1 -type f -size 0 -delete

            # -n l/N is a byte-exact partition, so this equality is exact rather
            # than a heuristic -- it is the whole reason the split is trustworthy.
            _srcb=$(stat -c %s "$_src")
            _sumb=$(find "$TOK_INPUT/parts.tmp" -maxdepth 1 -type f -printf '%s\n' \
                    | awk '{s+=$1} END{print s+0}')
            _np=$(find "$TOK_INPUT/parts.tmp" -maxdepth 1 -type f | wc -l)

            if [ "$_srcb" != "$_sumb" ]; then
                echo "presplit FAILED: parts total $_sumb bytes != source $_srcb." >&2
                echo "Keeping both $_src and $TOK_INPUT/parts.tmp for inspection." >&2
                return 1 2>/dev/null || exit 1
            fi

            mv "$TOK_INPUT/parts.tmp" "$TOK_INPUT/parts"   # atomic: parts/ == complete
            echo ">> pre-split: $_np parts, $_sumb bytes == source (verified)"

            if [ -n "${ROOT:-}" ] && [ -n "${LABEL:-}" ] && [ -d "$ROOT/$LABEL" ]; then
                {   echo "# source:     $_base"
                    echo "# src_bytes:  $_srcb"
                    echo "# splits:     $SPLITS  (split -d -a $_suf -n l/$SPLITS)"
                    echo "# parts:      $_np non-empty, prefix $_prefix"
                    echo "# sum_bytes:  $_sumb  (== src_bytes, verified)"
                    echo "# split_at:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
                    echo "# note:       raw-manifest.tsv was written before this split,"
                    echo "#             so it lists the monolith, not the parts."
                } > "$ROOT/$LABEL/split-manifest.txt"
            fi

            # The monolith is KEPT by default. Deleting it would save 120 GB, but
            # scratch has ~346 TB free, so there is no reason to destroy a
            # verified source to buy space we are not short of -- and keeping it
            # leaves invariant 7 ("verify before deleting raw") untouched. It
            # also makes a resubmit free: wget -c sees a complete file and skips.
            if [ -n "${DROP_MONOLITH:-}" ]; then
                rm -f "$_src"
                echo ">> pre-split: removed $_base (byte sum verified; re-fetchable from urls.txt)"
            fi

            TOK_INPUT="$TOK_INPUT/parts"; TOK_GLOB="*.jsonl"
            echo ">> pre-split: re-splitting later with a different SPLITS also needs"
            echo "              'rm -rf ${OUT_BIN:-\$OUT}.parts' -- parts are keyed on file index"
            ;;
        *)
            _msg="single COMPRESSED input ($_src): 'split -n l/' cannot line-split a compressed stream. Decompress it in place (gunzip / zstd -d) and re-run, or set PRESPLIT=off to accept a single-core tokenize."
            if [ "$PRESPLIT" = on ]; then
                echo "presplit: $_msg" >&2
                return 1 2>/dev/null || exit 1
            fi
            echo ">> WARNING: pre-split skipped -- $_msg"
            ;;
        esac
    fi
fi

unset _n _np _src _base _prefix _suf _srcb _sumb _msg
