#!/usr/bin/env bash
# Raw-text -> single token-array pipeline for infini-gram corpora.
#
# Per dataset:  (1) fetch a subset of raw JSONL,
#               (2) tokenize to ONE uint16-LE array via tokenize-corpus.py,
#               (3) verify token / separator counts.
#
# Usage:   ROOT=/scratch/$USER/corpora ./download-and-tokenize.sh {c4|pile|redpajama|dolma}
# Tune via env vars, e.g.:
#   ROOT=/scratch/$USER/corpora INCLUDE="en/c4-train.000*-of-01024.json.gz" \
#       TOKENIZER=allenai/OLMo-7B-hf WORKERS=64 ./download-and-tokenize.sh c4
#   # NB: C4 shard numbers are 5 digits -> 000* = 100 shards (~35 GB),
#   #     00* = 1000 shards (~340 GB, ~all of C4), * = all 1024.
#
# Prereqs:  pip install -U "huggingface_hub[hf_xet]" transformers tokenizers \
#                          numpy zstandard sentencepiece
#           (sentencepiece only needed for the Llama-2 tokenizer)
set -euo pipefail

# Absolute dir of this script, so its sibling tokenizer is found regardless of
# the caller's $PWD (both files live together in the repo's dataset/ dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------- CONFIG -------------------------------------
if [ $# -lt 1 ]; then
    echo "usage: ROOT=<dir> ./download-and-tokenize.sh {c4|pile|redpajama|dolma}" >&2
    exit 1
fi
DATASET="$1"
ROOT="${ROOT:?set ROOT to a working dir for raw data + the token array (no default)}"
TOKENIZER="${TOKENIZER:-allenai/OLMo-7B-hf}"  # HF tokenizer; keep ONE across corpora (-hf: no trust_remote_code, no torch)
WORKERS="${WORKERS:-$(nproc)}"                # tokenizer worker procs (CPU-bound; ~1/core)
DL_PROC="${DL_PROC:-8}"                        # parallel download streams (network-bound)
TOK_SCRIPT="${TOK_SCRIPT:-$SCRIPT_DIR/tokenize-corpus.py}"   # sibling tokenizer script

export HF_XET_HIGH_PERFORMANCE=1              # faster Xet downloads (replaces hf_transfer)

# Per-corpus label; redpajama is further split by subset so arxiv/github/etc.
# get isolated raw dirs and output arrays instead of clobbering each other.
LABEL="$DATASET"
if [ "$DATASET" = redpajama ]; then
    SUBSET="${SUBSET:-arxiv}"                  # RedPajama-V1 subset (see redpajama branch)
    LABEL="$DATASET-$SUBSET"
fi

RAW_DIR="$ROOT/$LABEL/raw"
OUT_BIN="$ROOT/$LABEL/${LABEL}.tokenized.bin"
mkdir -p "$RAW_DIR" "$(dirname "$OUT_BIN")"

# ------------------------------- FETCH --------------------------------------
case "$DATASET" in
  c4)
    # 1024 en train shards, ~319 MB each (~195M tokens/shard).
    #   5-digit shards: 000* -> 100 (~35 GB) | 00* -> 1000 (~340 GB, ~all) | * -> all 1024
    INCLUDE="${INCLUDE:-en/c4-train.000*-of-01024.json.gz}"
    hf download allenai/c4 --repo-type dataset \
        --include "$INCLUDE" --local-dir "$RAW_DIR"
    TOK_INPUT="$RAW_DIR/en"; TOK_GLOB="c4-train.*.json.gz"
    ;;

  pile)
    # monology/pile-uncopyrighted. Confirm exact file names first with:
    #   hf datasets ls monology/pile-uncopyrighted --tree -h
    INCLUDE="${INCLUDE:-train/0[0-3].jsonl.zst}"   # ~4 shards by default
    hf download monology/pile-uncopyrighted --repo-type dataset \
        --include "$INCLUDE" --local-dir "$RAW_DIR"
    TOK_INPUT="$RAW_DIR/train"; TOK_GLOB="*.jsonl.zst"
    ;;

  redpajama)
    # RedPajama-V1 raw text is served from data.together.xyz.
    # Multi-file subsets parallelize across cores in tokenize-corpus.py:
    #   arxiv ~28B (~56 GB) | github ~59B (~118 GB) | c4 ~175B | common_crawl ~878B
    # AVOID as-is: wikipedia (~24B) and stackexchange (~20B) are each a SINGLE
    # monolithic .jsonl -> one worker, single-core tokenize. Pre-split first
    # (e.g. `split -d -n l/64 file.jsonl part_`) if you need them. book: defunct.
    SUBSET="${SUBSET:-arxiv}"; NFILES="${NFILES:-all}"   # SUBSET resolved in CONFIG; kept for clarity
    curl -s https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt \
        | grep "/$SUBSET/" > "$RAW_DIR/urls.txt"
    if [ "$NFILES" != "all" ]; then
        head -n "$NFILES" "$RAW_DIR/urls.txt" > "$RAW_DIR/urls.head"
        mv "$RAW_DIR/urls.head" "$RAW_DIR/urls.txt"
    fi
    TOTAL=$(wc -l < "$RAW_DIR/urls.txt")
    echo ">> fetching $TOTAL files ($SUBSET) into $RAW_DIR, $DL_PROC parallel streams"
    xargs -P"$DL_PROC" -I{} sh -c \
        'wget -q -c -P "$1" "$2" && echo "${2##*/}"' _ "$RAW_DIR" {} < "$RAW_DIR/urls.txt" \
        | awk -v t="$TOTAL" '{printf ">>   [%d/%d] %s\n", NR, t, $0; fflush()}'
    TOK_INPUT="$RAW_DIR"; TOK_GLOB="*.jsonl*"
    ;;

  dolma)
    # allenai/dolma ships per-file URL lists at urls/<version>.txt
    VER="${VER:-v1_7}"; NFILES="${NFILES:-20}"
    hf download allenai/dolma "urls/${VER}.txt" --repo-type dataset --local-dir "$RAW_DIR"
    mkdir -p "$RAW_DIR/data"
    SEL="$RAW_DIR/dl_urls.txt"
    head -n "$NFILES" "$RAW_DIR/urls/${VER}.txt" > "$SEL"
    TOTAL=$(wc -l < "$SEL")
    echo ">> fetching $TOTAL files (dolma $VER) into $RAW_DIR/data, $DL_PROC parallel streams"
    xargs -P"$DL_PROC" -I{} sh -c \
        'wget -q -c -P "$1" "$2" && echo "${2##*/}"' _ "$RAW_DIR/data" {} < "$SEL" \
        | awk -v t="$TOTAL" '{printf ">>   [%d/%d] %s\n", NR, t, $0; fflush()}'
    TOK_INPUT="$RAW_DIR/data"; TOK_GLOB="*.json.gz"
    ;;

  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

# ---------------------------- RAW MANIFEST ----------------------------------
# Record what actually landed (byte sizes + filenames) as a SIBLING of raw/, so
# provenance survives deleting the raw text once the token array is verified.
# Raw text is free to re-fetch; the token array is the expensive artifact.
MANIFEST="$ROOT/$LABEL/raw-manifest.tsv"
{
    echo "# dataset:   $DATASET"
    [ "$DATASET" = redpajama ] && echo "# subset:    $SUBSET"
    [ -n "${INCLUDE:-}" ]      && echo "# include:   $INCLUDE"
    [ -n "${NFILES:-}"  ]      && echo "# nfiles:    $NFILES"
    [ -n "${VER:-}"     ]      && echo "# version:   $VER"
    echo "# fetched:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# raw_dir:   $RAW_DIR"
    echo "# tok_input: $TOK_INPUT (glob: $TOK_GLOB)"
    echo "# bytes	path   -- relative to raw_dir; .cache/ bookkeeping excluded"
} > "$MANIFEST"
find "$RAW_DIR" -name .cache -prune -o -type f -printf '%s\t%P\n' \
    | sort -k2 >> "$MANIFEST"
NFOUND=$(grep -cv '^#' "$MANIFEST" || true)
echo ">> manifest: $NFOUND files -> $MANIFEST"

# Keep the exact URL list too (RedPajama/Dolma): upstream subsets do disappear,
# and it's the record of precisely which files went in.
if [ -f "$RAW_DIR/urls.txt" ]; then
    cp -f "$RAW_DIR/urls.txt" "$ROOT/$LABEL/urls.txt"
    echo ">> urls:     $ROOT/$LABEL/urls.txt"
fi

# Download-only mode: stop here so the tokenize step can run separately as a
# scheduler-friendly job array (see submit-tokenize.sh). Fetch is network-bound,
# so this runs fine on a login node or a 1-CPU allocation.
if [ -n "${SKIP_TOKENIZE:-}" ]; then
    echo "SKIP_TOKENIZE set -> fetched into $TOK_INPUT (glob: $TOK_GLOB); skipping tokenize."
    echo "next: INPUT_DIR=$TOK_INPUT OUT=$OUT_BIN GLOB='$TOK_GLOB' ./submit-tokenize.sh"
    exit 0
fi

# ------------------------------- TOKENIZE -----------------------------------
python3 "$TOK_SCRIPT" \
    --input-dir "$TOK_INPUT" \
    --output    "$OUT_BIN" \
    --tokenizer "$TOKENIZER" \
    --glob      "$TOK_GLOB" \
    --workers   "$WORKERS"

# ------------------------------- VERIFY -------------------------------------
python3 - "$OUT_BIN" <<'PY'
import sys, numpy as np
a = np.memmap(sys.argv[1], dtype="<u2", mode="r")
n = a.shape[0]
seps = int((a == 0xFFFF).sum())
print(f"{sys.argv[1]}: {n:,} uint16 values "
      f"({n - seps:,} tokens + {seps:,} separators)")
PY
echo "done -> $OUT_BIN"
