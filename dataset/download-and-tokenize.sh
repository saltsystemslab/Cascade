#!/usr/bin/env bash
# Raw-text -> single token-array pipeline for infini-gram corpora.
#
# Per dataset:  (1) fetch a subset of raw JSONL,
#               (2) tokenize to ONE uint16-LE array via tokenize_corpus.py,
#               (3) verify token / separator counts.
#
# Usage:   ./download-and-tokenize.sh {c4|pile|redpajama|dolma}
# Tune via env vars, e.g.:
#   INCLUDE="en/c4-train.00*-of-01024.json.gz" TOKENIZER=allenai/OLMo-7B \
#       WORKERS=64 ./download-and-tokenize.sh c4
#
# Prereqs:  pip install -U "huggingface_hub[hf_xet]" transformers tokenizers \
#                          numpy zstandard sentencepiece
#           (sentencepiece only needed for the Llama-2 tokenizer)
set -euo pipefail

# ------------------------------- CONFIG -------------------------------------
if [ $# -lt 1 ]; then
    echo "usage: run_pipeline.sh {c4|pile|redpajama|dolma}" >&2
    exit 1
fi
DATASET="$1"
ROOT="${ROOT:-$PWD/corpora}"                  # working root
TOKENIZER="${TOKENIZER:-allenai/OLMo-7B}"     # keep ONE tokenizer across corpora
WORKERS="${WORKERS:-$(nproc)}"
DL_PROC="${DL_PROC:-8}"                        # parallel wget streams (URL-list datasets)
TOK_SCRIPT="${TOK_SCRIPT:-$PWD/tokenize-corpus.py}"

export HF_XET_HIGH_PERFORMANCE=1              # faster Xet downloads (replaces hf_transfer)

RAW_DIR="$ROOT/$DATASET/raw"
OUT_BIN="$ROOT/$DATASET/${DATASET}.tokenized.bin"
mkdir -p "$RAW_DIR" "$(dirname "$OUT_BIN")"

# ------------------------------- FETCH --------------------------------------
case "$DATASET" in
  c4)
    # 1024 en train shards, ~319 MB each (~195M tokens/shard).
    #   000* -> 10 shards (~2B tok) | 00* -> 100 (~20B) | * -> all (~198B)
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
    # Multi-file subsets parallelize across cores in tokenize_corpus.py:
    #   arxiv ~28B (~56 GB) | github ~59B (~118 GB) | c4 ~175B | common_crawl ~878B
    # AVOID as-is: wikipedia (~24B) and stackexchange (~20B) are each a SINGLE
    # monolithic .jsonl -> one worker, single-core tokenize. Pre-split first
    # (e.g. `split -d -n l/64 file.jsonl part_`) if you need them. book: defunct.
    SUBSET="${SUBSET:-arxiv}"; NFILES="${NFILES:-all}"
    curl -s https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt \
        | grep "/$SUBSET/" > "$RAW_DIR/urls.txt"
    if [ "$NFILES" != "all" ]; then
        head -n "$NFILES" "$RAW_DIR/urls.txt" > "$RAW_DIR/urls.head"
        mv "$RAW_DIR/urls.head" "$RAW_DIR/urls.txt"
    fi
    xargs -n1 -P"$DL_PROC" -I{} wget -q -c -P "$RAW_DIR" "{}" < "$RAW_DIR/urls.txt"
    TOK_INPUT="$RAW_DIR"; TOK_GLOB="*.jsonl*"
    ;;

  dolma)
    # allenai/dolma ships per-file URL lists at urls/<version>.txt
    VER="${VER:-v1_7}"; NFILES="${NFILES:-20}"
    hf download allenai/dolma "urls/${VER}.txt" --repo-type dataset --local-dir "$RAW_DIR"
    mkdir -p "$RAW_DIR/data"
    head -n "$NFILES" "$RAW_DIR/urls/${VER}.txt" \
        | xargs -n1 -P"$DL_PROC" -I{} wget -q -c -P "$RAW_DIR/data" "{}"
    TOK_INPUT="$RAW_DIR/data"; TOK_GLOB="*.json.gz"
    ;;

  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

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