#!/usr/bin/env bash
# Raw-text -> single token-array pipeline for infini-gram corpora.
#
# Per dataset:  (1) fetch a subset of raw JSONL,
#               (2) tokenize to ONE uint16-LE array via tokenize-corpus.py,
#               (3) verify token / separator counts.
#
# Usage:   ROOT=/scratch/$USER/corpora ./download-and-tokenize.sh {c4|pile|fineweb|redpajama|dolma}
# Tune via env vars, e.g.:
#   ROOT=/scratch/$USER/corpora INCLUDE="en/c4-train.000*-of-01024.json.gz" \
#       TOKENIZER=allenai/OLMo-7B-hf WORKERS=64 ./download-and-tokenize.sh c4
#   # NB: C4 shard numbers are 5 digits -> 000* = 100 shards (~35 GB),
#   #     00* = 1000 shards (~340 GB, ~all of C4), * = all 1024.
#
# Prereqs:  pip install -U "huggingface_hub[hf_xet]" transformers tokenizers \
#                          numpy pyarrow zstandard sentencepiece
#           sentencepiece: Llama-2 only | pyarrow: parquet corpora (pile,
#           fineweb) | zstandard: .jsonl.zst only. On this cluster the login
#           nodes kill pip, so use submit-setup-env.sh instead of installing by
#           hand, and pass PYTHON=<venv>/bin/python to every job.
set -euo pipefail

# Absolute dir of this script, so its sibling tokenizer is found regardless of
# the caller's $PWD (both files live together in the repo's dataset/ dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------- CONFIG -------------------------------------
if [ $# -lt 1 ]; then
    echo "usage: ROOT=<dir> ./download-and-tokenize.sh {c4|pile|fineweb|redpajama|dolma}" >&2
    exit 1
fi
DATASET="$1"
ROOT="${ROOT:?set ROOT to a working dir for raw data + the token array (no default)}"
TOKENIZER="${TOKENIZER:-allenai/OLMo-7B-hf}"  # HF tokenizer; keep ONE across corpora (-hf: no trust_remote_code, no torch)
WORKERS="${WORKERS:-$(nproc)}"                # tokenizer worker procs (CPU-bound; ~1/core)
DL_PROC="${DL_PROC:-8}"                        # parallel download streams (network-bound)
TOK_SCRIPT="${TOK_SCRIPT:-$SCRIPT_DIR/tokenize-corpus.py}"   # sibling tokenizer script

# The HF CLI lives in the venv, and its shebang points at the venv python, which
# needs the module's libpython on LD_LIBRARY_PATH. So a bare `hf` is neither on a
# batch node's PATH nor runnable directly -- setup-env.sbatch emits an `hfrun`
# wrapper that pins the library path, and HF_CLI should point at that.
HF_CLI="${HF_CLI:-hf}"
need_hf() {
    command -v "$HF_CLI" >/dev/null 2>&1 && return 0
    echo "HF CLI not found: '$HF_CLI' -- the c4/pile/fineweb branches need it." >&2
    echo "Run submit-setup-env.sh, then pass HF_CLI=<venv>/bin/hfrun. Not a bare" >&2
    echo "'hf' (absent from a batch node's PATH), and not <venv>/bin/hf (whose" >&2
    echo "shebang cannot find libpython without the wrapper)." >&2
    exit 1
}

export HF_XET_HIGH_PERFORMANCE=1              # faster Xet downloads (replaces hf_transfer)

# Per-corpus label; redpajama is further split by subset so arxiv/github/etc.
# get isolated raw dirs and output arrays instead of clobbering each other.
LABEL="$DATASET"
if [ "$DATASET" = redpajama ]; then
    SUBSET="${SUBSET:-arxiv}"                  # RedPajama-V1 subset (see redpajama branch)
    LABEL="$DATASET-$SUBSET"
fi
# LABEL is overridable because the four already-built corpora live in FLAT dirs
# on scratch -- arxiv/, github/, c4/, dolmasample/ -- not the redpajama-<subset>
# this computes. Without the override, a Llama-2 arxiv run would create a second
# directory (redpajama-arxiv/) beside the OLMo one, splitting an alphabet-ablation
# pair across two paths where nothing checks that they describe the same corpus.
#   LABEL=arxiv SUBSET=arxiv ./submit-download.sh redpajama
LABEL="${LABEL_OVERRIDE:-$LABEL}"

# Tokenizer tag: arrays from different tokenizers must not overwrite each other.
# raw/ stays UNtagged on purpose -- one download feeds every tokenization of it.
case "$TOKENIZER" in
    *[Oo][Ll][Mm][Oo]*)    TAG="${TAG:-olmo}"   ;;
    *[Ll]lama-2*|*llama*)  TAG="${TAG:-llama2}" ;;
    *gpt2*)                TAG="${TAG:-gpt2}"   ;;
    *) TAG="${TAG:-$(basename "$TOKENIZER" | tr '[:upper:]' '[:lower:]')}" ;;
esac

RAW_DIR="$ROOT/$LABEL/raw"
OUT_BIN="$ROOT/$LABEL/${LABEL}-${TAG}.tokenized.bin"
mkdir -p "$RAW_DIR" "$(dirname "$OUT_BIN")"

# ------------------------------- FETCH --------------------------------------
case "$DATASET" in
  c4)
    # 1024 en train shards, ~319 MB each (~195M tokens/shard).
    #   5-digit shards: 000* -> 100 (~35 GB) | 00* -> 1000 (~340 GB, ~all) | * -> all 1024
    INCLUDE="${INCLUDE:-en/c4-train.000*-of-01024.json.gz}"
    need_hf; "$HF_CLI" download allenai/c4 --repo-type dataset \
        --include "$INCLUDE" --local-dir "$RAW_DIR"
    TOK_INPUT="$RAW_DIR/en"; TOK_GLOB="c4-train.*.json.gz"
    ;;

  pile)
    # EleutherAI/the_pile_deduplicated -- the artifact Pythia was trained on, so
    # it is citable, unlike one maintainer's pick of "copyright-safe" subsets.
    # PARQUET: 1650 files, 451.1 GB, a single `text` column (verified 2026-09-11).
    # The names carry a content hash -- data/train-NNNNN-of-01650-<hash>.parquet
    # -- so the tokenize glob must be a wildcard; the hash is not derivable.
    #
    # Deduplication removes the near-duplicate spans that drive deep LCPs, so
    # expect Sigma h_i/n below raw Pile. State that as a property of the corpus
    # rather than discovering it in the numbers.
    INCLUDE="${INCLUDE:-data/*.parquet}"
    need_hf; "$HF_CLI" download EleutherAI/the_pile_deduplicated --repo-type dataset \
        --include "$INCLUDE" --local-dir "$RAW_DIR"
    case "$INCLUDE" in */*) TOK_INPUT="$RAW_DIR/${INCLUDE%/*}";; *) TOK_INPUT="$RAW_DIR";; esac
    TOK_GLOB="train-*.parquet"
    ;;

  fineweb)
    # HuggingFaceFW/fineweb. sample/350BT is 510 parquet files, 1061.4 GB
    # (verified 2026-09-11) -> ~360B Llama-2 tokens, ~720 GB of array. The full
    # corpus is 15T+ under data/CC-MAIN-*/ and is out of scope.
    #
    # Names are NNN_NNNNN.parquet -- no `train-` prefix and no hash, unlike the
    # Pile -- so the glob here is plain *.parquet. Getting this wrong fetches or
    # tokenizes nothing, silently.
    #
    # 9 columns ship; tokenize-corpus.py projects to `text` alone, so the rest
    # are never decompressed. That is a real saving on a terabyte, not tidiness.
    INCLUDE="${INCLUDE:-sample/350BT/*}"
    need_hf; "$HF_CLI" download HuggingFaceFW/fineweb --repo-type dataset \
        --include "$INCLUDE" --local-dir "$RAW_DIR"
    case "$INCLUDE" in */*) TOK_INPUT="$RAW_DIR/${INCLUDE%/*}";; *) TOK_INPUT="$RAW_DIR";; esac
    TOK_GLOB="*.parquet"
    ;;

  redpajama)
    # RedPajama-V1 raw text is served from data.together.xyz.
    # Multi-file subsets parallelize across cores in tokenize-corpus.py:
    #   arxiv ~28B (~56 GB) | github ~59B (~118 GB) | c4 ~175B | common_crawl ~878B
    # AVOID as-is: wikipedia (~24B) and stackexchange (~20B) are each a SINGLE
    # monolithic .jsonl -> one worker, single-core tokenize. Pre-split first
    # (e.g. `split -d -n l/64 file.jsonl part_`) if you need them. book: defunct.
    SUBSET="${SUBSET:-arxiv}"; NFILES="${NFILES:-all}"   # SUBSET resolved in CONFIG; kept for clarity
    # Re-fetching by REPLAYING the saved url list, not by re-deriving it. The
    # previous version always re-curled the live master list and re-grepped the
    # subset, so a re-fetch pulled whatever upstream serves today -- which is
    # exactly the confounding the runbook warns about for the ablation pairs,
    # with the urls.txt it carefully preserved sitting unused. When a saved list
    # exists we replay it; when it exists AND differs from upstream we say so
    # and stop, because a changed list means the corpus changed.
    SAVED="$ROOT/$LABEL/urls.txt"
    if [ -f "$SAVED" ] && [ -z "${REFRESH_URLS:-}" ]; then
        echo ">> replaying saved url list: $SAVED ($(wc -l < "$SAVED") files)"
        cp -f "$SAVED" "$RAW_DIR/urls.txt"
        if [ -n "${CHECK_URLS:-}" ]; then
            curl -s https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt \
                | grep "/$SUBSET/" > "$RAW_DIR/urls.upstream" || true
            if ! diff -q "$SAVED" "$RAW_DIR/urls.upstream" >/dev/null; then
                echo "upstream url list for '$SUBSET' differs from the saved one:" >&2
                diff "$SAVED" "$RAW_DIR/urls.upstream" | head -20 >&2
                echo "Refusing to proceed -- re-fetching different files would confound" >&2
                echo "any comparison against the existing array. Set REFRESH_URLS=1 to" >&2
                echo "deliberately adopt the new list (and expect different counts)." >&2
                exit 1
            fi
            echo ">> url list matches upstream"
            # Not corpus data: leaving it in raw/ puts it in the manifest, where
            # it shows up as a spurious extra entry in an old-vs-new join.
            rm -f "$RAW_DIR/urls.upstream"
        fi
    else
        curl -s https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt \
            | grep "/$SUBSET/" > "$RAW_DIR/urls.txt"
        if [ "$NFILES" != "all" ]; then
            head -n "$NFILES" "$RAW_DIR/urls.txt" > "$RAW_DIR/urls.head"
            mv "$RAW_DIR/urls.head" "$RAW_DIR/urls.txt"
        fi
    fi
    [ -s "$RAW_DIR/urls.txt" ] || { echo "no urls for subset '$SUBSET' -- check the name" >&2; exit 1; }
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
    need_hf; "$HF_CLI" download allenai/dolma "urls/${VER}.txt" --repo-type dataset --local-dir "$RAW_DIR"
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
# Archive any existing manifest before truncating it. Once a corpus's raw/ has
# been deleted, its manifest is the ONLY surviving evidence of what the verified
# array was built from -- and comparing old vs new byte sizes per shard is the
# cheapest way to catch a re-download that pulled different data, which is the
# one failure that would quietly invalidate an alphabet-ablation pair. Losing it
# to a re-fetch is unrecoverable, so this runs before the truncation, not after.
if [ -f "$MANIFEST" ] && [ ! -f "$ROOT/$LABEL/raw-manifest.orig.tsv" ]; then
    cp -n "$MANIFEST" "$ROOT/$LABEL/raw-manifest.orig.tsv"
    echo ">> archived the previous manifest -> $ROOT/$LABEL/raw-manifest.orig.tsv"
fi
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

# `hf download --include <glob>` exits 0 when the glob matches nothing, so a
# typo'd INCLUDE looks like a successful fetch and the failure only surfaces
# hours later as "No files matching" in every array task. Fail here instead.
if [ "$NFOUND" -eq 0 ]; then
    echo "nothing landed in $RAW_DIR -- INCLUDE='${INCLUDE:-<unset>}' matched no files." >&2
    echo "Confirm the real layout first:  hf datasets ls <repo> --tree -h" >&2
    exit 1
fi

# Keep the exact URL list too (RedPajama/Dolma): upstream subsets do disappear,
# and it's the record of precisely which files went in.
if [ -f "$RAW_DIR/urls.txt" ]; then
    cp -f "$RAW_DIR/urls.txt" "$ROOT/$LABEL/urls.txt"
    echo ">> urls:     $ROOT/$LABEL/urls.txt"
fi

# ---------------------------- PRE-SPLIT (auto) ------------------------------
# Single-file subsets (wikipedia, stackexchange) would otherwise tokenize on one
# core. presplit.sh rewrites TOK_INPUT/TOK_GLOB to point at the parts dir, so it
# is SOURCED, not executed. It lives in its own file because none of the GNU
# tools it needs (split -n l/N, --additional-suffix, find -printf, stat -c)
# exist on BSD/macOS, which makes it untestable on a dev Mac unless it can be
# driven directly -- see test-presplit.sh.
#
# Position matters twice over: after the manifest, so provenance records the
# file that was actually downloaded rather than 64 derived parts; and before the
# `next:` line below, which must carry the rewritten input dir and a glob that
# matches it.
. "$SCRIPT_DIR/presplit.sh"

# Download-only mode: stop here so the tokenize step can run separately as a
# scheduler-friendly job array (see submit-tokenize.sh). Fetch is network-bound,
# so this runs fine on a login node or a 1-CPU allocation.
if [ -n "${SKIP_TOKENIZE:-}" ]; then
    echo "SKIP_TOKENIZE set -> fetched into $TOK_INPUT (glob: $TOK_GLOB); skipping tokenize."
    # TOKENIZER is echoed because the tag in OUT is derived from it: pasting an
    # OUT that says -olmo while overriding TOKENIZER=...Llama-2... writes a
    # Llama-2 array to an olmo-tagged path, and the whole OLMo-vs-Llama-2
    # comparison is keyed on that tag.
    echo "next: TOKENIZER=$TOKENIZER INPUT_DIR=$TOK_INPUT OUT=$OUT_BIN GLOB='$TOK_GLOB' ./submit-tokenize.sh"
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
