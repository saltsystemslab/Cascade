#!/usr/bin/env python3
"""
Tokenize a JSONL corpus into ONE little-endian uint16 token array, matching the
infini-gram on-disk layout (2 bytes/token, 0xFFFF as the document separator).

Design: CPU-parallel by file.
  Phase 1 (parallel): each input file is tokenized in its own worker process
                      into a temporary uint16-LE chunk.
  Phase 2 (serial):   the per-file chunks are concatenated, in sorted-filename
                      order, into a single output array. This is just byte
                      copying (I/O-bound), so it does not undo the parallelism.

The document ORDER in the final array is a reproducibility choice, not a
correctness one for the suffix array. Sorting filenames makes reruns identical.

Usage:
  python tokenize_corpus.py \
      --input-dir /path/to/c4_samples \
      --output    /path/to/c4.tokenized.bin \
      --tokenizer meta-llama/Llama-2-7b-hf \
      --workers   64

Notes:
  * Llama-2 tokenizer is gated (free Meta license click-through) and needs
    `sentencepiece`. GPT-2 ("gpt2") and OLMo ("allenai/OLMo-7B") are ungated.
  * Token id 0xFFFF (65535) is reserved as the separator; the tokenizer vocab
    must therefore be <= 65535 (Llama-2 32k / GPT-2 50k / OLMo 50k all fit).
"""

import argparse
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SEP = 0xFFFF  # document separator token id (reserved; not a real token)

# Each worker's tokenizer stays single-threaded; we parallelize across files.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def open_text(path):
    """Open .jsonl, .jsonl.gz / .json.gz, or .jsonl.zst transparently as text."""
    p = str(path)
    if p.endswith(".zst"):
        import zstandard as zstd  # pip install zstandard
        fh = open(p, "rb")
        return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh),
                                encoding="utf-8")
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "rt", encoding="utf-8")


# --- worker globals (loaded once per process via the Pool initializer) -------
_TOK = None
_TEXT_FIELD = "text"


def _init_worker(tokenizer_name, text_field):
    global _TOK, _TEXT_FIELD
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    _TEXT_FIELD = text_field
    if len(_TOK) > SEP:  # max valid id must be < 65535
        raise ValueError(
            f"Tokenizer vocab {len(_TOK)} does not fit in uint16 below the "
            f"separator id {SEP}; pick a tokenizer with vocab <= 65535.")


def tokenize_file(task):
    in_path, out_path, batch_size = task
    n_ids = 0   # total uint16 values written (tokens + separators)
    n_docs = 0
    with open_text(in_path) as f, open(out_path, "wb") as out:
        batch = []

        def flush(texts):
            nonlocal n_ids, n_docs
            if not texts:
                return
            encoded = _TOK(texts, add_special_tokens=False)["input_ids"]
            buf = []
            for ids in encoded:
                buf.extend(ids)
                buf.append(SEP)
                n_ids += len(ids) + 1
                n_docs += 1
            np.asarray(buf, dtype="<u2").tofile(out)

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(_TEXT_FIELD)
            if not text:
                continue
            batch.append(text)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        flush(batch)
    return out_path, n_ids, n_docs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True,
                    help="path to the single final token array, e.g. c4.tokenized.bin")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="documents per tokenizer call")
    ap.add_argument("--glob", default="*.jsonl*",
                    help="file pattern; also matches .json.gz, .jsonl.gz, .jsonl.zst")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input_dir).glob(args.glob) if p.is_file())
    if not files:
        sys.exit(f"No files matching {args.glob!r} in {args.input_dir}")
    print(f"Found {len(files)} files; tokenizing with '{args.tokenizer}' "
          f"on {args.workers} workers", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    tmpdir = tempfile.mkdtemp(prefix="tok_", dir=out_dir)
    tasks = [(str(f), os.path.join(tmpdir, f"part_{i:06d}.bin"), args.batch_size)
             for i, f in enumerate(files)]

    results = {}
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(args.tokenizer, args.text_field)) as pool:
        for out_path, n_ids, n_docs in pool.imap_unordered(tokenize_file, tasks):
            results[out_path] = (n_ids, n_docs)
            print(f"  done {os.path.basename(out_path)}: "
                  f"{n_ids:,} ids, {n_docs:,} docs", flush=True)

    # Phase 2: concatenate chunks in deterministic order into one array.
    total_docs = 0
    with open(args.output, "wb") as out:
        for _, tmp, _ in tasks:               # tasks already in sorted order
            with open(tmp, "rb") as src:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
            total_docs += results[tmp][1]
    if not args.keep_temp:
        shutil.rmtree(tmpdir, ignore_errors=True)

    size = os.path.getsize(args.output)
    assert size % 2 == 0, "output is not 2-byte aligned"
    print(f"\nWrote {args.output}")
    print(f"  {size:,} bytes = {size // 2:,} uint16 values "
          f"({total_docs:,} docs, including {SEP} separators)")
    print("  load with:  np.memmap(path, dtype='<u2', mode='r')")


if __name__ == "__main__":
    main()
