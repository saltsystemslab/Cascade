#!/usr/bin/env python3
"""
Tokenize a JSONL corpus into ONE little-endian uint16 token array, matching the
infini-gram on-disk layout (2 bytes/token, 0xFFFF as the document separator).

Many independent 1-CPU tasks, then a concat.
    # Phase 1 (run as a job array, task i of N):
    python tokenize-corpus.py --input-dir DIR --output OUT --shard i/N --workers 1
    # Phase 2 (once all shards are done):
    python tokenize-corpus.py --input-dir DIR --output OUT --concat
Use the submit-tokenize.sh / tokenize.sbatch wrappers for Slurm.

This mode writes per-file chunks into a STABLE parts dir ("OUT.parts/") using
atomic rename (part_N.bin.tmp -> part_N.bin) and skip any part that already
exists. That makes the whole thing resumable: a killed run (walltime, preemption)
is recovered by simply rerunning -- only missing parts are redone. The parts dir
is removed only after a successful concat (unless --keep-temp).

File order in the final array is the sorted input-file order (deterministic /
reproducible). All tasks must see the SAME --input-dir and --glob so index i
refers to the same file everywhere; downloads must be complete before tokenizing.

Notes:
  * Llama-2 tokenizer is gated and needs `sentencepiece`. GPT-2 ("gpt2") and OLMo
    ("allenai/OLMo-7B-hf") are ungated; use the -hf OLMo repo (no trust_remote_code).
  * Token id 0xFFFF (65535) is the reserved separator; tokenizer vocab must be
    <= 65535 (Llama-2 32k / GPT-2 50k / OLMo 50k all fit).
"""

import argparse
import gzip
import io
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

SEP = 0xFFFF  # document separator token id (reserved; not a real token)

# Each worker's tokenizer stays single-threaded; parallelism is across files/tasks.
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
    """Tokenize one input file -> its part file, via a .tmp + atomic rename."""
    import numpy as np
    idx, in_path, out_path, batch_size, allow_empty = task
    tmp_path = out_path + ".tmp"
    n_ids = 0   # uint16 values written (tokens + separators)
    n_docs = 0
    with open_text(in_path) as f, open(tmp_path, "wb") as out:
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
    # A zero-document part is the one failure this design cannot otherwise see:
    # skip-if-exists counts it as done, do_concat accepts it, and the array comes
    # out short by exactly one file's documents while every other check still
    # passes. A wrong --text-field would do this to EVERY file. So refuse to
    # promote it, and leave no part behind for a resume to trust.
    if n_docs == 0 and not allow_empty:
        os.unlink(tmp_path)
        raise RuntimeError(
            f"{in_path}: produced 0 documents. Refusing to write an empty part -- "
            f"it would look 'done' to the resume logic and silently drop this file "
            f"from the array. Check --text-field (currently {_TEXT_FIELD!r}) and the "
            f"input file; pass --allow-empty-parts if an empty input is expected.")
    os.replace(tmp_path, out_path)   # atomic within the same directory/filesystem
    return idx, os.path.basename(in_path), n_ids, n_docs


def parse_shard(s):
    try:
        a, b = s.split("/")
        shard_id, num_shards = int(a), int(b)
    except Exception:
        raise argparse.ArgumentTypeError("shard must look like I/N, e.g. 3/32")
    if not (0 <= shard_id < num_shards):
        raise argparse.ArgumentTypeError(f"need 0 <= I < N (got {s!r})")
    return shard_id, num_shards


def build_file_list(input_dir, glob):
    return sorted(p for p in Path(input_dir).glob(glob) if p.is_file())


def part_path(parts_dir, idx):
    return os.path.join(parts_dir, f"part_{idx:06d}.bin")


def run_tasks(todo, workers, tokenizer, text_field):
    if not todo:
        return
    if workers <= 1:
        _init_worker(tokenizer, text_field)
        for t in todo:
            idx, name, n_ids, n_docs = tokenize_file(t)
            print(f"  [done] {name}: {n_ids:,} ids, {n_docs:,} docs", flush=True)
    else:
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(tokenizer, text_field)) as pool:
            for idx, name, n_ids, n_docs in pool.imap_unordered(tokenize_file, todo):
                print(f"  [done] {name}: {n_ids:,} ids, {n_docs:,} docs", flush=True)


def do_concat(files, parts_dir, output, keep_temp):
    n = len(files)
    missing = [i for i in range(n) if not os.path.exists(part_path(parts_dir, i))]
    if missing:
        sys.exit(f"cannot concat: {len(missing)}/{n} parts missing "
                 f"(e.g. indices {missing[:10]}). Re-run the shards that cover them.")
    with open(output, "wb") as out:
        for i in range(n):
            with open(part_path(parts_dir, i), "rb") as src:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
    if not keep_temp:
        shutil.rmtree(parts_dir, ignore_errors=True)
    size = os.path.getsize(output)
    if size % 2:
        sys.exit(f"output {output} is not 2-byte aligned ({size} bytes)")
    print(f"\nConcatenated {n} parts -> {output}")
    print(f"  {size:,} bytes = {size // 2:,} uint16 values")
    print("  load with:  np.memmap(path, dtype='<u2', mode='r')")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True,
                    help="final token array path; parts live in OUTPUT.parts/")
    ap.add_argument("--tokenizer", default="allenai/OLMo-7B-hf",
                    help="HF tokenizer; keep ONE across corpora so arrays stay "
                         "comparable (-hf: no trust_remote_code, no torch)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="documents per tokenizer call")
    ap.add_argument("--glob", default="*.jsonl*",
                    help="file pattern; also matches .json.gz, .jsonl.gz, .jsonl.zst")
    ap.add_argument("--shard", type=parse_shard, default=None, metavar="I/N",
                    help="Phase 1: tokenize only files with index%%N==I (stride)")
    ap.add_argument("--concat", action="store_true",
                    help="Phase 2: stitch existing parts into the final array")
    ap.add_argument("--keep-temp", action="store_true",
                    help="keep the parts dir after a successful concat")
    ap.add_argument("--allow-empty-parts", action="store_true",
                    help="permit a 0-document input file to produce an empty part "
                         "(default: hard error, since an empty part looks 'done' to "
                         "the resume logic and silently shortens the array)")
    args = ap.parse_args()

    if args.shard and args.concat:
        ap.error("--shard and --concat are mutually exclusive")

    files = build_file_list(args.input_dir, args.glob)
    if not files:
        sys.exit(f"No files matching {args.glob!r} in {args.input_dir}")
    parts_dir = args.output + ".parts"

    # ---- Phase 2 only ----
    if args.concat:
        do_concat(files, parts_dir, args.output, args.keep_temp)
        return

    os.makedirs(parts_dir, exist_ok=True)

    if args.shard is not None:
        shard_id, num_shards = args.shard
        my_idx = [i for i in range(len(files)) if i % num_shards == shard_id]
        role = f"shard {shard_id}/{num_shards}"
    else:
        my_idx = list(range(len(files)))
        role = "all files"

    doneset = {i for i in my_idx if os.path.exists(part_path(parts_dir, i))}
    todo = [(i, str(files[i]), part_path(parts_dir, i), args.batch_size,
             args.allow_empty_parts)
            for i in my_idx if i not in doneset]

    print(f"{len(files)} files total; this run: {role} -> {len(my_idx)} assigned, "
          f"{len(doneset)} already done, {len(todo)} to do; "
          f"tokenizer '{args.tokenizer}', {args.workers} workers", flush=True)

    run_tasks(todo, args.workers, args.tokenizer, args.text_field)

    if args.shard is None:
        # Mode A: single invocation also concatenates + cleans up.
        do_concat(files, parts_dir, args.output, args.keep_temp)
    else:
        print(f"{role} complete ({len(todo)} files tokenized this run). "
              f"Run --concat once all shards finish.", flush=True)


if __name__ == "__main__":
    main()
