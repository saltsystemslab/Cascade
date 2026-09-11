#!/usr/bin/env python3
"""Verify a GPU_SACA token array: little-endian uint16, 0xFFFF (65535) written
AFTER every document. Full chunked scan; numpy only.

Exit status is the point of this script: 0 = every check passed, 1 = a check
FAILED, 2 = usage/config error. That is what lets "verify before deleting raw"
(invariant 7) be a gate rather than a suggestion.

    python3 verifier.py --tokenizer llama2 --expect-docs 1,558,306 arxiv-llama2.tokenized.bin
    python3 verifier.py --vocab 50280 c4.tokenized.bin       # pre-tag arrays are untagged
    python3 verifier.py --sep-first dolmasample.tokenized.bin

Document counts are tokenizer-invariant: an OLMo and a Llama-2 array over the
same raw text MUST report the same --expect-docs, or the re-download pulled
different data and the alphabet-ablation pair is confounded. Known-good:

    arxiv 1,558,306 | github 28,793,312 | c4 (1024 shards) 364,868,892

Token counts are NOT valid pass/fail targets. The runbook's 1024-shard OLMo
figure is the 1000-shard count x 1.024, and its "198.1B (exact)" for Llama-2 is
that times 1.133 -- estimates, not measurements. Gate on documents.
"""

import argparse
import hashlib
import os
import re
import sys

import numpy as np

SEP = 0xFFFF            # reserved document separator; not a real token id
CHUNK = 1 << 28         # 256M elems/chunk -> ~512 MB + a bool mask per chunk

# max valid id + 1. Verify against the real tokenizer before trusting these:
#   python3 -c "from transformers import AutoTokenizer as A; \
#               print(len(A.from_pretrained('meta-llama/Llama-2-7b-hf')))"
VOCAB = {"olmo": 50280, "llama2": 32000, "gpt2": 50257}


def _int(s):
    """Accept 1_558_306 and 1,558,306 as well as 1558306."""
    return int(s.replace(",", "").replace("_", ""))


def _tag_from_name(path):
    m = re.search(r"[-_](olmo|llama2|gpt2)\b", os.path.basename(path).lower())
    return m.group(1) if m else None


def _sha256(path, blocks=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(blocks), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--vocab", type=_int,
                    help="max valid id + 1; overrides --tokenizer and the filename tag")
    ap.add_argument("--tokenizer", choices=sorted(VOCAB),
                    help="vocab bound by name (default: the -olmo/-llama2/-gpt2 tag "
                         "in the filename)")
    ap.add_argument("--expect-docs", type=_int, help="exact document count")
    ap.add_argument("--expect-tokens", type=_int,
                    help="exact content-token count (rarely a known-good figure)")
    ap.add_argument("--avg-range", metavar="LO:HI",
                    help="FAIL outside this avg tokens/doc band "
                         "(default: WARN outside 10:50000)")
    ap.add_argument("--sep-first", action="store_true",
                    help="array PREPENDS the separator (infini-gram's pre-built "
                         "dolmasample): element 0 is 65535 and the last element is a "
                         "real token id")
    ap.add_argument("--quick", action="store_true",
                    help="structural checks only: first + last chunk, no full scan")
    ap.add_argument("--sha256", action="store_true",
                    help="also hash the file (a full extra read; ~40 min for 341 GB)")
    ap.add_argument("--tsv", metavar="FILE",
                    help="on PASS, append a corpora.tsv row: "
                         "label, tokenizer, tokens, docs, bytes, sha256, timestamp")
    ap.add_argument("--label", help="label for the --tsv row (default: filename stem)")
    args = ap.parse_args()

    # --- resolve the vocab bound -------------------------------------------------
    vocab, how = args.vocab, "--vocab"
    if vocab is None:
        tag = args.tokenizer or _tag_from_name(args.path)
        if tag is None:
            print(f"config: cannot determine the vocab bound for {args.path}. Pass "
                  f"--vocab N or --tokenizer {{{'|'.join(sorted(VOCAB))}}} -- the four "
                  f"pre-tag arrays are untagged on disk.", file=sys.stderr)
            return 2
        vocab = VOCAB[tag]
        how = "--tokenizer" if args.tokenizer else f"filename tag {tag!r}"

    fails = []

    def rec(ok, label, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            fails.append(label)

    def warn(label, detail):
        print(f"  [WARN] {label}: {detail}")

    def verdict():
        print(f"\n{'FAIL' if fails else 'PASS'}: {args.path}"
              + (f"   failed: {', '.join(fails)}" if fails else ""))
        return 1 if fails else 0

    # --- size first: np.memmap refuses an empty or odd-sized file, which would
    # --- pre-empt the very check meant to catch a truncated write ---------------
    if not os.path.isfile(args.path):
        print(f"config: no such file: {args.path}", file=sys.stderr)
        return 2
    size = os.path.getsize(args.path)
    print(f"{args.path}\n  {size:,} bytes ({size / 1e9:.1f} GB)   "
          f"vocab bound {vocab} (from {how})")

    if size == 0 or size % 2:
        rec(False, "even byte count",
            f"{size:,} bytes -- {'empty file' if size == 0 else 'odd, truncated write'}")
        return verdict()
    rec(True, "even byte count", f"{size:,} bytes = {size // 2:,} uint16 values")

    a = np.memmap(args.path, dtype="<u2", mode="r")
    n = a.shape[0]
    first, last = int(a[0]), int(a[-1])
    print(f"  first element: {first}   last element: {last}")

    # --- separator convention ---------------------------------------------------
    if args.sep_first:
        rec(first == SEP, "leading separator", f"a[0] = {first} (expect 65535)")
        rec(last != SEP, "unterminated last doc",
            f"a[-1] = {last} (prepend convention: expect a real token id)")
    else:
        rec(last == SEP, "trailing separator", f"a[-1] = {last} (expect 65535)")
        if first == SEP:
            warn("leading element", "a[0] = 65535, but this pipeline never writes a "
                                    "leading separator -- wrong convention, or a "
                                    "zero-length first document?")

    # --- chunked scan -----------------------------------------------------------
    if args.quick:
        spans = [(0, min(CHUNK, n))]
        if n > CHUNK:
            spans.append((max(0, n - CHUNK), n))
    else:
        spans = [(i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)]

    seps, mx = 0, 0
    for lo, hi in spans:
        c = np.asarray(a[lo:hi])
        seps += int(np.count_nonzero(c == SEP))
        # where=/initial= avoids materialising a second filtered array per chunk;
        # initial=0 covers an all-separator chunk, where the max is undefined.
        mx = max(mx, int(np.max(c, where=(c != SEP), initial=0)))

    rec(mx < vocab, "max non-separator id",
        f"{mx} (must be < {vocab}) -- any non-separator value at or above the vocab "
        f"means garbage in the stream")

    if args.quick:
        print(f"  [SKIP] document count: --quick scanned {len(spans)} chunk(s), not all")
        return verdict()

    # --- document accounting ----------------------------------------------------
    print(f"  documents (separators): {seps:,}   content tokens: {n - seps:,}")
    rec(seps > 0, "separators present",
        f"{seps:,} -- zero means this is not a document-separated array")
    if seps == 0:
        return verdict()

    avg = (n - seps) / seps
    if args.expect_docs is not None:
        ok = seps == args.expect_docs
        rec(ok, "document count", f"{seps:,} vs expected {args.expect_docs:,}"
            + ("" if ok else f"  (delta {seps - args.expect_docs:+,})"))
    if args.expect_tokens is not None:
        ok = (n - seps) == args.expect_tokens
        rec(ok, "content tokens", f"{n - seps:,} vs expected {args.expect_tokens:,}"
            + ("" if ok else f"  (delta {(n - seps) - args.expect_tokens:+,})"))

    if args.avg_range:
        lo_s, _, hi_s = args.avg_range.partition(":")
        lo_a, hi_a = float(lo_s), float(hi_s)
        rec(lo_a <= avg <= hi_a, "avg tokens/doc", f"{avg:.1f} in [{lo_a}, {hi_a}]")
    else:
        rec(avg >= 1.0, "avg tokens/doc", f"{avg:.1f} (must be >= 1.0)")
        if not 10 <= avg <= 50000:
            warn("avg tokens/doc", f"{avg:.1f} outside the plausible 10-50000 band")

    digest = _sha256(args.path) if args.sha256 else "-"
    if args.sha256:
        print(f"  sha256: {digest}")

    status = verdict()

    # --- durable record ---------------------------------------------------------
    # corpora.tsv is the authoritative "what exists" record; the runbook's tables
    # are a narrative view of it. Only PASSing arrays get a row.
    if args.tsv and status == 0:
        label = args.label or re.sub(r"\.tokenized\.bin$", "",
                                     os.path.basename(args.path))
        tag = args.tokenizer or _tag_from_name(args.path) or "-"
        row = "\t".join(str(x) for x in (
            label, tag, n - seps, seps, size, digest,
            __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        # Several verify jobs can finish at once (the chained pipeline runs one
        # per corpus), and this lives on NFS, so serialise the append: a torn
        # line in the authoritative record is worse than a failed write.
        import fcntl
        with open(args.tsv, "a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except OSError as e:          # some filesystems refuse locks
                warn("tsv lock", f"{e}; appending unlocked")
            if f.tell() == 0:
                f.write("# label\ttokenizer\ttokens\tdocs\tbytes\tsha256\tverified_at\n")
            f.write(row + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"  recorded -> {args.tsv}")

    return status


if __name__ == "__main__":
    sys.exit(main())
