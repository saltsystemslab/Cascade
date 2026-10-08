#!/usr/bin/env python3
"""Decode each corpus' heaviest K-gram, once the suffix array confirms it.

    kgram_decode.py <out.tsv> <K> <corpus>=<structure.json>,<text>,<sa>,<tokenizer>...

<structure.json> is `sa_structure`'s output for the corpus: it gives the heaviest
K-gram's frequency f_max, and the text-position of its first suffix in SA-order.
<text> is the corpus' token array (little-endian uint16), <sa> its suffix array
(little-endian entries of size(<sa>) / n bytes each), and <tokenizer> the Hugging
Face tokenizer that produced the array, loaded from the local cache (set HF_HOME
and HF_HUB_OFFLINE=1). Runs where `transformers` and `numpy` are installed, such
as the dataset pipeline's venv.

The suffixes that start with a K-gram form one SA-interval. So the check
binary-searches the SA for the K-gram at the recorded position: the interval must
hold f_max suffixes and start at that position, which also confirms that <text>
is the array that <sa> was built from.

Writes one row per corpus: the K-gram's frequency, its position and SA-rank, the
span of text-positions that its occurrences cover, the share of them that the
next text-position also starts (as in a run of one repeated token), and its token
IDs, tokens and text (the last two as JSON strings, where 0xFFFF, the document
separator, is "<sep>"). Prints some occurrences in context, drawn from the
interval at random, to interpret the K-gram by.
"""
import json
import random
import sys

import numpy as np

SEP = 0xFFFF
CONTEXT = 24    # Tokens of context on each side of a printed occurrence.
SAMPLES = 40
CHUNK = 1 << 24 # SA-entries read at a time, to find the occurrences.


def decode(tok, ids):
    """Returns the tokens and the text of `ids`, with the separators marked."""
    pieces = ["<sep>" if i == SEP else tok.convert_ids_to_tokens(int(i)) for i in ids]
    text, run = [], []
    for i in list(ids) + [SEP]:
        if i != SEP:
            run.append(int(i))
            continue
        if run:
            # Llama-2's decoder drops a leading space; a known first token undoes that.
            lead = tok.convert_tokens_to_ids("a")
            text.append(tok.decode([lead] + run, clean_up_tokenization_spaces=False)[1:])
            run = []
        text.append("<sep>")
    return pieces, "".join(text[:-1])


def check(name, spec, K, tok):
    """Returns the corpus' row, after checking its heaviest K-gram against its SA."""
    tool_path, text_path, sa_path, _ = spec
    tool = json.load(open(tool_path))
    n, f_max, pos = tool["n"], tool["kgram_max"][K - 1], tool["kgram_pos"][K - 1]
    T = np.memmap(text_path, dtype="<u2", mode="r")
    SA = np.memmap(sa_path, dtype=np.uint8, mode="r")
    w = len(SA) // n
    if len(T) != n or len(SA) != w * n:
        sys.exit(f"{name}: the text holds {len(T)} tokens and the SA {len(SA)} bytes, for n = {n}")

    def suffix(r):
        return int.from_bytes(SA[r * w:(r + 1) * w].tobytes(), "little")

    def prefix(r):  # Its first K tokens; fewer at the text's end, which sort first.
        j = suffix(r)
        return tuple(int(x) for x in T[j:j + K])

    g = tuple(int(x) for x in T[pos:pos + K])

    def bound(strict):  # The first SA-rank whose prefix exceeds g (strictly or not).
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            p = prefix(mid)
            if p < g or (strict and p == g):
                lo = mid + 1
            else:
                hi = mid
        return lo

    lo, hi = bound(False), bound(True)
    if len(g) != K or hi - lo != f_max or suffix(lo) != pos:
        sys.exit(f"{name}: the SA holds {hi - lo} suffixes from SA-rank {lo} (text-position {suffix(lo)}) with the"
                 f" K-gram at {pos}; the tool reports {f_max}")

    occ = np.concatenate([SA[r * w:min(r + CHUNK, hi) * w].reshape(-1, w).astype(np.int64)
                          @ (np.int64(1) << np.arange(0, 8 * w, 8, dtype=np.int64)) for r in range(lo, hi, CHUNK)])
    occ.sort()
    first, last = int(occ[0]), int(occ[-1])
    overlap = float(np.mean(occ[1:] - occ[:-1] == 1)) if f_max > 1 else 0.0

    pieces, text = decode(tok, g)
    print(f"{name}: the heaviest {K}-gram occurs {f_max} times, at SA-ranks [{lo}, {hi}); tokens {json.dumps(pieces)},"
          f" text {json.dumps(text)}", flush=True)
    print(f"    its occurrences span text-positions [{first}, {last}], {(last - first + 1) / n:.2%} of the text, and"
          f" {overlap:.1%} of them are followed by another at the next position", flush=True)
    rng = random.Random(f"{name}-{K}")
    for r in sorted(rng.randrange(lo, hi) for _ in range(SAMPLES)):
        j = suffix(r)
        left = [int(x) for x in T[max(0, j - CONTEXT):j]]
        right = [int(x) for x in T[j + K:j + K + CONTEXT]]
        print(f"    at {j}: {json.dumps(decode(tok, left)[1])} [{json.dumps(text)}] {json.dumps(decode(tok, right)[1])}")
    return {"corpus": name, "K": K, "fmax": f_max, "pos": pos, "sa_rank": lo, "first": first, "last": last,
            "overlap": f"{overlap:.6g}", "ids": " ".join(map(str, g)), "tokens": json.dumps(pieces), "text": json.dumps(text)}


def main(argv):
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    if len(argv) < 3:
        print(__doc__)
        return 2
    from transformers import AutoTokenizer

    out, K = argv[0], int(argv[1])
    specs = {}
    for a in argv[2:]:
        name, _, spec = a.partition("=")
        specs[name] = spec.split(",")
        if len(specs[name]) != 4:
            sys.exit(f"{a}: expected <corpus>=<structure.json>,<text>,<sa>,<tokenizer>")

    toks, rows = {}, []
    for name, spec in specs.items():
        if spec[3] not in toks:
            toks[spec[3]] = AutoTokenizer.from_pretrained(spec[3], use_fast=True)
        rows.append(check(name, spec, K, toks[spec[3]]))

    cols = list(rows[0])
    with open(out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
    print(f"Wrote {len(rows)} corpora's heaviest {K}-grams to {out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
