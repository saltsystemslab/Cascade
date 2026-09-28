#!/usr/bin/env python3
"""Summarize a results directory of timed.sbatch.

    summarize.py <results-dir>

Reads the key=value logs of `query find` (ours/) and of bench_infinigram_query
(ig/), and writes runs.tsv (a line per timed batch) and summary.md. Throughputs
are in millions of queries per second: a batch's size over its median
repetition. The main runs report the median over their processes, with min-max.
Deliberately no means: at three processes they are not meaningful.

Every batch's counts hash must equal infini-gram's for the same queries (its
workload, and as many of them); summary.md lists the batches whose hash differs,
and the count of our batches failing the certificate.
"""
import math
import pathlib
import re
import statistics
import sys

KEY = re.compile(r"([a-z][a-z0-9_]*)=(.*)")
MODE_KEYS = ("stride", "key_words", "key_symbols", "sort", "table_bytes", "t_table")
KINDS = ("present", "mutated", "random", "heldout", "unigram", "batch")
SHOWN = ("present-m1", "present-m10", "random-m5", "present-m100", "present-m1000")
NAN = math.nan


def pairs(path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = KEY.fullmatch(line)
        if m:
            yield m.group(1), m.group(2)


def ours_batches(path, run):
    """The batches of a `query find` log, each with its load's and its mode's keys."""
    load, mode, cur, out = {}, {}, None, []
    for k, v in pairs(path):
        if k in ("stride", "queries"):
            if cur is not None:
                out.append(cur)
            cur = dict(load, **mode) if k == "queries" else None
            if k == "stride":
                mode = {}
        if cur is not None:
            cur[k] = v
        elif k in MODE_KEYS:
            mode[k] = v
        else:
            load[k] = v
    if cur is not None:
        out.append(cur)
    for b in out:
        b.update(engine="ours", run=run, config=f"s{b.get('stride')}w{b.get('key_words')}o{b.get('sort')}")
    return out


def ig_batch(path, run):
    b = dict(pairs(path))
    if "qps" not in b:
        return []
    b.update(engine="ig", run=run, config=f"{b.get('mode')}-t{b.get('threads')}-p{b.get('prefetch')}-{b.get('load')}")
    return [b]


def num(b, k):
    try:
        return float(b[k])
    except (KeyError, ValueError):
        return NAN


def load(d):
    """The batches of every log under `d`, and the logs that hold none or report errors."""
    bs, bad = [], []
    logs = [(p, p.parent.name if p.name == "find.log" else p.stem, ours_batches)
            for p in sorted((d / "ours").glob("*.log")) + sorted((d / "ours").glob("*/find.log"))]
    logs += [(p, p.parent.name, ig_batch) for p in sorted((d / "ig").glob("*/*.log"))]
    for p, run, parse in logs:
        got = parse(p, run)
        errors = [line for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("error")]
        if not got or errors:
            bad.append(f"{p.relative_to(d)}: {errors[0] if errors else 'no results'}")
        bs += got
    for b in bs:
        b["workload"] = pathlib.Path(b.get("queries", "?")).stem
        q, med, best = num(b, "q"), num(b, "t_find_median"), num(b, "t_find_min")
        b["qps_median"] = q / med if med > 0 else NAN
        b["qps_best"] = q / best if best > 0 else NAN
        if b["engine"] == "ours":
            t = num(b, "t_h2d") + med + num(b, "t_d2h")
            b["qps_e2e"] = q / t if t > 0 else NAN
    return bs, bad


def order(w):
    m = re.fullmatch(r"([a-z]+)-[mq](\d+)", w)
    kind = w if not m else m.group(1)
    return (KINDS.index(kind) if kind in KINDS else len(KINDS), int(m.group(2)) if m else 0, w)


def median(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else NAN


def mq(x):
    """Millions per second, to about three significant digits."""
    if math.isnan(x):
        return "-"
    x /= 1e6
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def spread(xs):
    """Median with min-max, over processes."""
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return mq(xs[0]) if xs else "-"
    return f"{mq(statistics.median(xs))} ({mq(min(xs))}–{mq(max(xs))})"


def times(a, b):
    if math.isnan(a) or math.isnan(b) or b <= 0:
        return "-"
    x = a / b
    return f"{x:.0f}×" if x >= 100 else f"{x:.1f}×" if x >= 1 else f"{x:.2f}×"


def sel(bs, engine, run, config, workload, key="qps_median"):
    """Values of `key` of the batches matching; `run` is a name or a predicate."""
    ok = run if callable(run) else (lambda r: r == run)
    return [b[key] for b in bs if b["engine"] == engine and ok(b["run"]) and b["config"] == config
            and b["workload"] == workload and key in b]


def is_main(run):
    return run.startswith("main-r")


def table(head, rows, align):
    return [f"| {' | '.join(head)} |", f"|{'|'.join(align)}|"] + [f"| {' | '.join(r)} |" for r in rows]


def agreement(bs, d):
    # Batches of the same queries: a batch is checked if infini-gram counted them in another batch too.
    same = {}
    for b in bs:
        if b.get("counts_hash"):
            same.setdefault((b["workload"], b.get("q")), []).append(b)
    agree, differ, unchecked = 0, [], 0
    for group in same.values():
        for b in group:
            ref = {x["counts_hash"] for x in group if x["engine"] == "ig" and x is not b}
            if not ref:
                unchecked += 1
            elif ref == {b["counts_hash"]}:
                agree += 1
            else:
                differ.append(f"{b['engine']} {b['run']} {b['config']} {b['workload']}")
    unchecked += sum(1 for b in bs if not b.get("counts_hash"))
    cert = [b.get("validate", "-") for b in bs if b["engine"] == "ours"]
    cmp_lines = (d / "compare.txt").read_text(encoding="utf-8").split() if (d / "compare.txt").exists() else []
    out = [f"- Count files of the first main processes: {cmp_lines.count('same')} same, "
           f"{cmp_lines.count('DIFFERENT')} DIFFERENT.",
           f"- Counts hashes against infini-gram's: {agree} batches agree, {len(differ)} differ, "
           f"{unchecked} without a reference.",
           f"- Certificate: {cert.count('pass')} of our batches pass, {cert.count('FAIL')} fail."]
    out += [f"  - differs: {x}" for x in differ]
    return out


def main_rows(bs, t_all):
    ig_cfg = f"direct-t{t_all}-p0-mmap"
    rows, speedups = [], []
    for w in sorted({b["workload"] for b in bs if is_main(b["run"])}, key=order):
        ig = sel(bs, "ig", is_main, ig_cfg, w)
        tp, e2e = sel(bs, "ours", is_main, "s32w2o0", w), sel(bs, "ours", is_main, "s32w2o0", w, "qps_e2e")
        q = next(b["q"] for b in bs if b["workload"] == w and is_main(b["run"]))
        speedups.append((median(tp) / median(ig) if ig else NAN, median(e2e) / median(ig) if ig else NAN))
        rows.append([w, f"{int(q):,}", spread(ig), spread(sel(bs, "ours", is_main, "s0w0o0", w)), spread(tp), spread(e2e),
                     times(median(tp), median(ig)), times(median(e2e), median(ig))])
    return rows, speedups


def batch_rows(bs, t_all):
    rows = []
    for w in sorted({b["workload"] for b in bs if b["run"] == "batch"}, key=order):
        q = int(re.sub(r"\D", "", w))
        t = min(t_all, (q + 255) // 256)
        ig = [median(sel(bs, "ig", "batch", f"direct-t{x}-p0-mmap", w)) for x in sorted({t_all, t})]
        ig_best = max((x for x in ig if not math.isnan(x)), default=NAN)
        tp = median(sel(bs, "ours", "batch", "s32w2o0", w))
        rows.append([f"{q:,}", mq(ig[-1]), mq(ig[0]) if t < t_all else "-", mq(median(sel(bs, "ours", "batch", "s0w0o0", w))),
                     mq(tp), mq(median(sel(bs, "ours", "batch", "s32w2o1", w))),
                     mq(median(sel(bs, "ours", "batch", "s32w2o0", w, "qps_e2e"))), times(tp, ig_best)])
    return rows


def ablation_rows(bs):
    abl = [b for b in bs if b["run"].startswith("ablation-")]
    any_run = lambda r: True
    ws = sorted({b["workload"] for b in abl}, key=order)
    # The default mode's throughput in the ablation, else in the main runs.
    base = {w: median(sel(abl, "ours", any_run, "s32w2o0", w)) for w in ws}
    base = {w: x if not math.isnan(x) else median(sel(bs, "ours", is_main, "s32w2o0", w)) for w, x in base.items()}
    rows = []
    for c in sorted({b["config"] for b in abl}, key=lambda c: tuple(map(int, re.findall(r"\d+", c)))):
        per = {w: median(sel(abl, "ours", any_run, c, w)) for w in ws}
        rel = median([per[w] / base[w] for w in ws if base[w] > 0])
        first = next(b for b in abl if b["config"] == c)
        rows.append([c, f"{num(first, 'table_bytes') / 1e9:.1f}", f"{num(first, 't_table'):.3f}",
                     "-" if math.isnan(rel) else f"{rel:.2f}"] + [mq(per[w]) if w in per else "-" for w in SHOWN])
    return rows


def thread_rows(bs):
    th = [b for b in bs if b["engine"] == "ig" and b["run"] == "threads"]
    ws = sorted({b["workload"] for b in th}, key=order)
    rows = [[str(t)] + [mq(median(sel(th, "ig", "threads", f"direct-t{t}-p0-mmap", w))) for w in ws]
            for t in sorted({int(b["threads"]) for b in th})]
    return ws, rows


def variant_rows(bs, t_all):
    ws = sorted({b["workload"] for b in bs if b["engine"] == "ig" and b["run"] in ("find", "prefetch", "heap")}, key=order)
    rows = []
    for w in ws:
        direct = median(sel(bs, "ig", is_main, f"direct-t{t_all}-p0-mmap", w))
        find = median(sel(bs, "ig", "find", f"find-t{t_all}-p0-mmap", w))
        rows.append([w, mq(direct), mq(find), mq(median(sel(bs, "ig", "prefetch", f"direct-t{t_all}-p1-mmap", w))),
                     mq(median(sel(bs, "ig", "heap", f"direct-t{t_all}-p0-ram", w))), times(find, direct)])
    return rows


def setup_lines(bs):
    """Load times: ours per process of the main runs; infini-gram's page-cache warm-up and heap loads."""
    procs = {}
    for b in bs:
        if b["engine"] == "ours" and is_main(b["run"]):
            procs.setdefault(b["run"], b)
    ours = [(num(b, "t_load_text"), num(b, "t_load_sa"), num(b, "t_spot")) for b in procs.values()]
    tab = [num(b, "t_table") for b in bs if b["engine"] == "ours" and is_main(b["run"]) and b["config"] == "s32w2o0"]
    warm = [num(b, "t_prewarm") for b in bs if b["engine"] == "ig" and num(b, "prewarm_bytes") > 0]
    heap = [num(b, "t_load") for b in bs if b["engine"] == "ig" and b["run"] == "heap"]
    out = []
    if ours:
        out.append(f"- Ours, per process (median of {len(ours)}): text {median([x[0] for x in ours]):.1f} s, "
                   f"suffix array {median([x[1] for x in ours]):.1f} s, order spot-check {median([x[2] for x in ours]):.3f} s; "
                   f"sample table (stride 32) {median(tab):.3f} s.")
    if warm:
        out.append(f"- infini-gram: page-cache warm-up {max(warm):.1f} s, once; index read into the heap "
                   f"{median(heap):.1f} s per process." if heap else f"- infini-gram: page-cache warm-up {max(warm):.1f} s, once.")
    return out


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    d = pathlib.Path(argv[1])
    env = dict(pairs(d / "env.txt")) if (d / "env.txt").exists() else {}
    t_all = int(env.get("threads", "0") or 0)
    bs, bad = load(d)

    cols = ("engine", "run", "config", "workload", "q", "sym_c", "reps", "t_h2d", "t_find_min", "t_find_median", "t_d2h",
            "qps_best", "qps_median", "qps_e2e", "counts_hash", "validate", "hits", "occurrences")
    with open(d / "runs.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for b in bs:
            f.write("\t".join(f"{b[c]:.6g}" if isinstance(b.get(c), float) else str(b.get(c, "")) for c in cols) + "\n")

    md = [f"# Timed query runs: {env.get('host', '?')}, {env.get('gpu', '?')}", "",
          f"Commit {env.get('commit', '?')}, job {env.get('job', '?')}, {env.get('date', '?')}. "
          f"{env.get('reps', '?')} repetitions per batch; infini-gram's pool {t_all} threads, its first repetition dropped; "
          f"ours bound with `{env.get('ours_numactl', 'none')}`. Throughput in millions of queries per second.", "",
          "## Correctness", ""] + agreement(bs, d) + [f"- Logs without results or with errors: {len(bad)}."] \
        + [f"  - {x}" for x in bad]

    rows, speedups = main_rows(bs, t_all)
    if rows:
        md += ["", "## Main: infini-gram's `_find_thread` against ours", "",
               "Median over processes (min-max). Ours without transfers: the search on device-resident queries; with "
               "transfers: plus the batch's copy to the device and the intervals' copy back, unoverlapped, from pageable "
               "memory.", ""]
        md += table(["workload", "q", "infini-gram", "ours, direct", "ours, two-phase", "two-phase with transfers",
                     "speedup", "with transfers"], rows, ["---", "--:", "--:", "--:", "--:", "--:", "--:", "--:"])
        sp = [x for x, _ in speedups if not math.isnan(x)]
        sp_e2e = [y for _, y in speedups if not math.isnan(y)]
        if sp and sp_e2e:
            md += ["", f"Two-phase over infini-gram: {times(min(sp), 1)} to {times(max(sp), 1)} across workloads, median "
                       f"{times(statistics.median(sp), 1)}; with transfers {times(min(sp_e2e), 1)} to {times(max(sp_e2e), 1)}, "
                       f"median {times(statistics.median(sp_e2e), 1)}."]

    rows = batch_rows(bs, t_all)
    if rows:
        md += ["", "## Batch size: present patterns, m = 10", "",
               "infini-gram with every core, and with a pool no larger than the batch's 256-query chunks; the speedup is "
               "two-phase's over the better of the two.", ""]
        md += table(["q", "infini-gram", "infini-gram, small pool", "ours, direct", "two-phase", "two-phase, sorted",
                     "two-phase with transfers", "speedup"], rows, ["--:"] * 8)

    rows = ablation_rows(bs)
    if rows:
        md += ["", "## Ablation: stride (s), key words (w), sorted batches (o)", "",
               "Relative: the median over workloads of the throughput relative to the default mode's, s32w2o0.", ""]
        md += table(["mode", "table GB", "build s", "relative"] + list(SHOWN), rows, ["---"] + ["--:"] * (3 + len(SHOWN)))

    ws, rows = thread_rows(bs)
    if rows:
        md += ["", "## infini-gram's pool size", ""] + table(["threads"] + ws, rows, ["--:"] * (1 + len(ws)))

    rows = variant_rows(bs, t_all)
    if rows:
        md += ["", "## infini-gram's variants", "",
               "`_find_thread` is the main runs'; `find` is the engine's API, which starts a thread per shard for every "
               "query; prefetching at the engine's default depths; the index read into the heap instead of mapped.", ""]
        md += table(["workload", "`_find_thread`", "`find`", "prefetching", "index in heap", "`find` / `_find_thread`"], rows,
                    ["---"] + ["--:"] * 5)

    lines = setup_lines(bs)
    if lines:
        md += ["", "## Setup", ""] + lines

    (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
