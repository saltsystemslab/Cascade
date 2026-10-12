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
and the counts of our batches failing the certificate, and of those whose
intervals through the pipeline differ from those found from device memory.

If the host searches our index, ours is reported end to end only: from device
memory, its batches go to the host and back. With our suffix array mapped from
its drive, the main runs are reported per cache state, cold (every repetition
first dropping the suffix array's pages, and infini-gram's table's) and hot,
with the pages that each engine read from storage per query; and the other runs
are cold.

Runs that time the phases of our calls (`query find --phase-times 1`) are
reported where the time goes: per batch, the busy time of every phase, summed
over its chunks. The memory that our processes report goes in the setup.
"""
import math
import pathlib
import re
import statistics
import sys

KEY = re.compile(r"([a-z][a-z0-9_]*)=(.*)")
MODE_KEYS = ("stride", "key_words", "key_symbols", "sort", "gather", "table_bytes", "t_table")
KINDS = ("present", "mutated", "random", "heldout", "unigram", "batch")
SHOWN = ("present-m1", "present-m10", "random-m5", "present-m100", "present-m1000")
NAN = math.nan
PLACED = {"mmap": "mapped", "ram": "in the heap"}
TIERS = {"device": "device memory", "pinned": "pinned host memory", "pageable": "pageable host memory",
         "mapped": "a mapped file on a node-local drive"}
PAGE = 4096
# The phases of our calls that `query find --phase-times 1` times, as it names them, and as the report does.
PHASES = (("stage", "staging"), ("h2d", "H2D"), ("rank", "rank"), ("search", "search"), ("d2h", "D2H"), ("prefetch", "prefetch"),
          ("host_search", "host search"), ("wait", "host waits"))
GB = 1e9


def pairs(path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = KEY.fullmatch(line)
        if m:
            yield m.group(1), m.group(2)


def batches(path, engine, run):
    """The batches of a log of `query find` or of infini-gram's driver, each with
    its load's keys, and ours with its mode's."""
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
        b.update(engine=engine, run=run, log=str(path))
        b["cache"] = "cold" if b.get("sa_cold" if engine == "ours" else "cold") == "1" else "hot"
        if engine == "ours" and b.get("stride") == "0":
            b["config"] = "s0w0o0"  # The direct mode neither sorts nor gathers, whatever the options of its run.
        elif engine == "ours":
            # Brackets are gathered iff the suffix array is pinned, unless a mode says otherwise; only then does its name say so.
            placed = "1" if b.get("sa_res") == "pinned" else "0"
            forced = b.get("gather", placed) != placed
            b["config"] = f"s{b.get('stride')}w{b.get('key_words')}o{b.get('sort')}" + (f"g{b['gather']}" if forced else "")
        else:
            b["config"] = f"{b.get('mode')}-t{b.get('threads')}-p{b.get('prefetch')}-{b.get('load')}"
    return [b for b in out if "qps" in b]


def num(b, k):
    try:
        return float(b[k])
    except (KeyError, ValueError):
        return NAN


def load(d):
    """The batches of every log under `d`, and the logs that hold none or report errors."""
    bs, bad = [], []
    logs = [(p, "ours", p.parent.name if p.name == "find.log" else p.stem)
            for p in sorted((d / "ours").glob("*.log")) + sorted((d / "ours").glob("*/find.log"))]
    logs += [(p, "ig", p.parent.name) for p in sorted((d / "ig").glob("*/*.log"))]
    for p, engine, run in logs:
        got = batches(p, engine, run)
        errors = [line for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("error")]
        if not got or errors:
            bad.append(f"{p.relative_to(d)}: {errors[0] if errors else 'no results'}")
        bs += got
    for b in bs:
        b["workload"] = pathlib.Path(b.get("queries", "?")).stem
        q, med, best = num(b, "q"), num(b, "t_find_median"), num(b, "t_find_min")
        b["qps_median"] = q / med if med > 0 else NAN
        b["qps_best"] = q / best if best > 0 else NAN
        if b["engine"] == "ours" and "t_e2e_median" in b:  # Through the pipeline, from pinned memory, and from pageable.
            e2e, pageable = num(b, "t_e2e_median"), num(b, "t_e2e_pageable_median")
            b["qps_e2e"] = q / e2e if e2e > 0 else NAN
            b["qps_e2e_pageable"] = q / pageable if pageable > 0 else NAN
        elif b["engine"] == "ours":    # From before the pipeline: the transfers and the search, unoverlapped, from pageable memory.
            t = num(b, "t_h2d") + med + num(b, "t_d2h")
            b["qps_e2e"] = q / t if t > 0 else NAN
        # Pages read from storage per query: ours end to end from pinned memory; infini-gram's but its first repetition, which its
        # times drop too.
        io = [int(x) for x in b.get("io_bytes_e2e" if b["engine"] == "ours" else "io_bytes", "").split(",") if x]
        if b["engine"] == "ig" and len(io) > 1:
            io = io[1:]
        b["pages"] = median(io) / PAGE / q if io and q > 0 else NAN
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


def ms(x):
    """Seconds as milliseconds, to about three significant digits; `-` if none."""
    if math.isnan(x) or x <= 0:
        return "-"
    x *= 1e3
    return f"{x:.{max(0, 2 - math.floor(math.log10(x)))}f}"


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
    return re.fullmatch(r"main(-cold|-hot)?-r\d+", run) is not None


def pages(x):
    return "-" if math.isnan(x) else f"{x:.2f}"


def table(head, rows, align):
    return [f"| {' | '.join(head)} |", f"|{'|'.join(align)}|"] + [f"| {' | '.join(r)} |" for r in rows]


def filled(head, rows, align):
    """The table without the columns that hold no value: runs that a campaign does not make."""
    keep = [i for i in range(len(head)) if i == 0 or any(r[i] != "-" for r in rows)]
    return [head[i] for i in keep], [[r[i] for i in keep] for r in rows], [align[i] for i in keep]


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
    piped = [b["e2e_check"] for b in bs if b["engine"] == "ours" and "e2e_check" in b]
    if piped:
        out.append(f"- Pipeline: {piped.count('pass')} of our batches' intervals equal those from device memory, "
                   f"{piped.count('FAIL')} differ.")
    return out


def main_rows(bs, t_all, ig_load, piped, two, host):
    """With the host searching, ours end to end only: from pinned memory, and from pageable."""
    ig_cfg = f"direct-t{t_all}-p0-{ig_load}"
    rows, speedups = [], []
    for w in sorted({b["workload"] for b in bs if is_main(b["run"])}, key=order):
        ig = sel(bs, "ig", is_main, ig_cfg, w)
        tp, e2e = sel(bs, "ours", is_main, two, w), sel(bs, "ours", is_main, two, w, "qps_e2e")
        pageable = sel(bs, "ours", is_main, two, w, "qps_e2e_pageable")
        q = next(b["q"] for b in bs if b["workload"] == w and is_main(b["run"]))
        if host:
            speedups.append((median(e2e) / median(ig) if ig else NAN, median(pageable) / median(ig) if ig else NAN))
            rows.append([w, f"{int(q):,}", spread(ig), spread(sel(bs, "ours", is_main, "s0w0o0", w, "qps_e2e")), spread(e2e), spread(pageable),
                         times(median(e2e), median(ig)), times(median(pageable), median(ig))])
            continue
        speedups.append((median(tp) / median(ig) if ig else NAN, median(e2e) / median(ig) if ig else NAN))
        rows.append([w, f"{int(q):,}", spread(ig), spread(sel(bs, "ours", is_main, "s0w0o0", w)), spread(tp), spread(e2e)]
                    + ([spread(pageable)] if piped else []) + [times(median(tp), median(ig)), times(median(e2e), median(ig))])
    return rows, speedups


def batch_rows(bs, t_all, ig_load, host, two="s32w2o0"):
    """With the host searching, ours end to end only: the direct mode, two phases unsorted and sorted, and unsorted from pageable memory.
    Two phases in the stride and key width of `two`."""
    rows = []
    two0, two1 = two[:two.index("o")] + "o0", two[:two.index("o")] + "o1"
    best = lambda xs: max((x for x in xs if not math.isnan(x)), default=NAN)
    for w in sorted({b["workload"] for b in bs if b["run"] == "batch"}, key=order):
        q = int(re.sub(r"\D", "", w))
        t = min(t_all, (q + 255) // 256)
        ig = [median(sel(bs, "ig", "batch", f"direct-t{x}-p0-{ig_load}", w)) for x in sorted({t_all, t})]
        ig_best = max((x for x in ig if not math.isnan(x)), default=NAN)
        tp = [median(sel(bs, "ours", "batch", c, w)) for c in (two0, two1)]
        e2e = [median(sel(bs, "ours", "batch", c, w, "qps_e2e")) for c in ("s0w0o0", two0, two1)]
        igs = [mq(ig[-1]), mq(ig[0]) if t < t_all else "-"]
        if host:
            pageable = median(sel(bs, "ours", "batch", two0, w, "qps_e2e_pageable"))
            rows.append([f"{q:,}"] + igs + [mq(x) for x in e2e] + [mq(pageable), times(best(e2e), ig_best), times(pageable, ig_best)])
            continue
        rows.append([f"{q:,}"] + igs + [mq(median(sel(bs, "ours", "batch", "s0w0o0", w))), mq(tp[0]), mq(tp[1]), mq(e2e[0]), mq(e2e[1]),
                     mq(e2e[2]), times(best(tp), ig_best), times(best(e2e), ig_best)])
    return rows


def ablation_rows(bs, key, two="s32w2o0"):
    abl = [b for b in bs if b["run"].startswith("ablation-")]
    any_run = lambda r: True
    ws = sorted({b["workload"] for b in abl}, key=order)
    # The main runs' mode's throughput in the ablation, else in the main runs.
    base = {w: median(sel(abl, "ours", any_run, two, w, key)) for w in ws}
    base = {w: x if not math.isnan(x) else median(sel(bs, "ours", is_main, two, w, key)) for w, x in base.items()}
    rows = []
    for c in sorted({b["config"] for b in abl}, key=lambda c: tuple(map(int, re.findall(r"\d+", c)))):
        per = {w: median(sel(abl, "ours", any_run, c, w, key)) for w in ws}
        rel = median([per[w] / base[w] for w in ws if base[w] > 0])
        first = next(b for b in abl if b["config"] == c)
        rows.append([c, f"{num(first, 'table_bytes') / 1e9:.1f}", f"{num(first, 't_table'):.3f}",
                     "-" if math.isnan(rel) else f"{rel:.2f}"] + [mq(per[w]) if w in per else "-" for w in SHOWN])
    return rows


def host_rows(bs, prefix, knob):
    """Our host search's sweep of the runs `<prefix><x>`, over `knob`: end to end from pinned memory, the direct mode and two phases."""
    runs = [b for b in bs if b["engine"] == "ours" and re.fullmatch(prefix + r"\d+", b["run"])]
    ws = sorted({b["workload"] for b in runs}, key=order)
    rows = []
    for x in sorted({int(b[knob]) for b in runs if knob in b}):
        at = [b for b in runs if b.get(knob) == str(x)]
        rows.append([str(x)] + [mq(median(sel(at, "ours", lambda r: True, c, w, "qps_e2e"))) for w in ws for c in ("s0w0o0", "s32w2o0")])
    return ws, rows


def thread_rows(bs, ig_load):
    th = [b for b in bs if b["engine"] == "ig" and b["run"] == "threads"]
    ws = sorted({b["workload"] for b in th}, key=order)
    rows = [[str(t)] + [mq(median(sel(th, "ig", "threads", f"direct-t{t}-p0-{ig_load}", w))) for w in ws]
            for t in sorted({int(b["threads"]) for b in th})]
    return ws, rows


def variant_rows(bs, t_all, ig_load, other):
    placement = lambda r: r in ("placement", "heap")  # `heap`: its name in results of before the placement was chosen.
    ws = sorted({b["workload"] for b in bs if b["engine"] == "ig" and (b["run"] in ("find", "prefetch") or placement(b["run"]))},
                key=order)
    rows = []
    for w in ws:
        direct = median(sel(bs, "ig", is_main, f"direct-t{t_all}-p0-{ig_load}", w))
        find = median(sel(bs, "ig", "find", f"find-t{t_all}-p0-{ig_load}", w))
        rows.append([w, mq(direct), mq(find), mq(median(sel(bs, "ig", "prefetch", f"direct-t{t_all}-p1-mmap", w))),
                     mq(median(sel(bs, "ig", placement, f"direct-t{t_all}-p0-{other}", w))), times(find, direct)])
    return rows


def ssd_main_rows(bs, cache, t_all, two):
    """Ours end to end from pinned memory against infini-gram, in the cache state `cache`, with the pages read per query."""
    at = [b for b in bs if b["cache"] == cache and is_main(b["run"])]
    rows, speedups = [], []
    for w in sorted({b["workload"] for b in at}, key=order):
        ig = sel(at, "ig", is_main, f"direct-t{t_all}-p0-mmap", w)
        ours = sel(at, "ours", is_main, two, w, "qps_e2e")
        q = next(b["q"] for b in at if b["workload"] == w)
        speedups.append(median(ours) / median(ig) if ig else NAN)
        rows.append([w, f"{int(q):,}", spread(ig), spread(ours), times(median(ours), median(ig)),
                     pages(median(sel(at, "ig", is_main, f"direct-t{t_all}-p0-mmap", w, "pages"))),
                     pages(median(sel(at, "ours", is_main, two, w, "pages")))])
    return rows, speedups


def ssd_ablation_rows(bs, two):
    """Ours cold, end to end from pinned memory: the main runs' mode, not asking for pages ahead, the suffix array split over the
    drives, asking and not, and the direct mode; each with its pages read per query."""
    runs = (("main", lambda r: is_main(r), two), ("prefetch-p0", lambda r: r == "prefetch-p0", two),
            ("stripe-p1", lambda r: r == "stripe-p1", two), ("stripe-p0", lambda r: r == "stripe-p0", two),
            ("direct", lambda r: r == "direct", "s0w0o0"))
    cold = [b for b in bs if b["engine"] == "ours" and b["cache"] == "cold"]
    present = [name for name, run, c in runs if any(run(b["run"]) and b["config"] == c for b in cold)]
    rows = []
    for w in sorted({b["workload"] for b in cold if any(run(b["run"]) for _, run, _ in runs)}, key=order):
        row = [w]
        for name, run, c in runs:
            if name in present:
                row += [mq(median(sel(cold, "ours", run, c, w, "qps_e2e"))), pages(median(sel(cold, "ours", run, c, w, "pages")))]
        rows.append(row)
    return present, rows


def phased(b):
    return b["engine"] == "ours" and b.get("phase_times") == "1" and "t_phase_wall_median" in b


def phase_rows(bs, cache, pageable):
    """Ours in the batches that time their phases, in the cache state `cache` (any if None), end to end from pinned memory, or from
    pageable if `pageable`: per workload and mode, the median over such batches of each phase's median time, and of the wall time, over
    the main runs' end to end."""
    at = [b for b in bs if phased(b) and (cache is None or b["cache"] == cache)]
    sfx, e2e = ("_pageable", "t_e2e_pageable_median") if pageable else ("", "t_e2e_median")
    rows = []
    for w in sorted({b["workload"] for b in at}, key=order):
        for c in sorted({b["config"] for b in at if b["workload"] == w}, key=lambda c: (c == "s0w0o0", c)):
            got = [b for b in at if b["workload"] == w and b["config"] == c]
            main = median([num(b, e2e) for b in bs if b["engine"] == "ours" and is_main(b["run"]) and b["config"] == c and b["workload"] == w
                           and (cache is None or b["cache"] == cache)])
            vs = median([num(b, e2e) for b in got]) / main if main > 0 else NAN
            rows.append([w, c, f"{int(got[0]['q']):,}", got[0].get("phase_chunks", "-"), ms(median([num(b, f"t_phase_wall{sfx}_median") for b in got]))]
                        + [ms(median([num(b, f"t_phase_{p}{sfx}_median") for b in got])) for p, _ in PHASES]
                        + ["-" if math.isnan(vs) else f"{vs:.2f}"])
    return rows


def phase_lines(bs, cache=None, explain=True):
    """Where our time goes, in the batches that time their phases: from pinned memory, and from pageable; explained if `explain`."""
    md = []
    for pageable in (False, True):
        rows = phase_rows(bs, cache, pageable)
        if not rows:
            continue
        where = "pageable" if pageable else "pinned"
        md += ["", f"## Where the time goes{', ' + cache if cache else ''}: ours end to end, from {where} memory", ""]
        if explain and not pageable:
            md += ["A process of ours that times the phases of every call (`--phase-times 1`), over the main workloads in the main runs' "
                   "modes. Per batch, the median over repetitions of each phase's busy time in ms, summed over the batch's chunks, and "
                   "over the host's threads that run them: the device's between events on its streams, the host's on its clock. Staging: "
                   "the host's copies between the batch's pageable memory and pinned buffers; rank: packing, sorting, and ranking in the "
                   "sample table; search: the device's exact search, with its gathers; prefetch: the host's asking the kernel for a mapped "
                   "suffix array's pages ahead; host waits: on the device. Phases overlap across the streams and the host, so that they "
                   "may add up to more than the wall time. Last column: the process's end-to-end time over the main runs' median, the "
                   "cost of the timing, within the noise of processes.", ""]
        md += table(*filled(["workload", "mode", "q", "chunks", "wall"] + [name for _, name in PHASES] + ["/ main runs"], rows,
                            ["---", "---"] + ["--:"] * (4 + len(PHASES))))
    return md


def memory_lines(bs):
    """Our processes' memory, per cache state: the median over the main runs' processes, or else those of any run. Of the engine's own
    device and pinned memory, the parts that hold any."""
    procs = {}
    for b in bs:
        if b["engine"] == "ours" and "mem_device_used_load" in b:
            p = procs.setdefault(b["log"], {"cache": b["cache"], "main": is_main(b["run"])})
            p.update((k, num(b, k)) for k in b if k.startswith("mem_"))
    out = []
    for cache in sorted({p["cache"] for p in procs.values()}):
        at = [p for p in procs.values() if p["cache"] == cache]
        at = [p for p in at if p["main"]] or at

        def gb(k):
            x = median([p.get(f"mem_{k}", NAN) for p in at]) / GB
            return "-" if math.isnan(x) else f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"

        def parts(names):
            return ", ".join(f"{name} {gb(k)}" for k, name in names if median([p.get(f"mem_{k}", NAN) for p in at]) > 0)

        device = parts((("text_device_end", "text"), ("sa_device_end", "suffix array"), ("table_device_end", "sample table"),
                        ("lanes_device_end", "lanes"), ("pool_used_high_end", "the memory pool's high-water mark")))
        pinned = parts((("text_pinned_end", "text"), ("sa_pinned_end", "suffix array"), ("lanes_pinned_end", "lanes")))
        state = f", {cache}" if any(b.get("sa_res") == "mapped" for b in bs) else ""
        out.append(f"- Ours, memory per process{state} (median of {len(at)}{' main' if at[0]['main'] else ''} processes), GB: the device's in use "
                   f"{gb('device_used_load')} after the load and {gb('device_used_end')} after the last batch, of which the engine's own "
                   f"{gb('engine_device_end')}" + (f" ({device})" if device else "") + f", its memory pool having reserved at most "
                   f"{gb('pool_reserved_high_end')}; the engine's pinned host memory {gb('engine_pinned_end')}" + (f" ({pinned})" if pinned else "")
                   + f". Peak resident set {gb('peak_rss_end')}; at the end anonymous {gb('rss_anon_end')}, file-backed {gb('rss_file_end')}, "
                   f"shared {gb('rss_shmem_end')}.")
    return out


def setup_lines(bs, two):
    """Load times: ours per process of the main runs; infini-gram's page-cache warm-up and heap loads."""
    procs = {}
    for b in bs:
        if b["engine"] == "ours" and is_main(b["run"]):
            procs.setdefault(b["run"], b)
    ours = [(num(b, "t_load_text"), num(b, "t_load_sa"), num(b, "t_spot")) for b in procs.values()]
    tab = [num(b, "t_table") for b in bs if b["engine"] == "ours" and is_main(b["run"]) and b["config"] == two]
    stride = next((b.get("stride") for b in bs if b["engine"] == "ours" and b["config"] == two), "?")
    warm = [num(b, "t_prewarm") for b in bs if b["engine"] == "ig" and num(b, "prewarm_bytes") > 0]
    lock = list({b["log"]: num(b, "t_lock") for b in bs if b["engine"] == "ig" and b.get("lock_text") == "1"}.values())
    heap = list({b["log"]: num(b, "t_load") for b in bs if b["engine"] == "ig" and b.get("load") == "ram"}.values())
    out = []
    if ours:
        out.append(f"- Ours, per process (median of {len(ours)}): text {median([x[0] for x in ours]):.1f} s, "
                   f"suffix array {median([x[1] for x in ours]):.1f} s, order spot-check {median([x[2] for x in ours]):.3f} s; "
                   f"sample table (stride {stride}) {median(tab):.3f} s.")
    if warm:
        out.append(f"- infini-gram: page-cache warm-up {max(warm):.1f} s, once; index read into the heap "
                   f"{median(heap):.1f} s per process." if heap else f"- infini-gram: page-cache warm-up {max(warm):.1f} s, once.")
    if lock:
        out.append(f"- infini-gram: text read in and locked in memory in {median(lock):.1f} s per process (median of {len(lock)}).")
    return out + memory_lines(bs)


def ssd_lines(bs, t_all, two):
    """The report of runs with our suffix array mapped from its drive."""
    md = []
    for cache in ("cold", "hot"):
        rows, speedups = ssd_main_rows(bs, cache, t_all, two)
        if not rows:
            continue
        md += ["", f"## Main, {cache}: infini-gram's `_find_thread` against ours", "",
               "Median over processes (min-max). Ours end to end from pinned memory, in two phases (" + two + "): the device ranks "
               "every chunk in the sample table, the host asks the kernel for the pages of its brackets, and the host's cores search "
               "them. " + ("Every repetition starts with none of the suffix array's pages in the page cache, nor of infini-gram's "
               "table." if cache == "cold" else "The repetitions after the first find their pages cached.") + " Pages: read from "
               "storage per query, median over repetitions.", ""]
        md += table(["workload", "q", "infini-gram", "ours", "speedup", "pages, infini-gram", "pages, ours"], rows, ["---"] + ["--:"] * 6)
        sp = [x for x in speedups if not math.isnan(x)]
        if sp:
            md += ["", f"Ours over infini-gram, {cache}: {times(min(sp), 1)} to {times(max(sp), 1)} across workloads, median "
                       f"{times(statistics.median(sp), 1)}."]

    explained = False
    for cache in ("cold", "hot"):
        lines = phase_lines(bs, cache, not explained)
        explained = explained or bool(lines)
        md += lines

    cold = [b for b in bs if b["cache"] == "cold"]
    rows = batch_rows(cold, t_all, "mmap", True, two)
    if rows:
        md += ["", "## Batch size, cold: present patterns, m = 10", "",
               "infini-gram with every core, and with a pool no larger than the batch's 256-query chunks; ours end to end. The "
               "speedups are over the better of infini-gram's two.", ""]
        md += table(*filled(["q", "infini-gram", "infini-gram, small pool", "ours, direct", "two-phase", "two-phase, sorted",
                             "two-phase, pageable", "speedup", "pageable"], rows, ["--:"] * 9))

    present, rows = ssd_ablation_rows(bs, two)
    if len(present) > 1:
        names = {"main": "one drive", "prefetch-p0": "one drive, not asking ahead", "stripe-p1": "two drives",
                 "stripe-p0": "two drives, not asking ahead", "direct": "direct mode"}
        md += ["", "## Our suffix array's drive, cold", "",
               "Ours end to end from pinned memory, and the pages read per query: the suffix array on one drive, asking the kernel "
               "for the brackets' pages ahead (the main runs) and not; split over two drives, asking and not; and the direct mode, "
               "which binary-searches the whole suffix array with no table, on a prefix of the queries.", ""]
        md += table(["workload"] + [x for n in present for x in (names[n], "pages")], rows, ["---"] + ["--:"] * (2 * len(present)))

    rows = ablation_rows(cold, "qps_e2e", two)
    if rows:
        md += ["", "## Ablation, cold: stride (s), key words (w)", "",
               f"Relative: the median over workloads of the throughput relative to the main runs' mode, {two}, end to end.", ""]
        md += table(["mode", "table GB", "build s", "relative"] + list(SHOWN), rows, ["---"] + ["--:"] * (3 + len(SHOWN)))

    for prefix, knob, title in (("host-l", "host_lanes", "searches each thread interleaves"), ("host-t", "host_threads", "threads")):
        runs = [b for b in bs if b["engine"] == "ours" and re.fullmatch(prefix + r"\d+", b["run"])]
        ws = sorted({b["workload"] for b in runs}, key=order)
        rows = [[str(x)] + [mq(median(sel([b for b in runs if b.get(knob) == str(x)], "ours", lambda r: True, two, w, "qps_e2e"))) for w in ws]
                for x in sorted({int(b[knob]) for b in runs if knob in b})]
        if rows:
            md += ["", f"## Our host search, cold: {title}", "", f"End to end from pinned memory, in {two}.", ""]
            md += table([knob.split("_")[1]] + ws, rows, ["--:"] * (1 + len(ws)))

    ws, rows = thread_rows(bs, "mmap")
    if rows:
        md += ["", "## infini-gram's pool size, cold", ""] + table(["threads"] + ws, rows, ["--:"] * (1 + len(ws)))

    rows = variant_rows(cold, t_all, "mmap", "ram")
    if rows:
        md += ["", "## infini-gram's variants, cold", "",
               "`_find_thread` is the main runs'; `find` is the engine's API, which starts a thread per shard for every query; "
               "prefetching is at the engine's default depths, on the first 1e5 queries.", ""]
        md += table(*filled(["workload", "`_find_thread`", "`find`", "prefetching", f"`_find_thread`, {PLACED['ram']}",
                             "`find` / `_find_thread`"], rows, ["---"] + ["--:"] * 5))

    lines = setup_lines(bs, two)
    if lines:
        md += ["", "## Setup", ""] + lines
    return md


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    d = pathlib.Path(argv[1])
    env = dict(pairs(d / "env.txt")) if (d / "env.txt").exists() else {}
    t_all = int(env.get("threads", "0") or 0)
    ig_load = env.get("ig_load", "mmap")    # Absent from results of before the placement was chosen.
    other = "mmap" if ig_load == "ram" else "ram"
    bs, bad = load(d)

    cols = ("engine", "run", "config", "workload", "q", "sym_c", "reps", "t_h2d", "t_find_min", "t_find_median", "t_d2h",
            "qps_best", "qps_median", "t_e2e_median", "qps_e2e", "t_e2e_pageable_median", "qps_e2e_pageable", "e2e_check",
            "counts_hash", "validate", "hits", "occurrences")
    if any(phased(b) for b in bs):  # Only then: the columns of results without them stay as they were.
        cols += ("phase_chunks",) + tuple(f"t_phase_{p}{x}_median" for x in ("", "_pageable") for p in [p for p, _ in PHASES] + ["wall"])
    with open(d / "runs.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for b in bs:
            f.write("\t".join(f"{b[c]:.6g}" if isinstance(b.get(c), float) else str(b.get(c, "")) for c in cols) + "\n")

    text, sa = (TIERS[env.get(k, "device")] for k in ("ours_text_res", "ours_sa_res"))
    placement = f"both its arrays in {text}" if text == sa else f"its text in {text} and its suffix array in {sa}"
    # The host searches our index: ours is end to end only. Its threads and lanes are those of the main runs.
    host = any(b.get("host_search") == "1" for b in bs if b["engine"] == "ours")
    if host:
        first = next((b for b in bs if b["engine"] == "ours" and is_main(b["run"])), {})
        placement += f", searched by the host's {first.get('host_threads', '?')} threads of {first.get('host_lanes', '?')} lanes each"
    ig_numa = env.get("ig_numactl", "none")
    md = [f"# Timed query runs: {env.get('host', '?')}, {env.get('gpu', '?')}", "",
          f"Commit {env.get('commit', '?')}, job {env.get('job', '?')}, {env.get('date', '?')}. "
          f"{env.get('reps', '?')} repetitions per batch; infini-gram's pool {t_all} threads, its index {PLACED[ig_load]}, "
          f"its first repetition dropped" + (f", bound with `{ig_numa}`" if ig_numa != "none" else "") + "; "
          f"ours bound with `{env.get('ours_numactl', 'none')}`, {placement}. Throughput in millions of queries per second.", "",
          "## Correctness", ""] + agreement(bs, d) + [f"- Logs without results or with errors: {len(bad)}."] \
        + [f"  - {x}" for x in bad]

    # End to end: through the pipeline if the results have it; from before it, the transfers and the search added up.
    piped = any("t_e2e_median" in b for b in bs)
    e2e = "end to end" if piped else "with transfers"
    # Our two-phase mode of the main runs: unsorted if they ran it, else sorted; stride 32 and 2-word keys if they ran several.
    modes = {b["config"] for b in bs if b["engine"] == "ours" and is_main(b["run"]) and b["config"] != "s0w0o0"}
    two = next((c for c in ("s32w2o0", "s32w2o1") if c in modes), None) or min(modes, key=lambda c: (not c.endswith("o0"), c), default="s32w2o0")
    mapped = any(b.get("sa_res") == "mapped" for b in bs if b["engine"] == "ours")
    if mapped:
        md += ssd_lines(bs, t_all, two)
        (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        return 0
    rows, speedups = main_rows(bs, t_all, ig_load, piped, two, host)
    sorted_ = ", sorted" if two.endswith("o1") else ""
    if rows and host:
        md += ["", "## Main: infini-gram's `_find_thread` against ours", "",
               "Median over processes (min-max). Ours end to end: from the batch in host memory to its intervals in host memory; the "
               "device ranks every chunk in the sample table, and the host's cores search the brackets over the arrays, while the "
               "device ranks the next chunks. The direct mode runs on the host alone. From pinned memory, and from pageable "
               "memory in its own column.", ""]
        md += table(["workload", "q", f"infini-gram, {PLACED[ig_load]}", "ours, direct", f"ours, two-phase{sorted_}", "two-phase, pageable",
                     "speedup", "pageable"], rows, ["---"] + ["--:"] * 7)
    elif rows:
        md += ["", "## Main: infini-gram's `_find_thread` against ours", "",
               "Median over processes (min-max). Ours from device memory: the search on device-resident queries. "
               + ("End to end: from the batch in host memory to its intervals in host memory, through the pipeline of chunks "
                  "over streams; from pinned memory, and from pageable memory in its own column." if piped else
                  "With transfers: plus the batch's copy to the device and the intervals' copy back, unoverlapped, from pageable "
                  "memory."), ""]
        md += table(["workload", "q", f"infini-gram, {PLACED[ig_load]}", "ours, direct", f"ours, two-phase{sorted_}", f"two-phase, {e2e}"]
                    + (["end to end, pageable"] if piped else []) + ["speedup", e2e], rows, ["---"] + ["--:"] * (7 + piped))
    sp = [x for x, _ in speedups if not math.isnan(x)]
    sp_e2e = [y for _, y in speedups if not math.isnan(y)]
    if rows and sp and sp_e2e:
        md += ["", (f"Two-phase over infini-gram, end to end: {times(min(sp), 1)} to {times(max(sp), 1)} across workloads, median "
                    f"{times(statistics.median(sp), 1)}; from pageable memory {times(min(sp_e2e), 1)} to {times(max(sp_e2e), 1)}, median "
                    f"{times(statistics.median(sp_e2e), 1)}.") if host else
                   (f"Two-phase over infini-gram: {times(min(sp), 1)} to {times(max(sp), 1)} across workloads, median "
                    f"{times(statistics.median(sp), 1)}; {e2e} {times(min(sp_e2e), 1)} to {times(max(sp_e2e), 1)}, "
                    f"median {times(statistics.median(sp_e2e), 1)}.")]

    md += phase_lines(bs)

    rows = batch_rows(bs, t_all, ig_load, host)
    if rows and host:
        md += ["", "## Batch size: present patterns, m = 10", "",
               "infini-gram with every core, and with a pool no larger than the batch's 256-query chunks. Ours end to end. The "
               "speedups are over the better of infini-gram's two: our best mode's from pinned memory, and two phases from pageable.", ""]
        md += table(["q", "infini-gram", "infini-gram, small pool", "ours, direct", "two-phase", "two-phase, sorted", "two-phase, pageable",
                     "speedup", "pageable"], rows, ["--:"] * 9)
    elif rows:
        md += ["", "## Batch size: present patterns, m = 10", "",
               "infini-gram with every core, and with a pool no larger than the batch's 256-query chunks. The speedups are over "
               f"the better of the two: our better two-phase mode's from device memory, and our best mode's {e2e}.", ""]
        md += table(["q", "infini-gram", "infini-gram, small pool", "ours, direct", "two-phase", "two-phase, sorted", f"direct, {e2e}",
                     f"two-phase, {e2e}", f"sorted, {e2e}", "speedup", e2e], rows, ["--:"] * 11)

    rows = ablation_rows(bs, "qps_e2e" if host else "qps_median")
    if rows:
        md += ["", "## Ablation: stride (s), key words (w), sorted batches (o)", "",
               "Relative: the median over workloads of the throughput relative to the default mode's, s32w2o0"
               + (", end to end." if host else "."), ""]
        md += table(["mode", "table GB", "build s", "relative"] + list(SHOWN), rows, ["---"] + ["--:"] * (3 + len(SHOWN)))

    for prefix, knob, title in (("host-l", "host_lanes", "searches each thread interleaves"), ("host-t", "host_threads", "threads")):
        ws, rows = host_rows(bs, prefix, knob)
        if rows:
            md += ["", f"## Our host search: {title}", "",
                   "End to end from pinned memory: the direct mode (s0), and two phases (s32).", ""]
            md += table([knob.split("_")[1]] + [f"{w}, {c}" for w in ws for c in ("s0", "s32")], rows, ["--:"] * (1 + 2 * len(ws)))

    ws, rows = thread_rows(bs, ig_load)
    if rows:
        md += ["", "## infini-gram's pool size", ""] + table(["threads"] + ws, rows, ["--:"] * (1 + len(ws)))

    rows = variant_rows(bs, t_all, ig_load, other)
    if rows:
        md += ["", "## infini-gram's variants", "",
               f"`_find_thread` with the index {PLACED[ig_load]} is the main runs'; `find` is the engine's API, which starts a "
               "thread per shard for every query; prefetching is at the engine's default depths, with the index mapped, on the "
               f"first 1e5 queries; the last `_find_thread` has the index {PLACED[other]}.", ""]
        md += table(["workload", f"`_find_thread`, {PLACED[ig_load]}", "`find`", "prefetching", f"`_find_thread`, {PLACED[other]}",
                     "`find` / `_find_thread`"], rows, ["---"] + ["--:"] * 5)

    lines = setup_lines(bs, two)
    if lines:
        md += ["", "## Setup", ""] + lines

    (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
