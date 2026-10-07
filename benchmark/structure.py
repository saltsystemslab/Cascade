#!/usr/bin/env python3
"""Tabulate the structural statistics of GPU-SACA runs, for the paper's figures.

    structure.py <out-dir> <corpus>.log... [--sweep <corpus>_<label>.log...]
                 [--filtered <corpus>_<rep>.log...]

Each run log is a `caps_sa_gpu` STATS run of one corpus, and is named after it.
A `<corpus>.structure.json` next to it is `sa_structure`'s output for that run:
it adds the repeat-depth, look-back and run statistics, each checked against the
log exactly. The logs after `--sweep` are partition-only runs (with
`GPU_SACA_PARTITION_ONLY=1`), named `<corpus>_<label>.log`: the labels `p...`
form the partition-count sweep, and `k<K>` the pivot-width sweep. The logs after
`--filtered` are runs with the liveness filter, named `<corpus>_<rep>.log`, one
per replicate; the `<corpus>.structure.json` next to them is `sa_structure`'s
output for their partitioning, whose unfiltered messages they are held against.

Writes pgfplots-ready TSVs into <out-dir>, with `nan` for what is unavailable:
    decay.tsv           the active set after each generation: the repeat-depth
                        bounds S(c_t), S(c_t + K), the tool's prediction, the
                        tertiary shift K'_t, and the skew over the partitions
    depth_ccdf.tsv      the repeat-depth CCDF S(c)/n, log-thinned
    amplification.tsv   each generation's amplification, against its decay bounds
    lookback.tsv        each generation's cohort by look-back channel, and its
                        forwarded-pill stream, from the tool and from the log
    lookback_hist.tsv   each generation's full look-back histogram
    runs.tsv            the runs of consecutive text-positions in each active set
    skew.tsv            the partition sizes of every run and sweep point, against
                        the size bound B; the `p...` and `k...` points also go to
                        skew_p.tsv and skew_k.tsv
    partition_sizes.tsv each partition's size over B, for CDFs
    datasets.tsv        one row per corpus; also typeset as datasets.tex
    budget.tsv          each corpus' messages per suffix, live and dead, in order of E[d],
                        and the dead ones the liveness filter skips
    filter.tsv          each generation's messages with the liveness filter, over the
                        replicates, against the unfiltered and the live ones
    kgram.tsv           each corpus' heaviest K-gram for every K, against its run's size
                        bound B; the skew rows and datasets.tsv also carry it at their K
    checks.txt          the consistency checks, also printed
Exits 1 if any check fails.

Notation follows the theory writeup. A^(t) is the active set after generation t,
at context c_t = ell * 2^t. Generation t's cohort is P^(t) = A^(t-1), with
A^(-1) = [0, n). M_g is the inbox that generation g reads, i.e. what generation
g - 1 emitted. A = sum_t |P^(t)| is the total cohort mass, and M = sum_g M_g. A
query of generation t has look-back lambda = t - d(target) if its target
retired at generation d(target) < t, and 0 otherwise.
"""
import bisect
import json
import math
import pathlib
import re
import sys

RE_N = re.compile(r"Sequence length: (\d+)")
RE_STAT = {k: re.compile(rf"\t{k}:\s+(\d+)") for k in ("Size", "Minimum", "Maximum")}
# Unanchored, as a warning without a trailing newline can glue itself to the
# front of the next line.
RE_ACTIVE = re.compile(r"fully-sorted by context-size (\d+): (\d+)\.(?: Time taken in iteration: \S+s\.)?"
                       r" Out of (\d+) suffixes, (\d+) \(")
RE_INBOX = re.compile(r"Total inbox-size for generation (\d+): (\d+); number of recipients: (\d+)\."
                      r" Amplification: (\S+)x\.")
RE_BOUND = re.compile(r"Partition-size bound: (\d+) suffixes, from minimum partition-count (\d+)"
                      r" \(device-budget floor: (\d+)\)\.")
RE_PIVOTS = re.compile(r"Pivots selection: (\d+) partitions from (\d+) samples of (\d+)-symbol prefixes;"
                       r" initial sort-keys of (\d+) symbols\.")
RE_SIZES = re.compile(r"Partition sizes: ([\d ]*)")
RE_VEC = re.compile(r"Per-partition active-set sizes by context-size (\d+): ([\d ]*)")
RE_PILLS = re.compile(r"Active suffixes with poison-pill received for generation (\d+): (\d+)\.")
RE_ABORT = re.compile(r"In pivots-selection, (.*?)\. Increase `K`\. Aborting\.")

# The log prints amplification to 6 significant digits.
AMP_RTOL = 1e-5

NAN = float("nan")


def parse(path):
    """Returns what the `caps_sa_gpu` log at `path` reports; `None` if it is not one."""
    txt = path.read_text(errors="replace")
    n = RE_N.search(txt)
    if not n:
        return None

    act = RE_ACTIVE.findall(txt)
    stat = {k: int(m.group(1)) for k, rx in RE_STAT.items() for m in [rx.search(txt)] if m}
    bound, pivots, abort = RE_BOUND.search(txt), RE_PIVOTS.search(txt), RE_ABORT.search(txt)
    return {"name": path.stem, "path": path, "n": int(n.group(1)), "p": stat.get("Size"), "stat": stat,
            "ctx": [int(c) for c, _, _, _ in act],
            "sorted": [int(s) for _, s, _, _ in act],
            "total": [int(x) for _, _, x, _ in act],
            "active": [int(a) for _, _, _, a in act],
            "inbox": [(int(g), int(m), int(rc), float(amp)) for g, m, rc, amp in RE_INBOX.findall(txt)],
            "bound": tuple(map(int, bound.groups())) if bound else None,
            "pivots": tuple(map(int, pivots.groups())) if pivots else None,
            "sizes": [[int(x) for x in s.split()] for s in RE_SIZES.findall(txt)],
            "vec": {int(c): [int(x) for x in v.split()] for c, v in RE_VEC.findall(txt)},
            "pills": [(int(g), int(x)) for g, x in RE_PILLS.findall(txt)],
            "abort": abort.group(1) if abort else None}


def derive(r):
    """Adds the per-generation rows and the corpus aggregates to the parsed log `r`.

    Returns the checks that failed. The amplification bounds are, for t >= 1,
        (|P^(t)| - E_t) / |A^(t)| <= amp_{t+1} <= (2 |P^(t)| - |A^(t)|) / |A^(t)|,
    as M_{t+1} = payloads_t + forwards_t. Here, payloads_t = |P^(t)| - E_t with
    E_t = #{j in P^(t) : j < c_t} <= c_t; and 0 <= forwards_t <= |P^(t)| - |A^(t)|,
    since receiving a pill implies retirement. So the forwarded-pill stream is
    forwards_t = M_{t+1} - |P^(t)| + E_t, which the log brackets to within c_t.

    The per-partition active-set sizes give each generation's skew over the
    partitions, max / mean; the corpus' skew weighs it by |A^(t)|, i.e. by work.

    The message budget per suffix: each active suffix receives one live message
    per generation, A / n in all. A suffix sends its rank in each generation it
    takes part in, d + 1 of them, but receives only d; so the dead messages are
    one per suffix, less the last generation's cohort and the E_t edges, plus
    the forwarded pills. Generation 0's dead ones are the flood's.
    """
    c, n, a, inbox = r["name"], r["n"], r["active"], r["inbox"]
    ell = r["ctx"][0]
    G = len(a) - 1  # Doubling generations; the last one leaves no active suffix.
    fails = []

    def cohort(t):  # |P^(t)|
        return n if t == 0 else a[t - 1]

    if any(x != n for x in r["total"]):
        fails.append(f"{c}: suffix count differs from the sequence length {n}")
    if any(r["ctx"][t] != ell << t for t in range(len(a))):
        fails.append(f"{c}: contexts are not ell * 2^t")
    if any(a[t] > cohort(t) for t in range(len(a))):
        fails.append(f"{c}: active set grows")
    if a[-1] != 0:
        fails.append(f"{c}: run is incomplete, {a[-1]} suffixes still active")
    if len(inbox) != G or any(g != i + 1 for i, (g, *_) in enumerate(inbox)):
        fails.append(f"{c}: {len(inbox)} inbox lines for {G} doubling generations")
        return fails

    m = [None] + [x for _, x, _, _ in inbox]  # m[g] = M_g.
    for g, _, recips, amp in inbox:
        if recips != a[g - 1]:
            fails.append(f"{c}: generation {g} has {recips} recipients, but |A^({g - 1})| = {a[g - 1]}")
        if abs(amp - m[g] / a[g - 1]) > AMP_RTOL * amp:
            fails.append(f"{c}: generation {g} amplification {amp} is not M_g / |A^({g - 1})|")
    if m[1] != n - ell:
        fails.append(f"{c}: M_1 = {m[1]}, not n - ell = {n - ell}")

    vec = r["vec"]
    if vec:
        bad = [ctx for ctx, s, x in zip(r["ctx"], r["sorted"], a)
               if ctx not in vec or len(vec[ctx]) != r["p"] or sum(vec[ctx]) != x or vec[ctx].count(0) != s]
        if len(vec) != len(a) or bad:
            fails.append(f"{c}: per-partition active-set sizes disagree with the active sets, at contexts {bad}")

    rows_decay, rows_amp, rows_lb = [], [], []
    skew_work = 0.0
    for t in range(len(a)):
        v = vec.get(ell << t)
        skew = max(v) * len(v) / a[t] if v and a[t] > 0 else NAN
        skew_work += a[t] * skew if skew == skew else 0
        rows_decay.append({"corpus": c, "t": t, "ctx": ell << t, "active": a[t], "frac": a[t] / n,
                           "rho": a[t] / cohort(t), "S_ctx": NAN, "S_ctxK": NAN, "pred_active": NAN, "kshift": NAN,
                           "skew_max_mean": skew, "parts_live": sum(x > 0 for x in v) if v else NAN})

    pills = 0   # Forwarded pills over the run, as bracketed from below by the log.
    for t in range(G):  # Generation t emits the inbox M_{t+1}.
        g, ctx, coh = t + 1, ell << t, cohort(t)
        amp = m[g] / a[t]
        if t == 0:  # Every suffix sends a payload, and no pill is forwarded yet.
            lo = hi = amp
            pill_lo = pill_hi = 0
        else:
            lo = (coh - min(ctx, coh)) / a[t]
            hi = (2 * coh - a[t]) / a[t]
            pill_lo = max(0, m[g] - coh)
            pill_hi = min(m[g] - coh + min(ctx, coh), coh - a[t])
            pills += pill_lo
            if not lo <= amp <= hi:
                fails.append(f"{c}: amplification {amp:.6g} of generation {g} falls outside [{lo:.6g}, {hi:.6g}]")
            rows_lb.append({"corpus": c, "t": t, "cohort": coh, "lam0": NAN, "lam1": NAN, "lam2p": NAN,
                            "fwd_pred": NAN, "fwd_log_lo": pill_lo / coh, "fwd_log_hi": pill_hi / coh})
        rows_amp.append({"corpus": c, "t": t, "g": g, "rho": a[t] / coh, "amp": amp, "lo": lo, "hi": hi,
                         "cohort": coh, "pill_lo": pill_lo / coh, "pill_hi": pill_hi / coh})

    if pills > a[0]:    # Prop. pillvol: every forwarder retires, and each retires once.
        fails.append(f"{c}: forwarded pills {pills} exceed |A^(0)| = {a[0]}")

    A, M = sum(a), sum(m[1:])
    payloads_hi = sum(cohort(t) for t in range(G)) - ell  # Over-counts the logged payloads by the E_t edge senders.
    r["decay"], r["amplification"], r["lookback"] = rows_decay, rows_amp, rows_lb
    r["lookback_hist"], r["ccdf"], r["runs"] = [], [], []
    r["summary"] = {"corpus": c, "n": n, "p": r["p"] if r["p"] is not None else NAN, "ell": ell,
                    "K": r["pivots"][2] if r["pivots"] else NAN, "G": G, "surv0": a[0] / n, "Ed": A / n,
                    "depth_lo": ell << (G - 1), "depth_hi": (ell << G) + ell, "depth_max": NAN,
                    "amp_run": M / A, "waste": 1 - A / M, "pill_ovh": (M - payloads_hi) / payloads_hi,
                    "pill_ovh_exact": NAN, "flood_share": m[1] / M, "flood_waste": 1 - a[0] / m[1],
                    "skew_work": skew_work / A if vec else NAN, "max_B": NAN,
                    "live": A / n, "dead0": (m[1] - a[0]) / n, "dead_surv": (sum(m[2:]) - sum(a[1:]) - pills) / n,
                    "dead_pill": pills / n, "total": M / n}
    return fails


def quantile(xs, q):
    """Returns the `q`-quantile of the sorted `xs`, interpolated linearly."""
    pos = q * (len(xs) - 1)
    lo = int(pos)
    return xs[lo] + (pos - lo) * (xs[min(lo + 1, len(xs) - 1)] - xs[lo])


def partitioning(r, corpus, label):
    """Returns the size-bound row and the per-partition rows of the log `r`'s
    partitioning lines, and the checks that failed; no rows for a log older than
    those lines.

    The greedy fill packs each full partition with h = floor(B (s + 1) / n) of
    the s samples' gaps, so a full partition's size has a relative deviation of
    about 1 / sqrt(h), and the largest of the p - 1 full ones about
    sqrt(2 ln(p - 1) / h). `env_B` is that envelope over the full partitions'
    mean. A pivot-selection abort, as for too narrow a `K`, is an infeasible row.
    """
    name, n = r["name"], r["n"]
    k = re.fullmatch(r"k(\d+)", label)
    if r["abort"] and not r["pivots"]:  # Keeps the size bound, printed before the abort, that the run failed.
        row = {"corpus": corpus, "label": label, "feasible": 0, "K": int(k.group(1)) if k else NAN}
        if r["bound"]:
            row.update(dict(zip(("B", "p_min", "floor"), r["bound"])), n=n)
        return row, [], []
    if r["bound"] is None and r["pivots"] is None and not r["sizes"]:
        return None, [], []

    missing = [x for x in ("bound", "pivots") if r[x] is None] + ([] if r["sizes"] else ["partition-sizes"])
    if missing or len(r["sizes"]) != 1:
        return None, [], [f"{name}: " + (f"lacks the {', '.join(missing)} line" if missing
                                         else f"has {len(r['sizes'])} partition-sizes lines")]

    (B, p_min, floor), (p, s, K, ell), sz = r["bound"], r["pivots"], r["sizes"][0]
    fails = []
    if B != -(-n // p_min) or floor > p_min:
        fails.append(f"{name}: size bound {B} is not ceil(n / {p_min}), or the floor {floor} exceeds {p_min}")
    st = r["stat"]
    if (len(sz) != p or sum(sz) != n or min(sz) == 0
            or (st.get("Size"), st.get("Minimum"), st.get("Maximum")) != (p, min(sz), max(sz))):
        fails.append(f"{name}: its {len(sz)} partition sizes disagree with p = {p}, n or the size stats")

    full = sorted(sz[:-1] if p > 1 else sz)  # All but the leftover last partition.
    h = B * (s + 1) // n
    mean_full = sum(full) / len(full)
    row = {"corpus": corpus, "label": label, "feasible": 1, "K": K, "ell": ell, "n": n, "p": p, "p_min": p_min,
           "floor": floor, "B": B, "s": s, "h": h, "p_over_pmin": p / p_min, "max_sz": max(sz), "max_B": max(sz) / B,
           "mean_full_B": mean_full / B, "min_full_B": full[0] / B, "q05_B": quantile(full, 0.05) / B,
           "q50_B": quantile(full, 0.5) / B, "q95_B": quantile(full, 0.95) / B, "fill": n / (p * B),
           "env_B": mean_full / B * (1 + math.sqrt(2 * math.log(len(full)) / h)) if h > 0 else NAN}
    sizes = [{"corpus": corpus, "label": label, "i": i, "size_B": x / B} for i, x in enumerate(sz)]
    return row, sizes, fails


def thinned(hi):
    """Returns the context-sizes up to `hi` to sample a CCDF at: each one up to 64, then 32 per octave."""
    cs = list(range(1, min(hi, 64) + 1))
    x = 64.0
    while True:
        x *= 2 ** (1 / 32)
        v = round(x)
        if v > hi:
            return cs
        if v > cs[-1]:
            cs.append(v)


def attach(r, tool):
    """Adds `sa_structure`'s output `tool` for the run to the rows of its log `r`.

    Returns the checks that failed: the tool must reproduce every |A^(t)| and
    M_g of the log, and the pill receipts of a TESTS log, exactly; and its
    repeat-depths must bound the active sets, S(c_t + K) <= |A^(t)| <= S(c_t).
    The tertiary shift K'_t is the (interpolated) k with S(c_t + k) = |A^(t)|.
    """
    c, n, a, s = r["name"], r["n"], r["active"], r["summary"]
    ell, G = r["ctx"][0], len(a) - 1
    K = r["pivots"][2] if r["pivots"] else tool["K"]
    if (tool["n"], tool["ell"], tool["K"], tool["p"]) != (n, ell, K, r["p"]):
        return [f"{c}: the tool's n, ell, K, p = {tool['n']}, {tool['ell']}, {tool['K']}, {tool['p']} are not the log's"]

    fails = []
    retire, pay, fwd, lb, runs = tool["retire_hist"], tool["payloads"], tool["forwards"], tool["lookback"], tool["runs"]
    D = len(retire) - 1
    pred = [sum(retire[t + 1:]) for t in range(D + 1)]  # |A^(t)| = #{j : d(j) > t}.
    if pred != a:
        bad = [t for t in range(max(D, G) + 1) if t > min(D, G) or pred[t] != a[t]]
        fails.append(f"{c}: the tool's active sets differ from the log's at generations {bad}")
    m = [None] + [x for _, x, _, _ in r["inbox"]]
    bad = [g for g in range(1, G + 1) if g > D or pay[g - 1] + fwd[g - 1] != m[g]]
    if bad:
        fails.append(f"{c}: the tool's inbox-sizes differ from the log's at generations {bad}")
    bad = [g for g, x in r["pills"] if g > D or sum(lb[g][1:]) != x]
    if bad:
        fails.append(f"{c}: the tool's pill receipts differ from the TESTS log's at generations {bad}")

    xs = [x for x, _ in tool["depth_hist"]]
    tail = [0] * (len(xs) + 1)  # tail[i] = the count of the depths at least xs[i].
    for i in range(len(xs) - 1, -1, -1):
        tail[i] = tail[i + 1] + tool["depth_hist"][i][1]

    def S(ctx):     # #{j : depth(j) >= ctx}; exact up to the tool's depth cap.
        return tail[bisect.bisect_left(xs, ctx)] if ctx <= tool["depth_cap"] else NAN

    def kshift(ctx, x):
        prev = S(ctx)
        if not x or prev != prev:
            return NAN
        for k in range(K + 1):
            cur = S(ctx + k)
            if cur <= x:
                return 0.0 if k == 0 else k - 1 + (prev - x) / (prev - cur)
            prev = cur
        return NAN

    bounds = [(t, S((ell << t) + K), S(ell << t)) for t in range(G + 1)]
    bad = [t for t, lo, hi in bounds if lo == lo and hi == hi and not lo <= a[t] <= hi]    # Unless past the cap.
    if bad:
        fails.append(f"{c}: the active sets fall outside their repeat-depth bounds at generations {bad}")
    if not s["depth_lo"] <= tool["depth_max"] < (ell << G) + K:
        fails.append(f"{c}: maximum repeat-depth {tool['depth_max']} lies outside [{s['depth_lo']}, {(ell << G) + K})")
    if sum(fwd[1:]) > a[0]:
        fails.append(f"{c}: the tool's forwarded pills {sum(fwd[1:])} exceed |A^(0)| = {a[0]}")
    bad = [t for t in range(1, D) if sum(lb[t + 1][2:]) > fwd[t]]
    if bad:
        fails.append(f"{c}: more pills consumed than forwarded, after generations {bad}")

    for row in r["decay"]:
        t, ctx = row["t"], row["ctx"]
        row.update(S_ctx=S(ctx) / n, S_ctxK=S(ctx + K) / n, pred_active=pred[t] if t <= D else NAN,
                   kshift=kshift(ctx, a[t]))
    log_pills = {row["t"]: row for row in r["lookback"]}
    r["lookback"] = []
    for t in range(1, D + 1):
        coh = sum(lb[t])
        row = log_pills.get(t, {})
        r["lookback"].append({"corpus": c, "t": t, "cohort": coh, "lam0": lb[t][0] / coh,
                              "lam1": lb[t][1] / coh, "lam2p": sum(lb[t][2:]) / coh,
                              "fwd_pred": fwd[t] / coh, "fwd_log_lo": row.get("fwd_log_lo", NAN),
                              "fwd_log_hi": row.get("fwd_log_hi", NAN)})
        r["lookback_hist"] += [{"corpus": c, "t": t, "lam": lam, "count": x, "frac": x / coh}
                               for lam, x in enumerate(lb[t])]
    r["ccdf"] = [{"corpus": c, "c": x, "frac": S(x) / n} for x in thinned(min(tool["depth_max"], tool["depth_cap"]))]
    r["runs"] = [{"corpus": c, "t": t, "ctx": ell << t, "runs": runs[t], "mean_len": a[t] / runs[t] if runs[t] else NAN}
                 for t in range(min(D, G))]
    s["depth_max"] = tool["depth_max"]
    r["kgram"] = tool.get("kgram_max")
    s["pill_ovh_exact"] = sum(fwd[1:G]) / sum(pay[:G])     # Over the logged inboxes, M_2 to M_G.
    s["dead_pill"] = sum(fwd[1:G]) / n                      # Exact, in place of the log's bracket.
    s["dead_surv"] = (sum(m[2:]) - sum(a[1:]) - sum(fwd[1:G])) / n
    return fails


def ablation(corpus, reps, tool):
    """Returns the liveness filter's per-generation rows for the corpus' filtered
    runs `reps`, its per-suffix summary, and the checks that failed; `tool` is
    `sa_structure`'s output for their partitioning.

    The filter skips the messages whose recipients are already marked retired.
    So it changes no active set and skips no live message: generation g's inbox
    lies between |A^(g-1)| and the unfiltered M_g, which the tool predicts
    exactly. Whether a recipient that retires in its sender's generation is
    marked in time depends on the pipeline's timing, so the replicates differ
    slightly; the rows give their mean and range. The summary is per suffix, so
    it holds for an unfiltered run of a slightly different partitioning too.
    """
    n, retire, pay, fwd = tool["n"], tool["retire_hist"], tool["payloads"], tool["forwards"]
    D = len(retire) - 1
    active = [sum(retire[t + 1:]) for t in range(D + 1)]
    unf = [pay[t] + fwd[t] for t in range(D)]  # unf[g - 1] = the unfiltered M_g.
    fails, inboxes = [], []
    for r in reps:
        if (r["n"], r["p"], r["active"]) != (n, tool["p"], active):
            fails.append(f"{r['name']}: its n, p or active sets differ from the tool's for its partitioning")
            continue
        if [g for g, *_ in r["inbox"]] != list(range(1, D + 1)):
            fails.append(f"{r['name']}: {len(r['inbox'])} inbox lines for {D} doubling generations")
            continue
        m = [x for _, x, _, _ in r["inbox"]]
        bad = [g for g in range(1, D + 1) if not active[g - 1] <= m[g - 1] <= unf[g - 1]]
        if bad:
            fails.append(f"{r['name']}: inbox-sizes outside [|A^(g-1)|, unfiltered M_g] at generations {bad}")
        inboxes.append(m)
    if not inboxes:
        return [], None, fails

    rows = []
    for g in range(1, D + 1):
        ms = [m[g - 1] for m in inboxes]
        live, u, mean = active[g - 1], unf[g - 1], sum(ms) / len(ms)
        rows.append({"corpus": corpus, "g": g, "live": live, "unfiltered": u, "filtered": mean,
                     "filtered_lo": min(ms), "filtered_hi": max(ms),
                     "skipped_dead": (u - mean) / (u - live) if u > live else NAN})
    U, F, A = sum(unf), sum(row["filtered"] for row in rows), sum(active)
    summary = {"reps": len(inboxes), "skipped0": (unf[0] - rows[0]["filtered"]) / n,
               "skipped_later": (U - unf[0] - F + rows[0]["filtered"]) / n,
               "msg_cut": 1 - F / U, "dead_cut": (U - F) / (U - A)}
    return rows, summary, fails


def heaviest(kgram, skew, summary):
    """Adds each corpus' heaviest K-gram frequency f_max to its partitionings in
    `skew` and to its summary in `summary`, as f_max / B at their K; returns the
    kgram.tsv rows, and the checks that failed. `kgram` maps a corpus to its
    `sa_structure` output's `kgram_max`, f_max for K = 1, 2, ....

    The suffixes that share a K-gram share a partition, so no partitioning by
    K-prefixes has a partition smaller than f_max. A pivot-selection abort is the
    sampled counterpart of f_max > B.
    """
    fails, run_B = [], {}
    run_K = {s["corpus"]: s["K"] for s in summary}
    for row in skew:
        c = row["corpus"]
        K = row["K"] if row["K"] == row["K"] else run_K.get(c, NAN)
        if c not in kgram or "B" not in row or not 1 <= K <= len(kgram[c]):
            continue
        f = kgram[c][int(K) - 1]
        row["fmax_B"] = f / row["B"]
        if row["label"] == "full":
            run_B[c] = row["B"]
        if row["feasible"] and row["max_sz"] < f:
            fails.append(f"{c} {row['label']}: its largest partition, {row['max_sz']}, is below f_max = {f} at K = {K}")
    for s in summary:
        c = s["corpus"]
        if c in kgram and 1 <= s["K"] <= len(kgram[c]):
            s["fmax"] = kgram[c][int(s["K"]) - 1]
            s["fmax_B"] = s["fmax"] / run_B[c] if c in run_B else NAN
    n = {s["corpus"]: s["n"] for s in summary}
    rows = [{"corpus": c, "K": k, "fmax": f, "frac": f / n[c] if c in n else NAN,
             "fmax_B": f / run_B[c] if c in run_B else NAN}
            for c in sorted(kgram) for k, f in enumerate(kgram[c], 1)]
    return rows, fails


def fmt(x):
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def write_tsv(path, cols, rows):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(k, NAN)) for k in cols) + "\n")


def pow2(x):
    """Returns `x` typeset as a power of two if it is one, else as is."""
    return f"2^{{{x.bit_length() - 1}}}" if x & (x - 1) == 0 else str(x)


def write_tex(path, rows):
    """Typesets the per-corpus summary as a booktabs tabular.

    Waste is the share of messages dropped at the join, 1 - A/M; Pills is the
    forwarded-pill stream over the logged payloads, exact with the tool and
    bracketed from below otherwise; Flood is generation 1's share of M; and Skew
    is the work-weighted max / mean of the per-partition active-set sizes.

    delta_max is exact with the tool, and only bracketed by a log otherwise: the
    last generation's cohort A^(G-1) is non-empty, so some lcp is at least
    c_{G-1}; and no pair is linked at G, so every lcp is below c_G + K, with
    K <= ell. The table shows the bracket as c_{G-1}--c_G.
    """
    def pct(x):
        return f"{100 * x:.1f}\\%"

    lines = [r"\begin{tabular}{lrrrrrcrrrrr}",
             r"\toprule",
             r"Corpus & $n$ ($10^9$) & $p$ & $G$ & $|\mathcal{A}^{(0)}|/n$ & $\mathrm{E}[d]$ & $\delta_{\max}$"
             r" & $M/A$ & Waste & Pills & Flood & Skew \\",
             r"\midrule"]
    for s in rows:
        exact = s["depth_max"] == s["depth_max"]
        depth = (f"{s['depth_max']:,}".replace(",", r"\,") if exact
                 else f"${pow2(s['depth_lo'])}$--${pow2(2 * s['depth_lo'])}$")
        pills = s["pill_ovh_exact"] if exact else s["pill_ovh"]
        skew = f"{s['skew_work']:.2f}" if s["skew_work"] == s["skew_work"] else "--"
        p = str(s["p"]) if s["p"] == s["p"] else "--"
        lines.append(" & ".join([s["corpus"].replace("_", r"\_"), f"{s['n'] / 1e9:.2f}", p, str(s["G"]),
                                 pct(s["surv0"]), f"{s['Ed']:.3f}", depth, f"{s['amp_run']:.2f}", pct(s["waste"]),
                                 pct(pills), pct(s["flood_share"]), skew]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


DECAY = ["corpus", "t", "ctx", "active", "frac", "rho", "S_ctx", "S_ctxK", "pred_active", "kshift", "skew_max_mean",
         "parts_live"]
AMPLIFICATION = ["corpus", "t", "g", "rho", "amp", "lo", "hi", "cohort", "pill_lo", "pill_hi"]
LOOKBACK = ["corpus", "t", "cohort", "lam0", "lam1", "lam2p", "fwd_pred", "fwd_log_lo", "fwd_log_hi"]
SKEW = ["corpus", "label", "feasible", "K", "ell", "n", "p", "p_min", "floor", "B", "s", "h", "p_over_pmin", "max_B",
        "mean_full_B", "min_full_B", "q05_B", "q50_B", "q95_B", "fill", "env_B", "fmax_B"]
DATASETS = ["corpus", "n", "p", "ell", "G", "surv0", "Ed", "depth_lo", "depth_hi", "amp_run", "waste", "pill_ovh",
            "flood_share", "flood_waste", "K", "depth_max", "pill_ovh_exact", "skew_work", "max_B",
            "msg_cut", "dead_cut", "fmax", "fmax_B"]
BUDGET = ["corpus", "Ed", "live", "dead0", "dead_surv", "dead_pill", "total", "amp_run",    # Messages per suffix.
          "skipped0", "skipped_later"]
FILTER = ["corpus", "g", "live", "unfiltered", "filtered", "filtered_lo", "filtered_hi", "skipped_dead"]


def main(argv):
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2

    out = pathlib.Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    args, flag = {"": [], "--sweep": [], "--filtered": []}, ""
    for a in argv[1:]:
        if a in args:
            flag = a
        else:
            args[flag].append(a)

    runs, skew, sizes, fails = [], [], [], []
    for f in args[""]:
        path = pathlib.Path(f)
        r = parse(path)
        if r is None or not r["active"]:
            fails.append(f"{f}: no STATS generation lines; is it a STATS build's log?")
            continue
        fails += derive(r)
        if "summary" not in r:
            continue
        row, rows, bad = partitioning(r, r["name"], "full")
        fails += bad
        if row:
            skew.append(row)
            sizes += rows
            r["summary"]["max_B"] = row["max_B"]
        tool = path.with_suffix(".structure.json")
        if tool.exists():
            fails += attach(r, json.loads(tool.read_text()))
        runs.append(r)

    for f in args["--sweep"]:
        path = pathlib.Path(f)
        corpus, _, label = path.stem.rpartition("_")
        r = parse(path)
        row, rows, bad = partitioning(r, corpus or label, label if corpus else "") if r else (None, [], [])
        fails += bad
        if row is None and not bad:
            fails.append(f"{f}: no partitioning lines; is it a STATS build's partition-only log?")
        if row:
            skew.append(row)
            sizes += rows

    filtered, filt_rows, filt_sum = {}, [], {}
    kgram = {r["name"]: [r["kgram"]] for r in runs if r.get("kgram")}
    for f in args["--filtered"]:
        path = pathlib.Path(f)
        r = parse(path)
        if r is None or not r["active"]:
            fails.append(f"{f}: no STATS generation lines; is it a STATS build's log?")
            continue
        filtered.setdefault(path.stem.rpartition("_")[0] or path.stem, []).append(r)
    for corpus, reps in filtered.items():
        path = reps[0]["path"].parent / f"{corpus}.structure.json"
        if not path.exists():
            fails.append(f"{corpus}: no {path.name} next to its filtered runs")
            continue
        tool = json.loads(path.read_text())
        rows, summ, bad = ablation(corpus, reps, tool)
        fails += bad
        if tool.get("kgram_max"):
            kgram.setdefault(corpus, []).append(tool["kgram_max"])
        filt_rows += rows
        if summ:
            filt_sum[corpus] = summ
    for r in runs:
        r["summary"].update(filt_sum.get(r["name"], {}))
    fails += [f"{c}: its tool runs disagree on the heaviest K-grams" for c, ks in kgram.items()
              if ks.count(ks[0]) != len(ks)]
    kgram_rows, bad = heaviest({c: ks[0] for c, ks in kgram.items()}, skew, [r["summary"] for r in runs])
    fails += bad

    runs.sort(key=lambda r: r["n"])
    skew.sort(key=lambda row: (row["corpus"], row["label"][:1], row.get("K", 0), row.get("p_min", 0)))

    write_tsv(out / "decay.tsv", DECAY, [row for r in runs for row in r["decay"]])
    write_tsv(out / "depth_ccdf.tsv", ["corpus", "c", "frac"], [row for r in runs for row in r["ccdf"]])
    write_tsv(out / "amplification.tsv", AMPLIFICATION, [row for r in runs for row in r["amplification"]])
    write_tsv(out / "lookback.tsv", LOOKBACK, [row for r in runs for row in r["lookback"]])
    write_tsv(out / "lookback_hist.tsv", ["corpus", "t", "lam", "count", "frac"],
              [row for r in runs for row in r["lookback_hist"]])
    write_tsv(out / "runs.tsv", ["corpus", "t", "ctx", "runs", "mean_len"], [row for r in runs for row in r["runs"]])
    write_tsv(out / "skew.tsv", SKEW, skew)
    write_tsv(out / "skew_p.tsv", SKEW, [row for row in skew if row["label"].startswith("p")])
    write_tsv(out / "skew_k.tsv", SKEW, [row for row in skew if row["label"].startswith("k")])
    write_tsv(out / "partition_sizes.tsv", ["corpus", "label", "i", "size_B"], sizes)
    summary = [r["summary"] for r in runs]
    write_tsv(out / "datasets.tsv", DATASETS, summary)
    write_tsv(out / "budget.tsv", BUDGET, sorted(summary, key=lambda s: s["Ed"]))
    write_tsv(out / "filter.tsv", FILTER, sorted(filt_rows, key=lambda row: (row["corpus"], row["g"])))
    write_tsv(out / "kgram.tsv", ["corpus", "K", "fmax", "frac", "fmax_B"], kgram_rows)
    write_tex(out / "datasets.tex", summary)

    report = [f"{r['name']}: n = {r['n']}, ell = {r['ctx'][0]}, {len(r['active']) - 1} doubling generations"
              + ("; with the tool's statistics" if r["ccdf"] else "") for r in runs]
    report += [f"sweep {row['corpus']} {row['label']}: " + (f"p = {row['p']}, max/B = {row['max_B']:.4f}"
                                                              if row["feasible"] else "infeasible")
               for row in skew if row["label"] != "full"]
    report += [f"filtered {c}: {s['reps']} replicate(s); {100 * s['msg_cut']:.1f}% of the messages skipped,"
               f" {100 * s['dead_cut']:.1f}% of the dead ones" for c, s in sorted(filt_sum.items())]
    report += [f"heaviest {s['K']}-gram of {s['corpus']}: {s['fmax']}, {100 * s['fmax_B']:.2f}% of B"
               for s in sorted((r["summary"] for r in runs), key=lambda s: s["n"]) if "fmax" in s]
    report += [f"FAIL {x}" for x in fails]
    report.append(f"{len(fails)} check(s) failed." if fails else "All checks pass.")
    (out / "checks.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
