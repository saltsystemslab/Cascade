#!/usr/bin/env python3
"""Tabulate the structural statistics of GPU-SACA runs, for the paper's figures.

    structure.py <out-dir> <corpus>.log...

Each log is a `caps_sa_gpu` STATS run of one corpus, and is named after it.
Writes pgfplots-ready TSVs into <out-dir>:
    decay.tsv           the active set after each generation
    amplification.tsv   each generation's amplification, against its decay bounds
    datasets.tsv        one row per corpus; also typeset as datasets.tex
    checks.txt          the consistency checks, also printed
Exits 1 if any check fails.

Notation follows the theory writeup. A^(t) is the active set after generation t,
at context c_t = ell * 2^t. Generation t's cohort is P^(t) = A^(t-1), with
A^(-1) = [0, n). M_g is the inbox that generation g reads, i.e. what generation
g - 1 emitted. A = sum_t |P^(t)| is the total cohort mass, and M = sum_g M_g.
"""
import pathlib
import re
import sys

RE_N = re.compile(r"Sequence length: (\d+)")
RE_P = re.compile(r"\tSize:\s+(\d+)")
# Unanchored, as a warning without a trailing newline can glue itself to the
# front of the next line.
RE_ACTIVE = re.compile(r"fully-sorted by context-size (\d+): \d+\.(?: Time taken in iteration: \S+s\.)?"
                       r" Out of (\d+) suffixes, (\d+) \(")
RE_INBOX = re.compile(r"Total inbox-size for generation (\d+): (\d+); number of recipients: (\d+)\."
                      r" Amplification: (\S+)x\.")

# The log prints amplification to 6 significant digits.
AMP_RTOL = 1e-5


def parse(path):
    """Returns the per-generation series of the STATS log at `path`; `None` if it carries none."""
    txt = path.read_text()
    n = RE_N.search(txt)
    act = RE_ACTIVE.findall(txt)
    if not n or not act:
        return None

    p = RE_P.search(txt)
    return {"corpus": path.stem,
            "n": int(n.group(1)),
            "p": int(p.group(1)) if p else None,
            "ctx": [int(c) for c, _, _ in act],
            "total": [int(x) for _, x, _ in act],
            "active": [int(a) for _, _, a in act],
            "inbox": [(int(g), int(m), int(rc), float(amp)) for g, m, rc, amp in RE_INBOX.findall(txt)]}


def derive(r):
    """Adds the per-generation rows and the corpus aggregates to the parsed log `r`.

    Returns the checks that failed. The amplification bounds are, for t >= 1,
        (|P^(t)| - E_t) / |A^(t)| <= amp_{t+1} <= (2 |P^(t)| - |A^(t)|) / |A^(t)|,
    as M_{t+1} = payloads_t + forwards_t. Here, payloads_t = |P^(t)| - E_t with
    E_t = #{j in P^(t) : j < c_t} <= c_t; and 0 <= forwards_t <= |P^(t)| - |A^(t)|,
    since receiving a pill implies retirement. So the forwarded-pill stream is
    forwards_t = M_{t+1} - |P^(t)| + E_t, which the log brackets to within c_t.
    """
    c, n, a, inbox = r["corpus"], r["n"], r["active"], r["inbox"]
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

    rows_decay, rows_amp = [], []
    for t in range(len(a)):
        rows_decay.append({"corpus": c, "t": t, "ctx": ell << t, "active": a[t],
                           "frac": a[t] / n, "rho": a[t] / cohort(t)})

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
        rows_amp.append({"corpus": c, "t": t, "g": g, "rho": a[t] / coh, "amp": amp, "lo": lo, "hi": hi,
                         "cohort": coh, "pill_lo": pill_lo / coh, "pill_hi": pill_hi / coh})

    if pills > a[0]:    # Prop. pillvol: every forwarder retires, and each retires once.
        fails.append(f"{c}: forwarded pills {pills} exceed |A^(0)| = {a[0]}")

    A, M = sum(a), sum(m[1:])
    payloads_hi = sum(cohort(t) for t in range(G)) - ell  # Over-counts the logged payloads by the E_t edge senders.
    r["decay"], r["amplification"] = rows_decay, rows_amp
    r["summary"] = {"corpus": c, "n": n, "p": r["p"] if r["p"] is not None else "nan", "ell": ell, "G": G,
                    "surv0": a[0] / n, "Ed": A / n, "depth_lo": ell << (G - 1), "depth_hi": (ell << G) + ell,
                    "amp_run": M / A, "waste": 1 - A / M, "pill_ovh": (M - payloads_hi) / payloads_hi,
                    "flood_share": m[1] / M, "flood_waste": 1 - a[0] / m[1]}
    return fails


def fmt(x):
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def write_tsv(path, cols, rows):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row[k]) for k in cols) + "\n")


def pow2(x):
    """Returns `x` typeset as a power of two if it is one, else as is."""
    return f"2^{{{x.bit_length() - 1}}}" if x & (x - 1) == 0 else str(x)


def write_tex(path, rows):
    """Typesets the per-corpus summary as a booktabs tabular.

    Waste is the share of messages dropped at the join, 1 - A/M; Pills is the
    forwarded-pill stream over the logged payloads, as bracketed from below; and
    Flood is generation 1's share of M.

    delta_max is only bracketed by a log: the last generation's cohort A^(G-1)
    is non-empty, so some lcp is at least c_{G-1}; and no pair is linked at G,
    so every lcp is below c_G + K, with K <= ell. The table shows c_{G-1}--c_G.
    """
    lines = [r"\begin{tabular}{lrrrrrcrrrr}",
             r"\toprule",
             r"Corpus & $n$ ($10^9$) & $p$ & $G$ & $|\mathcal{A}^{(0)}|/n$ & $\mathrm{E}[d]$ & $\delta_{\max}$"
             r" & $M/A$ & Waste & Pills & Flood \\",
             r"\midrule"]
    for s in rows:
        p = s["p"] if s["p"] != "nan" else "--"
        depth = f"${pow2(s['depth_lo'])}$--${pow2(2 * s['depth_lo'])}$"
        lines.append(" & ".join([s["corpus"].replace("_", r"\_"), f"{s['n'] / 1e9:.2f}", str(p), str(s["G"]),
                                 f"{100 * s['surv0']:.1f}\\%", f"{s['Ed']:.3f}", depth, f"{s['amp_run']:.2f}",
                                 f"{100 * s['waste']:.1f}\\%", f"{100 * s['pill_ovh']:.1f}\\%",
                                 f"{100 * s['flood_share']:.1f}\\%"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main(argv):
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2

    out = pathlib.Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)

    runs, fails = [], []
    for f in argv[1:]:
        r = parse(pathlib.Path(f))
        if r is None:
            fails.append(f"{f}: no STATS generation lines; is it a STATS build's log?")
            continue
        fails += derive(r)
        if "summary" in r:
            runs.append(r)
    runs.sort(key=lambda r: r["n"])

    write_tsv(out / "decay.tsv", ["corpus", "t", "ctx", "active", "frac", "rho"],
              [row for r in runs for row in r["decay"]])
    write_tsv(out / "amplification.tsv", ["corpus", "t", "g", "rho", "amp", "lo", "hi", "cohort", "pill_lo", "pill_hi"],
              [row for r in runs for row in r["amplification"]])
    summary = [r["summary"] for r in runs]
    write_tsv(out / "datasets.tsv", ["corpus", "n", "p", "ell", "G", "surv0", "Ed", "depth_lo", "depth_hi", "amp_run",
                                     "waste", "pill_ovh", "flood_share", "flood_waste"], summary)
    write_tex(out / "datasets.tex", summary)

    report = [f"{r['corpus']}: n = {r['n']}, ell = {r['ctx'][0]}, {len(r['active']) - 1} doubling generations"
              for r in runs] + [f"FAIL {x}" for x in fails]
    report.append(f"{len(fails)} check(s) failed." if fails else "All checks pass.")
    (out / "checks.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
