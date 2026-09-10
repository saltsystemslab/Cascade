#!/usr/bin/env python3
"""Summarize a baseline results directory.

    summarize.py <results-dir> [--tsv out.tsv]

Reads the one-JSON-per-run records the drivers write, groups replicates, and
prints median t_core with min-max. Deliberately reports no mean and no sd: at
n=3 they are not meaningful, and a column that does not exist cannot drift into
the paper.
"""
import json
import pathlib
import statistics
import sys

GiB = 1 << 30


def load(d):
    runs, bad = [], []
    for f in sorted(pathlib.Path(d).glob("*.json")):
        try:
            runs.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            bad.append(f"{f.name}: {e}")
    return runs, bad


def check(runs):
    """Provenance guards. Each catches a whole class of invalid table."""
    warn = []
    commits = {r.get("commit", "unknown") for r in runs}
    if "unknown" in commits:
        warn.append("some records have commit=unknown: CMake did not pass BENCH_COMMIT")
    if any(c.endswith("-dirty") for c in commits):
        warn.append("some records come from a dirty working tree")
    if len(commits) > 1:
        warn.append(f"records span {len(commits)} commits {sorted(commits)}: not comparable")
    hosts = {r.get("host") for r in runs}
    if len(hosts) > 1:
        warn.append(f"runs spanned hosts {sorted(hosts)}: timings are not comparable")
    # `off` means verification was never requested (a recorded policy choice,
    # e.g. github, where the +8n ISA would not fit). `fail` and `skip` mean it
    # was wanted and did not succeed -- those are the ones worth shouting about.
    failed = [r["run_id"] for r in runs if r.get("verify") == "fail"]
    if failed:
        warn.append(f"VERIFICATION FAILED for {failed}")
    skipped = [r["run_id"] for r in runs if r.get("verify") == "skip"]
    if skipped:
        warn.append(f"verification requested but not completed for {skipped}")
    return warn


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    d = argv[0]
    tsv = argv[argv.index("--tsv") + 1] if "--tsv" in argv else None

    runs, bad = load(d)
    if not runs:
        print(f"no records in {d}")
        return 1

    # Group replicates. `threads` is part of the key so the thread-choice pair
    # does not get folded into the headline rows.
    groups = {}
    for r in runs:
        key = (r["method"], r["corpus"], r["alphabet"], r["n_tokens"], r["threads"])
        groups.setdefault(key, []).append(r)

    hdr = ("method", "corpus", "alphabet", "n_tokens", "thr", "reps", "status",
           "verify", "t_core_med", "min", "max", "Mtok/s", "peak_rss", "B/tok", "peak_dev")
    rows = []
    for (method, corpus, alphabet, ntok, thr), rs in sorted(groups.items(), key=lambda kv: kv[0][3]):
        ok = [r for r in rs if r["status"] == "ok"]
        # A group is only "ok" if every replicate was; otherwise show the failure.
        status = "ok" if len(ok) == len(rs) else next(r["status"] for r in rs if r["status"] != "ok")
        cores = [r["t_core"] for r in ok if r["t_core"] >= 0]
        rss = max(r["peak_rss"] for r in rs)
        dev = max(r["peak_device"] for r in rs)
        med = statistics.median(cores) if cores else None
        rows.append([
            method, corpus, alphabet, f"{ntok:.3e}", thr, len(rs), status,
            "/".join(sorted({r["verify"] for r in rs})),
            f"{med:.2f}" if med else "-",
            f"{min(cores):.2f}" if cores else "-",
            f"{max(cores):.2f}" if cores else "-",
            f"{ntok / med / 1e6:.2f}" if med else "-",
            f"{rss / GiB:.2f}G" if rss else "-",
            f"{rss / ntok:.2f}" if rss and ntok else "-",
            f"{dev / GiB:.2f}G" if dev > 0 else "-",
        ])

    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(hdr)]
    line = lambda c: "  ".join(str(x).ljust(w[i]) for i, x in enumerate(c))
    print(line(hdr))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))

    ok_runs = [r for r in runs if r["status"] == "ok"]
    verified = [r for r in ok_runs if r["verify"] == "pass"]
    print(f"\n{len(runs)} records in {len(groups)} groups; "
          f"{len(verified)}/{len(ok_runs)} ok runs verified")
    for b in bad:
        print(f"UNPARSEABLE {b}")
    for m in check(runs):
        print(f"WARN: {m}")

    # A failed run is the E2.1 result for its row, so surface the reason.
    for r in runs:
        if r["status"] != "ok":
            print(f"  {r['status']:10} {r['run_id']}: {r['detail']}")

    if tsv:
        with open(tsv, "w") as f:
            f.write("\t".join(hdr) + "\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"\nwrote {tsv}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
