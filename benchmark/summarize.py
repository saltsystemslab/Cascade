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
SCHEMA = 4

# Accessed unconditionally below, so a record missing any of them would take the
# whole summary down rather than reporting itself as the one bad record.
REQUIRED = ("run_id", "method", "corpus", "alphabet", "n_tokens", "threads",
            "status", "t_core", "peak_rss", "peak_device", "verify", "detail")


def load(d):
    runs, bad = [], []
    for f in sorted(pathlib.Path(d).glob("*.json")):
        # The manifest lives in the same directory and is a JSON list.
        if f.name == "manifest.json":
            continue
        try:
            r = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            bad.append(f"{f.name}: {e}")
            continue
        if not isinstance(r, dict):
            bad.append(f"{f.name}: not a JSON object")
            continue
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            bad.append(f"{f.name}: missing {','.join(missing)}")
            continue
        runs.append(r)
    return runs, bad


def check(runs):
    """Provenance guards. Each catches a whole class of invalid table."""
    warn = []
    schemas = {r.get("schema_version") for r in runs}
    if schemas - {SCHEMA}:
        warn.append(f"records span schema versions {sorted(map(str, schemas))}, "
                    f"expected {SCHEMA}: fields may not mean the same thing")
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


def inconsistencies(rs):
    """Fields that must be identical across replicates. A difference here is a
    harness fault by construction, not noise—replicates of one configuration
    cannot legitimately run a different binary or a different environment."""
    out = []
    for k in ("exe_hash", "env_tuning", "threads_avail", "cpus_allowed"):
        vals = {r.get(k) for r in rs}
        if len(vals) > 1:
            out.append(f"{k}={sorted(map(str, vals))}")
    return out


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
        key = (r["method"], r["corpus"], r.get("tokenizer", ""), r["alphabet"],
               r["n_tokens"], r["threads"])
        groups.setdefault(key, []).append(r)

    hdr = ("method", "corpus", "tok", "alphabet", "n_tokens", "thr", "thr_av", "reps",
           "status", "verify", "t_core_med", "min", "max", "Mtok/s", "cpu_s", "peak_rss",
           "B/tok", "peak_dev", "!")
    rows, flagged = [], []
    for (method, corpus, tok, alphabet, ntok, thr), rs in sorted(groups.items(),
                                                                key=lambda kv: kv[0][4]):
        ok = [r for r in rs if r["status"] == "ok"]
        # A group is only "ok" if every replicate was; otherwise show the failure.
        status = "ok" if len(ok) == len(rs) else next(r["status"] for r in rs if r["status"] != "ok")
        cores = [r["t_core"] for r in ok if r["t_core"] >= 0]
        rss = max(r["peak_rss"] for r in rs)
        dev = max(r["peak_device"] for r in rs)
        med = statistics.median(cores) if cores else None
        # user+sys: separates "ran on fewer cores" and "spun instead of slept" from a genuine slowdown.
        cpu = [r["cpu_user"] + r["cpu_sys"] for r in ok
               if r.get("cpu_user", -1) >= 0 and r.get("cpu_sys", -1) >= 0]
        avail = sorted({r.get("threads_avail", -1) for r in rs})
        bad_fields = inconsistencies(rs)
        if bad_fields:
            flagged.append((method, corpus, bad_fields))
        rows.append([
            method, corpus, tok, alphabet, f"{ntok:.3e}", thr,
            avail[0] if len(avail) == 1 else "/".join(map(str, avail)),
            len(rs), status,
            "/".join(sorted({r["verify"] for r in rs})),
            f"{med:.2f}" if med else "-",
            f"{min(cores):.2f}" if cores else "-",
            f"{max(cores):.2f}" if cores else "-",
            f"{ntok / med / 1e6:.2f}" if med else "-",
            f"{statistics.median(cpu):.0f}" if cpu else "-",
            f"{rss / GiB:.2f}G" if rss else "-",
            f"{rss / ntok:.2f}" if rss and ntok and status == "ok" else "-",    # Only for runs that complete; OOMs won't report their intensity consistently.
            f"{dev / GiB:.2f}G" if dev > 0 else "-",
            "!" if bad_fields else "",
        ])

    w = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(hdr)]
    line = lambda c: "  ".join(str(x).ljust(w[i]) for i, x in enumerate(c))
    print(line(hdr))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))

    # The matrix is no longer derivable from the script, so the manifest is the
    # only thing that distinguishes "never requested" from "requested but died".
    mpath = pathlib.Path(d) / "manifest.json"
    if mpath.exists():
        try:
            inv = json.loads(mpath.read_text())
            expected = sum(i.get("runs_expected", 0) for i in inv)
            print(f"manifest: {len(inv)} invocation(s), {expected} runs expected, {len(runs)} present")
            if expected > len(runs):
                print(f"  {expected - len(runs)} run(s) missing -- the job may have died; check slurm-*.out")
        except (json.JSONDecodeError, OSError) as e:
            print(f"manifest unreadable: {e}")
    else:
        print("no manifest.json -- expected-vs-present cannot be checked")

    ok_runs = [r for r in runs if r["status"] == "ok"]
    verified = [r for r in ok_runs if r["verify"] == "pass"]
    print(f"\n{len(runs)} records in {len(groups)} groups; "
          f"{len(verified)}/{len(ok_runs)} ok runs verified")
    for b in bad:
        print(f"UNPARSEABLE {b}")
    for m in check(runs):
        print(f"WARN: {m}")

    # `!` above means replicates of one configuration did not agree on something
    # that cannot legitimately vary between them.
    for method, corpus, fields in flagged:
        print(f"WARN: {method}/{corpus} replicates differ in {'; '.join(fields)}")

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
