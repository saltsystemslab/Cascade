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

# (major, minor), as in semantic versioning. Mixing across one major is the
# normal case and must not warn: our method is re-measured at every tag while a
# baseline's numbers stand, so a campaign pair will routinely span minors. Only
# a major says a field changed meaning, and only then are the two sets of
# numbers actually incomparable.
SCHEMA = (6, 2)

# A run is comparable to another run only if the machine was in the same state.
# thp_alloc + thp_fallback is a constant of the workload separates the good runs
# from the bad ones:
#
#      0.0%  every run on a healthy node (2026-09-19 placement test x4,
#            and the same campaign's dolmasample x3)
#      6.4%  node marginally crowded                    t_core   +8%
#     45.0%  forced local against a full node                   +35%
#    24-78%  the 2026-09-19 campaign's degraded runs     +36% to +124%
THP_REFUSED_MAX = 0.10

# A refusal rate over the bar means one of two different things, and they call
# for opposite responses, so the report must not conflate them:
#
#   inherited  the node was already fragmented when the run started. The number
#              is invalid and re-running on a clean node will fix it.
#   intrinsic  the working set is close enough to the node's supply of 2 MiB
#              blocks that the run exhausts it as it allocates. Every honest
#              measurement of this corpus on this machine will look like this,
#              and re-running will not help.
#
# `buddy_free_huge` at run start against the run's own `peak_rss` separates them.
# Below 1.0 the shortfall is arithmetic -- the blocks to serve the working set
# were never there. github sits at 1.01-1.21x and refuses 41-45%; the tightest
# clean run on record is arxiv at 2.2x, refusing nothing. So the boundary lies
# somewhere in (1.21, 2.2) and 1.25 is deliberately near the bottom of it: the
# conservative error is to call a dirty node "intrinsic" and excuse a number
# that deserved a re-run.
INTRINSIC_HEADROOM_MAX = 1.25

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


def schema_of(r):
    """-> (major, minor). A record without `schema_minor` predates the scheme."""
    major = r.get("schema_version")
    if not isinstance(major, int):
        return None
    minor = r.get("schema_minor", 0)
    return (major, minor if isinstance(minor, int) else 0)


def check(runs):
    """Provenance guards. Each catches a whole class of invalid table."""
    warn = []
    seen = {schema_of(r) for r in runs}
    if None in seen:
        warn.append("some records carry no usable schema_version")
        seen.discard(None)
    majors = {v[0] for v in seen}
    if len(majors) > 1:
        warn.append(f"records span schema majors {sorted(majors)}: a field changed "
                    "meaning between them, so these numbers are not comparable")
    elif majors and majors != {SCHEMA[0]}:
        warn.append(f"records are schema major {majors.pop()}, this summarizer is "
                    f"{SCHEMA[0]}: a field may not mean what is printed here")
    ahead = sorted(v for v in seen if v[0] == SCHEMA[0] and v[1] > SCHEMA[1])
    if ahead:
        warn.append(f"records at {ahead} were written by a newer harness than this "
                    f"summarizer ({SCHEMA[0]}.{SCHEMA[1]}); fields it does not know are ignored")
    commits = {r.get("commit", "unknown") for r in runs}
    if "unknown" in commits:
        warn.append("some records have commit=unknown: CMake did not pass BENCH_COMMIT")
    if any(c.endswith("-dirty") for c in commits):
        warn.append("some records come from a dirty working tree")
    if len(commits) > 1:
        warn.append(f"records span {len(commits)} commits {sorted(commits)}: not comparable")
    # One commit is not one binary. `bench.sbatch`'s --ours path configured
    # without PREF_KEY_WORDS for a long time, so the same source could be built
    # narrow or wide with nothing in the record to tell them apart.
    # Records predating the field do not carry it; absent is "not measured",
    # not a bad value, so they sit this check out the way pre-schema-6 records
    # sit out the fragmentation one.
    configs = {r["build_config"] for r in runs if "build_config" in r}
    if "unknown" in configs:
        warn.append("some records have build_config=unknown: CMake did not pass BENCH_BUILD_CONFIG")
    if len(configs - {"unknown"}) > 1:
        warn.append(f"records span build configurations {sorted(configs)}: not comparable")
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


def refused_fraction(r):
    """Share of this run's huge-page requests the kernel turned down, or None
    where nothing was measured (pre-schema-5, or not Linux)."""
    alloc, fb = r.get("thp_alloc", -1), r.get("thp_fallback", -1)
    if alloc < 0 or fb < 0 or alloc + fb == 0:
        return None
    return fb / (alloc + fb)


def fragmentation(rs):
    """Runs that could not get the huge pages the clean runs got. Unlike
    `inconsistencies` this is not a harness fault: it is the machine reporting
    its own state. Each entry says whether a re-run would help."""
    out = []
    for r in rs:
        # An OOM never ran the algorithm -- it failed in the allocator -- so its
        # refusal rate describes the node, not a result. Same reason `B/tok` is
        # suppressed for OOMs.
        if r.get("status") != "ok":
            continue
        frac = refused_fraction(r)
        if frac is None or frac <= THP_REFUSED_MAX:
            continue

        huge, rss = r.get("buddy_free_huge", -1), r.get("peak_rss", 0)
        if huge < 0 or rss <= 0:
            why = "headroom unknown (pre-schema-6 record)"
        else:
            head = huge / rss
            why = (f"huge@start {huge / GiB:.0f}G vs peak_rss {rss / GiB:.0f}G "
                   f"= {head:.2f}x headroom: "
                   + ("intrinsic to this corpus on this node, a re-run will not help"
                      if head <= INTRINSIC_HEADROOM_MAX else
                      "the node was already fragmented, so re-run on a clean one"))
        out.append(f"{r['run_id']} refused {frac:.1%} ({why})")
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
        frag = fragmentation(rs)
        if bad_fields:
            flagged.append((method, corpus, "replicates differ in", bad_fields))
        if frag:
            flagged.append((method, corpus, "ran on a fragmented node "
                            f"(>{THP_REFUSED_MAX:.0%} of huge pages refused):", frag))
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
            "!" if bad_fields or frag else "",
        ])

    w = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(hdr)]
    line = lambda c: "  ".join(str(x).ljust(w[i]) for i, x in enumerate(c))
    print(line(hdr))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))

    mem = []
    for (method, corpus, tok, alphabet, ntok, thr), rs in sorted(groups.items(),
                                                                 key=lambda kv: kv[0][4]):
        def med(key):
            v = [r[key] for r in rs if r.get(key, -1) >= 0]
            return statistics.median(v) if v else None

        alloc, fb = med("thp_alloc"), med("thp_fallback")
        loc, oth = med("numa_local"), med("numa_other")
        scan, comp = med("pgscan_direct"), med("compact_stall")
        if alloc is None and loc is None:
            continue

        # The worst replicate, not the median: this is a health check, and a group
        # with one degraded run is a group holding one number that cannot be
        # pooled with the rest.
        ratios = [f for f in (refused_fraction(r) for r in rs if r.get("status") == "ok")
                  if f is not None]
        refused = max(ratios) if ratios else None

        # The replicate that started with the least huge-page-capable memory,
        # reported with its own free total so the pair stays coherent. Read them
        # together: small `huge` beside a large `free` is a fragmented node,
        # small beside small is merely a full one, and the two want different
        # remedies. This is the state the run inherited -- `thp_refused` is what
        # that state then cost it.
        starts = [(r["buddy_free_huge"], r["buddy_free"]) for r in rs
                  if r.get("buddy_free_huge", -1) >= 0 and r.get("buddy_free", -1) >= 0
                  and r.get("status") == "ok"]
        huge0, free0 = min(starts) if starts else (None, None)

        mem.append([
            method, corpus,
            f"{refused:.1%}{'*' if refused > THP_REFUSED_MAX else ''}" if refused is not None else "-",
            f"{oth / (loc + oth):.1%}" if loc is not None and (loc + oth) else "-",
            f"{scan:.3e}" if scan else "-",
            f"{comp:.3e}" if comp else "-",
            f"{free0 / GiB:.0f}G" if free0 is not None else "-",
            f"{huge0 / GiB:.0f}G" if huge0 is not None else "-",
        ])

    if mem:
        mh = ("method", "corpus", "thp_refused", "numa_remote", "direct_reclaim",
              "compaction", "free@start", "huge@start")
        mw = [max([len(str(h))] + [len(str(r[i])) for r in mem]) for i, h in enumerate(mh)]
        print("\nmemory behaviour (node-wide deltas; meaningful only under --exclusive)")
        print(f"thp_refused and the @start pair are the worst replicate, the rest are "
              f"medians.  * over {THP_REFUSED_MAX:.0%}: the node could not serve that "
              "run's huge pages,\n  so its t_core is not comparable to one that got them."
              "  huge@start well under free@start means the node was fragmented, not full")
        print("  ".join(str(x).ljust(mw[i]) for i, x in enumerate(mh)))
        print("  ".join("-" * x for x in mw))
        for r in mem:
            print("  ".join(str(x).ljust(mw[i]) for i, x in enumerate(r)))

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

    # `!` above means either that replicates of one configuration did not agree
    # on something that cannot legitimately vary between them, or that the node
    # was fragmented when the run started.
    for method, corpus, why, details in flagged:
        print(f"WARN: {method}/{corpus} {why} {'; '.join(details)}")

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
