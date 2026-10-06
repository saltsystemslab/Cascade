#!/usr/bin/env python3
"""infini-gram baseline harness: prep, kappa, check, run. Never edits their source."""


import argparse, hashlib, json, math, os, re, resource, shutil, subprocess, sys, threading, time
import numpy as np


# Benchmarked against infini-gram v2.6.0 (2025-05-15). Every file:line citation
# in this file is relative to that tag; the indexing.py numbers differ on master.
PIN_TAG = "v2.6.0"
PIN_SHA = "9567c3d7cad09a9750bc288bcce6f63adb768390"

SEP = 0xFFFF
HACK = 100000   # indexing.py:15; do not raise
ULIMIT = 262144 # their 1048576 default exceeds the hard limit; clamped at run time
THREADS = 128   # Override with --cpus.
RESERVED = ("tokenized", "table", "offset", "metadata", "metaoff", "unigram")
ALIAS = {"c4-1000shards": "c4", "c4-1024shards": "c4", "fineweb-350BT": "fineweb"}
TSV = ("label", "tokenizer", "sep", "tokens", "docs", "bytes", "sha256", "verified_at")


# Aborts, reporting `m`.
def die(m): sys.exit(f"FATAL: {m}")

# Returns the table record width in bytes that infini-gram derives for a
# token array of `ds` bytes.
def ratio_of(ds): return int(math.ceil(math.log2(ds) / 8))  # indexing.py:163

# Returns the per-token sort key their builder actually uses: it reads the
# little-endian array with `u16::from_be_bytes``, so the key is the byte-swapped
# token and the order is byte-wise over the on-disk stream.
def kappa(a): return a.byteswap()   # main.rs:243-246


# Aborts unless `ig_dir` is checked out at `PIN_TAG`.
def check_pin(ig_dir):
    r = subprocess.run(["git", "-C", ig_dir, "rev-parse", "HEAD"], capture_output=True, text=True)
    sha = r.stdout.strip() if r.returncode == 0 else ""
    if sha != PIN_SHA:
        die(f"{ig_dir} is {f'at {sha[:12]}' if sha else 'not a git checkout'}, not "
            f"{PIN_TAG} ({PIN_SHA[:12]}).\n"
            f"       git -C {ig_dir} fetch --tags && git -C {ig_dir} checkout {PIN_TAG}")


# Returns the SHA-256 hex digest of the file at `p`.
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)

    return h.hexdigest()


# Returns the `/proc/meminfo`` field `k` in kB, or `None` when absent.
def meminfo(k):
    try:
        return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith(k))
    except (OSError, StopIteration):
        return None


# Returns the live corpora.tsv row for `label`. The file is append-only and
# keeps pre-conversion rows, so the latest wins; a row that is not leading-
# separator is fatal, since infini-gram requires one.
def row(root, label, tok=None):
    hits = []
    for l in open(os.path.join(root, "corpora.tsv")):
        f = l.rstrip("\n").split("\t")
        if l.startswith("#") or len(f) < len(TSV):
            continue

        r = dict(zip(TSV, f))
        r.update({k: int(r[k]) for k in ("tokens", "docs", "bytes")})
        if r["label"] == label and tok in (None, r["tokenizer"]):
            hits.append(r)

    if not hits:
        die(f"no corpora.tsv row for {label}/{tok}")

    r = max(hits, key=lambda r: r["verified_at"])
    if r["sep"] != "first":
        die(f"{label} live row says sep={r['sep']!r}; infini-gram needs leading separators")
    return r


# Returns the path to the token array described by row `r`.
def find_input(root, r):
    lab, tok, short = r["label"], r["tokenizer"], ALIAS.get(r["label"], r["label"])
    for d in dict.fromkeys([short, lab]):
        for n in (f"{lab}-{tok}.tokenized.bin", f"{short}-{tok}.tokenized.bin", f"{short}.tokenized.bin", f"{lab}.tokenized.bin"):
            p = os.path.join(root, d, n)
            if os.path.isfile(p) and os.path.getsize(p) == r["bytes"]:
                if int(np.memmap(p, dtype="<u2", mode="r")[0]) != SEP:
                    die(f"{p} has no leading separator.")

                return p

    die(f"no {r['bytes']}-byte token array for {lab}/{tok} under {root}")


# Returns their batch count, total job count, and per-batch bytes for a `ds`-
# byte array under `mem_gib` and `cpus` (indexing.py:165-169).
def sizing(ds, mem_gib, cpus):
    nb = 1
    while nb * ((mem_gib * 1024**3) // 8) < ds: # indexing.py:165-167
        nb *= 2
    return nb, nb * cpus, ds / nb


# Returns the index base, baseline arm, token-array, and table paths for
# corpus row `r`, refusing a path holding a word their discovery reserves.
def paths(a, r):
    base = os.path.join(a.index_root, f"{r['label']}-{r['tokenizer']}")
    save = os.path.join(base, "baseline")

    # The engine classifies index files by substring over the full path
    # (cpp_engine.h:162-176) and build_sa() derives the table path the same way
    # (indexing.py:149), so a reserved word in a directory breaks both.
    bad = [w for w in RESERVED if w in base]
    if bad:
        die(f"reserved word {bad} in index path {base}")

    return base, save, os.path.join(save, "tokenized.0"), os.path.join(save, "table.0")


# Writes `offset.0` for `src`: one uint64 LE per document, the byte offset of
# its leading separator. Aborts unless every invariant the engine and the
# builder rely on holds, including a document count of `expect_docs`.
def emit_offsets(src, out, expect_docs):
    a = np.memmap(src, dtype="<u2", mode="r")
    docs = 0
    with open(out + ".part", "wb") as f:
        for i in range(0, a.shape[0], 1 << 27):
            # uint16, not bytes: a byte search for FF FF also matches across a
            # token boundary when one token ends in 0xFF and the next starts with it.
            idx = np.flatnonzero(np.asarray(a[i:i + (1 << 27)]) == SEP)
            (2 * (idx.astype(np.uint64) + np.uint64(i))).astype("<u8").tofile(f)
            docs += idx.size

    os.replace(out + ".part", out)
    od = np.memmap(out, dtype="<u8", mode="r")
    bad = [m for m, ok in (
        ("offset[0] != 0", int(od[0]) == 0),
        ("not ascending", bool(np.all(od[1:] > od[:-1]))),
        ("odd offset", not bool(np.any(od % 2))),
        ("out of range", not bool(np.any(od >= a.nbytes))),
        ("not on a separator", bool(np.all(np.asarray(a)[(od // 2).astype(np.int64)] == SEP))),
        (f"{docs} docs, expected {expect_docs}", docs == expect_docs)) if not ok]
    if bad:
        die("offset.0 invalid: " + "; ".join(bad))

    print(f"  offset.0    {docs:,} docs, all invariants ok")


# Returns the suffix array of `keys` by prefix doubling, ordering a shorter
# suffix first, as their SA-IS does by appending no sentinel.
def suffix_array(keys):
    n = keys.shape[0]
    _, rank = np.unique(keys, return_inverse=True)
    rank = np.ascontiguousarray(rank, dtype=np.int64)

    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        if k < n:
            second[:n - k] = rank[k:]
        o = np.lexsort((second, rank))      # last key is primary
        f, s = rank[o], second[o]
        new = np.empty(n, dtype=np.int64)
        new[o] = np.concatenate(([0], np.cumsum((f[1:] != f[:-1]) | (s[1:] != s[:-1]))))
        rank = new
        if int(rank[o[-1]]) == n - 1:
            return o
        k *= 2


# Returns `sa` in their table format: N x `ratio` bytes, little-endian, each
# entry the byte offset 2 * token_index (main.rs:252, to_bytes).
def pack(sa, ratio):
    v = (2 * sa.astype(np.uint64)).astype("<u8")
    return np.ascontiguousarray(v.view(np.uint8).reshape(-1, 8)[:, :ratio]).tobytes()


# Validates a table against its token array: the width the engine would
# derive, and the record invariants that detect a concat killed mid-write.
def check_table(ds_path, sa_path):
    a = np.memmap(ds_path, dtype="<u2", mode="r")
    n, sz = a.shape[0], os.path.getsize(sa_path)
    if sz % n:
        die(f"table {sz} not divisible by {n} tokens; the engine's assert would fire")

    k = sz // n
    if k != ratio_of(a.nbytes):
        die(f"ptr_size {k} != derived ratio {ratio_of(a.nbytes)}")

    zeros = odd = oob = 0
    with open(sa_path, "rb") as f:
        while True:
            buf = f.read(k * (1 << 24))
            if not buf:
                break
            w = np.zeros((len(buf) // k, 8), np.uint8)
            w[:, :k] = np.frombuffer(buf, np.uint8).reshape(-1, k)
            v = w.view("<u8").ravel()
            zeros += int(np.count_nonzero(v == 0))
            odd += int(np.count_nonzero(v % 2))
            oob += int(np.count_nonzero(v >= a.nbytes))

    # concat set_len's table.0 to full size before writing (main.rs:518-520), so a
    # killed run leaves a correct-sized, zero-tailed file that size cannot detect.
    # Only one record may be zero: the suffix at byte 0.
    bad = [m for m, ok in ((f"{zeros} zero records", zeros == 1),
                           (f"{odd} odd records", odd == 0),
                           (f"{oob} out of range", oob == 0)) if not ok]
    if bad:
        die("table corrupt: " + "; ".join(bad) + " -- delete it and rebuild")

    print(f"  table.0     ok ({n:,} records, ptr_size {k})")


# Builds our reference table for a token array, then either writes it or
# checks it byte-for-byte against one their builder produced.
def cmd_kappa(a):
    tok = np.asarray(np.memmap(a.tokenized, dtype="<u2", mode="r"))
    ratio = a.ratio or ratio_of(tok.nbytes)
    if a.selftest:
        m = min(4096, tok.shape[0])
        raw = np.ascontiguousarray(tok[:m]).tobytes()

        # Sorting kappa keys must equal sorting the raw LE byte stream.
        if list(suffix_array(kappa(tok[:m]))) != sorted(range(m), key=lambda i: raw[2 * i:]):
            die("suffix sort disagrees with a brute-force byte-slice sort")

        print(f"  selftest    ok ({m:,} tokens vs byte-slice sort)")

    blob = pack(suffix_array(kappa(tok)), ratio)
    print(f"  ours        {tok.shape[0]:,} tokens, ratio {ratio}, {len(blob):,} bytes")
    if a.emit:
        open(a.table, "wb").write(blob)
        return print(f"  wrote       {a.table}")

    theirs = open(a.table, "rb").read()
    if theirs == blob:
        return print("  PASS        byte-identical")

    n = ("size %d vs %d" % (len(theirs), len(blob)) if len(theirs) != len(blob) else
         "%d bytes" % int((np.frombuffer(theirs, np.uint8) != np.frombuffer(blob, np.uint8)).sum()))
    die(f"table differs ({n}). At fixture scale HACK is not in play, so this is a real bug.")


# Stages a corpus's input to the node-local drive. Called once before its
# replicates.
def cmd_stage(a):
    r = row(a.corpus_root, a.corpus, a.tokenizer)
    _, save, ds, _ = paths(a, r)
    d, secs, n = stage(os.path.realpath(ds),
                       os.path.realpath(os.path.join(save, "offset.0")),
                       a.stage_root, f"{r['label']}-{r['tokenizer']}")
    print(f"  staged      {n / 2**30:.1f} GiB to {d} in {secs}s" if n else
          f"  staged      already present at {d}")


# Lays out the baseline and gpu_saca index arms for a corpus: resolves the
# array, makes it read-only, writes offset.0, and links both arms to them.
def cmd_prep(a):
    r = row(a.corpus_root, a.corpus, a.tokenizer)
    src = find_input(a.corpus_root, r)
    base, _, _, _ = paths(a, r)
    print(f"  source      {src}\n  index       {base}")

    # tokenize() skips only when both tokenized.{i} and offset.{i} exist; else it
    # runs open(ds_path,'wb') and truncates every array for the worker. A symlink
    # does not protect its target, so make the inode read-only.
    if not a.no_chmod and (os.stat(src).st_mode & 0o777) != 0o444:
        os.chmod(src, 0o444)
        print("  chmod 444   source now read-only")

    od = os.path.join(base, "offset.0.shared")
    os.makedirs(os.path.join(base, "empty-data-dir"), exist_ok=True)
    if not os.path.exists(od) or a.force:
        emit_offsets(src, od, r["docs"])

    for arm in ("baseline", "gpu_saca"):
        d = os.path.join(base, arm)
        os.makedirs(d, exist_ok=True)
        for name, tgt in (("tokenized.0", src), ("offset.0", od)):
            p = os.path.join(d, name)
            if os.path.islink(p):
                os.unlink(p)
            os.symlink(tgt, p)

        has = os.path.exists(os.path.join(d, "table.0"))
        print(f"  {arm:<11} ready" + (" (table.0 present)" if has else ""))


# Node-wide counters `bench.hpp` records for every method. Cumulative since
# boot, so only the delta across a run matters, while under `--exclusive`.
def kernel_counters():
    c = dict.fromkeys(("thp_alloc", "thp_fallback", "pgscan_direct",
                       "compact_stall", "numa_local", "numa_other"), -1)
    vm = {"thp_fault_alloc": "thp_alloc", "thp_fault_fallback": "thp_fallback",
          "pgscan_direct": "pgscan_direct", "compact_stall": "compact_stall"}
    try:
        for line in open("/proc/vmstat"):
            k, _, v = line.partition(" ")
            if k in vm:
                c[vm[k]] = int(v)
    except OSError:
        pass

    n = 0
    while os.path.exists(f"/sys/devices/system/node/node{n}/numastat"):
        try:
            for line in open(f"/sys/devices/system/node/node{n}/numastat"):
                k, _, v = line.partition(" ")
                if k in ("local_node", "other_node"):
                    key = "numa_local" if k == "local_node" else "numa_other"
                    c[key] = max(c[key], 0) + int(v)
        except OSError:
            pass

        n += 1

    return c


# Free bytes across every node, zone and order, and the share of those sitting
# in 2 MiB-or-larger blocks (bench.hpp:220-256). Absolute at run start, not a
# delta: summarize.py reads it against peak_rss to tell a fragmented node from
# a full one.
def buddyinfo():
    c = {"buddy_free": -1, "buddy_free_huge": -1}
    page = os.sysconf("SC_PAGE_SIZE")
    huge = 0
    while (page << huge) < (2 << 20):
        huge += 1
    try:
        for line in open("/proc/buddyinfo"):
            z = line.find("zone")
            if z < 0:
                continue
            for order, blocks in enumerate(map(int, line[z + 4:].split()[1:])):
                b = blocks * (1 << order) * page
                c["buddy_free"] = max(c["buddy_free"], 0) + b
                if order >= huge:
                    c["buddy_free_huge"] = max(c["buddy_free_huge"], 0) + b
    except OSError:
        pass

    return c


# Node memory low-water at 1 Hz. Runs are --exclusive, so this is the whole
# build; a single-process peak does not exist here (make-part is `cpus`
# subprocesses, ru_maxrss is a max not a sum, no memory cgroup on this node).
class MemSampler(threading.Thread):

    def __init__(self):
        super().__init__(daemon=True)
        self.stop, self.lo = threading.Event(), None

    def run(self):
        """Samples MemAvailable once a second until stopped."""
        while not self.stop.is_set():
            v = meminfo("MemAvailable:")
            self.lo = v if self.lo is None else min(self.lo, v or self.lo)
            self.stop.wait(1.0)


# Copies the input to a node-local drive, returning its directory and the
# seconds taken. Replicates reuse a staged copy.
def stage(src_ds, src_od, root, tag):
    if not os.path.isdir(root) or not os.access(root, os.W_OK):
        die(f"stage root {root} is not a writable directory on "
            f"{os.uname().nodename}. The node-local drives differ per node "
            f"-- /tmp2 exists on d4067 only. Set --stage-root/IG_STAGE_ROOT.")

    parent = os.path.join(root, "ig-stage")
    os.makedirs(parent, exist_ok=True)
    for other in os.listdir(parent):
        if other != tag:
            shutil.rmtree(os.path.join(parent, other), ignore_errors=True)

    d = os.path.join(parent, tag)
    os.makedirs(d, exist_ok=True)
    t, copied = time.monotonic(), 0
    for name, src in (("tokenized.0", src_ds), ("offset.0", src_od)):
        dst = os.path.join(d, name)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        if os.path.exists(dst):
            os.unlink(dst)               # It is 444; copyfile would need write

        shutil.copyfile(src, dst)
        fd = os.open(dst, os.O_RDONLY)   # Flush now; writeback must not
        os.fsync(fd)                     # bleed into the first replicate
        os.close(fd)
        os.chmod(dst, 0o444)             # Same guard as the source: tokenize()
        copied += os.path.getsize(dst)   # Must fail and not truncate

    return d, (round(time.monotonic() - t, 1) if copied else 0.0), copied


# Runs their unmodified builder over one corpus and writes a benchmark
# record plus a sidecar. Skips itself when a record already exists.
def cmd_run(a):
    r = row(a.corpus_root, a.corpus, a.tokenizer)
    base, save, ds, sa = paths(a, r)
    if not os.path.exists(os.path.join(save, "offset.0")):
        die("offset.0 missing -- run `prep` first. Without it their tokenize() "
            "runs open(ds_path,'wb') and truncates the corpus.")

    n, ratio = r["bytes"] // 2, ratio_of(r["bytes"])
    cpus = a.cpus or THREADS
    avail, total = meminfo("MemAvailable:"), meminfo("MemTotal:")
    mem = a.mem or (avail and int(0.9 * avail / 1048576))
    if not mem:
        die("cannot read MemAvailable; pass --mem")

    nb, jobs, batch = sizing(r["bytes"], mem, cpus)
    # indexing.py:292 raises both soft and hard nofile, so asking above the hard
    # limit aborts at startup. Login nodes allow 524288 here, compute nodes 131072.
    fds = jobs + cpus * (jobs + 1)
    ulimit = min(ULIMIT, resource.getrlimit(resource.RLIMIT_NOFILE)[1])
    if fds > ulimit:
        die(f"merge needs ~{fds:,} fds but the hard limit is {ulimit:,}; "
            f"lower --cpus (fds go as cpus^2 * nb)")

    tag = f"{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    work = os.path.join(a.temp_root, f"ig-{tag}")
    temp, work_save = os.path.join(work, "temp"), os.path.join(work, "baseline")
    work_sa, empty = os.path.join(work_save, "table.0"), os.path.join(work, "empty")
    short = ALIAS.get(r["label"], r["label"])   # The other baselines record the aliased name, so group on it too.
    rid = a.run_id or f"infinigram_{short}-{r['tokenizer']}_token16_n{n}_t{cpus}_r{a.rep}"

    # Before the binary check: at the wrong tag the built binary is the wrong
    # binary, so "go build it" would be the wrong advice.
    check_pin(a.infinigram_dir)
    binary = os.path.join(a.infinigram_dir, "pkg", "infini_gram", "rust_indexing")
    if not os.access(binary, os.X_OK):
        die(f"{binary} missing; build it on a login node (see README)")

    if os.path.exists(os.path.join(a.results_dir, rid + ".json")) and not a.rerun:
        return print(f"== {rid}\n   skip (record exists)")

    if os.path.exists(sa):
        # A published table with no record is from an attempt that died before
        # writing one. Refuse rather than silently replace a good index.
        if a.dry_run:
            print(f"   note        {sa} exists; a real run needs --force")
        elif not a.force:
            die(f"{sa} exists without a record. Verify it with `check`, or pass "
                f"--force to keep it and run anyway.")

    print(f"== {rid}\n   {r['bytes'] / 2**30:.1f} GiB, {n:,} tokens, ratio {ratio}\n"
          f"   cpus {cpus}  mem {mem} GiB  nb {nb}  jobs {jobs}  "
          f"fds ~{fds:,}/{ulimit:,}\n"
          f"   make-part peak {batch * 8 / 2**30:.0f} GiB @8x, "
          f"{batch * 12.5 / 2**30:.0f} GiB @12.5x"
          + (f", node {total / 1048576:.0f} GiB" if total else ""))
    cmd = [sys.executable, "-m", "infini_gram.indexing",
           "--data_dir", empty, "--save_dir", work_save,
           "--temp_dir", temp, "--version", "4", "--token_dtype", "u16",
           "--shards", "1", "--workers", "1", "--cpus", str(cpus),
           "--mem", str(mem), "--ulimit", str(ulimit)]
    if a.dry_run:
        return print("   " + " ".join(cmd) + "\nDRY RUN")

    os.makedirs(a.results_dir, exist_ok=True)
    stage_dir, stage_s, staged = stage(
        os.path.realpath(ds), os.path.realpath(os.path.join(save, "offset.0")),
        a.stage_root, f"{r['label']}-{r['tokenizer']}")

    if staged:
        print(f"   staged      {staged / 2**30:.1f} GiB to {a.stage_root} in {stage_s}s")

    for d in (temp, work_save, empty):
        os.makedirs(d, exist_ok=True)

    for name in ("tokenized.0", "offset.0"):    # infini-gram's tokenize() skips only when both exist in save_dir; symlinks satisfy it.
        link = os.path.join(work_save, name)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)

        os.symlink(os.path.join(stage_dir, name), link)

    real = os.path.join(stage_dir, "tokenized.0")
    log = os.path.join(a.results_dir, rid + ".log")
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    k0, buddy = kernel_counters(), buddyinfo()
    s = MemSampler()
    s.start()
    with open(log, "wb") as lf:
        p = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=os.path.join(
            a.infinigram_dir, "pkg")), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            sys.stdout.write(line.decode("utf-8", "replace"))
        code = p.wait()

    s.stop.set()
    s.join(5)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    k1 = kernel_counters()
    kd = {k: (k1[k] - k0[k] if k0[k] >= 0 and k1[k] >= 0 else -1) for k in k0}

    # main.rs has no fsync, sync_all or flush anywhere, and BufWriter's drop
    # swallows errors, so concat's reported time is a copy into page cache, not a
    # durable write. Timing the flush here quantifies what their t_write omits,
    # and keeps one run's writeback out of the next run's make-part.
    fsync_s = None
    if os.path.exists(work_sa):
        try:
            t = time.monotonic()
            fd = os.open(work_sa, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
            fsync_s = round(time.monotonic() - t, 3)
        except OSError:
            pass

    out = open(log, errors="replace").read()
    g = lambda p: (lambda m: float(m.group(1)) if m else None)(re.search(p, out))
    ph = {"make_part": g(r"2\.1 \(make-part\): Done\. Took ([\d.]+)"),
          "merge": g(r"2\.2 \(merge\): Done\. Took ([\d.]+)"),
          "concat": g(r"2\.3 \(concat\): Done\. Took ([\d.]+)"),
          "build_sa": g(r"Step 2 \(build_sa\): Done\. Took ([\d.]+)")}

    status, detail = "ok", ""
    if code:
        status = "oom_host" if "memory allocation of" in out else "lib_error"
        detail = f"exit={code}"
    elif "Step 1 (tokenize): Skipped" not in out:
        status, detail = "bad_param", "tokenize() did not skip; verify the corpus"
    elif "Step 2 (build_sa): Skipped" in out or ph["build_sa"] is None:
        status, detail = "bad_param", "build_sa() skipped or never completed"
    elif not os.path.exists(work_sa) or os.path.getsize(work_sa) != n * ratio:
        status = "lib_error"
        detail = f"table.0 is {os.path.getsize(work_sa) if os.path.exists(work_sa) else 0}, want {n * ratio}"

    # Verify on the local copy while it is cheap, then publish to `/scratch`.
    copyout_s = None
    if status == "ok":
        check_table(real, work_sa)
        # infini-gram's SA build is deterministic with identical sha256 across runs.
        # Replicates should have the same bytes, so publish once per corpus.
        if os.path.exists(sa) and os.path.getsize(sa) == n * ratio:
            print(f"   published   already present; copy skipped")
        else:
            t = time.monotonic()
            shutil.copyfile(work_sa, sa)
            copyout_s = round(time.monotonic() - t, 1)
            print(f"   published   table.0 -> {sa} in {copyout_s}s")
    else:
        # A surviving table.0 would be taken for a finished index later.
        for f in (work_sa, sa):
            if os.path.exists(f):
                print(f"   removing {f}: run failed")
                os.unlink(f)

    if not a.keep_temp:
        shutil.rmtree(work, ignore_errors=True)

    # `bench.hpp`'s field set. The fault and switch counters come from RUSAGE_CHILDREN,
    # not RUSAGE_SELF: their pipeline is a subprocess tree, and these counters sum across children.
    rec = {
        "schema_version": 6, "run_id": rid, "method": "infinigram",
        "build_type": "Release",
        "method_version": PIN_TAG,
        "commit": os.environ.get("IG_COMMIT", "unknown"),
        "host": os.uname().nodename,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": real, "corpus": short, "tokenizer": r["tokenizer"],
        "alphabet": "token16", "symbol_bytes": 2, "n_symbols": n, "n_tokens": n,
        "threads": cpus,
        "threads_avail": cpus,
        "cpus_allowed": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else -1,
        "status": status, "code": code, "detail": detail, "verify": "off",
        "t_load": -1,   # Initial text read sits inside make-part and cannot be separated.
        "t_core": (round(ph["build_sa"] + (fsync_s or 0), 2)
                   if ph["build_sa"] is not None else -1),  # build_sa plus the flush. Their concat never fsyncs,
                                                            # so the output SA does not persist yet when they finish.
                                                            # and concat cannot be skipped: an empty path fails its `open()`` and
                                                            # `/dev/null`` fails its set_len(); concat is in the sidecar either way.
        "t_write": -1,
        "peak_rss": 0,                      # 0, not -1: summarize.py renders 0 as "-"
        "peak_device": -1,
        # CPU time sums across children, unlike ru_maxrss.
        "cpu_user": round(r1.ru_utime - r0.ru_utime, 6),
        "cpu_sys": round(r1.ru_stime - r0.ru_stime, 6),
        "minflt": r1.ru_minflt - r0.ru_minflt, "majflt": r1.ru_majflt - r0.ru_majflt,
        "nvcsw": r1.ru_nvcsw - r0.ru_nvcsw, "nivcsw": r1.ru_nivcsw - r0.ru_nivcsw,
        **kd, **buddy,
        # bench.hpp emits GPU_SACA's phase timers for every method; libsais
        # leaves them at -1 and so does this.
        **{k: -1.0 for k in ("t_prepage_pid", "t_pivots", "t_partition",
                             "t_device_setup", "t_initial_rank",
                             "t_exponentiate", "t_concatenate")},
        "n_partitions": -1, "n_generations": -1,
        "exe_hash": int(sha256(binary)[:16], 16),
        "env_tuning": ";".join(sorted(f"{k}={v}" for k, v in os.environ.items()
                                      if k.startswith(("OMP_", "GPU_SACA_")))),
    }

    peak = (total - s.lo) * 1024 if (s.lo and total) else None
    if peak:
        # bench.hpp reads VmHWM of one process. That does not exist here: their pipeline is `cpus` concurrent
        # subprocesses and this node has no memory cgroup, so the node-level low-water is the whole-build figure.
        rec["peak_rss"] = peak

    rec["notes"] = ("peak_rss = MemTotal - min(MemAvailable), node-level under "
                    "--exclusive, not a single-process VmHWM; "
                    f"phases make_part={ph['make_part']} merge={ph['merge']} "
                    f"concat={ph['concat']} (all three + fsync in t_core)")
    side = {"run_id": rid, "phases_s": ph, "mem_gib": mem, "nb": nb, "jobs": jobs,
            "cpus": cpus, "ulimit": ulimit, "hacksize": HACK, "ratio": ratio,
            "per_batch_bytes": int(batch), "node_peak_bytes": peak,
            "observed_multiplier": round(peak / batch, 2) if peak else None,
            "table_fsync_s": fsync_s,
            "status": status, "detail": detail,
            "exit_code": code, "command": cmd,
            "table_sha256": sha256(sa) if status == "ok" else None,
            "stage_root": a.stage_root, "stage_s": stage_s, "copyout_s": copyout_s,
            "infinigram_version": PIN_TAG}

    # Sidecar in a subdirectory: summarize.py globs *.json in the results dir and
    # would read it as a malformed record.
    sd = os.path.join(a.results_dir, "sidecars")
    os.makedirs(sd, exist_ok=True)
    for obj, path in ((rec, os.path.join(a.results_dir, rid + ".json")),
                      (side, os.path.join(sd, rid + ".json"))):
        json.dump(obj, open(path + ".tmp", "w"), indent=2)
        os.replace(path + ".tmp", path)

    print(f"   status      {status} {detail}\n   t_core      {rec['t_core']}s "
          f"= make-part {ph['make_part']} + merge {ph['merge']} "
          f"+ concat {ph['concat']} + fsync {fsync_s}"
          + (f"\n   node peak   {peak / 2**30:.1f} GiB = {peak / batch:.2f}x per-batch"
             if peak else ""))
    sys.exit(0 if status == "ok" else 1)


# Runs our own builder end to end on the staged input, dumping the suffix array
# to the node-local drive so the output cost lands inside its concatenate phase
# (bench_gpu_saca.cu:129), as their concat does in build_sa.
def cmd_ours(a):
    if not a.ours_bin or not os.access(a.ours_bin, os.X_OK):
        die(f"--ours-bin/OURS_BIN must name a built bench_gpu_saca; got {a.ours_bin!r}. "
            f"bench.sbatch builds one per tag under ~/gpu-saca-worktrees/<tag>/build.")

    r = row(a.corpus_root, a.corpus, a.tokenizer)
    _, save, ds, _ = paths(a, r)
    short = ALIAS.get(r["label"], r["label"])
    cpus = a.cpus or THREADS
    n = r["bytes"] // 2
    # --n 0 lets the driver size from the file, which is what bench.sbatch passes.
    rid = a.run_id or f"gpu_saca_{short}-{r['tokenizer']}_token16_n0_t{cpus}_r{a.rep}"
    rec = os.path.join(a.results_dir, rid + ".json")
    if os.path.exists(rec) and not a.rerun:
        return print(f"== {rid}\n   skip (record exists)")

    if a.dry_run:   # Staging is `run`'s path too; this checks the invocation.
        print(f"== {rid}\n   " + " ".join(
            [a.ours_bin, "--input", f"{a.stage_root}/ig-stage/"
             f"{r['label']}-{r['tokenizer']}/tokenized.0", "--corpus", short,
             "--tokenizer", r["tokenizer"], "--run-id", rid, "--alphabet", "token16",
             "--n", "0", "--threads", str(cpus), "--results-dir", a.results_dir,
             "--sa-out", os.path.join(a.temp_root, "ours-<jobid>.sa"),
             "--method-version", a.method_version]))
        return print(f"   {r['bytes'] / 2**30:.1f} GiB in, "
                     f"{n * 5 / 2**30:.1f} GiB out")

    # The same copy their runs read, staged once by the caller before the replicates.
    stage_dir, stage_s, staged = stage(os.path.realpath(ds),
                                       os.path.realpath(os.path.join(save, "offset.0")),
                                       a.stage_root, f"{r['label']}-{r['tokenizer']}")
    real = os.path.join(stage_dir, "tokenized.0")
    sa_out = os.path.join(a.temp_root, f"ours-{os.environ.get('SLURM_JOB_ID', os.getpid())}.sa")
    print(f"== {rid}\n   {r['bytes'] / 2**30:.1f} GiB, {n:,} symbols, "
          f"{n * 5 / 2**30:.1f} GiB of output -> {sa_out}\n"
          f"   cpus {cpus}  input {real}"
          + (f"  (staged {staged / 2**30:.1f} GiB in {stage_s}s)" if staged else ""))

    cmd = [a.ours_bin, "--input", real, "--corpus", short,
           "--tokenizer", r["tokenizer"], "--run-id", rid, "--alphabet", "token16",
           "--n", "0", "--threads", str(cpus), "--results-dir", a.results_dir,
           "--sa-out", sa_out, "--method-version", a.method_version]
    t = time.monotonic()
    try:
        with open(os.path.join(a.results_dir, rid + ".log"), "wb") as lf:
            code = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    finally:
        # A surviving dump would fill the drive the next replicate spills onto.
        if os.path.exists(sa_out):
            print(f"   dumped      {os.path.getsize(sa_out):,} bytes, want {n * 5:,}")
            os.unlink(sa_out)

    print(f"   exit {code}  wall {time.monotonic() - t:.1f}s"
          + ("" if os.path.exists(rec) else "  -- no record written"))
    sys.exit(0 if code == 0 else 1)


# Appends this invocation to the campaign's manifest, the only thing that tells
# "never requested" apart from "requested but died" (summarize.py:278-291).
# Appended, not overwritten: several jobs accumulate into one directory.
def cmd_manifest(a):
    pairs = [x.split(":") for x in a.corpora.split()]
    methods = a.methods.split(":")
    rows = [row(a.corpus_root, lab, tok) for lab, tok in pairs]
    path = os.path.join(a.results_dir, "manifest.json")
    m = json.load(open(path)) if os.path.exists(path) else []
    m.append({"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
              "corpora": [lab for lab, _ in pairs],
              "tokenizer": ":".join(dict.fromkeys(t for _, t in pairs)),
              "methods": methods,
              "ours_tag": os.environ.get("OURS_TAG", ""),
              "reps": a.reps, "runs_expected": a.reps * len(pairs) * len(methods),
              "corpus_root": a.corpus_root, "results": a.results_dir,
              "sizes_bytes": [r["bytes"] for r in rows],
              "commit": os.environ.get("IG_COMMIT", "unknown")})
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")
    print(f"  manifest    {len(m)} invocation(s), {sum(i['runs_expected'] for i in m)} runs expected")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Shared via parents= so they may follow the subcommand, which is where
    # anyone would naturally type them. Defaults come from the environment.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus-root", default=os.environ.get(
        "CORPUS_ROOT", f"/scratch/{os.environ.get('USER', '')}/corpora"))
    common.add_argument("--index-root", default=os.environ.get(
        "IG_INDEX_ROOT", f"/scratch/{os.environ.get('USER', '')}/ig-index"))
    common.add_argument("--infinigram-dir", default=os.environ.get(
        "INFINIGRAM_DIR", os.path.expanduser("~/infini-gram")))
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("prep", parents=[common], help="lay out both arms, generate offset.0, chmod 444")
    q.add_argument("corpus")
    q.add_argument("--tokenizer")
    q.add_argument("--no-chmod", action="store_true")
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_prep)

    q = s.add_parser("stage", parents=[common], help="copy a corpus's input to the node-local drive")
    q.add_argument("corpus")
    q.add_argument("--tokenizer")
    q.add_argument("--stage-root", default=os.environ.get("IG_STAGE_ROOT", "/tmp2"))
    q.set_defaults(fn=cmd_stage)

    q = s.add_parser("ours", parents=[common], help="run our builder end to end on the staged input")
    q.add_argument("corpus")
    q.add_argument("--tokenizer")
    q.add_argument("--results-dir", required=True)
    q.add_argument("--ours-bin", default=os.environ.get("OURS_BIN"))
    q.add_argument("--method-version", default=os.environ.get("OURS_TAG", "unknown"))
    q.add_argument("--cpus", type=int, help=f"default {THREADS}")
    q.add_argument("--temp-root", default=os.environ.get("IG_TEMP_ROOT", "/tmp"))
    q.add_argument("--stage-root", default=os.environ.get("IG_STAGE_ROOT", "/tmp2"))
    q.add_argument("--run-id")
    q.add_argument("--rep", type=int, default=1)
    q.add_argument("--rerun", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(fn=cmd_ours)

    q = s.add_parser("manifest", parents=[common], help="append this job to the campaign manifest")
    q.add_argument("corpora", help="space-separated label:tokenizer pairs")
    q.add_argument("--results-dir", required=True)
    q.add_argument("--reps", type=int, default=1)
    q.add_argument("--methods", default="infinigram")
    q.set_defaults(fn=cmd_manifest)


    q = s.add_parser("kappa", help="reference table; byte-identity check against theirs")
    q.add_argument("tokenized")
    q.add_argument("table")
    q.add_argument("--emit", action="store_true", help="write ours instead of comparing")
    q.add_argument("--ratio", type=int)
    q.add_argument("--selftest", action="store_true")
    q.set_defaults(fn=cmd_kappa)

    q = s.add_parser("check", help="table integrity")
    q.add_argument("tokenized")
    q.add_argument("table")
    q.set_defaults(fn=lambda a: check_table(a.tokenized, a.table))

    q = s.add_parser("run", parents=[common], help="run their unmodified builder and record it")
    q.add_argument("corpus")
    q.add_argument("--tokenizer")
    q.add_argument("--results-dir", required=True)
    q.add_argument("--mem", type=int)
    q.add_argument("--cpus", type=int, help=f"default {THREADS}")
    q.add_argument("--temp-root", default=os.environ.get("IG_TEMP_ROOT", "/tmp"),
                   help="node-local drive for temp and output (default /tmp)")
    q.add_argument("--stage-root", default=os.environ.get("IG_STAGE_ROOT", "/tmp2"),
                   help="node-local drive for the staged input (default /tmp2)")
    q.add_argument("--run-id")
    q.add_argument("--rep", type=int, default=1)
    q.add_argument("--keep-temp", action="store_true")
    q.add_argument("--force", action="store_true", help="delete a stale table.0 first")
    q.add_argument("--rerun", action="store_true", help="redo a run that has a record")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(fn=cmd_run)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
