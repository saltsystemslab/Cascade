# Baseline harness notes

Facts that cost time to find. `bench.hpp` and the drivers point here rather than
carrying the explanations inline.

Node hardware, the walls derived from it, and site configuration live in
`local-node.md` and `local.env`, both untracked — nothing here identifies a
machine or a user.

## Running it

Campaigns go through the API, which validates everything on the submit host
before anything is queued:

```
cp benchmark/local.env.example benchmark/local.env   # once: partition, node, roots
. benchmark/local.env
cmake -S . -B build -DBENCHMARK=ON -DCMAKE_BUILD_TYPE=Release   # configure only

benchmark/bench.sh --corpora dolmasample,arxiv --methods libsais --reps 3
benchmark/bench.sh --corpora c4 --ours v0.5.0 --reps 1 --results-dir <same dir>

python3 benchmark/summarize.py <results-dir>         # [--tsv out.tsv]
sbatch benchmark/verify.sbatch                       # correctness, own directory
```

`bench.sh --help` lists the flags. `--dry-run` prints the expanded run set and the
per-corpus sizes without submitting. `--methods` defaults to **empty**, so
`--corpora X --ours TAG` runs only our method; at least one of `--methods` /
`--ours` must be given.

**Our method is run by tag.** `--ours TAG` builds it from a git worktree checked
out at that tag, under `WORKTREE_ROOT` (default `${HOME}/gpu-saca-worktrees}`),
reused across campaigns. The binary therefore provably corresponds to the version
its records name, and re-running an older tag later reproduces its numbers. A tag
that predates `src/bench_gpu_saca.cu` cannot build, which is the intended signal
that it is not benchmarkable under this policy -- the driver, the no-dump mode and
the timings accessor all have to be present.

**The thread-choice control is two invocations**, not a separate experiment — it
exists so "libsais, all cores" is a measured choice rather than an assumption:

```
benchmark/bench.sh --corpora dolmasample --methods libsais --reps 3 --name thr
benchmark/bench.sh --corpora dolmasample --methods libsais --reps 3 --threads 64 \
                   --results-dir <the same dir>
```

The run-id carries `t<threads>`, so the two land as separate rows in one summary.

**Two invocations can share a `--results-dir`.** Run-ids already carry the
replicate index and every run is idempotent, so small corpora at 3 replicates and
a large one at 1 merge into a single campaign, and the summary reports differing
replicate counts per row with no special handling. A resubmission after a job
dies skips what already completed.

**Where things land.** `RESULTS_ROOT` (default `${HOME}/gpu-saca-results`) holds
one directory per campaign, `<UTC>[-<name>]`, containing the per-run JSON records,
`slurm-<jobid>.out/.err`, and `manifest.json`. Nothing is written inside the repo:
the sbatches require the directory as an argument rather than defaulting to a
repo-relative path, since that would have made the results location depend on
which directory `sbatch` was run from.

`manifest.json` records each invocation's corpora, tokenizer, methods, tag,
replicate count and resolved sizes, so `summarize.py` can report expected-vs-present
and tell "never requested" apart from "requested but the job died".

**Configure on a host with network; compile inside the job.** Both baselines are
fetched, so `cmake -S -B` needs internet — which compute nodes often lack — but it
only downloads two small tarballs. Compiling the CUDA sources is the CPU-heavy half
and would blow a login node's CPU-time limit, so the sbatch does it in-job with
every core of the allocation, which also guarantees the commit stamped into the
binary is the one that built it. `BUILD_IN_JOB=0` skips it.

**Submit from the repo root**, or `export REPO=/path/to/repo` first. Slurm runs a
*copy* of the batch script from a spool directory, so it cannot locate the repo
from its own path; it uses `SLURM_SUBMIT_DIR` and refuses up front if that does
not look like the repo root.

Corpora live at `${CORPUS_ROOT}/<corpus>/<corpus>-<tokenizer>.tokenized.bin`. The
tokenizer is one per invocation and is recorded in every record and run-id: the
same corpus under two tokenizers is two different inputs with two different n, so
mixing them within one table would confound both n and throughput.

## libsais 2.10.4

- **Dynamic threads must be off, and the driver is what enforces it.** The main
  parallel region is guarded by `threads > 1 && n >= 65536 && omp_get_dynamic() == 0`.
  With dynamic threads on it silently runs serial: a thread sweep comes back flat
  with no error anywhere. `bench_libsais` refuses and records `bad_param`, so the
  wrong number cannot be produced. The harness deliberately does **not** export
  `OMP_DYNAMIC=false` on top of that: libgomp already defaults it off, and the
  export bought nothing while making the harness look like it tunes OpenMP.
- **`--threads 0` is not serial** — it resolves to `omp_get_max_threads()`
  (`libsais16x64.c:7204`). Pass it explicitly anyway so the record is unambiguous.
- **Peak is 10n–12.4n and corpus-dependent.** `libsais16x64.c:6860,6891` allocate
  `k * sizeof(sa_sint_t)` when `fs < k`, where `k` is the reduced-problem
  alphabet — up to ~2.4n extra. It happens *inside the recursion*, i.e. late, so
  a near-wall run can burn a long time before dying. Do not hardcode 11n; calibrate
  `peak_rss/n` per corpus from that corpus's own smaller runs. **The model is
  asymptotic**: measured on a 5e6-token corpus the peak is ~20n, not 11n, so do
  not extrapolate the wall from a small run.
- **Return codes:** `0` ok, `-2` allocation failure, `-1` bad arguments — where
  `-1` also covers null pointers, `n < 0`, `fs < 0` *and* `threads < 0`. The
  driver puts the arguments in `detail` so a `-1` is diagnosable from the record.
- **There is a throughput discontinuity at n = 2^31.** Below it,
  `libsais16x64`/`libsais64` delegate to the 32-bit implementation
  (`libsais16x64.c:7207`), writing int32 into the int64 SA and converting in
  place. If a curve kinks near 2e9, this is why.
- Measured: 1 -> 4 threads moves a 5e6-token run only 12.2 -> 14.7 Mtok/s,
  which is the next point in concrete form.
- **`libsais16x64` stays serial much longer than `libsais64`.** The induced-sort
  regions guard on `block_size >= 64 * max(k, 256)` with `ALPHABET_SIZE = 65536`,
  so they need `block_size >= 4 194 304` versus 16 384 for a byte alphabet. A
  thread sweep below ~1e7 tokens measures nothing. Any "libsais16x64 scales
  poorly" claim has to state this.
- **Linking:** upstream's target is named `libsais` with `PREFIX ""` and exports
  `PUBLIC LIBSAIS_OPENMP` + `OpenMP::OpenMP_C`. That OpenMP interface is
  `$<COMPILE_LANGUAGE:C>`-guarded and so never reaches our `.cpp`, which calls
  `omp_get_dynamic()` — hence the explicit `OpenMP::OpenMP_CXX`. `LIBSAIS_OPENMP`
  gates the `_omp` *declarations*, so forgetting it is a compile error rather
  than a silent serial run.

## libcubwt 1.0.0 (`b1324f4d`)

- **v1.0.0 is the only release that ever supported suffix-array construction.**
  1.5.0 removed the SA/ISA entry points and no 1.1–1.4 tags exist. This preempts
  "why not a newer libcubwt?"
- **It is the only actively maintained single-GPU SACA.** Büren et al. (HPDC'19)
  reaches further but is multi-GPU (DGX-1 + NVLink); its own baselines, CUDPP and
  NVBIO, are both unmaintained (NVBIO archived by NVIDIA). Osipov'12,
  Deo & Keely'13, Wang'16 are unmaintained research code. So Büren is the right
  thing to *cite* for reach, not to run.
- **The fetched v1.0.0 tarball is byte-identical to what this repo used to
  vendor, and it builds for `sm_120` unpatched** despite predating Blackwell. No
  patch to disclose.
- `ptxas` emits ~20 `.minnctapersm will be ignored` warnings on `sm_120`. They
  come from upstream's own `__launch_bounds__` and are expected.
- **Device budget is ~30 B per input symbol** and is a pure function of
  `max_length`, so the wall is exactly locatable — and on any card in the ~100 GB
  class it binds around 3e9 symbols, i.e. well below the API ceiling below.
- **API limits:** `n >= 16` required; `max_length` is rounded up to
  `CUDA_DEVICE_PADDING = 8192` and rejected at `>= UINT32_MAX`, giving an exact
  ceiling of **4 294 959 104** symbols. uint8 alphabet, uint32 SA.
- **All nine `cudaMalloc`s happen inside `libcubwt_allocate_device_storage`;
  `libcubwt_sa` allocates nothing.** So peak device usage *is* post-allocate
  usage, and the `cudaMemGetInfo` delta across that call is exact. Force context
  creation with `cudaFree(nullptr)` before sampling the baseline, or libcubwt is
  charged for the context's few hundred MB. Do not compute the budget instead:
  two of the nine sizes come from CUB temp-storage queries and would be
  CCCL-version dependent.
- **Its SA is allocated on the host at 4n bytes**, so on C4-read-as-bytes
  (n = 3.4e11) that is 1366 GB on top of a 341 GB corpus — over host memory. The
  driver therefore allocates *device* storage first: otherwise the C4 row would
  report `oom_host` for what is really a GPU wall.
- **Its H2D copy of `T` is from unpinned memory** (`libcubwt.cu:675`), so it runs
  at pageable-copy bandwidth. Not currently addressed; `cudaHostRegister` is the
  experiment to try, and registering a multi-GB buffer may well be a net loss
  for a single-shot run.
- **Do not put the source directory on the include path.** It contains a
  plain-text file named `VERSION`, and on a case-insensitive filesystem — which
  the macOS bind mount behind `./build.sh` is — libstdc++'s `#include <version>`
  resolves to it and detonates inside `<tuple>`. This breaks the local
  compile-check while building fine on the Linux cluster. `CMakeLists.txt` stages
  the header into its own directory to avoid it.
- **`CUDA_SEPARABLE_COMPILATION` must be OFF.** A `-dc` static library cannot
  link into a pure-C++ executable: CMake emits the CUDA device-link step only for
  a target that itself has CUDA sources. Off also matches upstream's own
  `nvcc -c` build. This is what lets `bench_libcubwt` be a `.cpp`, so nvcc never
  compiles `bench.hpp`.

## The assumption the cross-check rests on

**libsais and libcubwt order suffixes the same way: shorter-suffix-first, with no
appended sentinel.** The `reference_sa()` in the driver this harness replaced
compared with `memcmp` over `min(la, lb)` and then `la < lb`, and matched
libcubwt's output; libsais uses the same convention, and `verify_SA`'s
`rank_next() == -1` past the end encodes it.

This is *why* comparing the two on identical byte input is valid at all, and it
was previously recorded only in a commented-out line. Both drivers write the SA
as canonical little-endian uint64 regardless of native width (libcubwt's is
uint32), so the cross-check is:

```
cmp libsais64.sa libcubwt.sa
```

**One pairing is not legal:** `libsais16x64` on tokens against `libcubwt` on the
same file as bytes. Tokens are little-endian uint16, so byte-wise suffix order
compares the low byte first and is *not* token order, and odd-offset suffixes are
mid-token. Such a check fails correctly and wastes a day.

## Harness conventions

- **A missing record means the process died.** Records are written to a temporary
  and `rename`d, so absence is never a half-written file. This is what makes the
  runner's log interpretable without a Slurm accounting query.
- **Failure is a result.** The drivers exit 0 whenever a record was written,
  including for OOM — an OOM at C4 is precisely the datum E2.1 rests on. A
  non-zero exit means *no record exists*, the one case the runner cannot
  reconstruct.
- **Timing:** `-1` is "not measured", `0.0` is "measured as zero". There is no
  `t_e2e` field; it is the sum of the buckets that actually ran, derived in the
  summarizer so the two definitions cannot drift.
- **`t_core` is the headline, not `t_e2e`.** `t_e2e` folds in `t_load` and
  `t_write`, and libcubwt's native SA is 4n against libsais's 8n, so `t_write`
  differs 2x for reasons unrelated to either algorithm — and at the large token
  rungs no SA is written at all. (`t_e2e` remains right for E1, a pipeline claim.)
- **`peak_rss` is read at emission, after everything the run did.** Correct,
  because `VmHWM` is a lifetime high-water mark and not an instantaneous sample.
  Do not "fix" this by reading it earlier. `/proc` is Linux-only; the macOS
  fallback is `getrusage`, whose `ru_maxrss` is bytes there but KiB on Linux.
- Verification never runs in a timing job — see "Running it". `verify.sbatch`
  owns it.
- Our method aborts via `std::exit` from inside `core` on algorithmic failures
  (unsuitable partition sizes, too little device memory), which leaves **no
  record** -- read the job log for those. The driver pre-checks the cheap,
  common environmental failures instead (input readable and an even byte count,
  `GPU_SACA_SCRATCH` set and writable, a usable CUDA device), so those become
  records the way the baselines' do. `core`'s own scratch check fires only after
  the text load and the partition-ID pass, i.e. minutes of wasted work.
- **`verify` has four states**, and the distinction matters: `off` (never
  requested — a recorded policy choice), `pass`, `fail`, and `skip` (requested
  but infeasible, e.g. the ISA would not allocate). `summarize.py` shouts only
  about `fail` and `skip`, and always reports the verified fraction — a warning
  that fires on every campaign is a warning nobody reads.
- `--verify` roughly doubles peak RSS: measured `peak_rss/n` went 9.9 -> 18.1
  B/token with it on, matching the predicted ~18n. It fits dolma (146 GB) and
  arxiv (509 GB) but not github (1068 GB > 1032 GB), so `bench.sbatch` drops it
  there and the record says `verify: skip`.
- **How an over-budget libsais run fails, now that no `RLIMIT_AS` is imposed.**
  Slurm is not enforcing a memory bound here: there is no `cgroup.conf`, so
  `--mem` is advisory and `--mem=0` asks for the whole node anyway. That leaves
  the kernel, and the outcome depends on `vm.overcommit_memory`. In heuristic
  mode (0), which is what this node runs, a single allocation larger than
  RAM+swap is refused outright -- and libsais's SA *is* a single allocation, 1.37
  TB for C4 -- so it returns -2 and the driver records `oom_host` with an exact
  `VmHWM`. That is the clean failure, achieved without a limit.
  The risk the limit used to cover is narrower than it looks: only in
  always-overcommit mode (1) would the allocation succeed and the process fault
  until the OOM killer fires, and SIGKILL would then skip the emission path, so
  the row becomes a gap rather than a result. If a node ever runs mode 1, prefer
  fixing the sysctl or reading `sacct` (a kernel OOM appears as `FAILED / 0:9`)
  over reintroducing a knob that is inert on every node we actually use.
  The limit was removed because it is one more unmeasured setting: it was never
  observed to fire, and the harness now sets nothing that is not an input.
- **`--n` counts symbols, not tokens.** For `byte_from_token16` that means bytes,
  i.e. twice the token count.
- **`--alphabet` is recorded, and `n_tokens` is the throughput denominator.**
  `byte_from_token16` has `n_symbols = 2 * n_tokens`; using `n_symbols` as the
  denominator would double libcubwt's apparent work. The `alphabet` field keeps a
  distorted run out of a native run's table column mechanically.
- Our own feature macros (`-DSTATS`, `-DPROBE`, `-DPROFILE`, `-DTESTS`, …) must
  not reach a baseline binary. `add_compile_definitions` at the root is a
  *directory* property and is inherited, so `benchmark/CMakeLists.txt` clears it;
  excluding `${COMPILE_FLAGS}` from a target is not sufficient.
- A dirty working tree gets a `-dirty` suffix on the recorded commit, so a record
  cannot quietly claim to come from a clean one.

## I/O and environment

- `pread`/`pwrite` transfer at most ~2 GB per call; both loops chunk. A `pread`
  returning 0 is a short file (stop), `< 0` is an error.
- **Cache state is not a controlled variable, and the harness no longer pretends
  it is.** A `warm` policy is unachievable at these scales — page cache is a small
  fraction of the working set and is evicted immediately. A `cold` policy was
  attempted via `posix_fadvise(DONTNEED)`, which needs no root but is frequently a
  **no-op on network filesystems** — and the corpora are on one. It was also
  honoured only by the baselines, never by our driver, so it made the two sides of
  a comparison read under different conditions. Removed. This costs nothing that
  matters: it moved `t_load`, and `t_load` is already unusable on this node, where
  back-to-back reads of the same file measured 0.70 and 1.64 GB/s.
- **Every knob the harness used to set has now been measured, and all of them
  were removed.** Four conditions on the benchmark node, libsais over a
  2e9-token prefix, medians of 3 replicates; our method over full dolmasample,
  medians of 2:

  | condition | libsais | ours |
  |---|---:|---:|
  | nothing set | 129.1 s | 92.9 s |
  | `+ OMP_DYNAMIC=false` | 127.9 s (−0.9%) | — |
  | `+ numactl --interleave=all` | 200.3 s (**+57%**) | 103.0 s (**+11%**) |
  | `+ OMP_PROC_BIND=close, OMP_PLACES=cores` | 204.9 s (+2.3%) | 391 s (**+235%**) |

  Three conclusions. `OMP_DYNAMIC=false` restated libgomp's default — the ICV
  dump with nothing exported reads `OMP_DYNAMIC = 'FALSE'` — so it never did
  anything; the driver's `omp_get_dynamic()` refusal is the real protection.
  The placement knobs cost libsais 2% but cost our method 3.35x, which is why a
  harness-wide `export` was the wrong shape for any of this. And `numactl
  --interleave=all` is harmful to *both*, contradicting the comment that had
  justified it — the harness was penalising the CPU baseline by more than half
  and inflating our own speedup by ~1.57x.

  **The open question is scale.** Those peaks (20 GB for the libsais arm,
  193 GB for ours) fit inside one socket's ~500 GiB, where first-touch is
  already local and interleaving only adds remote traffic. `github` is the first
  corpus whose libsais peak (639–793 GB) cannot fit one socket, so interleaving
  could plausibly win there. Until that is measured, no numactl anywhere.
- **`nproc` is not the core count under `--exclusive --ntasks=1`.** This node
  reports `SLURM_CPUS_ON_NODE=128` while `nproc` returns **65**, so a thread
  count derived from `nproc` runs the campaign on half the machine. Every
  measurement in the table above was taken at 65 threads for that reason. Derive
  it from `SLURM_CPUS_ON_NODE`.
- **lmod's `module` is a shell function that non-interactive shells never
  source**, and `module load` must run *inside* the job: a module-provided
  libgomp/libstdc++ lives outside the default loader path, so its absence is a
  runtime linker failure after a clean build. Bracket the bootstrap with
  `set +u` (module init scripts reference unset variables freely) and try, in
  order: `/usr/share/lmod/lmod/init/bash`, `/etc/profile.d/lmod.sh`,
  `/etc/profile.d/modules.sh`, `/usr/share/Modules/init/bash`.
- github's SA at 5 B/suffix is **298 GiB**, so `--sa-out` stays off in
  `bench.sbatch`'s campaigns, which time construction alone. The end-to-end runs
  (`infinigram/e2e.sbatch`, METHODS=libsais) turn it on, as ours and
  infini-gram's builds both write theirs.
