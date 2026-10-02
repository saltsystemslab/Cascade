#!/bin/bash
#
# Timed runs of query counting on the benchmark node: our engine against
# infini-gram's, over the workloads of a cross-check (see xcheck.sbatch) and
# over batches of growing size. Both indexes are first staged onto node-local
# NVMe. Every batch's counts must match infini-gram's, and our intervals must
# pass the certificate.
#
# Submit from the repo root:
#
#   . benchmark/local.env && sbatch benchmark/query/timed.sbatch INDEX IG_INDEX QRY_DIR [OUT_DIR]
#
#   INDEX     our query index (a directory with index.meta)
#   IG_INDEX  infini-gram's index of the same token array (2-byte symbols)
#   QRY_DIR   the workloads: a directory of .qry files, e.g. qry/ of a cross-check
#   OUT_DIR   default ${RESULTS_ROOT}/timed-<INDEX's name>-<UTC timestamp>
#
# Runs, most important first; TM_RUNS selects:
#   main      ours in TM_MAIN_MODES, by default the direct mode and two phases
#             (stride 32, 2-word keys), and infini-gram's `_find_thread` from a
#             pool of every core, its index placed as TM_IG_LOAD says; in
#             TM_REPLICATES alternating processes of each
#   batch     both, over present-pattern batches (m = 10) of TM_BATCHES queries,
#             ours in TM_BATCH_MODES; infini-gram also with a pool no larger
#             than the batch's chunks
#   ablation  ours over strides, key widths, and batch orders
#   threads   infini-gram over pool sizes, on the TM_SWEEP workloads
#   variants  on the TM_SWEEP workloads, infini-gram's own `find`, its
#             prefetching (on a prefix of 1e5 queries, the index mapped), and
#             its index in the other placement
#
# Results in OUT_DIR: env.txt (provenance), ours/ and ig/ (logs, and the counts
# of the first main replicate), compare.txt (count files: same or DIFFERENT),
# and from summarize.py, runs.tsv (a line per timed batch) and summary.md.
#
# Environment (all optional):
#   TM_STAGE        node-local directory to stage the indexes and the batches
#                   under, kept for later runs; default /tmp/${USER}/query-stage;
#                   `none` reads the indexes in place. STAGE_STREAMS sets the
#                   pieces copied at a time (see stage.sh)
#   TM_RUNS         default main,batch,ablation,threads,variants
#   TM_IG_LOAD      infini-gram's index in the main, batch, and thread runs: mmap,
#                   mapped as its engine defaults to (default), or ram, read into
#                   the heap
#   TM_TEXT_RES     tier of our text: device (default), pinned, or pageable
#   TM_SA_RES       tier of our suffix array: device (default), pinned, or
#                   pageable; with both in host memory and either pageable, the
#                   host searches our index
#   TM_NUMA         our process's NUMA placement: gpu, on the GPU's node
#                   (default); interleave, its memory interleaved over every
#                   node and its threads on every core, as the host's search
#                   wants; or none
#   TM_IG_NUMA      infini-gram's: none (default) or interleave
#   TM_MAIN_MODES   our modes of the main runs, as `query find` options; default
#                   `--stride 0,32 --key-words 2 --sort 0`
#   TM_BATCH_MODES  our modes of the batch runs; default `--stride 0,32 --key-words
#                   2 --sort 0,1`
#   TM_REPS         repetitions per batch; default 6 (infini-gram's driver drops the first)
#   TM_REPLICATES   processes per engine of the main runs; default 3
#   TM_BATCHES      default 1000,10000,100000,1000000,10000000,100000000
#   TM_STRIDES      strides of the ablation; default 8,16,32,64,128,256
#   TM_THREADS      pool sizes of the thread sweep; default 1,2,4,8,16,32,64,96,128,256
#   TM_SWEEP        workloads of the thread sweep and the variants; default
#                   present-m5,present-m10,present-m100,random-m5
#   TM_ARCH         CUDA architecture of the in-job build; default 120-real
#   TM_CMAKE_ARGS   extra configure arguments, e.g. FETCHCONTENT_SOURCE_DIR_* offline
#   INFINIGRAM_DIR  infini-gram checkout at v2.6.0; default ~/infini-gram

#SBATCH --job-name=qtimed
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -uo pipefail    # Not -e: a failed run is data, and the others must still run.

REPO="${REPO:-${SLURM_SUBMIT_DIR:-${PWD}}}"
if [ ! -r "${REPO}/benchmark/query/timed.sbatch" ]; then
    echo "FATAL: '${REPO}' does not look like the repository root; submit from it, or export REPO=/path/to/repo"
    exit 1
fi

USAGE="usage: timed.sbatch INDEX IG_INDEX QRY_DIR [OUT_DIR]"
IDX="${1:?${USAGE}}"
IG="${2:?${USAGE}}"
QRY="${3:?${USAGE}}"

if [ -r "${REPO}/benchmark/local.env" ]; then
    set +u
    # shellcheck source=/dev/null
    . "${REPO}/benchmark/local.env"
    set -u
fi

MODULES="${MODULES:-}"
if [ -n "${MODULES}" ]; then
    set +u
    for i in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh \
             /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        [ -r "${i}" ] && . "${i}" && break
    done
    # shellcheck disable=SC2086
    module load ${MODULES} || echo "WARN: module load failed; using the ambient environment"
    set -u
fi

OUT="${4:-${RESULTS_ROOT:-${HOME}/gpu-saca-results}/timed-$(basename "${IDX}")-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE="${TM_STAGE:-/tmp/${USER}/query-stage}"
RUNS=",${TM_RUNS:-main,batch,ablation,threads,variants},"
REPS="${TM_REPS:-6}"
REPL="${TM_REPLICATES:-3}"
BATCHES="${TM_BATCHES:-1000,10000,100000,1000000,10000000,100000000}"
SWEEP="${TM_SWEEP:-present-m5,present-m10,present-m100,random-m5}"
IG_LOAD="${TM_IG_LOAD:-mmap}"
TEXT_RES="${TM_TEXT_RES:-device}"
SA_RES="${TM_SA_RES:-device}"
OURS_NUMA="${TM_NUMA:-gpu}"
IG_NUMA="${TM_IG_NUMA:-none}"
read -r -a MAIN_MODES <<< "${TM_MAIN_MODES:---stride 0,32 --key-words 2 --sort 0}"
read -r -a BATCH_MODES <<< "${TM_BATCH_MODES:---stride 0,32 --key-words 2 --sort 0,1}"
if [ "${IG_LOAD}" != mmap ] && [ "${IG_LOAD}" != ram ]; then
    echo "FATAL: TM_IG_LOAD must be mmap or ram, not '${IG_LOAD}'"
    exit 1
fi
if [[ ! "${OURS_NUMA}" =~ ^(gpu|interleave|none)$ ]] || [[ ! "${IG_NUMA}" =~ ^(interleave|none)$ ]]; then
    echo "FATAL: TM_NUMA must be gpu, interleave, or none, and TM_IG_NUMA interleave or none"
    exit 1
fi
ARCH="${TM_ARCH:-120-real}"
T_ALL="$(nproc)"
B="${REPO}/build-timed-sm${ARCH%%-*}"
mkdir -p "${OUT}/ours" "${OUT}/ig"
cp "$0" "${OUT}/timed.sbatch" 2> /dev/null   # The script as submitted.

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
want() { [[ "${RUNS}" == *",$1,"* ]]; }

# Our process runs on the GPU's NUMA node, or over every node; infini-gram's pool spans the machine, its memory as placed or interleaved.
bdf=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1)
node=$(cat "/sys/bus/pci/devices/$(echo "0000:${bdf#*:}" | tr 'A-F' 'a-f')/numa_node" 2> /dev/null || echo -1)
NUMA=()
IG_NUMACTL=()
if command -v numactl > /dev/null; then
    if [ "${OURS_NUMA}" = gpu ] && [ "${node}" -ge 0 ] 2> /dev/null; then
        NUMA=(numactl --cpunodebind="${node}" --membind="${node}")
    elif [ "${OURS_NUMA}" = interleave ]; then
        NUMA=(numactl --interleave=all)
    fi
    [ "${IG_NUMA}" = interleave ] && IG_NUMACTL=(numactl --interleave=all)
fi

{
    echo "date=$(date -u +%FT%TZ)"
    echo "host=$(hostname)"
    echo "job=${SLURM_JOB_ID:-none}"
    echo "commit=$(git -C "${REPO}" rev-parse HEAD)$(git -C "${REPO}" diff --quiet HEAD || echo -dirty)"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "index=${IDX}"
    echo "ig_index=${IG}"
    echo "qry_dir=${QRY}"
    echo "reps=${REPS}"
    echo "replicates=${REPL}"
    echo "threads=${T_ALL}"
    echo "ig_load=${IG_LOAD}"
    echo "gpu_numa_node=${node}"
    echo "ours_numactl=${NUMA[*]:-none}"
    echo "ig_numactl=${IG_NUMACTL[*]:-none}"
    echo "ours_text_res=${TEXT_RES}"
    echo "ours_sa_res=${SA_RES}"
    echo "ours_main_modes=${MAIN_MODES[*]}"
    echo "ours_batch_modes=${BATCH_MODES[*]}"
    echo "kernel=$(uname -r)"
    echo "cpu=$(lscpu | sed -n 's/^Model name: *//p')"
    echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2> /dev/null || echo n/a)"
    echo "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2> /dev/null || echo n/a)"
    echo "memlock=$(ulimit -l)"
    echo "gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
    echo
    nvidia-smi --query-gpu=name,pci.bus_id,driver_version,memory.total,persistence_mode,clocks.max.sm,clocks.max.mem,power.limit --format=csv
    echo
    numactl -H
    echo
    free -g
    echo
    ls -l "${QRY}"
} > "${OUT}/env.txt" 2>&1
git -C "${REPO}" diff --quiet HEAD || git -C "${REPO}" diff HEAD > "${OUT}/uncommitted.diff"
stamp "start: $(hostname), $(sed -n 's/^gpu=//p' "${OUT}/env.txt"), commit $(sed -n 's/^commit=//p' "${OUT}/env.txt" | cut -c1-7), results ${OUT}"

stamp "build"
# shellcheck disable=SC2086
{ cmake -S "${REPO}" -B "${B}" -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH="${ARCH}" -DBENCHMARK=ON \
        -DFETCHCONTENT_SOURCE_DIR_INFINIGRAM="${INFINIGRAM_DIR:-${HOME}/infini-gram}" ${TM_CMAKE_ARGS:-} \
    && cmake --build "${B}" -j "${T_ALL}" --target query bench_infinigram_query; } > "${OUT}/build.log" 2>&1 \
    || { stamp "FATAL: build failed; see ${OUT}/build.log"; exit 1; }
stamp "build: $(grep -c -i warning "${OUT}/build.log") warnings"

# shellcheck source=stage.sh
. "${REPO}/benchmark/query/stage.sh"

SIDX="${IDX}"
SIG="${IG}"
BQ="${OUT}/batch-qry"
if [ "${STAGE}" != none ]; then
    stamp "stage: ${STAGE}"
    if stage_indexes "${IDX}" "${IG}" "${STAGE}"; then
        BQ="${STAGE}/batch-qry-$(basename "${IDX}")"
        stamp "stage: done; $(du -shL "${SIDX}" | cut -f1) ours, $(du -shL "${SIG}" | cut -f1) infini-gram"
    else
        SIDX="${IDX}"
        SIG="${IG}"
        stamp "WARN: staging failed; the indexes are read in place"
    fi
fi
{ echo "index_read=${SIDX}"; echo "ig_index_read=${SIG}"; echo "batch_qry=${BQ}"; } >> "${OUT}/env.txt"

# Runs `query find` over the comma-separated query files $2 into the log $1,
# with the options "${@:3}", every interval certified.
ours()
{
    local log="$1" files="$2"
    shift 2
    "${NUMA[@]}" "${B}/src/query" find --index "${SIDX}" --queries "${files}" --text-res "${TEXT_RES}" --sa-res "${SA_RES}" --reps "${REPS}" \
        --validate "$@" > "${log}" 2>&1
    local rc=$?
    stamp "ours ${log#"${OUT}"/}: rc=${rc}"
}

# Runs infini-gram's driver over the comma-separated query files $2 into the
# log $1, with the options "${@:3}": a process, loading the index once.
ig()
{
    local log="$1" files="$2"
    shift 2
    "${IG_NUMACTL[@]}" "${B}/benchmark/bench_infinigram_query" --index "${SIG}" --queries "${files}" --reps "${REPS}" "$@" > "${log}" 2>&1 \
        || stamp "infini-gram failed: ${log}"
}

files=$(ls "${QRY}"/*.qry | paste -sd, -)
sweep_files=$(for w in ${SWEEP//,/ }; do echo "${QRY}/${w}.qry"; done | paste -sd, -)

# infini-gram's index into the page cache, once for all of its runs: the warm-up, and then a single query.
if want main || want batch || want threads || want variants; then
    mkdir -p "${OUT}/ig/prewarm"
    ig "${OUT}/ig/prewarm/prewarm.log" "${files%%,*}" --mode direct --threads "${T_ALL}" --prewarm 1 --reps 1 --q 1
    stamp "infini-gram prewarm: $(sed -n 's/^prewarm_bytes=//p' "${OUT}/ig/prewarm/prewarm.log") bytes in $(sed -n 's/^t_prewarm=//p' "${OUT}/ig/prewarm/prewarm.log") s"
fi

if want main; then
    for ((r = 1; r <= REPL; ++r)); do
        d_ours="${OUT}/ours/main-r${r}"
        d_ig="${OUT}/ig/main-r${r}"
        mkdir -p "${d_ours}" "${d_ig}"
        cnt=()
        [ "${r}" -eq 1 ] && cnt=(--cnt-dir "${d_ours}")
        ours "${d_ours}/find.log" "${files}" "${MAIN_MODES[@]}" "${cnt[@]}"
        cnt=()
        [ "${r}" -eq 1 ] && cnt=(--cnt-dir "${d_ig}")
        ig "${d_ig}/${IG_LOAD}.log" "${files}" --mode direct --threads "${T_ALL}" --load "${IG_LOAD}" --prewarm 0 "${cnt[@]}"
        stamp "infini-gram main-r${r}: done"
    done
fi

if want batch; then
    stamp "gen: batches of ${BATCHES}"
    mkdir -p "${BQ}"
    for q in ${BATCHES//,/ }; do
        f="${BQ}/batch-q${q}.qry"
        [ -s "${f}" ] && continue   # Kept from an earlier run: the same seed, so the same queries.
        "${B}/src/query" gen --index "${SIDX}" --out "${f}.part" --kind substring --len 10 --q "${q}" --seed "${q}" > /dev/null \
            && mv "${f}.part" "${f}" || stamp "gen failed: q=${q}"
    done

    bfiles=$(for q in ${BATCHES//,/ }; do echo "${BQ}/batch-q${q}.qry"; done | paste -sd, -)
    ours "${OUT}/ours/batch.log" "${bfiles}" "${BATCH_MODES[@]}"
    mkdir -p "${OUT}/ig/batch"
    ig "${OUT}/ig/batch/t${T_ALL}.log" "${bfiles}" --mode direct --threads "${T_ALL}" --load "${IG_LOAD}" --prewarm 0
    for q in ${BATCHES//,/ }; do
        t=$(((q + 255) / 256))  # The driver's pool threads take 256 queries at a time.
        if [ "${t}" -lt "${T_ALL}" ]; then
            ig "${OUT}/ig/batch/batch-q${q}-t${t}.log" "${BQ}/batch-q${q}.qry" --mode direct --threads "${t}" --load "${IG_LOAD}" --prewarm 0
        fi
    done
    stamp "infini-gram batch: done"
fi

if want ablation; then
    # A process per stride: a table that does not fit the device fails its stride only.
    strides="${TM_STRIDES:-8,16,32,64,128,256}"
    for s in ${strides//,/ }; do
        ours "${OUT}/ours/ablation-s${s}.log" "${files}" --stride "${s}" --key-words 1,2,4 --sort 0,1
    done
fi

if want threads; then
    mkdir -p "${OUT}/ig/threads"
    threads="${TM_THREADS:-1,2,4,8,16,32,64,96,128,256}"
    for t in ${threads//,/ }; do
        ig "${OUT}/ig/threads/t${t}.log" "${sweep_files}" --mode direct --threads "${t}" --load "${IG_LOAD}" --prewarm 0 --reps 3
    done
    stamp "infini-gram threads: done"
fi

if want variants; then
    mkdir -p "${OUT}/ig/find" "${OUT}/ig/prefetch" "${OUT}/ig/placement"
    other=$([ "${IG_LOAD}" = ram ] && echo mmap || echo ram)
    ig "${OUT}/ig/find/${IG_LOAD}.log" "${sweep_files}" --mode find --threads "${T_ALL}" --load "${IG_LOAD}" --prewarm 0
    # Prefetching issues about ten `madvise` calls per search step: on a prefix, 3 times.
    ig "${OUT}/ig/prefetch/mmap.log" "${sweep_files}" --mode direct --threads "${T_ALL}" --load mmap --prefetch 1 --prewarm 0 \
        --q 100000 --reps 3
    ig "${OUT}/ig/placement/${other}.log" "${sweep_files}" --mode direct --threads "${T_ALL}" --load "${other}" --prewarm 0
    stamp "infini-gram variants: done"
fi

# The first main replicates' count files must be byte-identical; summarize.py
# checks every batch's counts hash.
stamp "compare"
for c in "${OUT}"/ours/main-r1/*.cnt; do
    [ -e "${c}" ] || continue
    s=$(basename "${c}" .cnt)
    if cmp -s "${c}" "${OUT}/ig/main-r1/${s%@*}.cnt"; then echo "${s} same"; else echo "${s} DIFFERENT"; fi
done > "${OUT}/compare.txt"
stamp "compare: $(grep -c ' same$' "${OUT}/compare.txt") same, $(grep -c ' DIFFERENT$' "${OUT}/compare.txt") different"

stamp "summarize"
python3 "${REPO}/benchmark/query/summarize.py" "${OUT}" > "${OUT}/summarize.log" 2>&1 \
    && cat "${OUT}/summary.md" || stamp "WARN: summarize.py failed; see ${OUT}/summarize.log"
stamp "done"
