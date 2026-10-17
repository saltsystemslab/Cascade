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
#   phases    ours in TM_MAIN_MODES over the main workloads, timing the phases
#             of every call (`query find --phase-times 1`): a process, per cache
#             state of TM_PHASE_CACHES with our suffix array mapped
#   batch     both, over present-pattern batches (m = 10) of TM_BATCHES queries,
#             ours in TM_BATCH_MODES; infini-gram also with a pool no larger
#             than the batch's chunks
#   ablation  ours over strides, key widths, and batch orders
#   threads   infini-gram over pool sizes, on the TM_SWEEP workloads
#   variants  on the TM_SWEEP workloads, infini-gram's own `find`, its
#             prefetching (on a prefix of 1e5 queries, the index mapped), and
#             its index in the other placement
#   host      if the host searches our index, on the TM_SWEEP workloads in the
#             direct mode and in two phases, over the searches each thread
#             interleaves, TM_LANES, and over the threads, TM_HOST_THREADS
#   prefetch  with our suffix array mapped: ours not asking for its pages ahead
#   stripe    with our suffix array mapped and TM_STRIPE set: ours over a copy
#             of the suffix array split over the TM_STRIPE directories, asking
#             for its pages ahead and not
#   direct    with our suffix array mapped: ours in the direct mode, on the
#             first TM_DIRECT_Q queries of every workload
#
# With our suffix array mapped (TM_SA_RES=mapped), the runs time cold caches,
# every repetition first dropping our suffix array's pages and infini-gram's
# table's from the page cache, and the main runs hot ones too (TM_CACHES).
# infini-gram's index is mapped with its text locked in memory, as ours is in
# memory; its table, like our suffix array, is read from its drive. The two are
# staged on the same drive, and the text apart (TM_TEXT_STAGE).
#
# Results in OUT_DIR: env.txt (provenance), ours/ and ig/ (logs, and the counts
# of the first main replicate), compare.txt (count files: same or DIFFERENT),
# and from summarize.py, runs.tsv (a line per timed batch) and summary.md. Our
# logs report our processes' memory too.
#
# Environment (all optional):
#   TM_STAGE        node-local directory to stage the indexes and the batches
#                   under; default /tmp/${USER}/query-stage; `none` reads the
#                   indexes in place. STAGE_STREAMS sets the pieces copied at a
#                   time (see stage.sh). With our suffix array mapped, a failed
#                   staging ends the job
#   TM_KEEP_STAGE   1 keeps the staged copies for later runs; by default the job
#                   deletes them when it ends, as a site's epilog may anyway for
#                   /tmp after the user's last job on the node
#   TM_TEXT_STAGE   node-local directory to stage the text in, apart from the
#                   rest: our index's, and infini-gram's token array, linked to
#                   it if the same file; default with our index
#   TM_STRIPE       colon-separated node-local directories to split a copy of our
#                   suffix array over, for the stripe run; default none
#   TM_RUNS         default main,phases,batch,ablation,threads,variants,host, and
#                   with our suffix array mapped prefetch,stripe,direct too; host
#                   runs only if the host searches our index
#   TM_CACHES       cache states of the main runs with our suffix array mapped:
#                   cold, hot, or both (default)
#   TM_PHASE_CACHES cache states of the phases run with our suffix array mapped;
#                   default the first of TM_CACHES
#   TM_IG_LOCK_TEXT 1 locks infini-gram's text in memory; default 1 iff our
#                   suffix array is mapped
#   TM_DIRECT_Q     queries per workload of the direct run; default 100000
#   TM_IG_LOAD      infini-gram's index in the main, batch, and thread runs: mmap,
#                   mapped as its engine defaults to (default), or ram, read into
#                   the heap
#   TM_TEXT_RES     tier of our text: device (default), pinned, or pageable
#   TM_SA_RES       tier of our suffix array: device (default), pinned,
#                   pageable, or mapped; with both in host memory and either
#                   pageable or mapped, the host searches our index
#   TM_NUMA         our process's NUMA placement: gpu, on the GPU's node
#                   (default); interleave, its memory interleaved over every
#                   node and its threads on every core, as the host's search
#                   wants; or none
#   TM_IG_NUMA      infini-gram's: none (default) or interleave
#   TM_MAIN_MODES   our modes of the main runs, as `query find` options; default
#                   `--stride 0,32 --key-words 2 --sort 0`, or with our suffix
#                   array mapped `--stride 128 --key-words 4 --sort 0`
#   TM_BATCH_MODES  our modes of the batch runs; default `--stride 0,32 --key-words
#                   2 --sort 0,1`, or the main modes with our suffix array mapped
#   TM_REPS         repetitions per batch; default 6, or 3 with our suffix array
#                   mapped (infini-gram's driver drops the first)
#   TM_REPLICATES   processes per engine of the main runs; default 3
#   TM_BATCHES      default 1000,10000,100000,1000000,10000000,100000000, or up to
#                   1000000 with our suffix array mapped
#   TM_STRIDES      strides of the ablation; default 8,16,32,64,128,256, or 64,128
#                   with our suffix array mapped
#   TM_KEY_WORDS    key widths of the ablation; default 1,2,4, or 2,4 with our
#                   suffix array mapped
#   TM_SORTS        batch orders of the ablation; default 0,1, or 0 with our
#                   suffix array mapped
#   TM_THREADS      pool sizes of the thread sweep; default 1,2,4,8,16,32,64,96,128,256,
#                   or 64,128,256,512,1024 with our suffix array mapped
#   TM_SWEEP        workloads of the thread sweep, the variants, and our host
#                   search's sweeps; default present-m5,present-m10,present-m100,random-m5
#   TM_LANES        our host search's lanes per thread; default 1,4,8,16,32,64, or
#                   8,32 with our suffix array mapped
#   TM_HOST_THREADS our host search's threads; default 8,16,32,64,96,128, or 32,128
#                   with our suffix array mapped
#   TM_ARCH         CUDA architecture of the in-job build; default 120-real
#   TM_CMAKE_ARGS   extra configure arguments, e.g. FETCHCONTENT_SOURCE_DIR_* offline
#   INFINIGRAM_DIR  infini-gram checkout at v2.6.0; default ~/infini-gram

#SBATCH --job-name=qtimed
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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
TEXT_STAGE="${TM_TEXT_STAGE:-}"
STRIPE="${TM_STRIPE:-}"
KEEP_STAGE="${TM_KEEP_STAGE:-0}"
TEXT_RES="${TM_TEXT_RES:-device}"
SA_RES="${TM_SA_RES:-device}"
MAPPED=0    # Whether our suffix array is mapped from its drive: the runs time cold caches.
[ "${SA_RES}" = mapped ] && MAPPED=1
p() { [ "${MAPPED}" -eq 1 ] && echo "$2" || echo "$1"; }    # The default $1, or $2 with our suffix array mapped.
RUNS=",${TM_RUNS:-main,phases,batch,ablation,threads,variants,host$(p "" ,prefetch,stripe,direct)},"
REPS="${TM_REPS:-$(p 6 3)}"
REPL="${TM_REPLICATES:-3}"
BATCHES="${TM_BATCHES:-$(p 1000,10000,100000,1000000,10000000,100000000 1000,10000,100000,1000000)}"
SWEEP="${TM_SWEEP:-present-m5,present-m10,present-m100,random-m5}"
CACHES="${TM_CACHES:-$(p hot cold,hot)}"
PHASE_CACHES="${TM_PHASE_CACHES:-${CACHES%%,*}}"
IG_LOAD="${TM_IG_LOAD:-mmap}"
IG_LOCK_TEXT="${TM_IG_LOCK_TEXT:-${MAPPED}}"
DIRECT_Q="${TM_DIRECT_Q:-100000}"
OURS_NUMA="${TM_NUMA:-gpu}"
IG_NUMA="${TM_IG_NUMA:-none}"
HOST=0  # Whether the host searches our index: both arrays in host memory, either pageable or mapped.
if [ "${TEXT_RES}" != device ] && [ "${SA_RES}" != device ] && [ "${TEXT_RES}${SA_RES}" != pinnedpinned ]; then
    HOST=1
fi
read -r -a MAIN_MODES <<< "${TM_MAIN_MODES:-$(p "--stride 0,32 --key-words 2 --sort 0" "--stride 128 --key-words 4 --sort 0")}"
read -r -a BATCH_MODES <<< "${TM_BATCH_MODES:-$(p "--stride 0,32 --key-words 2 --sort 0,1" "${MAIN_MODES[*]}")}"
if [ "${IG_LOAD}" != mmap ] && [ "${IG_LOAD}" != ram ]; then
    echo "FATAL: TM_IG_LOAD must be mmap or ram, not '${IG_LOAD}'"
    exit 1
fi
if [ "${MAPPED}" -eq 1 ] && [ "${IG_LOAD}" != mmap ]; then
    echo "FATAL: with our suffix array mapped, infini-gram's table is mapped too: TM_IG_LOAD=mmap"
    exit 1
fi
if [[ ! ",${CACHES}," =~ ^(,(cold|hot))+,$ ]] || [[ ! ",${PHASE_CACHES}," =~ ^(,(cold|hot))+,$ ]]; then
    echo "FATAL: TM_CACHES and TM_PHASE_CACHES must list cold and hot, not '${CACHES}' and '${PHASE_CACHES}'"
    exit 1
fi
# The options of a cold repetition: ours drops its suffix array's pages, infini-gram its table's.
COLD_OURS=(--sa-cold 1)
COLD_IG=(--cold 1)
# infini-gram's placement beside a mapped suffix array: its text locked in memory, its table read from the drive as needed.
IG_PLACE=(--lock-text "${IG_LOCK_TEXT}")
[ "${MAPPED}" -eq 1 ] && IG_PLACE+=(--prewarm 0)
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
    echo "ours_host_search=${HOST}"
    echo "stage=${STAGE}"
    echo "text_stage=${TEXT_STAGE:-none}"
    echo "stripe=${STRIPE:-none}"
    echo "keep_stage=${KEEP_STAGE}"
    echo "caches=${CACHES}"
    echo "phase_caches=${PHASE_CACHES}"
    echo "ig_lock_text=${IG_LOCK_TEXT}"
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
SIDX_STRIPE=""  # Our index over a suffix array split over the TM_STRIPE directories, if staged.
BQ="${OUT}/batch-qry"
if [ "${STAGE}" != none ]; then
    name=$(basename "${IDX}")
    text_dir=""
    [ -n "${TEXT_STAGE}" ] && text_dir="${TEXT_STAGE}/text-${name}"
    stripe_dirs=""
    if [ -n "${STRIPE}" ] && [ "${MAPPED}" -eq 1 ] && want stripe; then
        stripe_dirs=$(for d in ${STRIPE//:/ }; do echo "${d}/sa-${name}"; done | paste -sd: -)
    fi

    # The copies go when the job does, however it ends, unless kept; and then the staging directories, if left empty.
    staged=("${STAGE}/ours-${name}" "${STAGE}/ig-$(basename "${IG}")" "${STAGE}/batch-qry-${name}" "${STAGE}/stripe" "${text_dir}" ${stripe_dirs//:/ })
    parents=("${TEXT_STAGE}" ${STRIPE//:/ })
    if [ "${KEEP_STAGE}" != 1 ]; then
        trap 'unstage "${STAGE}" "${staged[@]}"; for d in "${parents[@]}"; do [ -n "${d}" ] && rmdir -- "${d}" 2> /dev/null; done' EXIT
        trap 'exit 143' TERM
    fi

    stamp "stage: ${STAGE}${text_dir:+, the text into ${text_dir}}"
    if stage_indexes "${IDX}" "${IG}" "${STAGE}" "${text_dir}"; then
        BQ="${STAGE}/batch-qry-${name}"
        stamp "stage: done; $(staged_sizes "${SIDX}" "${SIG}")"
        if [ -n "${stripe_dirs}" ]; then
            # A copy of the staged suffix array, split over the drives; the text is the staged one.
            staged_idx="${SIDX}"
            stamp "stage: our suffix array split over ${stripe_dirs}"
            if stage_indexes "${staged_idx}" "" "${STAGE}/stripe" "$(dirname "$(sed -n 's/^text=//p' "${staged_idx}/index.meta")")" "${stripe_dirs}"; then
                SIDX_STRIPE="${SIDX}"
            else
                stamp "WARN: splitting our suffix array failed; no stripe run"
            fi
            SIDX="${staged_idx}"
        fi
    else
        # A mapped suffix array read in place would time the shared file system, not a drive.
        [ "${MAPPED}" -eq 1 ] && { stamp "FATAL: staging failed"; exit 1; }
        SIDX="${IDX}"
        SIG="${IG}"
        stamp "WARN: staging failed; the indexes are read in place"
    fi
fi
{ echo "index_read=${SIDX}"; echo "ig_index_read=${SIG}"; echo "index_stripe=${SIDX_STRIPE:-none}"; echo "batch_qry=${BQ}"; } >> "${OUT}/env.txt"

# Runs `query find` over the comma-separated query files $2 into the log $1,
# with the options "${@:3}", every interval certified; over our index at
# OURS_IDX if set, else the staged one.
ours()
{
    local log="$1" files="$2"
    shift 2
    "${NUMA[@]}" "${B}/src/query" find --index "${OURS_IDX:-${SIDX}}" --queries "${files}" --text-res "${TEXT_RES}" --sa-res "${SA_RES}" \
        --reps "${REPS}" --validate "$@" > "${log}" 2>&1
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

# infini-gram's index into the page cache, once for all of its runs: the warm-up, and then a single query. Not beside a mapped
# suffix array: infini-gram's table is read from its drive too, and its text locked by every process.
if [ "${MAPPED}" -eq 0 ] && { want main || want batch || want threads || want variants; }; then
    mkdir -p "${OUT}/ig/prewarm"
    ig "${OUT}/ig/prewarm/prewarm.log" "${files%%,*}" --mode direct --threads "${T_ALL}" --prewarm 1 --reps 1 --q 1
    stamp "infini-gram prewarm: $(sed -n 's/^prewarm_bytes=//p' "${OUT}/ig/prewarm/prewarm.log") bytes in $(sed -n 's/^t_prewarm=//p' "${OUT}/ig/prewarm/prewarm.log") s"
fi

# The main runs over the cache states, with our suffix array mapped: main-cold-r<k> and main-hot-r<k>; else main-r<k>.
main_dir()
{
    [ "${MAPPED}" -eq 1 ] && echo "main-$1-r$2" || echo "main-r$2"
}

if want main; then
    for cache in ${CACHES//,/ }; do
        cold_ours=()
        cold_ig=()
        [ "${cache}" = cold ] && cold_ours=("${COLD_OURS[@]}") && cold_ig=("${COLD_IG[@]}")
        for ((r = 1; r <= REPL; ++r)); do
            d_ours="${OUT}/ours/$(main_dir "${cache}" "${r}")"
            d_ig="${OUT}/ig/$(main_dir "${cache}" "${r}")"
            mkdir -p "${d_ours}" "${d_ig}"
            cnt=()
            [ "${r}" -eq 1 ] && [ "${cache}" = "${CACHES%%,*}" ] && cnt=(--cnt-dir "${d_ours}")
            ours "${d_ours}/find.log" "${files}" "${MAIN_MODES[@]}" "${cold_ours[@]}" "${cnt[@]}"
            cnt=()
            [ "${r}" -eq 1 ] && [ "${cache}" = "${CACHES%%,*}" ] && cnt=(--cnt-dir "${d_ig}")
            ig "${d_ig}/${IG_LOAD}.log" "${files}" --mode direct --threads "${T_ALL}" --load "${IG_LOAD}" --prewarm 0 "${IG_PLACE[@]}" \
                "${cold_ig[@]}" "${cnt[@]}"
            stamp "infini-gram $(main_dir "${cache}" "${r}"): done"
        done
    done
fi

# Where our time goes: a process timing every call's phases, in the main modes; per cache state with our suffix array mapped, as
# phases-cold and phases-hot.
if want phases; then
    for cache in $(p hot "${PHASE_CACHES//,/ }"); do
        cold_ours=()
        [ "${cache}" = cold ] && cold_ours=("${COLD_OURS[@]}")
        ours "${OUT}/ours/phases$(p "" "-${cache}").log" "${files}" "${MAIN_MODES[@]}" --phase-times 1 "${cold_ours[@]}"
    done
fi

# The other runs time cold caches with our suffix array mapped.
COLD_RUN_OURS=()
COLD_RUN_IG=()
if [ "${MAPPED}" -eq 1 ]; then
    COLD_RUN_OURS=("${COLD_OURS[@]}")
    COLD_RUN_IG=("${COLD_IG[@]}")
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
    ours "${OUT}/ours/batch.log" "${bfiles}" "${BATCH_MODES[@]}" "${COLD_RUN_OURS[@]}"
    mkdir -p "${OUT}/ig/batch"
    ig "${OUT}/ig/batch/t${T_ALL}.log" "${bfiles}" --mode direct --threads "${T_ALL}" --load "${IG_LOAD}" --prewarm 0 "${IG_PLACE[@]}" \
        "${COLD_RUN_IG[@]}"
    for q in ${BATCHES//,/ }; do
        t=$(((q + 255) / 256))  # The driver's pool threads take 256 queries at a time.
        if [ "${t}" -lt "${T_ALL}" ]; then
            ig "${OUT}/ig/batch/batch-q${q}-t${t}.log" "${BQ}/batch-q${q}.qry" --mode direct --threads "${t}" --load "${IG_LOAD}" --prewarm 0 \
                "${IG_PLACE[@]}" "${COLD_RUN_IG[@]}"
        fi
    done
    stamp "infini-gram batch: done"
fi

if want ablation; then
    # A process per stride: a table that does not fit the device fails its stride only.
    strides="${TM_STRIDES:-$(p 8,16,32,64,128,256 64,128)}"
    for s in ${strides//,/ }; do
        ours "${OUT}/ours/ablation-s${s}.log" "${files}" --stride "${s}" --key-words "${TM_KEY_WORDS:-$(p 1,2,4 2,4)}" --sort "${TM_SORTS:-$(p 0,1 0)}" \
            "${COLD_RUN_OURS[@]}"
    done
fi

if want threads; then
    mkdir -p "${OUT}/ig/threads"
    threads="${TM_THREADS:-$(p 1,2,4,8,16,32,64,96,128,256 64,128,256,512,1024)}"
    for t in ${threads//,/ }; do
        ig "${OUT}/ig/threads/t${t}.log" "${sweep_files}" --mode direct --threads "${t}" --load "${IG_LOAD}" --prewarm 0 --reps 3 "${IG_PLACE[@]}" \
            "${COLD_RUN_IG[@]}"
    done
    stamp "infini-gram threads: done"
fi

if want variants; then
    mkdir -p "${OUT}/ig/find" "${OUT}/ig/prefetch" "${OUT}/ig/placement"
    other=$([ "${IG_LOAD}" = ram ] && echo mmap || echo ram)
    ig "${OUT}/ig/find/${IG_LOAD}.log" "${sweep_files}" --mode find --threads "${T_ALL}" --load "${IG_LOAD}" --prewarm 0 "${IG_PLACE[@]}" \
        "${COLD_RUN_IG[@]}"
    # Prefetching issues about ten `madvise` calls per search step: on a prefix, 3 times.
    ig "${OUT}/ig/prefetch/mmap.log" "${sweep_files}" --mode direct --threads "${T_ALL}" --load mmap --prefetch 1 --prewarm 0 \
        --q 100000 --reps 3 "${IG_PLACE[@]}" "${COLD_RUN_IG[@]}"
    # Not in the heap if its table is read from its drive: it does not fit the memory.
    [ "${MAPPED}" -eq 0 ] && ig "${OUT}/ig/placement/${other}.log" "${sweep_files}" --mode direct --threads "${T_ALL}" --load "${other}" --prewarm 0
    stamp "infini-gram variants: done"
fi

# Our host search's lanes and threads, a process each, if the host searches; infini-gram's pool sizes are its threads run's.
if want host && [ "${HOST}" -eq 1 ]; then
    read -r -a host_modes <<< "$(p "--stride 0,32 --key-words 2 --sort 0" "${MAIN_MODES[*]}")"
    lanes="${TM_LANES:-$(p 1,4,8,16,32,64 8,32)}"
    for l in ${lanes//,/ }; do
        ours "${OUT}/ours/host-l${l}.log" "${sweep_files}" "${host_modes[@]}" --host-lanes "${l}" "${COLD_RUN_OURS[@]}"
    done
    host_threads="${TM_HOST_THREADS:-$(p 8,16,32,64,96,128 32,128)}"
    for t in ${host_threads//,/ }; do
        ours "${OUT}/ours/host-t${t}.log" "${sweep_files}" "${host_modes[@]}" --host-threads "${t}" "${COLD_RUN_OURS[@]}"
    done
fi

# Our mapped suffix array's ablations, cold: not asking for its pages ahead; split over the drives, asking or not; the direct mode.
if [ "${MAPPED}" -eq 1 ]; then
    want prefetch && ours "${OUT}/ours/prefetch-p0.log" "${files}" "${MAIN_MODES[@]}" --host-prefetch 0 "${COLD_OURS[@]}"
    if want stripe && [ -n "${SIDX_STRIPE}" ]; then
        for pf in 1 0; do
            OURS_IDX="${SIDX_STRIPE}" ours "${OUT}/ours/stripe-p${pf}.log" "${files}" "${MAIN_MODES[@]}" --host-prefetch "${pf}" "${COLD_OURS[@]}"
        done
    fi
    want direct && ours "${OUT}/ours/direct.log" "${files}" --stride 0 --q "${DIRECT_Q}" "${COLD_OURS[@]}"
fi

# The first main replicates' count files must be byte-identical; summarize.py
# checks every batch's counts hash.
stamp "compare"
first=$(main_dir "${CACHES%%,*}" 1)
for c in "${OUT}/ours/${first}"/*.cnt; do
    [ -e "${c}" ] || continue
    s=$(basename "${c}" .cnt)
    if cmp -s "${c}" "${OUT}/ig/${first}/${s%@*}.cnt"; then echo "${s} same"; else echo "${s} DIFFERENT"; fi
done > "${OUT}/compare.txt"
stamp "compare: $(grep -c ' same$' "${OUT}/compare.txt") same, $(grep -c ' DIFFERENT$' "${OUT}/compare.txt") different"

stamp "summarize"
python3 "${REPO}/benchmark/query/summarize.py" "${OUT}" > "${OUT}/summarize.log" 2>&1 \
    && cat "${OUT}/summary.md" || stamp "WARN: summarize.py failed; see ${OUT}/summarize.log"
stamp "done"
