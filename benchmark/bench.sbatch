#!/bin/bash
#
# Benchmark executor: walks one campaign's run set.
#
# Submit through the API, which validates everything first and passes the run set
# and the results directory in:
#
#   benchmark/bench.sh --corpora a,b --tokenizer llama2 --methods libsais --reps 3
#
# Invoking this directly works but requires CORPORA/TOKENIZER/METHODS/OURS_TAG/REPS
# in the environment and a results directory as $1.
#
# One job rather than one per run: `ulimit -v` turns an over-budget allocation
# into a failed malloc in seconds with a clean record, so a failure neither takes
# the job down nor leaves anything to reconcile. Runs are idempotent -- one whose
# record already exists is skipped -- so a job that dies partway can simply be
# resubmitted with the same results directory.
#
# Site-specific settings are deliberately not baked in. Supply them either by
# sourcing an (untracked) benchmark/local.env before submitting, or on the
# command line:
#
#   SBATCH_PARTITION   partition to submit to        (or sbatch --partition=)
#   SBATCH_NODELIST    pin to one node, so timings are comparable across runs
#   CORPUS_ROOT        holds <corpus>/<corpus>-<tokenizer>.tokenized.bin
#   MODULES            module names to load in-job   (default none)
#   MEM_RESERVE_GIB    held back from the RLIMIT_AS  (default 8)
#   BUILD_IN_JOB       compile the baselines in-job  (default 1; see below)
#
# `sbatch` exports the submitting environment by default, so sourcing local.env
# in your shell covers both the SBATCH_* directives and the in-job variables.

#SBATCH --job-name=saca_baselines
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Override with SBATCH_TIMELIMIT.
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail        # Not -e: a failed run is data, and the loop must continue.

# Slurm runs a COPY of this script from a spool directory, so the script's own
# path says nothing about where the repo is -- ${BASH_SOURCE[0]} is
# /var/spool/slurmd/.../slurm_script. Use the submit directory, and fall back to
# $PWD when the script is run directly (dry tests).
REPO="${REPO:-${SLURM_SUBMIT_DIR:-${PWD}}}"

# Check it, rather than letting a wrong path surface later as a confusing cmake
# or "not built" error.
if [ ! -r "${REPO}/CMakeLists.txt" ] || [ ! -d "${REPO}/benchmark" ]; then
    echo "FATAL: '${REPO}' does not look like the repository root"
    echo "       (no CMakeLists.txt, or no benchmark/ directory)."
    echo "       Submit from the repo root, or: export REPO=/path/to/repo"
    exit 1
fi

BUILD="${BUILD:-${REPO}/build}"
# Required, not defaulted: the old `${REPO}/results/...` fallback made the results
# location a function of which directory sbatch was run from, and could write into
# the repo. bench.sh computes it under RESULTS_ROOT and passes it explicitly.
RESULTS="${1:?results directory is required -- submit through benchmark/bench.sh}"

# Untracked site config, if present. Sourced here too so the in-job variables
# work even when it was not sourced in the submitting shell. Bracketed with
# `set +u` for the same reason the lmod bootstrap below is: a config file should
# not have to be `set -u`-clean, and an unbound variable in one would otherwise
# kill the job outright.
if [ -r "${REPO}/benchmark/local.env" ]; then
    set +u
    # shellcheck source=/dev/null
    . "${REPO}/benchmark/local.env"
    set -u
fi

CORPUS_ROOT="${CORPUS_ROOT:-/scratch/${USER}/corpora}"
MODULES="${MODULES:-}"
MEM_RESERVE_GIB="${MEM_RESERVE_GIB:-8}"
SAIS="${BUILD}/benchmark/bench_libsais"
CUBWT="${BUILD}/benchmark/bench_libcubwt"

# The run set, from bench.sh (which has already validated all of it).
# Split on comma OR colon: bench.sh transports these colon-separated, because
# Slurm's --export parser would treat a comma inside a value as a new entry.
IFS=',:' read -r -a CORPORA     <<< "${CORPORA:?CORPORA is required}"
IFS=',:' read -r -a METHOD_LIST <<< "${METHODS-}"
TOKENIZER="${TOKENIZER:?TOKENIZER is required}"
OURS_TAG="${OURS_TAG-}"
REPS="${REPS:?REPS is required}"
# Empty means "all cores on this node". Overriding it is how the thread-choice
# control is run: same corpus, half the cores, into the same results directory --
# the run-id carries t<threads>, so the two land as separate rows.
THREADS="${THREADS:-}"
[ -n "${THREADS}" ] || THREADS="$(nproc)"

# `module` is a shell function login shells get from /etc/profile.d; an sbatch
# script never sources it. Init scripts reference unset vars, hence `set +u`.
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

# Compile inside the job, not on the login node: the CUDA sources are CPU-heavy
# and login nodes are usually CPU-time limited, while here we have every core of
# an exclusive allocation. Configure is the opposite -- it fetches both baselines,
# so it needs network, which compute nodes often lack, but it costs seconds. So
# the split is: configure wherever there is network, compile here.
#
# Building in-job also guarantees the commit stamped into the binary is the one
# that actually built it.
if [ "${BUILD_IN_JOB:-1}" = 1 ]; then
    if [ ! -f "${BUILD}/CMakeCache.txt" ]; then
        echo "No ${BUILD}/CMakeCache.txt -- configuring here. This needs network;"
        echo "if this node has none, run it on a login node first and resubmit:"
        echo "    cmake -S '${REPO}' -B '${BUILD}' -DBENCHMARK=ON -DCMAKE_BUILD_TYPE=Release"
        cmake -S "${REPO}" -B "${BUILD}" -DBENCHMARK=ON -DCMAKE_BUILD_TYPE=Release \
            || { echo "FATAL: configure failed."; exit 1; }
    elif ! grep -q '^BENCHMARK:BOOL=ON' "${BUILD}/CMakeCache.txt"; then
        echo "FATAL: ${BUILD} was configured without -DBENCHMARK=ON, so the baseline"
        echo "       targets do not exist. Reconfigure:"
        echo "    cmake -S '${REPO}' -B '${BUILD}' -DBENCHMARK=ON"
        exit 1
    fi

    echo "compiling baselines with -j$(nproc)"
    cmake --build "${BUILD}" --target bench_libsais bench_libcubwt -j"$(nproc)" \
        || { echo "FATAL: build failed."; exit 1; }
fi

# OMP_DYNAMIC=false is a correctness precondition, not tuning: with it on libsais
# silently runs serial. The driver also refuses, so this is belt and braces.
export OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=passive

# First-touch placement of libsais's 8n SA matters on a multi-socket node.
# Without this the CPU baseline is undertuned, so a missing numactl has to be
# loud rather than silent.
if command -v numactl >/dev/null; then
    NUMA=(numactl --interleave=all)
else
    NUMA=()
    echo "WARN: numactl not found -- libsais will run UNPINNED across sockets."
    echo "WARN: its numbers will understate the CPU baseline. Install numactl or load its module."
fi

# Physical memory less a reserve, as a RLIMIT_AS. Derived at runtime rather than
# hardcoded, and taken from MemTotal rather than Slurm's RealMemory: without a
# memory cgroup nothing holds a process to RealMemory, and RealMemory is
# conventionally set several percent below physical, so using it would hand the
# baseline less memory than the machine actually has.
#
# This is what makes an over-budget allocation fail cleanly -- in seconds, with
# an exact VmHWM -- instead of inviting the OOM killer (which loses the record)
# or swap thrash (a garbage timing after hours). Note it bounds *virtual* address
# space, so the reserve must also cover thread stacks and allocator mappings.
# CPU only: never for a CUDA process, whose unified VA reserves terabytes.
MEM_TOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
RLIMIT_KIB=$(( MEM_TOTAL_KIB - MEM_RESERVE_GIB * 1024 * 1024 ))


mkdir -p "${RESULTS}"
printf 'results  %s\nhost     %s   job %s\nmemtotal %s KiB   rlimit %s KiB (reserve %s GiB)\n' \
    "${RESULTS}" "$(hostname)" "${SLURM_JOB_ID:-none}" \
    "${MEM_TOTAL_KIB}" "${RLIMIT_KIB}" "${MEM_RESERVE_GIB}"

for b in "${SAIS}" "${CUBWT}"; do
    [ -x "${b}" ] || { echo "FATAL: ${b} not built; configure -DBENCHMARK=ON first."; exit 1; }
done

corpus_path() { echo "${CORPUS_ROOT}/$1/$1-${TOKENIZER}.tokenized.bin"; }

# Pre-flight: a bad path should not be discovered three hours in. bench.sh has
# already checked these, so a failure here means the filesystem changed under us.
for c in "${CORPORA[@]}"; do
    p="$(corpus_path "${c}")"
    if [ -r "${p}" ]
        then printf '  %-12s %14s bytes\n' "${c}" "$(stat -c %s "${p}")"
        else printf '  %-12s %14s  <- runs will record bad_input\n' "${c}" MISSING
    fi
done

# run <bin> <tag> <corpus> <alphabet> <n-symbols> <threads> <rep> [rlimit]
#
# Timing only -- nothing here passes --verify. Correctness is verify.sbatch's
# job, for two reasons: verification allocates n/8 + 8n on top of the run's own
# peak, so a verified run's `peak_rss` would be the verification peak rather than
# the construction peak, and peak RSS is half of the E2.1 table; and correctness
# is a property of a *configuration*, so it needs one run, not one per replicate.
run()
{
    local bin=$1 n=$5 threads=$6 rlimit=${8:-0}
    local id="$2_$3-${TOKENIZER}_$4_n$5_t$6_r$7"

    [ -s "${RESULTS}/${id}.json" ] && { echo "skip (already done) ${id}"; return; }

    local -a cmd=("${bin}" --input "$(corpus_path "$3")" --corpus "$3"
                  --tokenizer "${TOKENIZER}" --run-id "${id}"
                  --alphabet "$4" --n "${n}" --threads "${threads}"
                  --results-dir "${RESULTS}" --cold)

    # The baselines report their library's own version; ours reports the tag it
    # was built from.
    [ "$2" = gpu_saca ] && cmd+=(--method-version "${OURS_TAG}")

    echo "=== ${id}"
    local t0=${SECONDS}
    if [ "${rlimit}" -eq 1 ]
        then ( ulimit -v "${RLIMIT_KIB}"; exec ${NUMA[@]+"${NUMA[@]}"} "${cmd[@]}" )
        else ${NUMA[@]+"${NUMA[@]}"} "${cmd[@]}"
    fi
    echo "    wall $(( SECONDS - t0 ))s"
}

# Ordered smallest corpus first, so the cheapest answers arrive in minutes rather
# than after the largest corpus has finished.
for c in "${CORPORA[@]}"; do
    for m in ${METHOD_LIST[@]+"${METHOD_LIST[@]}"}; do
        for r in $(seq 1 "${REPS}"); do
            case "${m}" in
                libsais)
                    # uint16 tokens natively; RLIMIT_AS on, so an over-budget
                    # allocation fails cleanly instead of being OOM-killed.
                    run "${SAIS}" libsais16x64 "${c}" token16 0 "${THREADS}" "${r}" 1
                    ;;
                libcubwt)
                    # uint8-only, so a token corpus is read as raw bytes -- a
                    # reach-only comparison, which the `alphabet` field records.
                    # No RLIMIT_AS: a CUDA process reserves terabytes of VA.
                    run "${CUBWT}" libcubwt "${c}" byte_from_token16 0 0 "${r}"
                    ;;
            esac
        done
    done
done

# Our method, if a tag was given. Built from a git worktree checked out at that
# tag, so the binary provably corresponds to the version its records name, and so
# that re-running an older tag later gives the same numbers. The worktree lives
# outside the repo, like the results, and is reused across campaigns.
if [ -n "${OURS_TAG}" ]; then
    WORKTREE_ROOT="${WORKTREE_ROOT:-${HOME}/gpu-saca-worktrees}"
    WT="${WORKTREE_ROOT}/${OURS_TAG}"
    OURS_BUILD="${WT}/build"
    OURS="${OURS_BUILD}/src/bench_gpu_saca"

    if [ ! -d "${WT}" ]; then
        echo "creating worktree for ${OURS_TAG} at ${WT}"
        mkdir -p "${WORKTREE_ROOT}"
        git -C "${REPO}" worktree add --detach "${WT}" "refs/tags/${OURS_TAG}" \
            || { echo "FATAL: git worktree add failed for tag ${OURS_TAG}."; exit 1; }
    fi

    if [ "${BUILD_IN_JOB:-1}" = 1 ]; then
        # Same split as the baselines: configure needs network (concurrentqueue is
        # fetched), compiling needs cores. A tag predating the driver cannot build,
        # which is the intended signal that it is not benchmarkable under this policy.
        if [ ! -f "${OURS_BUILD}/CMakeCache.txt" ]; then
            echo "configuring ${WT} (needs network; run it on a login node if this has none)"
            cmake -S "${WT}" -B "${OURS_BUILD}" -DCMAKE_BUILD_TYPE=Release \
                || { echo "FATAL: configure failed for tag ${OURS_TAG}."; exit 1; }
        fi
        echo "compiling bench_gpu_saca @ ${OURS_TAG} with -j$(nproc)"
        cmake --build "${OURS_BUILD}" --target bench_gpu_saca -j"$(nproc)" \
            || { echo "FATAL: build failed for tag ${OURS_TAG} -- does it contain src/bench_gpu_saca.cu?"; exit 1; }
    fi

    if [ ! -x "${OURS}" ]; then
        echo "FATAL: ${OURS} not built."
        exit 1
    fi
    for c in "${CORPORA[@]}"; do
        for r in $(seq 1 "${REPS}"); do
            run "${OURS}" gpu_saca "${c}" token16 0 "${THREADS}" "${r}"
        done
    done
fi

echo
echo "done -> ${RESULTS}"
echo "summarize: python3 ${REPO}/benchmark/summarize.py ${RESULTS}"
