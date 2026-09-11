#!/bin/bash
#
# Baseline SACA: one job walks the whole run set.
#
#   . benchmark/local.env            # optional site config; see below
#   sbatch benchmark/bench.sbatch [results-dir]
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
#   CORPUS_ROOT        directory holding <corpus>.bin (default /scratch/$USER/corpora)
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
RESULTS="${1:-${REPO}/results/$(date -u +%Y%m%dT%H%M%SZ)}"

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

CORPORA=(dolmasample arxiv github c4)

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

# libsais peaks at 10n-12.4n bytes, so it completes the corpora that fit and
# fails cleanly on the one that does not; that failure is the result for its row.
# Uniform across corpora and methods: a runtime-tiered count needs a
# justification table, and any cell at 1 cannot state its own dispersion. At the
# measured 5-20 M tokens/s this costs 4.4-16.9 h, inside the 24 h wall even
# pessimistically, and it lets the CV be reported as evidence rather than assumed.
REPS=3

mkdir -p "${RESULTS}"
printf 'results  %s\nhost     %s   job %s\nmemtotal %s KiB   rlimit %s KiB (reserve %s GiB)\n' \
    "${RESULTS}" "$(hostname)" "${SLURM_JOB_ID:-none}" \
    "${MEM_TOTAL_KIB}" "${RLIMIT_KIB}" "${MEM_RESERVE_GIB}"

for b in "${SAIS}" "${CUBWT}"; do
    [ -x "${b}" ] || { echo "FATAL: ${b} not built; configure -DBENCHMARK=ON first."; exit 1; }
done

# Pre-flight: a bad path should not be discovered three hours in.
for c in "${CORPORA[@]}"; do
    p="${CORPUS_ROOT}/${c}.bin"
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
    local id="$2_$3_$4_n$5_t$6_r$7"

    [ -s "${RESULTS}/${id}.json" ] && { echo "skip (already done) ${id}"; return; }

    local -a cmd=("${bin}" --input "${CORPUS_ROOT}/$3.bin" --corpus "$3" --run-id "${id}"
                  --alphabet "$4" --n "${n}" --threads "${threads}"
                  --results-dir "${RESULTS}" --cold)

    echo "=== ${id}"
    local t0=${SECONDS}
    if [ "${rlimit}" -eq 1 ]
        then ( ulimit -v "${RLIMIT_KIB}"; exec ${NUMA[@]+"${NUMA[@]}"} "${cmd[@]}" )
        else ${NUMA[@]+"${NUMA[@]}"} "${cmd[@]}"
    fi
    echo "    wall $(( SECONDS - t0 ))s"
}

# 1. libsais on every corpus -- the headline table. Ascending, so the smallest
#    corpus's answer arrives in minutes rather than after the largest.
for c in "${CORPORA[@]}"; do
    for r in $(seq 1 "${REPS}"); do
        run "${SAIS}" libsais16x64 "${c}" token16 0 "$(nproc)" "${r}" 1
    done
done

# 2. Thread choice. Not a figure: it makes "libsais, all cores" a measured choice
#    rather than an assumption, and guards against a NUMA-hostile access pattern
#    making half the cores faster than all of them.
for r in $(seq 1 "${REPS}"); do
    run "${SAIS}" libsais16x64 "${CORPORA[0]}" token16 0 "$(( $(nproc) / 2 ))" "${r}" 1
done

# 3. libcubwt on every corpus -- the GPU memory-wall row. Expected to fail at
#    device allocation in seconds: it needs ~30 B per input symbol, and reading a
#    uint16 corpus as bytes doubles the symbol count.
for c in "${CORPORA[@]}"; do
    for r in $(seq 1 "${REPS}"); do
        run "${CUBWT}" libcubwt "${c}" byte_from_token16 0 0 "${r}"
    done
done

# 4. libcubwt prefix ladder on the smallest corpus, for the supplement. n is in
#    SYMBOLS, i.e. bytes, i.e. twice the token count. The last rung sits on the
#    predicted device wall, which is why it is included.
for n in 250000000 500000000 1000000000 2000000000 3000000000; do
    for r in $(seq 1 "${REPS}"); do
        run "${CUBWT}" libcubwt "${CORPORA[0]}" byte_from_token16 "${n}" 0 "${r}"
    done
done

echo
echo "done -> ${RESULTS}"
echo "summarize: python3 ${REPO}/benchmark/summarize.py ${RESULTS}"
