#!/bin/bash
#
# Correctness runner. Separate from bench.sbatch on purpose:
#
#   . benchmark/local.env
#   sbatch benchmark/verify.sbatch [results-dir]
#
#  - Correctness is a property of a *configuration*, not of a replicate, so each
#    configuration is verified once. bench.sbatch can then re-run a method's
#    timings as often as the method evolves without re-paying for verification.
#  - Verification allocates a bitset (n/8) and an inverse SA (8n) on top of the
#    run's own peak, so a verified run's `peak_rss` is the verification peak, not
#    the construction peak. Keeping it out of the timing runs is what makes the
#    E2.1 memory column mean one thing.
#
# Records land in their own directory, so they never enter a timing summary.
#
# Scales are deliberately modest: libsais and libcubwt are fixed third-party
# code, so verifying them at full corpus scale costs hours and proves nothing
# that a small run does not. What is actually being checked here is (a) that each
# alphabet path through the harness is wired up correctly, and (b) the shared
# suffix-order convention that every cross-method comparison depends on. Our own
# method, once it joins, is the one that earns verification at scale.

#SBATCH --job-name=saca_verify
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-${PWD}}}"
if [ ! -r "${REPO}/CMakeLists.txt" ] || [ ! -d "${REPO}/benchmark" ]; then
    echo "FATAL: '${REPO}' does not look like the repository root"
    echo "       Submit from the repo root, or: export REPO=/path/to/repo"
    exit 1
fi

BUILD="${BUILD:-${REPO}/build}"
RESULTS="${1:-${REPO}/results/verify-$(date -u +%Y%m%dT%H%M%SZ)}"

if [ -r "${REPO}/benchmark/local.env" ]; then
    set +u
    # shellcheck source=/dev/null
    . "${REPO}/benchmark/local.env"
    set -u
fi

CORPUS_ROOT="${CORPUS_ROOT:-/scratch/${USER:-$(id -un)}/corpora}"
MODULES="${MODULES-}"
MEM_RESERVE_GIB="${MEM_RESERVE_GIB:-8}"
SAIS="${BUILD}/benchmark/bench_libsais"
CUBWT="${BUILD}/benchmark/bench_libcubwt"

# Symbols to verify per configuration. Small by design (see the header); raise
# only when verifying something whose correctness is scale-dependent.
VERIFY_N="${VERIFY_N:-100000000}"
# Scratch for the two SA dumps the cross-check compares. 8 B/symbol each.
SA_DIR="${SA_DIR:-${RESULTS}/sa}"

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

if [ "${BUILD_IN_JOB:-1}" = 1 ]; then
    if [ ! -f "${BUILD}/CMakeCache.txt" ]; then
        echo "No ${BUILD}/CMakeCache.txt -- configuring here. This needs network;"
        echo "if this node has none, configure on a login node first and resubmit."
        cmake -S "${REPO}" -B "${BUILD}" -DBENCHMARK=ON -DCMAKE_BUILD_TYPE=Release \
            || { echo "FATAL: configure failed."; exit 1; }
    elif ! grep -q '^BENCHMARK:BOOL=ON' "${BUILD}/CMakeCache.txt"; then
        echo "FATAL: ${BUILD} was configured without -DBENCHMARK=ON."
        exit 1
    fi
    cmake --build "${BUILD}" --target bench_libsais bench_libcubwt -j"$(nproc)" \
        || { echo "FATAL: build failed."; exit 1; }
fi

export OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=passive

if command -v numactl >/dev/null
    then NUMA=(numactl --interleave=all)
    else NUMA=(); echo "WARN: numactl not found."
fi

MEM_TOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
RLIMIT_KIB=$(( MEM_TOTAL_KIB - MEM_RESERVE_GIB * 1024 * 1024 ))

mkdir -p "${RESULTS}" "${SA_DIR}"
printf 'results  %s\nhost     %s   job %s\nverify n %s symbols\n' \
    "${RESULTS}" "$(hostname)" "${SLURM_JOB_ID:-none}" "${VERIFY_N}"

for b in "${SAIS}" "${CUBWT}"; do
    [ -x "${b}" ] || { echo "FATAL: ${b} not built."; exit 1; }
done

CORPUS="${VERIFY_CORPUS:-dolmasample}"
IN="${CORPUS_ROOT}/${CORPUS}.bin"
[ -r "${IN}" ] || { echo "FATAL: ${IN} is unreadable."; exit 1; }

fail=0

# verify <bin> <tag> <alphabet> <threads> [sa-out] [rlimit]
verify()
{
    local bin=$1 tag=$2 alphabet=$3 threads=$4 sa_out=${5:-} rlimit=${6:-0}
    local id="${tag}_${CORPUS}_${alphabet}_n${VERIFY_N}_t${threads}_verify"

    local -a cmd=("${bin}" --input "${IN}" --corpus "${CORPUS}" --run-id "${id}"
                  --alphabet "${alphabet}" --n "${VERIFY_N}" --threads "${threads}"
                  --results-dir "${RESULTS}" --verify)
    [ -n "${sa_out}" ] && cmd+=(--sa-out "${sa_out}")

    echo "=== ${id}"
    if [ "${rlimit}" -eq 1 ]
        then ( ulimit -v "${RLIMIT_KIB}"; exec ${NUMA[@]+"${NUMA[@]}"} "${cmd[@]}" )
        else ${NUMA[@]+"${NUMA[@]}"} "${cmd[@]}"
    fi

    # The record is authoritative; read the verdict back out of it. Report the
    # run's status alongside, so "the run failed" is distinguishable from
    # "the run succeeded but its output is wrong" -- very different problems.
    local v st
    v="$(sed -n 's/.*"verify":"\([a-z]*\)".*/\1/p' "${RESULTS}/${id}.json" 2>/dev/null)"
    st="$(sed -n 's/.*"status":"\([a-z_]*\)".*/\1/p' "${RESULTS}/${id}.json" 2>/dev/null)"
    if [ "${v}" = pass ]
        then echo "    verify: pass"
    elif [ -z "${st}" ]
        then echo "    verify: NO RECORD  <-- FAILED (the process died; see the log)"; fail=1
    elif [ "${st}" != ok ]
        then echo "    verify: n/a  <-- FAILED: the run itself did not complete (status=${st})"; fail=1
        else echo "    verify: ${v}  <-- FAILED: run completed but its SA is wrong"; fail=1
    fi
}

# 1. Self-verification, one run per (method, alphabet) path through the harness.
verify "${SAIS}"  libsais16x64 token16           "$(nproc)" ""                     1
verify "${SAIS}"  libsais64    byte_from_token16 "$(nproc)" "${SA_DIR}/sais.sa"    1
verify "${CUBWT}" libcubwt     byte_from_token16 0          "${SA_DIR}/cubwt.sa"

# 2. Cross-verification. This is the load-bearing one: every cross-method
#    comparison assumes libsais and libcubwt order suffixes the same way
#    (shorter-suffix-first, no appended sentinel). Both drivers write canonical
#    little-endian uint64 regardless of native width, so agreement is a `cmp`.
#
#    Note the pairing: both sides read the SAME bytes. libsais16x64 on tokens
#    against libcubwt on those tokens-as-bytes would NOT be comparable, because
#    little-endian byte order is not token order.
echo "=== cross-check libsais64 vs libcubwt on identical bytes"
if [ -s "${SA_DIR}/sais.sa" ] && [ -s "${SA_DIR}/cubwt.sa" ]; then
    if cmp -s "${SA_DIR}/sais.sa" "${SA_DIR}/cubwt.sa"
        then echo "    cross-check: identical"
        else echo "    cross-check: DIFFER  <-- FAILED"; cmp "${SA_DIR}/sais.sa" "${SA_DIR}/cubwt.sa" | head -3; fail=1
    fi
else
    echo "    cross-check: skipped (one or both SA dumps are missing)"
    fail=1
fi

echo
if [ "${fail}" -eq 0 ]
    then echo "ALL CHECKS PASSED -> ${RESULTS}"
    else echo "SOME CHECKS FAILED -> ${RESULTS}"
fi
echo "SA dumps left in ${SA_DIR} (delete when done: $(du -sh "${SA_DIR}" 2>/dev/null | cut -f1))"
exit "${fail}"
