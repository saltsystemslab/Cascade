#!/bin/bash
#
# Cross-check of query counts: our engine against infini-gram's, on workloads
# drawn from one corpus: present, mutated, and random patterns of the lengths of
# infini-gram's evaluation, every single symbol, and, given a held-out text of the
# same tokenizer, held-out substrings. Every interval also goes through the
# certificate. Correctness only: the timings it logs are informative at best.
#
# Submit from the repo root:
#
#   . benchmark/local.env && sbatch benchmark/query/xcheck.sbatch INDEX IG_INDEX [OUT_DIR]
#
#   INDEX     our query index (a directory with index.meta)
#   IG_INDEX  infini-gram's index of the same token array (2-byte symbols)
#   OUT_DIR   default ${RESULTS_ROOT}/xcheck-<INDEX's name>-<UTC timestamp>
#
# Results in OUT_DIR: qry/ (workloads), ours/ and ig/ (counts and logs),
# summary.txt (per workload: same or DIFFERENT, and both engines' rates).
#
# Environment (all optional):
#   XC_HELDOUT      held-out text of the same symbols, for the held-out workloads
#   XC_Q            queries per workload of patterns up to 10 long; a tenth of it
#                   for longer ones; default 1000000
#   XC_STRIDE       search modes, as `query find --stride`; default 0,32
#   XC_SORT         batch orders of the modes, as `query find --sort`; default 0
#   XC_RES          tier of both arrays: device (default), pinned, or pageable
#   XC_TEXT_RES     tier of the text; default XC_RES
#   XC_SA_RES       tier of the suffix array; default XC_RES
#   XC_NUMA         our process's NUMA placement: none (default), or interleave,
#                   its memory interleaved over every node, as the host's search
#                   wants
#   XC_STAGE        node-local directory to stage both indexes and the held-out
#                   text under first, as timed.sbatch does, and to keep them for
#                   later runs; default none: they are read in place
#   XC_SPOT         suffix-array order spot-checks at load; default 100000
#   XC_ARCH         CUDA architecture of the in-job build; default 120-real
#   XC_THREADS      infini-gram's pool size; default the job's CPUs
#   XC_CMAKE_ARGS   extra configure arguments, e.g. FETCHCONTENT_SOURCE_DIR_* offline
#   INFINIGRAM_DIR  infini-gram checkout at v2.6.0; default ~/infini-gram

#SBATCH --job-name=xcheck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -uo pipefail    # Not -e: a failed workload is data, and the others must still run.

REPO="${REPO:-${SLURM_SUBMIT_DIR:-${PWD}}}"
if [ ! -r "${REPO}/benchmark/query/xcheck.sbatch" ]; then
    echo "FATAL: '${REPO}' does not look like the repository root; submit from it, or export REPO=/path/to/repo"
    exit 1
fi

IDX="${1:?usage: xcheck.sbatch INDEX IG_INDEX [OUT_DIR]}"
IG="${2:?usage: xcheck.sbatch INDEX IG_INDEX [OUT_DIR]}"

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

OUT="${3:-${RESULTS_ROOT:-${HOME}/gpu-saca-results}/xcheck-$(basename "${IDX}")-$(date -u +%Y%m%dT%H%M%SZ)}"
Q="${XC_Q:-1000000}"
ARCH="${XC_ARCH:-120-real}"
RES="${XC_RES:-device}"
TEXT_RES="${XC_TEXT_RES:-${RES}}"
SA_RES="${XC_SA_RES:-${RES}}"
THREADS="${XC_THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
B="${REPO}/build-xcheck-sm${ARCH%%-*}"
NUMA=()
[ "${XC_NUMA:-none}" = interleave ] && command -v numactl > /dev/null && NUMA=(numactl --interleave=all)
mkdir -p "${OUT}/qry" "${OUT}/ours" "${OUT}/ig"

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
stamp "start: $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), commit $(git -C "${REPO}" rev-parse --short HEAD)$(git -C "${REPO}" diff --quiet HEAD || echo -dirty), text ${TEXT_RES}, suffix array ${SA_RES}, numa ${NUMA[*]:-none}, results ${OUT}"

stamp "build"
# shellcheck disable=SC2086
{ cmake -S "${REPO}" -B "${B}" -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH="${ARCH}" -DBENCHMARK=ON \
        -DFETCHCONTENT_SOURCE_DIR_INFINIGRAM="${INFINIGRAM_DIR:-${HOME}/infini-gram}" ${XC_CMAKE_ARGS:-} \
    && cmake --build "${B}" -j "${THREADS}" --target query bench_infinigram_query; } > "${OUT}/build.log" 2>&1 \
    || { stamp "FATAL: build failed; see ${OUT}/build.log"; exit 1; }

# shellcheck source=stage.sh
. "${REPO}/benchmark/query/stage.sh"

SIDX="${IDX}"
SIG="${IG}"
HELDOUT="${XC_HELDOUT:-}"
if [ "${XC_STAGE:-none}" != none ]; then
    stamp "stage: ${XC_STAGE}"
    if stage_indexes "${IDX}" "${IG}" "${XC_STAGE}" && { [ -z "${HELDOUT}" ] || stage_file "${HELDOUT}" "${XC_STAGE}/heldout"; }; then
        [ -n "${HELDOUT}" ] && HELDOUT="${XC_STAGE}/heldout/$(basename "${HELDOUT}")"
        stamp "stage: done; $(du -shL "${SIDX}" | cut -f1) ours, $(du -shL "${SIG}" | cut -f1) infini-gram"
    else
        SIDX="${IDX}"
        SIG="${IG}"
        HELDOUT="${XC_HELDOUT:-}"
        stamp "WARN: staging failed; the inputs are read in place"
    fi
fi

# Workloads: name, kind, length, count, seed, and extra options of `query gen`.
gen()
{
    "${B}/src/query" gen --index "${SIDX}" --out "${OUT}/qry/$1.qry" --kind "$2" --len "$3" --q "$4" --seed "$5" "${@:6}" > /dev/null \
        || stamp "gen failed: $1"
}

stamp "gen"
seed=1
for m in 1 2 5 10; do gen "present-m${m}" substring ${m} "${Q}" $((seed++)); done
for m in 100 1000; do gen "present-m${m}" substring ${m} $((Q / 10)) $((seed++)); done
for m in 2 5 10; do gen "mutated-m${m}" mutated ${m} "${Q}" $((seed++)); done
for m in 100 1000; do gen "mutated-m${m}" mutated ${m} $((Q / 10)) $((seed++)); done
for m in 1 2 5 10; do gen "random-m${m}" random ${m} "${Q}" $((seed++)); done
gen unigram unigram 1 1 $((seed++))
if [ -n "${HELDOUT}" ]; then
    gen heldout-m10 substring 10 "${Q}" $((seed++)) --source "${HELDOUT}"
    gen heldout-m100 substring 100 $((Q / 10)) $((seed++)) --source "${HELDOUT}"
fi
files=$(ls "${OUT}"/qry/*.qry | paste -sd, -)

stamp "ours"
"${NUMA[@]}" "${B}/src/query" find --index "${SIDX}" --queries "${files}" --text-res "${TEXT_RES}" --sa-res "${SA_RES}" --spot "${XC_SPOT:-100000}" \
    --stride "${XC_STRIDE:-0,32}" --sort "${XC_SORT:-0}" --validate --reps 1 --cnt-dir "${OUT}/ours" > "${OUT}/ours/find.log" 2>&1
stamp "ours: rc=$?"

stamp "infini-gram"
first=1
for f in "${OUT}"/qry/*.qry; do
    s=$(basename "${f}" .qry)
    "${B}/benchmark/bench_infinigram_query" --index "${SIG}" --queries "${f}" --mode direct --threads "${THREADS}" --reps 1 \
        --prewarm ${first} --cnt-out "${OUT}/ig/${s}.cnt" > "${OUT}/ig/${s}.log" 2>&1 || stamp "infini-gram failed: ${s}"
    first=0
done

# Every mode's counts of a workload must equal infini-gram's.
stamp "compare"
fail=0
for c in "${OUT}"/ours/*.cnt; do
    s=$(basename "${c}" .cnt)
    w=${s%@*}
    if cmp -s "${c}" "${OUT}/ig/${w}.cnt"; then r=same; else r=DIFFERENT; fail=$((fail + 1)); fi
    printf '%-24s %-9s ig %s q/s\n' "${s}" "${r}" "$(grep -m1 '^qps=' "${OUT}/ig/${w}.log" | cut -d= -f2)"
done > "${OUT}/summary.txt"
cat "${OUT}/summary.txt"
stamp "compare: ${fail} of $(ls "${OUT}"/ours/*.cnt | wc -l) count files differ; certificate: $(grep -c '^validate=pass' "${OUT}/ours/find.log") pass, $(grep -c '^validate=FAIL' "${OUT}/ours/find.log") fail"
stamp "done"
