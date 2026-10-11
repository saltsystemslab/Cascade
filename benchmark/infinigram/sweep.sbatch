#!/usr/bin/env bash
# Host-budget sweep of our method, end to end as in e2e.sbatch: one run per plan
# line, into a results directory per corpus and budget, as neither the run-ids
# nor summarize.py's groups tell budgets apart. Submit from the repo root:
#   sbatch --export=ALL,OURS_BIN=<bench_gpu_saca>,OURS_TAG=<tag> \
#          benchmark/infinigram/sweep.sbatch <results-root> <plan>
# A plan line is `label:tokenizer point rep [VAR=value ...]`, `#` starting a
# comment. The point is `default`, `forced` (no RAM-tier, and Π on disk), or a
# budget in GiB, passed on in bytes: GPU_SACA_RAM_BUDGET parses "400G" as 400.
# The plan is re-read before every run, so lines can be added, reordered or
# commented out while the job runs. A line whose record exists is skipped, and so
# is one this job has attempted already, so a resubmit retries only what has no
# record. A file named STOP in the results root ends the job before its next run.
# RESET=0 skips the memory reset before each run.

#SBATCH --job-name=sweep
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail        # not -e: a failed run is data

REPO="${REPO:-${SLURM_SUBMIT_DIR:-${PWD}}}"   # Slurm runs a spooled copy of this
HERE="${REPO}/benchmark/infinigram"
ROOT="${1:?results root required}"
PLAN="${2:?plan file required}"
[ -f "${HERE}/igbench.py" ] || { echo "FATAL: ${HERE} not found; set REPO"; exit 1; }
[ -r "${PLAN}" ] || { echo "FATAL: plan '${PLAN}' is unreadable"; exit 1; }

if [ -r "${REPO}/benchmark/local.env" ]; then
    set +u; . "${REPO}/benchmark/local.env"; set -u
fi

# The points set these per run. Inherited through --export=ALL, they would turn
# every `default` run into something else.
for v in GPU_SACA_RAM_BUDGET GPU_SACA_FORCE_SPILL GPU_SACA_PID_SPILL GPU_SACA_P_TARGET; do
    [ -n "${!v:-}" ] && echo "note: unsetting the inherited ${v}=${!v}"
    unset "${v}"
done

[ -x "${OURS_BIN:-}" ] || { echo "FATAL: OURS_BIN must name a built bench_gpu_saca; got '${OURS_BIN:-}'"; exit 1; }
export OURS_BIN OURS_TAG="${OURS_TAG:-unknown}"
export IG_TEMP_ROOT="${IG_TEMP_ROOT:-/tmp}"
export IG_STAGE_ROOT="${IG_STAGE_ROOT:-/tmp2}"
export GPU_SACA_SCRATCH="${GPU_SACA_SCRATCH:-/tmp:/tmp2}"

if [ -n "${MODULES:-}" ]; then
    set +u
    for i in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh \
             /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        [ -r "${i}" ] && . "${i}" && break
    done
    module load ${MODULES} || echo "WARN: module load failed"
    set -u
fi

# Checked before the first run, which is past staging the corpus: the binary
# links libcudart, which only a loaded cuda module puts on the loader path.
"${OURS_BIN}" >/dev/null 2>&1
if [ $? -eq 127 ]; then
    echo "FATAL: ${OURS_BIN} will not load. MODULES='${MODULES:-}'; missing:"
    ldd "${OURS_BIN}" 2>&1 | grep 'not found'
    exit 1
fi

# As e2e.sbatch's RESET, but with no re-read of the input after it: ours reads
# the input with O_DIRECT, past the page cache.
RESET_BIN=""
if [ "${RESET:-1}" = 1 ]; then
    RESET_BIN="/tmp/reset_memory-${SLURM_JOB_ID:-$$}"
    g++ -std=c++17 -O2 -fopenmp -I "${REPO}/benchmark" "${REPO}/benchmark/reset_memory.cpp" -o "${RESET_BIN}" \
        || { echo "FATAL: cannot build reset_memory"; exit 1; }
fi

JOB="${SLURM_JOB_ID:-$$}"
mkdir -p "${ROOT}"
cp "${PLAN}" "${ROOT}/plan-${JOB}.txt"
echo "host $(hostname)  job ${JOB}  ours ${OURS_TAG} ${OURS_BIN}"
echo "plan ${PLAN}  results ${ROOT}  reset ${RESET_BIN:+on}${RESET_BIN:-off}"
echo "stage ${IG_STAGE_ROOT}  temp ${IG_TEMP_ROOT}  spill ${GPU_SACA_SCRATCH}"

# Prints the results directory of corpus label `$1` at point `$2`, the budgets
# zero-padded so that the directories sort by budget.
point_dir()
{
    case "$2" in
        default|forced) echo "${ROOT}/$1-$2" ;;
        *)              printf '%s/%s-%04d\n' "${ROOT}" "$1" "$((10#$2))" ;;
    esac
}

# Prints the plan's runs, one per line, without comments; malformed lines are
# reported to stderr and dropped.
plan_runs()
{
    sed -e 's/#.*//' "${PLAN}" | while read -r p q r x; do
        [ -n "${p}" ] || continue

        bad=""
        case "${q}" in
            default|forced) ;;
            *[!0-9]*|'')    bad="the point must be default, forced, or GiB" ;;
        esac
        case "${r}" in
            *[!0-9]*|''|0)  bad="the replicate must be a positive integer" ;;
        esac
        for a in ${x}; do
            [[ "${a}" =~ ^[A-Z_][A-Z0-9_]*= ]] || bad="'${a}' is not a VAR=value"
        done
        if [ -n "${bad}" ]; then
            echo "WARN: skipping plan line '${p} ${q} ${r} ${x}': ${bad}" >&2
            continue
        fi

        echo "${p} ${q} ${r} ${x}"
    done
}

staged=""
while :; do
    [ -e "${ROOT}/STOP" ] && { echo "STOP found in ${ROOT}; ending before the next run"; break; }

    # The first line with neither a record nor an attempt by this job.
    pair="" point="" rep="" extra=""
    while read -r p q r x; do
        d="$(point_dir "${p%%:*}" "${q}")"
        compgen -G "${d}/gpu_saca_*_r${r}.json" >/dev/null && continue
        [ -e "${d}/.attempt-r${r}-${JOB}" ] && continue
        pair="${p}" point="${q}" rep="${r}" extra="${x}"
        break
    done < <(plan_runs)
    [ -n "${pair}" ] || { echo "plan exhausted"; break; }

    label="${pair%%:*}" tok="${pair##*:}"
    dir="$(point_dir "${label}" "${point}")"
    mkdir -p "${dir}"
    touch "${dir}/.attempt-r${rep}-${JOB}"

    # The plan's count of this point's lines is what its directory expects.
    if [ ! -e "${dir}/manifest.json" ]; then
        reps=$(plan_runs 2>/dev/null | awk -v p="${pair}" -v q="${point}" '$1 == p && $2 == q' | wc -l)
        python3 "${HERE}/igbench.py" manifest "${pair}" --results-dir "${dir}" --reps "${reps}" --methods gpu_saca
    fi

    # Staged before the reset, so that the copy's page cache goes with it. A
    # stage drops whichever corpus was staged before.
    if [ "${pair}" != "${staged}" ]; then
        python3 "${HERE}/igbench.py" stage "${label}" --tokenizer "${tok}" && staged="${pair}"
    fi

    declare -a penv=()
    case "${point}" in
        default) ;;
        forced)  penv=(GPU_SACA_FORCE_SPILL=1 GPU_SACA_PID_SPILL=1) ;;
        *)       penv=(GPU_SACA_RAM_BUDGET=$(( 10#${point} << 30 ))) ;;
    esac

    echo "=== ${label} ${point} r${rep} ${penv[*]-} ${extra}"
    [ -n "${RESET_BIN}" ] && "${RESET_BIN}"
    # shellcheck disable=SC2086
    env ${penv[@]+"${penv[@]}"} ${extra} python3 "${HERE}/igbench.py" ours "${label}" --tokenizer "${tok}" \
        --results-dir "${dir}" --rep "${rep}" ${CPUS:+--cpus "${CPUS}"}
    echo "    exit $?"
done

[ -z "${staged}" ] || rm -rf "${IG_STAGE_ROOT}/ig-stage/${staged%%:*}-${staged##*:}"
echo "done -> ${ROOT}"
