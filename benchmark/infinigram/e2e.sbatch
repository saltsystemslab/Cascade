#!/usr/bin/env bash
# End-to-end construction: input read through persistent output, off the same
# node-local copy of the corpus their runs read. Submit from the repo root:
#   sbatch --export=ALL,CORPORA="arxiv:llama2",REPS=3,OURS_TAG=v0.8.0,\
#OURS_BIN=${HOME}/gpu-saca-worktrees/v0.8.0/build/src/bench_gpu_saca \
#          benchmark/infinigram/e2e.sbatch <results-dir>
# CORPORA: space-separated label:tokenizer pairs, ascending size.
# METHODS: colon-separated, of infinigram, gpu_saca and libsais. Defaults to gpu_saca
# alone: `run.sbatch` already measures theirs end to end -- their t_core covers
# make-part's read of the staged copy, concat's write, and the fsync -- so their
# baseline is banked, and re-taking it on every version bump of ours would spend
# hours of node time to reproduce a number that has not moved.
# libsais needs LIBSAIS_BIN, a bench_libsais built at the commit under test, and
# takes NUMACTL (e.g. "--cpunodebind=0") as its placement. UNSTAGED=1 reads the
# corpus in place instead of staging it, for libsais runs refused or OOM-killed
# before they would write: their input path does not matter, and staging c4
# alone copies 396 GB.
# RESET=1 runs benchmark/reset_memory before every replicate, then re-reads the
# staged input into the page cache, as staging left it for the first; not for
# ours, which reads it with O_DIRECT, past the page cache. A run's
# huge-page supply otherwise depends on whatever ran before it: libsais's c4p64
# replicates took 13,157, 11,102 and 9,042 s in turn as the node recovered.
# --gres: ours needs the GPU; their baseline ignores it.

#SBATCH --job-name=e2e
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail        # not -e: a failed run is data

REPO="${REPO:-${SLURM_SUBMIT_DIR:-${PWD}}}"   # Slurm runs a spooled copy of this
HERE="${REPO}/benchmark/infinigram"
RESULTS="${1:?results directory required}"
[ -f "${HERE}/igbench.py" ] || { echo "FATAL: ${HERE} not found; set REPO"; exit 1; }

if [ -r "${REPO}/benchmark/local.env" ]; then
    set +u; . "${REPO}/benchmark/local.env"; set -u
fi

METHODS="${METHODS:-gpu_saca}"
declare -a BINS=()
UNSTAGED_ARG=""
if [ "${UNSTAGED:-0}" = 1 ]; then
    [ "${METHODS}" = libsais ] || { echo "FATAL: UNSTAGED=1 is for METHODS=libsais alone; the others read the staged copy"; exit 1; }
    UNSTAGED_ARG=--unstaged
fi

# Building is bench.sbatch's job, keyed on a tag; this consumes the artifact so
# the two campaigns cannot disagree about what the binary is.
case ":${METHODS}:" in
    *:gpu_saca:*)
        [ -x "${OURS_BIN:-}" ] || { echo "FATAL: OURS_BIN must name a built bench_gpu_saca; got '${OURS_BIN:-}'"; exit 1; }
        BINS+=("${OURS_BIN}")
        ;;
esac
case ":${METHODS}:" in
    *:libsais:*)
        [ -x "${LIBSAIS_BIN:-}" ] || { echo "FATAL: LIBSAIS_BIN must name a built bench_libsais; got '${LIBSAIS_BIN:-}'"; exit 1; }
        BINS+=("${LIBSAIS_BIN}")
        ;;
esac
export OURS_BIN="${OURS_BIN:-}" OURS_TAG="${OURS_TAG:-unknown}" LIBSAIS_BIN="${LIBSAIS_BIN:-}"

# Two node-local NVMe drives: input is staged on one, temp and output on the
# other. Ours stripes its spill across both, so the staged input shares /tmp2
# with it; Spill_Store statvfs's the roots and aborts at plan time when the
# striping will not fit, so a shortfall costs a submit rather than a campaign.
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

# Executable is not loadable: ours links libcudart, which only a loaded cuda
# module puts on the loader path, and `-x` above cannot see that. Checked here
# rather than at the first run, which is past staging the corpus.
for b in ${BINS[@]+"${BINS[@]}"}; do
    "${b}" >/dev/null 2>&1
    if [ $? -eq 127 ]; then
        echo "FATAL: ${b} will not load. MODULES='${MODULES:-}'; missing:"
        ldd "${b}" 2>&1 | grep 'not found'
        echo "Set MODULES (see benchmark/local.env) so the job loads the cuda runtime."
        exit 1
    fi
done

# Stamp the binary under test into every record.
if [ -r "${HERE}/pin" ]; then
    set -a; . "${HERE}/pin"; set +a
else
    echo "WARN: no ${HERE}/pin; records will say commit=unknown"
fi

RESET_BIN=""
if [ "${RESET:-0}" = 1 ]; then
    RESET_BIN="/tmp/reset_memory-${SLURM_JOB_ID:-$$}"
    g++ -std=c++17 -O2 -fopenmp -I "${REPO}/benchmark" "${REPO}/benchmark/reset_memory.cpp" -o "${RESET_BIN}" \
        || { echo "FATAL: cannot build reset_memory"; exit 1; }
fi

# igbench defaults to 128; set CPUS to override.
CPUS="${CPUS:-}"
mkdir -p "${RESULTS}"
[ -r "${HERE}/pin" ] && cp "${HERE}/pin" "${RESULTS}/pin"
echo "host $(hostname)  cpus ${CPUS:-<igbench default>}  methods ${METHODS}"
echo "ours ${OURS_TAG} ${OURS_BIN:-<none>}"
echo "libsais ${LIBSAIS_BIN:-<none>}  placement ${NUMACTL:-default}${UNSTAGED_ARG:+  unstaged}"
echo "stage ${IG_STAGE_ROOT}  temp ${IG_TEMP_ROOT}  spill ${GPU_SACA_SCRATCH}  results ${RESULTS}"
python3 "${HERE}/igbench.py" manifest "${CORPORA:?CORPORA required}" \
        --results-dir "${RESULTS}" --reps "${REPS:-1}" --methods "${METHODS}"
[ -r /proc/meminfo ] && awk '/^Mem(Total|Available):/ {print "  " $1 " " $2 " " $3}' /proc/meminfo
df -h "${IG_TEMP_ROOT}" "${IG_STAGE_ROOT}" 2>/dev/null | awk 'NR==1 || /^\/dev/ {print "  " $0}'

for pair in ${CORPORA:?CORPORA required}; do
    # Stage once, before the replicates, so none of them carries the copy. Same
    # layout `run.sbatch` stages, so ours reads the bytes off the drive theirs did.
    [ -n "${UNSTAGED_ARG}" ] || python3 "${HERE}/igbench.py" stage "${pair%%:*}" --tokenizer "${pair##*:}"

    # Both subcommands skip themselves when their record exists, so a resubmit
    # resumes. Each record carries thp_refused and huge@start, which is what
    # tells whether a campaign run weeks apart from another is comparable to it.
    for m in ${METHODS//:/ }; do
        for r in $(seq 1 "${REPS:-1}"); do
            if [ -n "${RESET_BIN}" ]; then
                echo "  reset before ${m} r${r}"
                "${RESET_BIN}"
                [ -n "${UNSTAGED_ARG}" ] || [ "${m}" = gpu_saca ] \
                    || cat "${IG_STAGE_ROOT}/ig-stage/${pair%%:*}-${pair##*:}/tokenized.0" > /dev/null
            fi
            case "${m}" in
                # --force: a table.0 from a killed earlier attempt must not make
                # build_sa() skip the shard and report success.
                infinigram) python3 "${HERE}/igbench.py" run "${pair%%:*}" \
                        --tokenizer "${pair##*:}" --results-dir "${RESULTS}" \
                        --rep "${r}" --force ${CPUS:+--cpus "${CPUS}"} \
                        ${MEM:+--mem "${MEM}"} ;;
                gpu_saca)   python3 "${HERE}/igbench.py" ours "${pair%%:*}" \
                        --tokenizer "${pair##*:}" --results-dir "${RESULTS}" \
                        --rep "${r}" ${CPUS:+--cpus "${CPUS}"} ;;
                # --numactl=, joined: argparse would read a separate
                # "--cpunodebind=0" as an option of its own.
                libsais)    python3 "${HERE}/igbench.py" libsais "${pair%%:*}" \
                        --tokenizer "${pair##*:}" --results-dir "${RESULTS}" \
                        --rep "${r}" ${CPUS:+--cpus "${CPUS}"} \
                        ${NUMACTL:+--numactl="${NUMACTL}"} ${UNSTAGED_ARG} ;;
                *) echo "FATAL: unknown method '${m}'"; exit 1 ;;
            esac
            echo "    exit $?"
        done
    done

    # Drop the staged input once its replicates are done.
    rm -rf "${IG_STAGE_ROOT}/ig-stage/${pair%%:*}-${pair##*:}"
done

python3 "${REPO}/benchmark/summarize.py" "${RESULTS}" || true
