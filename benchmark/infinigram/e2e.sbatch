#!/usr/bin/env bash
# End-to-end construction for both methods: input read through persistent output,
# from one node-local copy of the corpus. Submit from the repo root:
#   sbatch --export=ALL,CORPORA="arxiv:llama2",REPS=3,OURS_TAG=v0.8.0,\
#OURS_BIN=${HOME}/gpu-saca-worktrees/v0.8.0/build/src/bench_gpu_saca \
#          benchmark/infinigram/e2e.sbatch <results-dir>
# CORPORA: space-separated label:tokenizer pairs, ascending size.
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

# Building is bench.sbatch's job, keyed on a tag; this consumes the artifact so
# the two campaigns cannot disagree about what the binary is.
[ -x "${OURS_BIN:-}" ] || { echo "FATAL: OURS_BIN must name a built bench_gpu_saca; got '${OURS_BIN:-}'"; exit 1; }
export OURS_BIN OURS_TAG="${OURS_TAG:-unknown}"

# Two node-local NVMe drives: input is staged on one, temp and output on the
# other. Ours stripes its spill across both, so the staged input shares /tmp2
# with it -- check the free space the campaign prints below against the corpus.
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

# Stamp the binary under test into every record.
if [ -r "${HERE}/pin" ]; then
    set -a; . "${HERE}/pin"; set +a
else
    echo "WARN: no ${HERE}/pin; records will say commit=unknown"
fi

# igbench defaults to 128; set CPUS to override.
CPUS="${CPUS:-}"
mkdir -p "${RESULTS}"
[ -r "${HERE}/pin" ] && cp "${HERE}/pin" "${RESULTS}/pin"
echo "host $(hostname)  cpus ${CPUS:-<igbench default>}  ours ${OURS_TAG} ${OURS_BIN}"
echo "stage ${IG_STAGE_ROOT}  temp ${IG_TEMP_ROOT}  spill ${GPU_SACA_SCRATCH}  results ${RESULTS}"
python3 "${HERE}/igbench.py" manifest "${CORPORA:?CORPORA required}" \
        --results-dir "${RESULTS}" --reps "${REPS:-1}" --methods "infinigram:gpu_saca"
[ -r /proc/meminfo ] && awk '/^Mem(Total|Available):/ {print "  " $1 " " $2 " " $3}' /proc/meminfo
df -h "${IG_TEMP_ROOT}" "${IG_STAGE_ROOT}" 2>/dev/null | awk 'NR==1 || /^\/dev/ {print "  " $0}'

for pair in ${CORPORA:?CORPORA required}; do
    # Stage once, before the replicates, so none of them carries the copy and
    # both methods read the same bytes off the same drive.
    python3 "${HERE}/igbench.py" stage "${pair%%:*}" --tokenizer "${pair##*:}"

    for r in $(seq 1 "${REPS:-1}"); do
        # Interleaved by replicate rather than grouped by method: the node's
        # supply of 2 MiB blocks falls monotonically across a campaign, so
        # grouping would hand whichever method ran first every clean run.
        #
        # Both subcommands skip themselves when their record exists, so a
        # resubmit resumes. --force: a table.0 from a killed earlier attempt
        # must not make build_sa() skip the shard and report success.
        python3 "${HERE}/igbench.py" run "${pair%%:*}" --tokenizer "${pair##*:}" \
                --results-dir "${RESULTS}" --rep "${r}" --force ${CPUS:+--cpus "${CPUS}"} \
                ${MEM:+--mem "${MEM}"}
        echo "    exit $?"

        python3 "${HERE}/igbench.py" ours "${pair%%:*}" --tokenizer "${pair##*:}" \
                --results-dir "${RESULTS}" --rep "${r}" ${CPUS:+--cpus "${CPUS}"}
        echo "    exit $?"
    done

    # Drop the staged input once its replicates are done.
    rm -rf "${IG_STAGE_ROOT}/ig-stage/${pair%%:*}-${pair##*:}"
done

python3 "${REPO}/benchmark/summarize.py" "${RESULTS}" || true
