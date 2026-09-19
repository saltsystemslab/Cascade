#!/usr/bin/env bash
# infini-gram baseline runs. Submit from the repo root:
#   sbatch --export=ALL,CORPORA="dolmasample:olmo arxiv:olmo",REPS=1 \
#          benchmark/infinigram/run.sbatch <results-dir>
# CORPORA: space-separated label:tokenizer pairs, ascending size.
# No --gres: the baseline never touches a GPU.

#SBATCH --job-name=ig_baseline
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
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
export IG_TEMP_ROOT="${IG_TEMP_ROOT:-/tmp}"   # node-local NVMe; /tmp2 is shared, /scratch network

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

# igbench defaults to 64; set CPUS to override.
CPUS="${CPUS:-}"
mkdir -p "${RESULTS}"
[ -r "${HERE}/pin" ] && cp "${HERE}/pin" "${RESULTS}/pin"
echo "host $(hostname)  cpus ${CPUS:-<igbench default>}  temp ${IG_TEMP_ROOT}  results ${RESULTS}"
[ -r /proc/meminfo ] && awk '/^Mem(Total|Available):/ {print "  " $1 " " $2 " " $3}' /proc/meminfo

for pair in ${CORPORA:?CORPORA required}; do
    for r in $(seq 1 "${REPS:-1}"); do
        # run skips itself when its record exists, so a resubmit resumes.
        # --force: a table.0 from a killed earlier attempt must not make
        # build_sa() skip the shard and report success.
        python3 "${HERE}/igbench.py" run "${pair%%:*}" --tokenizer "${pair##*:}" \
                --results-dir "${RESULTS}" --rep "${r}" --force ${CPUS:+--cpus "${CPUS}"} \
                ${MEM:+--mem "${MEM}"}
        echo "    exit $?"
    done
done

python3 "${REPO}/benchmark/summarize.py" "${RESULTS}" || true
