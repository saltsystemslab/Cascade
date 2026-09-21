#!/usr/bin/env bash
#
# Benchmarking API: validate a campaign, then submit it.
#
# Everything checkable is checked here, on the submit host, before anything is
# queued.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"

usage()
{
    cat <<'EOF'
Usage: benchmark/bench.sh --corpora LIST [options]

  --corpora a,b,c      comma-separated corpus names; required, non-empty
  --tokenizer NAME     one per invocation (default: llama2)
  --methods a,b        baselines from {libsais, libcubwt}; default: none
  --ours TAG           also run our method, built at git tag TAG
  --reps N             replicates per configuration (default 3, minimum 1)
  --threads N          CPU threads for libsais/ours (default: all cores on the node)
  --name LABEL         human label for the campaign directory
  --results-dir DIR    override the computed directory; reuse one to accumulate
  --dry-run            print the plan and exit without submitting
  --yes                skip the confirmation prompt
  -h, --help           this text

At least one of --methods / --ours must be given. --methods defaults to empty so
that `--corpora X --ours TAG` runs only our method; a default of "all" would add
hours of baseline runs to every iteration on our side.

Site configuration (benchmark/local.env): CORPUS_ROOT, RESULTS_ROOT, MODULES,
SBATCH_PARTITION, SBATCH_NODELIST, GPU_SACA_SCRATCH (required by --ours).
EOF
}

die() { echo "error: $*" >&2; exit 1; }

CORPORA= TOKENIZER=llama2 METHODS= OURS_TAG= REPS=3 NAME= RESULTS= DRY=0 YES=0 THREADS=
while [ $# -gt 0 ]; do
    case "$1" in
        --corpora)     CORPORA="${2:?--corpora needs a value}"; shift 2;;
        --tokenizer)   TOKENIZER="${2:?--tokenizer needs a value}"; shift 2;;
        --methods)     METHODS="${2-}"; shift 2;;
        --ours)        OURS_TAG="${2:?--ours needs a tag}"; shift 2;;
        --reps)        REPS="${2:?--reps needs a value}"; shift 2;;
        --threads)     THREADS="${2:?--threads needs a value}"; shift 2;;
        --name)        NAME="${2:?--name needs a value}"; shift 2;;
        --results-dir) RESULTS="${2:?--results-dir needs a value}"; shift 2;;
        --dry-run)     DRY=1; shift;;
        --yes|-y)      YES=1; shift;;
        -h|--help)     usage; exit 0;;
        *)             usage >&2; die "unknown argument: $1";;
    esac
done

# Site config. Bracketed with `set +u` for the same reason the sbatch does it:
# a config file should not have to be `set -u`-clean.
if [ -r "${HERE}/local.env" ]; then
    set +u; . "${HERE}/local.env"; set -u
fi
CORPUS_ROOT="${CORPUS_ROOT:-/scratch/${USER:-$(id -un)}/corpora}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/gpu-saca-results}"

# ---- validation -------------------------------------------------------------
[ -n "${CORPORA}" ] || { usage >&2; die "--corpora is required and must be non-empty"; }
IFS=, read -r -a CORPUS_LIST <<< "${CORPORA}"
IFS=, read -r -a METHOD_LIST <<< "${METHODS}"

[ "${#METHOD_LIST[@]}" -gt 0 ] || [ -n "${OURS_TAG}" ] \
    || die "nothing to run: give --methods, --ours, or both"

for m in ${METHOD_LIST[@]+"${METHOD_LIST[@]}"}; do
    case "${m}" in
        libsais|libcubwt) ;;
        *) die "unknown method '${m}' (known: libsais, libcubwt)";;
    esac
done

case "${REPS}" in
    ''|*[!0-9]*) die "--reps must be a positive integer (got '${REPS}')";;
esac
[ "${REPS}" -ge 1 ] || die "--reps must be at least 1 (got ${REPS})"
[ "${REPS}" -gt 1 ] || echo "WARN: --reps 1 -- a single run cannot state its own dispersion."

if [ -n "${THREADS}" ]; then
    case "${THREADS}" in ''|*[!0-9]*) die "--threads must be a positive integer (got '${THREADS}')";; esac
    [ "${THREADS}" -ge 1 ] || die "--threads must be at least 1 (got ${THREADS})"
fi

# The corpus paths are the real check, and they double as the tokenizer check,
# so no hardcoded corpus or tokenizer list can go stale here.
declare -a SIZES=()
for c in "${CORPUS_LIST[@]}"; do
    f="${CORPUS_ROOT}/${c}/${c}-${TOKENIZER}.tokenized.bin"
    [ -r "${f}" ] || die "corpus not readable: ${f}
       (check --corpora, --tokenizer, and CORPUS_ROOT)"
    SIZES+=("$(stat -c %s "${f}" 2>/dev/null || stat -f %z "${f}")")
done

if [ -n "${OURS_TAG}" ]; then
    # An actual tag, not a branch or a floating ref: a record's method_version
    # has to stay reproducible.
    git -C "${REPO}" rev-parse --verify --quiet "refs/tags/${OURS_TAG}^{commit}" >/dev/null \
        || die "--ours '${OURS_TAG}' is not a git tag in ${REPO}"
    # Self-heals the moment Part 2 lands.
    [ -r "${REPO}/src/bench_gpu_saca.cu" ] \
        || die "--ours needs src/bench_gpu_saca.cu, which is not in this tree yet (Part 2)"
    # Only that it is set: the roots are node-local NVMe, so the submit host
    # cannot stat them. bench.sbatch checks writability once it is on the node.
    [ -n "${GPU_SACA_SCRATCH:-}" ] \
        || die "--ours needs GPU_SACA_SCRATCH set to colon-separated scratch roots
       (node-local NVMe; see benchmark/local.env)"
fi

[ -n "${RESULTS}" ] || RESULTS="${RESULTS_ROOT}/$(date -u +%Y%m%dT%H%M%SZ)${NAME:+-${NAME}}"

# ---- the plan ---------------------------------------------------------------
ALL_METHODS=(${METHOD_LIST[@]+"${METHOD_LIST[@]}"})
[ -n "${OURS_TAG}" ] && ALL_METHODS+=("gpu_saca")
RUNS=$(( ${#CORPUS_LIST[@]} * ${#ALL_METHODS[@]} * REPS ))

echo "corpora    ${CORPUS_LIST[*]}"
echo "tokenizer  ${TOKENIZER}"
echo "methods    ${ALL_METHODS[*]}${OURS_TAG:+  (gpu_saca @ ${OURS_TAG})}"
echo "reps       ${REPS}"
echo "threads    ${THREADS:-all cores on the node}"
echo "results    ${RESULTS}"
echo
printf '%-16s %10s %14s\n' corpus bytes "n (tokens)"
for i in "${!CORPUS_LIST[@]}"; do
    awk -v c="${CORPUS_LIST[$i]}" -v b="${SIZES[$i]}" 'BEGIN{
        printf "%-16s %9.1fGB %13.3e\n", c, b/1e9, b/2;
    }'
done
echo
awk -v runs="${RUNS}" 'BEGIN{ printf "%d runs queued\n", runs }'

[ "${DRY}" -eq 1 ] && { echo; echo "--dry-run: not submitting."; exit 0; }

if [ "${YES}" -eq 0 ]; then
    printf '\nsubmit? [y/N] '
    read -r reply
    case "${reply}" in [yY]*) ;; *) echo "aborted."; exit 1;; esac
fi

# ---- submit -----------------------------------------------------------------
mkdir -p "${RESULTS}" || die "cannot create ${RESULTS}"

# Appended, not overwritten: several invocations may accumulate into one
# campaign directory, and an absent record has to stay distinguishable from a
# run that was never requested.
python3 - "${RESULTS}/manifest.json" <<EOF
import json, sys, pathlib, datetime
p = pathlib.Path(sys.argv[1])
m = json.loads(p.read_text()) if p.exists() else []
m.append({
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "corpora": "${CORPORA}".split(","), "tokenizer": "${TOKENIZER}",
    "methods": [x for x in "${METHODS}".split(",") if x], "ours_tag": "${OURS_TAG}",
    "reps": ${REPS}, "runs_expected": ${RUNS},
    "corpus_root": "${CORPUS_ROOT}", "results": "${RESULTS}",
    "sizes_bytes": [int(x) for x in "${SIZES[*]}".split()],
    "commit": "$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)",
})
p.write_text(json.dumps(m, indent=2) + "\n")
EOF

# --output on the command line, not as an #SBATCH directive: directives cannot
# interpolate a shell variable, so this is the only way the logs land beside the
# records they belong to.
# Slurm splits --export on commas, so a comma-separated value would be parsed as
# further entries: `CORPORA=dolmasample,arxiv` becomes CORPORA=dolmasample plus a
# bare `arxiv`, silently dropping every corpus after the first. Transport the
# lists colon-separated; the sbatch splits on either.
CORPORA_X="${CORPORA//,/:}"
METHODS_X="${METHODS//,/:}"

jid=$(sbatch --parsable \
        --output="${RESULTS}/slurm-%j.out" --error="${RESULTS}/slurm-%j.err" \
        --export=ALL,CORPORA="${CORPORA_X}",TOKENIZER="${TOKENIZER}",METHODS="${METHODS_X}",OURS_TAG="${OURS_TAG}",REPS="${REPS}",THREADS="${THREADS}" \
        "${HERE}/bench.sbatch" "${RESULTS}")
rc=$?
[ "${rc}" -eq 0 ] || die "sbatch failed (rc=${rc})"

echo "submitted job ${jid}"
echo "  records   ${RESULTS}"
echo "  log       ${RESULTS}/slurm-${jid}.out"
echo "  summarize python3 ${REPO}/benchmark/summarize.py ${RESULTS}"
