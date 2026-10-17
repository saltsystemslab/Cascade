#!/usr/bin/env bash
# Measure real tokenization throughput from a finished (or running) tokenize array.
#
#   bash throughput.sh <arrayjobid> [logdir]
#
# Sizing walltime from a guess is how a 64-task array gets a TIMEOUT and costs a
# whole queue cycle. One finished corpus turns that guess into a number.
#
# Uses Elapsed at the allocation level, NOT TotalCPU, for two reasons:
#   * TotalCPU is recorded on the .batch step and reads 00:00:00 on the
#     allocation row, so -X gives zero.
#   * Slurm renders TotalCPU as MM:SS.mmm under an hour but HH:MM:SS above it,
#     and parsing it naively on [:.] inflates the result ~60x. That produced a
#     58x disagreement between the two methods before it was caught.
# Each array task is 1 CPU running sequentially, so Elapsed IS core-seconds,
# modulo a few seconds of interpreter and tokenizer startup per task.
#
# Throughput is strongly corpus-dependent -- long uncompressed documents tokenize
# far faster per token than short gzipped ones, because per-document overhead
# (decompress, JSON parse, dict lookup) dominates at small document sizes. Do NOT
# carry a number from one corpus to another; measure each.
set -uo pipefail

A="${1:?usage: throughput.sh <arrayjobid> [logdir]}"
LOGDIR="${2:-.}"

ids=$(grep -ho '[0-9,]\+ ids' "$LOGDIR"/slurm-tokenize-"$A"_*.out 2>/dev/null \
      | tr -d ',ids ' | paste -sd+ - | bc)
docs=$(grep -ho '[0-9,]\+ docs' "$LOGDIR"/slurm-tokenize-"$A"_*.out 2>/dev/null \
       | tr -d ',docs ' | paste -sd+ - | bc)
files=$(grep -ho '[0-9,]\+ ids' "$LOGDIR"/slurm-tokenize-"$A"_*.out 2>/dev/null | wc -l)

[ -n "${ids:-}" ] || { echo "no '<n> ids' lines under $LOGDIR for array $A" >&2; exit 1; }

read -r ntasks secs <<EOF
$(sacct -j "$A" -X -n -P -o State,Elapsed \
  | awk -F'|' '$1=="COMPLETED" {split($2,t,":"); s+=t[1]*3600+t[2]*60+t[3]; n++}
               END {print n+0, s+0}')
EOF

echo "array $A"
echo "  files tokenized   $files"
echo "  tasks COMPLETED   $ntasks"
echo "  uint16 written    $(printf "%'d" "$ids")   (tokens + separators)"
[ -n "${docs:-}" ] && echo "  documents         $(printf "%'d" "$docs")"
echo "  core-seconds      $(printf "%'d" "$secs")  (sum of Elapsed; 1 CPU/task)"

if [ "${secs:-0}" -gt 0 ]; then
    awk -v i="$ids" -v s="$secs" -v n="$ntasks" 'BEGIN {
        printf "  throughput        %\x27d tokens/s/core\n", i/s
        printf "  per task          %\x27d tokens, %d s avg\n", i/n, s/n
    }'
fi

# Per-task spread. The LAST task to finish gates the afterok concat, so the max
# is what walltime must cover -- the mean is not enough.
echo "  slowest tasks:"
sacct -j "$A" -X -n -P -o JobID,State,Elapsed \
  | awk -F'|' '$2=="COMPLETED"' | sort -t'|' -k3 -r | head -3 \
  | awk -F'|' '{printf "    %-16s %s\n", $1, $3}'
