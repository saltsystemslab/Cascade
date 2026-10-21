#!/usr/bin/env bash
#
# Structural-statistics runs of `cascade`, for `structure.py`. One run per
# invocation, on the node with the GPU and the local scratch; e.g. inside an
# allocation, through `srun --jobid=<id> --overlap`.

set -uo pipefail

usage()
{
    cat <<'EOF'
Usage: benchmark/structure.sh full <corpus> <cascade> <sa_structure>
       benchmark/structure.sh partition <corpus> <cascade> <label>

  full       a pinned run with its SA kept on $SA_DIR, then `sa_structure` on
             that SA; writes $RESULTS/runs/<corpus>.log and .structure.json,
             and deletes the SA unless KEEP_SA=1. With IO_TRACE=1, the run's
             /proc/<pid>/io counters and its log's length go to
             <corpus>.log.io.tsv every 0.25 s: its phase lines go to
             unbuffered stderr, so the log's length dates each phase, and the
             bytes can be attributed to the phases
  partition  a partition-only run (GPU_SACA_PARTITION_ONLY=1); writes
             $RESULTS/sweeps/<corpus>_<label>.log. Label `p<P>` for the
             partition-count sweep and `k<K>` for the pivot-width sweep.

Both need STATS builds. A run whose output exists is skipped, so a campaign
resumes when rerun; and a `full` run whose tool failed redoes only the tool.
Each output has a `.env` sidecar: the command, the pins, and the build.

Pins: OMP_NUM_THREADS=$THREADS (default 128), as the pivot samples depend on
it; and GPU_SACA_P_TARGET=$P_TARGET, as the partition-count floor depends on
the free device memory. P_TARGET defaults per corpus to the elastic-residency
validation's pins, and must be set for any other corpus.

Environment (defaults): RESULTS (~/structure-results), CORPUS_ROOT
(/scratch/$USER/corpora), TOKENIZER (olmo for dolmasample, else llama2),
SCRATCH_ROOTS (/tmp; colon-separated, one root per drive), SA_DIR
(/tmp/$USER-structure/sa; with KEEP_SA=1, a drive that outlives the job),
THREADS, P_TARGET, KEEP_SA (0), IO_TRACE (0).

The campaign; BIN_k is a STATS build with -DPIVOT_WORDS=k, i.e. K = 4k tokens:
  E1  for c in dolmasample stackexchange wikipedia arxiv github; do
          structure.sh full $c $BIN $TOOL; done
  E2  for c in ...; do for m in 1 2 4 8 16 32; do P=$((m * <c's pin>));
          P_TARGET=$P structure.sh partition $c $BIN p$P; done; done
  E3  for k in 1 2 3 4 6 8; do for c in ...; do
          structure.sh partition $c $BIN_k k$((4 * k)); done; done
  then  structure.py <out-dir> $RESULTS/runs/*.log --sweep $RESULTS/sweeps/*.log
EOF
}

die() { echo "error: $*" >&2; exit 1; }

case ${1-} in -h|--help) usage; exit 0;; esac
[ $# -eq 4 ] || { usage >&2; exit 2; }
MODE=$1 CORPUS=$2 BIN=$3 ARG=$4
[ "$MODE" = full ] || [ "$MODE" = partition ] || { usage >&2; exit 2; }
[ -x "$BIN" ] || die "no executable at $BIN"

RESULTS=${RESULTS:-$HOME/structure-results}
CORPUS_ROOT=${CORPUS_ROOT:-/scratch/$USER/corpora}
SA_DIR=${SA_DIR:-/tmp/$USER-structure/sa}
THREADS=${THREADS:-128}
KEEP_SA=${KEEP_SA:-0}
IO_TRACE=${IO_TRACE:-0}
if [ "$CORPUS" = dolmasample ]; then TOK=${TOKENIZER:-olmo}; else TOK=${TOKENIZER:-llama2}; fi
INPUT=$CORPUS_ROOT/$CORPUS/$CORPUS-$TOK.tokenized.bin
[ -r "$INPUT" ] || die "no corpus at $INPUT"

# The elastic-residency validation's pins, each at or above its corpus' floor.
case $CORPUS in
    dolmasample) PIN=33;; stackexchange) PIN=86;; wikipedia) PIN=95;; arxiv) PIN=118;; github) PIN=248;; *) PIN=;;
esac
P=${P_TARGET:-$PIN}
[ -n "$P" ] || die "set P_TARGET for $CORPUS: the runs must pin the partition count"

# Dedicated spill directories, so that clearing them after a run touches nothing else.
SPILL=()
IFS=':' read -r -a ROOTS <<< "${SCRATCH_ROOTS:-/tmp}"
for r in "${ROOTS[@]}"; do SPILL+=("$r/$USER-structure/spill"); done
mkdir -p "${SPILL[@]}" "$SA_DIR" || die "cannot create the scratch directories"
PINS=(OMP_NUM_THREADS="$THREADS" GPU_SACA_P_TARGET="$P" GPU_SACA_SCRATCH="$(IFS=:; echo "${SPILL[*]}")")


# Writes the sidecar `<output>.env` of a run: the command, the pins, and the build.
sidecar()
{
    local out=$1; shift
    {
        echo "# $(basename "$0") $MODE $CORPUS $BIN $ARG; host $(hostname), $(date "+%Y-%m-%dT%H:%M:%S%z")"
        echo "input: $INPUT"
        echo "run: $*"
        env | grep -E '^(OMP_|GPU_SACA_)' | sort | sed 's/^/inherited: /'
        echo "commit: $(git -C "$(dirname "$BIN")" rev-parse HEAD 2>/dev/null || echo unknown)" \
             "($(git -C "$(dirname "$BIN")" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ') modified files)"
        grep -E '^(CMAKE_BUILD_TYPE|STATS|TESTS|PREF_KEY_WORDS|PIVOT_WORDS)[:=]' "$(dirname "$BIN")/../CMakeCache.txt" 2>/dev/null | sed 's/^/cmake: /'
    } > "$out.env"
}

# Clears the spill directories, and says so if a run left anything there.
clear_spill()
{
    local left
    left=$(find "${SPILL[@]}" -mindepth 1 -maxdepth 1 2>/dev/null | head -3)
    [ -z "$left" ] || echo "warning: clearing spill left behind: $left" >&2
    for d in "${SPILL[@]}"; do rm -rf "${d:?}"/*; done
}

# Samples the run that records its PID in `$1.pid`, until it exits: every 0.25 s, the time, the run's /proc/<pid>/io
# counters, and the length of its log `$1`, as a row of `$1.io.tsv`.
io_trace()
{
    local log=$1 pid k v r w rb wb cw
    for _ in $(seq 200); do [ -s "$log.pid" ] && break; sleep 0.05; done
    pid=$(cat "$log.pid" 2>/dev/null) || return 0
    printf 't\trchar\twchar\tread_bytes\twrite_bytes\tcancelled_write_bytes\tlog_bytes\n' > "$log.io.tsv"
    while [ -r "/proc/$pid/io" ]; do
        while read -r k v; do
            case $k in rchar:) r=$v;; wchar:) w=$v;; read_bytes:) rb=$v;; write_bytes:) wb=$v;; cancelled_write_bytes:) cw=$v;; esac
        done 2>/dev/null < "/proc/$pid/io" || break
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$EPOCHREALTIME" "$r" "$w" "$rb" "$wb" "$cw" "$(stat -c %s "$log")" >> "$log.io.tsv"
        sleep 0.25
    done
}


if [ "$MODE" = full ]; then
    TOOL=$ARG
    [ -x "$TOOL" ] || die "no executable at $TOOL"
    RUNS=$RESULTS/runs; mkdir -p "$RUNS" || die "cannot create $RUNS"
    LOG=$RUNS/$CORPUS.log OUT=$RUNS/$CORPUS.structure.json SA=$SA_DIR/$CORPUS.SA.bin
    [ -s "$OUT" ] && { echo "skip: $OUT exists"; exit 0; }

    if [ -s "$SA" ] && grep -qx 'exit: 0' "$LOG" 2>/dev/null; then
        echo "reusing the run in $LOG and its SA"
    else
        sidecar "$LOG" "${PINS[*]} $BIN $INPUT $SA"
        echo "[$(date '+%F %T')] $CORPUS: cascade, P_TARGET=$P, $THREADS threads"
        cat "$INPUT" > /dev/null    # Warm the page cache, as the timed runs do.
        RUN=("$BIN" "$INPUT" "$SA")
        rm -f "$LOG.pid" "$LOG.io.tsv"
        if [ "$IO_TRACE" = 1 ]; then
            # The run records its PID for the trace, then becomes `$BIN`.
            RUN=(bash -c 'echo $$ > "$0.pid"; exec "$@"' "$LOG" "${RUN[@]}")
            io_trace "$LOG" &
        fi
        { env "${PINS[@]}" /usr/bin/time "${RUN[@]}"; echo "exit: $?"; } > "$LOG" 2>&1
        wait
        rm -f "$LOG.pid"
        clear_spill
        grep -qx 'exit: 0' "$LOG" || { rm -f "$SA"; die "the run failed; see $LOG"; }
        grep -q '^Partition sizes:' "$LOG" || { rm -f "$SA"; die "$LOG lacks the partitioning lines; is $BIN a STATS build?"; }
    fi

    echo "[$(date '+%F %T')] $CORPUS: sa_structure"
    { OMP_NUM_THREADS=$THREADS /usr/bin/time "$TOOL" "$INPUT" "$SA" "$LOG" "$OUT.tmp"; echo "exit: $?"; } > "$RUNS/$CORPUS.structure.out" 2>&1
    grep -qx 'exit: 0' "$RUNS/$CORPUS.structure.out" || die "sa_structure failed, keeping $SA for a rerun; see $RUNS/$CORPUS.structure.out"
    mv "$OUT.tmp" "$OUT"
    [ "$KEEP_SA" = 1 ] || rm -f "$SA"
    echo "[$(date '+%F %T')] $CORPUS: done"
else
    LABEL=$ARG
    SWEEPS=$RESULTS/sweeps; mkdir -p "$SWEEPS" || die "cannot create $SWEEPS"
    LOG=$SWEEPS/${CORPUS}_$LABEL.log UNUSED=$SA_DIR/$CORPUS.unused.SA.bin
    [ -s "$LOG" ] && { echo "skip: $LOG exists"; exit 0; }

    sidecar "$LOG" "${PINS[*]} GPU_SACA_PARTITION_ONLY=1 $BIN $INPUT $UNUSED"
    echo "[$(date '+%F %T')] $CORPUS $LABEL: partition-only, P_TARGET=$P, $THREADS threads"
    { env "${PINS[@]}" GPU_SACA_PARTITION_ONLY=1 /usr/bin/time "$BIN" "$INPUT" "$UNUSED"; echo "exit: $?"; } > "$LOG.tmp" 2>&1
    clear_spill
    rm -f "$UNUSED"
    mv "$LOG.tmp" "$LOG"
    # An abort in pivots-selection is an infeasible K, and a data point; any other failure is not.
    grep -qx 'exit: 0' "$LOG" || grep -q 'In pivots-selection' "$LOG" || echo "warning: the run failed; see $LOG" >&2
fi
