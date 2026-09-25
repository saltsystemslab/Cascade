#!/usr/bin/env bash
#
# Random-read IOPS and sequential bandwidth of the node-local NVMe
# drives. Feeds the SSD tier's cost model (4 KiB suffix-array page reads, and
# 8 KiB ones for windows straddling two pages) and the sizing of its reader pool.
#
# Usage: nvme.sh OUT_DIR DIR [DIR...]
#   Lays out an `HWPROBE_NVME_SIZE` test file in every DIR, then runs each test on
#   every DIR alone and on all DIRs at once. The files are removed at the end,
#   unless HWPROBE_NVME_KEEP=1.
#
# Environment:
#   FIO              fio binary; default /home/$USER/fio/fio, else `fio` on PATH
#   HWPROBE_NVME_SIZE     test-file size per drive; default 64g
#   HWPROBE_NVME_RUNTIME  seconds per test; default 20
#
# Output: OUT_DIR/nvme.csv, one row per (test, drive set), and fio's JSON per
# test in OUT_DIR/nvme-json/.

set -uo pipefail

usage() { echo "usage: nvme.sh OUT_DIR DIR [DIR...]" >&2; exit 2; }
[ $# -ge 2 ] || usage
out="$1"; shift
dirs=("$@")

FIO="${FIO:-/home/${USER}/fio/fio}"
[ -x "${FIO}" ] || FIO="$(command -v fio || true)"
[ -n "${FIO}" ] || { echo "error: fio not found; set FIO" >&2; exit 1; }
SIZE="${HWPROBE_NVME_SIZE:-64g}"
RUNTIME="${HWPROBE_NVME_RUNTIME:-20}"
mkdir -p "${out}/nvme-json"

# The asynchronous engine: io_uring if this fio build has it, else libaio.
ENGINE=libaio
"${FIO}" --enghelp=io_uring >/dev/null 2>&1 && ENGINE=io_uring
echo "fio: ${FIO} ($("${FIO}" --version 2>/dev/null)); async engine: ${ENGINE}; ${SIZE} per drive; ${RUNTIME} s per test"

# Lay out the test files.
for d in "${dirs[@]}"; do
    echo "laying out ${d}/hwprobe_nvme.dat"
    "${FIO}" --name=layout --filename="${d}/hwprobe_nvme.dat" --size="${SIZE}" --rw=write --bs=1m --direct=1 \
             --ioengine=psync --end_fsync=1 --output=/dev/null \
        || { echo "error: cannot lay out ${d}/hwprobe_nvme.dat" >&2; exit 1; }
done

# Drives are labeled `d<i>` by their position in DIR...; a drive set's label
# joins its drives' labels with `+`.
: > "${out}/nvme-drives.txt"
for i in "${!dirs[@]}"; do
    echo "d${i} ${dirs[i]}" >> "${out}/nvme-drives.txt"
done

# Runs the test `$1` over the drive set `$2` (colon-separated drive indices),
# with the extra fio options in "${@:3}"; every drive gets its own job, and the
# jobs are reported as one group.
run()
{
    local name="$1" set="$2"; shift 2
    local idx jobs=() label=""
    IFS=: read -r -a idx <<< "${set}"
    for i in "${idx[@]}"; do
        jobs+=(--name="d${i}" --filename="${dirs[i]}/hwprobe_nvme.dat")
        label+="${label:++}d${i}"
    done

    local tag="${name}@${label}"
    echo "running ${tag}"
    "${FIO}" --output-format=json --output="${out}/nvme-json/${tag}.json" --group_reporting --time_based \
             --runtime="${RUNTIME}" --size="${SIZE}" --randrepeat=0 --norandommap "$@" "${jobs[@]}" \
        || echo "warn: fio failed for ${tag}" >&2
}

# Every drive alone, then all of them at once.
sets=()
all=""
for i in "${!dirs[@]}"; do
    sets+=("${i}")
    all+="${all:+:}${i}"
done
[ "${#dirs[@]}" -gt 1 ] && sets+=("${all}")

for s in "${sets[@]}"; do
    # Asynchronous engine: a queue-depth sweep with one job per drive, then 4 jobs x 64.
    for qd in 1 16 64 256; do
        run "rand4k-${ENGINE}-qd${qd}" "${s}" --rw=randread --bs=4k --direct=1 --ioengine="${ENGINE}" --iodepth="${qd}" --numjobs=1
    done
    run "rand4k-${ENGINE}-4x64" "${s}" --rw=randread --bs=4k --direct=1 --ioengine="${ENGINE}" --iodepth=64 --numjobs=4

    # Synchronous reader pool: the queue depth is the thread count per drive.
    for nj in 16 64 128 256; do
        run "rand4k-psync-t${nj}" "${s}" --rw=randread --bs=4k --direct=1 --ioengine=psync --numjobs="${nj}"
    done

    # Windows straddling two pages.
    run "rand8k-${ENGINE}-4x64" "${s}" --rw=randread --bs=8k --direct=1 --ioengine="${ENGINE}" --iodepth=64 --numjobs=4

    # Sequential bandwidth: index loads and table builds.
    run "seq1m-${ENGINE}-qd32" "${s}" --rw=read --bs=1m --direct=1 --ioengine="${ENGINE}" --iodepth=32 --numjobs=1
done

# Warm page cache: buffered random 4 KiB reads of cached files, the SSD tier's
# warm mode. Cache the files with one buffered sequential pass first.
for d in "${dirs[@]}"; do
    "${FIO}" --name=warm --filename="${d}/hwprobe_nvme.dat" --size="${SIZE}" --rw=read --bs=1m --direct=0 --invalidate=0 \
             --ioengine=psync --output=/dev/null || true
done
for nj in 16 64; do
    run "rand4k-buffered-warm-t${nj}" "${all}" --rw=randread --bs=4k --direct=0 --invalidate=0 --ioengine=psync --numjobs="${nj}"
done

# Summarize the JSON reports: fio reports bandwidth in KiB/s and latencies in ns.
python3 - "${out}" <<'EOF' || echo "warn: python3 unavailable or failed; see nvme-json/" >&2
import glob, json, os, sys
out = sys.argv[1]
with open(os.path.join(out, 'nvme.csv'), 'w') as f:
    f.write('test,drives,iops,bw_mib_s,clat_p50_us,clat_p99_us\n')
    for p in sorted(glob.glob(os.path.join(out, 'nvme-json', '*.json'))):
        tag = os.path.basename(p)[:-len('.json')]
        name, drives = tag.split('@', 1)
        try:
            r = json.load(open(p))['jobs'][0]['read']
        except (ValueError, KeyError, IndexError):
            continue
        pct = r.get('clat_ns', {}).get('percentile', {})
        f.write('%s,%s,%.0f,%.1f,%.1f,%.1f\n' % (name, drives, r['iops'], r['bw'] / 1024.0,
                pct.get('50.000000', 0) / 1e3, pct.get('99.000000', 0) / 1e3))
EOF

if [ "${HWPROBE_NVME_KEEP:-0}" != 1 ]; then
    for d in "${dirs[@]}"; do rm -f "${d}/hwprobe_nvme.dat"; done
fi
