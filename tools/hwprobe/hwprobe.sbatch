#!/bin/bash
#
# Hardware probes of the memory tiers the query engine places its index in: the
# rates its cost model and placement planner are parameterized by.
#   gpu_memory   random access to device memory, against the footprint
#   pinned_host  zero-copy access to pinned host memory from the device
#   host_memory  random access to host memory from the cores
#   nvme.sh      random reads and sequential bandwidth of the node-local drives
#
# Submit from the repo root:
#
#   . benchmark/local.env && sbatch tools/hwprobe/hwprobe.sbatch [OUT_DIR]
#
# Results land in OUT_DIR, by default ${RESULTS_ROOT}/hwprobe-<UTC timestamp>:
# host.txt (node provenance), gpu_memory.csv, pinned_host-*.csv,
# host_memory-*.csv, nvme.csv, and nvme-json/; stderr of each probe in *.err.
#
# Environment (all optional):
#   HWPROBE_PINNED_GIB  footprints of pinned_host on the GPU's NUMA node; default 16,64,140,300
#   HWPROBE_HOST_GIB    footprint of host_memory; default 64
#   HWPROBE_NVME_DIRS   colon-separated scratch directories on the drives; default /tmp/$USER-hwprobe:/tmp2/$USER-hwprobe
#   HWPROBE_ARCH        nvcc target architecture; default sm_120
#   HWPROBE_SKIP        comma-separated subset of gpu,pinned,host,nvme to skip

#SBATCH --job-name=hwprobe
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail    # Not -e: one failed probe should not lose the others.

# Slurm runs a copy of this script from a spool directory; the repo is the
# submit directory.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-${PWD}}}"
HP="${REPO}/tools/hwprobe"
if [ ! -r "${HP}/gpu_memory.cu" ]; then
    echo "FATAL: '${REPO}' does not look like the repository root; submit from it, or export REPO=/path/to/repo"
    exit 1
fi

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

OUT="${1:-${RESULTS_ROOT:-${HOME}/gpu-saca-results}/hwprobe-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT}/bin"
NVME_DIRS="${HWPROBE_NVME_DIRS:-/tmp/${USER}-hwprobe:/tmp2/${USER}-hwprobe}"
skip() { [[ ",${HWPROBE_SKIP:-}," == *",$1,"* ]]; }
echo "results: ${OUT}"

# Pinning large buffers needs the memlock limit lifted, if the hard limit allows.
ulimit -l unlimited 2>/dev/null || echo "WARN: cannot lift the memlock limit; it is $(ulimit -l)"

# The GPU's NUMA node, and another node to contrast it with.
bdf="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr 'A-F' 'a-f')"
bdf="${bdf#0000}"   # nvidia-smi pads the PCI domain to 8 digits; sysfs uses 4.
GPU_NODE="$(cat "/sys/bus/pci/devices/${bdf}/numa_node" 2>/dev/null || echo 0)"
[ "${GPU_NODE}" -ge 0 ] 2>/dev/null || GPU_NODE=0
NODES="$(numactl -H 2>/dev/null | awk '/^available:/ {print $2}')"
NODES="${NODES:-1}"
OTHER_NODE=$(( (GPU_NODE + 1) % NODES ))
BIND=""; command -v numactl >/dev/null && BIND="numactl"
membind() { if [ -n "${BIND}" ]; then numactl --membind="$1" "${@:2}"; else "${@:2}"; fi; }

# Node provenance.
{
    echo "date: $(date -u +%FT%TZ)"
    echo "host: $(hostname)"
    uname -a
    echo "commit: $(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null)$(git -C "${REPO}" diff --quiet HEAD 2>/dev/null || echo '-dirty')"
    echo "nproc: $(nproc)"
    echo "gpu pci ${bdf}: numa node ${GPU_NODE} of ${NODES}"
    echo "memlock: $(ulimit -l)"
    echo "--- numactl --show";     numactl --show 2>&1
    echo "--- numactl -H";         numactl -H 2>&1
    echo "--- lscpu";              lscpu
    echo "--- memory";             free -g; grep -E '^(MemTotal|MemAvailable|Hugepagesize|HugePages_Total)' /proc/meminfo
    echo "--- THP";                cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
    echo "--- kernel cmdline";     cat /proc/cmdline
    echo "--- drives";             df -h ${NVME_DIRS//:/ } 2>&1; lsblk -o NAME,MODEL,SIZE,ROTA,TRAN,MOUNTPOINT 2>&1
    echo "--- nvidia-smi topo -m"; nvidia-smi topo -m
    echo "--- nvidia-smi -q";      nvidia-smi -q
} > "${OUT}/host.txt" 2>&1

# Build in-job, for the node's own toolchain.
ARCH="${HWPROBE_ARCH:-sm_120}"
nvcc -O3 -std=c++17 -arch="${ARCH}" "${HP}/gpu_memory.cu" -o "${OUT}/bin/gpu_memory" \
    && nvcc -O3 -std=c++17 -arch="${ARCH}" -Xcompiler -pthread "${HP}/pinned_host.cu" -o "${OUT}/bin/pinned_host" \
    && g++ -O3 -std=c++17 -march=native -pthread "${HP}/host_memory.cpp" -o "${OUT}/bin/host_memory" \
    || { echo "FATAL: build failed"; exit 1; }

if ! skip gpu; then
    echo "gpu_memory"
    "${OUT}/bin/gpu_memory" > "${OUT}/gpu_memory.csv" 2> "${OUT}/gpu_memory.err"
fi

if ! skip pinned; then
    echo "pinned_host"
    G="${HWPROBE_PINNED_GIB:-16,64,140,300}"
    membind "${GPU_NODE}" "${OUT}/bin/pinned_host" --alloc register --gib "${G}" --label "mem${GPU_NODE}_local" \
        > "${OUT}/pinned_host-register-local.csv" 2> "${OUT}/pinned_host-register-local.err"
    membind "${GPU_NODE}" "${OUT}/bin/pinned_host" --alloc hostalloc --gib 16,64 --label "mem${GPU_NODE}_local" \
        > "${OUT}/pinned_host-hostalloc-local.csv" 2> "${OUT}/pinned_host-hostalloc-local.err"
    if [ "${NODES}" -gt 1 ]; then
        membind "${OTHER_NODE}" "${OUT}/bin/pinned_host" --alloc register --gib 16,64 --label "mem${OTHER_NODE}_remote" \
            > "${OUT}/pinned_host-register-remote.csv" 2> "${OUT}/pinned_host-register-remote.err"
    fi
fi

if ! skip host; then
    echo "host_memory"
    for node in $(printf '%s\n' "${GPU_NODE}" "${OTHER_NODE}" | sort -u); do
        for pages in 4k thp; do
            membind "${node}" "${OUT}/bin/host_memory" --gib "${HWPROBE_HOST_GIB:-64}" --pages "${pages}" --label "mem${node}" \
                > "${OUT}/host_memory-mem${node}-${pages}.csv" 2> "${OUT}/host_memory-mem${node}-${pages}.err"
        done
    done
fi

if ! skip nvme; then
    echo "nvme"
    IFS=: read -r -a nvme_dirs <<< "${NVME_DIRS}"
    mkdir -p "${nvme_dirs[@]}"
    bash "${HP}/nvme.sh" "${OUT}" "${nvme_dirs[@]}" > "${OUT}/nvme.log" 2>&1
    rmdir "${nvme_dirs[@]}" 2>/dev/null || true
fi

echo "done: ${OUT}"
