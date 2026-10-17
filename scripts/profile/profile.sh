#!/bin/bash
#
THREADS=64
PREFETCH_DEPTH=0
FORCE_SPILL=0
READER_C=24
SCRATCH=/tmp:/tmp2
# P_TARGET=3

driver=/home/${USER}/GPU-Suffix-Array/build/src/cascade
input=${1}
output=/tmp/SA.bin
nsys_log=${2}
#
# TODO: check for CPU-node binding with `numactl`—
# NUMA nodes adjacent to the GPU can have significantly higher PCIe throughput.
#
# TODO: check if we need GPU-clock fixing.
#
# Warm page-cache
cat ${input} > /dev/null
#
env_vars=(
    "OMP_NUM_THREADS=${THREADS}"
    "GPU_SACA_PREFETCH_DEPTH=${PREFETCH_DEPTH}"
    "GPU_SACA_FORCE_SPILL=${FORCE_SPILL}"
    "GPU_SACA_READER_C=${READER_C}"
    "GPU_SACA_SCRATCH=${SCRATCH}"
)

cmd_args=(
    "${driver}"
    "${input}"
    "${output}"
)

echo "${env_vars[*]} nsys profile ${cmd_args[*]}"

env "${env_vars[@]}" \
    nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --output="${nsys_log}" \
    --force-overwrite=true \
    --cudabacktrace=true \
    "${cmd_args[@]}"
