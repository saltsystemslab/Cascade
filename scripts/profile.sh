#!/bin/bash
#
THREADS=128
PREFETCH_DEPTH=0
FORCE_SPILL=0
READER_C=24
SCRATCH=/tmp:/tmp2
P_TARGET=3

driver=/home/${USER}/GPU-Suffix-Array/build/src/caps_sa_gpu
input=${1}
output=${2}
#
# TODO: check for CPU-node binding with `numactl`—
# NUMA nodes adjacent to the GPU can have significantly higher PCIe throughput.
#
# TODO: check if we need GPU-clock fixing.
#
# Warm page-cache
cat ${input} > /dev/null
#
OMP_NUM_THREADS=${THREADS} \
GPU_SACA_PREFETCH_DEPTH=${PREFETCH_DEPTH} \
GPU_SACA_FORCE_SPILL=${FORCE_SPILL} \
GPU_SACA_READER_C=${READER_C} \
GPU_SACA_SCRATCH=${SCRATCH} \
GPU_SACA_P_TARGET=${P_TARGET} \
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --output="${output}" \
    --force-overwrite=true \
    --cudabacktrace=true \
    "${driver}" "${input}"
#