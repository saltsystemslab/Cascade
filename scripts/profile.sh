#!/bin/bash
#
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
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --output="${output}" \
    --force-overwrite=true \
    --cudabacktrace=true \
    ./build/src/caps_sa_gpu "${input}"
#