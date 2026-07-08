#!/bin/bash
#
input=${1}
output=${2}
#
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --output="${output}" \
    --force-overwrite=true \
    ./build/src/caps_sa_gpu "${input}"
#