#!/bin/bash
#
perf_log=${1}
#
cmd="nsys stats --report nvtx_sum ${perf_log}"
echo "${cmd}"
eval "$cmd"

cmd="nsys stats --report cuda_gpu_kern_sum ${perf_log}"
printf "\n\n\n\n"
echo "${cmd}"
eval "$cmd"

cmd="nsys stats --report cuda_gpu_mem_time_sum ${perf_log}"
printf "\n\n\n\n"
echo "${cmd}"
eval "$cmd"

cmd="nsys stats --report cuda_gpu_mem_size_sum ${perf_log}"
printf "\n\n\n\n"
echo "${cmd}"
eval "$cmd"
#