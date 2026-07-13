#!/bin/bash
#
perf_log=${1}
report=${2}
#
cmd="nsys stats --report nvtx_sum ${perf_log}"
{
    echo "${cmd}" >> "${report}"
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

    cmd="nsys stats --report cuda_gpu_trace ${perf_log}"
    printf "\n\n\n\n"
    echo "${cmd}"
    eval "$cmd" > "${perf_log%.nsys-rep}.gpu_trace.txt"
    echo "  → written to ${perf_log%.nsys-rep}.gpu_trace.txt"

    cmd="nsys stats --report cuda_api_sum ${perf_log}"
    printf "\n\n\n\n"
    echo "${cmd}"
    eval "$cmd"

    cmd="nsys stats --report osrt_sum ${perf_log}"
    printf "\n\n\n\n"
    echo "${cmd}"
    eval "$cmd"
} > "${report}"
#