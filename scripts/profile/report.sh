#!/bin/bash

perf_log="${1}"
report="${2}"

# Initialize report file
:> "${report}"

# Helper function to append reports cleanly
run_report() {
    local report_type="$1"
    echo "nsys stats --report ${report_type} ${perf_log}" >> "${report}"
    nsys stats --report "${report_type}" "${perf_log}" >> "${report}"
    printf "\n\n\n\n" >> "${report}"
}

# 1. Run standard reports that go directly into the main report file
run_report "nvtx_sum"
run_report "cuda_gpu_kern_sum"
run_report "cuda_gpu_mem_time_sum"
run_report "cuda_gpu_mem_size_sum"

# 2. Handle the massive GPU Trace separately
trace_file="${perf_log%.nsys-rep}.gpu_trace.txt"
echo "nsys stats --report cuda_gpu_trace ${perf_log}" > "${trace_file}"
nsys stats --report cuda_gpu_trace "${perf_log}" >> "${trace_file}"

# Log the status text into the main report file
echo " → written to ${trace_file}" >> "${report}"
printf "\n\n\n\n" >> "${report}"

# 3. Resume running the remaining standard reports
run_report "cuda_api_sum"
run_report "osrt_sum"
