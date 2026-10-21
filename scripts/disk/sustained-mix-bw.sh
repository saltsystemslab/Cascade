#!/bin/bash

driver="${FIO:-fio}"
test_dir="${1}"
ioengine="${2:-"psync"}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <test-dir> <(optional) io-engine>" >&2
    exit 1
fi

echo "Running disk-bandwidth measurement for sustained read-write mix over 5 minutes, at ${test_dir} for I/O engine ${ioengine}"

cmd="${driver} --name=sustained_mix --filename=${test_dir}/fio_test \
        --rw=rw --rwmixread=64 --direct=1 --ioengine=${ioengine} --bs=16m \
        --numjobs=24 --size=20g --group_reporting --time_based --runtime=300"
echo "$cmd"
eval "$cmd"
