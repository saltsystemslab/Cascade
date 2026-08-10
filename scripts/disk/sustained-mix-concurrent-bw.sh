#!/bin/bash

driver="/home/$USER/fio/fio"
test_dir_1="${1}"
test_dir_2="${2}"
ioengine="${3:-"psync"}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <test-dir-1> <test-dir-2> <(optional) io-engine>" >&2
    exit 1
fi

echo "Running disk-bandwidth measurement for sustained read-write mix over 5 minutes, at ${test_dir_1} and ${test_dir_2} concurrently, for I/O engine ${ioengine}"

cmd="   ${driver} --name=d1 --filename=${test_dir_1}/fio_test  --rw=rw --rwmixread=64 --direct=1 \
            --ioengine=${ioengine} --bs=16m --numjobs=12 --size=20g --time_based --runtime=120 &
        ${driver} --name=d2 --filename=${test_dir_2}/fio_test --rw=rw --rwmixread=64 --direct=1 \
            --ioengine=${ioengine} --bs=16m --numjobs=12 --size=20g --time_based --runtime=120 &
        wait"
echo "$cmd"
eval "$cmd"
