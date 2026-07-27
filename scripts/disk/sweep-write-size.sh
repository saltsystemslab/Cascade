#!/bin/bash

driver="/home/$USER/fio/fio"
test_dir=${1}
threads=${2}

if [ $# -lt 2 ]; then
    echo "Usage: $0 <test-dir> <thread-count>" >&2
    exit 1
fi

echo "Running bandwidth measurement sweep for write-sizes at ${test_dir}"

for bs in 4k 16k 64k 128k 256k 512k 1m; do
    ${driver}   --name=w_$bs --filename="${test_dir}/fio_test" \
                --rw=write --direct=1 --ioengine=psync --bs=$bs \
                --numjobs="${threads}" --size=256m --group_reporting --unlink=1 2>/dev/null \
    | grep -oE "bw=[0-9.]+[MG]iB/s" | head -1 | sed "s/^/$bs: /"
done
