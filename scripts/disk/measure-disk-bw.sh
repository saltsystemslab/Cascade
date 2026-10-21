#!/bin/bash

driver="${FIO:-fio}"
test_dir="${1}"
ioengine="${2:-"psync"}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <test-dir> <(optional) io-engine>" >&2
    exit 1
fi

echo "Running disk-bandwidth measurement tests at ${test_dir} for I/O engine ${ioengine}"

# Single write job
cmd="${driver} --name=sw --rw=write --bs=1M --size=64G --direct=1 --ioengine=${ioengine} --numjobs=1 --end_fsync=1 --filename=${test_dir}/fio_test"
echo "$cmd"
eval "$cmd"

# Multiple write jobs
cmd="${driver} --name=sw --rw=write --bs=1M --size=64G --direct=1 --ioengine=${ioengine} --numjobs=8 --group_reporting --end_fsync=1 --filename=${test_dir}/fio_test"
echo "$cmd"
eval "$cmd"

# Single read job
cmd="${driver} --name=sw --rw=read --bs=1M --size=64G --direct=1 --ioengine=${ioengine} --numjobs=1 --end_fsync=1 --filename=${test_dir}/fio_test"
echo "$cmd"
eval "$cmd"

# Multiple read jobs
cmd="${driver} --name=sw --rw=read --bs=1M --size=64G --direct=1 --ioengine=${ioengine} --numjobs=8 --group_reporting --end_fsync=1 --filename=${test_dir}/fio_test"
echo "$cmd"
eval "$cmd"
