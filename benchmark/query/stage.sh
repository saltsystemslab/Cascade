#!/bin/bash
#
# Staging of the query runs' inputs onto node-local storage, sourced by
# timed.sbatch and xcheck.sbatch: files copied in parallel pieces, since a
# single stream reads a shared file system at a fraction of what several do,
# and kept for later runs.
#
# Environment (optional):
#   STAGE_STREAMS   pieces copied at a time; default 16

STAGE_PIECE_MIB=8192    # MiB per piece: a whole number of 64-MiB blocks.


# Copies the file $1, following symlinks, into the directory $2, unless a copy
# of the same size and modification time is there already. Returns 1 on
# failure, leaving no copy under the file's name.
stage_file()
{
    local src dst size pieces blocks=$((STAGE_PIECE_MIB / 64))
    src=$(readlink -f "$1") || return 1
    dst="$2/$(basename "$1")"
    size=$(stat -c %s "${src}") || return 1
    if [ -f "${dst}" ] && [ "$(stat -c %s "${dst}")" = "${size}" ] && [ "$(stat -c %Y "${dst}")" = "$(stat -c %Y "${src}")" ]; then
        return 0
    fi

    mkdir -p "$2" && rm -f "${dst}" "${dst}.part" && truncate -s "${size}" "${dst}.part" || return 1
    pieces=$(((size + STAGE_PIECE_MIB * 1048576 - 1) / (STAGE_PIECE_MIB * 1048576)))
    # shellcheck disable=SC2016   # The piece's `sh` expands its own arguments.
    seq 0 $((pieces - 1)) | xargs -P "${STAGE_STREAMS:-16}" -I{} \
        sh -c 'dd if="$1" of="$2" bs=64M skip=$(($3 * $4)) seek=$(($3 * $4)) count="$4" conv=notrunc status=none' sh "${src}" "${dst}.part" {} "${blocks}" \
        || { rm -f "${dst}.part"; return 1; }

    touch -r "${src}" "${dst}.part" && mv "${dst}.part" "${dst}"
}


# Copies the files "${@:2}" into the directory $1, one after another, each in
# parallel pieces. Returns 1 if any failed.
stage_files()
{
    local d="$1" f rc=0
    shift
    for f in "$@"; do
        stage_file "${f}" "${d}" || { echo "stage: cannot copy ${f}"; rc=1; }
    done
    return ${rc}
}


# Stages our index $1 and infini-gram's index $2 under the directory $3: into
# $3/ours-<name of $1>, with its metadata's paths into the copies, and into
# $3/ig-<name of $2>, unless $2 is empty. Sets SIDX, and SIG unless $2 is
# empty, to the staged copies; sets neither, and returns 1, on failure.
stage_indexes()
{
    local idx="$1" ig="$2" stage="$3" f i parts files=()
    local s_ours="${stage}/ours-$(basename "${idx}")" s_ig="${stage}/ig-$(basename "${ig}")"
    files=("$(sed -n 's/^text=//p' "${idx}/index.meta" | head -1)")
    parts=$(sed -n 's/^sa_part_c=//p' "${idx}/index.meta" | head -1)
    for ((i = 0; i < ${parts:-0}; ++i)); do
        files+=("$(sed -n "s/^sa_part_${i}=//p" "${idx}/index.meta" | head -1)")
    done
    stage_files "${s_ours}" "${files[@]}" || return 1

    if [ -n "${ig}" ]; then
        files=()
        for f in "${ig}"/*; do
            [ -f "${f}" ] && files+=("${f}")
        done
        stage_files "${s_ig}" "${files[@]}" || return 1
    fi

    sync
    awk -F= -v d="${s_ours}" 'BEGIN { OFS = "=" } $1 == "text" || $1 ~ /^sa_part_[0-9]+$/ { n = split($2, p, "/"); $2 = d "/" p[n] } { print }' \
        "${idx}/index.meta" > "${s_ours}/index.meta" || return 1
    SIDX="${s_ours}"
    if [ -n "${ig}" ]; then
        SIG="${s_ig}"
    fi
}
