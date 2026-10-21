#!/bin/bash
#
# Staging of the query runs' inputs onto node-local storage, sourced by
# timed.sbatch and xcheck.sbatch: files copied in parallel pieces, since a
# single stream reads a shared file system at a fraction of what several do.
# Copies already there of the same size and time are used as they are; a job
# deletes its copies when it ends, unless told to keep them.
#
# Environment (optional):
#   STAGE_STREAMS   pieces copied at a time; default 16

STAGE_PIECE_MIB=8192    # MiB per piece: a whole number of 64-MiB blocks.


# Copies the bytes `[$3, $3 + $4)` of the file $1, following symlinks, into
# the file $2, unless a copy of that size and of the file's modification time
# is there already. Returns 1 on failure, leaving no copy under the name $2.
stage_range()
{
    local src dst="$2" off="$3" len="$4" piece=$((STAGE_PIECE_MIB * 1048576)) pieces
    src=$(readlink -f "$1") || return 1
    if [ -f "${dst}" ] && [ "$(stat -c %s "${dst}")" = "${len}" ] && [ "$(stat -c %Y "${dst}")" = "$(stat -c %Y "${src}")" ]; then
        return 0
    fi

    mkdir -p "$(dirname "${dst}")" && rm -f "${dst}" "${dst}.part" && truncate -s "${len}" "${dst}.part" || return 1
    pieces=$(((len + piece - 1) / piece))
    # shellcheck disable=SC2016   # The piece's `sh` expands its own arguments.
    seq 0 $((pieces - 1)) | xargs -P "${STAGE_STREAMS:-16}" -I{} \
        sh -c 'n=$(($4 - $1 * $5)); [ ${n} -gt $5 ] && n=$5; dd if="$2" of="$3" bs=64M iflag=skip_bytes,count_bytes,fullblock oflag=seek_bytes conv=notrunc skip=$(($6 + $1 * $5)) seek=$(($1 * $5)) count=${n} status=none' \
        sh {} "${src}" "${dst}.part" "${len}" "${piece}" "${off}" \
        || { rm -f "${dst}.part"; return 1; }

    touch -r "${src}" "${dst}.part" && mv "${dst}.part" "${dst}"
}


# Copies the file $1, following symlinks, into the directory $2, unless a copy
# of the same size and modification time is there already. Returns 1 on
# failure, leaving no copy under the file's name.
stage_file()
{
    local size
    size=$(stat -L -c %s "$1") || return 1
    stage_range "$1" "$2/$(basename "$1")" 0 "${size}"
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


# Deletes the staged directories "${@:2}", and then the staging directory $1
# if it is left empty.
unstage()
{
    local d
    for d in "${@:2}"; do
        [ -n "${d}" ] && rm -rf -- "${d}"
    done
    rmdir -- "$1" 2> /dev/null
    return 0
}


# Stages our index $1 and infini-gram's index $2 under the directory $3: ours
# into $3/ours-<name of $1>, with its metadata's paths into the copies; and
# infini-gram's into $3/ig-<name of $2>, unless $2 is empty. The text goes into
# the directory $4 if given, else with ours; infini-gram's token array, if the
# same file, is a link to it rather than a copy of its own. Our suffix array is
# split over the colon-separated directories $5 if given, a part into each, at
# the page-aligned offsets of `index_meta::part_offsets`; else copied as its
# parts are. Sets SIDX, and SIG unless $2 is empty, to the staged copies; sets
# neither, and returns 1, on failure.
stage_indexes()
{
    local idx="$1" ig="$2" stage="$3" text_dir="${4:-}" sa_dirs="${5:-}"
    local s_ours="${stage}/ours-$(basename "${idx}")" s_ig="${stage}/ig-$(basename "${ig}")"
    local text text_dst parts src size k i f o cut next dirs=() part_path=() part_off=()
    text=$(sed -n 's/^text=//p' "${idx}/index.meta" | head -1)
    parts=$(sed -n 's/^sa_part_c=//p' "${idx}/index.meta" | head -1)
    [ -n "${text_dir}" ] || text_dir="${s_ours}"
    mkdir -p "${s_ours}" || return 1
    stage_file "${text}" "${text_dir}" || { echo "stage: cannot copy ${text}"; return 1; }
    text_dst="${text_dir}/$(basename "${text}")"

    if [ -n "${sa_dirs}" ]; then
        [ "${parts}" = 1 ] || { echo "stage: only a suffix array in one part is split"; return 1; }
        src=$(sed -n 's/^sa_part_0=//p' "${idx}/index.meta" | head -1)
        size=$(stat -L -c %s "${src}") || return 1
        IFS=: read -r -a dirs <<< "${sa_dirs}"
        k=${#dirs[@]}
        for ((i = 0; i < k; ++i)); do
            cut=$((size * i / k - size * i / k % 4096))
            next=$((i + 1 < k ? size * (i + 1) / k - size * (i + 1) / k % 4096 : size))
            stage_range "${src}" "${dirs[i]}/sa.u40.part${i}" "${cut}" $((next - cut)) || { echo "stage: cannot copy part ${i} of ${src}"; return 1; }
            part_path+=("${dirs[i]}/sa.u40.part${i}")
            part_off+=("${cut}")
        done
    else
        for ((i = 0; i < ${parts:-0}; ++i)); do
            f=$(sed -n "s/^sa_part_${i}=//p" "${idx}/index.meta" | head -1)
            o=$(sed -n "s/^sa_part_off_${i}=//p" "${idx}/index.meta" | head -1)
            stage_file "${f}" "${s_ours}" || { echo "stage: cannot copy ${f}"; return 1; }
            part_path+=("${s_ours}/$(basename "${f}")")
            part_off+=("${o}")
        done
    fi

    if [ -n "${ig}" ]; then
        mkdir -p "${s_ig}" || return 1
        for f in "${ig}"/*; do
            [ -f "${f}" ] || continue
            if [ "$(readlink -f "${f}")" = "$(readlink -f "${text}")" ]; then
                ln -sfn "${text_dst}" "${s_ig}/$(basename "${f}")" || return 1
            else
                stage_file "${f}" "${s_ig}" || { echo "stage: cannot copy ${f}"; return 1; }
            fi
        done
    fi

    sync
    {
        grep -v -E '^(text|sa_part_c|sa_part_[0-9]+|sa_part_off_[0-9]+)=' "${idx}/index.meta"
        echo "text=${text_dst}"
        echo "sa_part_c=${#part_path[@]}"
        for ((i = 0; i < ${#part_path[@]}; ++i)); do
            echo "sa_part_${i}=${part_path[i]}"
            echo "sa_part_off_${i}=${part_off[i]}"
        done
    } > "${s_ours}/index.meta" || return 1

    SIDX="${s_ours}"
    if [ -n "${ig}" ]; then
        SIG="${s_ig}"
    fi
}


# Prints the sizes of the staged copies, each counted once: the text and the
# suffix array of our index $1, and infini-gram's index $2 unless empty, whose
# links to the text are not followed.
staged_sizes()
{
    local text parts
    text=$(sed -n 's/^text=//p' "$1/index.meta" | head -1)
    mapfile -t parts < <(sed -n 's/^sa_part_[0-9][0-9]*=//p' "$1/index.meta")
    printf '%s text, %s suffix array' "$(du -sh "${text}" | cut -f1)" "$(du -shc "${parts[@]}" | tail -1 | cut -f1)"
    [ -z "${2:-}" ] || printf ', %s infini-gram' "$(du -sh "$2" | cut -f1)"
    echo
}
