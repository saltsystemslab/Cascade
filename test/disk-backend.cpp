
#include "tests.hpp"
#include "lane.hpp"
#include "utility.hpp"

#include "omp.h"

#include <cstdint>
#include <algorithm>


namespace gpu_saca::test
{

bool validate_partition_contents(const idx_t n, const Partition_ID_Map& P_id, const std::vector<idx_t>& P_sz, const Partition_Backend& backend)
{
    const auto p_ = P_sz.size();

    const auto P_sz_max = *std::max_element(P_sz.cbegin(), P_sz.cend());
    auto* const suf_buf = util::aligned_allocate<uint8_t>(Direct_File::block_up(lane::bytes(P_sz_max)), Direct_File::DISK_BLOCK);   // Buffer to read-in the whole suffix-file of a partition.
    uint8_t* present = util::allocate<uint8_t>(n); // Whether each suffix is present in the partitions.

    #pragma omp parallel for schedule(static)
    for(idx_t i = 0; i < n; ++i)
        present[i] = 0;

    bool content_match = true;
    for(p_id_t p_id = 0; p_id < p_ && content_match; ++p_id)
    {
        const auto sz = P_sz[p_id];
        if(sz == 0) // TODO: check if there may be any required checks here.
            continue;

        backend.read_suf_chunk(p_id, 0, sz, suf_buf);

        #pragma omp parallel for schedule(static) reduction(&& : content_match)
        for(idx_t i = 0; i < sz; ++i)
        {
            const auto j = lane::load(suf_buf, i);  // A bare suffix-index: any tag-bits put it out of range.
            if(j >= n || P_id[j] != p_id || present[j])
                content_match = false;
            else
                present[j] = 1;
        }
    }

    util::deallocate(suf_buf);

    if(!content_match)
        return false;

    bool absent = false;
    #pragma omp parallel for schedule(static) reduction(|| : absent)
    for(idx_t i = 0; i < n; ++i)
        if(!present[i])
            absent = true;

    return !absent;
}

}
