
#include "tests.hpp"
#include "utility.hpp"

#include "omp.h"

#include <cstdint>
#include <algorithm>


namespace gpu_saca::test
{

template <typename T_sym_>
bool validate_partition_contents(const T_sym_* const T_h, const idx_t n, const p_id_t* const P_id, const std::vector<idx_t>& P_sz, const std::unique_ptr<Spill_Store>& spill)
{
    const auto p_ = P_sz.size();

    const auto P_sz_max = *std::max_element(P_sz.cbegin(), P_sz.cend());
    auto* const suf_buf = util::aligned_allocate<idx_t>(Direct_File::block_up(P_sz_max * sizeof(idx_t)), Direct_File::DISK_BLOCK);  // Buffer to read-in the whole suffix-file of a partition.
    auto* const pre_buf = util::aligned_allocate<Prefix_Key>(Direct_File::block_up(P_sz_max * sizeof(Prefix_Key)), Direct_File::DISK_BLOCK);    // Buffer to read-in the whole prefix-key-file of a partition.
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

        const auto suf_file = spill->open(p_id, Spill_Role::suf, Direct_File::Mode::read);
        const auto pre_file = spill->open(p_id, Spill_Role::prefix, Direct_File::Mode::read);
        suf_file.read_at(suf_buf, sz * sizeof(idx_t), 0);
        pre_file.read_at(pre_buf, sz * sizeof(Prefix_Key), 0);

        #pragma omp parallel for schedule(static) reduction(&& : content_match)
        for(idx_t i = 0; i < sz; ++i)
        {
            const auto j = suf_buf[i];
            if(j >= n || P_id[j] != p_id || present[j] || pre_buf[i] != Prefix_Key::pack_prefix(T_h + j))
                content_match = false;

            present[j] = 1;
        }
    }

    util::deallocate(suf_buf), util::deallocate(pre_buf);

    if(!content_match)
        return false;

    bool absent = false;
    #pragma omp parallel for schedule(static) reduction(|| : absent)
    for(idx_t i = 0; i < n; ++i)
        if(!present[i])
            absent = true;

    return !absent;
}



// Template instantiations for the required instances.
template bool validate_partition_contents(const char* const T_h_, const idx_t n_, const p_id_t* const P_id, const std::vector<idx_t>& P_sz, const std::unique_ptr<Spill_Store>& spill_);
template bool validate_partition_contents(const uint8_t* const T_h_, const idx_t n_, const p_id_t* const P_id, const std::vector<idx_t>& P_sz, const std::unique_ptr<Spill_Store>& spill_);
template bool validate_partition_contents(const uint16_t* const T_h_, const idx_t n_, const p_id_t* const P_id, const std::vector<idx_t>& P_sz, const std::unique_ptr<Spill_Store>& spill_);
template bool validate_partition_contents(const uint32_t* const T_h_, const idx_t n_, const p_id_t* const P_id, const std::vector<idx_t>& P_sz, const std::unique_ptr<Spill_Store>& spill_);
template bool validate_partition_contents(const uint64_t* const T_h_, const idx_t n_, const p_id_t* const P_id, const std::vector<idx_t>& P_sz, const std::unique_ptr<Spill_Store>& spill_);

}
