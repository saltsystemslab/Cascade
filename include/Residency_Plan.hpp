
#ifndef RESIDENCY_PLAN_HPP
#define RESIDENCY_PLAN_HPP



#include "types.hpp"

#include <cstddef>
#include <vector>

namespace gpu_saca
{

// Host-residency plan of the partitions' content at the first barrier: per
// field, a leading run of each partition's chunks starts in RAM.
class Residency_Plan
{
private:

    std::vector<idx_t> chunk_c_;    // Count of each partition's chunks.
    std::vector<idx_t> suf_ram_;    // Count of each partition's leading chunks whose suffixes start in RAM.
    std::vector<idx_t> key_ram_;    // Count of each partition's leading chunks whose keys start in RAM.
    const bool force_spill_;    // Whether all data spills regardless of the budget: `GPU_SACA_FORCE_SPILL`.
    const bool chunk_wise_; // Whether the suffixes alone are placed, chunk-wise: `GPU_SACA_CHUNK_PLAN`, on by default.
    std::size_t cap_;   // Bytes of host-memory for the partitions' data: the budget past the fixed allocation, or none if all data spills.
    std::size_t data_resident_; // Bytes of the partitions' data that are host-resident.


public:

    // Plans residency for partitions with capacities `P_cap`, in chunks of
    // `chunk_cap` elements, on host-budget `budget`, given an unavoidable host
    // allocation `fixed_bytes`. The suffixes alone are placed in RAM: whole
    // partitions' largest-first, then as many chunks of the partition next as
    // fit, with the keys starting in disk. Or, with `GPU_SACA_CHUNK_PLAN=0`,
    // whole partitions' data (suffixes and keys) is placed, largest-first, as
    // far as the budget admits.
    Residency_Plan(const std::vector<idx_t>& P_cap, idx_t chunk_cap, std::size_t budget, std::size_t fixed_bytes);

    // Returns the count of the `p_id`'th partition's chunks.
    idx_t chunk_c(const p_id_t p_id) const { return chunk_c_[p_id]; }

    // Returns the count of the `p_id`'th partition's leading chunks whose
    // suffixes start in RAM.
    idx_t suf_ram(const p_id_t p_id) const { return suf_ram_[p_id]; }

    // Returns the count of the `p_id`'th partition's leading chunks whose keys
    // start in RAM.
    idx_t key_ram(const p_id_t p_id) const { return key_ram_[p_id]; }

    // Returns whether all data spills regardless of the budget; the elastic
    // RAM-tiers then get no budget either.
    bool force_spill() const { return force_spill_; }

    // Returns whether the suffixes alone are placed, chunk-wise.
    bool chunk_wise() const { return chunk_wise_; }

    // Returns the bytes of host-memory for the partitions' data: the budget
    // past the fixed allocation, or none if all data spills.
    auto cap() const { return cap_; }

    // Returns the bytes of the partitions' data that are host-resident.
    auto data_resident() const { return data_resident_; }
};

}



#endif
