
#ifndef RESIDENCY_PLAN_HPP
#define RESIDENCY_PLAN_HPP



#include "types.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace gpu_saca
{

class Residency_Plan
{
private:

    std::vector<uint8_t> data_; // Whether each partition's data (suffixes and keys) are host-resident.
    const bool force_spill_;    // Whether all data spills regardless of the budget: `GPU_SACA_FORCE_SPILL`.
    std::size_t data_resident_; // Bytes of the partitions' data that are host-resident.


public:

    // Plans residency for partitions with capacities `P_cap` on host-budget
    // `budget`, given an unavoidable host allocation `fixed_bytes`.
    Residency_Plan(const std::vector<idx_t>& P_cap, std::size_t budget, std::size_t fixed_bytes);

    // Returns whether the `p_id`'th partition's data (suffixes and keys) are
    // host-resident.
    bool data(const p_id_t p_id) const { return data_[p_id]; }

    // Returns whether each partition's data (suffixes and keys) are host-
    // resident.
    const auto& data() const { return data_; }

    // Returns whether all data spills regardless of the budget; the elastic
    // RAM-tiers then get no budget either.
    bool force_spill() const { return force_spill_; }

    // Returns the bytes of the partitions' data that are host-resident.
    auto data_resident() const { return data_resident_; }
};

}



#endif
