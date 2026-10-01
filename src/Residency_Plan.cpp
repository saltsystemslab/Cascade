
#include "Residency_Plan.hpp"
#include "env.hpp"

#include <numeric>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Residency_Plan::Residency_Plan(const std::vector<idx_t>& P_cap, const idx_t chunk_cap, const std::size_t budget, const std::size_t fixed_bytes):
      chunk_c_(P_cap.size())
    , suf_ram_(P_cap.size(), 0)
    , key_ram_(P_cap.size(), 0)
    , force_spill_(env::get_u64("GPU_SACA_FORCE_SPILL").value_or(0) != 0)
    , chunk_wise_(env::get_u64("GPU_SACA_CHUNK_PLAN").value_or(1) != 0)
    , cap_(0)
    , data_resident_(0)
{
    constexpr auto GB = [](const std::size_t bytes){ return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); };

    if(budget < fixed_bytes)
    {
        std::cerr << "The host memory-budget of " << GB(budget) << "GBs cannot fit the fixed requirement of " << GB(fixed_bytes) << "GBs. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    cap_ = (force_spill_ ? 0 : budget - fixed_bytes);

    for(std::size_t p_id = 0; p_id < P_cap.size(); ++p_id)
        chunk_c_[p_id] = (P_cap[p_id] + chunk_cap - 1) / chunk_cap;

    constexpr auto data_elem_bytes = sizeof(idx_t) + sizeof(p_sz_t);    // Per-suffix cost of a partition's content: suffix index and local rank.

    auto available = budget - fixed_bytes;
    p_id_t whole_c = 0; // Count of partitions placed whole.
    bool split = false; // Whether a partition is placed in part.
    idx_t split_c = 0;  // Count of chunks placed of the partition split, if any.
    if(!force_spill_)   // `GPU_SACA_FORCE_SPILL` spills all data regardless of the budget.
    {
        std::vector<p_id_t> order(P_cap.size());    // Largest-first.
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](const auto x, const auto y){ return P_cap[x] > P_cap[y]; });

        if(!chunk_wise_)    // Admit as many partitions' data as the budget allows.
            for(const auto p_id : order)
            {
                const auto sz = data_elem_bytes * P_cap[p_id];
                if(sz <= available)
                    suf_ram_[p_id] = key_ram_[p_id] = chunk_c_[p_id], available -= sz, data_resident_ += sz, whole_c++;
            }
        else    // Admit the partitions' suffixes whole while the budget allows, and then as many chunks of the next as fit.
            for(const auto p_id : order)
            {
                const auto sz = sizeof(idx_t) * P_cap[p_id];
                if(sz <= available)
                {
                    suf_ram_[p_id] = chunk_c_[p_id], available -= sz, data_resident_ += sz, whole_c++;
                    continue;
                }

                split = true, split_c = suf_ram_[p_id] = available / (sizeof(idx_t) * chunk_cap);   // Whole chunks, all of full capacity.
                data_resident_ += split_c * chunk_cap * sizeof(idx_t);
                break;  // At most one partition splits.
            }
    }

#ifdef STATS
    const auto n = std::accumulate(P_cap.cbegin(), P_cap.cend(), idx_t(0));
    std::cerr << "Residency plan (budget: " << GB(budget) << "GBs, fixed: " << GB(fixed_bytes) << "GBs" << (chunk_wise_ ? ", chunk-wise" : "") << "):\n";
    if(!chunk_wise_)
        std::cerr   << "\tdata:   " << whole_c << "/" << P_cap.size() << " partitions, "
                    << GB(data_resident_) << "GBs resident, " << GB(n * data_elem_bytes - data_resident_) << "GBs spilled.\n";
    else
        std::cerr   << "\tsuffixes: " << whole_c << "/" << P_cap.size() << " partitions whole"
                    << (split ? ", and " + std::to_string(split_c) + " chunks of the next" : "") << "; " << GB(data_resident_) << "GBs resident, " << GB(n * sizeof(idx_t) - data_resident_) << "GBs spilled.\n"
                    << "\tkeys:     " << GB(n * sizeof(p_sz_t)) << "GBs, all spilled.\n";
#endif
    (void)whole_c, (void)split, (void)split_c;
}

}
