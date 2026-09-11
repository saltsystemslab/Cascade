
#include "Residency_Plan.hpp"
#include "env.hpp"

#include <numeric>
#include <iostream>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Residency_Plan::Residency_Plan(const std::vector<idx_t>& P_cap, const std::size_t budget, const std::size_t fixed_bytes):
      data_(P_cap.size(), 0)
{
    constexpr auto GB = [](const std::size_t bytes){ return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); };

    if(budget < fixed_bytes)
    {
        std::cerr << "The host memory-budget of " << GB(budget) << "GBs cannot fit the fixed requirement of " << GB(fixed_bytes) << "GBs. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    constexpr auto data_elem_bytes = sizeof(idx_t) + sizeof(p_sz_t);    // Per-suffix cost of a partition's content: suffix index and local rank.

    auto available = budget - fixed_bytes;
    std::size_t data_resident = 0;
    if(env::get_u64("GPU_SACA_FORCE_SPILL").value_or(0) == 0)   // `GPU_SACA_FORCE_SPILL` spills all data regardless of the budget.
    {
        // Admit as many partitions' data as the budget allows, largest-first.
        std::vector<p_id_t> order(P_cap.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](const auto x, const auto y){ return P_cap[x] > P_cap[y]; });

        for(const auto p_id : order)
        {
            const auto sz = data_elem_bytes * P_cap[p_id];
            if(sz <= available)
                data_[p_id] = 1, available -= sz, data_resident += sz;
        }
    }

#ifdef STATS
    const auto n = std::accumulate(P_cap.cbegin(), P_cap.cend(), idx_t(0));
    const auto data_c = std::count(data_.cbegin(), data_.cend(), 1);
        std::cerr << "Residency plan (budget: " << GB(budget) << "GBs, fixed: " << GB(fixed_bytes) << "GBs):\n"
              << "\tdata:   " << data_c << "/" << P_cap.size() << " partitions, "
              << GB(data_resident) << "GBs resident, " << GB(n * data_elem_bytes - data_resident) << "GBs spilled.\n";
#else
    (void)data_resident;
#endif
}

}
