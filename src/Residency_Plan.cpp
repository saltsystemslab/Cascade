
#include "Residency_Plan.hpp"
#include "utility.hpp"
#include "env.hpp"

#include <numeric>
#include <iostream>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Residency_Plan::Residency_Plan(const std::vector<idx_t>& P_cap, const std::size_t fixed_bytes, const std::size_t prefix_elem_bytes):
      data_(P_cap.size(), 0)
    , prefix_(P_cap.size(), 0)
{
    const auto GB = [](const std::size_t bytes){ return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); };

    const auto budget = env::get_u64("GPU_SACA_RAM_BUDGET").value_or(static_cast<uint64_t>(MEM_SAFETY_FRACTION * util::available_memory()));
    if(budget < fixed_bytes)
    {
        std::cerr << "The host memory-budget of " << GB(budget) << "GBs cannot fit the fixed requirement of " << GB(fixed_bytes) << "GBs. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    auto available = budget - fixed_bytes;
    std::vector<p_id_t> order(P_cap.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](const auto x, const auto y){ return P_cap[x] > P_cap[y]; });

    // Admit as many partitions' units, each of size `elem_bytes`, as the budget
    // fits, largest-first.
    const auto admit = [&](std::vector<uint8_t>& resident, const std::size_t elem_bytes)
    {
        std::size_t admitted = 0;
        for(const auto p_id : order)
        {
            const auto sz = elem_bytes * P_cap[p_id];
            if(sz <= available)
                resident[p_id] = 1, available -= sz, admitted += sz;
        }

        return admitted;
    };

#ifdef STATS
    const auto data_resident = admit(data_, 2 * sizeof(idx_t));
    const auto prefix_resident = admit(prefix_, prefix_elem_bytes);

    const auto n = std::accumulate(P_cap.cbegin(), P_cap.cend(), idx_t(0));
    const auto data_c = std::count(data_.cbegin(), data_.cend(), 1);
    const auto prefix_c = std::count(prefix_.cbegin(), prefix_.cend(), 1);
        std::cerr << "Residency plan (budget: " << GB(budget) << "GBs, fixed: " << GB(fixed_bytes) << "GBs):\n"
              << "\tdata:   " << data_c << "/" << P_cap.size() << " partitions, "
              << GB(data_resident) << "GBs resident, " << GB(2 * sizeof(idx_t) * n - data_resident) << "GBs spilled.\n"
              << "\tprefix: " << prefix_c << "/" << P_cap.size() << " partitions, "
              << GB(prefix_resident) << "GBs resident, " << GB(prefix_elem_bytes * n - prefix_resident) << "GBs spilled.\n";

#endif
}

}
