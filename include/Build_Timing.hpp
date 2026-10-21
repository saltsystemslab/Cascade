
#ifndef BUILD_TIMING_HPP
#define BUILD_TIMING_HPP



#include "types.hpp"

#include <cstddef>


namespace gpu_saca
{

// Per-phase wall-clock timing of the SA construction, in seconds.
struct Build_Timing
{
    double prepage_pid = -1;    // Time spent in pre-paging the partition-ID map.
    double text_load   = -1;    // Time spent in  allocating, pre-paging, and reading the text.
    double pivots      = -1;    // Time spent in selecting pivots.
    double partition   = -1;    // Time spent in partitioning.
    double device_setup = -1;   // Time spent in device allocation, plus in awaiting the staging setup pinned in the background.
    double initial_rank = -1;   // Time spent in initial ranking.
    double reload_pid  = 0;     // Time spent in reading the partition-ID map back from disk past the initial ranking; `0` if it stayed resident.
    double exponentiate = -1;   // Time spent in prefix exponentiation.
    double concatenate  = -1;   // Time spent in concatenating partitions.

    idx_t  n = 0;   // Count of symbols in the text.
    p_id_t p = 0;   // Count of partitions.
    idx_t  gen_c = 0;   // Doubling generations executed.
    bool   pid_spill = false;   // Whether the partition-ID map stayed on disk until the initial ranking completed.
    std::size_t host_budget = 0;    // Bytes of the host-memory budget.
    std::size_t host_fixed = 0; // Bytes of the budget the fixed allocation takes, the rest going to the RAM-tiers.

    // Returns the time spent in the core of the method: everything except text-
    // read.
    double core() const { return prepage_pid + pivots + partition + device_setup + initial_rank + reload_pid + exponentiate + concatenate; }
};

}



#endif
