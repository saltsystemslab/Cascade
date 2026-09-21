
#ifndef BUILD_TIMING_HPP
#define BUILD_TIMING_HPP



#include "types.hpp"


namespace gpu_saca
{

// Per-phase wall-clock timing of the SA construction, in seconds.
struct Build_Timing
{
    double prepage_pid = -1;    // Time spent in pre-paging the partition-ID map.
    double text_load   = -1;    // Time spent in  allocating, pre-paging, and reading the text.
    double pivots      = -1;    // Time spent in selecting pivots.
    double partition   = -1;    // Time spent in partitioning.
    double device_setup = -1;   // Time spent in device allocation and staging setup.
    double initial_rank = -1;   // Time spent in initial ranking.
    double exponentiate = -1;   // Time spent in prefix exponentiation.
    double concatenate  = -1;   // Time spent in concatenating partitions.

    idx_t  n = 0;   // Count of symbols in the text.
    p_id_t p = 0;   // Count of partitions.
    idx_t  gen_c = 0;   // Doubling generations executed.

    // Returns the time spent in the core of the method: everything except text-
    // read.
    double core() const { return prepage_pid + pivots + partition + device_setup + initial_rank + exponentiate + concatenate; }
};

}



#endif
