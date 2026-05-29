
#ifndef TYPES_HPP
#define TYPES_HPP



#include <cstdint>


namespace gpu_saca
{
    using p_id_t = uint16_t;        // Partition-ID type; restricts partition-count to be <= 2^16.
    using pivot_t = uint16_t;       // Pivot type; restricts `K` to be <= 16.
    using kgram_t = pivot_t;        // Quantized k-gram type; same as pivots.
    using freq_cnt_t = uint64_t;    // Global histogram counter type.
    using idx_t = uint64_t;         // Text-indexing type.
}



#endif
