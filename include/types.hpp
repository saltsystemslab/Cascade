
#ifndef TYPES_HPP
#define TYPES_HPP



#include <cstdint>


namespace gpu_saca
{
    using p_id_t = uint16_t;        // Partition-ID type; restricts partition-count to be <= 2^16.
    using idx_t = uint64_t;         // Text-indexing type.

    static constexpr idx_t MSB = (idx_t(1) << (sizeof(idx_t) * 8 - 1)); // Most significant bit of suffix index words.
}



#endif
