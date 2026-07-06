
#ifndef TYPES_HPP
#define TYPES_HPP



#include <cstdint>


namespace gpu_saca
{
    using p_id_t = uint16_t;        // Partition-ID type; restricts partition-count to be <= 2^16.
    using idx_t = uint64_t;         // Text-indexing type.
}



#endif
