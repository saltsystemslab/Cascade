
#ifndef TYPES_HPP
#define TYPES_HPP



#include <cstdint>


namespace gpu_saca
{
    using idx_t = uint64_t;     // Text-indexing type.
    using p_id_t = uint16_t;    // Partition-ID type; restricts partition-count to be <= 2^16.
    using p_sz_t = uint32_t;    // Type of partition-sizes; defines local ranks' type.
}



#endif
