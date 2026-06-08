
#ifndef SORT_KEY_HPP
#define SORT_KEY_HPP



#include "types.hpp"


namespace gpu_saca
{

// A 16-byte sort-key for a suffix.
struct Sort_Key
{
    idx_t primary;      // Primary sort-key
    idx_t secondary;    // Secondary sort-key.
};

}



#endif
