
#ifndef PREFIX_KEY_CUH
#define PREFIX_KEY_CUH



#include "types.hpp"

#include <cstddef>


namespace gpu_saca
{

// Fixed-width, MSB-first packed prefix of some suffix.
struct Prefix_Key
{
    static constexpr idx_t PREFIX_WORDS = 2;    // Number of 64-bit words per prefix key.
                                                // Currently set to `2` so that `Prefix_Key`s have 16-bytes, and thus it matches `Sort_Key`-width — no wasted resolution from the partitioning.

    idx_t word[PREFIX_WORDS];   // The packed-words representation of the given prefix; `word[0]` holds the most significant symbols at its highest bits.
};


__host__ __device__
inline bool operator<(const Prefix_Key& x, const Prefix_Key& y)
{
    #pragma unroll
    for(std::size_t i = 0; i < Prefix_Key::PREFIX_WORDS; ++i)
        if(x.word[i] != y.word[i])
            return x.word[i] < y.word[i];

    return false;
}


__host__ __device__
inline bool operator<=(const Prefix_Key& x, const Prefix_Key& y)
{
    #pragma unroll
    for(std::size_t i = 0; i < Prefix_Key::PREFIX_WORDS; ++i)
        if(x.word[i] != y.word[i])
            return x.word[i] < y.word[i];

    return true;
}

}



#endif
