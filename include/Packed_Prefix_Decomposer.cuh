
#ifndef PACKED_PREFIX_DECOMPOSER_CUH
#define PACKED_PREFIX_DECOMPOSER_CUH



#include "Packed_Prefix.hpp"

#include "cuda/std/tuple"

#include <cstddef>
#include <utility>


namespace gpu_saca
{

// Decomposer of a `Packed_Prefix<W_>` into its constituent 64-bit words.
// `word[0]` is treated as the most significant element.
template <idx_t W_>
struct Packed_Prefix_Decomposer
{
    template <std::size_t... Idx>
    __host__ __device__
    auto tie_words(Packed_Prefix<W_>& key, std::index_sequence<Idx...>) const
    {
        return cuda::std::tie(key.word[Idx]...);
    }

    __host__ __device__
    auto operator()(Packed_Prefix<W_>& key) const
    {
        return tie_words(key, std::make_index_sequence<W_>{});
    }
};


// Decomposer of the stored initial sort-key of a suffix.
using Prefix_Key_Decomposer = Packed_Prefix_Decomposer<ELL_WORDS>;

}



#endif
