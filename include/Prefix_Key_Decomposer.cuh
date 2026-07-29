
#ifndef PREFIX_KEY_DECOMPOSER_CUH
#define PREFIX_KEY_DECOMPOSER_CUH



#include "Prefix_Key.hpp"

#include "cuda/std/tuple"

#include <cstddef>
#include <utility>


namespace gpu_saca
{

// Decomposer of a `Prefix_Key` into its constituent 64-bit words. `word[0]` is
// treated as the most significant element.
struct Prefix_Key_Decomposer
{
    template <std::size_t... Idx>
    __host__ __device__
    auto tie_words(Prefix_Key& key, std::index_sequence<Idx...>) const
    {
        return cuda::std::tie(key.word[Idx]...);
    }

    __host__ __device__
    auto operator()(Prefix_Key& key) const
    {
        return tie_words(key, std::make_index_sequence<Prefix_Key::PREFIX_WORDS>{});
    }
};

}



#endif
