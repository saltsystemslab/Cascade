
#ifndef PREFIX_KEY_CUH
#define PREFIX_KEY_CUH



#include "cross-compile.hpp"
#include "types.hpp"
#include "utility.hpp"

#include "cuda/std/tuple"

#include <cstddef>
#include <utility>


namespace gpu_saca
{

// Fixed-width, MSB-first packed prefix of some suffix.
struct Prefix_Key
{
    static constexpr idx_t PREFIX_WORDS = 2;    // Number of 64-bit words per prefix key. It defines the context-size for the initial-ranking phase:
                                                // `PREFIX_WORDS * 8 / sizeof(T_sym)`, the `\ell` knob.
    static_assert(PREFIX_WORDS >= 2, "PREFIX_WORDS must be at least 2; device buffer reuses Prefix_Key space for two separate idx_t.");
    // NB: revisit this hard bound if/when we want to sweep `\ell` to measure convergence stats.

    idx_t word[PREFIX_WORDS];   // The packed-words representation of the given prefix; `word[0]` holds the most significant symbols at its highest bits.


    // Returns the prefix-key from `T`.
    // NB: it must have enough padding when extracting prefixes from the tail.
    template <typename T_sym_>
    CUDA_SPECIFIER
    static Prefix_Key pack_prefix(const T_sym_* T);

    CUDA_SPECIFIER
    bool operator<(const Prefix_Key& rhs) const;

    CUDA_SPECIFIER
    bool operator<=(const Prefix_Key& rhs) const;

    CUDA_SPECIFIER
    bool operator==(const Prefix_Key& rhs) const;

    CUDA_SPECIFIER
    bool operator!=(const Prefix_Key& rhs) const;
};


// Decomposer of a `Prefix_Key` into its constituent 64-bit words. `word[0]` is
// treated as the most significant element.
struct Prefix_Key_Decomposer
{
    template <std::size_t... Idx>
    CUDA_SPECIFIER
    auto tie_words(Prefix_Key& key, std::index_sequence<Idx...>) const
    {
        return cuda::std::tie(key.word[Idx]...);
    }

    CUDA_SPECIFIER
    auto operator()(Prefix_Key& key) const
    {
        return tie_words(key, std::make_index_sequence<Prefix_Key::PREFIX_WORDS>{});
    }
};


template <typename T_sym_>
inline Prefix_Key Prefix_Key::pack_prefix(const T_sym_* const T)
{
    constexpr idx_t sym_per_word = sizeof(idx_t) / sizeof(T_sym_);
    Prefix_Key pre_key;

    #pragma unroll
    for(idx_t w = 0; w < Prefix_Key::PREFIX_WORDS; ++w)
        pre_key.word[w] = util::pack_word(T + w * sym_per_word);

    return pre_key;
}


inline bool Prefix_Key::operator<(const Prefix_Key& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < PREFIX_WORDS; ++i)
        if(word[i] != rhs.word[i])
            return word[i] < rhs.word[i];

    return false;
}


inline bool Prefix_Key::operator<=(const Prefix_Key& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < PREFIX_WORDS; ++i)
        if(word[i] != rhs.word[i])
            return word[i] < rhs.word[i];

    return true;
}


inline bool Prefix_Key::operator==(const Prefix_Key& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < PREFIX_WORDS; ++i)
        if(word[i] != rhs.word[i])
            return false;

    return true;
}


inline bool Prefix_Key::operator!=(const Prefix_Key& rhs) const
{
    return !(*this == rhs);
}

}



#endif
