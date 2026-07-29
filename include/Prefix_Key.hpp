
#ifndef PREFIX_KEY_CUH
#define PREFIX_KEY_CUH



#include "cross-compile.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <cstddef>


namespace gpu_saca
{

// Fixed-width, MSB-first packed prefix of some suffix.
struct Prefix_Key
{
    // Number of 64-bit words per prefix key. It defines the context-size for the initial-ranking phase:
    // `PREFIX_WORDS * 8 / sizeof(T_sym)`, the `\ell` knob.
#ifndef ELL
    static constexpr idx_t PREFIX_WORDS = 2;
#else
    static constexpr idx_t PREFIX_WORDS = ELL;
#endif

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


template <typename T_sym_>
CUDA_SPECIFIER
inline Prefix_Key Prefix_Key::pack_prefix(const T_sym_* const T)
{
    constexpr idx_t sym_per_word = sizeof(idx_t) / sizeof(T_sym_);
    Prefix_Key pre_key;

    #pragma unroll
    for(idx_t w = 0; w < Prefix_Key::PREFIX_WORDS; ++w)
        pre_key.word[w] = util::pack_word(T + w * sym_per_word);

    return pre_key;
}


CUDA_SPECIFIER
inline bool Prefix_Key::operator<(const Prefix_Key& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < PREFIX_WORDS; ++i)
        if(word[i] != rhs.word[i])
            return word[i] < rhs.word[i];

    return false;
}


CUDA_SPECIFIER
inline bool Prefix_Key::operator<=(const Prefix_Key& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < PREFIX_WORDS; ++i)
        if(word[i] != rhs.word[i])
            return word[i] < rhs.word[i];

    return true;
}


CUDA_SPECIFIER
inline bool Prefix_Key::operator==(const Prefix_Key& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < PREFIX_WORDS; ++i)
        if(word[i] != rhs.word[i])
            return false;

    return true;
}


CUDA_SPECIFIER
inline bool Prefix_Key::operator!=(const Prefix_Key& rhs) const
{
    return !(*this == rhs);
}

}



#endif
