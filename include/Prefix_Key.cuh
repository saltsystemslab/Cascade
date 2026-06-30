
#ifndef PREFIX_KEY_CUH
#define PREFIX_KEY_CUH



#include "cross-compile.hpp"
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


    // Returns the prefix-key from `T`.
    template <typename T_sym_>
    CUDA_SPECIFIER
    static Prefix_Key pack_prefix(const T_sym_* T);

    CUDA_SPECIFIER
    bool operator<(const Prefix_Key& rhs) const;

    CUDA_SPECIFIER
    bool operator<=(const Prefix_Key& rhs) const;
};


template <typename T_sym_>
inline Prefix_Key Prefix_Key::pack_prefix(const T_sym_* const T)
{
    constexpr idx_t sym_per_word = sizeof(idx_t) / sizeof(T_sym_);
    Prefix_Key pre_key;

    #pragma unroll
    for(idx_t w = 0; w < Prefix_Key::PREFIX_WORDS; ++w)
    {
        idx_t word = 0;

        #pragma unroll
        for(idx_t j = 0; j < sym_per_word; ++j)
        {
            const idx_t shift = (sym_per_word - 1 - j) * sizeof(T_sym_) * 8;
            word |= (static_cast<idx_t>(T[(w * sym_per_word + j)]) << shift); // Note the endianness of the quantization.
        }

        pre_key.word[w] = word;
    }

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

}



#endif
