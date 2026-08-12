
#ifndef PACKED_PREFIX_HPP
#define PACKED_PREFIX_HPP



#include "cross-compile.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <cstddef>


namespace gpu_saca
{

// Number of 64-bit words in the stored initial sort-key of a suffix: the `\ell`
// knob. It defines the resolution of the initial-ranking phase:
// `ELL_WORDS * 8 / sizeof(T_sym)` symbols.
#ifndef ELL
    constexpr idx_t ELL_WORDS = 2;
#else
    constexpr idx_t ELL_WORDS = ELL;
#endif

// Number of 64-bit words in the partitioning prefix, i.e. in a pivot: the `K`
// knob. It defines the discriminating power of the pivots:
// `K_WORDS * 8 / sizeof(T_sym)` symbols.
#ifndef PIVOT_WORDS
    constexpr idx_t K_WORDS = ELL_WORDS;
#else
    constexpr idx_t K_WORDS = PIVOT_WORDS;
#endif


// Cap on the pivot-width. A thread of the partition-ID kernel materializes a
// pivot-wide prefix in registers and holds it across the whole binary search, at
// 2 registers per word; `32` words (`256` bytes) keeps that within ~1/3 of peak
// occupancy without spilling, and is the binding constraint among the walls that
// large `K` runs into (the others being the text-tile overhang of `K - 1` symbols
// per block and the pivot-sample footprint in `select_pivots`).
constexpr idx_t K_MAX_WORDS = 32;

static_assert(ELL_WORDS >= 1, "ELL_WORDS must be at least 1.");
static_assert(K_WORDS >= 1, "K_WORDS must be at least 1.");
static_assert(K_WORDS <= K_MAX_WORDS, "K_WORDS exceeds K_MAX_WORDS; see the register / occupancy rationale at its definition.");


// Number of symbols of type `T_sym_` the text must be padded with at its tail,
// so that prefix-packing at the last suffix stays in-bounds for both widths.
template <typename T_sym_>
constexpr idx_t tail_padding = (ELL_WORDS > K_WORDS ? ELL_WORDS : K_WORDS) * sizeof(idx_t) / sizeof(T_sym_);


// Fixed-width, MSB-first packed prefix of some suffix, spanning `W_` 64-bit
// words.
template <idx_t W_>
struct Packed_Prefix
{
    idx_t word[W_];   // The packed-words representation of the given prefix; `word[0]` holds the most significant symbols at its highest bits.


    // Returns the prefix-key from `T`.
    // NB: it must have enough padding when extracting prefixes from the tail.
    template <typename T_sym_>
    CUDA_SPECIFIER
    static Packed_Prefix pack_prefix(const T_sym_* T);

    CUDA_SPECIFIER
    bool operator<(const Packed_Prefix& rhs) const;

    CUDA_SPECIFIER
    bool operator<=(const Packed_Prefix& rhs) const;

    CUDA_SPECIFIER
    bool operator==(const Packed_Prefix& rhs) const;

    CUDA_SPECIFIER
    bool operator!=(const Packed_Prefix& rhs) const;
};


// The stored initial sort-key of a suffix, of width `\ell`.
using Prefix_Key = Packed_Prefix<ELL_WORDS>;

// The partitioning prefix of a suffix, of width `K`.
using Pivot_Key = Packed_Prefix<K_WORDS>;


template <idx_t W_>
template <typename T_sym_>
CUDA_SPECIFIER
inline Packed_Prefix<W_> Packed_Prefix<W_>::pack_prefix(const T_sym_* const T)
{
    constexpr idx_t sym_per_word = sizeof(idx_t) / sizeof(T_sym_);
    Packed_Prefix pre_key;

    #pragma unroll
    for(idx_t w = 0; w < W_; ++w)
        pre_key.word[w] = util::pack_word(T + w * sym_per_word);

    return pre_key;
}


template <idx_t W_>
CUDA_SPECIFIER
inline bool Packed_Prefix<W_>::operator<(const Packed_Prefix& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < W_; ++i)
        if(word[i] != rhs.word[i])
            return word[i] < rhs.word[i];

    return false;
}


template <idx_t W_>
CUDA_SPECIFIER
inline bool Packed_Prefix<W_>::operator<=(const Packed_Prefix& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < W_; ++i)
        if(word[i] != rhs.word[i])
            return word[i] < rhs.word[i];

    return true;
}


template <idx_t W_>
CUDA_SPECIFIER
inline bool Packed_Prefix<W_>::operator==(const Packed_Prefix& rhs) const
{
    #pragma unroll
    for(std::size_t i = 0; i < W_; ++i)
        if(word[i] != rhs.word[i])
            return false;

    return true;
}


template <idx_t W_>
CUDA_SPECIFIER
inline bool Packed_Prefix<W_>::operator!=(const Packed_Prefix& rhs) const
{
    return !(*this == rhs);
}

}



#endif
