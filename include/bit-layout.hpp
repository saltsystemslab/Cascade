
#ifndef BIT_LAYOUT_HPP
#define BIT_LAYOUT_HPP



#include "types.hpp"
#include "cross-compile.hpp"

#include <iostream>


// Bit-layout of the two packed 64-bit words the algorithm carries:
//      63 .. 48 :  forward routing tag;
//      47 .. 0 :   value
namespace gpu_saca::bit_layout
{

constexpr idx_t WORD_BITS = 64;
constexpr idx_t TAG_BITS = 16;  // Forward-tag: a partition-ID.
constexpr idx_t VAL_BITS = WORD_BITS - TAG_BITS;    // 48: suffix-indices and global ranks up to ~2.8e14.

static_assert(TAG_BITS + VAL_BITS == WORD_BITS, "fields must tile a full word");
static_assert(sizeof(p_id_t) * 8 <= TAG_BITS, "tag field must hold be able to hold `p_id_t`");
static_assert(sizeof(idx_t) * 8 == WORD_BITS, "bit_layout assumes a 64-bit `idx_t`");


constexpr idx_t TAG_SHIFT = VAL_BITS;   // Bit position of the forward-tag in a word.
constexpr idx_t TAG_MASK = (idx_t(1) << TAG_BITS) - 1;  // Mask to extract the forward-tag from a word.
constexpr idx_t VAL_MASK = (idx_t(1) << VAL_BITS) - 1;  // Mask to extract the actual values (bar the forward-tag) from a word.

// Largest text length and partition-count the bit_layout supports.
constexpr idx_t MAX_N = idx_t(1) << VAL_BITS;
constexpr idx_t MAX_P = idx_t(1) << TAG_BITS;


// Returns the word packing `val` with forward tag `tag`.
CUDA_SPECIFIER inline idx_t pack(const idx_t val, const idx_t tag) { return (tag << TAG_SHIFT) | val; }

// Returns the value held in word `w`.
CUDA_SPECIFIER inline idx_t value_of(const idx_t w) { return w & VAL_MASK; }

// Returns the forward tag held in word `w`.
CUDA_SPECIFIER inline idx_t tag_of(const idx_t w) { return w >> TAG_SHIFT; }

// Returns word `w` with its forward tag replaced by `tag`.
CUDA_SPECIFIER inline idx_t retag(const idx_t w, const idx_t tag) { return pack(w & VAL_MASK, tag); }

// Returns the suffix-index held in word `w`.
CUDA_SPECIFIER inline idx_t suffix_of(const idx_t w) { return value_of(w); }

// Returns the rank held in word `w`.
CUDA_SPECIFIER inline idx_t rank_of(const idx_t w) { return value_of(w); }


// Returns `true` iff a text of size `n` partitioned into `p` parts fits the
// bit-layout.
inline bool validate(const idx_t n, const idx_t p)
{
    if(n > MAX_N)
    {
        std::cerr << "Text length " << n << " exceeds the " << VAL_BITS << "-bit value field (max " << MAX_N << ") of the packed-bit layout.\n";
        return false;
    }

    if(p > MAX_P)
    {
        std::cerr << "Partition-count " << p << " exceeds the " << TAG_BITS << "-bit tag field (max " << MAX_P << ") of the packed-bit layout.\n";
        return false;
    }

    return true;
}

}



#endif
