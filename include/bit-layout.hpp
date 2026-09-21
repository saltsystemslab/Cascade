
#ifndef BIT_LAYOUT_HPP
#define BIT_LAYOUT_HPP



#include "types.hpp"
#include "cross-compile.hpp"

#include <iostream>


// Bit-layout of the two packed 64-bit words the algorithm carries:
// suffix:
//      63 .. 48 :  forward tag: (biased) partition-ID of the secondary-source of the next generation (see `fwd_tag` for the "bias");
//      47 .. 0 :   value
// key:
//      63 :        spare;
//      62 :        poison-pill marker;
//      61 .. 31 :  primary-key;
//      30 .. 0 :   secondary key.
namespace gpu_saca::bit_layout
{

constexpr idx_t WORD_BITS = 64;
constexpr idx_t TAG_BITS = 16;  // Forward-tag: a partition-ID, biased past the reserved band; see `fwd_tag`.
constexpr idx_t VAL_BITS = WORD_BITS - TAG_BITS;    // 48: suffix-indices and global ranks up to ~2.8e14.
constexpr idx_t SUF_BITS = VAL_BITS;

static_assert(TAG_BITS + VAL_BITS == WORD_BITS, "fields must tile a full word");
static_assert(sizeof(idx_t) * 8 == WORD_BITS, "bit_layout assumes a 64-bit `idx_t`");

// NB: necessary, but not sufficient: the tag also carries the `ell`-wide bias `fwd_tag` applies,
// so `validate` must be passed the biased partition-count.
static_assert(sizeof(p_id_t) * 8 <= TAG_BITS, "tag field must hold be able to hold `p_id_t`");


constexpr idx_t TAG_SHIFT = VAL_BITS;   // Bit position of the forward-tag in a word.
constexpr idx_t TAG_MASK = (idx_t(1) << TAG_BITS) - 1;  // Mask to extract the forward-tag from a word.
constexpr idx_t VAL_MASK = (idx_t(1) << VAL_BITS) - 1;  // Mask to extract the actual values (bar the forward-tag) from a word.

// Largest text length and partition-count the bit_layout supports.
constexpr idx_t MAX_N = idx_t(1) << VAL_BITS;
constexpr idx_t MAX_P = idx_t(1) << TAG_BITS;


constexpr idx_t SECONDARY_BITS = 31;
constexpr idx_t PRIMARY_BITS = 31;
constexpr idx_t SECONDARY_MASK = (idx_t(1) << SECONDARY_BITS) - 1;
constexpr idx_t PRIMARY_MASK = ((idx_t(1) << PRIMARY_BITS) - 1) << SECONDARY_BITS;

constexpr idx_t KEY_BITS = SECONDARY_BITS + PRIMARY_BITS;
constexpr idx_t POISON_PILL_RECEIPT = idx_t(1) << KEY_BITS;

// Reserved secondary-key value denoting "no rank-propagation message has joined
// this suffix at this generation". Complete coverage makes it unobservable.
constexpr idx_t SECONDARY_ABSENT = SECONDARY_MASK;

// Largest representable local rank.
constexpr idx_t MAX_LOCAL_RANK = SECONDARY_ABSENT - 1;

// Largest permissible partition-size, so that a local-rank fits key's and
// message's rank-field.
constexpr idx_t MAX_PART_SZ = MAX_LOCAL_RANK + 1;

// The primary and the secondary hold partition-local ranks, so they are in the
// same rank-space.
static_assert(PRIMARY_BITS == SECONDARY_BITS, "primary and secondary must share a single partition-size constraint");


constexpr idx_t BETA_BITS = 16; // Current generation's backward routing tag in low half of the pair, and preceding generation's one in high half.
constexpr idx_t BETA_CUR_GEN_MASK = (idx_t(1) << BETA_BITS) - 1;


// Returns the word packing `val` with forward tag `tag`.
CUDA_SPECIFIER inline idx_t pack(const idx_t val, const idx_t tag) { return (tag << TAG_SHIFT) | val; }

// Returns the value held in word `w`.
CUDA_SPECIFIER inline idx_t value_of(const idx_t w) { return w & VAL_MASK; }

// Returns the forward tag held in word `w`.
CUDA_SPECIFIER inline idx_t tag_of(const idx_t w) { return w >> TAG_SHIFT; }

// Returns word `w` with its forward tag replaced by `tag`.
CUDA_SPECIFIER inline idx_t retag(const idx_t w, const idx_t tag) { return pack(w & VAL_MASK, tag); }

// Returns the forward tag for a suffix whose secondary-source of the next
// generation is the suffix `tgt`, of an `n`-length text with partition-ID map
// `P_id`. Real partition-IDs are biased up by `ell`, reserving the bottom `ell`
// tag-values for sources lying past the text's end. Reserved value decreases as
// the source runs farther past, ordering the shorter of two such sources first.
// In absence of a text-terminator, this keeps the ranking prefix-aware.
inline idx_t fwd_tag(const p_id_t* const P_id, const idx_t tgt, const idx_t n, const idx_t ell)
{ return tgt < n ? static_cast<idx_t>(P_id[tgt]) + ell : (n + ell - 1 - tgt); }

// Returns the suffix-index held in word `w`.
CUDA_SPECIFIER inline idx_t suffix_of(const idx_t w) { return value_of(w); }

// Returns the rank held in word `w`.
CUDA_SPECIFIER inline idx_t rank_of(const idx_t w) { return value_of(w); }

// Returns the primary key packed in the composite key `key`.
CUDA_SPECIFIER inline idx_t primary_of(const idx_t key) { return (key & PRIMARY_MASK) >> SECONDARY_BITS; }

// Returns the secondary key packed in the composite key `key`.
CUDA_SPECIFIER inline idx_t secondary_of(const idx_t key) { return key & SECONDARY_MASK; }

// Returns `true` iff the composite key `key` had received a poison pill.
CUDA_SPECIFIER inline bool received_poison_pill(const idx_t key) { return key & POISON_PILL_RECEIPT; }

// Returns `true` iff no rank-propagation message joined the composite key `key`
// at this generation.
CUDA_SPECIFIER inline bool secondary_absent(const idx_t key) { return secondary_of(key) == SECONDARY_ABSENT; }

// Returns the composite key made from `primary` and `secondary`.
CUDA_SPECIFIER inline idx_t key(const idx_t primary, const idx_t secondary) { return (primary << SECONDARY_BITS) | secondary; }

// Returns the composite key `key` marking it as having received a poison pill.
CUDA_SPECIFIER inline idx_t mark_poison_pill_receipt(const idx_t key) { return key | POISON_PILL_RECEIPT; }

// Returns the backward routing tag pair for a suffix whose payload-message and
// forwarded poison-pill route to the `cur`'th partition and the `prev`'th
// partitions, respectively.
CUDA_SPECIFIER inline beta_t beta_pair(const p_id_t cur, const p_id_t prev) { return (static_cast<beta_t>(prev) << BETA_BITS) | cur; }

// Returns the current generation's backward routing tag from the tag-pair
// `beta`.
CUDA_SPECIFIER inline p_id_t beta_cur_gen(const beta_t beta) { return beta & BETA_CUR_GEN_MASK; }

// Returns the previous generation's backward routing tag from the tag-pair
// `beta`.
CUDA_SPECIFIER inline p_id_t beta_prev_gen(const beta_t beta) { return beta >> BETA_BITS; }


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
