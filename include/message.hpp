
#ifndef MESSAGE_HPP
#define MESSAGE_HPP



#include "types.hpp"
#include "bit-layout.hpp"
#include "Direct_File.hpp"
#include "cross-compile.hpp"

#include <algorithm>


namespace gpu_saca
{

namespace
{

// The MSB of a message's rank carries its finality, aka the poison-pill status.
constexpr inline msg_rank_t MSG_POISON_PILL_BIT = static_cast<msg_rank_t>(1) << (8 * sizeof(msg_rank_t) - 1);
constexpr inline msg_rank_t MSG_RANK_MASK = MSG_POISON_PILL_BIT - 1;    // Mask to extract actual rank-value from a message's rank field.

// A message's rank field carries the target's secondary rank.
static_assert(MSG_RANK_MASK == bit_layout::SECONDARY_MASK, "Message-rank and key-secondary must have the same width.");

}

// Size of a message.
constexpr inline std::size_t MESSAGE_BYTES = sizeof(idx_t) + sizeof(msg_rank_t);

// Size-granularity at which a destination-partition's message-group starts;
// required for `O_DIRECT` alignment purposes.
constexpr inline std::size_t MSG_ALIGN = Direct_File::DISK_BLOCK / std::min<idx_t>(sizeof(idx_t), sizeof(msg_rank_t));

// Count of RAM message-chunks tiling a staging-slab's inbox-range. Slab-
// capacities are multiples of `MSG_CHUNKS_PER_SLAB * MSG_ALIGN`, so that every
// message-chunk holds whole `MSG_ALIGN`-groups of messages.
constexpr inline std::size_t MSG_CHUNKS_PER_SLAB = 8;


// Returns `true` iff the message-rank `r` is a poison-pill.
CUDA_SPECIFIER
inline bool is_poison_pill(const msg_rank_t r) { return r & MSG_POISON_PILL_BIT; }

// Returns a poison-pill wrapping the message-rank `r`.
CUDA_SPECIFIER
inline msg_rank_t poison_pill(const msg_rank_t r) { return r | MSG_POISON_PILL_BIT; }

// Returns the actual rank-value wrapped in the message-rank `r`.
CUDA_SPECIFIER
inline msg_rank_t extract_rank(const msg_rank_t r) { return r & MSG_RANK_MASK; }

// Returns the inbox-parity a generation with index `gen` writes to.
static constexpr inline std::size_t write_parity(const idx_t gen) { return (gen & 1) ^ 1; }

// Returns the inbox-parity a generation with index `gen` reads from.
static constexpr inline std::size_t read_parity(const idx_t gen) { return gen & 1; }

}



#endif
