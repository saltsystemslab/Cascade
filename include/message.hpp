
#ifndef MESSAGE_HPP
#define MESSAGE_HPP



#include "types.hpp"
#include "Direct_File.hpp"

#include <algorithm>


namespace gpu_saca
{

// Size of a message.
constexpr inline std::size_t MESSAGE_BYTES = sizeof(idx_t) + sizeof(msg_rank_t);

// Size-granularity at which a destination-partition's message-group starts;
// required for `O_DIRECT` alignment purposes.
constexpr inline std::size_t MSG_ALIGN = Direct_File::DISK_BLOCK / std::min<idx_t>(sizeof(idx_t), sizeof(msg_rank_t));

// Returns the inbox-parity a generation with index `gen` writes to.
static constexpr inline std::size_t write_parity(const idx_t gen) { return (gen & 1) ^ 1; }

// Returns the inbox-parity a generation with index `gen` reads from.
static constexpr inline std::size_t read_parity(const idx_t gen) { return gen & 1; }

}



#endif
