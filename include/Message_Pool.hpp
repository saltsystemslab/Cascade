
#ifndef MESSAGE_POOL_HPP
#define MESSAGE_POOL_HPP



#include "types.hpp"
#include "lane.hpp"
#include "Spin_Lock.hpp"

#include <cstdint>
#include <cstddef>
#include <limits>
#include <atomic>
#include <vector>


namespace gpu_saca
{

// Pool of fixed-size message-chunks for the RAM-tier of the inboxes, carved from
// a single reservation of address-space. A chunk holds `chunk_cap` messages: all
// their target-lanes, followed by all their ranks. The pool only stores and
// charging the host-budget for the chunks is up to its users.
class Message_Pool
{
public:

    using chunk_t = uint32_t;   // Index of a chunk.

    static constexpr chunk_t NIL = std::numeric_limits<chunk_t>::max(); // Null chunk-index.

private:

    const std::size_t chunk_cap_;   // Messages per chunk.
    const std::size_t chunk_bytes_; // Bytes per chunk.
    const chunk_t chunk_c_; // Count of chunks the reservation holds.
    char* const base_;  // Base of the reservation.

    std::atomic<chunk_t> carved_;   // Count of chunks carved from the reservation so far.
    std::vector<chunk_t> free_; // Free chunks still resident, in LIFO order.
    std::vector<chunk_t> released_; // Free chunks whose memory went back to the kernel: to be taken anew.
    std::atomic<chunk_t> released_c_;   // Size of `released_`, readable outside the lock.
    util::Spin_Lock free_lock_; // Mutually-exclusive lock to `free_` and `released_`.

public:

    // Constructs a pool of chunks of `chunk_cap` messages each, reserving
    // address-space for `max_bytes` bytes worth of chunks.
    Message_Pool(std::size_t chunk_cap, std::size_t max_bytes);

    Message_Pool(const Message_Pool&) = delete;
    Message_Pool& operator=(const Message_Pool&) = delete;

    ~Message_Pool();

    // Returns the count of messages per chunk.
    auto chunk_cap() const { return chunk_cap_; }

    // Returns the count of bytes per chunk.
    auto chunk_bytes() const { return chunk_bytes_; }

    // Returns the count of chunks carved from the reservation so far.
    auto carved() const { return carved_.load(std::memory_order_relaxed); }

    // Returns the count of chunks released, and not taken anew since.
    auto released() const { return released_c_.load(std::memory_order_relaxed); }

    // Returns a free chunk still resident, or `NIL` if there is none.
    chunk_t take_free();

    // Returns a chunk not resident: a released one if any, else one never
    // carved before; `NIL` if the reservation is exhausted. Its memory must
    // have been charged beforehand separately.
    chunk_t take_new();

    // Returns the chunk `c` to the free-list; its memory stays resident.
    void give_back(chunk_t c);

    // Releases the memory of free chunks, until `b` bytes of it at least, or no
    // free chunk left; the released chunks are to be taken anew. Returns the
    // bytes released.
    std::size_t release_free(std::size_t b);

    // Returns the message-target lanes of the chunk `c`.
    uint8_t* tgt(const chunk_t c) const { return reinterpret_cast<uint8_t*>(base_ + c * chunk_bytes_); }

    // Returns the message-ranks of the chunk `c`.
    msg_rank_t* rank(const chunk_t c) const { return reinterpret_cast<msg_rank_t*>(base_ + c * chunk_bytes_ + lane::bytes(chunk_cap_)); }
};

}



#endif
