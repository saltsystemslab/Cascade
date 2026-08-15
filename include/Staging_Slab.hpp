
#ifndef STAGING_SLAB_HPP
#define STAGING_SLAB_HPP



#include "types.hpp"
#include "Packed_Prefix.hpp"

#include <atomic>
#include <vector>


namespace gpu_saca
{

// Staging slab to hold a chunk of a partition's data (keys and suffixes),
// bridging the partition storage backend and the device through host-pinned
// space.
struct Staging_Slab
{
private:

    Prefix_Key* key_;   // Keys of the partition-chunk; typed to the largest-width type used in sort (which is in initial-ranking).
    idx_t* suf_;    // Suffixes of the partition-chunk.
    p_id_t* beta_;  // Backward routing tags of the partition-chunk.
    p_id_t p_id_;   // ID of the partition this slab's content belongs to.
    idx_t off_; // Offset of the content of this slab within the corresponding partition.
    idx_t sz_;  // Number of elements in the slab.

public:

    // Constructs a null slab.
    Staging_Slab();

    ~Staging_Slab();

    // Returns the keys of the staged slab: typed to the largest-width
    // type used as keys.
    auto key() const { return key_; }

    // Returns the keys of the staged slab: typed to the largest-width
    // type used as keys.
    auto key() { return key_; }

    // Returns the suffixes of the staged slab.
    auto suf() const { return suf_; }

    // Returns the suffixes of the staged slab.
    auto suf() { return suf_; }

    // Returns the backward routing tags of the staged slab.
    auto beta() const { return beta_; }

    // Returns the backward routing tags of the staged slab.
    auto beta() { return beta_; }

    // Returns the ID of the partition this slab's content belongs to.
    auto p_id() const { return p_id_; }

    // Returns the offset of the content of this slab within the corresponding
    // partition.
    auto off() const { return off_; }

    // Returns the number of elements in the slab.
    auto size() const { return sz_; }

    // Allocates the slab for a `sz`-sized partition. Must be invoked before
    // accessing the slab.
    void allocate(idx_t sz);

    // Tags the slab with the partition-chunk it's carrying (from partition-ID
    // `p_id`, at its offset `off`, with `sz`-many elements).
    void assign(p_id_t p_id, idx_t off, idx_t sz);
};


// Assembly / chain of slabs making up partition.
struct Partition_Assembly
{
    std::vector<Staging_Slab*> slab;    // Collection of the slabs containing the partition's content.
    std::atomic<idx_t> chunks_to_read;  // Count of chunks still remaining to be read for the partition.
};

}



#endif
