
#ifndef STAGING_SLAB_HPP
#define STAGING_SLAB_HPP



#include "types.hpp"
#include "Prefix_Key.cuh"


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
    p_id_t p_id;    // ID of the partition this slab's content belongs to.
    idx_t off;  // Offset of the content of this slab within the corresponding partition.
    idx_t sz;   // Number of elements in the slab.

public:

    // Constructs a null slab.
    Staging_Slab();

    ~Staging_Slab();

    // Returns the keys of the staged partition: typed to the largest-width
    // type used as keys.
    auto key() const { return key_; }

    // Returns the keys of the staged partition: typed to the largest-width
    // type used as keys.
    auto key() { return key_; }

    // Returns the suffixes of the staged partition.
    auto suf() const { return suf_; }

    // Returns the suffixes of the staged partition.
    auto suf() { return suf_; }

    // Allocates the slab for a `sz`-sized partition. Must be invoked before
    // accessing the slab.
    void allocate(idx_t sz);
};

}



#endif
