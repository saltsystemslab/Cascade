
#ifndef PARTITION_CUH
#define PARTITION_CUH



#include "Sort_Key.hpp"
#include "types.hpp"


namespace gpu_saca
{

// A partition of suffixes.
class Partition
{
private:

    idx_t* suf_;    // The suffixes in this partition.
    Sort_Key* key_; // The sort-keys of the suffixes in the partition.
    const idx_t cap;    // Capacity of the partition.

public:

    // Allocates this partition for `cap` many suffixes.
    Partition(idx_t cap);

    Partition(const Partition&) = delete;
    Partition(Partition&&) = default;
    Partition& operator=(const Partition&) = delete;
    Partition& operator=(Partition&&) = delete;

    ~Partition();

    // Returns pointer to the (mutable) suffix data of the partition.
    auto suf() { return suf_; }

    // Returns pointer to the (mutable) sort-key data of the partition.
    auto key() { return key_; }
};

}



#endif