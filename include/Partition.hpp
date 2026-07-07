
#ifndef PARTITION_HPP
#define PARTITION_HPP



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
    const idx_t cap_;   // Capacity of the partition.
    idx_t sz_;  // Current size of the partition.

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

    // Returns pointer to the (immutable) suffix data of the partition.
    auto suf() const { return suf_; }

    // Returns pointer to the (immutable) sort-key data of the partition.
    auto key() const { return key_; }

    // Returns the capacity of the partition.
    auto capacity() const { return cap_; }

    // Returns the current size of the partition.
    auto size() const { return sz_; }

    // Returns the `i`'th suffix in the partition.
    auto suf(const idx_t i) const { return suf_[i]; }

    // Returns the `i`'th sort-key in the partition.
    auto key(const idx_t i) const { return key_[i]; }

    // Returns a mutable reference to `i`'th sort-key in the partition.
    auto& key(const idx_t i) { return key_[i]; }
};

}



#endif