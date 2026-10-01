
#ifndef PARTITION_ID_MAP_HPP
#define PARTITION_ID_MAP_HPP



#include "types.hpp"
#include "utility.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>
#include <algorithm>


namespace gpu_saca
{

// Partition-ID map `Π` of a text: the ID of the partition holding each suffix,
// bit-packed.
class Partition_ID_Map
{
public:

    static constexpr idx_t GROUP_SZ = 64;   // Suffixes per group: their IDs take up as many words as bits per ID.
                                            // IDs of `GROUP_SZ` consecutive suffixes take up whole words.

private:

    idx_t n_;   // Count of suffixes.
    uint32_t bits_; // Bits per partition-ID.
    uint64_t mask_; // Mask of a partition-ID's bits.
    std::size_t word_c_;    // Count of words in the flat array, including the padding at the end.
    uint64_t* word_;    // The packed partition-IDs: `bits_` words per group of suffixes, then a padding.


    // Returns the count of words of a map of `n` suffixes at `bits` bits per ID,
    // including the padding.
    static std::size_t word_c(const idx_t n, const uint32_t bits) { return ((n + GROUP_SZ - 1) / GROUP_SZ) * bits + 1; }

public:

    // Constructs an empty map.
    Partition_ID_Map();

    // Constructs a map of `n` suffixes across `p` partitions, unpopulated.
    Partition_ID_Map(idx_t n, p_id_t p);

    Partition_ID_Map(const Partition_ID_Map&) = delete;
    Partition_ID_Map& operator=(const Partition_ID_Map&) = delete;

    Partition_ID_Map(Partition_ID_Map&& rhs);
    Partition_ID_Map& operator=(Partition_ID_Map&& rhs);

    ~Partition_ID_Map();

    // Returns the bits per partition-ID across `p` partitions.
    static uint32_t bits_for(const p_id_t p) { return std::max<uint32_t>(1, util::math::bits_for(p)); }

    // Returns the bytes of a map of `n` suffixes across `p` partitions.
    static std::size_t bytes(const idx_t n, const p_id_t p) { return word_c(n, bits_for(p)) * sizeof(uint64_t); }

    // Returns the partition-ID of the `i`'th suffix.
    p_id_t operator[](idx_t i) const;

    // Stores the partition-IDs `src` of the `sz` suffixes from the `off`'th one
    // on. Both `off` and `off + sz` must be multiples of `GROUP_SZ`, unless the
    // latter is the count of suffixes. Thread-safe for disjoint ranges.
    void store(idx_t off, idx_t sz, const p_id_t* src);

    // Prepages the map.
    void prepage();

    // Returns the bytes of the map.
    std::size_t bytes() const { return word_c_ * sizeof(uint64_t); }
};


inline p_id_t Partition_ID_Map::operator[](const idx_t i) const
{
    const auto bit = i * bits_;
    uint64_t w;
    std::memcpy(&w, reinterpret_cast<const char*>(word_) + bit / 8, sizeof(w)); // Unaligned: the ID's bits start within its first byte.

    return static_cast<p_id_t>((w >> (bit % 8)) & mask_);
}

}



#endif
