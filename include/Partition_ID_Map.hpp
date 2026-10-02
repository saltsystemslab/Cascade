
#ifndef PARTITION_ID_MAP_HPP
#define PARTITION_ID_MAP_HPP



#include "types.hpp"
#include "utility.hpp"
#include "cross-compile.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
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

    // Returns the count of words that the partition-IDs of `sz` consecutive
    // suffixes take up, packed at `bits` bits per ID from a group's start.
    CUDA_SPECIFIER static std::size_t packed_word_c(const idx_t sz, const uint32_t bits) { return (sz * bits + 63) / 64; }

    // Packs the partition-IDs `src` of `sz` consecutive suffixes, from a
    // group's start, into the words `dst` at `bits` bits per ID, in the map's
    // layout.
    CUDA_SPECIFIER static void pack(const p_id_t* src, idx_t sz, uint32_t bits, uint64_t* dst);

    // Returns the bits per partition-ID.
    uint32_t bits() const { return bits_; }

    // Returns the partition-ID of the `i`'th suffix.
    p_id_t operator[](idx_t i) const;

    // Stores the partition-IDs of the `sz` suffixes from the `off`'th one on,
    // packed per `pack` in `src`. Both `off` and `off + sz` must be multiples
    // of `GROUP_SZ`, unless the latter is the count of suffixes. Thread-safe
    // for disjoint ranges.
    void store_packed(idx_t off, idx_t sz, const uint64_t* src);

    // Prepages the map.
    void prepage();

    // Returns the bytes of the map.
    std::size_t bytes() const { return word_c_ * sizeof(uint64_t); }
};


CUDA_SPECIFIER inline void Partition_ID_Map::pack(const p_id_t* const src, const idx_t sz, const uint32_t bits, uint64_t* dst)
{
    // The IDs fill the words least-significant bits first; an ID straddling two words leaves its high bits in the next.
    uint64_t acc = 0;   // Bits of the word being filled.
    uint32_t fill = 0;  // Count of the bits filled.
    for(idx_t i = 0; i < sz; ++i)
    {
        const uint64_t id = src[i];
        acc |= id << fill;
        fill += bits;
        if(fill >= 64)
        {
            *dst++ = acc;
            fill -= 64;
            acc = (fill > 0 ? id >> (bits - fill) : 0);
        }
    }

    if(fill > 0)    // The last word, partially filled.
        *dst = acc;
}


inline p_id_t Partition_ID_Map::operator[](const idx_t i) const
{
    const auto bit = i * bits_;
    uint64_t w;
    std::memcpy(&w, reinterpret_cast<const char*>(word_) + bit / 8, sizeof(w)); // Unaligned: the ID's bits start within its first byte.

    return static_cast<p_id_t>((w >> (bit % 8)) & mask_);
}

}



#endif
