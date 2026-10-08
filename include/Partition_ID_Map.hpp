
#ifndef PARTITION_ID_MAP_HPP
#define PARTITION_ID_MAP_HPP



#include "types.hpp"
#include "Direct_File.hpp"
#include "utility.hpp"
#include "cross-compile.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <algorithm>


namespace gpu_saca
{

// Partition-ID map `Π` of a text: the ID of the partition holding each suffix,
// and whether the suffix has retired, bit-packed.
class Partition_ID_Map
{
public:

    static constexpr idx_t GROUP_SZ = 64;   // Suffixes per group: their entries take up as many words as bits per entry.
                                            // Entries of `GROUP_SZ` consecutive suffixes take up whole words.

    static constexpr idx_t DISK_GROUP_SZ = 8 * Direct_File::DISK_BLOCK; // Suffixes per disk-group: their entries take up as many disk-blocks as bits per entry.

private:

    idx_t n_;   // Count of suffixes.
    uint32_t bits_; // Bits per entry: the partition-ID's, then the retirement-flag's.
    uint64_t mask_; // Mask of a partition-ID's bits.
    std::size_t word_c_;    // Count of words in the flat array, including the padding at the end.
    uint64_t* word_;    // The packed entries: `bits_` words per group of suffixes, then a padding.


    // Returns the count of words of a map of `n` suffixes at `bits` bits per
    // entry, including the padding.
    static std::size_t word_c(const idx_t n, const uint32_t bits) { return ((n + GROUP_SZ - 1) / GROUP_SZ) * bits + 1; }

    // Returns the index of the first word of the entries from the `off`'th
    // suffix on, a group's first, at `bits` bits per entry.
    static std::size_t word_of(const idx_t off, const uint32_t bits) { return (off / GROUP_SZ) * bits; }

    // Returns storage for `count` words, page-aligned and page-rounded for
    // direct reads.
    static uint64_t* allocate_words(std::size_t count);

    // Returns the bit-position of the `i`'th suffix's retirement-flag.
    idx_t flag_bit(const idx_t i) const { return i * bits_ + bits_ - 1; }

public:

    // Constructs an empty map.
    Partition_ID_Map();

    // Constructs a map of `n` suffixes across `p` partitions, unpopulated.
    Partition_ID_Map(idx_t n, p_id_t p);

    // Constructs a map of `n` suffixes holding their retirement-flags alone,
    // none marked: their partition-IDs read as `0`.
    explicit Partition_ID_Map(idx_t n);

    Partition_ID_Map(const Partition_ID_Map&) = delete;
    Partition_ID_Map& operator=(const Partition_ID_Map&) = delete;

    Partition_ID_Map(Partition_ID_Map&& rhs);
    Partition_ID_Map& operator=(Partition_ID_Map&& rhs);

    ~Partition_ID_Map();

    // Returns the bits per entry across `p` partitions: a partition-ID's, and
    // the retirement-flag's.
    static uint32_t bits_for(const p_id_t p) { return std::max<uint32_t>(1, util::math::bits_for(p)) + 1; }

    // Returns the bytes of a map of `n` suffixes across `p` partitions.
    static std::size_t bytes(const idx_t n, const p_id_t p) { return word_c(n, bits_for(p)) * sizeof(uint64_t); }

    // Returns the count of words that the entries of `sz` consecutive suffixes
    // take up, packed at `bits` bits per entry from a group's start.
    CUDA_SPECIFIER static std::size_t packed_word_c(const idx_t sz, const uint32_t bits) { return (sz * bits + 63) / 64; }

    // Packs the partition-IDs `src` of `sz` consecutive suffixes, from a
    // group's start, into the words `dst` at `bits` bits per entry, with the
    // retirement-flags clear, in the map's layout.
    CUDA_SPECIFIER static void pack(const p_id_t* src, idx_t sz, uint32_t bits, uint64_t* dst);

    // Returns the bits per entry.
    uint32_t bits() const { return bits_; }

    // Returns the partition-ID of the `i`'th suffix.
    p_id_t operator[](idx_t i) const;

    // Returns whether the `i`'th suffix is marked retired.
    bool retired(idx_t i) const;

    // Marks the `i`'th suffix retired. Thread-safe.
    void retire(idx_t i);

    // Marks retired the suffixes that the partition-ID map `flags`, of as many
    // suffixes, marks retired.
    void retire(const Partition_ID_Map& flags);

    // Prefetches the `i`'th suffix's retirement-flag, for marking it.
    void prefetch_flag(const idx_t i) const { __builtin_prefetch(word_ + flag_bit(i) / 64, 1, 0); }

    // Returns the count of the suffixes marked retired.
    idx_t retired_count() const;

    // Stores the partition-IDs of the `sz` suffixes from the `off`'th one on,
    // packed per `pack` in `src`. Both `off` and `off + sz` must be multiples
    // of `GROUP_SZ`, unless the latter is the count of suffixes. Thread-safe
    // for disjoint ranges.
    void store_packed(idx_t off, idx_t sz, const uint64_t* src);

    // Writes the partition-IDs of the `sz` suffixes from the `off`'th one on,
    // packed per `pack` at `bits` bits per entry in `src`, to their place in the
    // file `f` of a map's words. `off` must be a multiple of `DISK_GROUP_SZ`,
    // and `src` disk-block aligned, with room for its words rounded up to a
    // disk-block; it is zeroed past its words. Thread-safe for disjoint ranges.
    static void write_packed(Direct_File& f, idx_t off, idx_t sz, uint32_t bits, uint64_t* src);

    // Reads into the entries of the map's first `sz` suffixes those of the `sz`
    // suffixes from the `off`'th one on, from the file `f` of the words of a map
    // at the same width. `off` must be a multiple of `DISK_GROUP_SZ`.
    void read(const Direct_File& f, idx_t off, idx_t sz);

    // Reads the map's words from the file at path `path`, with `thread_c`
    // threads.
    void read(const std::string& path, std::size_t thread_c);

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
    // Relaxed atomic loads of whole words, as the retirement-flags in them may change concurrently; the IDs never do.
    const auto bit = i * bits_;
    const auto off = bit % 64;
    const uint64_t lo = __atomic_load_n(word_ + bit / 64, __ATOMIC_RELAXED) >> off;
    const uint64_t hi = (off + bits_ - 1 > 64 ? __atomic_load_n(word_ + bit / 64 + 1, __ATOMIC_RELAXED) << (64 - off) : 0); // The ID straddles two words.

    return static_cast<p_id_t>((lo | hi) & mask_);
}


inline bool Partition_ID_Map::retired(const idx_t i) const
{
    const auto bit = flag_bit(i);
    return (__atomic_load_n(word_ + bit / 64, __ATOMIC_RELAXED) >> (bit % 64)) & 1;
}


inline void Partition_ID_Map::retire(const idx_t i)
{
    const auto bit = flag_bit(i);
    __atomic_fetch_or(word_ + bit / 64, uint64_t(1) << (bit % 64), __ATOMIC_RELAXED);
}

}



#endif
