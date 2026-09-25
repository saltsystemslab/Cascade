
#ifndef SA40_HPP
#define SA40_HPP



#include "cross-compile.hpp"
#include "types.hpp"

#include <cstdint>
#include <cstring>


// The query index's suffix array: `5`-byte (40-bit) little-endian token-indices
// per suffix, in the numeric symbol order of the construction.

namespace gpu_saca
{

// Width in bytes of a suffix-array entry in the query index.
constexpr idx_t SA_WIDTH = 5;

// Exclusive upper bound of the text-length addressable by the query index.
constexpr idx_t SA_MAX_N = idx_t(1) << (8 * SA_WIDTH);

// Bytes of zero-padding kept in memory past the text's tail, so that unaligned
// key-loads at the last suffixes stay in-bounds. Correctness never depends on
// the padding's values: comparisons check the suffix-lengths explicitly.
constexpr idx_t TEXT_PAD_BYTES = 64;

// Bytes of padding kept in memory past the suffix array's tail: covers the
// two-word device decoding of the last entry, and rounds windows to 128-byte
// lines.
constexpr idx_t SA_PAD_BYTES = 128;

// Mask of the low `8 * SA_WIDTH` bits of a word.
constexpr idx_t SA_ENTRY_MASK = SA_MAX_N - 1;


namespace sa40
{

// Returns the entry at rank `r` of the 40-bit suffix array `SA`.
// NB: device-side, `SA` must be 4-byte aligned and padded with `SA_PAD_BYTES`.
CUDA_SPECIFIER
inline idx_t load(const uint8_t* const SA, const idx_t r)
{
#ifdef __CUDA_ARCH__
    // Two aligned 32-bit loads cover the 5 bytes at any offset `b % 4 <= 3`.
    const idx_t b = SA_WIDTH * r;
    const uint32_t* const w = reinterpret_cast<const uint32_t*>(SA + (b & ~idx_t(3)));
    const idx_t v = (static_cast<idx_t>(__ldg(w + 1)) << 32) | __ldg(w);
    return (v >> ((b & 3) * 8)) & SA_ENTRY_MASK;
#else
    idx_t v = 0;    // The host is little-endian: the low bytes of `v` receive the entry.
    std::memcpy(&v, SA + SA_WIDTH * r, SA_WIDTH);
    return v;
#endif
}


// Stores `v` as the entry at rank `r` of the 40-bit suffix array `SA`.
inline void store(uint8_t* const SA, const idx_t r, const idx_t v)
{
    std::memcpy(SA + SA_WIDTH * r, &v, SA_WIDTH);
}


// Returns `true` iff the host is little-endian, which the on-disk layouts of the
// query index require.
inline bool host_is_little_endian()
{
    const uint16_t x = 1;
    uint8_t b;
    std::memcpy(&b, &x, 1);
    return b == 1;
}

}

}



#endif
