
#ifndef INDEX_LANES_HPP
#define INDEX_LANES_HPP



#include "types.hpp"
#include "Direct_File.hpp"
#include "cross-compile.hpp"

#include <cstdint>
#include <cstddef>
#include <numeric>


namespace gpu_saca
{

// Bytes per index-lane: the width at which the suffix-indices and the message-
// targets are persisted and staged, in host and in disk.
#ifndef IDX_LANE_BYTES
    constexpr std::size_t LANE_BYTES = 8;
#else
    constexpr std::size_t LANE_BYTES = IDX_LANE_BYTES;
#endif

static_assert(LANE_BYTES == 5 || LANE_BYTES == sizeof(idx_t), "Index-lanes must be 5 or 8 bytes wide.");


// Fixed-width, little-endian lanes of indices: index `i` of a lane-array `l`
// takes up its bytes `[i * LANE_BYTES, (i + 1) * LANE_BYTES)`, holding the low
// `lane::BITS` bits of the index.
namespace lane
{

constexpr idx_t BITS = 8 * LANE_BYTES;  // 40 at 5 bytes: indices up to ~1.1e12.

// Count of lanes in the smallest disk-block aligned group of lanes: `4096` at 5
// bytes, `512` at 8.
constexpr idx_t ALIGN = Direct_File::DISK_BLOCK / std::gcd(Direct_File::DISK_BLOCK, LANE_BYTES);


// Returns the bytes of `n` lanes.
CUDA_SPECIFIER
constexpr std::size_t bytes(const idx_t n) { return n * LANE_BYTES; }


// Returns the index held at lane `i` of `l`.
CUDA_SPECIFIER
inline idx_t load(const uint8_t* const l, const idx_t i)
{
    if constexpr(LANE_BYTES == sizeof(idx_t))
        return reinterpret_cast<const idx_t*>(l)[i];    // Lanes are words: lane-arrays are word-aligned.
    else
    {
        idx_t v = 0;
        #pragma unroll
        for(std::size_t b = 0; b < LANE_BYTES; ++b)
            v |= static_cast<idx_t>(l[i * LANE_BYTES + b]) << (8 * b);

        return v;
    }
}


// Stores the low `BITS` bits of index `v` at lane `i` of `l`.
CUDA_SPECIFIER
inline void store(uint8_t* const l, const idx_t i, const idx_t v)
{
    if constexpr(LANE_BYTES == sizeof(idx_t))
        reinterpret_cast<idx_t*>(l)[i] = v; // Lanes are words: lane-arrays are word-aligned.
    else
    {
        #pragma unroll
        for(std::size_t b = 0; b < LANE_BYTES; ++b)
            l[i * LANE_BYTES + b] = static_cast<uint8_t>(v >> (8 * b));
    }
}

}

}



#endif
