
#ifndef GPU_SACA_HPP
#define GPU_SACA_HPP


#include "types.hpp"
#include "utility.cuh"

#include <cstdint>
#include <cstddef>
#include <limits>


// TODO: name algorithm.
namespace gpu_saca
{


constexpr p_id_t MAX_PART_COUNT = 1024; // Maximum number of allowed partitions. TODO: better fix / tune it.
extern __constant__ pivot_t pivots_d[];


// The suffix array constructor in GPUs for texts with symbols of type `T_sym_`.
// TODO:
//      1. Tune the number of partitions.
//      2. Batch out data to host memory; currently everything will be in device memory.
template <typename T_sym_>
class GPU_SACA
{
private:

    static constexpr double MEM_SAFETY_MARGIN = 0.1;    // Safety margin to spare some global memory for CUDA runtime overhead and possible fragmentation.

    T_sym_* T_d_;   // The text in device.
    const idx_t n_; // Size of the text.
    const T_sym_* const T_h_;   // Text at the host-side; required for async transfers and testing.

    const p_id_t p_;    // Number of partitions for the SA.

    static constexpr std::size_t K = 12;    // k-gram size for frequency counting of prefixes.

    std::vector<pivot_t> pivots;    // The pivots to break the SA into partitions. The pivots are inclusive upper bounds of their partition.

    std::vector<p_id_t> P_id;   // Partition-IDs of the suffixes.

    cudaStream_t stream[2]; // CUDA streams for double-buffered overlap of PCIe transfers and kernel executions.

    T_sym_* null_d; // Device-side buffer of `K - 1` null / terminator symbols, used to pad the tail of text chunks when required.


    // Selects the set of pivots for the partitioning of the suffixes.
    void select_pivots();

    // Computes the partition-IDs of the suffixes.
    void compute_partition_IDs();


public:

    // Constructs a GPU-SACA object for the input text `T` of size `n`.
    GPU_SACA(const T_sym_* T, idx_t n);

    ~GPU_SACA();

    // Returns `1` if `x` is in the upper half of `T_sym_`'s range, `0` otherwise.
    __device__ __host__
    static uint8_t quantize(T_sym_ x);

    // Builds the suffix array into `SA`.
    void build_SA(idx_t* SA);
};


template <typename T_sym_>
__device__ __host__
__forceinline__ uint8_t GPU_SACA<T_sym_>::quantize(const T_sym_ x)
{
    // Maximum element of the lower half: exactly `2^(N - 1)` values at each half.
    constexpr T_sym_ mid = std::numeric_limits<T_sym_>::min() / 2 + std::numeric_limits<T_sym_>::max() / 2;
    return x <= mid ? 0 : 1;
}


// Adds the histogram of the quantized `K`-grams of the text-chunk `T_chunk`, of
// symbols of type `T_sym_` and having `chunk_sz` many `K`-grams, into `H`. The
// chunk must be ensured to have `chunk_sz + K - 1` symbols with enough padding.
// Grid organization: one thread per `K`-gram of the chunk.
// Shared memory layout: quantized text-tile | block-local histogram | padding for CAS-safety
template <typename T_sym_, std::size_t K, std::size_t BLOCK_SZ>
__global__
void compute_kgram_histogram_kernel(const T_sym_* const __restrict__ T_chunk, const idx_t chunk_sz, freq_cnt_t* const __restrict__ H)
{
    constexpr auto HIST_SZ = 1llu << K; // Number of entries in the histogram.
    constexpr auto TILE_SZ = BLOCK_SZ + K - 1;  // Size of the text-tile processed by a block.

    extern __shared__ uint8_t s_mem[];
    uint8_t* tile       = s_mem;    // Quantized text-tile.
    uint8_t* local_hist = s_mem + TILE_SZ;  // The local histogram.
    static_assert(BLOCK_SZ < 256, "Block-size for building the histogram must not exceed 255 to avoid u8 block-local counter overflows in the worst-case.");

    const auto t_id = threadIdx.x;
    const idx_t block_off = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ;  // Offset of this block into the text-chunk.

    // Clear the block-local histogram cooperatively.
    {
        for(std::size_t j = t_id; j < HIST_SZ; j += BLOCK_SZ)
            local_hist[j] = 0;
    }

    // Load the corresponding text-tile cooperatively, quantizing it on the fly.
    {
        auto off = block_off + t_id;    // Offset into the text-chunk to load.
        tile[t_id] = off < chunk_sz ?
                        GPU_SACA<T_sym_>::quantize(T_chunk[off]) :
                        0u;

        if(BLOCK_SZ + t_id < TILE_SZ)   // Process the tile-segment of size `K - 1` overhanging from the block. The overhang has been ensured to exist.
        {
            off = block_off + BLOCK_SZ + t_id;
            tile[BLOCK_SZ + t_id] = off < chunk_sz + K - 1 ?
                                        GPU_SACA<T_sym_>::quantize(T_chunk[off]) :
                                        0u;
        }
    }

    __syncthreads();

    // Pack each k-gram and increment its frequency.
    {
        const auto off = block_off + t_id;  // Offset into the text-chunk being processed.
        if(off < chunk_sz)
        {
            kgram_t quant_kgram = 0;    // Quantized k-gram at chunk-offset `off`.

            #pragma unroll
            for(std::size_t i = 0; i < K; ++i)
                quant_kgram |= (static_cast<kgram_t>(tile[t_id + i]) << (K - 1 - i));   // Note the endianness of the quantization.

            // Atomic-add is not natively supported for `u8`s. Emulate via atomic-CAS on the enclosing 32-bit word.
            atomic_add_u8(local_hist + quant_kgram, 1u);
        }
    }

    __syncthreads();

    // Flush block-local histogram to global histogram cooperatively.
    for(std::size_t j = t_id; j < HIST_SZ; j += BLOCK_SZ)
        if(local_hist[j])
            atomicAdd(reinterpret_cast<unsigned long long*>(H + j), local_hist[j]);
}


// Computes the partition-IDs (in `[0, part_count)`) of the `K`-grams of the
// text-chunk `T`, of type `T_sym_` and having `chunk_sz` many `K-grams, into
// `P_id`.
// Grid organization: one thread per `K`-gram of the chunk.
// Shared memory layout: quantized text-tile
template <typename T_sym_, std::size_t K, std::size_t BLOCK_SZ>
__global__
void compute_partition_IDs_kernel(const T_sym_* const __restrict__ T_chunk, const idx_t chunk_sz, p_id_t* const __restrict__ P_id, const p_id_t part_count)
{
    constexpr auto TILE_SZ = BLOCK_SZ + K - 1;  // Size of the text-tile processed by a block.

    // Shared memory pattern: quantized text-tile of size `TILE_SZ`.
    extern __shared__ uint8_t s_mem[];
    uint8_t* tile = s_mem;  // Quantized text-tile.

    const auto t_id = threadIdx.x;
    const idx_t block_off = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ;  // Offset of this block into the text-chunk.

    // Load the corresponding text-tile cooperatively, quantizing it on the fly.
    {
        auto off = block_off + t_id;    // Offset into the text-chunk to load.
        tile[t_id] = off < chunk_sz ?
                        GPU_SACA<T_sym_>::quantize(T_chunk[off]) :
                        0u;

        if(BLOCK_SZ + t_id < TILE_SZ)   // Process the tile-segment of size `K - 1` overhanging from the block.
        {
            off = block_off + BLOCK_SZ + t_id;
            tile[BLOCK_SZ + t_id] = off < chunk_sz + K - 1 ?
                                        GPU_SACA<T_sym_>::quantize(T_chunk[off]) :
                                        0u;
        }
    }

    __syncthreads();

    // Pack each k-gram and compute its partition-ID.
    {
        const auto off = block_off + t_id;  // Offset into the text-chunk being processed.
        if(off < chunk_sz)
        {
            kgram_t quant_kgram = 0;    // Quantized k-gram at text-chunk offset `off`.

            #pragma unroll
            for(uint16_t i = 0; i < K; ++i)
                quant_kgram |= (static_cast<kgram_t>(tile[t_id + i]) << (K - 1 - i));   // Note the endianness of the quantization.

            // Binary search for `quant_kgram` into the pivots for the smallest `p_id` such that `quant_kgram <= pivots[p_id]`.
            p_id_t low = 0, high = part_count - 1;  // The solution lies in the range `[low, high]`.
            while(low < high)
            {
                const auto mid = low + (high - low) / 2;
                if(quant_kgram <= pivots_d[mid])    // The solution lies in the range `[low, mid]`.
                    high = mid;
                else    // It lies in the range `[mid + 1, high]`.
                    low = mid + 1;
            }

            P_id[off] = low;
        }
    }
}


}



#endif
