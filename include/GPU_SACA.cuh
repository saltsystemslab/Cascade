
#ifndef GPU_SACA_CUH
#define GPU_SACA_CUH


#include "types.hpp"
#include "Prefix_Key.cuh"
#include "Partition.hpp"
#include "Sort_Key.hpp"
#include "utility.cuh"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <limits>


// TODO: name algorithm.
namespace gpu_saca
{

constexpr p_id_t MAX_PIVOT_COUNT = CONST_MEM_SZ / sizeof(Prefix_Key);   // Maximum supported number of pivots.
extern __constant__ Prefix_Key pivots_d[];


// The suffix array constructor in GPUs for texts with symbols of type `T_sym_`.
// TODO:
//      1. Tune the number of partitions.
template <typename T_sym_>
class GPU_SACA
{
private:

    static constexpr double MEM_SAFETY_MARGIN = 0.1;    // Safety margin to spare some global memory for CUDA runtime overhead and possible fragmentation.

    const idx_t n_; // Size of the text.
    const T_sym_* const T_h_;   // Text at the host-side.

    p_id_t p_;  // Number of partitions for the SA.

    static constexpr idx_t K = Prefix_Key::PREFIX_WORDS * sizeof(idx_t) / sizeof(T_sym_);   // Prefix-size for pivots and partitioning.

    std::vector<Prefix_Key> pivots; // The pivots to break the SA into partitions. The pivots are inclusive upper bounds of their partition.

    p_id_t* P_id;   // Partition-IDs of the suffixes.

    std::vector<Partition> P_;  // The suffix partitions.
    std::vector<idx_t> P_sz;    // Sizes of the suffix partitions.
    std::vector<idx_t> P_off;   // Exclusive cumulative size of the partitions.

    idx_t* rho; // Global rank of the suffixes.

    static constexpr idx_t STREAM_C = 3;    // Number of CUDA-streams to support multi-buffered overlap of PCIe transfers and kernel executions.
    cudaStream_t stream[STREAM_C];  // CUDA streams for triple-buffered overlap of PCIe transfers and kernel executions.

    // Sampling parameters.
    // {
        static constexpr idx_t OVERSAMPLING_FACTOR = 32;    // Oversampling-factor.
        static constexpr idx_t PARTITION_SLACK_FACTOR = 4;  // Allowed slack factor for actual partition-count over the expected count.
        static constexpr uint64_t SAMPLE_SEED = 0x9E3779B97F4A7C15ull;  // Fixed seed for random sampling, for reproducibility.
    // }

    // Device-side data and related.
    // {
        T_sym_* null_d; // Buffer of `K - 1` null / terminator symbols, used to pad the tail of text chunks when required.

        Sort_Key* key_buf[STREAM_C][2]; // Input and output buffers for sort-keys of suffixes, per stream.
        idx_t* suf_buf[STREAM_C][2];    // Input and output buffers for suffixes, per stream.
        void* workspace_buf[STREAM_C];  // Scratch space for `cub` radix-sort and scan, per stream.
        std::size_t workspace_sz;       // Size of the `cub` workspace.
    // }

#ifdef TESTS
    Sort_Key* sorted_key = nullptr; // Single host-pinned buffer to save the sorted keys of all partitions before the ranks overwrite those.
#endif


    // Returns the `N` symbols situated at the index `idx` of the text as a 64-
    // bit word. The endian-ness of the word matches the lexicographical order
    // of the text—the symbol at index `idx` corresponds to the most significant
    // bits of the word.
    // NB: the text must have been appropriately padded at the tail.
    template <idx_t N = (8 / sizeof(T_sym_))>
    uint64_t pack_word(idx_t idx) const;

    // Selects the set of pivots for the partitioning of the suffixes.
    void select_pivots();

    // Returns the packed prefix-key filled with `K` symbols of the text from
    // the index `idx` onward.
    Prefix_Key pack_prefix(idx_t idx) const;

    // Returns the inclusive upper-bound for the maximum supported partition-
    // size. The rationale used in fixing this is that, in the case when the
    // partitions are large enough such that `STREAM_C = 3` partitions can fully
    // occupy the available device-memory, their sizes must be evenly balanced.
    idx_t partition_size_upper_bound() const;

    // Returns the required `cub` workspace-size in bytes, for sort and scan,
    // for `part_sz` many elements.
    static std::size_t cub_workspace_size(idx_t part_sz);

    // Allocates spaces for the required device data structures; returns `true`
    // iff the device had enough space for such.
    bool allocate_device_space();

    // Partitions the suffixes based on the selected pivots.
    void partition();

    // Ranks the suffixes in their partitions based on the prefixes associated
    // to them during partitioning.
    void initial_rank();

    // Computes the sizes of the suffix partitions based on the quantized `K`-
    // gram histogram `H` and the selected pivots.
    // void compute_partition_sizes(const std::vector<freq_cnt_t>& H);

    // Computes the partition-IDs of the suffixes.
    void compute_partition_IDs();

    // Sorts and ranks the `p_id`'th partition at the device.
    void sort_and_rank(p_id_t p_id);

    // Updates the global ranks of the suffixes based on the current status of
    // the partitions.
    void update_global_ranks();

    // Updates the global ranks of the suffixes in the `p_id`'th partition.
    void update_global_rank(p_id_t p_id);

    // Keeps exponentiating the effective prefix-context of the suffixes until
    // the suffixes are fully sorted.
    void exponentiate_prefix();

    // Exponentiates the effective prefix-context of the suffixes at the `p_id`
    // 'th partition from `k`.
    void exponentiate_prefix(p_id_t p_id, idx_t k);

    // Allocates enough space into `SA` and concatenates the partitions into it.
    void concatenate_partitions(idx_t*& SA);


public:

    // Constructs a GPU-SACA object for the input text `T` of size `n`.
    GPU_SACA(const T_sym_* T, idx_t n);

    ~GPU_SACA();

    // Returns `1` if `x` is in the upper half of `T_sym_`'s range, `0` otherwise.
    __device__ __host__
    static uint8_t quantize(T_sym_ x);

    // Builds the suffix array into `SA`.
    void build_SA(idx_t*& SA);
};


template <typename T_sym_>
__device__ __host__
__forceinline__ uint8_t GPU_SACA<T_sym_>::quantize(const T_sym_ x)
{
    // Maximum element of the lower half: exactly `2^(N - 1)` values at each half.
    constexpr T_sym_ mid = std::numeric_limits<T_sym_>::min() / 2 + std::numeric_limits<T_sym_>::max() / 2;
    return x <= mid ? 0 : 1;
}


template <typename T_sym_>
template <idx_t N>
inline uint64_t GPU_SACA<T_sym_>::pack_word(idx_t idx) const
{
    if constexpr(N == 0)
        return 0;
    else
    {
        constexpr auto shift = (N - 1) * sizeof(T_sym_) * 8;
        return (static_cast<uint64_t>(T_h_[idx]) << shift) | pack_word<N - 1>(idx + 1);
    }
}


template <typename T_sym_>
inline Prefix_Key GPU_SACA<T_sym_>::pack_prefix(const idx_t idx) const
{
    constexpr idx_t sym_per_word = 8 / sizeof(T_sym_);

    Prefix_Key pre_key;

    #pragma unroll
    for(std::size_t i = 0; i < Prefix_Key::PREFIX_WORDS; ++i)
        pre_key.word[i] = pack_word(idx + i * sym_per_word);

    return pre_key;
}


// Computes the partition-IDs (in `[0, part_count)`) of the suffixes of the
// text-chunk `T_chunk`, of type `T_sym_` and having `chunk_sz` many suffixes,
// into `P_id`.
// Grid organization: one thread per suffix of the chunk.
// Shared memory layout: text-tile
template <typename T_sym_, idx_t K, idx_t BLOCK_SZ>
__global__
void compute_partition_IDs_kernel(const T_sym_* const __restrict__ T_chunk, const idx_t chunk_sz, p_id_t* const __restrict__ P_id, const p_id_t part_count)
{
    constexpr auto TILE_SZ = BLOCK_SZ + K - 1;  // Size of the text-tile processed by a block.

    // Shared memory pattern: text-tile of size `TILE_SZ`.
    extern __shared__ uint8_t s_mem[];
    T_sym_* tile = s_mem;   // Text-tile.

    const auto t_id = threadIdx.x;
    const idx_t block_off = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ;  // Offset of this block into the text-chunk.

    // Load the corresponding text-tile cooperatively.
    {
        auto off = block_off + t_id;    // Offset into the text-chunk to load.
        tile[t_id] = off < chunk_sz ?
                        T_chunk[off] : std::numeric_limits<T_sym_>::min();

        if(BLOCK_SZ + t_id < TILE_SZ)   // Process the tile-segment of size `K - 1` overhanging from the block.
        {
            off = block_off + BLOCK_SZ + t_id;
            tile[BLOCK_SZ + t_id] = off < chunk_sz + K - 1 ?
                                        T_chunk[off] : std::numeric_limits<T_sym_>::min();
        }
    }

    __syncthreads();

    // Pack each suffix's `K`-prefix and compute its partition-ID.
    {
        const auto off = block_off + t_id;  // Offset into the text-chunk being processed.
        if(off < chunk_sz)
        {
            constexpr idx_t sym_per_word = sizeof(idx_t) / sizeof(T_sym_);
            Prefix_Key pre_key;

            #pragma unroll
            for(idx_t w = 0; w < Prefix_Key::PREFIX_WORDS; ++w)
            {
                idx_t word = 0;

                #pragma unroll
                for(idx_t j = 0; j < sym_per_word; ++j)
                {
                    const idx_t shift = (sym_per_word - 1 - j) * sizeof(T_sym_) * 8;
                    word |= (static_cast<idx_t>(tile[t_id + (w * sym_per_word + j)]) << shift); // Note the endianness of the quantization.
                }

                pre_key.word[w] = word;
            }


            // Binary search for the smallest `p_id` such that `pre_key <= pivots[p_id]`.
            p_id_t low = 0, high = part_count - 1;  // The solution lies in the range `[low, high]`.
            while(low < high)
            {
                const auto mid = low + (high - low) / 2;
                if(pre_key <= pivots_d[mid])    // The solution lies in the range `[low, mid]`.
                    high = mid;
                else    // It lies in the range `[mid + 1, high]`.
                    low = mid + 1;
            }

            P_id[off] = low;
        }
    }
}


// Flags the distinct sort-keys from the sorted `Sort_Key` array of size `n`,
// with their offsets, into the 64-bit array `off_u64` of size `2 * n`. `K` and
// `off_u64` will be byte-aligned: the `i`'th sort-key from `K` has its offset
// at the `2 * i`'th index of `off_u64`—thus the odd indices in `off_u64` will
// be 0-holes.
// Grid organization: one thread per sort-key.
// Shared memory layout: no usage.
template <std::size_t BLOCK_SZ>
__global__
void flag_distinct_key_offsets_kernel(const Sort_Key* const __restrict__ K, uint64_t* const __restrict__ off_u64, const idx_t n)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    off_u64[2 * i + 1] = 0; // Hole in the flag array corresponding to the `i`'th sort-key.
    if(i == 0)
        off_u64[0] = 0; // First group always starts at offset `0`.
    else
        off_u64[2 * i] = (K[i].primary != K[i - 1].primary || K[i].secondary != K[i - 1].secondary) ? i : 0;
}

}



#endif
