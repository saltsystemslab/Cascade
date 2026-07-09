
#ifndef GPU_SACA_CUH
#define GPU_SACA_CUH


#include "types.hpp"
#include "Prefix_Key.cuh"
#include "Partition.hpp"
#include "Sort_Key.hpp"
#include "utility.cuh"

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <vector>
#include <limits>


// TODO: name algorithm.
namespace gpu_saca
{

constexpr p_id_t MAX_PIVOT_COUNT = CONST_MEM_SZ / sizeof(Prefix_Key);   // Maximum supported number of pivots.
extern __constant__ Prefix_Key pivots_d[];


// The suffix array constructor in GPUs for texts with symbols of type `T_sym_`.
template <typename T_sym_>
class GPU_SACA
{
private:

    static constexpr double MEM_SAFETY_MARGIN = 0.1;    // Safety margin to spare some global memory for CUDA runtime overhead and possible fragmentation.

    // Assumed worst-case ratio of actual max partition-size to the target
    // estimated during pivots-selection.
    static constexpr double PART_SZ_IMBALANCE_SLACK = 1.20;

    const idx_t n_; // Size of the text.
    const T_sym_* const T_h_;   // Text at the host-side.

    p_id_t p_;  // Number of partitions for the SA.

    static constexpr idx_t K = Prefix_Key::PREFIX_WORDS * sizeof(idx_t) / sizeof(T_sym_);   // Prefix-size for pivots and partitioning.

    std::vector<Prefix_Key> pivots; // The pivots to break the SA into partitions. The pivots are inclusive upper bounds of their partition.

    p_id_t* P_id;   // Partition-IDs of the suffixes.

    std::vector<Partition> P_;  // The suffix partitions.
    std::vector<idx_t> P_sz;    // Sizes of the suffix partitions; tracks the active-set sizes of the partitions instead of the full size.
    std::vector<idx_t> P_off;   // Exclusive cumulative size of the partitions.

    idx_t* rho; // Global rank of the suffixes.

    std::atomic<idx_t> unsorted; // Count of suffixes that have not yet received their final rank.

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

        uint8_t* is_active_buf[STREAM_C];   // Activeness-marker buffers for suffixes, per stream.
        idx_t* active_set_sz_d; // Sizes of the new active-sets of the partitions.
    // }

#ifdef TESTS
    Sort_Key* sorted_key = nullptr; // Single host-pinned buffer to save the order of the sorted keys of all partitions, before the new ranks overwrite those.
    idx_t* sorted_suf = nullptr;    // Single host-pinned buffer to save the order of the suffix indices corresponding to sorted keys of all partitions, before they get shuffled.
#endif

    // Selects the set of pivots for the partitioning of the suffixes.
    void select_pivots();

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

    // Sorts and initially ranks the `p_id`'th partition, and filters the
    // entries in the partition with unique ranks by grouping them into the tail
    // of the partition.
    void sort_and_init_rank_and_filter(p_id_t p_id);

    // Sorts the `p_id`'th partition, re-ranks it based on the sorted order, and
    // filters the entries in the partition with unique ranks by grouping them
    // into the tail of the partition.
    void sort_and_re_rank_and_filter(p_id_t p_id);

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

    // Builds the suffix array into `SA`.
    void build_SA(idx_t*& SA);
};


// Computes the partition-IDs (in `[0, part_count)`) of the suffixes of the
// text-chunk `T_chunk`, of type `T_sym_` and having `chunk_sz` many suffixes,
// into `P_id`. Also accumulates the per-partition suffix counts of this chunk
// into the global histogram `P_sz`.
// Grid organization: one thread per suffix of the chunk.
// Shared memory layout: text-tile (`T_sym_`) | block-local histogram (`uint32_t`)
template <typename T_sym_, idx_t K, idx_t BLOCK_SZ>
__global__
void compute_partition_IDs_kernel(  const T_sym_* const __restrict__ T_chunk, const idx_t chunk_sz,
                                    p_id_t* const __restrict__ P_id, const p_id_t part_count, idx_t* const __restrict__ P_sz)
{
    constexpr auto TILE_SZ = BLOCK_SZ + K - 1;  // Size of the text-tile processed by a block.

    // Shared memory pattern: text-tile of size `TILE_SZ`, followed by block-local histogram of size `part_count`.
    extern __shared__ uint8_t s_mem[];
    T_sym_* const tile         = reinterpret_cast<T_sym_*>(s_mem);  // Text-tile.
    constexpr idx_t tile_bytes = TILE_SZ * sizeof(T_sym_);
    constexpr idx_t hist_off   = ((tile_bytes + alignof(uint32_t) - 1) / alignof(uint32_t)) * alignof(uint32_t);
    uint32_t* const local_hist = reinterpret_cast<uint32_t*>(s_mem + hist_off);

    const auto t_id = threadIdx.x;
    const idx_t block_off = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ;  // Offset of this block into the text-chunk.

    // Clear the block-local histogram cooperatively.
    {
        for(idx_t j = t_id; j < part_count; j += BLOCK_SZ)
            local_hist[j] = 0;
    }

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

    // Pack each suffix's `K`-prefix and compute its partition-ID, and count it.
    {
        const auto off = block_off + t_id;  // Offset into the text-chunk being processed.
        if(off < chunk_sz)
        {
            const auto pre_key = Prefix_Key::pack_prefix(tile + t_id);

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

            // Count this suffix into the block-local histogram.
            atomicAdd(local_hist + P_id[off], 1u);
        }
    }

    __syncthreads();

    // Flush the block-local counts into the global-local histogram.
    {
        for(idx_t j = t_id; j < part_count; j += BLOCK_SZ)
            if(local_hist[j])
                atomicAdd(reinterpret_cast<unsigned long long*>(P_sz + j), static_cast<unsigned long long>(local_hist[j]));
    }
}


// Initializes the ranks of the distinct sort-key groups from the sorted array
// `K` (of size `n`) into the 64-bit rank array `rank_u64` (of size `2 * n`).
// Also computes the sort-keys' activeness-flags (for future) into `is_active`.
// The first key of a group is ranked by its offset into `K`, whereas the rest
// of the keys in the group are ranked `0`. `K` and `rank_u64` will be byte-
// aligned: the `i`'th sort-key from `K` has its rank at the `2 * i`'th index of
// `rank_u64`—thus the odd indices in `rank_u64` will be 0-holes.
// Grid organization: one thread per sort-key.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void initialize_key_group_ranks_and_mark_active_flags_kernel(   const Sort_Key* const __restrict__ K, const idx_t n,
                                                                idx_t* const __restrict__ rank_u64,
                                                                uint8_t* const __restrict__ is_active)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    rank_u64[2 * i + 1] = 0;    // Hole in the flag array corresponding to the `i`'th sort-key.
    if(i == 0)
        rank_u64[0] = 0;    // First group always starts at offset `0`.
    else
        rank_u64[2 * i] = (K[i].primary != K[i - 1].primary || K[i].secondary != K[i - 1].secondary ? i : 0);

    const bool pre_tie = (i == 0        ? false : K[i].primary == K[i - 1].primary && K[i].secondary == K[i - 1].secondary);
    const bool suc_tie = (i == n - 1    ? false : K[i].primary == K[i + 1].primary && K[i].secondary == K[i + 1].secondary);
    is_active[i] = (pre_tie || suc_tie);
}


// Ranks the distinct sort-key groups and primary-key groups from the sorted
// array `K` (of size `n`) into `key_group_rank` and `primary_key_group_rank`,
// respectively. Also computes the sort-keys' activeness-flags (for future) into
// `is_active`. The first entry of a group is ranked by its offset into `K`,
// whereas the rest of the entries in the group are ranked `0`.
// Grid organization: one thread per sort-key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void rank_key_groups_and_primary_key_groups_and_mark_active_flags_kernel(   const Sort_Key* const __restrict__ K, const idx_t n,
                                                                            idx_t* const __restrict__ key_group_rank, idx_t* const __restrict__ primary_key_group_rank,
                                                                            uint8_t* const __restrict__ is_active)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    if(i == 0)
    {
        // First group always starts at offset `0`.
        key_group_rank[0] = 0;
        primary_key_group_rank[0] = 0;
    }
    else
    {
        const bool key_tie = (K[i].primary == K[i - 1].primary && K[i].secondary == K[i - 1].secondary);
        const bool primary_key_tie = (K[i].primary == K[i - 1].primary);
        key_group_rank[i] = (!key_tie ? i : 0);
        primary_key_group_rank[i] = (!primary_key_tie ? i : 0);
    }

    const bool pre_tie = (i == 0        ? false : K[i].primary == K[i - 1].primary && K[i].secondary == K[i - 1].secondary);
    const bool suc_tie = (i == n - 1    ? false : K[i].primary == K[i + 1].primary && K[i].secondary == K[i + 1].secondary);
    is_active[i] = (pre_tie || suc_tie);
}


// Refines the primary-keys of the sorted array `K` (of size `n`) based on its
// key groups' ranks `key_group_rank` and its primary-key groups' ranks
// `primary_key_group_rank`.
// Grid organization: one thread per sort-key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void refine_primary_keys_kernel(Sort_Key* const __restrict__ K, const idx_t n,
                                const idx_t* const __restrict__ key_group_rank, const idx_t* const __restrict__ primary_key_group_rank)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    K[i].primary += (key_group_rank[i] - primary_key_group_rank[i]);
}

}



#endif
