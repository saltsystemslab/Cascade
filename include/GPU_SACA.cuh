
#ifndef GPU_SACA_CUH
#define GPU_SACA_CUH


#include "types.hpp"
#include "Prefix_Key.hpp"
#include "Partition_Backend.hpp"
#include "Spill_Store.hpp"
#include "Staging_Slab.hpp"
#include "Read_Task.hpp"
#include "utility.cuh"

#include "concurrentqueue/blockingconcurrentqueue.h"

#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>


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

    const idx_t secondary_bits_;    // Bit-width of the secondary-key of a suffix: `\ceil(\log_2(n))`.
    idx_t key_bits_;    // Bit-width of the packed (primary, secondary) key of a suffix — primary: `\ceil(\log_2(P_sz_max))`, secondary: `\ceil(\log_2(n))`.

    static constexpr idx_t K = Prefix_Key::PREFIX_WORDS * sizeof(idx_t) / sizeof(T_sym_);   // Prefix-size for pivots and partitioning.

    std::vector<Prefix_Key> pivots; // The pivots to break the SA into partitions. The pivots are inclusive upper bounds of their partition.

    p_id_t* P_id;   // Partition-IDs of the suffixes.

    std::vector<idx_t> P_cap;   // Full initial sizes of the partitions.
    std::vector<idx_t> Active_sz;   // Active-set sizes of the suffix partitions.
    std::vector<idx_t> P_off;   // Exclusive cumulative size of the partitions.

    idx_t* rho; // Global rank of the suffixes.

    static constexpr idx_t STREAM_C = 3;    // Number of CUDA-streams to support multi-buffered overlap of PCIe transfers and kernel executions.
    cudaStream_t stream[STREAM_C];  // CUDA streams for triple-buffered overlap of PCIe transfers and kernel executions.

    std::unique_ptr<Partition_Backend> backend_;    // Partition storage backend.
    std::unique_ptr<Spill_Store> spill_;    // Partitions' on-disk spill-files manager.

    // Data structures to for the chunked staging of data between the partition-backend and the device buffers.
    // {
        // ~2M elements per chunk / slab: memory usage: 2M x (sizeof(Prefix_Key) + 8) (~48 MB at `ell = 2`).
        static constexpr idx_t CHUNK_CAP = (idx_t(1) << 21);
        static constexpr idx_t PREFETCH_DEPTH = 0;  // Complete partitions to stage ahead for the device-dispatcher.
        static constexpr idx_t WRITEBACK_DEPTH = 0; // Complete partitions to stage ahead to return to the backend.

        idx_t chunk_cap_;   // Chunk granularity (keys / suffixes per slab).
        idx_t slab_c_;  // |S|: slab pool size.
        idx_t reader_c_;    // |r|: reader-threads count.
        idx_t writer_c_;    // |w|: writer-threads count.

        std::vector<Staging_Slab> S;    // The pinned staging-slab pool, bridging partition-backend to device.

        template <typename T> using blocking_q_t = moodycamel::BlockingConcurrentQueue<T>;
        blocking_q_t<Read_Task> RT; // Chunk-read tasks.
        blocking_q_t<Staging_Slab*> F;  // Free slabs.
        blocking_q_t<Partition_Assembly*> R;    // Slabs with chunks read from partition-backend, ready to be uploaded to device.
        blocking_q_t<idx_t> FS; // IDs of the free device-streams.
        blocking_q_t<Staging_Slab*> W;  // Slabs with chunks downloaded from device, ready to be written to partition-backend.

        std::unique_ptr<Partition_Assembly[]> part_assembly;    // Per-partition assembly records (i.e. which chunks assemble them); `p_` of them.


        // Sets the capacity of each staging-slab.
        void set_slab_capacity();

        // Returns the size of the staging-slab pool.
        std::size_t slab_pool_size() const;

        // Returns the total memory used by the slab-pool.
        std::size_t slab_pool_memory() const;

        struct Partition_Download_Callback_Args // Argument bundle for the host-callback at download of a partition's chunks from the device.
        {
            GPU_SACA* const self;
            Partition_Assembly* const p_assembly;   // Assembly / chain of slabs making up the corresponding partition.
            const idx_t s_id;   // ID of the CUDA stream processing the partition.
        };

        // Host-callback once a partition's chunks get downloaded back from the
        // device.
        static void CUDART_CB on_partition_download(void* arg);
    // }

    // Sampling parameters.
    // {
        static constexpr idx_t OVERSAMPLING_FACTOR = 32;    // Oversampling-factor.
        static constexpr idx_t PARTITION_SLACK_FACTOR = 4;  // Allowed slack factor for actual partition-count over the expected count.
        static constexpr uint64_t SAMPLE_SEED = 0x9E3779B97F4A7C15ull;  // Fixed seed for random sampling, for reproducibility.
    // }

    // Device-side data and related.
    // {
        T_sym_* null_d; // Buffer of `K - 1` null / terminator symbols, used to pad the tail of text chunks when required.

        Prefix_Key* key_buf[STREAM_C][2];   // Input and output buffers for keys of suffixes, per stream; typed to the largest-width type used in sort (which is in initial-ranking).
        idx_t* suf_buf[STREAM_C][2];        // Input and output buffers for suffixes, per stream.
        void* workspace_buf[STREAM_C];      // Scratch space for `cub` radix-sort and scan, per stream.
        std::size_t workspace_sz;           // Size of the `cub` workspace.

        uint8_t* is_active_buf[STREAM_C];   // Activeness-marker buffers for suffixes, per stream.
        idx_t* active_set_sz_d; // Sizes of the new active-sets of the partitions.
    // }


    // Returns the fixed space-budget (in bytes) required in the algorithm.
    // Currently: the text (`T`), the partition-IDs (`Π`), the ISA (`ρ`), and
    // the scatter-buffers in partitioning.
    std::size_t fixed_budget() const;

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

    // Sets up the staging slabs and their readers-writers, for chunked staging
    // of data between the partition-backend and the device buffers.
    void setup_staging();

    // Partitions the suffixes based on the selected pivots.
    void partition();

    // Ranks the suffixes in their partitions based on the prefixes associated
    // to them during partitioning.
    void initial_rank();

    // Computes the partition-IDs of the suffixes.
    void compute_partition_IDs();

    // Drives a complete pipeline for all the partitions: chunked reading from
    // the backend, sort-and-rank in the device, and chunked write-back to the
    // backend. `init_rank_` determines whether the pipeline is being executed
    // for the initial-ranking phase.
    template <bool init_rank_> void drive_pipeline();

    // Sorts and initially ranks the `p_id`'th partition, and filters the
    // entries in the partition with unique ranks by grouping them into the tail
    // of the partition. The slab-assembly `p_assembly` contains the partition's
    // content before and after, and the `buf_id`'th device-buffer set is used
    // for device operations.
    void sort_and_init_rank_and_filter(p_id_t p_id, Partition_Assembly& p_assembly, idx_t buf_id);

    // Sorts the `p_id`'th partition, re-ranks it based on the sorted order, and
    // filters the entries in the partition with unique ranks by grouping them
    // into the tail of the partition. The slab-assembly `p_assembly` contains
    // the partition's content before and after, and the `buf_id`'th device-
    // buffer set is used for device operations.
    void sort_and_re_rank_and_filter(p_id_t p_id, Partition_Assembly& p_assembly, idx_t buf_id);

    // Updates the global ranks of the suffixes based on the current status of
    // the partitions.
    void update_global_ranks();

    // Updates the global ranks of the suffixes in the `p_id`'th partition.
    void update_global_rank(p_id_t p_id);

    // Keeps exponentiating the effective prefix-context of the suffixes until
    // the suffixes are fully sorted.
    void exponentiate_prefix();

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


// Initializes the ranks of the distinct prefix-key groups from the sorted array
// `K` (of size `n`) into `rank`, and marks the keys' activeness-flags into
// `is_active`. The first key of a group is ranked by its offset into `K`; the
// rest in the group are ranked `0`.
// Grid organization: one thread per key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void initialize_key_group_ranks_and_mark_active_flags_kernel(   const Prefix_Key* const __restrict__ K, const idx_t n,
                                                                idx_t* const __restrict__ rank,
                                                                uint8_t* const __restrict__ is_active)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    rank[i] = (i == 0 ? 0 : (K[i] != K[i - 1] ? i : 0));

    const bool pre_tie = (i == 0        ? false : K[i] == K[i - 1]);
    const bool suc_tie = (i == n - 1    ? false : K[i] == K[i + 1]);
    is_active[i] = (pre_tie || suc_tie);
}


// Ranks the distinct key groups and primary-key groups from the sorted array
// `K` (of size `n`) into `key_group_rank` and `primary_key_group_rank`,
// respectively, and marks the keys' activeness-flags into `is_active`. The low
// `secondary_bits` bits of a key are its secondary, and the rest are primary.
// The first entry of a group is ranked by its offset into `K`, whereas the rest
// of the entries in the group are ranked `0`.
// Grid organization: one thread per key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void rank_key_groups_and_primary_key_groups_and_mark_active_flags_kernel(   const idx_t* const __restrict__ K, const idx_t n,
                                                                            const idx_t secondary_bits,
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
        const bool key_tie = (K[i] == K[i - 1]);
        const bool primary_key_tie = ((K[i] >> secondary_bits) == (K[i - 1] >> secondary_bits));
        key_group_rank[i] = (!key_tie ? i : 0);
        primary_key_group_rank[i] = (!primary_key_tie ? i : 0);
    }

    const bool pre_tie = (i == 0        ? false : K[i] == K[i - 1]);
    const bool suc_tie = (i == n - 1    ? false : K[i] == K[i + 1]);
    is_active[i] = (pre_tie || suc_tie);
}


// Refines the primary-keys of the sorted array `K` (of size `n`) based on its
// key groups' ranks `key_group_rank` and its primary-key groups' ranks
// `primary_key_group_rank`, and writes back the refined *unpacked* primary-key.
// Grid organization: one thread per key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void refine_primary_keys_kernel(idx_t* const __restrict__ K, const idx_t n,
                                const idx_t secondary_bits,
                                const idx_t* const __restrict__ key_group_rank, const idx_t* const __restrict__ primary_key_group_rank)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    K[i] = (K[i] >> secondary_bits) + (key_group_rank[i] - primary_key_group_rank[i]);
}

}



#endif
