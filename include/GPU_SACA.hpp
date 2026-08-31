
#ifndef GPU_SACA_HPP
#define GPU_SACA_HPP



#include "types.hpp"
#include "Packed_Prefix.hpp"
#include "Partition_Backend.hpp"
#include "Spill_Store.hpp"
#include "Staging_Slab.hpp"
#include "Read_Task.hpp"
#include "env.hpp"
#include "tests.hpp"

#include "concurrentqueue/blockingconcurrentqueue.h"

#include "cuda_runtime_api.h"

#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>


// TODO: name algorithm.
namespace gpu_saca
{

constexpr p_id_t MAX_PART_COUNT = std::numeric_limits<p_id_t>::max();  // Maximum supported number of partitions.


// The suffix array constructor in GPUs for texts with symbols of type `T_sym_`.
template <typename T_sym_>
class GPU_SACA
{
private:

    static constexpr double MEM_SAFETY_MARGIN = 0.1;    // Safety margin to spare some global memory for CUDA runtime overhead and possible fragmentation.
                                                        // TODO: what happens if we bump it down? How far can we go?

    // Bytes per suffix required in each of the two device key-buffers, per
    // partition. The initial-ranking phase sorts the suffixes by their `\ell`-
    // length prefixes, whereas the re-ranking phase reuses one buffer as two
    // laid-flat `idx_t` arrays (the key-group and the primary-key-group ranks).
    // So the buffers are sized to whichever of the two demands is larger.
    static constexpr std::size_t DEV_KEY_BYTES = (sizeof(Prefix_Key) > 2 * sizeof(idx_t) ? sizeof(Prefix_Key) : 2 * sizeof(idx_t));

    // Assumed worst-case ratio of actual max partition-size to the target
    // estimated during pivots-selection.
    static constexpr double PART_SZ_IMBALANCE_SLACK = 1.20;

    const std::size_t host_budget;  // Host-memory budget available for the algorithm.

    const idx_t n_; // Size of the text.
    const T_sym_* const T_h_;   // Text at the host-side.

    p_id_t p_;  // Number of partitions for the SA.

    static constexpr idx_t ell = ELL_WORDS * sizeof(idx_t) / sizeof(T_sym_);    // Prefix-size of the initial sort-keys.
    static constexpr idx_t K = K_WORDS * sizeof(idx_t) / sizeof(T_sym_);    // Prefix-size of pivots.

    std::vector<Pivot_Key> pivots;  // The pivots to break the SA into partitions. The pivots are inclusive upper bounds of their partition.

    p_id_t* P_id;   // Partition-IDs of the suffixes.

    std::vector<idx_t> P_cap;   // Full initial sizes of the partitions.
    std::vector<idx_t> Active_sz;   // Active-set sizes of the suffix partitions.
    std::vector<idx_t> P_off;   // Exclusive cumulative size of the partitions.
    std::vector<idx_t> cohort_sz;   // Full cohort size per generation.

    idx_t ctx_; // Context-size the current generation is resolving.

    static constexpr idx_t STREAM_C = 3;    // Number of CUDA-streams to support multi-buffered overlap of PCIe transfers and kernel executions.
    cudaStream_t stream[STREAM_C];  // CUDA streams for triple-buffered overlap of PCIe transfers and kernel executions.

    std::unique_ptr<Partition_Backend> backend_;    // Partition storage backend.
    std::unique_ptr<Spill_Store> spill_;    // Partitions' on-disk spill-files manager.

    // Data structures to for the chunked staging of data between the partition-backend and the device buffers.
    // {
        // ~2M elements per chunk / slab: memory usage: 2M x (sizeof(Prefix_Key) + 8) (~48 MB at `ell = 2`).
        static constexpr idx_t CHUNK_CAP = (idx_t(1) << 21);
        inline static const idx_t PREFETCH_DEPTH = env::get_u64("GPU_SACA_PREFETCH_DEPTH").value_or(0); // Complete partitions to stage ahead for the device-dispatcher.
        inline static const idx_t WRITEBACK_DEPTH = env::get_u64("GPU_SACA_WRITEBACK_DEPTH").value_or(0);   // Complete partitions to stage ahead to return to the backend.

        idx_t chunk_cap_;   // Chunk granularity (keys / suffixes per slab).
        idx_t slab_c_;  // |S|: slab pool size.
        idx_t reader_c_;    // |r|: reader-threads count.
        idx_t writer_c_;    // |w|: writer-threads count.
        idx_t dispatcher_c; // |d|: dispatcher-threads count.
        idx_t compute_c;    // |c|: compute-threads count.

        std::vector<Staging_Slab> S;    // The pinned staging-slab pool, bridging partition-backend to device.

        // TODO: the following staging data structures need to move to their dedicated class.

        template <typename T> using blocking_q_t = moodycamel::BlockingConcurrentQueue<T>;
        blocking_q_t<Read_Task> RT; // Chunk-read tasks.
        blocking_q_t<Staging_Slab*> F;  // Free slabs.
        blocking_q_t<Partition_Assembly*> R;    // Slabs with chunks read from partition-backend, ready to be uploaded to device.
        blocking_q_t<Staging_Slab*> W;  // Slabs with chunks downloaded from device, ready to be written to partition-backend.

        std::unique_ptr<Partition_Assembly[]> part_assembly;    // Per-partition assembly records (i.e. which chunks assemble them); `p_` of them.


        // Sets the staging configurations: slab capacity, slab pool size, and
        // reader- / writer-pool sizes.
        void setup_staging_config();

        // Returns the total memory used by the slab-pool.
        std::size_t slab_pool_memory() const;
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

        Prefix_Key* key_buf[STREAM_C][2];   // Input and output buffers for keys of suffixes, per stream; typed to the initial-ranking sort key.
        idx_t* suf_buf[STREAM_C][2];        // Input and output buffers for suffixes, per stream.
        p_sz_t* perm_buf[STREAM_C][2];      // Input and output buffers for sort's permutation-indices, per stream; required for sorting with physically disjoint keys.
        beta_t* beta_buf[STREAM_C][2];      // Input and output buffers for the backward routing tags, per stream.
        p_sz_t* msg_off_buf[STREAM_C];      // Buffer for message offsets per destination-partition, per stream.
        void* workspace_buf[STREAM_C];      // Scratch space for `cub` radix-sort and scan, per stream.
        std::size_t workspace_sz;           // Size of the `cub` workspace.

        uint8_t* is_active_buf[STREAM_C];   // Activeness-marker buffers for suffixes, per stream.
        idx_t* active_set_sz_d; // Sizes of the new active-sets of the partitions.

        idx_t* unserved_c_d;                // Counters across partitions in a generation: active suffixes with no messaged received;
                                            // temporary, for verification purposes.
    // }


    // Returns the fixed RAM-budget (in bytes) required in the algorithm.
    // Currently: the text (`T`), the partition-IDs (`Π`), the scatter-buffers
    // in partitioning, and backend's fixed space.
    std::size_t fixed_budget() const;

    // Selects the set of pivots for the partitioning of the suffixes.
    void select_pivots();

    // Returns the inclusive upper-bound for the maximum supported partition-
    // size. The rationale used in fixing this is that, in the case when the
    // partitions are large enough such that `STREAM_C = 3` partitions can fully
    // occupy the available device-memory, their sizes must be evenly balanced.
    idx_t partition_size_upper_bound() const;

    // Returns `true` iff the computed partition-sizes fit all hard-constraints
    // from the implementation.
    bool partition_sizes_fit() const;

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

    // Sets the context-size that the current generation is resolving to `ctx`
    // and the generation's index to `gen`.
    void set_context(idx_t ctx, idx_t gen);

    // Ranks the suffixes in their partitions based on the prefixes associated
    // to them during partitioning.
    void initial_rank();

    // Computes the partition-IDs of the suffixes.
    void compute_partition_IDs();

    // Returns the active set size.
    idx_t active_set_size() const;

    // Drives a complete pipeline for all the partitions: chunked reading from
    // the backend, sort-and-rank in the device, and chunked write-back to the
    // backend. `init_rank_` determines whether the pipeline is being executed
    // for the initial-ranking phase.
    template <bool init_rank_> void drive_pipeline();

    // Stages the slab-assembly `p_assembly`'s content for upload, with required
    // compute / logic.
    template <bool init_rank_> void stage_slabs(Partition_Assembly& p_assembly);

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

    // Keeps exponentiating the effective prefix-context of the suffixes until
    // the suffixes are fully sorted.
    void exponentiate_prefix();

    // Allocates enough space into `SA` and concatenates the partitions into it.
    void concatenate_partitions(idx_t*& SA);


    // Provide access to private members for verification tests.

    friend bool test::validate_messages<T_sym_>(const Staging_Slab& s, const GPU_SACA<T_sym_>* self, const p_sz_t* msg_off);


public:

    // Constructs a GPU-SACA object for the input text `T` of size `n`.
    GPU_SACA(const T_sym_* T, idx_t n);

    ~GPU_SACA();

    // Builds the suffix array into `SA`.
    void build_SA(idx_t*& SA);
};

}



#endif
