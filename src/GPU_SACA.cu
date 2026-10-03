
#include "GPU_SACA.hpp"
#include "GPU_SACA.cuh"
#include "Suffix_Buffer.hpp"
#include "Packed_Prefix.hpp"
#include "Packed_Prefix_Decomposer.cuh"
#include "bit-layout.hpp"
#include "message.hpp"
#include "utility.hpp"
#include "utility.cuh"
#include "env.hpp"
#include "tests.hpp"
#include "Pipeline_Probe.hpp"
#include "profile.hpp"

#include "concurrentqueue/lightweightsemaphore.h"

#include "omp.h"
#include "cub/cub.cuh"
#include "thrust/iterator/zip_iterator.h"

#include <unistd.h>

#include <cstdint>
#include <cstddef>
#include <limits>
#include <atomic>
#include <utility>
#include <vector>
#include <cstring>
#include <sstream>
#include <memory>
#include <cstdlib>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>


namespace gpu_saca
{

template <typename T_sym_>
GPU_SACA<T_sym_>::GPU_SACA(const std::string& ip_path, const std::string& op_path):
      host_budget(env::get_u64("GPU_SACA_RAM_BUDGET").value_or(static_cast<uint64_t>((1 - MEM_SAFETY_MARGIN) * util::available_memory())))
    , n_(util::file_size<T_sym_>(ip_path))
    , T_h_(nullptr)
    , ip_path_(ip_path)
    , op_path_(op_path)
    , p_(0)
    , ctx_(0)
    , slab_pin_time_(0)
    , workspace_sz(0)
{
    if(n_ == 0)
    {
        std::cerr << "The text at " << ip_path_ << " is empty. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    for(std::size_t s = 0; s < STREAM_C; ++s)
        CUDA_CHECK(cudaStreamCreate(stream + s));

    for(std::size_t b = 0; b < STREAM_C; ++b)
    {
        for(std::size_t r = 0; r < 2; ++r)
            key_buf[b][r] = nullptr,
            suf_buf[b][r] = nullptr,
            perm_buf[b][r] = nullptr,
            beta_buf[b][r] = nullptr;

        workspace_buf[b] = nullptr;
        is_active_buf[b] = nullptr;
        msg_off_buf[b] = nullptr;
    }

    active_set_sz_d = nullptr;
    unserved_c_d = nullptr;
    pill_c_d = nullptr;
}


template <typename T_sym_>
GPU_SACA<T_sym_>::~GPU_SACA()
{
    if(slab_pinner_.joinable()) // The slab-pinner may still be alive if the build failed to reach its await.
        slab_pinner_.join();

    util::deallocate(T_h_);

    for(std::size_t s = 0; s < STREAM_C; ++s)
        CUDA_CHECK(cudaStreamDestroy(stream[s]));

    for(std::size_t b = 0; b < STREAM_C; ++b)
    {
        for(std::size_t r = 0; r < 2; ++r)
        {
            CUDA_CHECK(cudaFree(key_buf[b][r]));
            CUDA_CHECK(cudaFree(suf_buf[b][r]));
            CUDA_CHECK(cudaFree(perm_buf[b][r]));
            CUDA_CHECK(cudaFree(beta_buf[b][r]));
        }

        CUDA_CHECK(cudaFree(workspace_buf[b]));
        CUDA_CHECK(cudaFree(is_active_buf[b]));
        CUDA_CHECK(cudaFree(msg_off_buf[b]));
    }

    CUDA_CHECK(cudaFree(active_set_sz_d));
    CUDA_CHECK(cudaFree(unserved_c_d));
    CUDA_CHECK(cudaFree(pill_c_d));
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::build_SA()
{
    using namespace gpu_saca::timer;
    PROFILE_SCOPE("build_SA");

    std::cerr << "Sequence length: " << n_ << "\n";

    const auto t_l = now();
    {   PROFILE_SCOPE("load_text");                 load_text();    }

    const auto t_0 = now();
    build_timing_.text_load = duration(t_0 - t_l);

    {   PROFILE_SCOPE("select_pivots");             select_pivots();    }
    const auto t_1 = now();
    build_timing_.pivots = duration(t_1 - t_0);
    std::cerr << "Pivots selection completed. Time taken: " << build_timing_.pivots << "s.\n";

    {   PROFILE_SCOPE("fault_in_partition_IDs");    P_id = Partition_ID_Map(n_, p_), P_id.prepage();    }
    const auto t_p = now();
    build_timing_.prepage_pid = duration(t_p - t_1);
    std::cerr   << "Partition-ID map prepaging completed. Time taken: " << build_timing_.prepage_pid << "s"
                << " (" << (P_id.bytes() / (1024.0 * 1024.0 * 1024.0) / build_timing_.prepage_pid) << " GBs/s; " << Partition_ID_Map::bits_for(p_) << " bits per ID).\n";

    {   PROFILE_SCOPE("partition");                 partition();
        if(!partition_sizes_fit())
        {
            std::cerr << "Unsuitable partition-sizes for " << p_ << " partitions. Increase `K` and / or `GPU_SACA_P_TARGET`. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
    }
    const auto t_2 = now();
    build_timing_.partition = duration(t_2 - t_p);
    std::cerr << "Partitioning completed. Time taken: " << build_timing_.partition << "s.\n";

    {
        PROFILE_SCOPE("allocate_device_space");
        if(!allocate_device_space())
        {
            std::cerr << "Too low device memory available for the process. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
    }

    // Await allocation of the staging slab pool bridging the backend to the device.
    {
        PROFILE_SCOPE("await_staging");
        slab_pinner_.join();
    }
    const auto t_3 = now();
    build_timing_.device_setup = duration(t_3 - t_2);
    std::cerr   << "Slab-pool pinning completed in the background. Time taken: " << slab_pin_time_ << "s"
                << " (" << (slab_pool_memory() / (1024.0 * 1024.0 * 1024.0) / slab_pin_time_) << " GBs/s).\n";
    std::cerr << "Device-space allocated and staging-setup completed. Time taken: " << build_timing_.device_setup << "s.\n";

    {   PROFILE_SCOPE("initial_rank");              initial_rank();}
    const auto t_4 = now();
    build_timing_.initial_rank = duration(t_4 - t_3);
    std::cerr << "Initial ranking completed. Time taken: " << build_timing_.initial_rank << "s.\n";

    {   PROFILE_SCOPE("drop_text");                 util::deallocate(T_h_), T_h_ = nullptr;  }
    backend_->raise_cap(n_ * sizeof(T_sym_));   // The text's space now goes to the backend's RAM-tier.

    {   PROFILE_SCOPE("exponentiate_prefix");       exponentiate_prefix();}
    const auto t_5 = now();
    build_timing_.exponentiate = duration(t_5 - t_4);
    std::cerr << "Prefix-exponentiation completed. Time taken: " << build_timing_.exponentiate << "s.\n";

    // Free the inboxes' scratch-space ahead of the suffix array's output.
    {   PROFILE_SCOPE("retire_messages");           backend_->retire_messages();    }

    {   PROFILE_SCOPE("drop_partition_IDs");        P_id = Partition_ID_Map();  }

    {   PROFILE_SCOPE("concatenate_partitions");    concatenate_partitions(op_path_);}
    const auto t_6 = now();
    build_timing_.concatenate = duration(t_6 - t_5);
    std::cerr << "Partition-concatenation completed. Time taken: " << build_timing_.concatenate  << "s.\n";

    {   PROFILE_SCOPE("drop_backend");              backend_.reset();   spill_->cleanup();   }

    build_timing_.n = n_, build_timing_.p = p_, build_timing_.gen_c = cohort_sz.size();
}



template <typename T_sym_>
void GPU_SACA<T_sym_>::load_text()
{
    constexpr auto null_count = tail_padding<T_sym_>;   // Null symbols the prefix-packing requires past the text's tail.
    const auto sym_count = n_;  // Symbols stored in the file.
    const auto bytes = (sym_count + null_count) * sizeof(T_sym_);   // Size of the text's buffer.

    const auto t_0 = timer::now();
    T_h_ = util::checked_allocate<T_sym_>(sym_count + null_count);
    const auto t_1 = timer::now();

    util::prepage(T_h_, sym_count + null_count);
    const auto t_2 = timer::now();

    util::read_file(ip_path_, T_h_, sym_count);
    for(idx_t i = 0; i < null_count; ++i)
        T_h_[sym_count + i] = std::numeric_limits<T_sym_>::min();
    const auto t_3 = timer::now();

    std::cerr   << "Text-buffer allocation completed. Time taken: " << timer::duration(t_1 - t_0) << "s"
                << " (" << (bytes / (1024.0 * 1024.0 * 1024.0) / timer::duration(t_1 - t_0)) << " GBs/s).\n";
    std::cerr   << "Text-buffer prepaging completed. Time taken: " << timer::duration(t_2 - t_1) << "s"
                << " (" << (bytes / (1024.0 * 1024.0 * 1024.0) / timer::duration(t_2 - t_1)) << " GBs/s).\n";
    std::cerr   << "Text-read completed. Time taken: " << timer::duration(t_3 - t_2) << "s"
                << " (" << (bytes / (1024.0 * 1024.0 * 1024.0) / timer::duration(t_3 - t_2)) << " GBs/s).\n";
}


template <typename T_sym_>
std::size_t GPU_SACA<T_sym_>::fixed_budget() const
{
    return  n_ * sizeof(T_sym_) +   // `T`
            P_id.bytes() +  // `Π`
            slab_pool_memory() +
            Suffix_Buffer::scatter_buffer_memory(omp_get_max_threads(), p_) +
            Partition_Backend::fixed_budget(p_);
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::select_pivots()
{
    const idx_t p_sz_max = partition_size_upper_bound();    // Maximum supported partition-size based on the device-memory budget.
    const idx_t p_min = (n_ + p_sz_max - 1) / p_sz_max;     // Minimum number of partitions.
    const idx_t p_max = std::min<idx_t>(PARTITION_SLACK_FACTOR * p_min, MAX_PART_COUNT - ell);  // Maximum number of partitions; see `fwd_tag` for the `ell` bias.
    const idx_t s = OVERSAMPLING_FACTOR * p_max * std::log2(n_);    // Number of samples.
    std::vector<Pivot_Key> S(s);    // The samples.

    // Draw and sort the samples.
    {
        #pragma omp parallel
        {
            std::mt19937_64 rng(SAMPLE_SEED + omp_get_thread_num());    // Fixed per-thread RNG for reproducibility.
            std::uniform_int_distribution<idx_t> dist(0, n_ - 1);

            #pragma omp for schedule(static)
            for(idx_t i = 0; i < s; ++i)
                S[i] = Pivot_Key::pack_prefix(T_h_ + dist(rng));
        }

        std::sort(S.begin(), S.end());
    }

    // Pick the final pivots.
    {
        const double strip_sz_exp = static_cast<double>(n_) / (s + 1);  // Each gap between consecutive samples corresponds to a strip of the SA, of expected size `n / (s + 1)`.

        int64_t i = 0;
        int64_t part_beg = 0;   // Sample index where the current partition begins.
        while(i < static_cast<int64_t>(s))
        {
            const auto hops = i - part_beg + 1; // Number of sample-hops made in the current partition from the preceding partition.
            if(hops * strip_sz_exp <= p_sz_max) // Advance till the current partition's size remains at most `p_sz_max`.
                i++;
            else
            {
                const auto excl_idx = i;    // Sample to exclude from the current partition.
                auto pivot_idx = i - 1;

                // Backoff past all occurrences of the excluded sample's matches, so equal prefixes aren't split across partitions.
                while(pivot_idx >= part_beg && !(S[pivot_idx] < S[excl_idx]))
                    pivot_idx--;

                if(pivot_idx < part_beg)
                {
                    std::cerr << "In pivots-selection, a single `K`-prefix exceeds maximum supported partition-size. Increase `K`. Aborting.\n";
                    std::exit(EXIT_FAILURE);
                }

                pivots.push_back(S[pivot_idx]); // Inclusive upper-bound of the current partition.
                if(pivots.size() + 1 > p_max)
                {
                    std::cerr << "In pivots-selection, partition-count exceeds maximum supported partition-count. Increase `K`. Aborting.\n";
                    std::exit(EXIT_FAILURE);
                }

                i = part_beg = pivot_idx + 1;
            }
        }

        p_ = pivots.size() + 1;

        if(!bit_layout::validate(n_, p_ + ell)) // See `fwd_tag` for the `ell` bias.
        {
            std::cerr << "Bit-packing layout violated. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
    }

#ifdef STATS
    std::cerr   << "Pivots selection: " << p_ << " partitions from " << s << " samples of " << K << "-symbol prefixes;"
                << " initial sort-keys of " << ell << " symbols.\n";
#endif
}


template <typename T_sym_>
idx_t GPU_SACA<T_sym_>::partition_size_upper_bound() const
{
    // Device-space requirement for a suffix: it occupies space in two ping-pong partition-buffers for the radix-sorts, and one activeness-flag byte.
    // 2 x (key + suffix + permutation-index + backward tag) ping-pong + activeness flag.
    constexpr std::size_t bytes_per_suf = 2 * (DEV_KEY_BYTES + sizeof(idx_t) + sizeof(p_sz_t) + sizeof(beta_t)) + sizeof(uint8_t);
    std::size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const std::size_t raw_budget = free_mem * (1.0 - MEM_SAFETY_MARGIN);    // Total usable memory for the algorithm.
    const double imb_slack = env::get_f64("GPU_SACA_IMBALANCE_SLACK").value_or(PART_SZ_IMBALANCE_SLACK);
    const std::size_t budget = raw_budget / imb_slack;  // The actual partition-sizes, to be found out later, can be larger due to the probabilistic nature of pivoting.

    idx_t p_sz_max; // Maximum supported suffix-count per partition.
    {
        const idx_t from_device_mem = budget / (STREAM_C * bytes_per_suf);    // Max count so that `STREAM_C` partitions can fit concurrently in the worst-case.
        const idx_t from_rank_width = bit_layout::MAX_PART_SZ;
        p_sz_max = std::min(from_device_mem, from_rank_width);
    }

    while(true) // Keep refining `p_sz_max` until the device-budget can hold `STREAM_C`-many `p_sz_max`-sized partitions concurrently.
    {
        const auto req_workspace_sz = cub_workspace_size(p_sz_max);
        if(STREAM_C * (p_sz_max * bytes_per_suf + req_workspace_sz) <= budget)
            break;

        if(STREAM_C * req_workspace_sz >= budget)
        {
            std::cerr << "Device memory cannot even accommodate the CUB workspaces. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        const idx_t next_p_sz_max = (budget - STREAM_C * req_workspace_sz) / (STREAM_C * bytes_per_suf);
        if(next_p_sz_max >= p_sz_max)   // Converged to a fixed-point without being able to compute a suitable `p_sz_max`.
        {
            std::cerr << "Unable to determine a suitable partition-size upper bound. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        p_sz_max = next_p_sz_max;
    }


    const idx_t p_min_from_budget = std::max(STREAM_C, (n_ + p_sz_max - 1) / p_sz_max); // Lower-bound on the number partitions.
    idx_t p_min = p_min_from_budget;

    // Optional override of required minimum partition-count.
    {
        if(const auto p_target = env::get_u64("GPU_SACA_P_TARGET"); p_target)
        {
            if(*p_target < p_min_from_budget)
                std::cerr << "The provided required minimum partition-count falls below device-budget minimum. Discarding it.\n";
            else
                p_min = *p_target;
        }
    }

    // The (primary, secondary) key packing scheme, or very large `n`, can demand more partitions than the partition-ID type can index.
    if(p_min > MAX_PART_COUNT - ell)    // See `fwd_tag` for the `ell` bias.
    {
        std::cerr   << "Current partitioning policy requires at least " << p_min << " partitions for the " << n_ << "-length text. "
                    << "But the forward tags can index at most " << (MAX_PART_COUNT - ell) << " partitions. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    const auto p_sz_bound = (n_ + p_min - 1) / p_min;   // Maximum supported size of a partition based on the minimum partition-count.

#ifdef STATS
    std::cerr   << "Partition-size bound: " << p_sz_bound << " suffixes, from minimum partition-count " << p_min
                << " (device-budget floor: " << p_min_from_budget << ").\n";
#endif

    return p_sz_bound;
}


template <typename T_sym_>
bool GPU_SACA<T_sym_>::partition_sizes_fit() const
{
    const auto P_sz_max = *std::max_element(P_cap.cbegin(), P_cap.cend());

    if(P_sz_max > bit_layout::MAX_PART_SZ)
    {
        std::cerr << "Maximum partition-size " << P_sz_max << " exceeds hard-constraint " << bit_layout::MAX_PART_SZ << " set by local-rank bit-width.\n";
        return false;
    }

    return true;
}


template <typename T_sym_>
std::size_t GPU_SACA<T_sym_>::cub_workspace_size(const idx_t part_sz)
{
    cub::DoubleBuffer<idx_t> suf_dummy(nullptr, nullptr);
    cub::DoubleBuffer<p_sz_t> perm_dummy(nullptr, nullptr);

    std::size_t init_rank_workspace_sz = 0;
    {
        cub::DoubleBuffer<Prefix_Key> keys_dummy(nullptr, nullptr);

        std::size_t suf_perm_sort_space = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs( // Check space for sorting based on the forward tag.
            nullptr, suf_perm_sort_space,
            suf_dummy, perm_dummy,
            part_sz,
            bit_layout::TAG_SHIFT, bit_layout::WORD_BITS
        ));

        std::size_t pref_perm_sort_space = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs( // Check space for sorting based on the prefix-key.
            nullptr, pref_perm_sort_space,
            keys_dummy, perm_dummy,
            part_sz,
            Prefix_Key_Decomposer{},
            0, ELL_WORDS * 64
        ));

        init_rank_workspace_sz = std::max(suf_perm_sort_space, pref_perm_sort_space);
    }

    std::size_t exponentiation_workspace_size = 0;
    {
        cub::DoubleBuffer<idx_t> keys_dummy(nullptr, nullptr);
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, exponentiation_workspace_size,
            keys_dummy, perm_dummy,
            part_sz,
            0, 64
        ));
    }

    std::size_t scan_workspace_sz = 0;
    {
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(
            nullptr, scan_workspace_sz,
            static_cast<idx_t*>(nullptr), static_cast<idx_t*>(nullptr),
            cub::Max(),
            part_sz
        ));
    }

    std::size_t compaction_workspace_sz = 0;
    {
        auto in_zip  = thrust::make_zip_iterator(thrust::make_tuple(static_cast<idx_t*>(nullptr), static_cast<idx_t*>(nullptr), static_cast<p_id_t*>(nullptr)));
        auto zip_out = thrust::make_zip_iterator(thrust::make_tuple(static_cast<idx_t*>(nullptr), static_cast<idx_t*>(nullptr), static_cast<p_id_t*>(nullptr)));
        CUDA_CHECK(cub::DevicePartition::Flagged(
            nullptr, compaction_workspace_sz,
            in_zip,
            static_cast<uint8_t*>(nullptr),
            zip_out,
            static_cast<idx_t*>(nullptr),
            part_sz
        ));
    }

    std::size_t msg_off_scan_workspace_sz = 0;
    {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, msg_off_scan_workspace_sz,
            static_cast<p_sz_t*>(nullptr), static_cast<p_sz_t*>(nullptr),
            part_sz // Over-provisioned: message-offset sum-scans are over `p_ + 1` entries and `p_` isn't available here.
        ));
    }

    return std::max({init_rank_workspace_sz, exponentiation_workspace_size, scan_workspace_sz, compaction_workspace_sz, msg_off_scan_workspace_sz});
}


template <typename T_sym_>
bool GPU_SACA<T_sym_>::allocate_device_space()
{
    // Each suffix in each stream needs:
    //      1. `cub::DoubleBuffer` keys, input and output (2 x `DEV_KEY_BYTES`);
    //      2. `cub::DoubleBuffer` values, input and output (2 x `idx_t`);
    //      3. `cub::DoubleBuffer` permutation-indices, generated and transformed (2 x `p_sz_t`);
    //      4. `cub::DoubleBuffer` backward routing tag pairs, input and output (2 x `beta_t`);
    //      5. activeness-flag, output (1 x `uint8_t`).
    // Message-collation requires `p + 1` counts (per-partition offset).
    // `cub` itself requires a workspace for sort and scan.
    // Memory constraint: STREAM_C x (P_sz_max x (2 x (DEV_KEY_BYTES + sizeof(idx_t) + sizeof(p_sz_t)) + 2 * sizeof(uint8_t)) + (p + 1) * sizeof(p_sz_t) + workspace_sz) <= free_mem x (1 - MEM_SAFETY_MARGIN)

    const idx_t P_sz_max = *std::max_element(P_cap.cbegin(), P_cap.cend());

    // Compute required workspaces-sizes for the sort and the scan.
    workspace_sz = cub_workspace_size(P_sz_max);

    // Check if the device has enough space for the buffers.
    {
        std::size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        const std::size_t budget = free_mem * (1.0 - MEM_SAFETY_MARGIN);    // Total memory across all the streams.
        const std::size_t per_stream_budget = P_sz_max * (2 * (DEV_KEY_BYTES + sizeof(idx_t) + sizeof(p_sz_t) + sizeof(beta_t)) + sizeof(uint8_t)) + (p_ + 1) * sizeof(p_sz_t) + workspace_sz;
#ifdef STATS
        std::cerr << "Required memory:    " << ((STREAM_C * per_stream_budget) / (1024.0 * 1024.0 * 1024.0)) << "GBs.\n";
        std::cerr << "Device free-memory: " << (budget / (1024.0 * 1024.0 * 1024.0)) << "GBs.\n";
#endif
        if(STREAM_C * per_stream_budget > budget)
            return false;
    }


    // Allocate the required spaces in the device.
    {
        for(std::size_t b = 0; b < STREAM_C; ++b)   // For each stream
        {
            for(std::size_t r = 0; r < 2; ++r)  // For each buffer in the radix-sort ping-pong
            {
                CUDA_CHECK(cudaMalloc(&key_buf[b][r], P_sz_max * DEV_KEY_BYTES));
                CUDA_CHECK(cudaMalloc(&suf_buf[b][r], P_sz_max * sizeof(idx_t)));
                CUDA_CHECK(cudaMalloc(&perm_buf[b][r], P_sz_max * sizeof(p_sz_t)));
                CUDA_CHECK(cudaMalloc(&beta_buf[b][r], P_sz_max * sizeof(beta_t)));
            }

            CUDA_CHECK(cudaMalloc(&workspace_buf[b], workspace_sz));
            CUDA_CHECK(cudaMalloc(&is_active_buf[b], P_sz_max * sizeof(uint8_t)));
            CUDA_CHECK(cudaMalloc(&msg_off_buf[b], (p_ + 1) * sizeof(p_sz_t)));
        }

        CUDA_CHECK(cudaMalloc(&active_set_sz_d, p_ * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&unserved_c_d, sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&pill_c_d, sizeof(idx_t)));
    }

    return true;
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::setup_staging_config()
{
    const auto P_sz_max = *std::max_element(P_cap.cbegin(), P_cap.cend());

    // Sets the capacity of each staging-slab.
    {
        // A chunk-boundary `c * chunk_cap_` must be disk-block aligned in each
        // persisted field: the suffix-lanes, and the keys (`p_sz_t`).
        constexpr idx_t elems_per_disk_blk = std::max<idx_t>(lane::ALIGN, Direct_File::DISK_BLOCK / sizeof(p_sz_t));

        // A slab's inbox-range must also tile into `MSG_CHUNKS_PER_SLAB` message-chunks of whole `MSG_ALIGN`-groups.
        constexpr idx_t cap_granularity = MSG_CHUNKS_PER_SLAB * MSG_ALIGN;
        static_assert(cap_granularity % elems_per_disk_blk == 0, "Slab-capacity granularity must be disk-block aligned.");
        static_assert(CHUNK_CAP % cap_granularity == 0, "`CHUNK_CAP` must be a multiple of the slab-capacity granularity.");

        if(P_sz_max >= CHUNK_CAP)
            chunk_cap_ = CHUNK_CAP;
        else    // Small partitions can render disk-block-rounded reads to overflow the slab's capacity, by up-to 4095 bytes.
            chunk_cap_ = ((P_sz_max + cap_granularity - 1) / cap_granularity) * cap_granularity;

#ifdef TESTS
        // Small slabs make even small inputs span many content-chunks per partition.
        if(const auto cap = env::get_u64("GPU_SACA_CHUNK_CAP"); cap)
        {
            if(*cap == 0 || *cap % cap_granularity != 0)
            {
                std::cerr << "`GPU_SACA_CHUNK_CAP` must be a positive multiple of " << cap_granularity << ". Aborting.\n";
                std::exit(EXIT_FAILURE);
            }

            chunk_cap_ = *cap;
        }
#endif
    }

    // Set the reader- and writer-pool sizes.
    {
        constexpr idx_t read_poster_c = 1;
        reader_c_ = env::get_u64("GPU_SACA_READER_C").value_or(24); // TODO: tune default.
        writer_c_ = env::get_u64("GPU_SACA_WRITER_C").value_or(8);  // TODO: tune default.
        dispatcher_c_ = STREAM_C;
        stager_c_ = env::get_u64("GPU_SACA_STAGER_C").value_or(std::max<idx_t>(1, omp_get_max_threads() - (read_poster_c +reader_c_ + writer_c_ + dispatcher_c_)));
        if(read_poster_c + reader_c_ + writer_c_ + dispatcher_c_ + stager_c_ > static_cast<idx_t>(omp_get_max_threads()))
            std::cerr <<    "Warning: the staging workers (read-posters, readers, writers, stagers, and dispatchers)"
                            ": (" << read_poster_c << ", " << reader_c_ << ", " << writer_c_ << ", " << stager_c_ << ", " << dispatcher_c_ << ")"
                            " oversubscribe the " << omp_get_max_threads() << "available CPU-threads.\n";
    }

    // Sets the slab pool size.
    {
        const auto max_chunk_per_part = (P_sz_max + chunk_cap_ - 1) / chunk_cap_;

        // `STREAM_C` partitions in the GPU; `PREFETCH_DEPTH` partitions staged ahead in the ready-queue;
        // `WRITEBACK_DEPTH` partitions staged ahead in the write-queue; a slab per-reader and per-writer.
        // A slab lives in exactly one of: `F`, a reader, an assembly slot (covering its timeline in `C`,
        // at a stager, and at device), `W`, and a writer.
        slab_c_ = (STREAM_C + PREFETCH_DEPTH + WRITEBACK_DEPTH) * max_chunk_per_part + reader_c_ + writer_c_;

        admit_budget_ = slab_c_;
        if(admit_budget_ < max_chunk_per_part)
        {
            std::cerr << "Slab-pool cannot admit the largest partition's assembly. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        INSTRUMENT(probe::P().init(reader_c_, stager_c_, dispatcher_c_, writer_c_, slab_c_, admit_budget_));
    }

    // Log the staging configurations.
    {
#ifdef STATS
        std::cerr << "Staging-configuration between backend and device:\n";
        std::cerr << "\tReader-pool size: " << reader_c_ << "\n";
        std::cerr << "\tWriter-pool size: " << writer_c_ << "\n";
        std::cerr << "\tDevice-dispatcher-pool size: " << dispatcher_c_ << "\n";
        std::cerr << "\tStager-pool size: " << stager_c_ << "\n";
        std::cerr << "\tSlab-pool size: " << slab_c_ << "\n";
        std::cerr << "\tSlab-capacity:  " << chunk_cap_ << "\n";
        std::cerr << "\tSlab-pool memory: " << slab_pool_memory() << "B\n";
#endif
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::setup_staging()
{
    S.resize(slab_c_);
    for(auto& s : S)
    {
        s.allocate(chunk_cap_);
        F.enqueue(&s);
    }

    part_assembly = std::make_unique<Partition_Assembly[]>(p_);
}


template <typename T_sym_>
std::size_t GPU_SACA<T_sym_>::slab_pool_memory() const
{
    return slab_c_ * chunk_cap_ * (LANE_BYTES + sizeof(Prefix_Key) + sizeof(beta_t) + 2 * MESSAGE_BYTES);
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::partition()
{
    {
        PROFILE_SCOPE("compute_partition_IDs");
        compute_partition_IDs();

        // Partitioning sweeps need only the partition-sizes.
        if(env::get_u64("GPU_SACA_PARTITION_ONLY").value_or(0))
        {
            std::cerr << "Partition-IDs computed; stopping, as `GPU_SACA_PARTITION_ONLY` is set.\n";
            std::exit(partition_sizes_fit() ? EXIT_SUCCESS : EXIT_FAILURE);
        }
    }

    // Move the suffixes to their corresponding partitions.
    {
        PROFILE_SCOPE("scatter_to_partition");
        std::vector<util::Padded<Suffix_Buffer>> P_buf;

        // Initialize the partitions' spill-stores and their scatter buffers.
        {
            PROFILE_SCOPE("allocate_partition_buffers");

            const auto scratch = env::get("GPU_SACA_SCRATCH");
            if(!scratch)
            {
                std::cerr << "`GPU_SACA_SCRATCH` is not set; set it to colon-separated scratch roots for the disk-backend. Aborting.\n";
                std::exit(EXIT_FAILURE);
            }

            std::vector<std::string> roots;
            {
                std::stringstream ss(*scratch);
                for(std::string r; std::getline(ss, r, ':');)
                    if(!r.empty())
                        roots.push_back(r);
            }

            setup_staging_config();
            util::file::raise_open_files_limit(SPILL_ROLE_C * p_ + 64); // Spill-files of all partitions at once, plus slack for the rest of the process.
            spill_ = std::make_unique<Spill_Store>(std::move(roots), "run_" + std::to_string(::getpid()), op_path_, lane::bytes(n_));
            backend_ = std::make_unique<Partition_Backend>(P_cap, chunk_cap_, *spill_, host_budget, fixed_budget(), p_);

            P_buf.reserve(p_);
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                P_buf.emplace_back(Suffix_Buffer{*backend_, p_id});
        }


        // Pin the slab pool in the background, overlapped with the scatter.
        slab_pinner_ = std::thread(
            [this]
            {
                PROFILE_SCOPE("setup_staging");
                const auto t_0 = timer::now();
                setup_staging();
                slab_pin_time_ = timer::duration(timer::now() - t_0);
            }
        );


        // Add suffix indices to their corresponding partitions (through the buffers).
        {
            PROFILE_SCOPE("omp_scatter");
            #pragma omp parallel for schedule(static)
            for(idx_t i = 0; i < n_; ++i)
            {
                // TODO: skip staging for generation 0 this way:
                // const p_id_t beta = (i >= ell ? P_id[i - ell] : 0);  // Backward tag for generation 0.
                P_buf[P_id[i]].unwrap().add(i);
            }

            #pragma omp parallel for schedule(static)
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                P_buf[p_id].unwrap().flush();
        }

#ifdef TESTS
        {
            bool part_sz_match = true;
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                part_sz_match = part_sz_match && (P_buf[p_id].unwrap().flushed() == P_cap[p_id]);

            std::cerr << "Partition-sizes match verification: " << (part_sz_match ? "Passes" : "Fails") << "\n";
            std::cerr << "Partition-contents verification: " << (test::validate_partition_contents(n_, P_id, P_cap, *backend_) ? "Passes" : "Fails") << "\n";
        }
#endif
    }

    backend_->raise_cap(Suffix_Buffer::scatter_buffer_memory(omp_get_max_threads(), p_));  // The scatter buffers' space now goes to the backend's RAM-tier.
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::compute_partition_IDs()
{
    // Copy the pivots to the device.
    Pivot_Key* pivots_d = nullptr;
    if(!pivots.empty())
    {
        const auto pivots_bytes = pivots.size() * sizeof(Pivot_Key);
        CUDA_CHECK(cudaMalloc(&pivots_d, pivots_bytes));
        CUDA_CHECK(cudaMemcpy(pivots_d, pivots.data(), pivots_bytes, cudaMemcpyHostToDevice));
    }

    constexpr std::size_t BLOCK_SZ = 256;
    static_assert(BLOCK_SZ % Partition_ID_Map::GROUP_SZ == 0, "Text-chunks must hold whole groups of the partition-ID map.");
    idx_t CHUNK_SZ; // Number of suffixes at each text-chunk.
    T_sym_* T_buf[STREAM_C];    // The text-chunk buffers.
    p_id_t* P_id_buf[STREAM_C]; // The partition-ID buffers.
    uint64_t* P_id_word_buf[STREAM_C];  // The packed partition-ID buffers.
    T_sym_* T_stage[STREAM_C];  // Staging buffer for the text-chunks' uploads, per stream.
    uint64_t* P_id_stage[STREAM_C]; // Staging buffer for the packed partition-IDs' downloads, per stream.

    idx_t* P_sz_d;  // Global partition-size histogram at the device.
    {
        CUDA_CHECK(cudaMalloc(&P_sz_d, p_ * sizeof(idx_t)));
        CUDA_CHECK(cudaMemset(P_sz_d, 0, p_ * sizeof(idx_t)));
    }

    // Determine the chunks' sizes for processing them in a streaming manner, and allocate space for the corresponding buffers.
    {
        std::size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        const std::size_t budget = free_mem * (1.0 - MEM_SAFETY_MARGIN);    // Total memory for the chunk buffers.
        const std::size_t per_idx_mem = STREAM_C * (sizeof(T_sym_) + 2 * sizeof(p_id_t));   // Space required per-index across the three sets of buffers; the packed IDs are no wider.
        CHUNK_SZ = std::min<idx_t>(PID_CHUNK_CAP, budget / per_idx_mem);
        CHUNK_SZ = (CHUNK_SZ / BLOCK_SZ) * BLOCK_SZ;    // Round-down to the nearest multiple of `BLOCK_SZ` for clean grid-partitioning.
        if(CHUNK_SZ < BLOCK_SZ) // Too low memory in the GPU to support even one block.
        {
            std::cerr << "Too low device memory available for partition-ID computation streaming. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        const auto buf_sz = std::min(CHUNK_SZ, n_);
        const std::size_t text_buf_bytes = (buf_sz + K - 1) * sizeof(T_sym_);
        const std::size_t p_id_buf_bytes = buf_sz * sizeof(p_id_t);
        const auto p_id_word_c = Partition_ID_Map::packed_word_c(buf_sz, P_id.bits());
        for(std::size_t b = 0; b < STREAM_C; ++b)
        {
            CUDA_CHECK(cudaMalloc(T_buf + b, text_buf_bytes));
            CUDA_CHECK(cudaMalloc(P_id_buf + b, p_id_buf_bytes));
            CUDA_CHECK(cudaMalloc(P_id_word_buf + b, p_id_word_c * sizeof(uint64_t)));
            CUDA_CHECK(pinned_allocate_host(T_stage[b], buf_sz + K - 1));
            CUDA_CHECK(pinned_allocate_host(P_id_stage[b], p_id_word_c));
        }
    }


    // Stream text in chunks to device, compute partition-IDs and per-partition counts, and move the output to host.
    {
        const idx_t chunk_count = (n_ + CHUNK_SZ - 1) / CHUNK_SZ;   // Number of chunks the text is broken into for processing in a streaming manner.
        std::vector<std::thread> dispatchers;   // Dispatchers-pool; one dispatcher per device-stream.
        std::atomic<idx_t> chunk_claimed(0);    // Count of chunks claimed for dispatch.
        for(idx_t s_id = 0; s_id < STREAM_C; ++s_id)
            dispatchers.emplace_back(
                [&, s_id]
                {
                    while(true)
                    {
                        const auto c = chunk_claimed.fetch_add(1, std::memory_order_relaxed);   // Claim a chunk.
                        if(c >= chunk_count)
                            break;

                        const auto chunk_off = c * CHUNK_SZ;    // Offset of this chunk into the text.
                        const auto chunk_suf_count = std::min(CHUNK_SZ, n_ - chunk_off);    // Count of suffixes in this chunk.
                        const auto chunk_sym_count = chunk_suf_count + K - 1;   // Number of symbols this chunk requires. The text's tail-padding covers the overhang at the last chunk.

                        // Stage this chunk and asynchronously upload it to the device.
                        std::memcpy(T_stage[s_id], T_h_ + chunk_off, chunk_sym_count * sizeof(T_sym_));
                        CUDA_CHECK(cudaMemcpyAsync(T_buf[s_id], T_stage[s_id], chunk_sym_count * sizeof(T_sym_), cudaMemcpyHostToDevice, stream[s_id]));

                        const auto block_count = (chunk_suf_count + BLOCK_SZ - 1) / BLOCK_SZ;
                        compute_partition_IDs_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[s_id] >>>
                            (T_buf[s_id], chunk_suf_count, pivots_d, P_id_buf[s_id], p_, P_sz_d);
                        CUDA_CHECK(cudaGetLastError());

                        // Pack this chunk's partition-IDs, asynchronously download those to staging space, and drain them to `P_id`.
                        // The chunks start at multiples of `BLOCK_SZ`, so their packed IDs take up disjoint words.
                        const auto group_count = (chunk_suf_count + Partition_ID_Map::GROUP_SZ - 1) / Partition_ID_Map::GROUP_SZ;
                        pack_partition_IDs_kernel<BLOCK_SZ> <<< (group_count + BLOCK_SZ - 1) / BLOCK_SZ, BLOCK_SZ, 0, stream[s_id] >>>
                            (P_id_buf[s_id], chunk_suf_count, P_id.bits(), P_id_word_buf[s_id]);
                        CUDA_CHECK(cudaGetLastError());

                        const auto word_count = Partition_ID_Map::packed_word_c(chunk_suf_count, P_id.bits());
                        CUDA_CHECK(cudaMemcpyAsync(P_id_stage[s_id], P_id_word_buf[s_id], word_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream[s_id]));
                        CUDA_CHECK(cudaStreamSynchronize(stream[s_id]));
                        P_id.store_packed(chunk_off, chunk_suf_count, P_id_stage[s_id]);
                    }
                }
            );

        for(auto& d : dispatchers)  d.join();
    }


    // Free the buffers.
    CUDA_CHECK(cudaFree(pivots_d));
    for(std::size_t b = 0; b < STREAM_C; ++b)
    {
        CUDA_CHECK(cudaFree(T_buf[b]));
        CUDA_CHECK(cudaFree(P_id_buf[b]));
        CUDA_CHECK(cudaFree(P_id_word_buf[b]));
        CUDA_CHECK(deallocate_host(T_stage[b]));
        CUDA_CHECK(deallocate_host(P_id_stage[b]));
    }

    // Move the partition-sizes to the host.
    {
        P_cap.resize(p_);
        CUDA_CHECK(cudaMemcpy(P_cap.data(), P_sz_d, p_ * sizeof(idx_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(P_sz_d));

        P_off.resize(p_);
        P_off[0] = 0;
        for(p_id_t p_id = 1; p_id < p_; ++p_id)
            P_off[p_id] = P_off[p_id - 1] + P_cap[p_id - 1];

        Active_sz = P_cap;

#ifdef STATS
        std::cerr << "Partition-size stats:\n";
        util::Collection_Stats::get_collection_stat(P_cap).print();
        std::cerr << "Partition sizes: " << util::join(P_cap) << "\n";
#endif
    }

#ifdef TESTS
    std::cerr << "Partition-IDs verification: " << (test::validate_partition_IDs(T_h_, n_, pivots, P_id) ? "Passes" : "Fails") << "\n";
#endif
}


template <typename T_sym_>
idx_t GPU_SACA<T_sym_>::active_set_size() const
{
    idx_t unsorted = 0;
    for(const auto a : Active_sz)
        unsorted += a;

    return unsorted;
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::set_context(const idx_t ctx, const idx_t gen)
{
    ctx_ = ctx;
    backend_->open_generation(gen, Active_sz);

    cohort_sz.push_back(active_set_size());
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::initial_rank()
{
    INSTRUMENT(probe::P().begin_iteration(0, 0, n_));

#ifdef TESTS
    backend_->shuffle_tiers(0, Active_sz);
#endif
#ifdef STATS
    backend_->log_barrier(0, Active_sz);
#endif

    // Upload partitions to device in a triple-buffered manner, sort-and-rank based on the associated prefixes, and download the output to host.
    {
        PROFILE_SCOPE("sort_and_init_rank_dispatch");
        set_context(ell, 0);
        drive_pipeline<true>();
    }

#ifdef TESTS
    // std::cerr << "Initial rankings verification: " << (test::validate_local_rankings(P_, sorted_key, sorted_suf) ? "Passes" : "Fails") << "\n";
#endif

    // Resize the partitions to their active-set size.
    {
        PROFILE_SCOPE("collect_active_set_sizes");
        CUDA_CHECK(cudaMemcpy(Active_sz.data(), active_set_sz_d, p_ * sizeof(idx_t), cudaMemcpyDeviceToHost));
    }

#ifdef TESTS
    backend_->shuffle_tiers(1, Active_sz);
#endif


#ifdef STATS
    {
        p_id_t sorted_part = 0; // Count of partitions that are already fully-sorted (by prefix-context size `ell`).
        const auto unsorted = active_set_size();    // Count of suffixes with non-unique ranks.
        for(p_id_t p_id = 0; p_id < p_; ++p_id)
            sorted_part += (Active_sz[p_id] == 0);

        std::cerr   << "Number of partitions fully-sorted by context-size " << ell << ": " << sorted_part << "."
                    << " Out of " << n_ << " suffixes, " << unsorted << " (" << (unsorted * 100.0 / n_) << "%) still need more resolution.\n";
        std::cerr << "Per-partition active-set sizes by context-size " << ell << ": " << util::join(Active_sz) << "\n";
        backend_->log_barrier(1, Active_sz);
    }
#endif

    INSTRUMENT(probe::P().report_log());
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::sort_and_init_rank_and_filter(const p_id_t p_id, Partition_Assembly& p_assembly, const idx_t buf_id)
{
    PROFILE_SCOPE_FMT("sort_and_init_rank_and_filter_p%d", static_cast<int>(p_id));


    const auto P_sz_max = *std::max_element(P_cap.cbegin(), P_cap.cend());
    cub::DoubleBuffer<Prefix_Key> key(key_buf[buf_id][0], key_buf[buf_id][1]);
    cub::DoubleBuffer<idx_t> suf(suf_buf[buf_id][0], suf_buf[buf_id][1]);
    cub::DoubleBuffer<p_sz_t> perm(perm_buf[buf_id][0], perm_buf[buf_id][1]);
    cub::DoubleBuffer<beta_t> beta(beta_buf[buf_id][0], beta_buf[buf_id][1]);
    uint8_t* const suf_lanes = reinterpret_cast<uint8_t*>(suf.Alternate()); // Lanes of the suffixes, uploaded for unpacking.


    // Asynchronously upload this partition's chunks of suffixes and their prefix-keys to the device.
    {
        for(auto* s : p_assembly.slab)
        {
            CUDA_CHECK(cudaMemcpyAsync(key.Current() + s->off(), s->key(), s->size() * sizeof(Prefix_Key), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(suf_lanes + lane::bytes(s->off()), s->suf(), lane::bytes(s->size()), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(beta.Current() + s->off(), s->beta(), s->size() * sizeof(beta_t), cudaMemcpyHostToDevice, stream[buf_id]));
        }
    }


    const auto part_sz = Active_sz[p_id];   // Size of this partition.
    auto workspace = workspace_buf[buf_id]; // Scratch space for `cub`.
    auto ws_sz = workspace_sz;              // Size of the workspace; `cub` takes it via a non-const reference.
    auto is_active = is_active_buf[buf_id]; // Activeness-marker flags.

    constexpr std::size_t BLOCK_SZ = 256;
    const auto block_count = (part_sz + BLOCK_SZ - 1) / BLOCK_SZ;

    Prefix_Key* K_sorted = nullptr; // Sorted prefix-keys.
    idx_t* SA_sorted = nullptr; // Suffixes corresponding to the sorted prefix-keys.
    beta_t* beta_sorted = nullptr;  // Backward tags corresponding to the key-sorted suffixes.
    Prefix_Key* K_ranked = nullptr; // Ranked keys.
    idx_t* SA_ranked = nullptr; // Suffixes in ranked order (bar the active | retired split).
    beta_t* beta_ranked = nullptr;  // Backward tags corresponding to the ranked suffixes.
    uint8_t* msg_target = nullptr;  // Lanes of the target recipient suffixes for the messages.
    msg_rank_t* msg_rank = nullptr; // Local ranks sent with the messages.
    p_sz_t* msg_off = nullptr;  // Offsets for messages into message buffers per destination-partition.
    p_sz_t* local_rank = nullptr;   // Final primary ranks computed.
    uint8_t* SA_lanes = nullptr;    // Lanes of the ranked suffixes, for persisting.


    // Unpack the suffixes, each packed with its forward tag in its backward tag-pair.
    {
        unpack_tagged_lanes_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (suf_lanes, beta.Current(), part_sz, suf.Current());
        CUDA_CHECK(cudaGetLastError());
    }


    // Sort the partition by the (packed-prefix, forward-tag): two-phase radix-sort.
    {
        // Sort by the forward tag.
        {
            // Populate identity permutation.
            iota_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            // Sort the forward tags.
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                workspace, ws_sz,
                suf, perm,  // The forward tag is packed in suffix.
                part_sz,
                static_cast<int>(bit_layout::TAG_SHIFT), static_cast<int>(bit_layout::WORD_BITS),
                stream[buf_id]
            ));

            // Shuffle keys and backward tags per the forward-tag-order permutation.
            shuffle_kernel<Prefix_Key, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (key.Alternate(), key.Current(), perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());
            key.selector ^= 1;

            shuffle_kernel<beta_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (beta.Alternate(), beta.Current(), perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());
            beta.selector ^= 1;
        }

        // Sort by the packed-prefix.
        {
            // Populate identity permutation.
            iota_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            // Sort the packed-prefixes.
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                workspace, ws_sz,
                key, perm,
                part_sz,
                Prefix_Key_Decomposer{},
                0, ELL_WORDS * 64,
                stream[buf_id]
            ));

            // Shuffle suffixes and backward tags per the packed-prefix-order permutation.
            shuffle_kernel<idx_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (suf.Alternate(), suf.Current(), perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());
            suf.selector ^= 1;

            shuffle_kernel<beta_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (beta.Alternate(), beta.Current(), perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());
            beta.selector ^= 1;

            K_sorted = key.Current(), K_ranked = key.Alternate();
            SA_sorted = suf.Current(), SA_ranked = suf.Alternate();
            beta_sorted = beta.Current(), beta_ranked = beta.Alternate();
        }
    }


    // Rank and filter the sort-output.
    {
        idx_t* const key_group_rank = reinterpret_cast<idx_t*>(K_ranked);   // Distinct key groups' ranks.
        idx_t* const key_rank = reinterpret_cast<idx_t*>(K_sorted); // Distinct keys' ranks.

        // Track groups with distinct prefixes and rank by the groups' offsets, and mark non-unique keys as active.
        {
            initialize_key_group_ranks_and_mark_active_flags_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (K_sorted, SA_sorted, part_sz, key_group_rank, is_active);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cub::DeviceScan::InclusiveScan(
                workspace, ws_sz,
                key_group_rank, key_rank,
                cub::Max(),
                part_sz,
                stream[buf_id]
            ));
        }


        // Group non-unique rank keys and suffixes to the head and the rest to the tail of the partition.
        {
            CUDA_CHECK(cub::DevicePartition::Flagged(
                workspace, ws_sz,
                thrust::make_zip_iterator(thrust::make_tuple(key_rank, SA_sorted, beta_sorted)),
                is_active,
                thrust::make_zip_iterator(thrust::make_tuple(reinterpret_cast<idx_t*>(K_ranked), SA_ranked, beta_ranked)),
                active_set_sz_d + p_id,
                part_sz,
                stream[buf_id]
            ));
        }
    }


    // Emit this generation's cohort's rank-propagation messages, collated by their destination partitions.
    {
        msg_target = reinterpret_cast<uint8_t*>(K_sorted);
        msg_rank = reinterpret_cast<msg_rank_t*>(SA_sorted);
        msg_off = msg_off_buf[buf_id];

        CUDA_CHECK(cudaMemsetAsync(msg_off, 0, (p_ + 1) * sizeof(p_sz_t), stream[buf_id]));
        count_message_destination_distribution_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (SA_ranked, reinterpret_cast<const idx_t*>(K_ranked), beta_ranked, part_sz, ctx_, msg_off);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            workspace, ws_sz,
            msg_off, msg_off,
            p_ + 1,
            stream[buf_id]
        ));

        emit_messages_kernel<false, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (SA_ranked, reinterpret_cast<const idx_t*>(K_ranked), beta_ranked, part_sz, ctx_, active_set_sz_d + p_id, msg_target, msg_rank, msg_off);
        CUDA_CHECK(cudaGetLastError());
    }


    // Narrow the rank-width for persisting as `p_sz_t`.
    {
        local_rank = reinterpret_cast<p_sz_t*>(reinterpret_cast<idx_t*>(K_ranked) + P_sz_max);  // `K_ranked[0, part_sz)` contains primary keys.
        extract_primary_keys_kernel<false, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (reinterpret_cast<const idx_t*>(K_ranked), local_rank, part_sz);
        CUDA_CHECK(cudaGetLastError());
    }


    // Pack the suffixes into lanes for persisting, over the keys narrowed by now.
    {
        SA_lanes = reinterpret_cast<uint8_t*>(K_ranked);
        pack_lanes_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (SA_ranked, part_sz, SA_lanes);
        CUDA_CHECK(cudaGetLastError());
    }


    // Asynchronously download the ranked-and-compacted partition in a chunked manner to host slabs.
    {
        CUDA_CHECK(cudaMemcpyAsync(backend_->msg_off(p_id), msg_off, p_ * sizeof(p_sz_t), cudaMemcpyDeviceToHost, stream[buf_id]));
        CUDA_CHECK(cudaMemcpyAsync(backend_->active_slot(p_id), active_set_sz_d + p_id, sizeof(idx_t), cudaMemcpyDeviceToHost, stream[buf_id]));   // For the writers to mark the retired tails.
        CUDA_CHECK(cudaStreamSynchronize(stream[buf_id]));
        const idx_t outbox_sz = backend_->msg_off(p_id)[p_ - 1];

        idx_t c_id = 0; // Chunk-ID.
        for(auto* s : p_assembly.slab)
        {
            const auto chunk_outbox_off = 2 * c_id++ * chunk_cap_;  // Offset in the outbox for this slab.
            const auto chunk_outbox_sz = (chunk_outbox_off < outbox_sz ? std::min<idx_t>(2 * chunk_cap_, outbox_sz - chunk_outbox_off) : 0);    // Count of outbox messages for this slab.
            s->set_outbox_off_and_size(chunk_outbox_off, chunk_outbox_sz);

            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<p_sz_t*>(s->key()), local_rank + s->off(), s->size() * sizeof(p_sz_t), cudaMemcpyDeviceToHost, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(s->suf(), SA_lanes + lane::bytes(s->off()), lane::bytes(s->size()), cudaMemcpyDeviceToHost, stream[buf_id]));
            if(chunk_outbox_sz > 0)
            {
                CUDA_CHECK(cudaMemcpyAsync(s->msg_target(), msg_target + lane::bytes(chunk_outbox_off), lane::bytes(chunk_outbox_sz), cudaMemcpyDeviceToHost, stream[buf_id]));
                CUDA_CHECK(cudaMemcpyAsync(s->msg_rank(), msg_rank + chunk_outbox_off, chunk_outbox_sz * sizeof(msg_rank_t), cudaMemcpyDeviceToHost, stream[buf_id]));
            }
        }
    }
}


template <typename T_sym_>
template <bool init_rank_>
void GPU_SACA<T_sym_>::drive_pipeline()
{
#ifdef PROBE
    uint64_t pipeline_ns = 0;
#endif
{
    INSTRUMENT_TIMER(pipeline_ns);

    INSTRUMENT(probe::P().set_depth_reader([this](uint64_t* const d)
    {
        d[0] = F.size_approx();
        d[1] = C.size_approx();
        d[2] = R.size_approx();
        d[3] = W.size_approx();
    }));
    INSTRUMENT(probe::P().start_sampler());

    idx_t total_chunk = 0;  // Total count of chunks across all partitions.
    p_id_t active_part_c = 0;   // Count of active partitions in the current iteration.

    std::thread read_poster;
    moodycamel::LightweightSemaphore admit(admit_budget_);
    // Post read-tasks under admission control: a partition's read-tasks are posted only if
    // `admit_budget_` has room for its whole assembly.
    {
        // Prepare read-tasks in partition order.
        for(p_id_t p_id = 0; p_id < p_; ++p_id)
        {
            if(Active_sz[p_id] == 0)    // Skip partitions with empty active-sets.
                continue;

            active_part_c++;

            const auto active_set_sz = Active_sz[p_id];
            const auto inbox_sz = backend_->inbox_size(p_id);
            const idx_t content_chunk_c = (active_set_sz + chunk_cap_ - 1) / chunk_cap_;    // Number of slabs required for this partition's content.
            const idx_t inbox_chunk_c = (inbox_sz + chunk_cap_ - 1) / chunk_cap_; // Number of slabs required for this partition's inbox.
            const idx_t chunk_c = std::max(content_chunk_c, inbox_chunk_c); // Number of slabs required in aggregate to hold this partition.
            total_chunk += chunk_c;

            // Prepare skeleton chain of chunks of this partition.
            part_assembly[p_id].slab.assign(chunk_c, nullptr);
            part_assembly[p_id].slabs_to_stage.store(chunk_c, std::memory_order_relaxed);
        }

        read_poster = std::thread(
            [&]
            {
                for(p_id_t p_id = 0; p_id < p_; ++p_id)
                {
                    if(Active_sz[p_id] == 0)    // Skip partitions with empty active-sets.
                        continue;

                    const idx_t chunk_c = part_assembly[p_id].slab.size();  // Number of slabs required to hold this partition's content and inbox.
                    for(idx_t chunk_res = 0; chunk_res < chunk_c;)  // Reserve the assembly's slabs, piecemeal.
                        chunk_res += static_cast<idx_t>(admit.waitMany(chunk_c - chunk_res));

                    const auto active_set_sz = Active_sz[p_id];
                    const auto inbox_sz = backend_->inbox_size(p_id);
                    for(idx_t c = 0; c < chunk_c; ++c) // Post a read-task for each chunk of this partition.
                    {
                        const auto chunk_off = c * chunk_cap_;
                        const auto content_chunk_sz = (chunk_off < active_set_sz ? std::min(chunk_cap_, active_set_sz - chunk_off) : 0);
                        const auto msg_chunk_sz = (chunk_off < inbox_sz ? std::min<p_sz_t>(chunk_cap_, inbox_sz - chunk_off) : 0);
                        RT.enqueue(Read_Task{p_id, c, chunk_off, content_chunk_sz, static_cast<p_sz_t>(chunk_off), msg_chunk_sz});
                    }
                }
            }
        );
    }


    std::vector<std::thread> readers;   // Readers-pool.
    std::atomic<int64_t> rt_remain(total_chunk);    // Count of read-tasks left to claim.
    // Launch the readers thread-pool to read chunks from the partition-backend to staging slabs.
    {
        for(idx_t r = 0; r < reader_c_; ++r)
            readers.emplace_back(
                [&, r]
                {
                    INSTRUMENT(probe::P().bind(probe::Role::reader, r));
                    while(rt_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Read_Task rt;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_in_ns);
                            RT.wait_dequeue(rt);    // Get a read-task.
                        }

                        Staging_Slab* s;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_slab_ns);
                            F.wait_dequeue(s);      // Get a free slab.
                        }

                        INSTRUMENT(probe::P().me().items++);

                        s->assign(rt.p_id, rt.off, rt.sz, rt.msg_off, rt.msg_sz);
                        if constexpr(init_rank_)    backend_->read_suffixes(*s);
                        else                        backend_->read_keys(*s);

                        if(s->inbox_sz() > 0)
                            backend_->read_messages(*s);

                        auto& p_assembly = part_assembly[rt.p_id];
                        p_assembly.slab[rt.chunk_idx] = s;  // Add the slab to the partition's assembly / chain.
                        C.enqueue(s);   // This slab now moves to the staging-compute queue.
                    }
                }
            );
    }


    std::vector<std::thread> stagers;   // Stagers-pool.
    std::atomic<int64_t> st_remain(total_chunk);    // Count of slabs left to stage.
    // Launch the stagers thread-pool to apply host-side compute to each slab as soon as it's been read from
    // the backend, add it to its partition's assembly, and post completed assemblies for dispatch.
    {
        for(idx_t c = 0; c < stager_c_; ++c)
            stagers.emplace_back([&, c]
                {
                    INSTRUMENT(probe::P().bind(probe::Role::stager, c));
                    while(st_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Staging_Slab* s;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_in_ns);
                            C.wait_dequeue(s);  // Get a slab with read data.
                        }

                        INSTRUMENT(probe::P().me().items++);
#ifdef TESTS
                        if(!test::validate_inbox_routing(*s, P_id, n_))
                        {
                            std::cerr << "Inboxes contain messages routed to wrong partitions. Aborting.\n";
                            std::exit(EXIT_FAILURE);
                        }
#endif
                        stage_slab<init_rank_>(*s);

                        auto& p_assembly = part_assembly[s->p_id()];
                        const auto prev = p_assembly.slabs_to_stage.fetch_sub(1, std::memory_order_acq_rel);
                        INSTRUMENT(probe::P().track_assembly(prev, p_assembly.slab.size()));
                        if(prev == 1)   // Last slab staged for this partition; chunked-assembly is ready for the device.
                        {
                            const auto p_id = s->p_id();    // Read before the assembly, `s` in it, leaves for the device.
                            R.enqueue(&p_assembly);
                            admit.signal(p_assembly.slab.size());   // These slabs are guaranteed to return to the free-pool `F`, so early signalling is safe.
                            backend_->retire_inbox(p_id);   // Its slabs have all read the inbox: the disk-part's space frees up.
                        }
                    }
                }
            );
    }


    std::vector<std::thread> writers;   // Writers-pool.
    std::atomic<int64_t> dl_remain(total_chunk);    // Count of slabs left to download from the device.
    // Launch the writers thread-pool to drain the processed slabs to the partition-backend.
    {
        for(idx_t w = 0; w < writer_c_; ++w)
            writers.emplace_back(
                [&, w]
                {
                    INSTRUMENT(probe::P().bind(probe::Role::writer, w));
                    while(dl_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Staging_Slab* s;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_in_ns);
                            W.wait_dequeue(s);  // Get a processed slab.
                        }

                        INSTRUMENT(probe::P().me().items++);
#ifdef TESTS
                        if(!test::validate_messages(*s, this, backend_->msg_off(s->p_id())))
                        {
                            std::cerr << "Message consistency verification: " << "Fails" << ". Aborting.\n";
                            std::exit(EXIT_FAILURE);
                        }
#endif
                        backend_->write(*s);    // Drain the slab's partition-content.
                        backend_->write_messages(*s);   // Drain the slab's messages.
                        F.enqueue(s);   // This slab is free now.
                    }
                }
            );
    }


    std::vector<std::thread> dispatchers;   // Dispatchers-pool; one dispatcher per device-stream.
    std::atomic<int64_t> dp_remain(active_part_c);  // Count of partitions left to dispatch to the device.
    // Launch the dispatchers threads-pool to dispatch the device-tasks, by repeatedly
    // pulling a complete assembly, dispatching it to its device-stream, and then
    // handing it over to the writers after device-execution.
    {
        for(idx_t s_id = 0; s_id < STREAM_C; ++s_id)
            dispatchers.emplace_back(
                [&, s_id]
                {
                    INSTRUMENT(probe::P().bind(probe::Role::dispatcher, s_id));
                    while(dp_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Partition_Assembly* p_assembly;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_in_ns);
                            R.wait_dequeue(p_assembly); // Get the complete chunked-assembly of a partition.
                        }

                        INSTRUMENT(probe::P().me().items++);

                        const p_id_t p_id = p_assembly - part_assembly.get();       // ID of the partition this assembly corresponds to.
#ifdef TESTS
                        {
                            idx_t read_msg = 0;
                            uint64_t read_sum = 0;  // Order-independent checksum of the messages read.
                            for(auto* s : p_assembly->slab)
                                read_msg += s->inbox_sz(),
                                read_sum += test::message_checksum(s->msg_target(), s->msg_rank(), s->inbox_sz());

                            if(read_msg != backend_->inbox_size(p_id))
                            {
                                std::cerr << "Read inbox size at current generation does not equal to written inbox size at previous generation. Aborting.\n";
                                std::exit(EXIT_FAILURE);
                            }

                            // The size-check above is by construction, as the slabs' inbox sizes derive from `inbox_size`; this one is not.
                            if(read_sum != backend_->inbox_checksum(p_id))
                            {
                                std::cerr << "Read inbox messages at current generation differ from the ones written at previous generation. Aborting.\n";
                                std::exit(EXIT_FAILURE);
                            }
                        }
#endif
                        {
                            INSTRUMENT_TIMER(probe::P().me().device_ns);
                            if constexpr(init_rank_)    sort_and_init_rank_and_filter(p_id, *p_assembly, s_id);
                            else                        sort_and_re_rank_and_filter(p_id, *p_assembly, s_id);

                            CUDA_CHECK(cudaStreamSynchronize(stream[s_id]));
                        }

                        for(auto* s : p_assembly->slab) // The processed slabs now move to the write queue.
                            W.enqueue(s);
                    }
                }
            );
    }


    // Wait for the pipeline to finish.
    {
        read_poster.join();
        for(auto& r : readers)  r.join();
        for(auto& c : stagers)  c.join();
        for(auto& d : dispatchers)  d.join();   // Dispatchers feed the write-queue, so they must drain before the writers.
        for(auto& w : writers)  w.join();

        INSTRUMENT(probe::P().stop_sampler());

        backend_->close_generation();
    }
}

    INSTRUMENT(probe::P().accumulate_pipeline_ns(pipeline_ns));
}


template <typename T_sym_>
template <bool init_rank_>
void GPU_SACA<T_sym_>::stage_slab(Staging_Slab& s)
{
    INSTRUMENT_TIMER(probe::P().me().stage_ns);

    const auto stride = ctx_ / 2;   // Previous generation's context.
    const auto sz = s.size();
    const auto* const suf = s.suf();

#ifdef TESTS
    if constexpr(!init_rank_)
    {
        const auto* const key = reinterpret_cast<const p_sz_t*>(s.key());

        // Suffixes tied after generation `t - 1` must draw their secondaries at generation `t` from the same partition.
        for(idx_t i = 1; i < sz; ++i)
            if(!test::validate_coherent_targets(lane::load(suf, i - 1), key[i - 1], lane::load(suf, i), key[i], n_, stride, ell, P_id))
            {
                std::cerr << "Target coherence invariant broken. Aborting.\n";
                std::exit(EXIT_FAILURE);
            }
    }
#endif

    auto* const beta = s.beta();
    auto* const key = s.key();
    for(idx_t i = 0; i < sz; ++i)
    {
        const auto j = bit_layout::suffix_of(lane::load(suf, i));

        if constexpr(init_rank_)
        {
            if(i + PREFIX_KEY_GATHER_PREFETCH_D < sz)
                __builtin_prefetch(T_h_ + bit_layout::suffix_of(lane::load(suf, i + PREFIX_KEY_GATHER_PREFETCH_D)), 0, 0);

            key[i] = Prefix_Key::pack_prefix(T_h_ + j);
        }

        // At generation 0, which forwards no pills, the pair's high half carries the forward tag instead.
        beta[i] = bit_layout::beta_pair(j < ctx_ ? 0 : P_id[j - ctx_],
                                        init_rank_ ? bit_layout::fwd_tag(P_id, j + ell, n_, ell) : (j < stride ? 0 : P_id[j - stride]));
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::sort_and_re_rank_and_filter(const p_id_t p_id, Partition_Assembly& p_assembly, const idx_t buf_id)
{
    PROFILE_SCOPE_FMT("sort_and_re_rank_and_filter_p%d", static_cast<int>(p_id));


    cub::DoubleBuffer<idx_t> suf(suf_buf[buf_id][0], suf_buf[buf_id][1]);
    cub::DoubleBuffer<p_sz_t> perm(perm_buf[buf_id][0], perm_buf[buf_id][1]);
    cub::DoubleBuffer<beta_t> beta(beta_buf[buf_id][0], beta_buf[buf_id][1]);
    uint8_t* const suf_lanes = reinterpret_cast<uint8_t*>(suf.Alternate()); // Lanes of the suffixes, uploaded for unpacking.

    static_assert(DEV_KEY_BYTES >= 2 * sizeof(idx_t), "Current device floor-plan requires at least two `idx_t` per entry.");
    const auto P_sz_max = *std::max_element(P_cap.cbegin(), P_cap.cend());
    idx_t* const A = reinterpret_cast<idx_t*>(key_buf[buf_id][0]);
    idx_t* const B = reinterpret_cast<idx_t*>(key_buf[buf_id][1]);
    p_sz_t* const primary = reinterpret_cast<p_sz_t*>(A);   // Bare local ranks, staged in.
    p_sz_t* const primary_suf_ordered = primary + P_sz_max; // Local ranks in suffix-order.
    idx_t* const K = B; // Composite keys, produced from join.
    p_sz_t* local_rank = nullptr;   // Final primary ranks computed.


    // Asynchronously upload this partition's chunks of suffixes and their ranks to the device.
    {
        for(auto* s : p_assembly.slab)
        {
            CUDA_CHECK(cudaMemcpyAsync(primary + s->off(), s->key(), s->size() * sizeof(p_sz_t), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(suf_lanes + lane::bytes(s->off()), s->suf(), lane::bytes(s->size()), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(beta.Current() + s->off(), s->beta(), s->size() * sizeof(beta_t), cudaMemcpyHostToDevice, stream[buf_id]));
        }
    }


    const auto part_sz = Active_sz[p_id];   // Size of the active-set of this partition.
    auto workspace = workspace_buf[buf_id]; // Scratch space for `cub`.
    auto ws_sz = workspace_sz;              // Size of the workspace; `cub` takes it via a non-const reference.
    auto is_active = is_active_buf[buf_id]; // Activeness-marker flags.

    constexpr std::size_t BLOCK_SZ = 256;
    const auto block_count = (part_sz + BLOCK_SZ - 1) / BLOCK_SZ;

    idx_t* K_sorted = nullptr;  // Sorted keys.
    idx_t* SA_sorted = nullptr; // Suffixes corresponding to the sorted keys.
    beta_t* beta_sorted = nullptr;  // Backward tags corresponding to the key-sorted suffixes.
    idx_t* K_re_ranked = nullptr;   // Re-ranked keys.
    idx_t* SA_re_ranked = nullptr;  // Suffixes in re-ranked order (bar the active | retired split).
    beta_t* beta_re_ranked = nullptr;   // Backward tags corresponding to the re-ranked suffixes.
    p_sz_t* msg_off = nullptr;  // Offsets for messages into message buffers per destination-partition.
    uint8_t* msg_target = nullptr;  // Lanes of the target recipient suffixes for the messages.
    msg_rank_t* msg_rank = nullptr; // Local ranks sent with the messages.
    uint8_t* inbox_target = nullptr;    // Lanes of the message-targets in the inbox of this partition.
    msg_rank_t* inbox_rank = nullptr;   // Message-ranks in the inbox of this partition.
    uint8_t* SA_lanes = nullptr;    // Lanes of the re-ranked suffixes, for persisting.
    const auto inbox_sz = backend_->inbox_size(p_id);   // Size of the inbox.


    // Unpack the suffixes.
    {
        unpack_lanes_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (suf_lanes, part_sz, suf.Current());
        CUDA_CHECK(cudaGetLastError());
    }


    // Sort the partition by suffix indices and consume the received messages.
    {
        iota_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (perm.Current(), part_sz);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            workspace, ws_sz,
            suf, perm,
            part_sz,
            0, static_cast<int>(std::min<idx_t>(bit_layout::SUF_BITS, lane::BITS)),
            stream[buf_id]
        ));

        // Shuffle keys and backward tags per the suffix-index-order permutation.
        shuffle_kernel<p_sz_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (primary_suf_ordered, primary, perm.Current(), part_sz);
        CUDA_CHECK(cudaGetLastError());

        shuffle_kernel<beta_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (beta.Alternate(), beta.Current(), perm.Current(), part_sz);
        CUDA_CHECK(cudaGetLastError());
        beta.selector ^= 1;
    }


    // Asynchronously upload this partition's inbox to the device.
    {
        inbox_target = reinterpret_cast<uint8_t*>(suf.Alternate()); // The suffixes' sort has freed it.
        inbox_rank = reinterpret_cast<msg_rank_t*>(B + P_sz_max);
        if(inbox_sz > P_sz_max) // By injectivity, a partition cannot receive more messages than its capacity.
        {
            std::cerr <<    "Device-space for partition does not have enough space for its messages."
                            " This is impossible due to the injectivity of rank-propagation messaging in a generation. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        for(auto* s : p_assembly.slab)
            if(s->inbox_sz() > 0)
            {
                CUDA_CHECK(cudaMemcpyAsync(inbox_target + lane::bytes(s->inbox_off()), s->msg_target(), lane::bytes(s->inbox_sz()), cudaMemcpyHostToDevice, stream[buf_id]));
                CUDA_CHECK(cudaMemcpyAsync(inbox_rank + s->inbox_off(), s->msg_rank(), s->inbox_sz() * sizeof(msg_rank_t), cudaMemcpyHostToDevice, stream[buf_id]));
            }
    }


     // Join the inbox to the partition.
    {
        // Seed every key with `SECONDARY_ABSENT` to optionally catch any suffix which hasn't received a message, during the join.
        CUDA_CHECK(cudaMemsetAsync(K, 0xFF, part_sz * sizeof(idx_t), stream[buf_id]));

        if(inbox_sz > 0)
        {
            const auto block_count = (inbox_sz + BLOCK_SZ - 1) / BLOCK_SZ;
            join_messages_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (inbox_target, inbox_rank, inbox_sz, suf.Current(), primary_suf_ordered, K, part_sz);
            CUDA_CHECK(cudaGetLastError());
        }
    }


    cub::DoubleBuffer<idx_t> key(K, A);


    // Sort the partition by (primary, secondary).
    {
        // Populate identity permutation.
        iota_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (perm.Current(), part_sz);
        CUDA_CHECK(cudaGetLastError());

        // Sort the (primary, secondary) ranks.
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            workspace, ws_sz,
            key, perm,
            part_sz,
            0, bit_layout::KEY_BITS,
            stream[buf_id]
        ));

        // Shuffle suffixes and backward tags per the (primary, secondary)-order permutation.
        shuffle_kernel<idx_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (suf.Alternate(), suf.Current(), perm.Current(), part_sz);
        CUDA_CHECK(cudaGetLastError());
        suf.selector ^= 1;

        shuffle_kernel<beta_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (beta.Alternate(), beta.Current(), perm.Current(), part_sz);
        CUDA_CHECK(cudaGetLastError());
        beta.selector ^= 1;

        K_sorted = key.Current(), K_re_ranked = key.Alternate();
        SA_sorted = suf.Current(), SA_re_ranked = suf.Alternate();
        beta_sorted = beta.Current(), beta_re_ranked = beta.Alternate();
    }


    // Re-rank and filter the sort-output.
    {
        idx_t* const key_group_rank = K_re_ranked;    // Distinct key groups' ranks.
        idx_t* const primary_key_group_rank = key_group_rank + P_sz_max;    // Distinct primary-key groups' ranks.

        // Track groups with distinct keys and distinct primary-keys, and mark non-unique keys.
        {
            rank_key_groups_and_primary_key_groups_and_mark_active_flags_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (K_sorted, part_sz, key_group_rank, primary_key_group_rank, is_active, unserved_c_d, pill_c_d);
            CUDA_CHECK(cudaGetLastError());
#ifdef TESTS
            synchronize_or_abort(stream[buf_id], "Target Coherence Invariant violated");
#endif

            CUDA_CHECK(cub::DeviceScan::InclusiveScan(
                workspace, ws_sz,
                key_group_rank, key_group_rank,
                cub::Max(),
                part_sz,
                stream[buf_id]
            ));

            CUDA_CHECK(cub::DeviceScan::InclusiveScan(
                workspace, ws_sz,
                primary_key_group_rank, primary_key_group_rank,
                cub::Max(),
                part_sz,
                stream[buf_id]
            ));
        }


        // Refine rankings to finer resolution based on the in-primary-key-group offsets of different key groups.
        {
            refine_primary_keys_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (K_sorted, part_sz, key_group_rank, primary_key_group_rank);
            CUDA_CHECK(cudaGetLastError());
        }


        // TODO: refactor all the repeated code between sort-and-initial-rank and sort-and-rank, with a `bool` template parameter.
        // Group non-unique rank keys and suffixes to the head and the rest to the tail of the partition.
        {
            CUDA_CHECK(cub::DevicePartition::Flagged(
                workspace, ws_sz,
                thrust::make_zip_iterator(thrust::make_tuple(K_sorted, SA_sorted, beta_sorted)),
                is_active,
                thrust::make_zip_iterator(thrust::make_tuple(K_re_ranked, SA_re_ranked, beta_re_ranked)),
                active_set_sz_d + p_id,
                part_sz,
                stream[buf_id]
            ));
        }
    }


    // Emit this generation's cohort's rank-propagation messages, collated by their destination partitions.
    {
        msg_target = reinterpret_cast<uint8_t*>(K_sorted);
        msg_rank = reinterpret_cast<msg_rank_t*>(SA_sorted);
        msg_off = msg_off_buf[buf_id];

        CUDA_CHECK(cudaMemsetAsync(msg_off, 0, (p_ + 1) * sizeof(p_sz_t), stream[buf_id]));
        count_message_destination_distribution_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (SA_re_ranked, K_re_ranked, beta_re_ranked, part_sz, ctx_, msg_off);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            workspace, ws_sz,
            msg_off, msg_off,
            p_ + 1,
            stream[buf_id]
        ));

        emit_messages_kernel<true, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (SA_re_ranked, K_re_ranked, beta_re_ranked, part_sz, ctx_, active_set_sz_d + p_id, msg_target, msg_rank, msg_off);
        CUDA_CHECK(cudaGetLastError());
    }


    // Narrow the rank-width for persisting as `p_sz_t`.
    {
        local_rank = reinterpret_cast<p_sz_t*>(K_re_ranked + P_sz_max);
        extract_primary_keys_kernel<true, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (K_re_ranked, local_rank, part_sz);
        CUDA_CHECK(cudaGetLastError());
    }


    // Pack the suffixes into lanes for persisting, over the keys narrowed by now.
    {
        SA_lanes = reinterpret_cast<uint8_t*>(K_re_ranked);
        pack_lanes_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (SA_re_ranked, part_sz, SA_lanes);
        CUDA_CHECK(cudaGetLastError());
    }


    // Asynchronously download the ranked-and-compacted partition in a chunked manner to host slabs.
    {
        CUDA_CHECK(cudaMemcpyAsync(backend_->msg_off(p_id), msg_off, p_ * sizeof(p_sz_t), cudaMemcpyDeviceToHost, stream[buf_id]));
        CUDA_CHECK(cudaMemcpyAsync(backend_->active_slot(p_id), active_set_sz_d + p_id, sizeof(idx_t), cudaMemcpyDeviceToHost, stream[buf_id]));   // For the writers to mark the retired tails.
        CUDA_CHECK(cudaStreamSynchronize(stream[buf_id]));
        const idx_t outbox_sz = backend_->msg_off(p_id)[p_ - 1];

        idx_t c_id = 0; // Chunk-ID.
        for(auto* s : p_assembly.slab)
        {
            const auto chunk_outbox_off = 2 * c_id++ * chunk_cap_;  // Offset in the outbox for this slab.
            const auto chunk_outbox_sz = (chunk_outbox_off < outbox_sz ? std::min<idx_t>(2 * chunk_cap_, outbox_sz - chunk_outbox_off) : 0);    // Count of outbox messages for this slab.
            s->set_outbox_off_and_size(chunk_outbox_off, chunk_outbox_sz);

            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<p_sz_t*>(s->key()), local_rank + s->off(), s->size() * sizeof(p_sz_t), cudaMemcpyDeviceToHost, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(s->suf(), SA_lanes + lane::bytes(s->off()), lane::bytes(s->size()), cudaMemcpyDeviceToHost, stream[buf_id]));
            if(chunk_outbox_sz > 0)
            {
                CUDA_CHECK(cudaMemcpyAsync(s->msg_target(), msg_target + lane::bytes(chunk_outbox_off), lane::bytes(chunk_outbox_sz), cudaMemcpyDeviceToHost, stream[buf_id]));
                CUDA_CHECK(cudaMemcpyAsync(s->msg_rank(), msg_rank + chunk_outbox_off, chunk_outbox_sz * sizeof(msg_rank_t), cudaMemcpyDeviceToHost, stream[buf_id]));
            }
        }
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::exponentiate_prefix()
{
    idx_t gen = 1;
    for(idx_t k = ell; k < n_; k *= 2, ++gen)
    {
        PROFILE_SCOPE_FMT("doubling_iter_k_%lu", static_cast<unsigned long>(k));

#ifdef STATS
        {
            const idx_t msg_count = backend_->inbox_mass_last_gen();
            const idx_t active = active_set_size();
            std::cerr <<    "Total inbox-size for generation " << gen << ": " << msg_count << "; number of recipients: " << active_set_size() <<
                            ". Amplification: " << (static_cast<double>(msg_count) / active) << "x.\n";

            // A line of its own, as the split between the tiers depends on timing, unlike the rest.
            const idx_t disk_count = backend_->disk_mass_last_gen();
            const auto& budget = backend_->budget();
            std::cerr <<    "Inbox-tiers for generation " << gen << ": " << (msg_count - disk_count) << " messages in RAM, " << disk_count << " on disk."
                            " RAM-tier: " << (budget.used() / (1024.0 * 1024.0 * 1024.0)) << "GBs charged, of " << (budget.cap() / (1024.0 * 1024.0 * 1024.0)) << "GBs cap.\n";
        }
#endif

#ifdef TESTS
        if(!test::validate_inbox_mass(*backend_))
        {
            std::cerr << "Inbox-mass validation failed for generation " << gen << ". Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
#endif

        INSTRUMENT(probe::P().begin_iteration(gen, k, std::accumulate(Active_sz.cbegin(), Active_sz.cend(), idx_t(0))));
        const auto t_0 = gpu_saca::timer::now();

        // Pass current prefix-context size to the backend so that readers can exponentiate the context-sizes.
        {
            // PROFILE_SCOPE("cpu_gather"); // TODO: relocate properly.
            set_context(2 * k, gen);
        }

        // Sort-and-rank each partition.
        {
            PROFILE_SCOPE("sort_and_rank_dispatch");
#ifdef TESTS
            CUDA_CHECK(cudaMemset(unserved_c_d, 0, sizeof(idx_t)));
            CUDA_CHECK(cudaMemset(pill_c_d, 0, sizeof(idx_t)));
#endif
            drive_pipeline<false>();
        }

        // Resize the partitions to their active-set size.
        // NB: this compaction must land *after* the preceding global rankings have been updated.
        {
            PROFILE_SCOPE("collect_active_set_sizes");
            CUDA_CHECK(cudaMemcpy(Active_sz.data(), active_set_sz_d, p_ * sizeof(idx_t), cudaMemcpyDeviceToHost));
        }

#ifdef TESTS
        {
            idx_t unserved_c_h;
            CUDA_CHECK(cudaMemcpy(&unserved_c_h, unserved_c_d, sizeof(idx_t), cudaMemcpyDeviceToHost));
            if(unserved_c_h > 0)
            {
                std::cerr << "Active suffixes with no message received: " << unserved_c_h << ". Aborting.\n";
                std::exit(EXIT_FAILURE);
            }

            idx_t pill_c_h;
            CUDA_CHECK(cudaMemcpy(&pill_c_h, pill_c_d, sizeof(idx_t), cudaMemcpyDeviceToHost));
            std::cerr << "Active suffixes with poison-pill received for generation " << gen << ": " << pill_c_h << ".\n";
        }

        backend_->shuffle_tiers(gen + 1, Active_sz);
#endif

        const auto t_1 = gpu_saca::timer::now();
        (void)t_0, (void)t_1;


        // Check for early-exit.
        {
            p_id_t sorted_part = 0; // Count of partitions that are already fully-sorted (by prefix-context size `2k`).
            const auto unsorted = active_set_size();    // Count of suffixes with non-unique ranks.
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                sorted_part += (Active_sz[p_id] == 0);
#ifdef STATS
            std::cerr   << "Number of partitions fully-sorted by context-size " << (k * 2) << ": " << sorted_part << ". Time taken in iteration: " << gpu_saca::timer::duration(t_1 - t_0) << "s."
                        << " Out of " << n_ << " suffixes, " << unsorted << " (" << (unsorted * 100.0 / n_) << "%) still need more resolution.\n";
            std::cerr << "Per-partition active-set sizes by context-size " << (k * 2) << ": " << util::join(Active_sz) << "\n";
            backend_->log_barrier(gen + 1, Active_sz);
#endif

            INSTRUMENT(probe::P().report_log());

            if(sorted_part == p_)   // All partitions have already been sorted.
                break;
        }
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::scatter_to_rank_order(const p_id_t p_id, Partition_Assembly& p_assembly, const idx_t buf_id)
{
    PROFILE_SCOPE_FMT("scatter_to_rank_order_p%d", static_cast<int>(p_id));

    static_assert(DEV_KEY_BYTES >= sizeof(p_sz_t), "Current device floor-plan requires at least one local-rank per entry.");
    uint8_t* const suf = reinterpret_cast<uint8_t*>(suf_buf[buf_id][0]);    // Lanes of the suffixes.
    p_sz_t* const rank = reinterpret_cast<p_sz_t*>(key_buf[buf_id][0]);
    idx_t* const SA_local = suf_buf[buf_id][1]; // The partition's slice of the suffix array.
    uint8_t* const SA_lanes = suf;  // Lanes of the slice, for the output; over the suffix-lanes scattered by then.
    const auto part_sz = P_cap[p_id];

    constexpr std::size_t BLOCK_SZ = 256;
    const auto block_count = (part_sz + BLOCK_SZ - 1) / BLOCK_SZ;

    // Asynchronously upload this partition's chunks of suffixes and their ranks to the device.
    {
        for(auto* s : p_assembly.slab)
        {
            CUDA_CHECK(cudaMemcpyAsync(suf + lane::bytes(s->off()), s->suf(), lane::bytes(s->size()), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(rank + s->off(), s->key(), s->size() * sizeof(p_sz_t), cudaMemcpyHostToDevice, stream[buf_id]));
        }
    }


    // Shuffle the suffixes into their rank-order.
    {
        scatter_suffixes_to_rank_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (suf, rank, SA_local, part_sz);
        CUDA_CHECK(cudaGetLastError());
    }


    // Pack the rank-ordered suffixes into lanes for the output.
    {
        pack_lanes_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
            (SA_local, part_sz, SA_lanes);
        CUDA_CHECK(cudaGetLastError());
    }


    // Asynchronously download the rank-ordered partition in a chunked manner to host slabs.
    {
        for(auto* s : p_assembly.slab)
            CUDA_CHECK(cudaMemcpyAsync(s->suf(), SA_lanes + lane::bytes(s->off()), lane::bytes(s->size()), cudaMemcpyDeviceToHost, stream[buf_id]));
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::concatenate_partitions(const std::string& op)
{
#ifdef PROBE
    uint64_t pipeline_ns = 0;
#endif
{
    INSTRUMENT_TIMER(pipeline_ns);

    INSTRUMENT(probe::P().init(reader_c_, 0, dispatcher_c_, writer_c_, slab_c_, admit_budget_));
    INSTRUMENT(probe::P().begin_iteration(cohort_sz.size(), 0, n_));
    INSTRUMENT(probe::P().set_depth_reader([this](uint64_t* const d)
    {
        d[0] = F.size_approx();
        d[1] = probe::QUEUE_ABSENT; // No host-side staging-compute in concatenation.
        d[2] = R.size_approx();
        d[3] = W.size_approx();
    }));
    INSTRUMENT(probe::P().start_sampler());

    // Open the output file and preallocate it. An empty `op` instructs
    // construction without output dump.
    const auto dump = !op.empty();
    const auto fd = dump ? ::open(op.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644) : -1; // Only owner can read-write, others read-only.
    if(dump)
    {
        if(fd < 0)
            util::file::exit_on_io_failure("open", op);

        if(::fallocate(fd, 0, 0, lane::bytes(n_)) != 0)
            util::file::exit_on_io_failure("fallocate", op);
    }


    idx_t total_chunk = 0;  // Total count of chunks across all partitions.
    p_id_t part_c = 0;  // Count of non-empty partitions.

    std::thread read_poster;
    moodycamel::LightweightSemaphore admit(admit_budget_);
    // Post read-tasks under admission control: a partition's read-tasks are posted only if
    // `admit_budget_` has room for its whole assembly.
    {
        // Prepare read-tasks in partition order.
        for(p_id_t p_id = 0; p_id < p_; ++p_id)
        {
            if(P_cap[p_id] == 0)    // Skip empty partitions; they have no content to concatenate.
                continue;

            part_c++;

            const idx_t chunk_c = (P_cap[p_id] + chunk_cap_ - 1) / chunk_cap_;  // Number of slabs required to hold this partition.
            total_chunk += chunk_c;

            // Prepare skeleton chain of chunks of this partition.
            part_assembly[p_id].slab.assign(chunk_c, nullptr);
            part_assembly[p_id].slabs_to_stage.store(chunk_c, std::memory_order_relaxed);
        }

        read_poster = std::thread(
            [&]
            {
                for(p_id_t p_id = 0; p_id < p_; ++p_id)
                {
                    if(P_cap[p_id] == 0)    // Skip empty partitions.
                        continue;

                    const idx_t chunk_c = part_assembly[p_id].slab.size();  // Number of slabs required to hold this partition.
                    for(idx_t chunk_res = 0; chunk_res < chunk_c;)  // Reserve the assembly's slabs, piecemeal.
                        chunk_res += static_cast<idx_t>(admit.waitMany(chunk_c - chunk_res));

                    const auto sz = P_cap[p_id];
                    for(idx_t c = 0; c < chunk_c; ++c) // Post a read-task for each chunk of this partition.
                    {
                        const auto chunk_off = c * chunk_cap_;
                        RT.enqueue(Read_Task{p_id, c, chunk_off, std::min(chunk_cap_, sz - chunk_off), 0, 0});
                    }
                }
            }
        );
    }


    std::vector<std::thread> readers;   // Readers-pool.
    std::atomic<int64_t> rt_remain(total_chunk);    // Count of read-tasks left to claim.
    // Launch the readers thread-pool to read chunks from the partition-backend to staging slabs, and post
    // completed assemblies for dispatch.
    {
        for(idx_t r = 0; r < reader_c_; ++r)
            readers.emplace_back(
                [&, r]
                {
                    INSTRUMENT(probe::P().bind(probe::Role::reader, r));
                    while(rt_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Read_Task rt;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_in_ns);
                            RT.wait_dequeue(rt);    // Get a read-task.
                        }

                        Staging_Slab* s;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_slab_ns);
                            F.wait_dequeue(s);      // Get a free slab.
                        }

                        INSTRUMENT(probe::P().me().items++);

                        s->assign(rt.p_id, rt.off, rt.sz, rt.msg_off, rt.msg_sz);
                        backend_->read_keys(*s);
                        backend_->retire_chunk(rt.p_id, rt.chunk_idx);  // Read for the last time: its memory goes now.

                        auto& p_assembly = part_assembly[rt.p_id];
                        p_assembly.slab[rt.chunk_idx] = s;  // Add the slab to the partition's assembly / chain.
                        const auto prev = p_assembly.slabs_to_stage.fetch_sub(1, std::memory_order_acq_rel);
                        INSTRUMENT(probe::P().track_assembly(prev, p_assembly.slab.size()));
                        if(prev == 1)   // Last slab read for this partition; chunked-assembly is ready for the device.
                        {
                            backend_->retire_partition(rt.p_id);    // The partition's backend is dead onward.
                            R.enqueue(&p_assembly);
                            admit.signal(p_assembly.slab.size());   // These slabs are guaranteed to return to the free-pool `F`, so early signalling is safe.
                        }
                    }
                }
            );
    }


    std::vector<std::thread> writers;   // Writers-pool.
    std::atomic<int64_t> dl_remain(total_chunk);    // Count of slabs left to download from the device.
    // Launch the writers thread-pool to drain the processed slabs to the output file.
    {
        for(idx_t w = 0; w < writer_c_; ++w)
            writers.emplace_back(
                [&, w]
                {
                    INSTRUMENT(probe::P().bind(probe::Role::writer, w));
                    while(dl_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Staging_Slab* s;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_in_ns);
                            W.wait_dequeue(s);  // Get a processed slab.
                        }

                        INSTRUMENT(probe::P().me().items++);

                        {
                            INSTRUMENT_TIMER(probe::P().me().io_ns);
                            INSTRUMENT(probe::P().me().bytes += lane::bytes(s->size()));
                            INSTRUMENT(probe::P().me().disk_bytes += (dump ? lane::bytes(s->size()) : 0));

                            // Drain the slab's suffixes.
                            if(dump)
                                util::file::write_and_flush(fd, op, s->suf(), lane::bytes(s->size()), lane::bytes(P_off[s->p_id()] + s->off()));
                        }

                        F.enqueue(s);   // This slab is free now.
                    }
                }
            );
    }


    std::vector<std::thread> dispatchers;   // Dispatchers-pool; one dispatcher per device-stream.
    std::atomic<int64_t> dp_remain(part_c); // Count of partitions left to dispatch to the device.
    // Launch the dispatchers threads-pool to dispatch the device-tasks, by repeatedly
    // pulling a complete assembly, dispatching it to its device-stream, and then
    // handing it over to the writers after device-execution.
    {
        for(idx_t s_id = 0; s_id < STREAM_C; ++s_id)
            dispatchers.emplace_back(
                [&, s_id]
                {
                    INSTRUMENT(probe::P().bind(probe::Role::dispatcher, s_id));
                    while(dp_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Partition_Assembly* p_assembly;
                        {
                            INSTRUMENT_TIMER(probe::P().me().wait_in_ns);
                            R.wait_dequeue(p_assembly); // Get the complete chunked-assembly of a partition.
                        }

                        INSTRUMENT(probe::P().me().items++);

                        const p_id_t p_id = p_assembly - part_assembly.get();       // ID of the partition this assembly corresponds to.

                        {
                            INSTRUMENT_TIMER(probe::P().me().device_ns);
                            scatter_to_rank_order(p_id, *p_assembly, s_id);

                            CUDA_CHECK(cudaStreamSynchronize(stream[s_id]));
                        }

                        for(auto* s : p_assembly->slab) // The processed slabs now move to the write queue.
                            W.enqueue(s);
                    }
                }
            );
    }


    // Wait for the pipeline to finish.
    {
        read_poster.join();
        for(auto& r : readers)  r.join();
        for(auto& d : dispatchers)  d.join();   // Dispatchers feed the write-queue, so they must drain before the writers.
        for(auto& w : writers)  w.join();

        INSTRUMENT(probe::P().stop_sampler());
    }

#ifdef TESTS
    if(backend_->budget().used() != 0)  // Every chunk is retired, and every charge returned.
    {
        std::cerr << "The host-memory budget has " << backend_->budget().used() << " bytes charged past the concatenation. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }
#endif


    // Close the output file.
    if(dump)
    {
        // The earlier per-chunk `sync_file_range`s only queued writebacks, with no guarantees on
        // actual completions. Therefore block until the writes physically complete.
        if(::fsync(fd) != 0)
            util::file::exit_on_io_failure("fsync", op);

        if(::close(fd) != 0)
            util::file::exit_on_io_failure("close", op);
    }
}

    INSTRUMENT(probe::P().accumulate_pipeline_ns(pipeline_ns));
    INSTRUMENT(probe::P().report_log());
}



// Template instantiations for the required instances.
template class GPU_SACA<uint8_t>;
template class GPU_SACA<uint16_t>;
template class GPU_SACA<uint32_t>;

}
