
#include "GPU_SACA.cuh"
#include "Suffix_Buffer.hpp"
#include "Packed_Prefix_Decomposer.cuh"
#include "bit-layout.hpp"
#include "utility.hpp"
#include "utility.cuh"
#include "env.hpp"
#include "tests.hpp"
#include "isa-instrument.hpp"
#include "profile.hpp"

#include "omp.h"
#include "cub/cub.cuh"
#include "thrust/iterator/zip_iterator.h"

#include <unistd.h>

#include <cstdint>
#include <cstddef>
#include <limits>
#include <atomic>
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
GPU_SACA<T_sym_>::GPU_SACA(const T_sym_* const T, const idx_t n):
      n_(n)
    , T_h_(T)
    , p_(0)
    , secondary_bits_(util::math::bits_for(n_))
    , key_bits_(0)
    , P_id(nullptr)
    , rho(nullptr)
    , null_d(nullptr)
    , workspace_sz(0)
{
    P_id = util::allocate<p_id_t>(n_);
    rho = util::allocate<idx_t>(n_);

    for(std::size_t s = 0; s < STREAM_C; ++s)
        CUDA_CHECK(cudaStreamCreate(stream + s));

    CUDA_CHECK(cudaMalloc(&null_d, (K - 1) * sizeof(T_sym_)));
    std::vector<T_sym_> null_h(K - 1, std::numeric_limits<T_sym_>::min());
    CUDA_CHECK(cudaMemcpy(null_d, null_h.data(), (K - 1) * sizeof(T_sym_), cudaMemcpyHostToDevice));

    for(std::size_t b = 0; b < STREAM_C; ++b)
    {
        for(std::size_t r = 0; r < 2; ++r)
            key_buf[b][r] = nullptr,
            suf_buf[b][r] = nullptr,
            perm_buf[b][r] = nullptr,
            beta_buf[b][r] = nullptr;

        workspace_buf[b] = nullptr;
        is_active_buf[b] = nullptr;
    }

    active_set_sz_d = nullptr;
}


template <typename T_sym_>
GPU_SACA<T_sym_>::~GPU_SACA()
{
    CUDA_CHECK(cudaFree(null_d));

    util::deallocate(P_id);
    util::deallocate(rho);

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
    }

    CUDA_CHECK(cudaFree(active_set_sz_d));
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::build_SA(idx_t*& SA)
{
    using namespace gpu_saca::timer;
    PROFILE_SCOPE("build_SA");

    // Pin the host-side text so that PCIe transfers of text can be overlapped kernel execution.
    {
        PROFILE_SCOPE("host_register_text");
        CUDA_CHECK(cudaHostRegister(const_cast<T_sym_*>(T_h_), n_ * sizeof(T_sym_), cudaHostRegisterDefault));
    }

    const auto t_0 = now();

    {   PROFILE_SCOPE("select_pivots");             select_pivots();    }
    const auto t_1 = now();
    std::cerr << "Pivots selection completed. Time taken: " << duration(t_1 - t_0) << "s.\n";

    {   PROFILE_SCOPE("partition");                 partition();    }
    const auto t_2 = now();
    std::cerr << "Partitioning completed. Time taken: " << duration(t_2 - t_1) << "s.\n";

    // Unpin the host-side text as we do not need the text anymore.
    {
        PROFILE_SCOPE("host_unregister_text");
        CUDA_CHECK(cudaHostUnregister(const_cast<T_sym_*>(T_h_)));
    }

    {
        PROFILE_SCOPE("allocate_device_space");
        if(!allocate_device_space())
        {
            std::cerr << "Too low device memory available for the process. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
    }

    // Prepare the partition-backend and allocate staging slabs bridging the backend to the device.
    {
        PROFILE_SCOPE("setup_staging");
        setup_staging();
    }

    {   PROFILE_SCOPE("initial_rank");              initial_rank();}
    const auto t_3 = now();
    std::cerr << "Initial ranking completed. Time taken: " << duration(t_3 - t_2) << "s.\n";

    {   PROFILE_SCOPE("exponentiate_prefix");       exponentiate_prefix();}
    const auto t_4 = now();
    std::cerr << "Prefix-exponentiation completed. Time taken: " << duration(t_4 - t_3) << "s.\n";

    {   PROFILE_SCOPE("concatenate_partitions");    concatenate_partitions(SA);}

    {   PROFILE_SCOPE("drop_backend");              backend_.reset();   spill_->cleanup();   }

    const auto t_5 = now();
    std::cerr << "Partition-concatenation completed. Time taken: " << duration(t_5 - t_4) << "s.\n";
}


template <typename T_sym_>
std::size_t GPU_SACA<T_sym_>::fixed_budget() const
{
    return  n_ * sizeof(T_sym_) +   // `T`
            n_ * sizeof(p_id_t) +   // `Π`
            n_ * sizeof(idx_t) +    // `ρ`
            slab_pool_memory() +
            Suffix_Buffer::scatter_buffer_memory(omp_get_max_threads(), p_);
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::select_pivots()
{
    const idx_t p_sz_max = partition_size_upper_bound();    // Maximum supported partition-size based on the device-memory budget.
    const idx_t p_min = (n_ + p_sz_max - 1) / p_sz_max;     // Minimum number of partitions.
    const idx_t p_max = std::min<idx_t>(PARTITION_SLACK_FACTOR * p_min, MAX_PART_COUNT);    // Maximum number of partitions.
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

        if(!bit_layout::validate(n_, p_))
        {
            std::cerr << "Bit-packing layout violated. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
    }
}


template <typename T_sym_>
idx_t GPU_SACA<T_sym_>::partition_size_upper_bound() const
{
    // Device-space requirement for a suffix: it occupies space in two ping-pong partition-buffers for the radix-sort, plus one flag byte to mark its activeness in subsequent steps.
    // 2 x (key + suffix + permutation-index + backward tag) ping-pong + activeness flag.
    constexpr std::size_t bytes_per_suf = 2 * (DEV_KEY_BYTES + sizeof(idx_t) + sizeof(p_sz_t) + sizeof(p_id_t)) + sizeof(uint8_t);
    std::size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const std::size_t raw_budget = free_mem * (1.0 - MEM_SAFETY_MARGIN);    // Total usable memory for the algorithm.
    const double imb_slack = env::get_f64("GPU_SACA_IMBALANCE_SLACK").value_or(PART_SZ_IMBALANCE_SLACK);
    const std::size_t budget = raw_budget / imb_slack;  // The actual partition-sizes, to be found out later, can be larger due to the probabilistic nature of pivoting.

    idx_t p_sz_max = budget / (STREAM_C * bytes_per_suf);   // Max count of suffixes per partition so that `STREAM_C` partitions can fit concurrently in the worst-case.
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

    // Cap partition-sizes so the 64-bit packed (primary, secondary) key of suffixes does not overflow.
    {
        const idx_t p_sz_max_from_packing = (secondary_bits_ == 0 ? std::numeric_limits<idx_t>::max() :
                                                (secondary_bits_ == 64 ? 1 :
                                                    static_cast<idx_t>((idx_t(1) << (64 - secondary_bits_)) / PART_SZ_IMBALANCE_SLACK)));
        p_sz_max = std::min(p_sz_max, p_sz_max_from_packing);
    }

    const idx_t p_min_from_budget = std::max(STREAM_C, (n_ + p_sz_max - 1) / p_sz_max); // Lower-bound on the number partitions.
    idx_t p_min = p_min_from_budget;

    // Optional override of required minimum partition-count.
    {
        if(const auto p_target = env::get_u64("GPU_SACA_P_TARGET"); p_target)
        {
            if(*p_target < p_min_from_budget)
                std::cerr << "The provided required minimum partition-count falls below device-budget minimum. Discarding it.";
            else
                p_min = *p_target;
        }
    }

    // The (primary, secondary) key packing scheme, or very large `n`, can demand more partitions than the partition-ID type can index.
    if(p_min > MAX_PART_COUNT)
    {
        std::cerr   << "Current partitioning policy requires at least " << p_min << " partitions for the " << n_ << "-length text. "
                    << "But `p_id_t` can index at most " << MAX_PART_COUNT << " partitions. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    const auto p_sz_bound = (n_ + p_min - 1) / p_min;   // Maximum supported size of a partition based on the minimum partition-count.

    return p_sz_bound;
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
        auto in_zip  = thrust::make_zip_iterator(thrust::make_tuple(static_cast<idx_t*>(nullptr), static_cast<idx_t*>(nullptr)));
        auto zip_out = thrust::make_zip_iterator(thrust::make_tuple(static_cast<idx_t*>(nullptr), static_cast<idx_t*>(nullptr)));
        CUDA_CHECK(cub::DevicePartition::Flagged(
            nullptr, compaction_workspace_sz,
            in_zip, static_cast<uint8_t*>(nullptr),
            zip_out,
            static_cast<idx_t*>(nullptr),
            part_sz
        ));
    }

    return std::max({init_rank_workspace_sz, exponentiation_workspace_size, scan_workspace_sz, compaction_workspace_sz});
}


template <typename T_sym_>
bool GPU_SACA<T_sym_>::allocate_device_space()
{
    // Each suffix in each stream needs:
    //      1. `cub::DoubleBuffer` keys, input and output (2 x `DEV_KEY_BYTES`);
    //      2. `cub::DoubleBuffer` values, input and output (2 x `idx_t`);
    //      3. `cub::DoubleBuffer` permutation-indices, generated and transformed (2 x `p_sz_t`);
    //      4. `cub::DoubleBuffer` backward routing tags, input and output (2 x `p_id_t`);
    //      5. activeness-flag, output (1 x `uint8_t`).
    // `cub` itself requires a workspace for sort and scan.
    // Memory constraint: STREAM_C x (P_sz_max x (2 x (DEV_KEY_BYTES + sizeof(idx_t) + sizeof(p_sz_t)) + sizeof(uint8_t)) + workspace_sz) <= free_mem x (1 - MEM_SAFETY_MARGIN)

    const idx_t P_sz_max = *std::max_element(P_cap.cbegin(), P_cap.cend());

    // Compute required workspaces-sizes for the sort and the scan.
    workspace_sz = cub_workspace_size(P_sz_max);

    // Check if the device has enough space for the buffers.
    {
        std::size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        const std::size_t budget = free_mem * (1.0 - MEM_SAFETY_MARGIN);    // Total memory across all the streams.
        const std::size_t per_stream_budget = P_sz_max * (2 * (DEV_KEY_BYTES + sizeof(idx_t) + sizeof(p_sz_t) + sizeof(p_id_t)) + sizeof(uint8_t)) + workspace_sz;
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
                CUDA_CHECK(cudaMalloc(&beta_buf[b][r], P_sz_max * sizeof(p_id_t)));
            }

            CUDA_CHECK(cudaMalloc(&workspace_buf[b], workspace_sz));
            CUDA_CHECK(cudaMalloc(&is_active_buf[b], P_sz_max * sizeof(uint8_t)));
        }

        CUDA_CHECK(cudaMalloc(&active_set_sz_d, p_ * sizeof(idx_t)));
    }

    return true;
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::setup_staging_config()
{
    const auto P_sz_max = *std::max_element(P_cap.cbegin(), P_cap.cend());

    // Sets the capacity of each staging-slab.
    {
        if(P_sz_max >= CHUNK_CAP)
            chunk_cap_ = CHUNK_CAP;
        else    // Small partitions can render disk-block-rounded reads to overflow the slab's capacity, by up-to 4095 bytes.
        {
            constexpr idx_t elems_per_disk_blk = Direct_File::DISK_BLOCK / sizeof(idx_t);
            chunk_cap_ = ((P_sz_max + elems_per_disk_blk - 1) / elems_per_disk_blk) * elems_per_disk_blk;
        }
    }

    // Set the reader- and writer-pool sizes.
    {
        writer_c_ = env::get_u64("GPU_SACA_WRITER_C").value_or(8);  // TODO: tune default.
        reader_c_ = env::get_u64("GPU_SACA_READER_C").value_or(std::max<idx_t>(1, omp_get_max_threads()));
        if(reader_c_ + writer_c_ + 1 > static_cast<idx_t>(omp_get_max_threads()))
            std::cerr << "Warning: the staging workers (readers, writers, and dispatcher) oversubscribes the " << omp_get_max_threads() << "available CPU-threads.\n";

        INSTRUMENT(isa_instrument::ISA_Instrument().init(reader_c_));
    }

    // Sets the slab pool size.
    {
        const auto max_chunk_per_part = (P_sz_max + chunk_cap_ - 1) / chunk_cap_;

        // `STREAM_C` partitions in the GPU; `PREFETCH_DEPTH` partitions staged ahead in the ready-queue;
        // `WRITEBACK_DEPTH` partitions staged ahead in the write-queue; a slab per-reader and per-writer
        slab_c_ = (STREAM_C + PREFETCH_DEPTH + WRITEBACK_DEPTH) * max_chunk_per_part + reader_c_ + writer_c_;
    }

    // Log the staging configurations.
    {
#ifdef STATS
        std::cerr << "Staging-configuration between backend and device:\n";
        std::cerr << "\tReader-pool size: " << reader_c_ << "\n";
        std::cerr << "\tWriter-pool size: " << writer_c_ << "\n";
        std::cerr << "\tSlab-pool size: " << slab_c_ << "\n";
        std::cerr << "\tSlab-capacity:  " << chunk_cap_ << "\n";
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

    for(idx_t s_id = 0; s_id < STREAM_C; ++s_id)
        FS.enqueue(s_id);

    part_assembly = std::make_unique<Partition_Assembly[]>(p_);
}


template <typename T_sym_>
std::size_t GPU_SACA<T_sym_>::slab_pool_memory() const
{
    return slab_c_ * chunk_cap_ * (sizeof(idx_t) + sizeof(Prefix_Key));
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::partition()
{
    {
        PROFILE_SCOPE("compute_partition_IDs");
        compute_partition_IDs();
    }

    // Move the suffixes (with some associated prefixes) to their corresponding partitions.
    {
        PROFILE_SCOPE("scatter_to_partition");
        std::vector<util::Padded<Suffix_Buffer>> P_buf;

        // Initialize the partitions' spill-stores and their prefix-key buffers.
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
            spill_ = std::make_unique<Spill_Store>(std::move(roots), "run_" + std::to_string(::getpid()));
            backend_ = std::make_unique<Partition_Backend>(P_cap, *spill_, fixed_budget(), rho, n_, secondary_bits_, P_id);

            P_buf.reserve(p_);
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                P_buf.emplace_back(Suffix_Buffer{backend_->resident_suf(p_id), backend_->resident_prefix(p_id), backend_->spilled_suf(p_id), backend_->spilled_prefix(p_id)});
        }


        // Add suffix indices along with their `ELL_WORDS x 8`-byte prefixes to their corresponding partitions (through the buffers).
        {
            PROFILE_SCOPE("omp_scatter");
            #pragma omp parallel for schedule(static)
            for(idx_t i = 0; i < n_; ++i)
            {
                const p_id_t phi = (i + ell < n_ ? P_id[i + ell] : 0);  // Forward tag for generation 0.
                P_buf[P_id[i]].unwrap().add(bit_layout::pack(i, phi), Prefix_Key::pack_prefix(T_h_ + i));   // Suffix packed with its forward tag, seeded at generation 0.
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
            std::cerr << "Partition-contents verification: " << (test::validate_partition_contents(T_h_, n_, P_id, P_cap, *backend_) ? "Passes" : "Fails") << "\n";
        }
#endif
    }
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
    idx_t CHUNK_SZ; // Number of suffixes at each text-chunk.
    T_sym_* T_buf[STREAM_C];    // The text-chunk buffers.
    p_id_t* P_id_buf[STREAM_C]; // The partition-ID buffers.

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
        const std::size_t per_idx_mem = STREAM_C * (sizeof(T_sym_) + sizeof(p_id_t));   // Space required per-index across the three sets of buffers.
        CHUNK_SZ = budget / per_idx_mem;
        CHUNK_SZ = (CHUNK_SZ / BLOCK_SZ) * BLOCK_SZ;    // Round-down to the nearest multiple of `BLOCK_SZ` for clean grid-partitioning.
        if(CHUNK_SZ < BLOCK_SZ) // Too low memory in the GPU to support even one block.
        {
            std::cerr << "Too low device memory available for partition-ID computation streaming. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        const std::size_t text_buf_bytes = (CHUNK_SZ + K - 1) * sizeof(T_sym_);
        const std::size_t p_id_buf_bytes = CHUNK_SZ * sizeof(p_id_t);
        for(std::size_t b = 0; b < STREAM_C; ++b)
        {
            CUDA_CHECK(cudaMalloc(T_buf + b, text_buf_bytes));
            CUDA_CHECK(cudaMalloc(P_id_buf + b, p_id_buf_bytes));
        }
    }


    // Stream text in chunks to device, compute partition-IDs and per-partition counts, and move the output to host.
    {
        const idx_t chunk_count = (n_ + CHUNK_SZ - 1) / CHUNK_SZ;   // Number of chunks the text is broken into for processing in a streaming manner.
        for(idx_t c = 0; c < chunk_count; ++c)  // Triple-buffered chunk loop.
        {
            const auto buf_id = c % STREAM_C;   // Which set of buffers (and stream) to use for this chunk.
            const auto chunk_off = c * CHUNK_SZ;    // Offset of this chunk into the text.
            const auto chunk_suf_count = std::min(CHUNK_SZ, n_ - chunk_off);    // Count of suffixes in this chunk.
            const auto chunk_symbol_req = chunk_suf_count + K - 1;  // Number of symbols this chunk is required to have.
            const auto chunk_symbol_count = std::min(chunk_symbol_req, n_ - chunk_off);    // Number of symbols to transfer from host to device for this chunk.

            //  Asynchronously transfer this chunk to the device while the other stream's kernel is being executed (from the last iteration).
            CUDA_CHECK(cudaMemcpyAsync(T_buf[buf_id], T_h_ + chunk_off, chunk_symbol_count * sizeof(T_sym_), cudaMemcpyHostToDevice, stream[buf_id]));

            // Pad the tail of the chunk with null / terminator symbols if required.
            if(chunk_symbol_count < chunk_symbol_req)   // Can happen at the last and at the previous-to-last chunks.
                CUDA_CHECK(cudaMemcpyAsync(T_buf[buf_id] + chunk_symbol_count, null_d, (chunk_symbol_req - chunk_symbol_count) * sizeof(T_sym_), cudaMemcpyDeviceToDevice, stream[buf_id]));

            const auto block_count = (chunk_suf_count + BLOCK_SZ - 1) / BLOCK_SZ;
            compute_partition_IDs_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (T_buf[buf_id], chunk_suf_count, pivots_d, P_id_buf[buf_id], p_, P_sz_d);
            CUDA_CHECK(cudaGetLastError());

            // Asynchronously copy this chunk's partition-IDs to the host.
            CUDA_CHECK(cudaMemcpyAsync(P_id + chunk_off, P_id_buf[buf_id], chunk_suf_count * sizeof(p_id_t), cudaMemcpyDeviceToHost, stream[buf_id]));
        }
    }


    // Wait for the streams to finish.
    for(std::size_t s = 0; s < STREAM_C; ++s)
        CUDA_CHECK(cudaStreamSynchronize(stream[s]));

    // Free the buffers.
    CUDA_CHECK(cudaFree(pivots_d));
    for(std::size_t b = 0; b < STREAM_C; ++b)
    {
        CUDA_CHECK(cudaFree(T_buf[b]));
        CUDA_CHECK(cudaFree(P_id_buf[b]));
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
#endif
    }

    // Determine bit-width of primary-keys of suffixes.
    {
        const idx_t P_sz_max = *std::max_element(P_cap.cbegin(), P_cap.cend());
        key_bits_ = secondary_bits_ + util::math::bits_for(P_sz_max);
        if(key_bits_ > 64)
        {
            std::cerr << "Packed (primary, secondary) key needs " << key_bits_ << ". The skew in partitioning is adversarial. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
    }

#ifdef TESTS
    std::cerr << "Partition-IDs verification: " << (test::validate_partition_IDs(T_h_, n_, pivots, P_id) ? "Passes" : "Fails") << "\n";
#endif
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::initial_rank()
{
    INSTRUMENT(isa_instrument::ISA_Instrument().begin_iteration(0, 0, n_));

    // Upload partitions to device in a triple-buffered manner, sort-and-rank based on the associated prefixes, and download the output to host.
    {
        PROFILE_SCOPE("sort_and_init_rank_dispatch");
        backend_->set_context(ell);
        drive_pipeline<true>();
    }

    // Drop the prefix-keys (used in seeding) from the backend.
    {
        PROFILE_SCOPE("drop_prefixes");
        backend_->drop_prefixes();
    }

#ifdef TESTS
    // std::cerr << "Initial rankings verification: " << (test::validate_local_rankings(P_, sorted_key, sorted_suf) ? "Passes" : "Fails") << "\n";
#endif


    // Compute the initial global ranking.
    {
        PROFILE_SCOPE("update_global_ranks");
        update_global_ranks();}

    // Resize the partitions to their active-set size.
    // NB: this compaction must land *after* the preceding global rankings have been updated.
    {
        PROFILE_SCOPE("collect_active_set_sizes");
        CUDA_CHECK(cudaMemcpy(Active_sz.data(), active_set_sz_d, p_ * sizeof(idx_t), cudaMemcpyDeviceToHost));
    }


#ifdef STATS
    {
        p_id_t sorted_part = 0; // Count of partitions that are already fully-sorted (by prefix-context size `ell`).
        idx_t unsorted = 0; // Count of suffixes with non-unique ranks.
        for(p_id_t p_id = 0; p_id < p_; ++p_id)
            sorted_part += (Active_sz[p_id] == 0),
            unsorted += Active_sz[p_id];

        std::cerr   << "Number of partitions fully-sorted by context-size " << ell << ": " << sorted_part << "."
                    << " Out of " << n_ << " suffixes, " << unsorted << " (" << (unsorted * 100.0 / n_) << "%) still need more resolution.\n";
    }
#endif

    INSTRUMENT(isa_instrument::ISA_Instrument().report_log());
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::sort_and_init_rank_and_filter(const p_id_t p_id, Partition_Assembly& p_assembly, const idx_t buf_id)
{
    PROFILE_SCOPE_FMT("sort_and_init_rank_and_filter_p%d", static_cast<int>(p_id));


    // Asynchronously upload this partition's chunks of suffixes and their prefix-keys to the device.
    {
        for(auto* s : p_assembly.slab)
        {
            CUDA_CHECK(cudaMemcpyAsync(key_buf[buf_id][0] + s->off(), s->key(), s->size() * sizeof(Prefix_Key), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(suf_buf[buf_id][0] + s->off(), s->suf(), s->size() * sizeof(idx_t), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(beta_buf[buf_id][0] + s->off(), s->beta(), s->size() * sizeof(p_id_t), cudaMemcpyHostToDevice, stream[buf_id]));
        }
    }


    const auto part_sz = Active_sz[p_id];   // Size of this partition.
    auto workspace = workspace_buf[buf_id]; // Scratch space for `cub`.
    auto is_active = is_active_buf[buf_id]; // Activeness-marker flags.

    constexpr std::size_t BLOCK_SZ = 256;
    const auto block_count = (part_sz + BLOCK_SZ - 1) / BLOCK_SZ;

    Prefix_Key* K_sorted = nullptr; // Sorted prefix-keys.
    idx_t* SA_sorted = nullptr; // Sorted suffixes.
    Prefix_Key* K_ranked = nullptr; // Ranked keys.
    idx_t* SA_ranked = nullptr; // Suffixes in ranked order (bar the active | retired split).
    p_id_t* beta_ranked = nullptr;  // Backward tags corresponding to the ranked suffixes.


    // Sort the partition by the (packed-prefix, forward-tag): two-phase radix-sort.
    {
        cub::DoubleBuffer<idx_t> suf(suf_buf[buf_id][0], suf_buf[buf_id][1]);
        cub::DoubleBuffer<p_sz_t> perm(perm_buf[buf_id][0], perm_buf[buf_id][1]);
        cub::DoubleBuffer<p_id_t> beta(beta_buf[buf_id][0], beta_buf[buf_id][1]);

        // Sort by the forward tag.
        {
            // Populate identity permutation.
            iota_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            // Sort the forward tags.
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                workspace, workspace_sz,
                suf, perm,  // The forward tag is packed in suffix.
                part_sz,
                static_cast<int>(bit_layout::TAG_SHIFT), static_cast<int>(bit_layout::WORD_BITS),
                stream[buf_id]
            ));

            // Shuffle keys and backward tags per the forward-tag-sorted permutation.
            shuffle_kernel<Prefix_Key, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (key_buf[buf_id][1], key_buf[buf_id][0], perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            shuffle_kernel<p_id_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (beta_buf[buf_id][1], beta_buf[buf_id][0], perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());
        }

        // Sort by the packed-prefix.
        {
            // Populate identity permutation.
            iota_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            // Sort the packed-prefixes.
            cub::DoubleBuffer<Prefix_Key> key(key_buf[buf_id][1], key_buf[buf_id][0]);
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                workspace, workspace_sz,
                key, perm,
                part_sz,
                Prefix_Key_Decomposer{},
                0, ELL_WORDS * 64,
                stream[buf_id]
            ));

            // Shuffle suffixes and backward tags per the packed-prefix-sorted permutation.
            shuffle_kernel<idx_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (suf.Alternate(), suf.Current(), perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            shuffle_kernel<p_id_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (beta_buf[buf_id][0], beta_buf[buf_id][1], perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            K_sorted = key.Current(), K_ranked = key.Alternate();
            SA_sorted = suf.Alternate(), SA_ranked = suf.Current();
            beta_ranked = beta_buf[buf_id][0];
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
                workspace, workspace_sz,
                key_group_rank, key_rank,
                cub::Max(),
                part_sz,
                stream[buf_id]
            ));
        }


        // Group non-unique rank keys and suffixes to the head and the rest to the tail of the partition.
        {
            CUDA_CHECK(cub::DevicePartition::Flagged(
                workspace, workspace_sz,
                thrust::make_zip_iterator(thrust::make_tuple(static_cast<const idx_t*>(key_rank), static_cast<idx_t*>(SA_sorted))),
                is_active,
                thrust::make_zip_iterator(thrust::make_tuple(reinterpret_cast<idx_t*>(K_ranked), SA_ranked)),
                active_set_sz_d + p_id,
                part_sz,
                stream[buf_id]
            ));
        }
    }


    // Asynchronously download the ranked-and-compacted partition in a chunked manner to host slabs.
    {
        for(auto* s : p_assembly.slab)
        {
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<idx_t*>(s->key()), reinterpret_cast<idx_t*>(K_ranked) + s->off(), s->size() * sizeof(idx_t), cudaMemcpyDeviceToHost, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(s->suf(), SA_ranked + s->off(), s->size() * sizeof(idx_t), cudaMemcpyDeviceToHost, stream[buf_id]));
            (void)beta_ranked;
        }
    }
}


template <typename T_sym_>
template <bool init_rank_>
void GPU_SACA<T_sym_>::drive_pipeline()
{
#ifdef ISA_INSTRUMENT
    uint64_t pipeline_ns = 0;
#endif
{
    INSTRUMENT_TIMER(pipeline_ns);

    idx_t total_chunk = 0;  // Total count of chunks across all partitions.
    p_id_t active_part_c = 0;   // Count of active partitions in the current iteration.

    // Post read-tasks in partition order.
    {
        for(p_id_t p_id = 0; p_id < p_; ++p_id)
        {
            if(Active_sz[p_id] == 0)    // Skip partitions with empty active-sets.
                continue;

            active_part_c++;
            const idx_t part_chunk_c = (Active_sz[p_id] + chunk_cap_ - 1) / chunk_cap_; // Number of chunks / slabs required for this partition.
            total_chunk += part_chunk_c;

            // Prepare skeleton chain of chunks of this partition.
            part_assembly[p_id].slab.assign(part_chunk_c, nullptr);
            part_assembly[p_id].chunks_to_read.store(part_chunk_c, std::memory_order_relaxed);

            for(idx_t c = 0; c < part_chunk_c; ++c) // Post a read-task for each chunk of this partition.
            {
                const auto chunk_off = c * chunk_cap_;
                const auto chunk_sz = std::min(chunk_cap_, Active_sz[p_id] - chunk_off);
                RT.enqueue(Read_Task{p_id, c, chunk_off, chunk_sz});
            }
        }
    }


    std::vector<std::thread> readers;   // Readers-pool.    // TODO: why new pool per iteration? Fix.
    std::atomic<int64_t> rt_remain(total_chunk);    // Count of read-tasks left to claim.
    // Launch readers thread-pool to read chunks from the partition-backend to staging slabs.
    {
        for(idx_t r = 0; r < reader_c_; ++r)
            readers.emplace_back(
                [&, r]
                {
                    INSTRUMENT(isa_instrument::ISA_Instrument().bind(r));
                    while(rt_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Read_Task rt;       RT.wait_dequeue(rt);    // Get a read-task.
                        Staging_Slab* s;
                        {
                            INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().wait_ns);
                            F.wait_dequeue(s);      // Get a free slab for the chunk.
                        }

                        s->assign(rt.p_id, rt.off, rt.sz);
                        if constexpr(init_rank_)    backend_->read_prefixes(*s);
                        else                        backend_->read_keys(*s);

                        auto& p_assembly = part_assembly[rt.p_id];
                        p_assembly.slab[rt.chunk_idx] = s;  // Add the slab to the partition's assembly / chain.
                        if(p_assembly.chunks_to_read.fetch_sub(1, std::memory_order_acq_rel) == 1)  // Last chunk read off for this partition; chunked-assembly is ready for the device.
                            R.enqueue(&p_assembly);
                    }
                }
            );
    }


    std::vector<std::thread> writers;   // Writers-pool.    // TODO: why new pool per iteration?
    std::atomic<int64_t> dl_remain(total_chunk);    // Count of chunks left to download from the device.
    // Launch writers thread-pool to drain the processed chunks to the partition-backend from staging slabs.
    {
        for(idx_t w = 0; w < writer_c_; ++w)
            writers.emplace_back(
                [&]
                {
                    while(dl_remain.fetch_sub(1, std::memory_order_relaxed) > 0)    // Claim a ticket.
                    {
                        Staging_Slab* s;
                        W.wait_dequeue(s);  // Get a processed slab.
                        backend_->write(*s);    // Drain the slab.
                        F.enqueue(s);   // This slab is free now.
                    }
                }
            );
    }


    // Current thread itself is the GPU-dispatcher thread. It dispatches device-tasks by repeatedly:
    // pulling a complete assembly and a free device-stream, and then dispatching these to the device
    // (with an attached callback to add processed slabs to write-queue and free the device-stream).
    {
        for(p_id_t dispatched = 0; dispatched < active_part_c; ++dispatched)
        {
            idx_t s_id;                     FS.wait_dequeue(s_id);      // Get a free device-stream.
            Partition_Assembly* p_assembly; R.wait_dequeue(p_assembly); // Get the complete chunked-assembly of a partition.
            const p_id_t p_id = p_assembly - part_assembly.get();       // ID of the partition this assembly corresponds to.

            if constexpr(init_rank_)    sort_and_init_rank_and_filter(p_id, *p_assembly, s_id);
            else                        sort_and_re_rank_and_filter(p_id, *p_assembly, s_id);

            CUDA_CHECK(cudaLaunchHostFunc(stream[s_id], on_partition_download, new Partition_Download_Callback_Args{this, p_assembly, s_id}));
        }
    }


    // Wait for the pipeline to finish.
    {
        for(idx_t s_id = 0; s_id < STREAM_C; ++s_id)
            CUDA_CHECK(cudaStreamSynchronize(stream[s_id]));

        for(auto& r : readers)  r.join();
        for(auto& w : writers)  w.join();
    }
}

    INSTRUMENT(isa_instrument::ISA_Instrument().accumulate_pipeline_ns(pipeline_ns));
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::sort_and_re_rank_and_filter(const p_id_t p_id, Partition_Assembly& p_assembly, const idx_t buf_id)
{
    PROFILE_SCOPE_FMT("sort_and_re_rank_and_filter_p%d", static_cast<int>(p_id));


    // Asynchronously upload this partition's chunks of suffixes and their ranks to the device.
    {
        for(auto* s : p_assembly.slab)
        {
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<idx_t*>(key_buf[buf_id][0]) + s->off(), s->key(), s->size() * sizeof(idx_t), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(suf_buf[buf_id][0] + s->off(), s->suf(), s->size() * sizeof(idx_t), cudaMemcpyHostToDevice, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(beta_buf[buf_id][0] + s->off(), s->beta(), s->size() * sizeof(p_id_t), cudaMemcpyHostToDevice, stream[buf_id]));
        }
    }


    const auto part_sz = Active_sz[p_id];   // Size of the active-set of this partition.
    auto workspace = workspace_buf[buf_id]; // Scratch space for `cub`.
    auto is_active = is_active_buf[buf_id]; // Activeness-marker flags.

    constexpr std::size_t BLOCK_SZ = 256;
    const auto block_count = (part_sz + BLOCK_SZ - 1) / BLOCK_SZ;

    idx_t* K_sorted = nullptr;  // Sorted keys.
    idx_t* SA_sorted = nullptr; // Sorted suffixes.
    idx_t* K_re_ranked = nullptr;   // Re-ranked keys.
    idx_t* SA_re_ranked = nullptr;  // Suffixes in re-ranked order (bar the active | retired split).
    p_id_t* beta_re_ranked = nullptr;   // Backward tags corresponding to the re-ranked suffixes.


    // Sort the partition by (primary, secondary, tertiary): two phase radix sort.
    {
        cub::DoubleBuffer<idx_t> suf(suf_buf[buf_id][0], suf_buf[buf_id][1]);
        cub::DoubleBuffer<p_sz_t> perm(perm_buf[buf_id][0], perm_buf[buf_id][1]);
        cub::DoubleBuffer<p_id_t> beta(beta_buf[buf_id][0], beta_buf[buf_id][1]);

        // Sort by the tertiary (forward tag).
        {
            // Populate identity permutation.
            iota_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            // Sort the forward tags.
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                workspace, workspace_sz,
                suf, perm,  // The tertiary (forward tag) is packed in suffix.
                part_sz,
                static_cast<int>(bit_layout::TAG_SHIFT), static_cast<int>(bit_layout::WORD_BITS),
                stream[buf_id]
            ));

            // Shuffle keys and backward tags per the forward-tag-sorted permutation.
            shuffle_kernel<idx_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (reinterpret_cast<idx_t*>(key_buf[buf_id][1]), reinterpret_cast<const idx_t*>(key_buf[buf_id][0]), perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            shuffle_kernel<p_id_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (beta_buf[buf_id][1], beta_buf[buf_id][0], perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());
        }


        // Sort by (primary, secondary).
        {
            // Populate identity permutation.
            iota_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            // Sort the (primary, secondary) ranks.
            cub::DoubleBuffer<idx_t> key(reinterpret_cast<idx_t*>(key_buf[buf_id][1]), reinterpret_cast<idx_t*>(key_buf[buf_id][0]));
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                workspace, workspace_sz,
                key, perm,
                part_sz,
                0, static_cast<int>(key_bits_),
                stream[buf_id]
            ));

            // Shuffle suffixes and backward tags per the (primary, secondary)-sorted permutation.
            shuffle_kernel<idx_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (suf.Alternate(), suf.Current(), perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            shuffle_kernel<p_id_t, BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (beta_buf[buf_id][0], beta_buf[buf_id][1], perm.Current(), part_sz);
            CUDA_CHECK(cudaGetLastError());

            K_sorted = key.Current(), K_re_ranked = key.Alternate();
            SA_sorted = suf.Alternate(), SA_re_ranked = suf.Current();
            beta_re_ranked = beta_buf[buf_id][0];
        }
    }


    // Re-rank and filter the sort-output.
    {
        idx_t* const key_group_rank = K_re_ranked;    // Distinct key groups' ranks.
        idx_t* const primary_key_group_rank = key_group_rank + part_sz; // Distinct primary-key groups' ranks.

        // Track groups with distinct keys and distinct primary-keys, and mark non-unique keys.
        {
            rank_key_groups_and_primary_key_groups_and_mark_active_flags_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (K_sorted, SA_sorted, part_sz, secondary_bits_, key_group_rank, primary_key_group_rank, is_active);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cub::DeviceScan::InclusiveScan(
                workspace, workspace_sz,
                key_group_rank, key_group_rank,
                cub::Max(),
                part_sz,
                stream[buf_id]
            ));

            CUDA_CHECK(cub::DeviceScan::InclusiveScan(
                workspace, workspace_sz,
                primary_key_group_rank, primary_key_group_rank,
                cub::Max(),
                part_sz,
                stream[buf_id]
            ));
        }


        // Refine rankings to finer resolution based on the in-primary-key-group offsets of different key groups.
        {
            refine_primary_keys_kernel<BLOCK_SZ> <<< block_count, BLOCK_SZ, 0, stream[buf_id] >>>
                (K_sorted, part_sz, secondary_bits_, key_group_rank, primary_key_group_rank);
            CUDA_CHECK(cudaGetLastError());
        }


        // TODO: refactor all the repeated code between sort-and-initial-rank and sort-and-rank, with a `bool` template parameter.
        // Group non-unique rank keys and suffixes to the head and the rest to the tail of the partition.
        {
            CUDA_CHECK(cub::DevicePartition::Flagged(
                workspace, workspace_sz,
                thrust::make_zip_iterator(thrust::make_tuple(static_cast<const idx_t*>(K_sorted), static_cast<idx_t*>(SA_sorted))),
                is_active,
                thrust::make_zip_iterator(thrust::make_tuple(K_re_ranked, SA_re_ranked)),
                active_set_sz_d + p_id,
                part_sz,
                stream[buf_id]
            ));
        }
    }


    // Asynchronously download the ranked-and-compacted partition in a chunked manner to host slabs.
    {
        for(auto* s : p_assembly.slab)
        {
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<idx_t*>(s->key()), K_re_ranked + s->off(), s->size() * sizeof(idx_t), cudaMemcpyDeviceToHost, stream[buf_id]));
            CUDA_CHECK(cudaMemcpyAsync(s->suf(), SA_re_ranked + s->off(), s->size() * sizeof(idx_t), cudaMemcpyDeviceToHost, stream[buf_id]));
            (void)beta_re_ranked;
        }
    }
}


template <typename T_sym_>
void CUDART_CB GPU_SACA<T_sym_>::on_partition_download(void* const arg)
{
    auto* const cb_arg = static_cast<Partition_Download_Callback_Args*>(arg);
    auto* const self = cb_arg->self;
    auto* const p_assembly = cb_arg->p_assembly;
    const auto s_id = cb_arg->s_id;

    for(auto* s : p_assembly->slab) // The processed slabs move to the write queue.
        self->W.enqueue(s);

    self->FS.enqueue(s_id);   // This stream is now free to process a new partition.
    delete cb_arg;
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::update_global_ranks()
{
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        if(Active_sz[p_id] != 0)    // Skip partitions with empty active-sets.
            update_global_rank(p_id);
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::update_global_rank(const p_id_t p_id)
{
    const auto sz = Active_sz[p_id];
    const auto rank_off = P_off[p_id];

#ifdef ISA_INSTRUMENT
    uint64_t read_ns = 0, scatter_ns = 0;
#endif

    if(backend_->resident_suf(p_id) != nullptr) // Resident fast path.
    {
        const auto* const suf = backend_->resident_suf(p_id);
        const auto* const key = backend_->resident_key(p_id);

        {
            INSTRUMENT_TIMER(scatter_ns);

            #pragma omp parallel for schedule(static)
            for(idx_t i = 0; i < sz; ++i)
            {
                const auto j = bit_layout::suffix_of(suf[i]);   // Suffix.
                const auto phi = bit_layout::tag_of(suf[i]);    // Forward tag used at current generation.
                const auto r_j = key[i];    // Local rank.
                rho[j] = bit_layout::pack(rank_off + r_j, phi); // Global rank, packed with the tag (being made available for the next generation's consumer, if any).
            }
        }

        INSTRUMENT(isa_instrument::ISA_Instrument().accumulate_scatter_ns(scatter_ns));
        return;
    }

    auto& s = S[0];
    idx_t* const suf = s.suf();
    idx_t* const key = reinterpret_cast<idx_t*>(s.key());
    for(idx_t off = 0; off < sz;)
    {
        const idx_t to_read = std::min(chunk_cap_, sz - off);

        // Read the chunk of suffixes and their keys from offset `off`.
        {
            INSTRUMENT_TIMER(read_ns);
            backend_->read_chunk(p_id, off, to_read, suf, key);
        }

        // Update `rho`-entries corresponding to the chunk.
        {
            INSTRUMENT_TIMER(scatter_ns);

            #pragma omp parallel for schedule(static)
            for(idx_t i = 0; i < to_read; ++i)
            {
                const auto j = bit_layout::suffix_of(suf[i]);   // Suffix.
                const auto phi = bit_layout::tag_of(suf[i]);    // Forward tag used at current generation.
                const auto r_j = key[i];    // Local rank.
                rho[j] = bit_layout::pack(rank_off + r_j, phi); // Global rank, packed with the tag (being made available for the next generation's consumer, if any).
            }
        }

        off += to_read;
    }

    INSTRUMENT(isa_instrument::ISA_Instrument().accumulate_read_chunk_ns(read_ns));
    INSTRUMENT(isa_instrument::ISA_Instrument().accumulate_scatter_ns(scatter_ns));
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::exponentiate_prefix()
{
    idx_t iter = 1;
    for(idx_t k = ell; k < n_; k *= 2, ++iter)
    {
        PROFILE_SCOPE_FMT("doubling_iter_k_%lu", static_cast<unsigned long>(k));

        INSTRUMENT(isa_instrument::ISA_Instrument().begin_iteration(iter, k, std::accumulate(Active_sz.cbegin(), Active_sz.cend(), idx_t(0))));
        const auto t_0 = gpu_saca::timer::now();

        // Pass current prefix-context size to the backend so that readers can exponentiate the context-sizes.
        {
            // PROFILE_SCOPE("cpu_gather"); // TODO: relocate properly.
            backend_->set_stride(k);
        }

        // Sort-and-rank each partition.
        {
            PROFILE_SCOPE("sort_and_rank_dispatch");
            drive_pipeline<false>();
        }

        // Update global-ranks based on the updated local-ranks.
        {
            PROFILE_SCOPE("update_global_ranks");
            update_global_ranks();
        }

        // Resize the partitions to their active-set size.
        // NB: this compaction must land *after* the preceding global rankings have been updated.
        {
            PROFILE_SCOPE("collect_active_set_sizes");
            CUDA_CHECK(cudaMemcpy(Active_sz.data(), active_set_sz_d, p_ * sizeof(idx_t), cudaMemcpyDeviceToHost));
        }

        const auto t_1 = gpu_saca::timer::now();
        (void)t_0, (void)t_1;


        // Check for early-exit.
        {
            p_id_t sorted_part = 0; // Count of partitions that are already fully-sorted (by prefix-context size `2k`).
            idx_t unsorted = 0; // Count of suffixes with non-unique ranks.
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                sorted_part += (Active_sz[p_id] == 0),
                unsorted += Active_sz[p_id];
#ifdef STATS
            std::cerr   << "Number of partitions fully-sorted by context-size " << (k * 2) << ": " << sorted_part << ". Time taken in iteration: " << gpu_saca::timer::duration(t_1 - t_0) << "s."
                        << " Out of " << n_ << " suffixes, " << unsorted << " (" << (unsorted * 100.0 / n_) << "%) still need more resolution.\n";
#endif

            INSTRUMENT(isa_instrument::ISA_Instrument().report_log());

            if(sorted_part == p_)   // All partitions have already been sorted.
                break;
        }
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::concatenate_partitions(idx_t*& SA)
{
    SA = rho;
    rho = nullptr;

    auto& s = S[0];
    idx_t* const suf = s.suf();
    idx_t* const key = reinterpret_cast<idx_t*>(s.key());

    // Place the partitions at their appropriate offsets and shuffle the in-
    // partition suffixes per their local-rank.
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
    {
        const auto sz = P_cap[p_id];
        const auto off = P_off[p_id];

        if(backend_->resident_suf(p_id) != nullptr) // Resident fast path.
        {
            const auto* const suf = backend_->resident_suf(p_id);
            const auto* const key = backend_->resident_key(p_id);

            #pragma omp parallel for schedule(static)
            for(idx_t i = 0; i < sz; ++i)
            {
                const auto j = bit_layout::suffix_of(suf[i]);   // Suffix.
                const auto r_j = key[i];    // Local rank; so global rank is `off + r_j`.
                SA[off + r_j] = j;
            }

            continue;
        }

        for(idx_t chunk_off = 0; chunk_off < sz;)
        {
            const auto chunk_sz = std::min(chunk_cap_, sz - chunk_off);
            backend_->read_chunk(p_id, chunk_off, chunk_sz, suf, key);
            chunk_off += chunk_sz;

            #pragma omp parallel for schedule(static)
            for(idx_t i = 0; i < chunk_sz; ++i)
            {
                const auto j = bit_layout::suffix_of(suf[i]);   // Suffix.
                const auto r_j = key[i];    // Local rank; so global rank is `off + r_j`.
                SA[off + r_j] = j;
            }
        }
    }
}



// Template instantiations for the required instances.
template class GPU_SACA<char>;
template class GPU_SACA<uint8_t>;
template class GPU_SACA<uint16_t>;
template class GPU_SACA<uint32_t>;

}
