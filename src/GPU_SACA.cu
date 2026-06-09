
#include "GPU_SACA.cuh"
#include "Suffix_Buffer.hpp"
#include "Sort_Key.cuh"
#include "utility.hpp"
#include "utility.cuh"
#include "tests.hpp"

#include "omp.h"
#include "cub/cub.cuh"

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <iostream>


namespace gpu_saca
{

__constant__ pivot_t pivots_d[MAX_PART_COUNT];


template <typename T_sym_>
GPU_SACA<T_sym_>::GPU_SACA(const T_sym_* const T, const idx_t n):
      n_(n)
    , T_h_(T)
    , p_(128)   // TODO: tune this.
    , null_d(nullptr)
{
    for(std::size_t s = 0; s < STREAM_C; ++s)
        CUDA_CHECK(cudaStreamCreate(stream + s));

    CUDA_CHECK(cudaMalloc(&null_d, (K - 1) * sizeof(T_sym_)));
    std::vector<T_sym_> null_h(K - 1, std::numeric_limits<T_sym_>::min());
    CUDA_CHECK(cudaMemcpy(null_d, null_h.data(), (K - 1) * sizeof(T_sym_), cudaMemcpyHostToDevice));
}


template <typename T_sym_>
GPU_SACA<T_sym_>::~GPU_SACA()
{
    for(std::size_t s = 0; s < STREAM_C; ++s)
        CUDA_CHECK(cudaStreamDestroy(stream[s]));

    CUDA_CHECK(cudaFree(null_d));
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::build_SA(idx_t* const SA)
{
    // Pin the host-side text so that PCIe transfers of text can be overlapped kernel execution.
    CUDA_CHECK(cudaHostRegister(const_cast<T_sym_*>(T_h_), n_ * sizeof(T_sym_), cudaHostRegisterDefault));

    select_pivots();

    partition();

    // Unpin the host-side text as we do not need the text anymore.
    CUDA_CHECK(cudaHostUnregister(const_cast<T_sym_*>(T_h_)));

    initial_rank();

    (void)SA;
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::select_pivots()
{
    const auto hist_sz = 1llu << K;
    freq_cnt_t* H_d;    // The global histogram.

    // Allocate the global histogram.
    CUDA_CHECK(cudaMalloc(&H_d, hist_sz * sizeof(freq_cnt_t)));
    CUDA_CHECK(cudaMemset(H_d, 0, hist_sz * sizeof(freq_cnt_t)));

    constexpr std::size_t BLOCK_SZ = 128;
    idx_t CHUNK_SZ; // Number of `K`-grams at each text-chunk.
    T_sym_* T_buf[2];   // The text-chunk buffers.

    // Determine the text-chunks' sizes for processing them in a streaming manner, and allocate space for the corresponding buffers.
    {
        std::size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        const std::size_t budget = free_mem * (1.0 - MEM_SAFETY_MARGIN);    // Total memory for the chunk buffers.
        const std::size_t per_sym_mem = 2 * sizeof(T_sym_); // Space required per-symbol across the two buffers.
        CHUNK_SZ = budget / per_sym_mem;
        CHUNK_SZ = (CHUNK_SZ / BLOCK_SZ) * BLOCK_SZ;    // Round-down to the nearest multiple of `BLOCK_SZ` for clean grid-partitioning.
        if(CHUNK_SZ < BLOCK_SZ) // Too low memory in the GPU to support even one block.
        {
            std::cerr << "Too low device memory available for histogram streaming. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        const std::size_t buf_bytes = (CHUNK_SZ + K - 1) * sizeof(T_sym_);
        for(std::size_t b = 0; b < 2; ++b)
            CUDA_CHECK(cudaMalloc(T_buf + b, buf_bytes));
    }

    // Count the quantized k-gram frequencies into the histogram.
    {
        const auto s_mem_sz = (BLOCK_SZ + K - 1) + (1llu << K) + 4; // Shared memory size:  quantized text-tile, followed by block-local histogram with `u8` counters,
                                                                    //                      followed by some padding for atomic-add workaround of `u8`s.

        const idx_t chunk_count = (n_ + CHUNK_SZ - 1) / CHUNK_SZ;   // Number of chunks the text is broken into for processing in a streaming manner.
        for(idx_t c = 0; c < chunk_count; ++c)  // Double-buffered chunk loop.
        {
            const auto buf_id = c % 2;  // Which buffer (and stream) to use for this chunk.
            const auto chunk_off = c * CHUNK_SZ;    // Offset of this chunk into the text.
            const auto chunk_kgram_count = std::min(CHUNK_SZ, n_ - chunk_off);  // Count of `K`-grams in this chunk.
            const auto chunk_symbol_req = chunk_kgram_count + K - 1;    // Number of symbols this chunk is required to have.
            const auto chunk_symbol_count = std::min(chunk_symbol_req, n_ - chunk_off);    // Number of symbols to transfer from host to device for this chunk.

            //  Asynchronously transfer this chunk to the device while the other stream's kernel is being executed (from the last iteration).
            CUDA_CHECK(cudaMemcpyAsync(T_buf[buf_id], T_h_ + chunk_off, chunk_symbol_count * sizeof(T_sym_), cudaMemcpyHostToDevice, stream[buf_id]));

            // Pad the tail of the chunk with null / terminator symbols if required.
            if(chunk_symbol_count < chunk_symbol_req)   // Can happen at the last and at the previous-to-last chunks.
                CUDA_CHECK(cudaMemcpyAsync(T_buf[buf_id] + chunk_symbol_count, null_d, (chunk_symbol_req - chunk_symbol_count) * sizeof(T_sym_), cudaMemcpyDeviceToDevice, stream[buf_id]));

            const auto block_count = (chunk_kgram_count + BLOCK_SZ - 1) / BLOCK_SZ;
            compute_kgram_histogram_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, s_mem_sz, stream[buf_id] >>>
                (T_buf[buf_id], chunk_kgram_count, H_d);
            CUDA_CHECK(cudaGetLastError());
        }

        // Wait for the streams to finish.
        for(std::size_t s = 0; s < 2; ++s)
            CUDA_CHECK(cudaStreamSynchronize(stream[s]));

        // Free the buffers.
        for(std::size_t b = 0; b < 2; ++b)
            CUDA_CHECK(cudaFree(T_buf[b]));
    }

    // Select `p - 1` uniformly spread out (in terms of histogram mass) pivots.
    {
        std::vector<freq_cnt_t> H_h(hist_sz);   // Histogram at the device.
        CUDA_CHECK(cudaMemcpy(H_h.data(), H_d, hist_sz * sizeof(freq_cnt_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(H_d));

#ifdef TESTS
        std::cerr << "Histogram verification: " << (test::validate_kgram_histogram(T_h_, n_, K, H_h.data()) ? "Passes" : "Fails") << "\n";
#endif

        const p_id_t pivot_count = p_ - 1;
        pivots.resize(pivot_count);
        freq_cnt_t cum_sum = 0;
        std::size_t j = 0;  // Number of collected pivots.
        for(std::size_t i = 0; i < hist_sz && j < pivot_count; ++i)
        {
            cum_sum += H_h[i];
            if(j < pivot_count && cum_sum >= (j + 1) * n_ / p_)
            {
                const auto target = (j + 1) * n_ / p_;
                const auto before = cum_sum - H_h[i];
                const auto after  = cum_sum;

                // Place the split point before or after bin `i`, where ever the mass sum is closer to the target sum.
                pivot_t split = (target - before <= after - target ?
                                    (i > 0 ? i - 1 : 0) :
                                    i);
                if(j > 0 && split <= pivots[j - 1]) // Nudge the pivot forward if the bin is really heavy.
                    split = pivots[j - 1] + 1;

                pivots[j++] = split;
            }
        }

        // Fill any remaining pivots with consecutive bin indices following the last pivot.
        // These will correspond to empty partition but keeps the pivot-set fully populated and non-decreasing.
        // Special case when partition-count is larger than count of unique quantized `k`-grams.
        const pivot_t last_pivot = (j > 0 ? pivots[j - 1] : 0);
        const pivot_t max_bin = (1llu << K) - 1;
        for(pivot_t rem_slot = 0; j + rem_slot < pivot_count; ++rem_slot)
            pivots[j + rem_slot] = std::min(static_cast<pivot_t>(last_pivot + rem_slot + 1), max_bin);


        compute_partition_sizes(H_h);

#ifdef TESTS
        const auto stats = util::Collection_Stats::get_collection_stat(H_h);
        stats.print();
#endif
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::compute_partition_sizes(const std::vector<freq_cnt_t>& H)
{
    P_sz.resize(p_, 0);
    p_id_t p_id = 0;    // Which partition we're at right now.
    for(std::size_t bin = 0; bin < H.size(); ++bin)
    {
        if(p_id < pivots.size() && bin > pivots[p_id])  // Moving onto the next partition.
            p_id++;

        P_sz[p_id] += H[bin];   // The pivots are inclusive upper bounds of their partition.
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::partition()
{
    compute_partition_IDs();

    // Move the suffixes (with some associated prefixes) to their corresponding partitions.
    {
        std::vector<util::Padded<Suffix_Buffer>> P_buf;

        // Allocate the partitions with required capacities, and create concurrent buffer for each.
        {
            P_.reserve(p_);
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                P_.emplace_back(P_sz[p_id]);

            P_buf.reserve(p_);
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                P_buf.emplace_back(Suffix_Buffer{P_[p_id].suf(), P_[p_id].key()});
        }


        // Add suffix indices along with their 16-byte prefixes to their corresponding partitions (through the buffers).
        {
            #pragma omp parallel for schedule(static)
            for(idx_t i = 0; i < n_; ++i)
                P_buf[P_id[i]].unwrap().add(i, pack_word(i), pack_word(i + 8 / sizeof(T_sym_)));

            #pragma omp parallel for schedule(static)
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                P_buf[p_id].unwrap().flush();
        }

#ifdef TESTS
        {
            bool part_sz_match = true;
            for(p_id_t p_id = 0; p_id < p_; ++p_id)
                part_sz_match = part_sz_match && (P_buf[p_id].unwrap().flushed() == P_sz[p_id]);

            std::cerr << "Partition-sizes match verification: " << (part_sz_match ? "Passes" : "Fails") << "\n";
        }
#endif
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::compute_partition_IDs()
{
    // Copy the pivots to the device.
    CUDA_CHECK(cudaMemcpyToSymbol(pivots_d, pivots.data(), pivots.size() * sizeof(pivot_t)));

    constexpr std::size_t BLOCK_SZ = 256;
    idx_t CHUNK_SZ; // Number of suffixes at each text-chunk.
    T_sym_* T_buf[STREAM_C];    // The text-chunk buffers.
    p_id_t* P_id_buf[STREAM_C]; // The partition-ID buffers.

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

        P_id.resize(n_);
    }


    // Move text in chunks to device, compute partition-IDs of the suffixes, and move the IDs to host.
    {
        const auto s_mem_sz = BLOCK_SZ + K - 1; // Shared memory size: quantized text-tile.
        const idx_t chunk_count = (n_ + CHUNK_SZ - 1) / CHUNK_SZ;   // Number of chunks the text is broken into for processing in a streaming manner.
        for(idx_t c = 0; c < chunk_count; ++c)  // Triple-buffered chunk loop.
        {
            const auto buf_id = c % 3;  // Which set of buffers (and stream) to use for this chunk.
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
            compute_partition_IDs_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, s_mem_sz, stream[buf_id] >>>
                (T_buf[buf_id], chunk_suf_count, P_id_buf[buf_id], p_);
            CUDA_CHECK(cudaGetLastError());

            // Asynchronously copy this chunk's partition-IDs to the host.
            CUDA_CHECK(cudaMemcpyAsync(P_id.data() + chunk_off, P_id_buf[buf_id], chunk_suf_count * sizeof(p_id_t), cudaMemcpyDeviceToHost, stream[buf_id]));
        }
    }


    // Wait for the streams to finish.
    for(std::size_t s = 0; s < STREAM_C; ++s)
        CUDA_CHECK(cudaStreamSynchronize(stream[s]));

    // Free the buffers.
    for(std::size_t b = 0; b < 3; ++b)
    {
        CUDA_CHECK(cudaFree(T_buf[b]));
        CUDA_CHECK(cudaFree(P_id_buf[b]));
    }


#ifdef TESTS
    std::cerr << "Partition-IDs verification: " << (test::validate_partition_IDs(T_h_, n_, K, pivots, P_id.data()) ? "Passes" : "Fails") << "\n";
#endif
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::initial_rank()
{
    Sort_Key* key_buf[STREAM_C][2]; // Sort-keys of suffixes.
    idx_t* suf_buf[STREAM_C][2];    // Suffixes.
    void* workspace_buf[STREAM_C];  // Scratch space for `cub` radix-sort.

    // Verify device memory budget and allocate device buffers for the streams.
    {
        // Each suffix in each stream needs: 1. `DoubleBuffer` keys, input and output (2 x `Sort_key`);
        //      2. `DoubleBuffer` values, input and output (2 x `idx_t`).
        // `cub` itself requires a sort-workspace.
        // Memory constraint: STREAM_C x (2 x P_sz_max * (sizeof(Sort_Key) + sizeof(idx_t)) + workspace_sz) <= free_mem x (1 - MEM_SAFETY_MARGIN)

        const idx_t P_sz_max = *std::max_element(P_sz.cbegin(), P_sz.cend());
        std::size_t workspace_sz = 0;
        {
            cub::DoubleBuffer<Sort_Key> keys;
            cub::DoubleBuffer<idx_t> vals;
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                nullptr, workspace_sz,
                keys, vals,
                P_sz_max,
                Sort_Key_Decomposer{}
            ));
        }

        std::size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        const std::size_t budget = free_mem * (1.0 - MEM_SAFETY_MARGIN);    // Total memory across all the streams.
        const std::size_t per_stream_budget = 2 * P_sz_max * (sizeof(Sort_Key) + sizeof(idx_t)) + workspace_sz;
        if(STREAM_C * per_stream_budget > budget)
        {
            std::cerr << "Too low device memory available for initial-ranking process streaming. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }


        for(std::size_t b = 0; b < STREAM_C; ++b)   // For each stream
        {
            for(std::size_t r = 0; r < 1; ++r)  // For each buffer in the radix-sort ping-pong
            {
                CUDA_CHECK(cudaMalloc(&key_buf[b][r], P_sz_max * sizeof(Sort_Key)));
                CUDA_CHECK(cudaMalloc(&suf_buf[b][r], P_sz_max * sizeof(idx_t)));
            }

            CUDA_CHECK(cudaMalloc(&workspace_buf[b], workspace_sz));
        }
    }
}



// Template instantiations for the required instances.
template class GPU_SACA<char>;
template class GPU_SACA<uint8_t>;
template class GPU_SACA<uint16_t>;
template class GPU_SACA<uint32_t>;

}
