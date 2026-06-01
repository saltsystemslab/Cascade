
#include "GPU_SACA.cuh"
#include "utility.hpp"
#include "utility.cuh"
#include "tests.hpp"

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <iostream>


namespace gpu_saca
{

__constant__ pivot_t pivots_d[MAX_PART_COUNT];


template <typename T_sym_>
GPU_SACA<T_sym_>::GPU_SACA(const T_sym_* const T, const idx_t n):
      T_d_(nullptr)
    , n_(n)
    , T_h_(T)
    , p_(128)   // TODO: tune this.
    , null_d(nullptr)
{
    CUDA_CHECK(cudaStreamCreate(stream + 0));
    CUDA_CHECK(cudaStreamCreate(stream + 1));

    CUDA_CHECK(cudaMalloc(&null_d, (K - 1) * sizeof(T_sym_)));
    std::vector<T_sym_> null_h(K - 1, std::numeric_limits<T_sym_>::min());
    CUDA_CHECK(cudaMemcpy(null_d, null_h.data(), (K - 1) * sizeof(T_sym_), cudaMemcpyHostToDevice));
}


template <typename T_sym_>
GPU_SACA<T_sym_>::~GPU_SACA()
{
    CUDA_CHECK(cudaStreamDestroy(stream[0]));
    CUDA_CHECK(cudaStreamDestroy(stream[1]));

    CUDA_CHECK(cudaFree(null_d));
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::build_SA(idx_t* const SA)
{
    // Pin the host-side text so that PCIe transfers of text can be overlapped kernel execution.
    CUDA_CHECK(cudaHostRegister(const_cast<T_sym_*>(T_h_), n_ * sizeof(T_sym_), cudaHostRegisterDefault));

    select_pivots();

    // compute_partition_IDs();

    // Unpin the host-side text as we do not need the text anymore.
    CUDA_CHECK(cudaHostUnregister(const_cast<T_sym_*>(T_h_)));

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
        CUDA_CHECK(cudaMalloc(T_buf + 0, buf_bytes));
        CUDA_CHECK(cudaMalloc(T_buf + 1, buf_bytes));
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
            compute_kgram_histogram_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, s_mem_sz, stream[buf_id] >>> (T_buf[buf_id], chunk_kgram_count, H_d);
            CUDA_CHECK(cudaGetLastError());
        }

        // Wait for the streams to finish.
        CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        CUDA_CHECK(cudaStreamSynchronize(stream[1]));

        // Free the buffers.
        CUDA_CHECK(cudaFree(T_buf[0]));
        CUDA_CHECK(cudaFree(T_buf[1]));


#ifdef TESTS
            std::cerr << "Histogram verification: " << (test::validate_kgram_histogram(T_h_, n_, K, H_d) ? "Passes" : "Fails") << "\n";
#endif
    }

    // Select `p - 1` uniformly spread out (in terms of histogram mass) pivots.
    {
        std::vector<freq_cnt_t> H_h(hist_sz);   // Histogram at the device.
        CUDA_CHECK(cudaMemcpy(H_h.data(), H_d, hist_sz * sizeof(freq_cnt_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(H_d));

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


#ifdef TESTS
        const auto part_sz = test::get_partition_sizes(H_h, pivots);
        const auto stats = util::Collection_Stats::get_collection_stat(part_sz);
        stats.print();
#endif
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::compute_partition_IDs()
{
    CUDA_CHECK(cudaMemcpyToSymbol(pivots_d, pivots.data(), pivots.size() * sizeof(pivot_t)));

    constexpr std::size_t BLOCK_SZ = 256;
    const auto suf_count = n_;
    const auto block_count = (suf_count + BLOCK_SZ - 1) / BLOCK_SZ;
    const auto s_mem_sz = BLOCK_SZ + K - 1; // Shared memory size: quantized text-tile.

    p_id_t* P_id_d;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&P_id_d), n_ * sizeof(p_id_t)));

    compute_partition_IDs_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, s_mem_sz >>> (T_d_, n_, P_id_d, p_);
    CUDA_CHECK(cudaGetLastError());

    P_id.resize(n_);
    CUDA_CHECK(cudaMemcpy(P_id.data(), P_id_d, n_ * sizeof(p_id_t), cudaMemcpyDeviceToHost));

#ifdef TESTS
    std::cerr << "Partition-IDs verification: " << (test::validate_partition_IDs(T_h_, n_, K, pivots, P_id_d) ? "Passes" : "Fails") << "\n";
#endif

    CUDA_CHECK(cudaFree(P_id_d));
}



// Template instantiations for the required instances.
template class GPU_SACA<char>;
template class GPU_SACA<uint8_t>;
template class GPU_SACA<uint16_t>;
template class GPU_SACA<uint32_t>;

}
