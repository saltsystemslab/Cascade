
#include "GPU_SACA.cuh"
#include "utility.hpp"
#include "utility.cuh"
#include "tests.hpp"

#include <cstdint>
#include <cstddef>
#include <iostream>


namespace gpu_saca
{

__constant__ uint32_t pivots_d[MAX_PART_COUNT];


template <typename T_sym_>
GPU_SACA<T_sym_>::GPU_SACA(const T_sym_* const T, const std::size_t n):
      T_d_(nullptr)
    , n_(n + 1) // Account for the terminator symbol with `+1`
    , T_h_(T)
    , p_(128)   // TODO: tune this.
{
    // Copy over text from host to device

    const auto bytes = n_ * sizeof(T_sym_);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&T_d_), bytes));
    CUDA_CHECK(cudaMemcpy(T_d_, T, bytes, cudaMemcpyHostToDevice));
}


template <typename T_sym_>
GPU_SACA<T_sym_>::~GPU_SACA()
{
    CUDA_CHECK(cudaFree(T_d_));
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::build_SA(std::size_t* const SA)
{
    select_pivots();

    compute_partition_IDs();

    (void)SA;
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::select_pivots()
{
    const auto hist_sz = 1llu << K;
    uint64_t* H_d;  // The global histogram.

    CUDA_CHECK(cudaMalloc(&H_d, hist_sz * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(H_d, 0, hist_sz * sizeof(uint64_t)));

    // Count the quantized k-gram frequencies into the histogram.
    {
        constexpr std::size_t BLOCK_SZ = 128;
        const auto kgram_count = n_;
        const auto block_count = (kgram_count + BLOCK_SZ - 1) / BLOCK_SZ;
        const auto s_mem_sz = (BLOCK_SZ + K - 1) + (1llu << K) + 4; // Shared memory size:  quantized text-tile, followed by block-local histogram with `u8` counters,
                                                                    //                      followed by some padding for atomic-add workaround of `u8`s.

        kgram_histogram_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, s_mem_sz >>> (T_d_, n_, H_d);
        CUDA_CHECK(cudaGetLastError());

#ifdef TESTS
            std::cerr << "Histogram verification: " << (test::validate_kgram_histogram(T_h_, n_, K, H_d) ? "Passes" : "Fails") << "\n";
#endif
    }

    // Select `p - 1` uniformly spread out (in terms of histogram mass) pivots.
    {
        std::vector<uint64_t> H_h(hist_sz); // Histogram at the device.
        CUDA_CHECK(cudaMemcpy(H_h.data(), H_d, hist_sz * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        pivots.resize(p_ - 1);
        uint64_t cum_sum = 0;
        std::size_t j = 0;  // Number of collected pivots.
        for(std::size_t i = 0; i < hist_sz && j < p_ - 1; ++i)
        {
            cum_sum += H_h[i];
            if(j < p_ - 1 && cum_sum >= (j + 1) * n_ / p_)
            {
                const auto target = (j + 1) * n_ / p_;
                const auto before = cum_sum - H_h[i];
                const auto after  = cum_sum;

                // Place the split point before or after bin `i`, where ever the mass sum is closer to the target sum.
                auto split = (target - before <= after - target ? i - 1 : i);
                if(j > 0 && split <= pivots[j - 1]) // Nudge the pivot forward if the bin is really heavy.
                    split = pivots[j - 1] + 1;

                pivots[j++] = split;
            }
        }

#ifdef TESTS
        const auto part_sz = test::get_partition_sizes(H_h, pivots);
        const auto stats = Collection_Stats::get_collection_stat(part_sz);
        stats.print();
#endif
    }
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::compute_partition_IDs()
{
    CUDA_CHECK(cudaMemcpyToSymbol(pivots_d, pivots.data(), pivots.size() * sizeof(uint32_t)));

    constexpr std::size_t BLOCK_SZ = 256;
    const auto suf_count = n_;
    const auto block_count = (suf_count + BLOCK_SZ - 1) / BLOCK_SZ;
    const auto s_mem_sz = BLOCK_SZ + K - 1; // Shared memory size: quantized text-tile.

    uint32_t* P_id_d;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&P_id_d), n_ * sizeof(uint32_t)));

    compute_partition_IDs_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, s_mem_sz >>> (T_d_, n_, P_id_d, p_);
    CUDA_CHECK(cudaGetLastError());
}



// Template instantiations for the required instances.
template class GPU_SACA<char>;
template class GPU_SACA<uint8_t>;
template class GPU_SACA<uint16_t>;
template class GPU_SACA<uint32_t>;

}
