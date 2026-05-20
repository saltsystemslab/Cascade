
#include "GPU_SACA.cuh"
#include "utility.cuh"

#include <cstdint>
#include <cstddef>


namespace gpu_saca
{


template <typename T_sym_>
GPU_SACA<T_sym_>::GPU_SACA(const T_sym_* const T_h, const std::size_t n):
      T_d_(nullptr)
    , n_(n + 1) // Account for the terminator symbol with `+1`
    , p_(128)   // TODO: tune this.
{
    // Copy over text from host to device

    const auto bytes = n * sizeof(T_sym_);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&T_d_), bytes));
    CUDA_CHECK(cudaMemcpy(T_d_, T_h, bytes, cudaMemcpyHostToDevice));
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

    (void)SA;
}


template <typename T_sym_>
void GPU_SACA<T_sym_>::select_pivots()
{
    const auto hist_sz = 1llu << K;
    uint64_t* H_d;  // The global histogram.

    CUDA_CHECK(cudaMalloc(&H_d, hist_sz * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(H_d, 0, hist_sz * sizeof(uint64_t)));

    // Count the k-gram frequencies into the histogram.
    {
        constexpr std::size_t BLOCK_SZ = 128;
        const auto kgram_count = n_ - K + 1;
        const auto block_count = (kgram_count + BLOCK_SZ - 1) / BLOCK_SZ;

        const auto s_mem_sz = (BLOCK_SZ + K - 1) + (1llu << K) + 4; // Shared memory size:  binarized text-tile, followed by block-local histogram with `u8` counters,
                                                                    //                      followed by some padding for atomic-add workaround of `u8`s.

        kgram_histogram_kernel<T_sym_, K, BLOCK_SZ> <<< block_count, BLOCK_SZ, s_mem_sz >>> (T_d_, n_, H_d);
        CUDA_CHECK(cudaGetLastError());
    }
}


// Template instantiations for the required instances.
template class GPU_SACA<char>;
template class GPU_SACA<uint8_t>;
template class GPU_SACA<uint16_t>;
template class GPU_SACA<uint32_t>;

}
