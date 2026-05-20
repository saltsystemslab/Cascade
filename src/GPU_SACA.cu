
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
    (void)SA;
}


// Template instantiations for the required instances.
template class GPU_SACA<char>;
template class GPU_SACA<uint8_t>;
template class GPU_SACA<uint16_t>;
template class GPU_SACA<uint32_t>;

}
