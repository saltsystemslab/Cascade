
#include "GPU_SACA.cuh"
#include "utility.hpp"


namespace gpu_saca
{


GPU_SACA::GPU_SACA(const T_alpha_* const T_h, const std::size_t n):
      T_d_(nullptr)
    , n_(n + 1) // Account for the terminator symbol with `+1`
{
    // Copy over text from host to device

    const auto bytes = n * sizeof(T_alpha_);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&T_d_), bytes));
    CUDA_CHECK(cudaMemcpy(T_d_, T_h, bytes, cudaMemcpyHostToDevice));
}


GPU_SACA::~GPU_SACA()
{
    CUDA_CHECK(cudaFree(T_d_));
}


}
