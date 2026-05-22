
#include "tests.hpp"
#include "utility.cuh"

#include <cstdint>
#include <cstddef>
#include <vector>


namespace gpu_saca::test
{

template <typename T_sym_>
bool validate_kgram_histogram(const T_sym_* const T_h, const std::size_t n, const std::size_t k, const uint64_t* const H_d)
{
    const std::size_t hist_sz = 1llu << k;
    std::vector<uint64_t> gpu_hist(hist_sz);
    CUDA_CHECK(cudaMemcpy(gpu_hist.data(), H_d, hist_sz * sizeof(uint64_t), cudaMemcpyDeviceToHost));   // Copy GPU-histogram to host.

    std::vector<uint64_t> cpu_hist(hist_sz);
    kgram_histogram(T_h, n, cpu_hist.data(), k);
    return gpu_hist == cpu_hist;
}



// Template instantiations for the required instances.
template bool validate_kgram_histogram(const char*, std::size_t, std::size_t, const uint64_t*);
template bool validate_kgram_histogram(const uint8_t*, std::size_t, std::size_t, const uint64_t*);
template bool validate_kgram_histogram(const uint16_t*, std::size_t, std::size_t, const uint64_t*);
template bool validate_kgram_histogram(const uint32_t*, std::size_t, std::size_t, const uint64_t*);

}
