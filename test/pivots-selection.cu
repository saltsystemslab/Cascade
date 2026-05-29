
#include "tests.hpp"
#include "utility.cuh"

#include <cstdint>
#include <cstddef>
#include <vector>


namespace gpu_saca::test
{

template <typename T_sym_>
bool validate_kgram_histogram(const T_sym_* const T_h, const idx_t n, const std::size_t k, const freq_cnt_t* const H_d)
{
    const std::size_t hist_sz = 1llu << k;
    std::vector<freq_cnt_t> gpu_hist(hist_sz);
    CUDA_CHECK(cudaMemcpy(gpu_hist.data(), H_d, hist_sz * sizeof(freq_cnt_t), cudaMemcpyDeviceToHost));

    std::vector<freq_cnt_t> cpu_hist(hist_sz);
    kgram_histogram(T_h, n, cpu_hist.data(), k);
    return gpu_hist == cpu_hist;
}


template <typename T_sym_>
bool validate_partition_IDs(const T_sym_* const T, const idx_t n, const std::size_t k, const std::vector<pivot_t>& X, const p_id_t* const P_id_d)
{
    std::vector<p_id_t> gpu_P_id(n);
    CUDA_CHECK(cudaMemcpy(gpu_P_id.data(), P_id_d, n * sizeof(p_id_t), cudaMemcpyDeviceToHost));

    std::vector<p_id_t> cpu_P_id(n);
    compute_partition_IDs(T, n, cpu_P_id.data(), k, X);
    return gpu_P_id == cpu_P_id;
}



// Template instantiations for the required instances.
template bool validate_kgram_histogram(const char*, idx_t, std::size_t, const freq_cnt_t*);
template bool validate_kgram_histogram(const uint8_t*, idx_t, std::size_t, const freq_cnt_t*);
template bool validate_kgram_histogram(const uint16_t*, idx_t, std::size_t, const freq_cnt_t*);
template bool validate_kgram_histogram(const uint32_t*, idx_t, std::size_t, const freq_cnt_t*);

template bool validate_partition_IDs(const char*, idx_t, std::size_t, const std::vector<pivot_t>&, const p_id_t*);
template bool validate_partition_IDs(const uint8_t*, idx_t, std::size_t, const std::vector<pivot_t>&, const p_id_t*);
template bool validate_partition_IDs(const uint16_t*, idx_t, std::size_t, const std::vector<pivot_t>&, const p_id_t*);
template bool validate_partition_IDs(const uint32_t*, idx_t, std::size_t, const std::vector<pivot_t>&, const p_id_t*);

}
