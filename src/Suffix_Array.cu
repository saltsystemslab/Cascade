

#include "Suffix_Array.hpp"
#include "utility.hpp"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <curand_kernel.h>
#include <cstddef>
#include <cstdlib>
#include <cmath>


namespace caps_sa
{

__host__
Suffix_Array::Suffix_Array(const char* const T, const idx_t n_):
      T_h(T)
    , n(n_ + 1)  // Account for the terminator symbol
    , chunk_count((n + chunk_sz - 1) / chunk_sz)
    , partition_count(chunk_count)  // TODO: number of final partitions is kept the same as the number of first-phase chunks now.
    , k(((c * std::log2(n) + 31 ) / 32) * 32)   // `c log n`
{
    // Copy over text to device
    const auto bytes = n * sizeof(char);
    cudaMalloc(reinterpret_cast<void**>(&T_d), bytes);
    cudaMemcpy(T_d, T_h, bytes, cudaMemcpyHostToDevice);

    // Allocate space for suffix array
    cudaMalloc(reinterpret_cast<void**>(&SA_d), n * sizeof(idx_t));
}


__host__
Suffix_Array::~Suffix_Array()
{
    cudaFree(T_d), cudaFree(SA_d);
}


__host__
void Suffix_Array::build_SA(idx_t* const SA)
{
    // Get the sample suffix collection.
    sample_suffixes();

    // Select the final pivots.
    select_pivots();

    // Sort the chunks.
    sort_chunks(SA_d, chunk_sz);

    // Copy back the chunked suffix array to host.
    cudaMemcpy(SA, SA_d, n * sizeof(idx_t), cudaMemcpyDeviceToHost);
}


__host__
void Suffix_Array::sort_chunks(idx_t* const SA_d, const idx_t chunk_sz) const
{
    for(std::size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
    {
        const auto chunk_begin = chunk_idx * chunk_sz;
        const auto chunk_len   = std::min(chunk_sz, n - chunk_begin);
        auto* const SA = SA_d + chunk_begin;

        // Populate the initial suffix array for this chunk.
        thrust::sequence(thrust::device, SA, SA + chunk_len, chunk_begin);

        // Sort each chunk successively.
        thrust::sort(thrust::device, SA, SA + chunk_len, Suffix_Comparator(T_d, n));
        cudaDeviceSynchronize();
    }
}


__host__
void Suffix_Array::sample_suffixes()
{
    const std::size_t k = ((c * std::log2(n) + 31 ) / 32) * 32;  // Sample per chunk: `c log n`

    cudaMalloc(reinterpret_cast<void**>(&S_d), k * chunk_count * sizeof(idx_t));

    sample_with_replacement<<<chunk_count, blk_sz>>>(chunk_sz, k, S_d, 0);
    cudaDeviceSynchronize();
}


__host__
void Suffix_Array::select_pivots()
{
    // Sort the collected samples.
    const auto sample_count = k * chunk_count;
    thrust::sort(thrust::device, S_d, S_d + sample_count, Suffix_Comparator(T_d, n));
    cudaDeviceSynchronize();

    std::vector<idx_t> S_h(sample_count);
    cudaMemcpy(S_h.data(), S_d, sample_count * sizeof(idx_t), cudaMemcpyDeviceToHost);

    const auto p = chunk_count; // Number of final partitions.
    P_h = reinterpret_cast<idx_t*>(std::malloc((p - 1) * sizeof(idx_t)));

    const auto gap = sample_count / p;  // Distance-gap between pivots.
    const auto quant_err = sample_count % p;    // Numerator of the quantization error per partition.
    idx_t unaccounted = 0;  // Running total number of samples not properly accounted for due to quantization of partition sizes.
    std::size_t idx = 0;    // Index of the next pivot to choose from the samples.
    for(idx_t i = 0; i < p - 1; ++i)
    {
        idx += gap;
        unaccounted += quant_err;
        if(unaccounted >= p)    // Fix the cumulative error.
            idx++,
            unaccounted %= p;

        P_h[i] = S_h[idx - 1];
    }
}


__global__
void sample_from_chunk_with_replacement(const idx_t* const input, idx_t* const output, const idx_t chunk_sz, const idx_t k, const uint64_t seed)
{
    const auto chunk_id =  blockIdx.x;
    const auto thread_id = threadIdx.x;

    const uint64_t off = blockIdx.x * blockDim.x + threadIdx.x; // Unique sequence-offset per thread for random number generation.
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, off, 0, &rng);

    auto* const chunk_ip = input + chunk_id * chunk_sz;
    auto* const chunk_op = output + chunk_id * k;
    for(uint32_t i = thread_id; i < k; i += blockDim.x)  // A thread samples the `[t_id, t_id + dim.x, ...]`'th samples. A chunk can have a larger size than the block.
    {
        const auto sample_idx = curand(&rng) & (chunk_sz - 1);
        chunk_op[i] = chunk_ip[sample_idx];
    }
}


__global__
void sample_with_replacement(const idx_t chunk_sz, const idx_t k, idx_t* const output, const uint64_t seed)
{
    const auto chunk_id = blockIdx.x;
    const auto thread_id = threadIdx.x;

    const uint64_t off = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;  // Unique sequence-offset per thread for random number generation.
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, off, 0, &rng);

    assert((chunk_sz & (chunk_sz - 1)) == 0);   // `chunk_sz` must be a power-of-two.
    auto* const chunk_op = output + chunk_id * k;
    const auto shift = chunk_id * chunk_sz;
    for(uint32_t i = thread_id; i < k; i += blockDim.x) // A thread samples the `[t_id, t_id + dim.x, ...]`'th samples, since a chunk may have larger size than the thread-block.
        chunk_op[i] = (curand(&rng) & (chunk_sz - 1)) + shift;
}


__global__
void validate_kernel(const char* const T_d, const idx_t* const SA_d, bool* const is_sorted, const idx_t n)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        if((idx & (chunk_sz - 1)) != 0)
            is_sorted[idx] = Suffix_Comparator(T_d, n)(SA_d[idx - 1], SA_d[idx]);
        else
            is_sorted[idx] = true;
}


__host__
bool Suffix_Array::validate()
{
    bool* is_sorted_d;
    cudaMalloc(reinterpret_cast<void**>(&is_sorted_d), n * sizeof(bool));

    validate_kernel<<<(n + blk_sz - 1) / blk_sz, blk_sz>>>(T_d, SA_d, is_sorted_d, n);
    cudaDeviceSynchronize();

    bool* is_sorted_h = reinterpret_cast<bool*>(std::malloc(n * sizeof(bool)));
    cudaMemcpy(is_sorted_h, is_sorted_d, n * sizeof(bool), cudaMemcpyDeviceToHost);

    bool r = true;
    for(idx_t i = 0; i < n; ++i)
        r = r && is_sorted_h[i];

    cudaFree(is_sorted_d);
    std::free(is_sorted_h);

    return r;
}

}

