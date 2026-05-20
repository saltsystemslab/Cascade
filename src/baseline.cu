

#include "utility.cuh"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <curand_kernel.h>
#include <cstddef>
#include <utility>
#include <iostream>
#include <cstdlib>


namespace caps_sa
{

typedef std::size_t idx_t;

__host__
void build_SA_naive(const char* const T_h, idx_t n, idx_t* const SA_h)
{
    const auto t_0 = now();

    n++;    // Account for the terminator symbol '\0'.

    // Copy text to device memory

    char* T_d;
    const auto bytes = n * sizeof(char);
    cudaMalloc(reinterpret_cast<void**>(&T_d), bytes);
    cudaMemcpy(T_d, T_h, bytes, cudaMemcpyHostToDevice);


    const auto t_s = now();

    // Initialize SA with `[0, n)`
    thrust::device_vector<idx_t> SA_d(n);
    thrust::sequence(SA_d.begin(), SA_d.end());

    // Sort suffixes
    thrust::sort(SA_d.begin(), SA_d.end(),
    [T_d, n] __device__ (idx_t i, idx_t j)
    {
        for(; i < n && j < n; ++i, ++j)
            if(T_d[i] != T_d[j])
                return T_d[i] < T_d[j]; // First mismatch.

        return i > j;   // Shorter suffixes come before in sorted order.
    });

    cudaDeviceSynchronize();
    const auto err = cudaGetLastError();    // Errors from inside the kernel, if any
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }

    const auto t_e = now();


    cudaFree(T_d);
    cudaMemcpy(SA_h, thrust::raw_pointer_cast(SA_d.data()), n * sizeof(idx_t), cudaMemcpyDeviceToHost);

    const auto t_1 = now();
    std::cerr << "Time taken for parallel SA construction and data movement: " << duration(t_1 - t_0) << " s.\n";
    std::cerr << "Time taken for parallel SA construction kernel: " << duration(t_e - t_s) << " s.\n";
}


// Initializes the suffix array with `[0, n)` and populates the initial rank of
// the suffixes.
__global__
void initialize_kernel(const char* const T, idx_t* const SA, idx_t* const R, const idx_t n)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        SA[idx] = idx,
        R[idx] = T[idx];    // A suffix's initial rank is its first symbol.
}


// Get the pair of ranks for the suffixes when the prefix length is doubled.
__global__
void build_rank_pairs_kernel(const idx_t* const R, std::pair<int64_t, int64_t>* const P, const idx_t n, const idx_t prefix_sz)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        P[idx].first  = R[idx],
        P[idx].second = (idx + prefix_sz < n ? static_cast<int64_t>(R[idx + prefix_sz]) : -1);
}


// Mark the stretches of suffixes equal up-to the current prefix size.
__global__
void mark_rank_boundaries_kernel(const idx_t* SA, const std::pair<int64_t, int64_t>* const P, idx_t* const C, const idx_t n)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        C[idx] = (idx == 0 ? 0 : P[SA[idx]] != P[SA[idx - 1]]);
}


// Get the new ranks of the suffixes from the equal-prefix-stretch boundaries.
__global__
void rank_from_scan_kernel(const idx_t* const SA, const idx_t* const S, idx_t* const R, const idx_t n)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        R[SA[idx]] = S[idx];
}


__host__
void build_SA_prefix_doubling(const char* T_h, std::size_t n, std::size_t* SA_h)
{
    const auto t_0 = now();

    n++;    // Account for the terminator symbol '\0'.

    constexpr std::size_t blk_sz = 256;
    const std::size_t grid_sz = (n + blk_sz - 1) / blk_sz;

    char* T_d;
    const auto bytes = n * sizeof(char);

    // Copy text to device memory.
    cudaMalloc(reinterpret_cast<void**>(&T_d), bytes);
    cudaMemcpy(T_d, T_h, bytes, cudaMemcpyHostToDevice);


    const auto t_s = now();

    thrust::device_vector<std::size_t> SA_vec_d(n); // The suffix array.
    thrust::device_vector<std::size_t> R_vec_d(n);  // The rank array.
    thrust::device_vector<std::pair<int64_t, int64_t>> P_vec_d(n);  // Rank pairs.
    thrust::device_vector<std::size_t> C_vec_d(n); // Changing points of ranks.
    thrust::device_vector<std::size_t> S_vec_d(n);  // Prefix-sum (scan) of the rank-change boundaries.

    auto* const SA_d = thrust::raw_pointer_cast(SA_vec_d.data());
    auto* const R_d = thrust::raw_pointer_cast(R_vec_d.data());
    auto* const P_d = thrust::raw_pointer_cast(P_vec_d.data());
    auto* const C = thrust::raw_pointer_cast(C_vec_d.data());
    auto* const S_d = thrust::raw_pointer_cast(S_vec_d.data());

    // Initial ranking and SA population.
    {
        initialize_kernel<<<grid_sz, blk_sz>>>(T_d, SA_d, R_d, n);
        cudaDeviceSynchronize();
    }

    // Prefix-doubling
    {
        for(std::size_t prefix_sz = 1; prefix_sz < n; prefix_sz *= 2)
        {
            // Get the pairs
            build_rank_pairs_kernel<<<grid_sz, blk_sz>>>(R_d, P_d, n, prefix_sz);
            cudaDeviceSynchronize();

            // Sort suffixes by the rank pairs
            thrust::sort(SA_vec_d.begin(), SA_vec_d.end(),
                [P_d] __device__ (const std::size_t s_x, const std::size_t s_y)
                {
                    const auto p_x = P_d[s_x], p_y = P_d[s_y];
                    return p_x < p_y;
                }
            );
            cudaDeviceSynchronize();

            // Re-rank the suffixes
            {
                // Mark new rank boundaries
                mark_rank_boundaries_kernel<<<grid_sz, blk_sz>>>(SA_d, P_d, C, n);
                cudaDeviceSynchronize();

                 // Prefix-sum (scan) of the rank-changes to get new ranks.
                thrust::inclusive_scan(C_vec_d.begin(), C_vec_d.end(), S_vec_d.begin());
                cudaDeviceSynchronize();

                // Re-rank
                rank_from_scan_kernel<<<grid_sz, blk_sz>>>(SA_d, S_d, R_d, n);
                cudaDeviceSynchronize();
            }

            // Early-exit if already sorted
            std::size_t max_suf, max_suf_rank;
            cudaMemcpy(&max_suf, SA_d + n - 1, sizeof(max_suf), cudaMemcpyDeviceToHost);
            cudaMemcpy(&max_suf_rank, R_d + max_suf, sizeof(max_suf_rank), cudaMemcpyDeviceToHost);
            if(max_suf_rank == n - 1)
                break;
        }
    }

    const auto t_e = now();


    cudaFree(T_d);
    cudaMemcpy(SA_h, SA_d, n * sizeof(std::size_t), cudaMemcpyDeviceToHost);


    const auto t_1 = now();
    std::cerr << "Time taken for parallel SA construction and data movement: " << duration(t_1 - t_0) << " s.\n";
    std::cerr << "Time taken for parallel SA construction: " << duration(t_e - t_s) << " s.\n";
}

}

