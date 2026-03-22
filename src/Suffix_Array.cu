

#include "Suffix_Array.hpp"
#include "utility.hpp"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <iostream>


namespace caps_sa
{

void build_SA(const char* const T_h, std::size_t n, std::size_t* const SA_h)
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
    thrust::device_vector<std::size_t> SA_d(n);
    thrust::sequence(SA_d.begin(), SA_d.end());

    // Sort suffixes
    thrust::sort(SA_d.begin(), SA_d.end(),
    [T_d, n] __device__ (std::size_t i, std::size_t j)
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
    cudaMemcpy(SA_h, thrust::raw_pointer_cast(SA_d.data()), n * sizeof(std::size_t), cudaMemcpyDeviceToHost);

    const auto t_1 = now();
    std::cerr << "Time taken for parallel SA construction and data movement: " << duration(t_1 - t_0) << " s.\n";
    std::cerr << "Time taken for parallel SA construction kernel: " << duration(t_e - t_s) << " s.\n";
}

}

