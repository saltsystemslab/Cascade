
#include "Parser.hpp"
#include "utility.hpp"

#include <cstddef>
#include <string>
#include <iostream>


__host__
void transform_seq(const std::string& path, const std::size_t trials)
{
    double sum_time = 0;

    for(std::size_t trial = 0; trial < trials; ++trial)
    {
        caps_sa::Parser input(path);
        while(input.read_next_seq())
        {
            auto T = input.seq();
            auto const len = input.seq_len();

            const auto t_s = caps_sa::now();
            for(std::size_t i = 0; i < len; ++i)
                T[i] &= 0b0101'1111;

            const auto t_e = caps_sa::now();
            sum_time += caps_sa::duration(t_e - t_s);
        }

        input.close();
    }

    std::cerr << "Mean time taken for sequential transform: " << (sum_time / trials) << " s.\n";
}


__global__
void transform_kernel(char* const T, const std::size_t n)
{
    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < n)
        T[thread_id] &= 0b0101'1111;
}



__host__
void transform_par(const std::string& path, const std::size_t trials)
{
    double sum_time_kernel = 0;
    double sum_time_whole = 0;

    for(std::size_t trial = 0; trial < trials; ++trial)
    {
        caps_sa::Parser input(path);
        while(input.read_next_seq())
        {
            auto const T_h = input.seq();
            auto const n = input.seq_len();
            const auto bytes = n * sizeof(char);

            const auto t_0 = caps_sa::now();

            char* T_d;
            cudaMalloc(reinterpret_cast<void**>(&T_d), bytes);
            cudaMemcpy(T_d, T_h, bytes, cudaMemcpyHostToDevice);

            const auto t_s = caps_sa::now();

            constexpr auto blk_sz = 1024;
            transform_kernel<<< (n + blk_sz - 1) / blk_sz, blk_sz >>>(T_d, n);
            cudaDeviceSynchronize();
            const auto err = cudaGetLastError();    // Errors from inside the kernel, if any
            if (err != cudaSuccess)
            {
                std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << "\n";
                std::exit(EXIT_FAILURE);
            }

            const auto t_e = caps_sa::now();
            sum_time_kernel += caps_sa::duration(t_e - t_s);

            cudaMemcpy(T_h, T_d, bytes, cudaMemcpyDeviceToHost);
            const auto t_1 = caps_sa::now();
            sum_time_whole += caps_sa::duration(t_1 - t_0);
        }

        input.close();
    }

    std::cerr << "Mean time taken for parallel transform kernel: " << (sum_time_kernel / trials) << " s.\n";
    std::cerr << "Mean time taken for parallel transform: " << (sum_time_whole / trials) << " s.\n";
}


__host__
void transform_exp(const std::string& path, const std::size_t trials)
{
    transform_seq(path, trials);
    transform_par(path, trials);
}


int main(int argc, char** argv)
{
    (void)argc, (void)argv;

    return 0;
}
