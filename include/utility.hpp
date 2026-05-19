
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>


namespace caps_sa
{
    // Fields for profiling time.

    typedef std::chrono::high_resolution_clock::time_point time_point_t;
    const static auto now = std::chrono::high_resolution_clock::now;
    const static auto duration = [](const std::chrono::nanoseconds& d) { return std::chrono::duration_cast<std::chrono::duration<double>>(d).count(); };

    // CUDA error-checker.
    #define CUDA_CHECK(call) \
    do \
    { \
        cudaError_t err = (call); \
        if(err != cudaSuccess) \
        { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " — " << cudaGetErrorString(err) << "\n"; \
            std::exit(EXIT_FAILURE); \
        } \
    } \
    while(0)
}
