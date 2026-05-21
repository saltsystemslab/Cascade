
#ifndef UTILITY_CUH
#define UTILITY_CUH



#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <chrono>


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


namespace caps_sa
{
    // Fields for profiling time.

    typedef std::chrono::high_resolution_clock::time_point time_point_t;
    const static auto now = std::chrono::high_resolution_clock::now;
    const static auto duration = [](const std::chrono::nanoseconds& d) { return std::chrono::duration_cast<std::chrono::duration<double>>(d).count(); };
}


namespace gpu_saca
{
    // Atomically adds `val` to the value at address `addr`, and returns the
    // earlier value from there.
    __device__ __forceinline__
    uint8_t atomic_add_u8(uint8_t* addr, const uint8_t val)
    {
        constexpr uintptr_t byte_off_mask = 0b11llu;    // Mask to extract the byte of interest's offset within the enclosing `u32` word.
        uint32_t* const word_addr = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(addr) & ~byte_off_mask);    // Address of the enclosing `u32` word.
        const uint32_t bit_off    = (reinterpret_cast<uintptr_t>(addr) & byte_off_mask) * 8u;   // Bit-offset of the byte of interest within the enclosing `u32` word.
        const uint32_t clear_mask = ~(0xFFu << bit_off);    // Mask to clear the byte of interest.

        uint32_t old = *word_addr, expected;
        do
        {
            expected = old;
            const uint8_t byte = (expected >> bit_off) & 0xFFu;
            const uint8_t new_byte = byte + val;    // Overflows handled implicitly.

            const uint32_t updated = (expected & clear_mask) | (static_cast<uint32_t>(new_byte) << bit_off);
            old = atomicCAS(word_addr, expected, updated);
        }
        while(old != expected);

        return (old >> bit_off) & 0xFFu;
    }
}



#endif
