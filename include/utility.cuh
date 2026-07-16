
#ifndef UTILITY_CUH
#define UTILITY_CUH



#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <cstdlib>


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



namespace gpu_saca  // TODO: why are all the following in the parent namespace?
{
    constexpr std::size_t CONST_MEM_SZ = 64 * 1024; // 64-KB of constant memory across all CUDA compute capabilities.

    // Allocates memory for `sz` elements of type `T_` at the host, pins it, and
    // places its address at `ptr`. Returns CUDA error-type encountered, if any.
    template <typename T_>
    auto pinned_allocate_host(T_*& ptr, const std::size_t sz)
    {
        return cudaHostAlloc(reinterpret_cast<void**>(&ptr), sz * sizeof(T_), cudaHostAllocDefault);
    }

    // Deallocates the pointer `ptr`, allocated with `pinned_allocate_host`.
    // Returns CUDA error-type encountered, if any.
    inline auto deallocate_host(void* const ptr)
    {
        return cudaFreeHost(ptr);
    }


    // Logs the current amount of free device-memory.
    static void log_free_mem(const char* const where)
    {
        std::size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr   << "[mem] " << where << ": free = "
                    << (free_mem / (1024.0*1024.0*1024.0)) << " GBs.\n";
    }


    // Atomically adds `val` to the value at address `addr`, and returns the
    // earlier value from there.
    __device__
    __forceinline__ uint8_t atomic_add_u8(uint8_t* addr, const uint8_t val)
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
