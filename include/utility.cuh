
#ifndef UTILITY_CUH
#define UTILITY_CUH



#include "utility.hpp"

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

    // Allocates space for `n` elements into `buf`, disk-block aligned for
    // `O_DIRECT`, out of the pinned allocation `alloc` it makes with a disk-
    // block of slack.
    template <typename T_>
    void pinned_allocate_aligned_host(T_*& buf, void*& alloc, const std::size_t n)
    {
        constexpr std::size_t DISK_BLOCK = 4096;

        char* raw;
        CUDA_CHECK(pinned_allocate_host(raw, util::math::block_up<DISK_BLOCK>(n * sizeof(T_)) + DISK_BLOCK));

        alloc = raw;
        buf = reinterpret_cast<T_*>((reinterpret_cast<std::uintptr_t>(raw) + DISK_BLOCK - 1) & ~(DISK_BLOCK - 1));
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


    // Blocks until the work enqueued on stream `s` completes, and aborts if it
    // failed, reporting `reason` as the cause.
    void synchronize_or_abort(cudaStream_t s, const char* reason);
}



#endif
