
#ifndef UTILITY_CUH
#define UTILITY_CUH



#include "utility.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <mutex>
#include <unordered_map>


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


    // Device-memory accounting, so a record can report the peak the run actually
    // held.
    namespace dev_mem
    {
        inline std::mutex mut;
        inline std::unordered_map<const void*, std::size_t> block;
        inline std::size_t live = 0;
        inline std::size_t peak = 0;
    }


    // Allocates `bytes` of device-memory and places its address at `ptr`,
    // counting it toward the peak. Returns CUDA error-type encountered, if any.
    template <typename T_>
    auto device_allocate(T_** const ptr, const std::size_t bytes)
    {
        const auto err = cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
        if(err != cudaSuccess)
            return err;

        const std::lock_guard<std::mutex> lock(dev_mem::mut);
        dev_mem::block.emplace(*ptr, bytes);
        dev_mem::live += bytes;
        if(dev_mem::live > dev_mem::peak)
            dev_mem::peak = dev_mem::live;

        return err;
    }


    // Deallocates the device-pointer `ptr`, allocated with `device_allocate`.
    // Returns CUDA error-type encountered, if any.
    inline auto device_free(const void* const ptr)
    {
        {
            const std::lock_guard<std::mutex> lock(dev_mem::mut);
            const auto it = dev_mem::block.find(ptr);
            if(it != dev_mem::block.end())
            {
                dev_mem::live -= it->second;
                dev_mem::block.erase(it);
            }
        }

        return cudaFree(const_cast<void*>(ptr));
    }


    // Restarts the high-water mark from what is held right now, so a mark read
    // after one run does not carry another's.
    inline void reset_device_peak()
    {
        const std::lock_guard<std::mutex> lock(dev_mem::mut);
        dev_mem::peak = dev_mem::live;
    }


    // Returns the most device-memory held at once, in bytes, since the last
    // `reset_device_peak`, or since the process started if there was none.
    inline std::size_t device_peak_bytes()
    {
        const std::lock_guard<std::mutex> lock(dev_mem::mut);
        return dev_mem::peak;
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
