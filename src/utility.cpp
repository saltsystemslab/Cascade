
#include "utility.hpp"
#include "utility.cuh"

#include <fstream>
#include <cstdlib>


namespace gpu_saca
{

void CUDART_CB fatal_error_callback(void* const user_data)
{
    // Check if the stream or device has fallen into an error state.
    const auto err = cudaGetLastError();

    // If the kernel prior to this `__trap()`-ed, `err` will be `cudaErrorLaunchFailure`.
    if(err != cudaSuccess)
    {
        std::cerr <<    "\n[ASYNC BACKDOOR] Fatal GPU error detected: " << cudaGetErrorString(err) <<
                        "; due to: " << static_cast<const char*>(user_data) << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }
}


namespace util
{

std::size_t available_memory()
{
    constexpr const char* field = "MemAvailable:";
    std::ifstream meminfo("/proc/meminfo");
    std::string line;

    while(std::getline(meminfo, line))
        if(line.rfind(field, 0) == 0)
        {
            std::stringstream ss(line.substr(sizeof("MemAvailable:") - 1));
            std::size_t memory_KB = 0;
            if(ss >> memory_KB)
                return memory_KB * 1024;
        }

    std::cerr << "Could not read `MemAvailable` from /proc/meminfo. Aborting.\n";
    std::exit(EXIT_FAILURE);
}


std::pair<std::size_t, std::size_t> thp_coverage(const void* const p)
{
    std::ifstream smaps("/proc/self/smaps");
    if(!smaps)
        return {0, 0};

    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    bool in_target_block = false;
    std::size_t rss = 0, thp = 0;

    std::string line;
    while(std::getline(smaps, line))
    {
        std::uintptr_t low, high;
        if(std::sscanf(line.c_str(), "%lx-%lx", &low, &high) == 2)  // A mapping header.
        {
            if(in_target_block) // Just walked past the target mapping block.
                break;

            in_target_block = (addr >= low && addr < high);
            continue;
        }

        if(!in_target_block)
            continue;


        // Parse metrics from the target block.

        std::size_t kb;
        if(std::sscanf(line.c_str(), "Rss: %zu kB", &kb) == 1)
            rss = (kb << 10);
        else if(std::sscanf(line.c_str(), "AnonHugePages: %zu kB", &kb) == 1 ||
                std::sscanf(line.c_str(), "ShmemHugePages: %zu kB", &kb) == 1 ||
                std::sscanf(line.c_str(), "FileHugeMapped: %zu kB", &kb) == 1) 
            thp += (kb << 10);
    }

    return {rss, thp};
}

}


}
