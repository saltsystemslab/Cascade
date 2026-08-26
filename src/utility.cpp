
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

}


}
