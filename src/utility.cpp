
#include "utility.hpp"
#include "utility.cuh"

#include <unistd.h>
#include <fcntl.h>

#include <cstring>
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


namespace util::file
{

void exit_on_io_failure(const char* const op, const std::string& path)
{
    std::cerr << op << " failed at filesystem-path: " << path << "; error: " << std::strerror(errno) << ". Aborting.\n";
    std::exit(EXIT_FAILURE);
}


void write_and_flush(const int fd, const std::string& path, const void* const buf, const std::size_t bytes, const std::size_t off)
{
    std::size_t written = 0;
    while(written < bytes)  // `pwrite` writes at most ~2GB.
    {
        const auto written_bytes = ::pwrite(fd, static_cast<const char*>(buf) + written, bytes - written, off + written);
        if(written_bytes <= 0)  exit_on_io_failure("pwrite", path);
        written += written_bytes;
    }

    // Queue these bytes for writeback right away, instead of leaving it for the flusher.
    // This does not wait, hence the dirty set stays small without the writer stalling.
    if(::sync_file_range(fd, off, bytes, SYNC_FILE_RANGE_WRITE) != 0)
        exit_on_io_failure("sync_file_range", path);
}

}


}
