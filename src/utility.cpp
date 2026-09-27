
#include "utility.hpp"
#include "utility.cuh"
#include "env.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>

#include <cstring>
#include <fstream>
#include <cerrno>
#include <cstdlib>


namespace gpu_saca
{

void synchronize_or_abort(const cudaStream_t s, const char* const reason)
{
    const auto err = cudaStreamSynchronize(s);

    // If a kernel on `s` `__trap()`-ed, `err` will be `cudaErrorLaunchFailure`.
    if(err != cudaSuccess)
    {
        std::cerr << "\nFatal GPU error detected: " << cudaGetErrorString(err) << "; due to: " << reason << ". Aborting.\n";
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


void* reserve_memory(const std::size_t bytes)
{
    if(bytes == 0)
        return nullptr;

    // Over-map by a huge page, and trim the mapping down to a huge-page aligned base.
    const auto res_bytes = (bytes + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
    const auto map_bytes = res_bytes + HUGE_PAGE_SIZE;
    void* const map = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if(map == MAP_FAILED)
    {
        std::cerr   << "Could not reserve " << (res_bytes / (1024.0 * 1024.0 * 1024.0)) << "GBs of address-space: " << std::strerror(errno) << "."
                    << " Address-space reservations require `vm.overcommit_memory` to be 0 or 1. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    const auto map_addr = reinterpret_cast<std::uintptr_t>(map);
    const auto base = (map_addr + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
    const auto head = base - map_addr;  // Unaligned head to trim.
    const auto tail = map_bytes - head - res_bytes; // Surplus tail to trim.
    if(head > 0)
        ::munmap(map, head);
    if(tail > 0)
        ::munmap(reinterpret_cast<void*>(base + res_bytes), tail);

    if(env::get_u64("GPU_SACA_MADV_HUGEPAGE").value_or(0) != 0) // Best-effort: opts into synchronous compaction on faults too, under `defrag=madvise`.
        ::madvise(reinterpret_cast<void*>(base), res_bytes, MADV_HUGEPAGE);

    return reinterpret_cast<void*>(base);
}


void unreserve_memory(void* const ptr, const std::size_t bytes)
{
    if(ptr != nullptr)
        ::munmap(ptr, (bytes + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1));
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


void raise_open_files_limit(const std::size_t required)
{
    rlimit lim;
    if(::getrlimit(RLIMIT_NOFILE, &lim) != 0)
    {
        std::cerr << "getrlimit(RLIMIT_NOFILE) failed; error: " << std::strerror(errno) << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    if(lim.rlim_cur == RLIM_INFINITY || lim.rlim_cur >= required)
        return;

    if(lim.rlim_max != RLIM_INFINITY && lim.rlim_max < required)
    {
        std::cerr << "The hard limit of " << lim.rlim_max << " open files falls below the " << required << " the spill-store may require. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    lim.rlim_cur = (lim.rlim_max == RLIM_INFINITY ? required : lim.rlim_max);   // The open-files limit cannot go infinite.
    if(::setrlimit(RLIMIT_NOFILE, &lim) != 0)
    {
        std::cerr << "setrlimit(RLIMIT_NOFILE) failed; error: " << std::strerror(errno) << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }
}

}


}
