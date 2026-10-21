
// Allocates and touches all of MemAvailable but a margin, then frees it. The
// kernel reclaims the page cache to make room, and with everything freed at
// once the buddy allocator re-merges free memory into 2 MiB blocks, so the run
// after it starts from the same memory state whatever ran before. Reports the
// node's free and 2 MiB-block totals before and after.
//
//   reset_memory [margin-GiB]      (default 32)

#include "bench.hpp"

#include <sys/mman.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>


namespace
{

constexpr double GiB = 1024.0 * 1024.0 * 1024.0;


int64_t mem_available()
{
    std::ifstream f("/proc/meminfo");
    for(std::string k; f >> k; )
        if(k == "MemAvailable:")
        {
            int64_t kb = 0;
            f >> kb;
            return kb * 1024;
        }

    return -1;
}


void report(const char* const when)
{
    const auto k = bench::read_kernel_counters();
    std::printf("  %-6s  free %7.1f GiB   in 2 MiB blocks %7.1f GiB   available %7.1f GiB\n",
                when, k.buddy_free / GiB, k.buddy_free_huge / GiB, mem_available() / GiB);
}

}


int main(int argc, char** argv)
{
    const int64_t margin = static_cast<int64_t>(argc > 1 ? std::atoll(argv[1]) : 32) << 30;
    report("before");

    const int64_t avail = mem_available();
    if(avail <= margin)
    {
        std::printf("  nothing to touch: %.1f GiB available, margin %.1f GiB\n", avail / GiB, margin / GiB);
        return 0;
    }

    const auto sz = static_cast<std::size_t>(avail - margin);
    void* const m = ::mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if(m == MAP_FAILED)
    {
        std::perror("mmap");
        return 1;
    }

    const auto t0 = bench::Clock::now();
    auto* const p = static_cast<volatile char*>(m);
    const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    #pragma omp parallel for schedule(dynamic, 4096)
    for(std::size_t i = 0; i < sz; i += page)
        p[i] = 1;

    ::munmap(m, sz);
    std::printf("  touched %.1f GiB in %.1fs\n", sz / GiB, bench::since(t0));
    report("after");
    return 0;
}
