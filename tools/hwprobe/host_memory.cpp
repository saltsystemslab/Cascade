
// Random-access rates of host memory from the CPU cores. Feeds the CPU
// executor's cost model (phase 2 over pageable host memory) and the depth of
// its software pipelining.
//
// Usage: host_memory [--gib F] [--pages {4k,thp}] [--threads T1,T2,...] [--lanes G1,G2,...] [--seconds S] [--label L]
//   --gib      footprint in GiB; default 64
//   --pages    4k: `MADV_NOHUGEPAGE`; thp: `MADV_HUGEPAGE`; default thp
//   --threads  thread counts, `all` being the affinity mask's CPU count;
//              default 1,all
//   --lanes    independent dependent-chains per thread, i.e. the software
//              memory-level parallelism; default 1,2,4,8,16,32
//   --label    tag for the rows, e.g. the NUMA placement of the buffer
//
// Every thread follows `lanes` interleaved chains of dependent random 8-byte
// loads, each touching a random 64-byte line; the next address of a chain
// derives from its loaded value. CSV to stdout, one row per (threads, lanes):
//   label,pages,thp_frac,gib,threads,lanes,loads,seconds,mloads,mloads_per_thread,ns_per_load_per_lane
// `ns_per_load_per_lane` approximates the loaded latency of a miss.

#include <sys/mman.h>
#ifdef __linux__
    #include <sched.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>


namespace
{

constexpr double GIB = 1024.0 * 1024.0 * 1024.0;


inline uint64_t mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// Returns an index in `[0, n)` derived from the random word `h`.
inline uint64_t pick(const uint64_t h, const uint64_t n)
{
    return static_cast<uint64_t>((static_cast<unsigned __int128>(h) * n) >> 64);
}


// Follows `G_` interleaved chains of `steps` dependent random loads each over
// `buf[0, n)`, and returns a digest of their ends.
template <int G_>
uint64_t chase(const uint64_t* const buf, const uint64_t n, const uint64_t steps, const uint64_t seed)
{
    uint64_t x[G_];
    for(int g = 0; g < G_; ++g)
        x[g] = mix(seed * 64 + g);

    for(uint64_t s = 0; s < steps; ++s)
        for(int g = 0; g < G_; ++g)
            x[g] = mix(x[g] ^ buf[pick(x[g], n)]);

    uint64_t acc = 0;
    for(int g = 0; g < G_; ++g)
        acc ^= x[g];

    return acc;
}

using Chase_Fn = uint64_t (*)(const uint64_t*, uint64_t, uint64_t, uint64_t);

Chase_Fn chase_for(const int g)
{
    switch(g)
    {
    case 1:     return chase<1>;
    case 2:     return chase<2>;
    case 4:     return chase<4>;
    case 8:     return chase<8>;
    case 16:    return chase<16>;
    case 32:    return chase<32>;
    default:    return nullptr;
    }
}


// Returns the number of CPUs in the affinity mask.
unsigned cpu_count()
{
#ifdef __linux__
    cpu_set_t s;
    if(sched_getaffinity(0, sizeof(s), &s) == 0)
        return std::max(1, CPU_COUNT(&s));
#endif
    return std::max(1u, std::thread::hardware_concurrency());
}


double seconds_since(const std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


// Returns the wall-clock seconds of `t_c` threads running `f` for `steps`
// steps each from a common start.
double run(const Chase_Fn f, const uint64_t* const buf, const uint64_t n, const uint64_t steps, const unsigned t_c)
{
    std::atomic<unsigned> ready{0};
    std::atomic<bool> go{false};
    std::atomic<uint64_t> sink{0};
    std::vector<std::thread> th;
    for(unsigned t = 0; t < t_c; ++t)
        th.emplace_back([&, t]
        {
            ready.fetch_add(1);
            while(!go.load(std::memory_order_acquire))
                ;
            sink.fetch_xor(f(buf, n, steps, t + 1), std::memory_order_relaxed);
        });

    while(ready.load() < t_c)
        ;
    const auto t0 = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for(auto& t : th)
        t.join();

    const double s = seconds_since(t0);
    if(sink.load() == 42)   // Keeps the loads observable.
        std::fprintf(stderr, " ");
    return s;
}


// Returns the fraction of the mapping containing `p` backed by transparent huge
// pages, from `/proc/self/smaps`; `-1` if unknown.
double thp_fraction(const void* const p)
{
    std::ifstream f("/proc/self/smaps");
    if(!f)
        return -1;

    const auto a = reinterpret_cast<uintptr_t>(p);
    bool in = false;
    std::size_t rss = 0, thp = 0;
    for(std::string line; std::getline(f, line); )
    {
        unsigned long lo = 0, hi = 0;
        const auto dash = line.find('-'), space = line.find(' ');
        if(dash != std::string::npos && dash < space && std::sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2)
        {
            if(in)  break;
            in = (a >= lo && a < hi);
            continue;
        }

        if(!in) continue;
        std::size_t kb = 0;
        if(std::sscanf(line.c_str(), "Rss: %zu kB", &kb) == 1)  rss = kb;
        else if(std::sscanf(line.c_str(), "AnonHugePages: %zu kB", &kb) == 1)  thp = kb;
    }

    return rss ? double(thp) / rss : -1;
}


// Returns the comma-separated integers of `s`, with `all` read as `all_val`.
std::vector<unsigned> parse_list(const std::string& s, const unsigned all_val)
{
    std::vector<unsigned> v;
    for(std::size_t b = 0; b <= s.size(); )
    {
        const auto e = s.find(',', b);
        const std::string x = s.substr(b, (e == std::string::npos ? s.size() : e) - b);
        v.push_back(x == "all" ? all_val : static_cast<unsigned>(std::strtoul(x.c_str(), nullptr, 10)));
        if(e == std::string::npos)  break;
        b = e + 1;
    }

    return v;
}

}


int main(int argc, char** argv)
{
    double gib = 64, target = 0.5;
    std::string pages = "thp", label = "-", threads_s = "1,all", lanes_s = "1,2,4,8,16,32";
    for(int i = 1; i < argc; ++i)
    {
        const std::string k = argv[i];
        if(k == "--gib" && i + 1 < argc)            gib = std::atof(argv[++i]);
        else if(k == "--pages" && i + 1 < argc)     pages = argv[++i];
        else if(k == "--threads" && i + 1 < argc)   threads_s = argv[++i];
        else if(k == "--lanes" && i + 1 < argc)     lanes_s = argv[++i];
        else if(k == "--seconds" && i + 1 < argc)   target = std::atof(argv[++i]);
        else if(k == "--label" && i + 1 < argc)     label = argv[++i];
        else
        {
            std::fprintf(stderr, "usage: host_memory [--gib F] [--pages {4k,thp}] [--threads T,...|all] [--lanes G,...] [--seconds S] [--label L]\n");
            return 2;
        }
    }

    const unsigned cpus = cpu_count();
    const auto thread_counts = parse_list(threads_s, cpus);
    const auto lane_counts = parse_list(lanes_s, 1);
    for(const auto g : lane_counts)
        if(!chase_for(g))   { std::fprintf(stderr, "unsupported lane count %u; use 1, 2, 4, 8, 16, or 32\n", g); return 2; }

    // Allocate, advise the page size, and first-touch from all CPUs in parallel.
    const std::size_t bytes = (static_cast<std::size_t>(gib * GIB) >> 21) << 21;
    void* const m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(m == MAP_FAILED) { std::fprintf(stderr, "mmap of %.1f GiB failed\n", bytes / GIB); return 1; }
#ifdef MADV_HUGEPAGE
    madvise(m, bytes, pages == "4k" ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
#endif

    auto* const buf = static_cast<uint64_t*>(m);
    const uint64_t n = bytes / 8;
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> th;
        for(unsigned t = 0; t < cpus; ++t)
            th.emplace_back([=]
            {
                for(uint64_t i = n * t / cpus; i < n * (t + 1) / cpus; ++i)
                    buf[i] = mix(i);
            });
        for(auto& t : th)
            t.join();
    }

    const double thp = thp_fraction(m);
    std::printf("# setup: %.1f GiB, pages %s, THP fraction %.3f, first-touch %.2f s on %u cpus\n", bytes / GIB, pages.c_str(), thp, seconds_since(t0), cpus);
    std::printf("label,pages,thp_frac,gib,threads,lanes,loads,seconds,mloads,mloads_per_thread,ns_per_load_per_lane\n");

    for(const unsigned t_c : thread_counts)
        for(const unsigned g : lane_counts)
        {
            const auto f = chase_for(g);

            // Calibrate the steps per thread to about `target` seconds.
            uint64_t steps = 1 << 10;
            double s = run(f, buf, n, steps, t_c);
            while(s < target / 8)
            {
                steps *= 8;
                s = run(f, buf, n, steps, t_c);
            }
            steps = std::max<uint64_t>(1, static_cast<uint64_t>(steps * (target / s)));
            s = run(f, buf, n, steps, t_c);

            const double loads = double(steps) * g * t_c;
            std::printf("%s,%s,%.3f,%.1f,%u,%u,%.0f,%.6f,%.2f,%.2f,%.2f\n", label.c_str(), pages.c_str(), thp, bytes / GIB, t_c, g, loads, s,
                        loads / s / 1e6, loads / s / 1e6 / t_c, s * 1e9 / steps);
            std::fflush(stdout);
        }

    munmap(m, bytes);
    return 0;
}
