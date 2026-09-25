
// Zero-copy access rates of pinned host memory from the device. Feeds
// the host tier's cost model: a suffix array in pinned host memory, read per
// probe or per window, and a text split across the device and the host.
//
// Usage: pinned_host --alloc {hostalloc,register} --gib F1,F2,... [--seconds S] [--label L]
//   --alloc    hostalloc: `cudaHostAlloc(Mapped)`. register: anonymous `mmap`
//              with transparent huge pages, touched in parallel, then
//              `cudaHostRegister(Mapped | ReadOnly)` in 16 GiB chunks.
//   --gib      footprints in GiB; the buffer is allocated at the largest.
//   --label    tag for the rows, e.g. the NUMA placement of the buffer.
//
// CSV to stdout, one row per (test, footprint):
//   alloc,label,test,footprint_gib,access_bytes,threads,ops,seconds,mops,payload_gbps
// Tests:
//   thread_<B>  every thread issues independent random `B`-byte loads: per-probe
//               suffix-array reads.
//   warp_<B>    every warp reads `B / 128` consecutive 128-byte lines at a random
//               128-byte-aligned offset, coalesced: window gathers.
//   memcpy_h2d  sequential `cudaMemcpy` from the buffer, in 1 GiB chunks.
// Setup lines (`# ...`) report the allocation, first-touch, and registration
// times, the transparent-huge-page coverage, and the device-pointer mode.

#include <cuda_runtime.h>

#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>


#define CK(call) \
do \
{ \
    const cudaError_t err_ = (call); \
    if(err_ != cudaSuccess) \
    { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        std::exit(EXIT_FAILURE); \
    } \
} \
while(0)


namespace
{

constexpr int BLOCK = 256;
constexpr int LANES = 8;    // Independent loads in flight per thread, per iteration.
constexpr int WINDOWS = 4;  // Independent windows in flight per warp, per iteration.
constexpr double GIB = 1024.0 * 1024.0 * 1024.0;


__host__ __device__ inline uint64_t mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__device__ inline uint64_t pick(const uint64_t h, const uint64_t n) { return __umul64hi(h, n); }

struct alignas(32) Sector { uint4 a, b; };

__device__ inline uint64_t fold(const uint64_t v) { return v; }
__device__ inline uint64_t fold(const Sector v) { return (uint64_t(v.a.x) << 32 | v.a.y) ^ (uint64_t(v.b.z) << 32 | v.b.w); }


// Every thread issues `iters x LANES` independent random loads over `buf[0, n)`.
template <typename T_>
__global__ void thread_kernel(const T_* const __restrict__ buf, const uint64_t n, const uint64_t iters, const uint64_t seed,
                              uint64_t* const __restrict__ sink)
{
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t h[LANES];
    #pragma unroll
    for(int l = 0; l < LANES; ++l)
        h[l] = mix(seed ^ (tid * LANES + l));

    uint64_t acc = 0;
    for(uint64_t it = 0; it < iters; ++it)
    {
        T_ v[LANES];
        #pragma unroll
        for(int l = 0; l < LANES; ++l)
        {
            h[l] = mix(h[l]);
            v[l] = buf[pick(h[l], n)];
        }

        #pragma unroll
        for(int l = 0; l < LANES; ++l)
            acc ^= fold(v[l]);
    }

    sink[tid] = acc;
}


// Every warp reads `iters x WINDOWS` windows of `L_` consecutive 128-byte lines
// at random line-aligned offsets over `n_lines` lines of `buf`; each lane reads
// 4 bytes of each line, so that every line is one coalesced request.
template <int L_>
__global__ void warp_kernel(const uint32_t* const __restrict__ buf, const uint64_t n_lines, const uint64_t iters, const uint64_t seed,
                            uint64_t* const __restrict__ sink)
{
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31;
    uint64_t h = mix(seed ^ (tid >> 5));    // Uniform across the warp.

    uint64_t acc = 0;
    for(uint64_t it = 0; it < iters; ++it)
    {
        uint32_t v[WINDOWS][L_];
        #pragma unroll
        for(int w = 0; w < WINDOWS; ++w)
        {
            h = mix(h);
            const uint64_t line = pick(h, n_lines - L_ + 1);
            #pragma unroll
            for(int l = 0; l < L_; ++l)
                v[w][l] = buf[(line + l) * 32 + lane];
        }

        #pragma unroll
        for(int w = 0; w < WINDOWS; ++w)
            #pragma unroll
            for(int l = 0; l < L_; ++l)
                acc ^= v[w][l];
    }

    sink[tid] = acc;
}


template <typename F_>
double time_kernel(F_&& launch)
{
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    launch();
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return ms / 1e3;
}


template <typename F_>
std::pair<uint64_t, double> calibrated(F_&& launch, const uint64_t work0, const double target)
{
    uint64_t work = work0;
    double t = time_kernel([&] { launch(work); });
    while(t < target / 8)
    {
        work *= 8;
        t = time_kernel([&] { launch(work); });
    }

    work = std::max<uint64_t>(1, static_cast<uint64_t>(work * (target / t)));
    t = time_kernel([&] { launch(work); });
    return {work, t};
}


double seconds_since(const std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


// Returns the number of CPUs in the affinity mask.
unsigned cpu_count()
{
    cpu_set_t s;
    if(sched_getaffinity(0, sizeof(s), &s) == 0)
        return std::max(1, CPU_COUNT(&s));

    return std::max(1u, std::thread::hardware_concurrency());
}


// Fills `buf[0, bytes / 8)` with random words from all CPUs in parallel, which
// also faults the pages in from the threads' NUMA policy.
void touch(uint64_t* const buf, const std::size_t bytes)
{
    const std::size_t n = bytes / 8;
    const unsigned t_c = cpu_count();
    std::vector<std::thread> th;
    for(unsigned t = 0; t < t_c; ++t)
        th.emplace_back([=]
        {
            const std::size_t b = n * t / t_c, e = n * (t + 1) / t_c;
            for(std::size_t i = b; i < e; ++i)
                buf[i] = mix(i);
        });

    for(auto& t : th)
        t.join();
}


// Returns `(RSS, AnonHugePages)` bytes of the mapping containing `p`, from
// `/proc/self/smaps`.
std::pair<std::size_t, std::size_t> thp_coverage(const void* const p)
{
    std::ifstream f("/proc/self/smaps");
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
        if(std::sscanf(line.c_str(), "Rss: %zu kB", &kb) == 1)  rss = kb * 1024;
        else if(std::sscanf(line.c_str(), "AnonHugePages: %zu kB", &kb) == 1)  thp = kb * 1024;
    }

    return {rss, thp};
}


void row(const std::string& alloc, const std::string& label, const char* const test, const double gib, const uint64_t bytes,
         const uint64_t threads, const uint64_t ops, const double s)
{
    std::printf("%s,%s,%s,%.3f,%llu,%llu,%llu,%.6f,%.3f,%.3f\n", alloc.c_str(), label.c_str(), test, gib,
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(threads), static_cast<unsigned long long>(ops),
                s, ops / s / 1e6, ops * double(bytes) / s / 1e9);
    std::fflush(stdout);
}


// Measures `warp_kernel<L_>` over the first `fp` bytes of `buf`, and reports it
// as `name`.
template <int L_>
void run_warp(const char* const name, const uint64_t* const buf, const std::size_t fp, const int grid, uint64_t* const sink,
              const double target, uint64_t& seed, const std::string& alloc, const std::string& label)
{
    const uint64_t warps = uint64_t(grid) * BLOCK / 32;
    const auto [iters, s] = calibrated([&](const uint64_t w)
        { warp_kernel<L_><<<grid, BLOCK>>>(reinterpret_cast<const uint32_t*>(buf), fp / 128, w, ++seed, sink); }, 4, target);
    row(alloc, label, name, fp / GIB, 128 * L_, warps, warps * iters * WINDOWS, s);
}

}


int main(int argc, char** argv)
{
    std::string alloc = "register", label = "-", gib_list = "16,64";
    double target = 0.5;
    for(int i = 1; i < argc; ++i)
    {
        const std::string k = argv[i];
        if(k == "--alloc" && i + 1 < argc)          alloc = argv[++i];
        else if(k == "--gib" && i + 1 < argc)       gib_list = argv[++i];
        else if(k == "--seconds" && i + 1 < argc)   target = std::atof(argv[++i]);
        else if(k == "--label" && i + 1 < argc)     label = argv[++i];
        else    { std::fprintf(stderr, "usage: pinned_host --alloc {hostalloc,register} --gib F1,F2,... [--seconds S] [--label L]\n"); return 2; }
    }

    if(alloc != "hostalloc" && alloc != "register") { std::fprintf(stderr, "unknown --alloc %s\n", alloc.c_str()); return 2; }

    std::vector<double> gibs;
    for(std::size_t b = 0; b <= gib_list.size(); )
    {
        const auto e = gib_list.find(',', b);
        gibs.push_back(std::atof(gib_list.substr(b, (e == std::string::npos ? gib_list.size() : e) - b).c_str()));
        if(e == std::string::npos)  break;
        b = e + 1;
    }

    const double max_gib = *std::max_element(gibs.begin(), gibs.end());
    const std::size_t bytes = (static_cast<std::size_t>(max_gib * GIB) >> 21) << 21;   // A multiple of 2 MiB.

    int dev = 0, can_use_host_ptr = 0, ro_supported = 0;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    CK(cudaDeviceGetAttribute(&can_use_host_ptr, cudaDevAttrCanUseHostPointerForRegisteredMem, dev));
    CK(cudaDeviceGetAttribute(&ro_supported, cudaDevAttrHostRegisterReadOnlySupported, dev));
    CK(cudaFree(nullptr));  // Create the context before timing allocations.
    std::printf("# device: %s; can-use-host-pointer %d; register-read-only %d; cpus %u\n", p.name, can_use_host_ptr, ro_supported, cpu_count());

    // Allocate, touch, and pin the buffer.
    uint64_t* host = nullptr;
    const uint64_t* dev_ptr = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    if(alloc == "hostalloc")
    {
        CK(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        const double t_alloc = seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        touch(host, bytes);
        std::printf("# setup: hostalloc %.1f GiB in %.2f s; first-touch %.2f s\n", bytes / GIB, t_alloc, seconds_since(t0));
        void* d = nullptr;
        CK(cudaHostGetDevicePointer(&d, host, 0));
        dev_ptr = static_cast<const uint64_t*>(d);
    }
    else
    {
        void* const m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if(m == MAP_FAILED) { std::fprintf(stderr, "mmap of %.1f GiB failed\n", bytes / GIB); return 1; }
        host = static_cast<uint64_t*>(m);
        madvise(m, bytes, MADV_HUGEPAGE);
        touch(host, bytes);
        const double t_touch = seconds_since(t0);
        const auto [rss, thp] = thp_coverage(host);

        // Chunks keep each call bounded; with host-pointer use, the chunks form one device range.
        const std::size_t chunk = can_use_host_ptr ? (std::size_t(16) << 30) : bytes;
        const unsigned flags = cudaHostRegisterMapped | (ro_supported ? cudaHostRegisterReadOnly : 0);
        t0 = std::chrono::steady_clock::now();
        for(std::size_t off = 0; off < bytes; off += chunk)
            CK(cudaHostRegister(reinterpret_cast<char*>(m) + off, std::min(chunk, bytes - off), flags));
        const double t_reg = seconds_since(t0);

        void* d = nullptr;
        if(can_use_host_ptr)
            d = m;
        else
            CK(cudaHostGetDevicePointer(&d, m, 0));
        dev_ptr = static_cast<const uint64_t*>(d);
        std::printf("# setup: mmap+touch %.1f GiB in %.2f s; THP %.1f%% of RSS %.1f GiB; register %.2f s (%.1f GiB/s) in %.0f GiB chunks, flags 0x%x\n",
                    bytes / GIB, t_touch, rss ? 100.0 * thp / rss : 0.0, rss / GIB, t_reg, bytes / GIB / t_reg, chunk / GIB, flags);
    }

    int bps = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, thread_kernel<Sector>, BLOCK, 0));
    const int grid = p.multiProcessorCount * std::max(bps, 1);
    const uint64_t threads = uint64_t(grid) * BLOCK;
    uint64_t* sink = nullptr;
    CK(cudaMalloc(&sink, threads * sizeof(uint64_t)));

    std::printf("alloc,label,test,footprint_gib,access_bytes,threads,ops,seconds,mops,payload_gbps\n");

    uint64_t seed = 1;
    for(const double gib : gibs)
    {
        const std::size_t fp = std::min(bytes, (static_cast<std::size_t>(gib * GIB) >> 12) << 12);
        const double fg = fp / GIB;

        {
            const auto [iters, s] = calibrated([&](const uint64_t w)
                { thread_kernel<uint64_t><<<grid, BLOCK>>>(dev_ptr, fp / 8, w, ++seed, sink); }, 4, target);
            row(alloc, label, "thread_8", fg, 8, threads, threads * iters * LANES, s);
        }

        {
            const auto [iters, s] = calibrated([&](const uint64_t w)
                { thread_kernel<Sector><<<grid, BLOCK>>>(reinterpret_cast<const Sector*>(dev_ptr), fp / 32, w, ++seed, sink); }, 4, target);
            row(alloc, label, "thread_32", fg, 32, threads, threads * iters * LANES, s);
        }

        run_warp<1>("warp_128", dev_ptr, fp, grid, sink, target, seed, alloc, label);
        run_warp<2>("warp_256", dev_ptr, fp, grid, sink, target, seed, alloc, label);
        run_warp<3>("warp_384", dev_ptr, fp, grid, sink, target, seed, alloc, label);

        if(fp >= (std::size_t(1) << 30))
        {
            const std::size_t chunk = std::size_t(1) << 30, span = std::min(fp, std::size_t(16) << 30);
            void* dst = nullptr;
            CK(cudaMalloc(&dst, chunk));
            const double s = time_kernel([&]
            {
                for(std::size_t off = 0; off + chunk <= span; off += chunk)
                    CK(cudaMemcpyAsync(dst, reinterpret_cast<const char*>(host) + off, chunk, cudaMemcpyHostToDevice));
            });
            row(alloc, label, "memcpy_h2d", fg, chunk, 1, span / chunk, s);
            CK(cudaFree(dst));
        }
    }

    CK(cudaFree(sink));
    if(alloc == "hostalloc")
        CK(cudaFreeHost(host));
    else
    {
        const std::size_t chunk = can_use_host_ptr ? (std::size_t(16) << 30) : bytes;
        for(std::size_t off = 0; off < bytes; off += chunk)
            CK(cudaHostUnregister(reinterpret_cast<char*>(host) + off));
        munmap(host, bytes);
    }

    return 0;
}
