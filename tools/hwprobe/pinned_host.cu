
// Zero-copy access rates of pinned host memory from the device. Feeds
// the host tier's cost model: a suffix array in pinned host memory, read per
// probe or per window, and a text split across the device and the host.
//
// Usage: pinned_host --alloc {hostalloc,register,vmm} --gib F1,F2,... [--tests G1,G2,...] [--numa N] [--seconds S] [--label L]
//   --alloc    hostalloc: `cudaHostAlloc(Mapped)`. register: anonymous `mmap`
//              with transparent huge pages, touched in parallel, then
//              `cudaHostRegister(Mapped | ReadOnly)` in 16 GiB chunks. vmm:
//              `cuMemCreate` of pinned memory on host NUMA node `--numa`
//              (default 0), mapped into the device's address space at the
//              recommended granularity, and filled from the device.
//   --gib      footprints in GiB; the buffer is allocated at the largest.
//   --tests    test groups to run at every footprint, or `all`; default: basic
//   --label    tag for the rows, e.g. the NUMA placement of the buffer.
//
// CSV to stdout, one row per (test, parameter, footprint):
//   alloc,label,test,footprint_gib,access_bytes,threads,ops,seconds,mops,payload_gbps,param,ns_per_op
// `ns_per_op` is the time per op of one thread, or of one warp for the warp
// tests; for the dependent chains, it is the latency of a hop.
// Test groups:
//   basic   thread_<B>  every thread issues independent random `B`-byte loads: per-probe
//                       suffix-array reads.
//           warp_<B>    every warp reads `B / 128` consecutive 128-byte lines at a random
//                       128-byte-aligned offset, coalesced: window gathers.
//           memcpy_h2d  sequential `cudaMemcpy` from the buffer, in 1 GiB chunks.
//   chain   chain_8     `T` threads (param) each follow a chain of dependent random 8-byte
//                       loads: the latency of a request, and the number of requests in
//                       flight at which the rate saturates.
//   reuse   reuse_128   every warp keeps 16 line reads in flight: `16 / K` random regions
//                       of `R` bytes (param `R:K`), with `K` lines in each, spaced `R / K`
//                       apart. `K > 1` reuses an address translation without making the
//                       lines adjacent, unless `R = 128 K`.
//   sorted  sorted_<B>  `N` windows (param) of `B` bytes at sorted, stratified random
//                       offsets, gathered in order by consecutive warps: the window
//                       gathers of a batch sorted by key.
//   seq     seq_128     warps read consecutive lines, coalesced: the ceiling of the path.
//   pages   warp_<B>    as in `basic`, for 512-byte to 4 KiB windows.
// Setup lines (`# ...`) report the allocation, first-touch, and registration
// times, the transparent-huge-page coverage, and the device-pointer mode.

#include <cuda.h>
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

#define CKD(call) \
do \
{ \
    const CUresult res_ = (call); \
    if(res_ != CUDA_SUCCESS) \
    { \
        const char* msg_ = nullptr; \
        cuGetErrorString(res_, &msg_); \
        std::fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__, __LINE__, msg_ ? msg_ : "unknown"); \
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


// Fills `buf[0, n)` with random words from the device.
__global__ void fill_kernel(uint64_t* const __restrict__ buf, const uint64_t n)
{
    for(uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x)
        buf[i] = mix(i);
}


// Every one of the first `active` threads follows a chain of `hops` dependent
// random 8-byte loads over `buf[0, n)`: the next address derives from the loaded
// value, so that a thread has one request in flight at a time.
__global__ void chain_kernel(const uint64_t* const __restrict__ buf, const uint64_t n, const uint64_t hops, const uint64_t seed,
                             const uint64_t active, uint64_t* const __restrict__ sink)
{
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid >= active)
        return;

    uint64_t h = mix(seed ^ tid), v = 0;
    for(uint64_t k = 0; k < hops; ++k)
    {
        h = mix(h ^ v);
        v = buf[pick(h, n)];
    }

    sink[tid] = v;
}


// Every warp reads `iters` rounds of 16 lines over `n_lines` lines of `buf`: in
// each round, `16 / K_` random regions of `region` lines, with `K_` lines in each
// region, spaced `region / K_` lines apart from a random phase.
template <int K_>
__global__ void reuse_kernel(const uint32_t* const __restrict__ buf, const uint64_t n_lines, const uint64_t region, const uint64_t iters,
                             const uint64_t seed, uint64_t* const __restrict__ sink)
{
    constexpr int G = 16 / K_;
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31;
    const uint64_t regions = n_lines / region, gap = region / K_;
    uint64_t h = mix(seed ^ (tid >> 5));    // Uniform across the warp.

    uint64_t acc = 0;
    for(uint64_t it = 0; it < iters; ++it)
    {
        uint32_t v[16];
        #pragma unroll
        for(int g = 0; g < G; ++g)
        {
            h = mix(h);
            const uint64_t first = pick(h, regions) * region + pick(mix(h), gap);
            #pragma unroll
            for(int k = 0; k < K_; ++k)
                v[g * K_ + k] = buf[(first + k * gap) * 32 + lane];
        }

        #pragma unroll
        for(int i = 0; i < 16; ++i)
            acc ^= v[i];
    }

    sink[tid] = acc;
}


// Gathers `passes` times `n_win` windows of `L_` lines over `n_lines` lines of
// `buf`, window `j` at the stratified random line offset `(j + u_j) (n_lines - L_ + 1) / n_win`,
// with `u_j` in [0, 1) drawn afresh every pass. The windows go in order: warp
// `w` of `W` takes windows `w, w + W, w + 2W, ...`, `WINDOWS` at a time.
template <int L_>
__global__ void sorted_kernel(const uint32_t* const __restrict__ buf, const uint64_t n_lines, const uint64_t n_win, const uint64_t passes,
                              const uint64_t seed, uint64_t* const __restrict__ sink)
{
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t warp = tid >> 5, warps = (uint64_t(gridDim.x) * blockDim.x) >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint64_t last = n_lines - L_;
    const double step = double(last + 1) / n_win;

    uint64_t acc = 0;
    for(uint64_t pass = 0; pass < passes; ++pass)
        for(uint64_t i = warp; i < n_win; i += WINDOWS * warps)
        {
            uint32_t v[WINDOWS][L_];
            #pragma unroll
            for(int w = 0; w < WINDOWS; ++w)
            {
                const uint64_t j = (i + w * warps < n_win ? i + w * warps : n_win - 1);
                const uint64_t x = static_cast<uint64_t>((j + (mix((seed + pass) ^ j) >> 11) * 0x1p-53) * step);
                const uint64_t line = (x < last ? x : last);
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


// Warps read the `n_lines` lines of `buf` in order, `passes` times: warp `w` of
// `W` reads lines `w, w + W, w + 2W, ...`, `WINDOWS` at a time.
__global__ void seq_kernel(const uint32_t* const __restrict__ buf, const uint64_t n_lines, const uint64_t passes,
                           uint64_t* const __restrict__ sink)
{
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t warp = tid >> 5, warps = (uint64_t(gridDim.x) * blockDim.x) >> 5;
    const uint32_t lane = threadIdx.x & 31;

    uint64_t acc = 0;
    for(uint64_t pass = 0; pass < passes; ++pass)
        for(uint64_t i = warp; i < n_lines; i += WINDOWS * warps)
        {
            uint32_t v[WINDOWS];
            #pragma unroll
            for(int w = 0; w < WINDOWS; ++w)
                v[w] = buf[(i + w * warps < n_lines ? i + w * warps : n_lines - 1) * 32 + lane];

            #pragma unroll
            for(int w = 0; w < WINDOWS; ++w)
                acc ^= v[w];
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
         const uint64_t threads, const uint64_t ops, const double s, const std::string& param = "-")
{
    std::printf("%s,%s,%s,%.4f,%llu,%llu,%llu,%.6f,%.3f,%.3f,%s,%.1f\n", alloc.c_str(), label.c_str(), test, gib,
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(threads), static_cast<unsigned long long>(ops),
                s, ops / s / 1e6, ops * double(bytes) / s / 1e9, param.c_str(), s * threads / ops * 1e9);
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


// Measures `reuse_kernel<K_>` with regions of `r_bytes` over the first `fp`
// bytes of `buf`.
template <int K_>
void run_reuse(const uint64_t r_bytes, const uint64_t* const buf, const std::size_t fp, const int grid, uint64_t* const sink,
               const double target, uint64_t& seed, const std::string& alloc, const std::string& label)
{
    const uint64_t warps = uint64_t(grid) * BLOCK / 32;
    const auto [iters, s] = calibrated([&](const uint64_t w)
        { reuse_kernel<K_><<<grid, BLOCK>>>(reinterpret_cast<const uint32_t*>(buf), fp / 128, r_bytes / 128, w, ++seed, sink); }, 4, target);
    row(alloc, label, "reuse_128", fp / GIB, 128, warps, warps * iters * 16, s, "R=" + std::to_string(r_bytes) + ":K=" + std::to_string(K_));
}


// Measures `sorted_kernel<L_>` with `n_win` windows over the first `fp` bytes of
// `buf`, and reports it as `name`.
template <int L_>
void run_sorted(const char* const name, const uint64_t n_win, const uint64_t* const buf, const std::size_t fp, const int grid,
                uint64_t* const sink, const double target, uint64_t& seed, const std::string& alloc, const std::string& label)
{
    const uint64_t warps = uint64_t(grid) * BLOCK / 32;
    const auto [passes, s] = calibrated([&](const uint64_t w)
        { sorted_kernel<L_><<<grid, BLOCK>>>(reinterpret_cast<const uint32_t*>(buf), fp / 128, n_win, w, seed += w, sink); }, 1, target);
    row(alloc, label, name, fp / GIB, 128 * L_, warps, passes * n_win, s, "N=" + std::to_string(n_win));
}

}


int main(int argc, char** argv)
{
    std::string alloc = "register", label = "-", gib_list = "16,64", tests = "basic";
    double target = 0.5;
    int numa = 0;
    for(int i = 1; i < argc; ++i)
    {
        const std::string k = argv[i];
        if(k == "--alloc" && i + 1 < argc)          alloc = argv[++i];
        else if(k == "--gib" && i + 1 < argc)       gib_list = argv[++i];
        else if(k == "--tests" && i + 1 < argc)     tests = argv[++i];
        else if(k == "--numa" && i + 1 < argc)      numa = std::atoi(argv[++i]);
        else if(k == "--seconds" && i + 1 < argc)   target = std::atof(argv[++i]);
        else if(k == "--label" && i + 1 < argc)     label = argv[++i];
        else
        {
            std::fprintf(stderr, "usage: pinned_host --alloc {hostalloc,register,vmm} --gib F1,F2,... [--tests G1,G2,...] [--numa N] [--seconds S] [--label L]\n"
                                 "  test groups: basic, chain, reuse, sorted, seq, pages, or all\n");
            return 2;
        }
    }

    if(alloc != "hostalloc" && alloc != "register" && alloc != "vmm")   { std::fprintf(stderr, "unknown --alloc %s\n", alloc.c_str()); return 2; }
    const auto want = [&](const char* const g)
        { return tests == "all" || ("," + tests + ",").find(std::string(",") + g + ",") != std::string::npos; };

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
    CUmemGenericAllocationHandle vmm_handle = 0;
    CUdeviceptr vmm_va = 0;
    std::size_t vmm_bytes = 0;
    auto t0 = std::chrono::steady_clock::now();
    if(alloc == "vmm")
    {
        CKD(cuInit(0));
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = numa;
        std::size_t gran = 0;
        CKD(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        vmm_bytes = (bytes + gran - 1) / gran * gran;
        CKD(cuMemCreate(&vmm_handle, vmm_bytes, &prop, 0));
        CKD(cuMemAddressReserve(&vmm_va, vmm_bytes, gran, 0, 0));
        CKD(cuMemMap(vmm_va, vmm_bytes, 0, vmm_handle, 0));
        CUmemAccessDesc acc = {};
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id = dev;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CKD(cuMemSetAccess(vmm_va, vmm_bytes, &acc, 1));
        const double t_map = seconds_since(t0);

        host = reinterpret_cast<uint64_t*>(vmm_va);    // Device-accessible only.
        dev_ptr = host;
        t0 = std::chrono::steady_clock::now();
        fill_kernel<<<p.multiProcessorCount * 8, BLOCK>>>(host, bytes / 8);
        CK(cudaDeviceSynchronize());
        std::printf("# setup: vmm %.1f GiB on host NUMA node %d, granularity %zu B, created and mapped in %.2f s; device fill %.2f s\n",
                    vmm_bytes / GIB, numa, gran, t_map, seconds_since(t0));
    }
    else if(alloc == "hostalloc")
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

    std::printf("alloc,label,test,footprint_gib,access_bytes,threads,ops,seconds,mops,payload_gbps,param,ns_per_op\n");

    uint64_t seed = 1;
    for(const double gib : gibs)
    {
        const std::size_t fp = std::min(bytes, (static_cast<std::size_t>(gib * GIB) >> 12) << 12);
        const double fg = fp / GIB;

        if(want("basic"))
        {
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
                        CK(cudaMemcpyAsync(dst, reinterpret_cast<const char*>(host) + off, chunk, cudaMemcpyDefault));
                });
                row(alloc, label, "memcpy_h2d", fg, chunk, 1, span / chunk, s);
                CK(cudaFree(dst));
            }
        }

        // One warp per block while the warps fit 8 to an SM, so that small counts
        // spread over the SMs.
        if(want("chain"))
            for(const uint64_t t_c : {uint64_t(1), uint64_t(4), uint64_t(16), uint64_t(64), uint64_t(256), uint64_t(1024), uint64_t(4096),
                                      uint64_t(16384), uint64_t(65536), threads})
            {
                const int blk = (t_c <= uint64_t(p.multiProcessorCount) * 8 * 32 ? 32 : BLOCK);
                const int g = static_cast<int>((t_c + blk - 1) / blk);
                const auto [hops, s] = calibrated([&](const uint64_t w)
                    { chain_kernel<<<g, blk>>>(dev_ptr, fp / 8, w, ++seed, t_c, sink); }, 4, target);
                row(alloc, label, "chain_8", fg, 8, t_c, t_c * hops, s, "T=" + std::to_string(t_c));
            }

        if(want("reuse"))
        {
            run_reuse<1>(4096, dev_ptr, fp, grid, sink, target, seed, alloc, label);
            for(const uint64_t r : {uint64_t(4 * 128), uint64_t(4096), uint64_t(64) << 10, uint64_t(2) << 20})
                run_reuse<4>(r, dev_ptr, fp, grid, sink, target, seed, alloc, label);
            for(const uint64_t r : {uint64_t(16 * 128), uint64_t(4096), uint64_t(64) << 10, uint64_t(2) << 20})
                run_reuse<16>(r, dev_ptr, fp, grid, sink, target, seed, alloc, label);
        }

        // Densities up to one window per window-length of lines.
        if(want("sorted"))
            for(const uint64_t n_win : {uint64_t(100000), uint64_t(1000000), uint64_t(10000000), uint64_t(100000000)})
            {
                if(n_win <= fp / 128)
                    run_sorted<1>("sorted_128", n_win, dev_ptr, fp, grid, sink, target, seed, alloc, label);
                if(n_win * 3 <= fp / 128)
                    run_sorted<3>("sorted_384", n_win, dev_ptr, fp, grid, sink, target, seed, alloc, label);
            }

        if(want("seq"))
        {
            const uint64_t n_lines = std::min(fp, std::size_t(16) << 30) / 128, warps = uint64_t(grid) * BLOCK / 32;
            const auto [passes, s] = calibrated([&](const uint64_t w)
                { seq_kernel<<<grid, BLOCK>>>(reinterpret_cast<const uint32_t*>(dev_ptr), n_lines, w, sink); }, 1, target);
            row(alloc, label, "seq_128", fg, 128, warps, passes * n_lines, s);
        }

        if(want("pages"))
        {
            run_warp<4>("warp_512", dev_ptr, fp, grid, sink, target, seed, alloc, label);
            run_warp<8>("warp_1024", dev_ptr, fp, grid, sink, target, seed, alloc, label);
            run_warp<32>("warp_4096", dev_ptr, fp, grid, sink, target, seed, alloc, label);
        }
    }

    CK(cudaFree(sink));
    if(alloc == "vmm")
    {
        CKD(cuMemUnmap(vmm_va, vmm_bytes));
        CKD(cuMemAddressFree(vmm_va, vmm_bytes));
        CKD(cuMemRelease(vmm_handle));
    }
    else if(alloc == "hostalloc")
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
