
// Random-access rates of the device memory, against the footprint
// accessed. Feeds the query engine's cost model: phase-1 table probes, phase-2
// suffix-array and text probes, and the device TLB's reach.
//
// Usage: gpu_memory [--gib F1,F2,...] [--seconds S]
//   --gib      footprints in GiB, or `max` for the whole free memory;
//              default: 0.125,1,4,16,32,48,64,80,max
//   --seconds  target duration of each measurement; default: 0.5
//
// CSV to stdout, one row per (test, footprint):
//   test,footprint_gib,access_bytes,threads,ops,seconds,mops,payload_gbps,ns_per_op
// Tests:
//   indep_<B>  every thread issues independent random `B`-byte loads; `ops`
//              counts loads.
//   chain      every thread follows a chain of dependent random 8-byte loads:
//              the next address derives from the loaded value. `ns_per_op` is
//              the latency of a hop under full load.
//   chain_1    the same with one thread: the unloaded latency of a hop.
//   probe      every thread follows a chain of phase-2 probes: a dependent 8-byte
//              load (a suffix-array entry) and then a dependent 16-byte
//              unaligned load (a text key); `ops` counts probes.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

constexpr int BLOCK = 256;  // Threads per block.
constexpr int LANES = 8;    // Independent loads in flight per thread, per iteration.
constexpr double GIB = 1024.0 * 1024.0 * 1024.0;


// Returns the splitmix64 finalization of `x`: a well-mixed 64-bit word.
__host__ __device__ inline uint64_t mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}


// Returns an index in `[0, n)` derived from the random word `h`.
__device__ inline uint64_t pick(const uint64_t h, const uint64_t n)
{
    return __umul64hi(h, n);
}


// A 32-byte sector.
struct alignas(32) Sector
{
    uint4 a, b;
};

__device__ inline uint64_t fold(const uint32_t v) { return v; }
__device__ inline uint64_t fold(const uint64_t v) { return v; }
__device__ inline uint64_t fold(const uint4 v) { return ((uint64_t(v.x) << 32) | v.y) ^ ((uint64_t(v.z) << 32) | v.w); }
__device__ inline uint64_t fold(const Sector v) { return fold(v.a) ^ fold(v.b); }

template <typename T_> __device__ inline T_ load(const T_* const p) { return __ldg(p); }
template <> __device__ inline Sector load<Sector>(const Sector* const p) { Sector s; s.a = __ldg(&p->a); s.b = __ldg(&p->b); return s; }


// Fills `buf[0, n)` with random words.
__global__ void fill_kernel(uint64_t* const buf, const uint64_t n)
{
    for(uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x)
        buf[i] = mix(i);
}


// Every thread issues `iters x LANES` independent random loads over `buf[0, n)`.
template <typename T_>
__global__ void indep_kernel(const T_* const __restrict__ buf, const uint64_t n, const uint64_t iters, const uint64_t seed,
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
            v[l] = load(buf + pick(h[l], n));
        }

        #pragma unroll
        for(int l = 0; l < LANES; ++l)
            acc ^= fold(v[l]);
    }

    sink[tid] = acc;
}


// Every thread follows a chain of `steps` dependent random loads over `buf[0, n)`.
__global__ void chain_kernel(const uint64_t* const __restrict__ buf, const uint64_t n, const uint64_t steps, const uint64_t seed,
                             uint64_t* const __restrict__ sink)
{
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t x = mix(seed ^ tid);
    for(uint64_t s = 0; s < steps; ++s)
        x = mix(x ^ __ldg(buf + pick(x, n)));

    sink[tid] = x;
}


// Every thread follows a chain of `steps` phase-2 probes over the 32-bit words
// `buf[0, n)`: an 8-byte load at a 4-byte-aligned offset, as the 40-bit
// suffix-array decode does, then a dependent 5-word load at an offset derived
// from it, as the unaligned 16-byte text-key load does.
__global__ void probe_kernel(const uint32_t* const __restrict__ buf, const uint64_t n, const uint64_t steps, const uint64_t seed,
                             uint64_t* const __restrict__ sink)
{
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t x = mix(seed ^ tid);
    for(uint64_t s = 0; s < steps; ++s)
    {
        const uint64_t a = pick(x, n - 1);
        const uint64_t e = (uint64_t(__ldg(buf + a + 1)) << 32) | __ldg(buf + a);

        const uint64_t b = pick(mix(e), n - 5);
        uint64_t k = 0;
        #pragma unroll
        for(int i = 0; i < 5; ++i)
            k = (k << 13) ^ __ldg(buf + b + i);

        x = mix(x ^ e ^ k);
    }

    sink[tid] = x;
}


// Returns the seconds taken by `launch()`, timed with events.
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


// Runs `launch(work)` with a work-amount calibrated so that it takes about
// `target` seconds, and returns `(work, seconds)` of the calibrated run.
template <typename F_>
std::pair<uint64_t, double> calibrated(F_&& launch, const uint64_t work0, const double target)
{
    uint64_t work = work0;
    double t = time_kernel([&] { launch(work); });
    while(t < target / 8)   // Too short to extrapolate from reliably.
    {
        work *= 8;
        t = time_kernel([&] { launch(work); });
    }

    work = std::max<uint64_t>(1, static_cast<uint64_t>(work * (target / t)));
    t = time_kernel([&] { launch(work); });
    return {work, t};
}


void row(const char* const test, const double gib, const int bytes, const uint64_t threads, const uint64_t ops, const double s,
         const double ns_per_op)
{
    std::printf("%s,%.3f,%d,%llu,%llu,%.6f,%.2f,%.2f,%.2f\n", test, gib, bytes, static_cast<unsigned long long>(threads),
                static_cast<unsigned long long>(ops), s, ops / s / 1e6, ops * double(bytes) / s / 1e9, ns_per_op);
    std::fflush(stdout);
}


// Measures `indep_kernel<T_>` over the first `bytes` bytes of `buf`, and reports
// it as `name`.
template <typename T_>
void run_indep(const char* const name, const uint64_t* const buf, const std::size_t bytes, const int grid, uint64_t* const sink,
               const double target, uint64_t& seed)
{
    const uint64_t n = bytes / sizeof(T_);
    const uint64_t threads = uint64_t(grid) * BLOCK;
    const auto [iters, s] = calibrated([&](const uint64_t w)
        { indep_kernel<T_><<<grid, BLOCK>>>(reinterpret_cast<const T_*>(buf), n, w, ++seed, sink); }, 16, target);
    const uint64_t ops = threads * iters * LANES;
    row(name, bytes / GIB, sizeof(T_), threads, ops, s, s * 1e9 / ops);
}

}


int main(int argc, char** argv)
{
    std::string gib_list = "0.125,1,4,16,32,48,64,80,max";
    double target = 0.5;
    for(int i = 1; i < argc; ++i)
    {
        const std::string k = argv[i];
        if(k == "--gib" && i + 1 < argc)            gib_list = argv[++i];
        else if(k == "--seconds" && i + 1 < argc)   target = std::atof(argv[++i]);
        else    { std::fprintf(stderr, "usage: gpu_memory [--gib F1,F2,...|max] [--seconds S]\n"); return 2; }
    }

    int dev = 0;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    std::size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("# device: %s, sm_%d%d, %d SMs, %.1f GiB total, %.1f GiB free, L2 %.1f MiB, %d-bit bus, ECC %s\n",
                p.name, p.major, p.minor, p.multiProcessorCount, total_b / GIB, free_b / GIB, p.l2CacheSize / (1024.0 * 1024.0),
                p.memoryBusWidth, p.ECCEnabled ? "on" : "off");

    // One buffer as large as the memory allows; footprints are its prefixes.
    const std::size_t reserve = std::size_t(3) << 29;   // 1.5 GiB for the context, the sinks, and slack.
    const std::size_t max_b = ((free_b - reserve) / (std::size_t(64) << 20)) * (std::size_t(64) << 20);
    uint64_t* buf = nullptr;
    CK(cudaMalloc(&buf, max_b));
    fill_kernel<<<p.multiProcessorCount * 8, BLOCK>>>(buf, max_b / 8);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    int bps = 0;    // Resident blocks per SM; the kernels are register-light.
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, indep_kernel<Sector>, BLOCK, 0));
    const int grid = p.multiProcessorCount * std::max(bps, 1);
    const uint64_t threads = uint64_t(grid) * BLOCK;
    uint64_t* sink = nullptr;
    CK(cudaMalloc(&sink, threads * sizeof(uint64_t)));

    std::printf("test,footprint_gib,access_bytes,threads,ops,seconds,mops,payload_gbps,ns_per_op\n");

    std::vector<double> gibs;
    for(std::size_t b = 0; b <= gib_list.size(); )
    {
        const auto e = gib_list.find(',', b);
        const std::string f = gib_list.substr(b, (e == std::string::npos ? gib_list.size() : e) - b);
        gibs.push_back(f == "max" ? max_b / GIB : std::atof(f.c_str()));
        if(e == std::string::npos)  break;
        b = e + 1;
    }

    uint64_t seed = 1;
    for(const double gib : gibs)
    {
        const std::size_t bytes = std::min<std::size_t>(max_b, static_cast<std::size_t>(gib * GIB));
        if(bytes < (std::size_t(1) << 20))
            continue;

        const double fp = bytes / GIB;
        run_indep<uint32_t>("indep_4", buf, bytes, grid, sink, target, seed);
        run_indep<uint64_t>("indep_8", buf, bytes, grid, sink, target, seed);
        run_indep<uint4>("indep_16", buf, bytes, grid, sink, target, seed);
        run_indep<Sector>("indep_32", buf, bytes, grid, sink, target, seed);

        {
            const auto [steps, s] = calibrated([&](const uint64_t w) { chain_kernel<<<grid, BLOCK>>>(buf, bytes / 8, w, ++seed, sink); }, 64, target);
            row("chain", fp, 8, threads, threads * steps, s, s * 1e9 / steps);
        }

        {
            const auto [steps, s] = calibrated([&](const uint64_t w) { chain_kernel<<<1, 1>>>(buf, bytes / 8, w, ++seed, sink); }, 1024, target / 2);
            row("chain_1", fp, 8, 1, steps, s, s * 1e9 / steps);
        }

        {
            const auto [steps, s] = calibrated([&](const uint64_t w)
                { probe_kernel<<<grid, BLOCK>>>(reinterpret_cast<const uint32_t*>(buf), bytes / 4, w, ++seed, sink); }, 32, target);
            row("probe", fp, 28, threads, threads * steps, s, s * 1e9 / steps);
        }
    }

    CK(cudaFree(sink));
    CK(cudaFree(buf));
    return 0;
}
