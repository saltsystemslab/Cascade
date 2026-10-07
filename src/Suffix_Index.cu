
#include "Suffix_Index.hpp"
#include "CPU_Reference.hpp"
#include "env.hpp"
#include "sa40.hpp"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>


namespace gpu_saca
{

namespace
{

constexpr std::size_t IO_ALIGN = 4096;  // Alignment of the offsets, lengths, and buffers of `O_DIRECT` reads.
constexpr std::size_t IO_PIECE = std::size_t(8) << 20;  // Bytes per read request.
constexpr std::size_t STAGE_BYTES = std::size_t(1) << 30;   // Bytes per pinned staging buffer of device-bound reads.
constexpr std::size_t REGISTER_CHUNK = std::size_t(16) << 30;   // Bytes per `cudaHostRegister` call.
constexpr std::size_t HUGE_PAGE = std::size_t(2) << 20;
constexpr idx_t SPOT_MAX_CMP = 1000;    // Symbols compared at most per spot-checked pair.
constexpr uint64_t SPOT_SEED = 0x5EED;


// Returns the count of parallel readers per file: `GPU_SACA_QUERY_IO_THREADS`,
// by default 16.
unsigned io_threads()
{
    return static_cast<unsigned>(std::max<uint64_t>(1, env::get_u64("GPU_SACA_QUERY_IO_THREADS").value_or(16)));
}


double seconds_since(const std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


std::size_t round_up(const std::size_t x, const std::size_t a)
{
    return (x + a - 1) / a * a;
}


// Returns `true` iff the CUDA call's result `e` is a success; sets `err` to a
// reason mentioning `what` otherwise.
bool cuda_ok(const cudaError_t e, const char* const what, std::string& err)
{
    if(e == cudaSuccess)
        return true;

    err = std::string(what) + " failed: " + cudaGetErrorString(e);
    return false;
}


// Allocates `bytes` of anonymous host memory with transparent huge pages into
// `base`. Returns `false` with a reason in `err` on failure.
bool map_host(const std::size_t bytes, void*& base, std::string& err)
{
    void* const p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if(p == MAP_FAILED)
    {
        err = "cannot map " + std::to_string(bytes) + " bytes of host memory: " + std::strerror(errno);
        return false;
    }

    ::madvise(p, bytes, MADV_HUGEPAGE);
    base = p;
    return true;
}


// Registers the host memory `[base, base + bytes)` with the device, mapped, in
// `REGISTER_CHUNK`-sized pieces, counting the bytes registered in `registered`;
// read-only for the device if `read_only` and the device supports it. The pages
// must have been written before: a read-only registration of untouched
// anonymous pages can pin the shared zero page, which later writes then
// copy-on-write away from, out of the device's sight. Returns `false` with a
// reason in `err` on failure.
bool register_host(void* const base, const std::size_t bytes, const bool read_only, std::size_t& registered, std::string& err)
{
    int ro = 0;
    if(read_only && !cuda_ok(cudaDeviceGetAttribute(&ro, cudaDevAttrHostRegisterReadOnlySupported, 0), "querying read-only registration", err))
        return false;

    const unsigned flags = cudaHostRegisterMapped | (ro ? cudaHostRegisterReadOnly : 0);
    for(registered = 0; registered < bytes; )
    {
        const std::size_t sz = std::min(REGISTER_CHUNK, bytes - registered);
        if(!cuda_ok(cudaHostRegister(static_cast<uint8_t*>(base) + registered, sz, flags), "cudaHostRegister", err))
            return false;

        registered += sz;
    }

    return true;
}


// Returns the SplitMix64 hash of `x`.
__host__ __device__ inline uint64_t mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}


// Checks `samples` adjacent pairs `(r, r + 1)` of the suffix array `SA` over
// the text `T[0, n)`: every pair if `exhaustive`, else the first and the last
// pair and random ones. Lowers `bad` to the smallest rank `r` of a pair out of
// order or out of range.
template <typename T_sym_>
__global__ void spot_check_kernel(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const idx_t samples, const bool exhaustive,
                                  const uint64_t seed, unsigned long long* const bad)
{
    for(idx_t k = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; k < samples; k += idx_t(gridDim.x) * blockDim.x)
    {
        const idx_t r = (exhaustive ? k : (k == 0 ? 0 : (k == 1 ? n - 2 : mix(seed ^ k) % (n - 1))));
        const idx_t i = sa40::load(SA, r), j = sa40::load(SA, r + 1);
        if(i >= n || j >= n || i == j || ref::cmp_suffixes(T, n, i, j, SPOT_MAX_CMP) > 0)
            atomicMin(bad, static_cast<unsigned long long>(r));
    }
}


// Runs `spot_check_kernel` for symbols of `T_sym_`. Returns the smallest bad
// rank found, or `n` if none; sets `err` on a CUDA failure.
template <typename T_sym_>
idx_t spot_check_device(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const idx_t samples, const bool exhaustive,
                        std::string& err)
{
    unsigned long long* bad_d = nullptr;
    unsigned long long bad = n;
    if(!cuda_ok(cudaMalloc(&bad_d, sizeof(bad)), "cudaMalloc", err))
        return n;

    const bool ok = cuda_ok(cudaMemcpy(bad_d, &bad, sizeof(bad), cudaMemcpyHostToDevice), "cudaMemcpy", err);
    if(ok)
    {
        const int block = 256;
        const int grid = static_cast<int>(std::min<idx_t>((samples + block - 1) / block, 65536));
        spot_check_kernel<<<grid, block>>>(T, n, SA, samples, exhaustive, SPOT_SEED, bad_d);
        if(cuda_ok(cudaGetLastError(), "spot-check launch", err))
            cuda_ok(cudaMemcpy(&bad, bad_d, sizeof(bad), cudaMemcpyDeviceToHost), "spot-check", err);
    }

    cudaFree(bad_d);
    return static_cast<idx_t>(bad);
}


// Checks the pairs of `spot_check_kernel` on the host's cores, over host-side
// arrays. Returns the smallest bad rank found, or `n` if none.
template <typename T_sym_>
idx_t spot_check_host(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const idx_t samples, const bool exhaustive)
{
    idx_t bad = n;

    #pragma omp parallel for schedule(static) reduction(min : bad)
    for(idx_t k = 0; k < samples; ++k)
    {
        const idx_t r = (exhaustive ? k : (k == 0 ? 0 : (k == 1 ? n - 2 : mix(SPOT_SEED ^ k) % (n - 1))));
        const idx_t i = sa40::load(SA, r), j = sa40::load(SA, r + 1);
        if(i >= n || j >= n || i == j || ref::cmp_suffixes(T, n, i, j, SPOT_MAX_CMP) > 0)
            bad = std::min(bad, r);
    }

    return bad;
}

}


const char* residency_name(const Residency r)
{
    switch(r)
    {
    case Residency::device:     return "device";
    case Residency::pinned:     return "pinned";
    case Residency::pageable:   return "pageable";
    case Residency::mapped:     return "mapped";
    }

    return "?";
}


bool parse_residency(const std::string& name, Residency& r)
{
    if(name == "device")        r = Residency::device;
    else if(name == "pinned")   r = Residency::pinned;
    else if(name == "pageable") r = Residency::pageable;
    else if(name == "mapped")   r = Residency::mapped;
    else    return false;

    return true;
}


namespace index_io
{

bool read_range(const std::string& path, const std::size_t off, const std::size_t len, uint8_t* const dst, std::string& err)
{
    if(len == 0)
        return true;

    const bool aligned = (off % IO_ALIGN == 0 && reinterpret_cast<std::uintptr_t>(dst) % IO_ALIGN == 0);
    int fd = (aligned ? ::open(path.c_str(), O_RDONLY | O_DIRECT) : -1);
    const bool direct = (fd >= 0);
    if(!direct)
        fd = ::open(path.c_str(), O_RDONLY);
    if(fd < 0)
    {
        err = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }

    const std::size_t pieces = (len + IO_PIECE - 1) / IO_PIECE;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex m;

    const auto fail = [&](const std::string& why)
    {
        std::lock_guard<std::mutex> lock(m);
        if(!failed.exchange(true))
            err = why;
    };

    const auto reader = [&]
    {
        for(std::size_t p; !failed.load(std::memory_order_relaxed) && (p = next.fetch_add(1)) < pieces; )
        {
            const std::size_t b = p * IO_PIECE, want = std::min(len - b, IO_PIECE);
            // A direct request is a whole number of blocks; the file's end cuts the last one short.
            const std::size_t req = (direct ? round_up(want, IO_ALIGN) : want);
            std::size_t got = 0;
            while(got < want)
            {
                const ssize_t r = ::pread(fd, dst + b + got, req - got, static_cast<off_t>(off + b + got));
                if(r < 0 && errno == EINTR)
                    continue;

                if(r < 0)   { fail("read failed on " + path + ": " + std::strerror(errno)); return; }
                if(r == 0)  { fail("unexpected end of file " + path); return; }
                got += static_cast<std::size_t>(r);
            }
        }
    };

    const unsigned t_c = static_cast<unsigned>(std::min<std::size_t>(io_threads(), pieces));
    std::vector<std::thread> threads;
    for(unsigned t = 0; t < t_c; ++t)
        threads.emplace_back(reader);
    for(auto& t : threads)
        t.join();

    ::close(fd);
    return !failed;
}


bool scan_sa(const Index_Meta& m, const std::size_t align, const std::function<void(idx_t, const uint8_t*, std::size_t)>& fn, std::string& err)
{
    if(align == 0 || align % IO_ALIGN != 0)
    {
        err = "the suffix array streams in pieces of whole 4 KB blocks";
        return false;
    }

    // The pieces of every part, taken in turns across the parts: every drive reads at once.
    struct Piece { std::size_t part; idx_t off; std::size_t len; };
    const std::size_t part_c = m.sa_part.size();
    std::vector<std::vector<Piece>> of_part(part_c);
    for(std::size_t i = 0; i < part_c; ++i)
        for(idx_t b = m.sa_part_off[i], e = b + m.part_bytes(i); b < e; )
        {
            const idx_t x = std::min<idx_t>(e, (b / align + 1) * align);
            of_part[i].push_back(Piece{i, b, static_cast<std::size_t>(x - b)});
            b = x;
        }

    std::vector<Piece> piece;
    for(std::size_t k = 0, left = part_c; left > 0; ++k)
    {
        left = 0;
        for(const auto& v : of_part)
            if(k < v.size())
                piece.push_back(v[k]), ++left;
    }

    // Parts start at pages, and pieces end at whole blocks into them but at the file's end: reads may bypass the page cache.
    std::vector<int> fd(part_c, -1);
    std::vector<char> direct(part_c, 0);
    for(std::size_t i = 0; i < part_c; ++i)
    {
        fd[i] = ::open(m.sa_part[i].c_str(), O_RDONLY | O_DIRECT);
        direct[i] = (fd[i] >= 0);
        if(!direct[i])
            fd[i] = ::open(m.sa_part[i].c_str(), O_RDONLY);
        if(fd[i] < 0)
        {
            err = "cannot open " + m.sa_part[i] + ": " + std::strerror(errno);
            for(std::size_t j = 0; j < i; ++j)
                ::close(fd[j]);
            return false;
        }
    }

    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex mtx;
    const auto fail = [&](const std::string& why)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if(!failed.exchange(true))
            err = why;
    };

    const auto reader = [&]
    {
        const std::size_t cap = align + round_up(SA_PAD_BYTES, IO_ALIGN);
        void* buf = nullptr;
        if(::posix_memalign(&buf, IO_ALIGN, cap) != 0)
        {
            fail("cannot allocate a buffer to stream the suffix array");
            return;
        }

        auto* const p = static_cast<uint8_t*>(buf);
        for(std::size_t k; !failed.load(std::memory_order_relaxed) && (k = next.fetch_add(1)) < piece.size(); )
        {
            const Piece& pc = piece[k];
            const std::size_t at = pc.off - m.sa_part_off[pc.part], req = (direct[pc.part] ? round_up(pc.len, IO_ALIGN) : pc.len);
            const std::string& path = m.sa_part[pc.part];
            std::size_t got = 0;
            while(got < pc.len)
            {
                const ssize_t r = ::pread(fd[pc.part], p + got, req - got, static_cast<off_t>(at + got));
                if(r < 0 && errno == EINTR)
                    continue;

                if(r < 0)   { fail("read failed on " + path + ": " + std::strerror(errno)); break; }
                if(r == 0)  { fail("unexpected end of file " + path); break; }
                got += static_cast<std::size_t>(r);
            }

            if(got < pc.len)
                break;

            std::memset(p + pc.len, 0, SA_PAD_BYTES);
            fn(pc.off, p, pc.len);
        }

        std::free(buf);
    };

    const unsigned t_c = static_cast<unsigned>(std::min<std::size_t>(io_threads() * part_c, piece.size()));
    std::vector<std::thread> threads;
    for(unsigned t = 0; t < t_c; ++t)
        threads.emplace_back(reader);
    for(auto& t : threads)
        t.join();

    for(const int f : fd)
        ::close(f);

    return !failed;
}

}


Suffix_Index::Suffix_Index(Suffix_Index&& o) noexcept:
      meta_(std::move(o.meta_))
    , text_(std::exchange(o.text_, Array()))
    , sa_(std::exchange(o.sa_, Array()))
    , stats_(o.stats_)
{}


Suffix_Index& Suffix_Index::operator=(Suffix_Index&& o) noexcept
{
    if(this != &o)
    {
        release();
        meta_ = std::move(o.meta_);
        text_ = std::exchange(o.text_, Array());
        sa_ = std::exchange(o.sa_, Array());
        stats_ = o.stats_;
    }

    return *this;
}


bool Suffix_Index::load(const std::string& dir, const Residency text_res, const Residency sa_res, const idx_t spot, std::string& err)
{
    release();

    if(!sa40::host_is_little_endian())
    {
        err = "the query index requires a little-endian host";
        return false;
    }

    if(text_res == Residency::mapped)
    {
        err = "only the suffix array can be mapped; the text must be in memory";
        return false;
    }

    const auto host_only = [](const Residency r) { return r == Residency::pageable || r == Residency::mapped; };
    const bool on_device = (!host_only(text_res) && !host_only(sa_res));
    const bool on_host = (text_res != Residency::device && sa_res != Residency::device);
    if(!on_device && !on_host)
    {
        err = std::string("unsupported placement: text ") + residency_name(text_res) + ", suffix array " + residency_name(sa_res)
            + "; both arrays must be device-accessible or host-accessible";
        return false;
    }

    if(!index_meta::read(dir, meta_, err) || !index_meta::validate_files(meta_, err))
    {
        meta_ = Index_Meta();
        return false;
    }

    // Pinned staging buffers for the device-bound reads, no larger than the arrays need.
    uint8_t* stage[2] = {nullptr, nullptr};
    void* stage_base = nullptr;
    std::size_t stage_registered = 0;
    const bool staged = (text_res == Residency::device || sa_res == Residency::device);
    const std::size_t stage_bytes = std::min(STAGE_BYTES, round_up(std::max(meta_.n * meta_.sym_bytes, meta_.sa_bytes()), IO_ALIGN));
    bool ok = true;
    if(staged)
    {
        ok = map_host(2 * stage_bytes, stage_base, err);
        if(ok)
            std::memset(stage_base, 0, 2 * stage_bytes);   // Fault the pages in before pinning them; see `register_host`.
        ok = ok && register_host(stage_base, 2 * stage_bytes, false, stage_registered, err);
        stage[0] = static_cast<uint8_t*>(stage_base);
        stage[1] = stage[0] + stage_bytes;
    }

    if(ok)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const std::size_t text_bytes = meta_.n * meta_.sym_bytes;
        ok = place(text_, text_res, {Source{meta_.text, 0, text_bytes}}, text_bytes, TEXT_PAD_BYTES, stage, stage_bytes, err);
        stats_.t_text = seconds_since(t0);
    }

    if(ok)
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Source> src;
        for(std::size_t i = 0; i < meta_.sa_part.size(); ++i)
            src.push_back(Source{meta_.sa_part[i], meta_.sa_part_off[i], meta_.part_bytes(i)});
        ok = place(sa_, sa_res, src, meta_.sa_bytes(), SA_PAD_BYTES, stage, stage_bytes, err);
        stats_.t_sa = seconds_since(t0);
    }

    if(staged)
    {
        for(std::size_t off = 0; off < stage_registered; off += REGISTER_CHUNK)
            cudaHostUnregister(static_cast<uint8_t*>(stage_base) + off);
        if(stage_base != nullptr)
            ::munmap(stage_base, 2 * stage_bytes);
    }

    if(ok && spot > 0)
        ok = spot_check(spot, err);

    if(!ok)
        release();

    return ok;
}


bool Suffix_Index::place(Array& a, const Residency res, const std::vector<Source>& src, const std::size_t bytes, const std::size_t pad,
                         uint8_t* const (&stage)[2], const std::size_t stage_bytes, std::string& err)
{
    a.res = res;
    if(res == Residency::mapped)
        return map_files(a, src, bytes, pad, err);

    if(res == Residency::device)
    {
        a.alloc_bytes = bytes + pad;
        if(!cuda_ok(cudaMalloc(&a.base, a.alloc_bytes), "cudaMalloc", err))
        {
            a.base = nullptr;
            return false;
        }

        a.dev = a.base;
        auto* const dst = static_cast<uint8_t*>(a.base);
        if(!cuda_ok(cudaMemset(dst + bytes, 0, pad), "cudaMemset", err))
            return false;

        // Stream every file through the two staging buffers: reading a chunk overlaps copying the previous one.
        cudaStream_t s;
        cudaEvent_t copied[2];
        if(!cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate", err))
            return false;

        bool ok = cuda_ok(cudaEventCreateWithFlags(&copied[0], cudaEventDisableTiming), "cudaEventCreate", err)
               && cuda_ok(cudaEventCreateWithFlags(&copied[1], cudaEventDisableTiming), "cudaEventCreate", err)
               && cuda_ok(cudaEventRecord(copied[0], s), "cudaEventRecord", err)
               && cuda_ok(cudaEventRecord(copied[1], s), "cudaEventRecord", err);

        std::size_t k = 0;
        for(std::size_t f = 0; ok && f < src.size(); ++f)
            for(std::size_t pos = 0; ok && pos < src[f].bytes; pos += stage_bytes, ++k)
            {
                const std::size_t b = k & 1, sz = std::min(stage_bytes, src[f].bytes - pos);
                ok = cuda_ok(cudaEventSynchronize(copied[b]), "staging", err)
                  && index_io::read_range(src[f].path, pos, sz, stage[b], err)
                  && cuda_ok(cudaMemcpyAsync(dst + src[f].off + pos, stage[b], sz, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync", err)
                  && cuda_ok(cudaEventRecord(copied[b], s), "cudaEventRecord", err);
            }

        ok = cuda_ok(cudaStreamSynchronize(s), "copying to the device", err) && ok;
        cudaEventDestroy(copied[0]);
        cudaEventDestroy(copied[1]);
        cudaStreamDestroy(s);
        return ok;
    }

    // Host tiers: read straight into the final buffer, sized for whole-block direct reads.
    a.alloc_bytes = round_up(std::max(round_up(bytes, IO_ALIGN), bytes + pad), HUGE_PAGE);
    if(!map_host(a.alloc_bytes, a.base, err))
    {
        a.base = nullptr;
        return false;
    }

    auto* const dst = static_cast<uint8_t*>(a.base);
    a.host = a.base;
    for(const auto& f : src)
        if(!index_io::read_range(f.path, 0, f.bytes, dst + f.off, err))
            return false;

    std::memset(dst + bytes, 0, a.alloc_bytes - bytes);
    if(res == Residency::pageable)
        return true;

    int host_ptr = 0;
    if(!cuda_ok(cudaDeviceGetAttribute(&host_ptr, cudaDevAttrCanUseHostPointerForRegisteredMem, 0), "querying host-pointer use", err)
        || !register_host(a.base, a.alloc_bytes, true, a.registered_bytes, err))
        return false;

    if(host_ptr)
        a.dev = a.base;
    else if(a.alloc_bytes > REGISTER_CHUNK)
    {
        err = "the device cannot use host pointers, and the pinned array spans several registrations";
        return false;
    }
    else if(!cuda_ok(cudaHostGetDevicePointer(&a.dev, a.base, 0), "cudaHostGetDevicePointer", err))
        return false;

    return true;
}


bool Suffix_Index::map_files(Array& a, const std::vector<Source>& src, const std::size_t bytes, const std::size_t pad, std::string& err)
{
    // Room for the files' pages, and then for the padding in anonymous pages, which read as zeros, as do the last file page's bytes
    // past its end.
    const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    a.mapped_bytes = round_up(bytes, page);
    a.alloc_bytes = a.mapped_bytes + round_up(pad, page);
    void* const p = ::mmap(nullptr, a.alloc_bytes, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if(p == MAP_FAILED)
    {
        err = "cannot map " + std::to_string(a.alloc_bytes) + " bytes of addresses: " + std::strerror(errno);
        return false;
    }

    a.base = a.host = p;
    for(const auto& f : src)
    {
        if(f.off % page != 0)
        {
            err = "the suffix array's part " + f.path + " does not start at a page of " + std::to_string(page) + " bytes";
            return false;
        }

        const int fd = ::open(f.path.c_str(), O_RDONLY);
        if(fd < 0)
        {
            err = "cannot open " + f.path + ": " + std::strerror(errno);
            return false;
        }

        a.fds.push_back(fd);
        if(::mmap(static_cast<uint8_t*>(p) + f.off, f.bytes, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED)
        {
            err = "cannot map " + f.path + ": " + std::strerror(errno);
            return false;
        }
    }

    if(::madvise(p, a.mapped_bytes, MADV_RANDOM) != 0)
    {
        err = std::string("cannot advise the kernel of random reads of the suffix array: ") + std::strerror(errno);
        return false;
    }

    return true;
}


bool Suffix_Index::drop_cached_sa(std::string& err) const
{
    if(sa_.res != Residency::mapped || sa_.base == nullptr)
    {
        err = "only a mapped suffix array has pages in the page cache";
        return false;
    }

    // The page cache keeps the pages that a process maps: this one's mappings of them go first.
    if(::madvise(sa_.base, sa_.mapped_bytes, MADV_DONTNEED) != 0)
    {
        err = std::string("cannot unmap the suffix array's pages: ") + std::strerror(errno);
        return false;
    }

    for(const int fd : sa_.fds)
        if(const int e = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); e != 0)
        {
            err = std::string("cannot drop the suffix array's pages from the page cache: ") + std::strerror(e);
            return false;
        }

    return true;
}


bool Suffix_Index::spot_check(const idx_t spot, std::string& err)
{
    const idx_t n = meta_.n;
    if(n < 2)
        return true;

    const auto t0 = std::chrono::steady_clock::now();
    const bool exhaustive = (spot + 2 >= n - 1);
    const idx_t samples = (exhaustive ? n - 1 : spot + 2);
    std::string e;
    idx_t bad = n;
    if(device_accessible())
        switch(meta_.sym_bytes)
        {
        case 1: bad = spot_check_device(text<uint8_t>(), n, sa(), samples, exhaustive, e); break;
        case 2: bad = spot_check_device(text<uint16_t>(), n, sa(), samples, exhaustive, e); break;
        default:    bad = spot_check_device(text<uint32_t>(), n, sa(), samples, exhaustive, e);
        }
    else
        switch(meta_.sym_bytes)
        {
        case 1: bad = spot_check_host(host_text<uint8_t>(), n, host_sa(), samples, exhaustive); break;
        case 2: bad = spot_check_host(host_text<uint16_t>(), n, host_sa(), samples, exhaustive); break;
        default:    bad = spot_check_host(host_text<uint32_t>(), n, host_sa(), samples, exhaustive);
        }

    if(!e.empty())
    {
        err = e;
        return false;
    }

    if(bad < n)
    {
        err = "suffix-array order spot-check failed at ranks " + std::to_string(bad) + " and " + std::to_string(bad + 1)
            + "; expected numeric symbol order with shorter suffixes first";
        return false;
    }

    stats_.t_spot = seconds_since(t0);
    stats_.spot_pairs = samples;
    stats_.spot_on_device = device_accessible();
    return true;
}


void Suffix_Index::release()
{
    free_array(text_);
    free_array(sa_);
    meta_ = Index_Meta();
    stats_ = Index_Load_Stats();
}


void Suffix_Index::free_array(Array& a)
{
    if(a.base != nullptr)
    {
        if(a.res == Residency::device)
            cudaFree(a.base);
        else
        {
            for(std::size_t off = 0; off < a.registered_bytes; off += REGISTER_CHUNK)
                cudaHostUnregister(static_cast<uint8_t*>(a.base) + off);
            ::munmap(a.base, a.alloc_bytes);    // With the files' mappings in it, if mapped.
        }
    }

    for(const int fd : a.fds)
        ::close(fd);

    a = Array();
}

}
