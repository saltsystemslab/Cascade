
// Prepares query indexes from suffix arrays: converts the construction's
// `u64` suffix array into the query index's 40-bit format, optionally split
// into page-aligned parts across directories, validates it, and writes the
// index metadata.

#include "Index_Meta.hpp"
#include "CPU_Reference.hpp"
#include "sa40.hpp"
#include "types.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif


namespace
{

using namespace gpu_saca;


const char* const USAGE =
R"(Usage:
  sa_convert --in SA.u64 --text TEXT --sym-bytes {1,2,4} --index-dir DIR
             [--parts K --part-dirs D0:D1:...] [--check none|range|perm]
             [--spot N] [--sep S] [--source STR]
      Converts the uint64 suffix array SA.u64 of TEXT into 40-bit entries, into
      DIR/sa.u40 (or K page-aligned parts D_i/sa.u40.part<i>), validates it,
      and writes DIR/index.meta.
  sa_convert --meta-only --sa SA.u40 --text TEXT --sym-bytes B --index-dir DIR
             [--spot N] [--sep S] [--source STR]
      Validates an existing 40-bit suffix array and writes DIR/index.meta.
  sa_convert --widen --in SA.u40 --out SA.u64
      Widens a 40-bit suffix array into uint64 entries.
Options:
  --check   Entry validation during conversion; default: perm (range + distinctness).
  --spot    Random adjacent-pair order checks; default: 100000.
)";


// Command-line arguments.
struct Args
{
    std::string mode = "convert";   // `convert`, `meta-only`, or `widen`.
    std::string in, out, sa, text, index_dir, source;
    std::vector<std::string> part_dirs;
    uint32_t    sym_bytes = 0;
    idx_t       parts = 1;
    std::string check = "perm";
    idx_t       spot = 100000;
    int64_t     sep = -1;
};


// Parses the command-line into `a`. Returns `false` with a reason in `err` on
// failure.
bool parse(const int argc, char** const argv, Args& a, std::string& err)
{
    for(int i = 1; i < argc; ++i)
    {
        const std::string k = argv[i];
        const auto val = [&]() -> std::string
        {
            if(i + 1 >= argc)   throw std::invalid_argument("missing value");
            return argv[++i];
        };

        try
        {
        if(k == "--meta-only")  a.mode = "meta-only";
        else if(k == "--widen") a.mode = "widen";
        else if(k == "--in")    a.in = val();
        else if(k == "--out")   a.out = val();
        else if(k == "--sa")    a.sa = val();
        else if(k == "--text")  a.text = val();
        else if(k == "--index-dir") a.index_dir = val();
        else if(k == "--source")    a.source = val();
        else if(k == "--sym-bytes") a.sym_bytes = static_cast<uint32_t>(std::stoul(val()));
        else if(k == "--parts") a.parts = std::stoull(val());
        else if(k == "--check") a.check = val();
        else if(k == "--spot")  a.spot = std::stoull(val());
        else if(k == "--sep")   a.sep = std::stoll(val());
        else if(k == "--part-dirs")
        {
            const std::string s = val();
            for(std::size_t b = 0; b <= s.size(); )
            {
                const auto e = s.find(':', b);
                a.part_dirs.push_back(s.substr(b, (e == std::string::npos ? s.size() : e) - b));
                if(e == std::string::npos)  break;
                b = e + 1;
            }
        }
        else if(k == "-h" || k == "--help") { err = ""; return false; }
        else    { err = "unknown argument: " + k; return false; }
        }
        catch(const std::exception&)
        {
            err = "missing or invalid value for " + k;
            return false;
        }
    }

    if(a.mode == "widen")
    {
        if(a.in.empty() || a.out.empty())   { err = "--widen requires --in and --out"; return false; }
        return true;
    }

    if(a.text.empty() || a.index_dir.empty() || a.sym_bytes == 0)   { err = "--text, --index-dir, and --sym-bytes are required"; return false; }
    if(a.sym_bytes != 1 && a.sym_bytes != 2 && a.sym_bytes != 4)    { err = "--sym-bytes must be 1, 2, or 4"; return false; }
    if(a.mode == "convert" && a.in.empty()) { err = "conversion requires --in"; return false; }
    if(a.mode == "meta-only" && a.sa.empty())   { err = "--meta-only requires --sa"; return false; }
    if(a.parts == 0)    { err = "--parts must be positive"; return false; }
    if(a.parts > 1 && a.part_dirs.size() != a.parts)    { err = "--parts K requires K --part-dirs"; return false; }
    if(a.parts > 1 && a.mode != "convert")  { err = "--parts is supported only for conversion"; return false; }
    if(a.check != "none" && a.check != "range" && a.check != "perm")    { err = "--check must be none, range, or perm"; return false; }

    return true;
}


// Returns `true` iff `bytes` bytes could be read from the file `fd` at offset
// `off` into `buf`.
bool pread_full(const int fd, void* const buf, const std::size_t bytes, const idx_t off)
{
    std::size_t done = 0;
    while(done < bytes)
    {
        const auto r = ::pread(fd, static_cast<char*>(buf) + done, bytes - done, static_cast<off_t>(off + done));
        if(r < 0 && errno == EINTR) continue;
        if(r <= 0)  return false;
        done += static_cast<std::size_t>(r);
    }

    return true;
}


// Returns `true` iff `bytes` bytes from `buf` could be written into the file
// `fd` at offset `off`.
bool pwrite_full(const int fd, const void* const buf, const std::size_t bytes, const idx_t off)
{
    std::size_t done = 0;
    while(done < bytes)
    {
        const auto r = ::pwrite(fd, static_cast<const char*>(buf) + done, bytes - done, static_cast<off_t>(off + done));
        if(r < 0 && errno == EINTR) continue;
        if(r <= 0)  return false;
        done += static_cast<std::size_t>(r);
    }

    return true;
}


// Read-only memory-mapping of a file.
class Mapping
{
private:

    void* p_ = MAP_FAILED;
    std::size_t len_ = 0;

public:

    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& o) noexcept : p_(o.p_), len_(o.len_) { o.p_ = MAP_FAILED; o.len_ = 0; }
    ~Mapping() { if(p_ != MAP_FAILED)   ::munmap(p_, len_); }

    // Maps the file at `path`. Returns `false` on failure.
    bool map(const std::string& path)
    {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if(fd < 0)  return false;
        struct stat st;
        if(::fstat(fd, &st) != 0 || st.st_size == 0)    { ::close(fd); return false; }
        len_ = static_cast<std::size_t>(st.st_size);
        p_ = ::mmap(nullptr, len_, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if(p_ != MAP_FAILED)
            ::madvise(p_, len_, MADV_RANDOM);
        return p_ != MAP_FAILED;
    }

    const uint8_t* data() const { return static_cast<const uint8_t*>(p_); }
};


// Accessor of the entries of a 40-bit suffix array split into byte-range parts.
struct Parted_SA
{
    std::vector<const uint8_t*> part;   // Mapped parts.
    std::vector<idx_t> off;             // Byte-offsets of the parts.
    idx_t bytes;                        // Size of the suffix array's byte-stream.

    // Returns the entry at rank `r`; an entry may straddle two parts.
    idx_t operator()(const idx_t r) const
    {
        const idx_t b = SA_WIDTH * r;
        std::size_t i = std::upper_bound(off.begin(), off.end(), b) - off.begin() - 1;
        const idx_t part_end = (i + 1 < off.size() ? off[i + 1] : bytes);
        if(b + SA_WIDTH <= part_end)
            return sa40::load(part[i] + (b - off[i]), 0);

        uint8_t e[8] = {};
        for(idx_t k = 0; k < SA_WIDTH; ++k)
        {
            const idx_t x = b + k;
            while(i + 1 < off.size() && x >= off[i + 1])    ++i;
            e[k] = part[i][x - off[i]];
        }

        return sa40::load(e, 0);
    }
};


// Returns seconds elapsed since `t0`.
double since(const std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


// Spot-checks the order of the 40-bit suffix array `sa` over the text at
// `text_path`, of `n` symbols of type `T_sym_`. Returns `false` with a reason in
// `err` on failure.
template <typename T_sym_>
bool spot_check(const std::string& text_path, const idx_t n, const Parted_SA& sa, const idx_t samples, std::string& err)
{
    Mapping text;
    if(!text.map(text_path))    { err = "cannot map text: " + text_path; return false; }

    const auto T = reinterpret_cast<const T_sym_*>(text.data());
    return ref::spot_check_sa_order(T, n, sa, samples, 1000, 0x5EED, err);
}


// Dispatches the order spot-check over the symbol width `sym_bytes`.
bool spot_check(const uint32_t sym_bytes, const std::string& text_path, const idx_t n, const Parted_SA& sa, const idx_t samples,
                std::string& err)
{
    switch(sym_bytes)
    {
    case 1: return spot_check<uint8_t>(text_path, n, sa, samples, err);
    case 2: return spot_check<uint16_t>(text_path, n, sa, samples, err);
    default:    return spot_check<uint32_t>(text_path, n, sa, samples, err);
    }
}


// Maps the parts of the 40-bit suffix array described by `m` into `maps` and
// returns an accessor over them.
bool map_parts(const Index_Meta& m, std::vector<Mapping>& maps, Parted_SA& sa, std::string& err)
{
    maps.clear();
    sa.part.clear();
    for(const auto& p : m.sa_part)
    {
        maps.emplace_back();
        if(!maps.back().map(p)) { err = "cannot map suffix-array part: " + p; return false; }
        sa.part.push_back(maps.back().data());
    }

    sa.off = m.sa_part_off;
    sa.bytes = m.sa_bytes();
    return true;
}


// Returns the text-length implied by the text at `path` of `sym_bytes`-wide
// symbols, or `0` with a reason in `err` if malformed.
idx_t text_length(const std::string& path, const uint32_t sym_bytes, std::string& err)
{
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(path, ec);
    if(ec)  { err = "cannot stat text: " + path; return 0; }
    if(bytes == 0 || bytes % sym_bytes != 0)    { err = "text size is not a positive multiple of the symbol width: " + path; return 0; }
    if(bytes / sym_bytes >= SA_MAX_N)   { err = "text too long for 40-bit suffix-array entries"; return 0; }

    return bytes / sym_bytes;
}


// Converts the uint64 suffix array per `a`. Returns the process exit code.
int convert(const Args& a)
{
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();

    const idx_t n = text_length(a.text, a.sym_bytes, err);
    if(n == 0)  { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    std::error_code ec;
    const auto in_bytes = std::filesystem::file_size(a.in, ec);
    if(ec || in_bytes != n * sizeof(uint64_t))
    {
        std::fprintf(stderr, "input suffix array %s has %llu bytes; expected %llu (n = %llu uint64 entries)\n", a.in.c_str(),
                     static_cast<unsigned long long>(ec ? 0 : in_bytes), static_cast<unsigned long long>(n * sizeof(uint64_t)),
                     static_cast<unsigned long long>(n));
        return 1;
    }

    std::filesystem::create_directories(a.index_dir, ec);

    // Lay out the output parts.
    Index_Meta m;
    m.n = n;
    m.sym_bytes = a.sym_bytes;
    m.text = std::filesystem::absolute(a.text).string();
    m.sep = a.sep;
    m.sa_source = a.source.empty() ? ("sa_convert from " + std::filesystem::absolute(a.in).string()) : a.source;
    m.sa_part_off = index_meta::part_offsets(m.sa_bytes(), a.parts);
    for(idx_t i = 0; i < a.parts; ++i)
    {
        const std::string dir = (a.parts == 1 ? a.index_dir : a.part_dirs[i]);
        std::filesystem::create_directories(dir, ec);
        const auto name = (a.parts == 1 ? std::string("sa.u40") : "sa.u40.part" + std::to_string(i));
        m.sa_part.push_back(std::filesystem::absolute(std::filesystem::path(dir) / name).string());
    }

    if(!index_meta::validate(m, err))   { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    const int in_fd = ::open(a.in.c_str(), O_RDONLY);
    if(in_fd < 0)   { std::fprintf(stderr, "cannot open %s\n", a.in.c_str()); return 1; }

    std::vector<int> out_fd;
    for(idx_t i = 0; i < a.parts; ++i)
    {
        const int fd = ::open(m.sa_part[i].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if(fd < 0 || ::ftruncate(fd, static_cast<off_t>(m.part_bytes(i))) != 0)
        {
            std::fprintf(stderr, "cannot create %s\n", m.sa_part[i].c_str());
            return 1;
        }

        out_fd.push_back(fd);
    }

    // Stream the input in chunks: validate and narrow each chunk in parallel, then
    // scatter its bytes into the parts.
    constexpr idx_t chunk = idx_t(1) << 24;
    std::vector<uint64_t> in_buf(chunk);
    std::vector<uint8_t> out_buf(chunk * SA_WIDTH + 8);
    std::vector<uint64_t> seen(a.check == "perm" ? (n + 63) / 64 : 0);   // Bitset of the entries seen.
    std::atomic<idx_t> bad_rank{UINT64_MAX};
    std::atomic<bool> dup{false};

    for(idx_t r0 = 0; r0 < n; r0 += chunk)
    {
        const idx_t e = std::min(chunk, n - r0);
        if(!pread_full(in_fd, in_buf.data(), e * sizeof(uint64_t), r0 * sizeof(uint64_t)))
        {
            std::fprintf(stderr, "cannot read %s at rank %llu\n", a.in.c_str(), static_cast<unsigned long long>(r0));
            return 1;
        }

        #pragma omp parallel for schedule(static)
        for(idx_t i = 0; i < static_cast<idx_t>(e); ++i)
        {
            const uint64_t v = in_buf[i];
            if(a.check != "none" && v >= n)
            {
                idx_t cur = bad_rank.load();
                while(r0 + i < cur && !bad_rank.compare_exchange_weak(cur, r0 + i)) {}
                continue;
            }

            if(!seen.empty())
            {
                const uint64_t bit = uint64_t(1) << (v & 63);
                if(__atomic_fetch_or(&seen[v >> 6], bit, __ATOMIC_RELAXED) & bit)
                    dup.store(true, std::memory_order_relaxed);
            }

            sa40::store(out_buf.data(), i, v);
        }

        if(bad_rank.load() != UINT64_MAX)
        {
            std::fprintf(stderr, "suffix-array entry out of range [0, n) at rank %llu\n", static_cast<unsigned long long>(bad_rank.load()));
            return 1;
        }

        if(dup.load())
        {
            std::fprintf(stderr, "suffix array has a duplicate entry in ranks [%llu, %llu); it is not a permutation\n",
                         static_cast<unsigned long long>(r0), static_cast<unsigned long long>(r0 + e));
            return 1;
        }

        // Scatter the chunk's bytes `[b0, b1)` into the parts it overlaps.
        const idx_t b0 = SA_WIDTH * r0, b1 = SA_WIDTH * (r0 + e);
        for(idx_t i = 0; i < a.parts; ++i)
        {
            const idx_t p0 = m.sa_part_off[i], p1 = p0 + m.part_bytes(i);
            const idx_t s = std::max(b0, p0), t = std::min(b1, p1);
            if(s < t && !pwrite_full(out_fd[i], out_buf.data() + (s - b0), t - s, s - p0))
            {
                std::fprintf(stderr, "cannot write %s\n", m.sa_part[i].c_str());
                return 1;
            }
        }

        const idx_t done = r0 + e;
        if((done / chunk) % 16 == 0 || done == n)
            std::fprintf(stderr, "[sa_convert] %.1f%% (%llu / %llu entries), %.1fs\n", 100.0 * done / n,
                         static_cast<unsigned long long>(done), static_cast<unsigned long long>(n), since(t0));
    }

    ::close(in_fd);
    for(const int fd : out_fd)
        if(::fsync(fd) != 0 || ::close(fd) != 0)
        {
            std::fprintf(stderr, "cannot flush the suffix-array parts\n");
            return 1;
        }

    // Distinct entries, all in `[0, n)`, and `n` of them: a permutation.
    if(a.check == "perm")
        std::fprintf(stderr, "[sa_convert] permutation check passed\n");

    // Spot-check the order against the text.
    std::vector<Mapping> maps;
    Parted_SA sa;
    if(!map_parts(m, maps, sa, err) || !spot_check(a.sym_bytes, m.text, n, sa, a.spot, err))
    {
        std::fprintf(stderr, "order check failed: %s\n", err.c_str());
        return 1;
    }

    std::fprintf(stderr, "[sa_convert] order spot-check passed (%llu pairs)\n", static_cast<unsigned long long>(a.spot));

    if(!index_meta::write(a.index_dir, m, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    std::fprintf(stderr, "[sa_convert] wrote %s/%s in %.1fs\n", a.index_dir.c_str(), INDEX_META_FILE, since(t0));
    return 0;
}


// Validates an existing 40-bit suffix array and writes its metadata per `a`.
// Returns the process exit code.
int meta_only(const Args& a)
{
    std::string err;
    const idx_t n = text_length(a.text, a.sym_bytes, err);
    if(n == 0)  { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    Index_Meta m;
    m.n = n;
    m.sym_bytes = a.sym_bytes;
    m.text = std::filesystem::absolute(a.text).string();
    m.sep = a.sep;
    m.sa_source = a.source.empty() ? ("native 40-bit suffix array " + std::filesystem::absolute(a.sa).string()) : a.source;
    m.sa_part = {std::filesystem::absolute(a.sa).string()};
    m.sa_part_off = {0};

    std::vector<Mapping> maps;
    Parted_SA sa;
    if(!index_meta::validate(m, err) || !index_meta::validate_files(m, err) || !map_parts(m, maps, sa, err)
       || !spot_check(a.sym_bytes, m.text, n, sa, a.spot, err))
    {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(a.index_dir, ec);
    if(!index_meta::write(a.index_dir, m, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    std::fprintf(stderr, "[sa_convert] order spot-check passed; wrote %s/%s\n", a.index_dir.c_str(), INDEX_META_FILE);
    return 0;
}


// Widens a 40-bit suffix array into uint64 entries per `a`. Returns the process
// exit code.
int widen(const Args& a)
{
    std::error_code ec;
    const auto in_bytes = std::filesystem::file_size(a.in, ec);
    if(ec || in_bytes % SA_WIDTH != 0)  { std::fprintf(stderr, "input is not a 40-bit suffix array: %s\n", a.in.c_str()); return 1; }

    const idx_t n = in_bytes / SA_WIDTH;
    const int in_fd = ::open(a.in.c_str(), O_RDONLY);
    const int out_fd = ::open(a.out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(in_fd < 0 || out_fd < 0) { std::fprintf(stderr, "cannot open the input or the output\n"); return 1; }

    constexpr idx_t chunk = idx_t(1) << 24;
    std::vector<uint8_t> in_buf(chunk * SA_WIDTH + 8);
    std::vector<uint64_t> out_buf(chunk);
    for(idx_t r0 = 0; r0 < n; r0 += chunk)
    {
        const idx_t e = std::min(chunk, n - r0);
        if(!pread_full(in_fd, in_buf.data(), e * SA_WIDTH, r0 * SA_WIDTH))  { std::fprintf(stderr, "read failed\n"); return 1; }

        #pragma omp parallel for schedule(static)
        for(idx_t i = 0; i < static_cast<idx_t>(e); ++i)
            out_buf[i] = sa40::load(in_buf.data(), i);

        if(!pwrite_full(out_fd, out_buf.data(), e * sizeof(uint64_t), r0 * sizeof(uint64_t)))   { std::fprintf(stderr, "write failed\n"); return 1; }
    }

    ::close(in_fd);
    if(::fsync(out_fd) != 0 || ::close(out_fd) != 0)   { std::fprintf(stderr, "cannot flush the output\n"); return 1; }

    return 0;
}

}


int main(int argc, char** argv)
{
    if(!gpu_saca::sa40::host_is_little_endian())
    {
        std::fprintf(stderr, "sa_convert requires a little-endian host\n");
        return 1;
    }

    Args a;
    std::string err;
    if(!parse(argc, argv, a, err))
    {
        std::fprintf(stderr, "%s%s%s", err.c_str(), err.empty() ? "" : "\n", USAGE);
        return 2;
    }

    if(a.mode == "widen")       return widen(a);
    if(a.mode == "meta-only")   return meta_only(a);
    return convert(a);
}
