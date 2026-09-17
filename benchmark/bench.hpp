
// Shared harness for the baseline SACA drivers. `finalize` owns the single
// record-emission path -- including our own driver, which has its own run flow
// but have the same record structure.

#ifndef BENCH_HPP
#define BENCH_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__) || defined(__APPLE__)
    #include <sys/resource.h>
#endif
#if defined(__linux__)
    #include <sched.h>
#endif

extern "C" char** environ;

#ifndef BENCH_COMMIT
    #define BENCH_COMMIT "unknown"
#endif
#ifndef BENCH_BUILD_TYPE
    #define BENCH_BUILD_TYPE "unknown"
#endif

// Corpora are little-endian on-disk arrays.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                  "Corpora are little-endian; a big-endian host would sort byte-swapped symbols.");
#endif


namespace bench
{

using Clock = std::chrono::steady_clock;

inline double since(const Clock::time_point& t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}


// RAII buffer.
template <typename T_>
struct Buffer
{
    T_* p = nullptr;
    int64_t n = 0;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { std::free(p); }

    bool allocate(const int64_t count)
    {
        std::free(p);
        p = static_cast<T_*>(std::malloc(static_cast<std::size_t>(count) * sizeof(T_)));
        n = (p ? count : 0);
        return p != nullptr;
    }
};


// JSON string escaping. Control bytes below 0x20 must be `\u`-escaped.
inline std::string esc(const std::string& s)
{
    static const char* const hex = "0123456789abcdef";
    std::string o;
    o.reserve(s.size() + 8);

    for(const char c : s)
    {
        const auto u = static_cast<unsigned char>(c);
        switch(c)
        {
            case '"':   o += "\\\"";    break;
            case '\\':  o += "\\\\";    break;
            case '\n':  o += "\\n";     break;
            case '\r':  o += "\\r";     break;
            case '\t':  o += "\\t";     break;
            default:
                if(u < 0x20)    { o += "\\u00"; o += hex[(u >> 4) & 0xF]; o += hex[u & 0xF]; }
                else            o += c;
        }
    }

    return o;
}


inline std::string now_utc()
{
    const auto t = std::time(nullptr);
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return std::string(b);
}


inline std::string host_name()
{
    char b[256] = {0};
    return (::gethostname(b, sizeof(b) - 1) == 0) ? std::string(b) : "unknown";
}


// Returns the peak RSS.
inline int64_t peak_rss_bytes()
{
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    for(std::string k; f >> k; )
        if(k == "VmHWM:")
        {
            int64_t kb = 0;
            f >> kb;
            return kb * 1024;
        }
#elif defined(__APPLE__)    // The macOS `getrusage` fallback to support local compilation.
    struct rusage ru;
    if(::getrusage(RUSAGE_SELF, &ru) == 0)  return static_cast<int64_t>(ru.ru_maxrss);
#endif
    return 0;
}


// Some kernel information: CPU time separates a thread-count change from a real
// slowdown; the fault counts separate a page placement change; the context-
// switch counts separate spinning from sleeping.
struct Usage
{
    double cpu_user = -1;
    double cpu_sys = -1;
    int64_t minflt = -1;    // Minor page faults: page table miss.
    int64_t majflt = -1;    // Major page faults: RAM miss.
    int64_t nvcsw = -1;     // Voluntary context switches.
    int64_t nivcsw = -1;    // Involuntary context switches.
};


// Returns some usage information of the process from the kernel.
inline Usage usage_self()
{
    Usage u;
#if defined(__linux__) || defined(__APPLE__)
    struct rusage r;
    if(::getrusage(RUSAGE_SELF, &r) == 0)   // RUSAGE_SELF sums all threads.
    {
        u.cpu_user = static_cast<double>(r.ru_utime.tv_sec) + r.ru_utime.tv_usec / 1e6;
        u.cpu_sys  = static_cast<double>(r.ru_stime.tv_sec) + r.ru_stime.tv_usec / 1e6;
        u.minflt   = r.ru_minflt;   u.majflt = r.ru_majflt;
        u.nvcsw    = r.ru_nvcsw;    u.nivcsw = r.ru_nivcsw;
    }
#endif
    return u;
}


// Returns the size of the affinity mask, i.e. what `nproc` reports, not
// necessarily the size of the machine: all-cores cannot be assumed.
inline int64_t cpus_allowed_count()
{
#if defined(__linux__)
    cpu_set_t set;
    if(::sched_getaffinity(0, sizeof(set), &set) == 0)  return CPU_COUNT(&set);
#endif
    return -1;
}


// Returns the FNV-1a hash of the running executable.
inline uint64_t executable_hash()
{
#if defined(__linux__)
    std::ifstream f("/proc/self/exe", std::ios::binary);
    if(!f)  return 0;

    uint64_t h = 14695981039346656037ULL;   // Offset basis (20 digits) for FNV-1a.
    char buf[1 << 16];
    while(f.read(buf, sizeof(buf)) || f.gcount())
        for(std::streamsize i = 0; i < f.gcount(); ++i)
            h = (h ^ static_cast<unsigned char>(buf[i])) * 1099511628211ULL;

    return h;
#else
    return 0;
#endif
}


// Returns very OMP_* and GPU_SACA_* variable actually in the environment.
inline std::string tuning_env()
{
    std::vector<std::string> v;
    for(char** e = ::environ; e != nullptr && *e != nullptr; ++e)
    {
        const std::string kv(*e);
        if(kv.rfind("OMP_", 0) == 0 || kv.rfind("GPU_SACA_", 0) == 0)
            v.push_back(kv);
    }

    std::sort(v.begin(), v.end());
    std::string r;
    for(const auto& kv : v)
        r += (r.empty() ? "" : ";") + kv;

    return r;
}


// The return statuses.
namespace st
{
    inline constexpr const char* OK         = "ok";
    inline constexpr const char* OOM_HOST   = "oom_host";
    inline constexpr const char* OOM_DEVICE = "oom_device";
    inline constexpr const char* BAD_INPUT  = "bad_input";
    inline constexpr const char* BAD_PARAM  = "bad_param";
    inline constexpr const char* LIB_ERROR  = "lib_error";
}


// How to interpret the input bytes. Recorded, not inferred.
namespace alpha
{
    inline constexpr const char* TOKEN16  = "token16";
    inline constexpr const char* BYTE     = "byte";
    inline constexpr const char* BYTE_T16 = "byte_from_token16";

    inline bool valid(const std::string& a) { return a == TOKEN16 || a == BYTE || a == BYTE_T16; }
    inline int  symbol_bytes(const std::string& a) { return a == TOKEN16 ? 2 : 1; }
}


// Each run produces a record, in a file. Timing: -1 is "not measured", 0.0 is
// "measured as zero". No `t_e2e` field: the summarizer sums the buckets.
struct Record
{
    static constexpr int schema_version = 4;

    std::string run_id, commit = BENCH_COMMIT, build_type = BENCH_BUILD_TYPE;
    std::string host = host_name(), timestamp_utc = now_utc();
    std::string method, method_version;
    std::string input, corpus, tokenizer, alphabet;
    int64_t     symbol_bytes = 0, n_symbols = 0, n_tokens = 0, threads = 0;
    std::string status;
    int64_t     code = 0;
    std::string detail, verify = "off";     // off (not requested) | pass | fail | skip (wanted, infeasible)
    double      t_load = -1, t_core = -1, t_write = -1;
    int64_t     peak_rss = 0, peak_device = -1;     // peak_device: -1 where inapplicable

    // What the run actually executed under, rather than what it was asked for.
    // `threads` above is the request; `threads_avail` is what the OpenMP runtime
    // resolves, and `cpus_allowed` what the kernel lets it use.
    double      cpu_user = -1, cpu_sys = -1;
    int64_t     minflt = -1, majflt = -1, nvcsw = -1, nivcsw = -1;
    int64_t     threads_avail = -1, cpus_allowed = -1;
    uint64_t    exe_hash = 0;
    std::string env_tuning;

    // Our method's phase breakdown. -1 for the baselines.
    double      t_prepage_pid = -1, t_pivots = -1, t_partition = -1, t_device_setup = -1,
                t_initial_rank = -1, t_exponentiate = -1, t_concatenate = -1;
    int64_t     n_partitions = -1, n_generations = -1;

    std::string notes;

    // Two macros so the JSON key and the C++ member are the same token, which
    // rules out a key that misnames its member. Note what it does not rule out:
    // a member with no line below compiles cleanly and is silently absent from
    // the JSON, so adding a field means adding it in both places.
    std::string to_json() const
    {
        std::ostringstream o;
        o.setf(std::ios::fixed);
        o.precision(6);
        const char* sep = "";
        o << "{";

        #define J_NUM(k) (o << sep << "\"" #k "\":"   << (k),             sep = ",")
        #define J_STR(k) (o << sep << "\"" #k "\":\"" << esc(k) << '"',   sep = ",")

        J_NUM(schema_version);
        J_STR(run_id);       J_STR(commit);        J_STR(build_type);
        J_STR(host);         J_STR(timestamp_utc);
        J_STR(method);       J_STR(method_version);
        J_STR(input);        J_STR(corpus);        J_STR(tokenizer);   J_STR(alphabet);
        J_NUM(symbol_bytes); J_NUM(n_symbols);     J_NUM(n_tokens);   J_NUM(threads);
        J_STR(status);       J_NUM(code);          J_STR(detail);     J_STR(verify);
        J_NUM(t_load);       J_NUM(t_core);        J_NUM(t_write);
        J_NUM(peak_rss);     J_NUM(peak_device);
        J_NUM(cpu_user);     J_NUM(cpu_sys);
        J_NUM(minflt);       J_NUM(majflt);        J_NUM(nvcsw);      J_NUM(nivcsw);
        J_NUM(threads_avail);J_NUM(cpus_allowed);  J_NUM(exe_hash);   J_STR(env_tuning);
        J_NUM(t_prepage_pid);  J_NUM(t_pivots);      J_NUM(t_partition);
        J_NUM(t_device_setup); J_NUM(t_initial_rank);J_NUM(t_exponentiate);
        J_NUM(t_concatenate);  J_NUM(n_partitions);  J_NUM(n_generations);
        J_STR(notes);

        #undef J_NUM
        #undef J_STR

        o << "}";
        return o.str();
    }

    // Temporary plus `rename`, so "absent" reliably means the process died
    // rather than that it was mid-write.
    bool write(const std::string& dir) const
    {
        const std::string path = dir + "/" + run_id + ".json", tmp = path + ".tmp";
        {
            std::ofstream f(tmp);
            if(!f)  return false;
            f << to_json() << "\n";
            if(!f)  return false;
        }
        return std::rename(tmp.c_str(), path.c_str()) == 0;
    }
};


struct Args
{
    std::string input, corpus, run_id, alphabet, tokenizer;
    std::string method_version;         // Overrides the library's own version string.
    std::string results_dir = ".";
    std::string sa_out;                 // Writing the SA is opt-in: github's would be 472 GiB.

    int64_t n = 0;                      // Symbols to read; <= 0 reads the whole file.
    int64_t threads = 0;                // 0 defers to the library's default.
    int64_t fs = 0;                     // libsais free-space parameter.

    bool verify = false;
};


inline const char* usage()
{
    return
    "  --input P --corpus NAME --run-id ID --alphabet token16|byte|byte_from_token16\n"
    "  [--tokenizer NAME] [--method-version V]\n"
    "  [--n N] [--threads T] [--fs F] [--results-dir D] [--sa-out P] [--verify]\n"
    "\n"
    "  --n counts SYMBOLS, not tokens. For byte_from_token16 that is bytes,\n"
    "  i.e. twice the token count.\n";
}


inline bool parse(const int argc, char** const argv, Args& a, std::string& err)
{
    for(int i = 1; i < argc; ++i)
    {
        const std::string k(argv[i]);
        const bool takes_val = (k != "--verify");

        if(takes_val && i + 1 >= argc)   { err = "missing value for " + k; return false; }
        const char* const v = takes_val ? argv[++i] : nullptr;

        if     (k == "--verify")        a.verify      = true;
        else if(k == "--input")         a.input       = v;
        else if(k == "--corpus")        a.corpus      = v;
        else if(k == "--tokenizer")     a.tokenizer   = v;
        else if(k == "--method-version") a.method_version = v;
        else if(k == "--run-id")        a.run_id      = v;
        else if(k == "--alphabet")      a.alphabet    = v;
        else if(k == "--results-dir")   a.results_dir = v;
        else if(k == "--sa-out")        a.sa_out      = v;
        else if(k == "--n")             a.n           = std::strtoll(v, nullptr, 10);
        else if(k == "--threads")       a.threads     = std::strtoll(v, nullptr, 10);
        else if(k == "--fs")            a.fs          = std::strtoll(v, nullptr, 10);
        else                            { err = "unknown argument: " + k; return false; }
    }

    if(a.input.empty())     { err = "--input is required";  return false; }
    if(a.corpus.empty())    { err = "--corpus is required"; return false; }
    if(a.run_id.empty())    { err = "--run-id is required"; return false; }
    if(!alpha::valid(a.alphabet))
    {
        err = "--alphabet must be token16|byte|byte_from_token16 (got \"" + a.alphabet + "\")";
        return false;
    }

    return true;
}


// Reads up to `n_req` elements of `T_sym_`; `buf.n` is what was actually read.
template <typename T_sym_>
inline bool load_prefix(const std::string& path, const int64_t n_req, Buffer<T_sym_>& buf, std::string& err)
{
    const int fd = ::open(path.c_str(), O_RDONLY);
    if(fd < 0)  { err = "open failed: " + path; return false; }

    const auto fail = [&](const std::string& m) { err = m; ::close(fd); return false; };

    struct stat sb;
    if(::fstat(fd, &sb) != 0)   return fail("fstat failed: " + path);

    const int64_t avail = static_cast<int64_t>(sb.st_size) / static_cast<int64_t>(sizeof(T_sym_));
    const int64_t n     = (n_req <= 0) ? avail : std::min(n_req, avail);

    if(n <= 0)          return fail("empty corpus: " + path);
    if(!buf.allocate(n))return fail("allocation failed for " + std::to_string(n) + " symbols");

    const std::size_t want = static_cast<std::size_t>(n) * sizeof(T_sym_);
    std::size_t got = 0;
    auto* const dst = reinterpret_cast<uint8_t*>(buf.p);

    while(got < want)   // `pread` transfers at most ~2 GB per call.
    {
        const auto r = ::pread(fd, dst + got, want - got, static_cast<off_t>(got));
        if(r < 0)       return fail("pread failed at offset " + std::to_string(got));
        if(r == 0)      break;  // Shorter file than `fstat` implied.
        got += static_cast<std::size_t>(r);
    }

    ::close(fd);
    buf.n = static_cast<int64_t>(got / sizeof(T_sym_));
    return buf.n > 0;
}


// Always writes `n` little-endian uint64 values, whatever width the library
// produced, so cross-checking two methods is a plain `cmp` of two files.
template <typename T_sa_>
inline bool write_SA(const std::string& path, const T_sa_* const SA, const int64_t n, std::string& err)
{
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(fd < 0)  { err = "cannot open for write: " + path; return false; }

    const auto fail = [&](const std::string& m) { err = m; ::close(fd); return false; };

    // Chunked: keeps the widening buffer small and respects pwrite's ~2 GB cap.
    constexpr int64_t CHUNK = 1 << 20;
    std::vector<uint64_t> out(static_cast<std::size_t>(std::min(CHUNK, n)));

    for(int64_t off = 0; off < n; )
    {
        const int64_t m = std::min(CHUNK, n - off);
        for(int64_t i = 0; i < m; ++i)
            out[static_cast<std::size_t>(i)] = static_cast<uint64_t>(SA[off + i]);

        const char* src  = reinterpret_cast<const char*>(out.data());
        std::size_t want = static_cast<std::size_t>(m) * sizeof(uint64_t);
        off_t       at   = static_cast<off_t>(off) * static_cast<off_t>(sizeof(uint64_t));

        while(want > 0)
        {
            const auto w = ::pwrite(fd, src, want, at);
            if(w <= 0)  return fail("pwrite failed at offset " + std::to_string(at));
            src += w;   want -= static_cast<std::size_t>(w);     at += w;
        }

        off += m;
    }

    if(::close(fd) != 0)    { err = "close failed: " + path; return false; }
    return true;
}


// Range, then permutation, then adjacent order. The order check builds an ISA,
// so this adds n/8 + 8n on top of the run's own peak (~18n for libsais) -- hence
// opt-in, and dropped at the top of the ladder.
template <typename T_sym_, typename T_sa_>
inline const char* verify_SA(const T_sym_* const T, const T_sa_* const SA, const int64_t n,
                             std::string& detail)
{
    std::vector<bool> seen(static_cast<std::size_t>(n), false);

    for(int64_t i = 0; i < n; ++i)
    {
        const int64_t s = static_cast<int64_t>(SA[i]);

        if(s < 0 || s >= n)
        {
            detail = "SA[" + std::to_string(i) + "] = " + std::to_string(s) + " out of range";
            return "fail";
        }

        if(seen[static_cast<std::size_t>(s)])
        {
            detail = "not a permutation: " + std::to_string(s) + " repeats at index " + std::to_string(i);
            return "fail";
        }

        seen[static_cast<std::size_t>(s)] = true;
    }

    Buffer<int64_t> ISA;
    if(!ISA.allocate(n))    { detail = "ISA allocation failed; order check skipped"; return "skip"; }

    for(int64_t i = 0; i < n; ++i)
        ISA.p[static_cast<int64_t>(SA[i])] = i;

    // -1 past the end makes the empty suffix smallest, matching both libraries'
    // shorter-suffix-first, no-sentinel order (baselines.md).
    const auto rank_next = [&](const int64_t s) { return (s + 1 < n) ? ISA.p[s + 1] : int64_t(-1); };

    for(int64_t i = 1; i < n; ++i)
    {
        const int64_t p = static_cast<int64_t>(SA[i - 1]), q = static_cast<int64_t>(SA[i]);

        if(!((T[p] < T[q]) || (T[p] == T[q] && rank_next(p) < rank_next(q))))
        {
            detail = "order violated at index " + std::to_string(i);
            return "fail";
        }
    }

    return "pass";
}


// The single record-finalization path: stamps the status, reads required kernel
// information, writes the record atomically, and echoes it. Returns the process
// exit code: 0 whenever a record was written, including for a failed run. Non-
// zero means no record exists.
inline int finalize(Record& rec, const std::string& results_dir,
                    const char* const status, const int64_t code, const std::string& detail)
{
    rec.status = status;
    rec.code   = code;
    rec.detail = detail;

    rec.peak_rss = peak_rss_bytes();

    const auto u   = usage_self();
    rec.cpu_user   = u.cpu_user;    rec.cpu_sys = u.cpu_sys;
    rec.minflt     = u.minflt;      rec.majflt  = u.majflt;
    rec.nvcsw      = u.nvcsw;       rec.nivcsw  = u.nivcsw;
    rec.cpus_allowed = cpus_allowed_count();
    rec.exe_hash     = executable_hash();
    rec.env_tuning   = tuning_env();

    if(!rec.write(results_dir))
    {
        std::fprintf(stderr, "FATAL: could not write record to %s/%s.json\n",
                     results_dir.c_str(), rec.run_id.c_str());
        return 1;
    }

    std::fprintf(stderr, "%s\n", rec.to_json().c_str());
    return 0;
}


// `code` is the library's own return value, verbatim, so a record can be matched
// against upstream source.
struct Outcome
{
    const char* status = st::OK;
    int64_t     code   = 0;
    std::string detail;
};

template <typename T_sym_>
struct Ctx
{
    const Args&   arg;
    Record&       rec;      // The driver fills its own t_core/t_write/peak_device/verify.
    const T_sym_* T;
    int64_t       n;        // Symbols actually loaded.
};


// Owns the corpus load and the run/argument fields, then hands off to
// `finalize`. Exit-code policy lives there.
template <typename T_sym_, typename F_core_>
int bench_main(const Args& arg, const char* const method, const char* const method_version,
               F_core_&& core)
{
    Record rec;
    rec.run_id       = arg.run_id;
    rec.method       = method;
    rec.method_version = method_version;
    rec.input        = arg.input;
    rec.corpus       = arg.corpus;
    rec.tokenizer    = arg.tokenizer;
    rec.alphabet     = arg.alphabet;
    rec.symbol_bytes = alpha::symbol_bytes(arg.alphabet);
    rec.threads      = arg.threads;

    const auto emit = [&](const char* const status, const int64_t code, const std::string& detail)
    { return finalize(rec, arg.results_dir, status, code, detail); };

    Buffer<T_sym_> T;
    std::string    err;

    const auto t0 = Clock::now();
    if(!load_prefix(arg.input, arg.n, T, err))              return emit(st::BAD_INPUT, 0, err);
    rec.t_load = since(t0);

    rec.n_symbols = T.n;
    // byte_from_token16 reads a uint16 corpus as bytes, so symbols = 2 x tokens.
    // Throughput must be per token or its apparent work doubles.
    rec.n_tokens  = (arg.alphabet == alpha::BYTE_T16) ? T.n / 2 : T.n;

    if(arg.n > 0 && T.n < arg.n)
        rec.notes = "short corpus: requested " + std::to_string(arg.n)
                  + " symbols, read " + std::to_string(T.n);

    Ctx<T_sym_> ctx{arg, rec, T.p, T.n};
    const Outcome out = core(ctx);

    return emit(out.status, out.code, out.detail);
}

}



#endif
