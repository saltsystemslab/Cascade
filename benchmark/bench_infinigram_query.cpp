
// Baseline query driver: infini-gram's unmodified engine (`cpp_engine.h`,
// v2.6.0) counting batches of `.qry` queries over an infini-gram index, from a
// pool of threads. The index is loaded once for every query file. Output is
// `key=value` lines on stdout, as `query find` prints: the load's, and then
// every batch's. `--cnt-out` and `--cnt-dir` write the counts, byte-comparable
// with those of `query find`.
//
// Usage: bench_infinigram_query --index DIR --queries Q1.qry[,Q2.qry,...] [--q N] [--mode find|direct]
//            [--threads T] [--load mmap|ram] [--prefetch 0|1] [--prewarm 0|1] [--reps K]
//            [--lock-text 0|1] [--cold 0|1] [--cnt-out F] [--cnt-dir D]
//   --q         the first N queries of every file; default all.
//   --mode      find: the engine's `find`, which starts a thread per shard for
//               every query (default). direct: the engine's `_find_thread` over
//               every whole shard, called from the pool.
//   --threads   pool size; default: the hardware concurrency.
//   --load      mmap: the files memory-mapped with `MADV_RANDOM`, the engine's
//               default (default). ram: read into the heap.
//   --prefetch  0: no prefetching, as the public API server runs (default).
//               1: the engine's default depths (1, 3, 3).
//   --prewarm   1: read the index files through the page cache before the runs
//               (default). The first of the K runs (default 3) warms up too.
//   --lock-text 1: lock the token arrays' pages in memory before the engine maps
//               them, so that its text never waits on the drives, as when the
//               text is in memory and the table mapped from a drive.
//   --cold      1: drop the tables' pages from the page cache before every run,
//               the engine's mappings of them first; with --load mmap.
//   Every batch reports the bytes that each run had read from storage.
//   --cnt-out   the counts of a single query file; --cnt-dir those of each, to
//               D/<stem>.cnt.

#include "query-file.hpp"

#include "cpp_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>


namespace
{

const char* const USAGE =
R"(Usage: bench_infinigram_query --index DIR --queries Q1.qry[,Q2.qry,...] [--q N] [--mode find|direct]
           [--threads T] [--load mmap|ram] [--prefetch 0|1] [--prewarm 0|1] [--reps K]
           [--lock-text 0|1] [--cold 0|1] [--cnt-out F] [--cnt-dir D]
)";

constexpr std::size_t CHUNK = 256;  // Queries a pool thread takes at a time.


double seconds_since(const std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


// Returns the real paths of the regular files of the index directory `dir`
// whose names contain `kind`, as infini-gram classifies them: "tokenized",
// "table", or "" for all.
std::vector<std::string> index_files(const std::string& dir, const std::string& kind)
{
    std::vector<std::string> files;
    for(const auto& e : std::filesystem::directory_iterator(dir))
        if(e.path().filename().string().find(kind) != std::string::npos && std::filesystem::is_regular_file(e.path()))
            files.push_back(std::filesystem::canonical(e.path()).string());

    return files;
}


// Returns the bytes that the process has had read from storage, per
// `/proc/self/io`; `0` if unknown.
uint64_t storage_read_bytes()
{
    std::ifstream is("/proc/self/io");
    for(std::string k; is >> k; )
    {
        uint64_t v = 0;
        is >> v;
        if(k == "read_bytes:")
            return v;
    }

    return 0;
}


// Maps the files `paths` and locks their pages in memory: an engine's own
// mappings of them then find the pages resident. The mappings stay for the
// process's life. Returns `false` with a reason in `err` on failure.
bool lock_files(const std::vector<std::string>& paths, std::string& err)
{
    for(const auto& p : paths)
    {
        const uint64_t bytes = std::filesystem::file_size(p);
        const int fd = ::open(p.c_str(), O_RDONLY);
        void* const m = (fd >= 0 && bytes > 0 ? ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED);
        if(fd >= 0)
            ::close(fd);    // The mapping keeps the file.
        if(m == MAP_FAILED || ::mlock(m, bytes) != 0)
        {
            err = "cannot lock " + p + " in memory: " + std::strerror(errno);
            return false;
        }
    }

    return true;
}


// Drops the pages of the files `paths` from the page cache, first unmapping
// them from this process, whose mappings would keep them. Returns `false` with
// a reason in `err` on failure.
bool drop_cached(const std::vector<std::string>& paths, std::string& err)
{
    std::ifstream maps("/proc/self/maps");
    for(std::string line; std::getline(maps, line); )   // `start-end perms offset dev inode path`
    {
        std::istringstream is(line);
        std::string range, perms, offset, dev, inode, path;
        is >> range >> perms >> offset >> dev >> inode;
        std::getline(is >> std::ws, path);
        if(std::find(paths.begin(), paths.end(), path) == paths.end())
            continue;

        const auto dash = range.find('-');
        const uint64_t start = std::stoull(range.substr(0, dash), nullptr, 16), end = std::stoull(range.substr(dash + 1), nullptr, 16);
        if(::madvise(reinterpret_cast<void*>(start), end - start, MADV_DONTNEED) != 0)
        {
            err = "cannot unmap the pages of " + path + ": " + std::strerror(errno);
            return false;
        }
    }

    for(const auto& p : paths)
    {
        const int fd = ::open(p.c_str(), O_RDONLY);
        const int e = (fd >= 0 ? ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) : errno);
        if(fd >= 0)
            ::close(fd);
        if(e != 0)
        {
            err = "cannot drop the pages of " + p + " from the page cache: " + std::strerror(e);
            return false;
        }
    }

    return true;
}


// Reads the files `paths` through the page cache with `threads` parallel
// readers. Returns the bytes read.
uint64_t prewarm(const std::vector<std::string>& paths, const unsigned threads)
{
    constexpr std::size_t PIECE = std::size_t(8) << 20;
    uint64_t total = 0;
    for(const auto& path : paths)
    {
        const uint64_t bytes = std::filesystem::file_size(path);
        const int fd = ::open(path.c_str(), O_RDONLY);
        if(fd < 0)
            continue;

        std::atomic<uint64_t> next{0};
        std::vector<std::thread> pool;
        for(unsigned t = 0; t < threads; ++t)
            pool.emplace_back([&]
            {
                std::vector<char> buf(PIECE);
                for(uint64_t off; (off = next.fetch_add(PIECE)) < bytes; )
                    if(::pread(fd, buf.data(), PIECE, static_cast<off_t>(off)) < 0)
                        break;
            });
        for(auto& t : pool)
            t.join();

        ::close(fd);
        total += bytes;
    }

    return total;
}


// Counts the first `q_max` queries of the file `path` with `engine`, `reps`
// times, through its `find` if `find` and else through `_find_thread`, from a
// pool of `threads` threads; prints the batch's `key=value` lines, and writes
// its counts to `cnt_path` unless empty. Returns the exit status.
int count_batch(const Engine<U16>& engine, const std::string& path, const uint64_t q_max, const bool find, const uint64_t threads,
                const uint64_t reps, const std::vector<std::string>& cold, const std::string& cnt_path)
{
    std::vector<uint64_t> off;
    std::vector<U16> sym;
    gpu_saca::Query_File_Header h;
    std::string err;
    if(!gpu_saca::query_file::read(path, q_max, off, sym, h, err))
    {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    const uint64_t q = off.size() - 1;
    const std::size_t shards = engine.get_num_shards();
    std::printf("queries=%s\nq=%" PRIu64 "\nsym_c=%" PRIu64 "\n", path.c_str(), q, off[q]);

    // One run over the batch: every pool thread takes `CHUNK` queries at a time.
    std::vector<uint64_t> cnt(q);
    const auto run = [&]
    {
        std::atomic<uint64_t> next{0};
        std::vector<std::thread> pool;
        for(uint64_t t = 0; t < threads; ++t)
            pool.emplace_back([&]
            {
                for(uint64_t b; (b = next.fetch_add(CHUNK)) < q; )
                    for(uint64_t i = b; i < std::min(q, b + CHUNK); ++i)
                    {
                        const U16* const P = sym.data() + off[i];
                        const uint64_t m = off[i + 1] - off[i];
                        if(find)
                            cnt[i] = engine.find(std::vector<U16>(P, P + m)).cnt;
                        else
                        {
                            uint64_t c = 0;
                            for(std::size_t s = 0; s < shards; ++s)
                            {
                                std::pair<U64, U64> seg;
                                engine._find_thread(s, reinterpret_cast<const U8*>(P), m * sizeof(U16), {0, engine.get_tok_cnt(s)}, &seg);
                                c += seg.second - seg.first;
                            }

                            cnt[i] = c;
                        }
                    }
            });
        for(auto& t : pool)
            t.join();
    };

    std::vector<double> t_run;
    std::vector<uint64_t> io;
    for(uint64_t r = 0; r < reps; ++r)
    {
        if(!cold.empty() && !drop_cached(cold, err))
        {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }

        const uint64_t b0 = storage_read_bytes();
        const auto t0 = std::chrono::steady_clock::now();
        run();
        t_run.push_back(seconds_since(t0));
        io.push_back(storage_read_bytes() - b0);
    }

    std::string io_list;
    for(const auto b : io)
        io_list += (io_list.empty() ? "" : ",") + std::to_string(b);

    const double t_first = t_run.front();
    std::vector<double> timed(t_run.begin() + (reps > 1 ? 1 : 0), t_run.end());    // The first run warms up.
    std::sort(timed.begin(), timed.end());
    uint64_t hits = 0, occ = 0;
    for(const auto c : cnt)
        hits += (c > 0), occ += c;

    std::printf("reps=%" PRIu64 "\nt_first=%.6f\nt_find_min=%.6f\nt_find_median=%.6f\nqps=%.0f\nhits=%" PRIu64 "\noccurrences=%" PRIu64
                "\ncounts_hash=0x%016" PRIx64 "\nio_bytes=%s\n",
                reps, t_first, timed.front(), timed[timed.size() / 2], q / timed.front(), hits, occ,
                gpu_saca::query_file::counts_hash(cnt.data(), q), io_list.c_str());
    std::fflush(stdout);

    if(!cnt_path.empty() && !gpu_saca::query_file::write_counts(cnt_path, cnt.data(), q))
    {
        std::fprintf(stderr, "error: cannot write %s\n", cnt_path.c_str());
        return 1;
    }

    return 0;
}

}


int main(const int argc, char** const argv)
{
    std::map<std::string, std::string> opt;
    for(int i = 1; i + 1 < argc; i += 2)
    {
        const std::string k = argv[i];
        if(k.compare(0, 2, "--") != 0)
        {
            std::fprintf(stderr, "error: unexpected argument `%s`\n%s", k.c_str(), USAGE);
            return 2;
        }

        opt[k.substr(2)] = argv[i + 1];
    }

    const auto get = [&](const std::string& k, const std::string& dflt) { return opt.count(k) ? opt.at(k) : dflt; };
    static const std::set<std::string> known = {"index", "queries", "q", "mode", "threads", "load", "prefetch", "prewarm", "reps", "lock-text",
                                                "cold", "cnt-out", "cnt-dir"};
    for(const auto& [k, v] : opt)
        if(!known.count(k))
        {
            std::fprintf(stderr, "error: unknown option --%s\n%s", k.c_str(), USAGE);
            return 2;
        }

    const std::string mode = get("mode", "find"), load = get("load", "mmap");
    if((argc - 1) % 2 != 0 || !opt.count("index") || !opt.count("queries") || (mode != "find" && mode != "direct")
        || (load != "mmap" && load != "ram"))
    {
        std::fprintf(stderr, "%s", USAGE);
        return 2;
    }

    uint64_t q_max = UINT64_MAX, threads = std::max(1u, std::thread::hardware_concurrency()), reps = 3;
    bool prefetch = false, warm = true, lock_text = false, cold = false;
    try
    {
        if(opt.count("q"))          q_max = std::stoull(opt["q"]);
        if(opt.count("threads"))    threads = std::max<uint64_t>(1, std::stoull(opt["threads"]));
        if(opt.count("reps"))       reps = std::max<uint64_t>(1, std::stoull(opt["reps"]));
        prefetch = std::stoul(get("prefetch", "0")) != 0;
        warm = std::stoul(get("prewarm", "1")) != 0;
        lock_text = std::stoul(get("lock-text", "0")) != 0;
        cold = std::stoul(get("cold", "0")) != 0;
    }
    catch(const std::exception&)
    {
        std::fprintf(stderr, "error: malformed numeric option\n%s", USAGE);
        return 2;
    }

    std::vector<std::string> files;
    const std::string list = get("queries", "");
    for(std::size_t b = 0; b < list.size(); )
    {
        const auto e = std::min(list.find(',', b), list.size());
        if(e > b)
            files.push_back(list.substr(b, e - b));
        b = e + 1;
    }

    if(files.empty() || (files.size() > 1 && opt.count("cnt-out")))
    {
        std::fprintf(stderr, "error: %s\n%s", files.empty() ? "no query file" : "--cnt-out takes a single query file; use --cnt-dir", USAGE);
        return 2;
    }

    for(const auto& f : files)  // Before the load, which can take minutes.
        if(!std::filesystem::is_regular_file(f))
        {
            std::fprintf(stderr, "error: no query file %s\n", f.c_str());
            return 1;
        }

    if(cold && load != "mmap")
    {
        std::fprintf(stderr, "error: --cold 1 drops the pages of a mapped table; use --load mmap\n%s", USAGE);
        return 2;
    }

    const std::string dir = get("index", "");
    double t_prewarm = 0, t_lock = 0;
    uint64_t prewarm_bytes = 0, locked_bytes = 0;
    if(warm && load == "mmap")
    {
        const auto t0 = std::chrono::steady_clock::now();
        prewarm_bytes = prewarm(index_files(dir, ""), static_cast<unsigned>(threads));
        t_prewarm = seconds_since(t0);
    }

    // The token arrays read in by every reader at once, and then locked.
    if(lock_text)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const auto text = index_files(dir, "tokenized");
        std::string err;
        locked_bytes = prewarm(text, static_cast<unsigned>(threads));
        if(!lock_files(text, err))
        {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }

        t_lock = seconds_since(t0);
    }

    const auto t0 = std::chrono::steady_clock::now();
    const Engine<U16> engine({dir}, /* eos */ 0, /* vocab */ 65535, /* version */ 4, load == "ram",
                             prefetch ? 1 : 0, prefetch ? 3 : 0, prefetch ? 3 : 0, {}, 512, false, {});
    const double t_load = seconds_since(t0);

    const std::size_t shards = engine.get_num_shards();
    uint64_t tok_cnt = 0;
    for(std::size_t s = 0; s < shards; ++s)
        tok_cnt += engine.get_tok_cnt(s);

    std::printf("index=%s\nshards=%zu\ntok_cnt=%" PRIu64 "\nmode=%s\nload=%s\nprefetch=%d\nthreads=%" PRIu64 "\nprewarm_bytes=%" PRIu64
                "\nt_prewarm=%.3f\nlock_text=%d\nlocked_bytes=%" PRIu64 "\nt_lock=%.3f\ncold=%d\nt_load=%.3f\n",
                dir.c_str(), shards, tok_cnt, mode.c_str(), load.c_str(), prefetch, threads, prewarm_bytes, t_prewarm, lock_text, locked_bytes,
                t_lock, cold, t_load);
    std::fflush(stdout);
    if(tok_cnt == 0)    // The engine finds no shard in a directory without its files, and would count every query as absent.
    {
        std::fprintf(stderr, "error: no infini-gram index in %s\n", dir.c_str());
        return 1;
    }

    // Counts go to `--cnt-out`, or to `<stem>.cnt` of every query file under `--cnt-dir`.
    int rc = 0;
    for(const auto& f : files)
    {
        const std::string cnt = opt.count("cnt-dir") ? (std::filesystem::path(opt["cnt-dir"]) / std::filesystem::path(f).stem()).string() + ".cnt"
                                                     : get("cnt-out", "");
        rc |= count_batch(engine, f, q_max, mode == "find", threads, reps, cold ? index_files(dir, "table") : std::vector<std::string>(), cnt);
    }

    return rc;
}
