
// Baseline query driver: infini-gram's unmodified engine (`cpp_engine.h`,
// v2.6.0) counting a batch of `.qry` queries over an infini-gram index, from a
// pool of threads. Output is `key=value` lines on stdout, as `query find` prints;
// `--cnt-out` writes the counts, byte-comparable with `query find --cnt-out`.
//
// Usage: bench_infinigram_query --index DIR --queries Q.qry [--q N] [--mode find|direct]
//            [--threads T] [--load mmap|ram] [--prefetch 0|1] [--prewarm 0|1] [--reps K]
//            [--cnt-out F]
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

#include "query-file.hpp"

#include "cpp_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>


namespace
{

const char* const USAGE =
R"(Usage: bench_infinigram_query --index DIR --queries Q.qry [--q N] [--mode find|direct]
           [--threads T] [--load mmap|ram] [--prefetch 0|1] [--prewarm 0|1] [--reps K]
           [--cnt-out F]
)";

constexpr std::size_t CHUNK = 256;  // Queries a pool thread takes at a time.


double seconds_since(const std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


// Reads every regular file of `dir`, following symlinks, through the page cache
// with `threads` parallel readers. Returns the bytes read.
uint64_t prewarm(const std::string& dir, const unsigned threads)
{
    constexpr std::size_t PIECE = std::size_t(8) << 20;
    uint64_t total = 0;
    for(const auto& e : std::filesystem::directory_iterator(dir))
    {
        const auto path = std::filesystem::canonical(e.path());
        if(!std::filesystem::is_regular_file(path))
            continue;

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
    static const std::set<std::string> known = {"index", "queries", "q", "mode", "threads", "load", "prefetch", "prewarm", "reps", "cnt-out"};
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
    bool prefetch = false, warm = true;
    try
    {
        if(opt.count("q"))          q_max = std::stoull(opt["q"]);
        if(opt.count("threads"))    threads = std::max<uint64_t>(1, std::stoull(opt["threads"]));
        if(opt.count("reps"))       reps = std::max<uint64_t>(1, std::stoull(opt["reps"]));
        prefetch = std::stoul(get("prefetch", "0")) != 0;
        warm = std::stoul(get("prewarm", "1")) != 0;
    }
    catch(const std::exception&)
    {
        std::fprintf(stderr, "error: malformed numeric option\n%s", USAGE);
        return 2;
    }

    std::vector<uint64_t> off;
    std::vector<U16> sym;
    gpu_saca::Query_File_Header h;
    std::string err;
    if(!gpu_saca::query_file::read(get("queries", ""), q_max, off, sym, h, err))
    {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    const uint64_t q = off.size() - 1;
    const std::string dir = get("index", "");

    double t_prewarm = 0;
    uint64_t prewarm_bytes = 0;
    if(warm && load == "mmap")
    {
        const auto t0 = std::chrono::steady_clock::now();
        prewarm_bytes = prewarm(dir, static_cast<unsigned>(threads));
        t_prewarm = seconds_since(t0);
    }

    auto t0 = std::chrono::steady_clock::now();
    const Engine<U16> engine({dir}, /* eos */ 0, /* vocab */ 65535, /* version */ 4, load == "ram",
                             prefetch ? 1 : 0, prefetch ? 3 : 0, prefetch ? 3 : 0, {}, 512, false, {});
    const double t_load = seconds_since(t0);

    const std::size_t shards = engine.get_num_shards();
    uint64_t tok_cnt = 0;
    for(std::size_t s = 0; s < shards; ++s)
        tok_cnt += engine.get_tok_cnt(s);

    std::printf("index=%s\nshards=%zu\ntok_cnt=%" PRIu64 "\nmode=%s\nload=%s\nprefetch=%d\nthreads=%" PRIu64 "\nprewarm_bytes=%" PRIu64
                "\nt_prewarm=%.3f\nt_load=%.3f\nqueries=%s\nq=%" PRIu64 "\nsym_c=%" PRIu64 "\n",
                dir.c_str(), shards, tok_cnt, mode.c_str(), load.c_str(), prefetch, threads, prewarm_bytes, t_prewarm, t_load,
                get("queries", "").c_str(), q, off[q]);

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
                        if(mode == "find")
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
    for(uint64_t r = 0; r < reps; ++r)
    {
        t0 = std::chrono::steady_clock::now();
        run();
        t_run.push_back(seconds_since(t0));
    }

    const double t_first = t_run.front();
    std::vector<double> timed(t_run.begin() + (reps > 1 ? 1 : 0), t_run.end());    // The first run warms up.
    std::sort(timed.begin(), timed.end());
    uint64_t hits = 0, occ = 0;
    for(const auto c : cnt)
        hits += (c > 0), occ += c;

    std::printf("reps=%" PRIu64 "\nt_first=%.6f\nt_find_min=%.6f\nt_find_median=%.6f\nqps=%.0f\nhits=%" PRIu64 "\noccurrences=%" PRIu64
                "\ncounts_hash=0x%016" PRIx64 "\n",
                reps, t_first, timed.front(), timed[timed.size() / 2], q / timed.front(), hits, occ,
                gpu_saca::query_file::counts_hash(cnt.data(), q));

    if(opt.count("cnt-out") && !gpu_saca::query_file::write_counts(opt["cnt-out"], cnt.data(), q))
    {
        std::fprintf(stderr, "error: cannot write %s\n", opt["cnt-out"].c_str());
        return 1;
    }

    return 0;
}
