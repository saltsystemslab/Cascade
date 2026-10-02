
// Command-line interface of the query engine. Output is `key=value` lines on
// stdout; diagnostics go to stderr. Exits `0` on success.

#include "Query_Engine.hpp"
#include "Index_Meta.hpp"
#include "Suffix_Index.hpp"
#include "query-file.hpp"
#include "utility.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>


namespace
{

using namespace gpu_saca;


const char* const USAGE =
R"(Usage:
  query info --index DIR
      Prints the index's metadata and whether it fits the device.
  query find --index DIR --queries Q1.qry[,Q2.qry,...] [--q N] [--text-res R] [--sa-res R]
             [--spot N] [--stride S,...] [--key-words W,...] [--sort 0|1,...] [--gather 0|1,...]
             [--gather-queries G] [--reps K] [--chunk-queries C] [--chunk-symbols Y]
             [--min-chunk-queries C'] [--min-chunk-symbols Y'] [--host-threads H] [--host-lanes L]
             [--itv-out F] [--cnt-out F] [--cnt-dir D] [--validate]
      Loads the index once, and finds the SA-intervals of the first N (default:
      all) queries of every query file, K times (default 3) each way: from
      device memory, and end to end, from host memory through the pipeline, both
      pinned and pageable. Reports the load, transfer, search, and end-to-end
      times, and the counts' hash. The pipeline streams chunks of up to C
      queries (default 2^18, or 2^24 if gathering at load) of up to Y symbols
      (default 2^24, or 2^28) in all, two per stream when their shares of the
      batch are at least C' queries (default 2^16) or Y' symbols (default
      2^22), and whole chunks if gathering; its intervals must equal those
      from device memory. R is `device` (default), `pinned`, or `pageable`;
      --spot sets the suffix-array order spot-checks at load (default 100000).
      The search runs in two phases over a sample table of stride S (default
      32; 0 selects the direct mode) with keys of W words (1, 2, or 4; default
      2, or 4 for 4-byte symbols), with every batch sorted by its packed
      prefixes if --sort 1, and its brackets' suffix-array entries gathered
      into device memory, G queries at a time (default 2^18), if --gather 1
      (default: iff the suffix array is pinned); lists run every combination as
      a mode, on the index loaded once. With both arrays in host memory and
      either pageable, the host searches: the device ranks every chunk in the
      table, and H threads (default: every core) search its brackets, each
      interleaving L searches (default 32); the direct mode runs on the host
      alone, and from device memory means there and back. --validate checks
      every interval with the O(m) certificate on the device, or on the host
      if it searches.
      The counts of a single query file and mode go to --cnt-out, or those of
      each to D/<stem>.cnt, or D/<stem>@s<S>w<W>o<sort>g<gather>.cnt for several
      modes.
  query gen --index DIR --out Q.qry --kind K [--q N --len L1[,L2,...]] [--source F]
            [--vocab V] [--seed S]
      Writes N queries of lengths drawn from L1, L2, ...: K = substring (default)
      takes substrings of the index's text, or of the text F of the same symbols
      (held-out); mutated, substrings with one symbol replaced; random, symbols
      drawn from [0, V); unigram, every symbol of [0, V) once. V defaults to the
      symbol type's range.
)";


// Parsed command line: the command, and its `--key value` options.
struct Args
{
    std::string cmd;
    std::map<std::string, std::string> opt;

    bool has(const std::string& k) const { return opt.count(k) > 0; }
    std::string get(const std::string& k, const std::string& dflt = "") const { return has(k) ? opt.at(k) : dflt; }
};


// Parses the command line into `a`. Returns `false` with a reason in `err` on
// failure.
bool parse(const int argc, char** const argv, Args& a, std::string& err)
{
    static const std::set<std::string> flags = {"validate"};
    if(argc < 2)
    {
        err = "missing command";
        return false;
    }

    a.cmd = argv[1];
    for(int i = 2; i < argc; ++i)
    {
        const std::string k = argv[i];
        if(k.size() < 3 || k.compare(0, 2, "--") != 0)  { err = "unexpected argument `" + k + "`"; return false; }
        const std::string key = k.substr(2);
        if(flags.count(key))
            a.opt[key] = "1";
        else if(i + 1 < argc)
            a.opt[key] = argv[++i];
        else
        {
            err = "option " + k + " needs a value";
            return false;
        }
    }

    return true;
}


// Parses the unsigned integer option `k` of `a` into `v`, keeping `v` if absent.
// Returns `false` with a reason in `err` if malformed.
bool get_u64(const Args& a, const std::string& k, uint64_t& v, std::string& err)
{
    if(!a.has(k))
        return true;

    try
    {
        std::size_t used = 0;
        v = std::stoull(a.get(k), &used);
        if(used == a.get(k).size())
            return true;
    }
    catch(const std::exception&) {}

    err = "option --" + k + " expects an unsigned integer";
    return false;
}


// Parses the comma-separated unsigned integers of the option `k` of `a` into
// `v`, keeping `v` if absent. Returns `false` with a reason in `err` if
// malformed.
bool get_u64_list(const Args& a, const std::string& k, std::vector<uint64_t>& v, std::string& err)
{
    if(!a.has(k))
        return true;

    v.clear();
    const std::string l = a.get(k);
    try
    {
        for(std::size_t b = 0; b <= l.size(); )
        {
            const auto e = std::min(l.find(',', b), l.size());
            std::size_t used = 0;
            v.push_back(std::stoull(l.substr(b, e - b), &used));
            if(used != e - b)
                throw std::invalid_argument(k);
            b = e + 1;
        }

        return !v.empty();
    }
    catch(const std::exception&) {}

    err = "option --" + k + " expects comma-separated unsigned integers";
    return false;
}


// Returns the seconds between the events `a` and `b`.
double seconds(const cudaEvent_t a, const cudaEvent_t b)
{
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / 1e3;
}


int run_info(const Args& a)
{
    Index_Meta m;
    std::string err;
    if(!index_meta::read(a.get("index"), m, err) || !index_meta::validate_files(m, err))
    {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    std::size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const idx_t text_b = m.n * m.sym_bytes, need = text_b + TEXT_PAD_BYTES + m.sa_bytes() + SA_PAD_BYTES;

    std::printf("index=%s\nn=%" PRIu64 "\nsym_bytes=%u\nsa_order=%s\ntext=%s\ntext_bytes=%" PRIu64 "\nsa_bytes=%" PRIu64 "\nsa_parts=%zu\n",
                a.get("index").c_str(), m.n, m.sym_bytes, m.sa_order.c_str(), m.text.c_str(), text_b, m.sa_bytes(), m.sa_part.size());
    for(std::size_t i = 0; i < m.sa_part.size(); ++i)
        std::printf("sa_part_%zu=%s\n", i, m.sa_part[i].c_str());
    std::printf("sep=%" PRId64 "\nsa_source=%s\ncreated_utc=%s\ndevice_free_bytes=%zu\ndevice_total_bytes=%zu\nfits_device=%d\n",
                m.sep, m.sa_source.c_str(), m.created_utc.c_str(), free_b, total_b, need <= free_b);
    return 0;
}


// Finds the SA-intervals of the first `q_max` queries of the file `path` with
// `eng`, `reps` times, and prints the batch's `key=value` lines. Checks the
// intervals if `validate`, and writes them and their counts to `itv_path` and
// `cnt_path` unless empty. Returns the exit status.
template <typename T_sym_>
int find_batch(const Query_Engine<T_sym_>& eng, const std::string& path, const uint64_t q_max, const uint64_t reps, const bool validate,
               const std::string& itv_path, const std::string& cnt_path)
{
    std::vector<uint64_t> off;
    std::vector<T_sym_> sym;
    Query_File_Header h;
    std::string err;
    if(!query_file::read(path, q_max, off, sym, h, err))
    {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    const idx_t q = off.size() - 1;
    std::printf("queries=%s\nq=%" PRIu64 "\nsym_c=%" PRIu64 "\n", path.c_str(), q, off[q]);

    uint64_t* off_d = nullptr;
    T_sym_* sym_d = nullptr;
    SA_Interval* out_d = nullptr;
    CUDA_CHECK(cudaMalloc(&off_d, (q + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&sym_d, std::max<idx_t>(off[q], 1) * sizeof(T_sym_)));
    CUDA_CHECK(cudaMalloc(&out_d, std::max<idx_t>(q, 1) * sizeof(SA_Interval)));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(off_d, off.data(), (q + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sym_d, sym.data(), off[q] * sizeof(T_sym_), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const double t_h2d = seconds(e0, e1);

    std::vector<double> t_find;
    for(uint64_t r = 0; r < reps; ++r)
    {
        CUDA_CHECK(cudaEventRecord(e0));
        eng.find_async(off_d, sym_d, q, out_d, 0);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        t_find.push_back(seconds(e0, e1));
    }

    std::vector<SA_Interval> out(q);
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(out.data(), out_d, q * sizeof(SA_Interval), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const double t_d2h = seconds(e0, e1);

    auto sorted = t_find;
    std::sort(sorted.begin(), sorted.end());

    // End to end: from the batch in host memory to its intervals in host memory, through the pipeline; from pinned memory, and pageable.
    uint64_t* off_p = nullptr;
    T_sym_* sym_p = nullptr;
    SA_Interval* out_p = nullptr;
    CUDA_CHECK(pinned_allocate_host(off_p, q + 1));
    CUDA_CHECK(pinned_allocate_host(sym_p, std::max<idx_t>(off[q], 1)));
    CUDA_CHECK(pinned_allocate_host(out_p, std::max<idx_t>(q, 1)));
    std::memcpy(off_p, off.data(), (q + 1) * sizeof(uint64_t));
    if(off[q] > 0)
        std::memcpy(sym_p, sym.data(), off[q] * sizeof(T_sym_));

    const auto time_find = [&](const uint64_t* const o, const T_sym_* const y, SA_Interval* const r)
    {
        std::vector<double> t;
        for(uint64_t k = 0; k < reps; ++k)
        {
            const auto t0 = std::chrono::steady_clock::now();
            eng.find(o, y, q, r);
            t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }

        std::sort(t.begin(), t.end());
        return t;
    };

    std::vector<SA_Interval> out_pg(q);
    const auto t_e2e = time_find(off_p, sym_p, out_p);
    const auto t_e2e_pg = time_find(off.data(), sym.data(), out_pg.data());
    const bool e2e_same = (q == 0 || (std::memcmp(out_p, out.data(), q * sizeof(SA_Interval)) == 0
                                      && std::memcmp(out_pg.data(), out.data(), q * sizeof(SA_Interval)) == 0));
    CUDA_CHECK(deallocate_host(off_p));
    CUDA_CHECK(deallocate_host(sym_p));
    CUDA_CHECK(deallocate_host(out_p));

    idx_t hits = 0, occ = 0;
    for(const auto& I : out)
        hits += (I.hi > I.lo), occ += I.hi - I.lo;

    std::printf("reps=%" PRIu64 "\nt_h2d=%.6f\nt_find_min=%.6f\nt_find_median=%.6f\nt_d2h=%.6f\nqps=%.0f\nt_e2e_min=%.6f\nt_e2e_median=%.6f"
                "\nqps_e2e=%.0f\nt_e2e_pageable_min=%.6f\nt_e2e_pageable_median=%.6f\nqps_e2e_pageable=%.0f\ne2e_check=%s\nhits=%" PRIu64
                "\noccurrences=%" PRIu64 "\ncounts_hash=0x%016" PRIx64 "\n",
                reps, t_h2d, sorted.front(), sorted[sorted.size() / 2], t_d2h, q / sorted.front(), t_e2e.front(), t_e2e[t_e2e.size() / 2],
                q / t_e2e.front(), t_e2e_pg.front(), t_e2e_pg[t_e2e_pg.size() / 2], q / t_e2e_pg.front(), e2e_same ? "pass" : "FAIL", hits, occ,
                query_file::counts_hash(out.data(), q));

    int rc = 0;
    if(!e2e_same)
    {
        std::fprintf(stderr, "error: %s: the pipeline's intervals differ from those found from device memory\n", path.c_str());
        rc = 1;
    }

    if(validate)
    {
        unsigned long long* flags_d = nullptr;
        unsigned long long flags[2] = {0, q};  // Failing count, and the first failing query.
        CUDA_CHECK(cudaMalloc(&flags_d, sizeof(flags)));
        CUDA_CHECK(cudaMemcpy(flags_d, flags, sizeof(flags), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e0));
        eng.certify_async(off_d, sym_d, q, out_d, flags_d, flags_d + 1, 0);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaMemcpy(flags, flags_d, sizeof(flags), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(flags_d));
        std::printf("validate=%s\nvalidate_failed=%llu\nt_validate=%.6f\n", flags[0] == 0 ? "pass" : "FAIL", flags[0], seconds(e0, e1));
        if(flags[0] != 0)
        {
            std::fprintf(stderr, "error: %s: %llu intervals fail the certificate; the first is query %llu\n", path.c_str(), flags[0], flags[1]);
            rc = 1;
        }
    }

    if(!itv_path.empty() && !query_file::write_intervals(itv_path, out.data(), q))
    {
        std::fprintf(stderr, "error: cannot write %s\n", itv_path.c_str());
        rc = 1;
    }

    if(!cnt_path.empty() && !query_file::write_counts(cnt_path, out.data(), q))
    {
        std::fprintf(stderr, "error: cannot write %s\n", cnt_path.c_str());
        rc = 1;
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(off_d));
    CUDA_CHECK(cudaFree(sym_d));
    CUDA_CHECK(cudaFree(out_d));
    return rc;
}


template <typename T_sym_>
int run_find(const Args& a)
{
    std::string err;
    Query_Options opt;
    uint64_t q_max = std::numeric_limits<uint64_t>::max(), spot = 100000, reps = 3, host_threads = opt.host_threads, host_lanes = opt.host_lanes;
    std::vector<uint64_t> strides = {opt.stride}, key_words = {0}, sorts = {0}, gathers;
    if(!get_u64(a, "q", q_max, err) || !get_u64(a, "spot", spot, err) || !get_u64(a, "reps", reps, err)
        || !get_u64(a, "chunk-queries", opt.chunk_queries, err) || !get_u64(a, "chunk-symbols", opt.chunk_symbols, err)
        || !get_u64(a, "min-chunk-queries", opt.min_chunk_queries, err) || !get_u64(a, "min-chunk-symbols", opt.min_chunk_symbols, err)
        || !get_u64(a, "gather-queries", opt.gather_queries, err)
        || !get_u64(a, "host-threads", host_threads, err) || !get_u64(a, "host-lanes", host_lanes, err)
        || !get_u64_list(a, "stride", strides, err) || !get_u64_list(a, "key-words", key_words, err) || !get_u64_list(a, "sort", sorts, err)
        || !get_u64_list(a, "gather", gathers, err)
        || !parse_residency(a.get("text-res", "device"), opt.text_res) || !parse_residency(a.get("sa-res", "device"), opt.sa_res))
    {
        std::fprintf(stderr, "error: %s\n", err.empty() ? "unknown residency" : err.c_str());
        return 2;
    }

    // Every combination of the listed strides, key widths, orders, and gathers is a mode; the direct mode ignores the key width and
    // the gather.
    std::vector<Gather> gather = {Gather::automatic};
    if(!gathers.empty())
    {
        gather.clear();
        for(const auto x : gathers)
            gather.push_back(x != 0 ? Gather::always : Gather::never);
    }

    struct Mode { idx_t stride; uint32_t key_words; bool sort; Gather gather; };
    std::vector<Mode> modes;
    for(const auto st : strides)
        for(const auto w : (st == 0 ? std::vector<uint64_t>{0} : key_words))
            for(const auto so : sorts)
                for(const auto ga : (st == 0 ? std::vector<Gather>{gather.front()} : gather))
                    modes.push_back({st, static_cast<uint32_t>(w), so != 0, ga});

    opt.stride = modes.front().stride;
    opt.key_words = modes.front().key_words;
    opt.sort = modes.front().sort;
    opt.gather = modes.front().gather;

    std::vector<std::string> files;
    const std::string list = a.get("queries");
    for(std::size_t b = 0; b < list.size(); )
    {
        const auto e = std::min(list.find(',', b), list.size());
        if(e > b)
            files.push_back(list.substr(b, e - b));
        b = e + 1;
    }

    if(files.empty() || ((files.size() > 1 || modes.size() > 1) && (a.has("cnt-out") || a.has("itv-out"))))
    {
        std::fprintf(stderr, "error: %s\n", files.empty() ? "no query file" : "--cnt-out and --itv-out take a single query file and mode; use --cnt-dir");
        return 2;
    }

    opt.spot = spot;
    opt.host_threads = static_cast<unsigned>(std::min<uint64_t>(host_threads, std::numeric_limits<unsigned>::max()));
    opt.host_lanes = static_cast<unsigned>(std::min<uint64_t>(host_lanes, std::numeric_limits<unsigned>::max()));   // Too many is rejected at load.
    reps = std::max<uint64_t>(reps, 1);

    Query_Engine<T_sym_> eng;
    if(!eng.load(a.get("index"), opt, err))
    {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    const auto& idx = eng.index();
    const auto& st = idx.stats();
    std::printf("index=%s\nn=%" PRIu64 "\nsym_bytes=%u\ntext_res=%s\nsa_res=%s\nt_load_text=%.3f\nt_load_sa=%.3f\nspot_pairs=%" PRIu64 "\nt_spot=%.3f\n",
                a.get("index").c_str(), idx.n(), idx.sym_bytes(), residency_name(idx.text_residency()), residency_name(idx.sa_residency()),
                st.t_text, st.t_sa, st.spot_pairs, st.t_spot);
    std::printf("streams=%zu\nchunk_queries=%" PRIu64 "\nchunk_symbols=%" PRIu64 "\nmin_chunk_queries=%" PRIu64 "\nmin_chunk_symbols=%" PRIu64
                "\ngather_queries=%" PRIu64 "\n", eng.streams(), eng.options().chunk_queries, eng.options().chunk_symbols,
                eng.options().min_chunk_queries, eng.options().min_chunk_symbols, eng.options().gather_queries);
    std::printf("host_search=%d\nhost_threads=%u\nhost_lanes=%u\n", eng.host_searches(), eng.options().host_threads, eng.options().host_lanes);

    // Counts go to `--cnt-out`, or to `<stem>.cnt` of every query file under `--cnt-dir`; to `<stem>@<mode>.cnt` if several modes.
    int rc = 0;
    for(std::size_t k = 0; k < modes.size(); ++k)
    {
        const auto& md = modes[k];
        if(k > 0 && !eng.configure(md.stride, md.key_words, md.sort, md.gather, err))
        {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }

        const auto& tab = eng.table();
        const int gathered = (tab.built() && eng.options().gather == Gather::always);   // The direct mode has no brackets.
        std::printf("stride=%" PRIu64 "\nkey_words=%u\nkey_symbols=%" PRIu64 "\nsort=%d\ngather=%d\ntable_bytes=%zu\nt_table=%.3f\n",
                    tab.stride(), tab.key_words(), tab.key_symbols(), eng.options().sort, gathered, tab.bytes(), tab.t_build());
        const std::string tag = (modes.size() > 1 ? "@s" + std::to_string(tab.stride()) + "w" + std::to_string(tab.key_words())
                                                    + "o" + std::to_string(eng.options().sort) + "g" + std::to_string(gathered) : "");
        for(const auto& f : files)
        {
            const std::string cnt = a.has("cnt-dir") ? (std::filesystem::path(a.get("cnt-dir")) / std::filesystem::path(f).stem()).string() + tag + ".cnt"
                                                     : a.get("cnt-out");
            rc |= find_batch(eng, f, q_max, reps, a.has("validate"), a.get("itv-out"), cnt);
        }
    }

    return rc;
}


template <typename T_sym_>
int run_gen(const Args& a)
{
    std::string err;
    uint64_t q = 0, seed = 1, vocab = uint64_t(std::numeric_limits<T_sym_>::max()) + 1;
    std::vector<uint64_t> lens;
    const std::string kind = a.get("kind", "substring");
    try
    {
        const std::string l = a.get("len");
        for(std::size_t b = 0; b < l.size(); )
        {
            const auto e = std::min(l.find(',', b), l.size());
            lens.push_back(std::stoull(l.substr(b, e - b)));
            b = e + 1;
        }
    }
    catch(const std::exception&) { lens.clear(); }

    const bool unigram = (kind == "unigram"), from_text = (kind == "substring" || kind == "mutated");
    if(!get_u64(a, "q", q, err) || !get_u64(a, "seed", seed, err) || !get_u64(a, "vocab", vocab, err) || !a.has("out")
        || (kind != "substring" && kind != "mutated" && kind != "random" && !unigram) || vocab == 0
        || (!unigram && (q == 0 || lens.empty())))
    {
        std::fprintf(stderr, "error: %s\n", err.empty() ? "gen needs --out, --kind substring|mutated|random|unigram, and --q > 0 and --len unless unigram"
                                                        : err.c_str());
        return 2;
    }

    Index_Meta m;
    if(!index_meta::read(a.get("index"), m, err) || !index_meta::validate_files(m, err))
    {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    // The text to draw substrings from: the index's, or a held-out one of the same symbols.
    const std::string source = a.get("source", m.text);
    std::vector<T_sym_> T;
    if(from_text)
    {
        std::error_code ec;
        const auto bytes = std::filesystem::file_size(source, ec);
        if(ec || bytes == 0 || bytes % sizeof(T_sym_) != 0)
        {
            std::fprintf(stderr, "error: cannot use %s as a text of %zu-byte symbols\n", source.c_str(), sizeof(T_sym_));
            return 1;
        }

        T.resize(bytes / sizeof(T_sym_));
        if(!index_io::read_range(source, 0, bytes, reinterpret_cast<uint8_t*>(T.data()), err))
        {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
    }

    std::mt19937_64 rng(seed);
    const auto symbol = [&] { return static_cast<T_sym_>(rng() % vocab); };
    std::vector<uint64_t> off = {0};
    std::vector<T_sym_> sym;
    if(unigram)
    {
        q = vocab;
        for(uint64_t c = 0; c < vocab; ++c)
        {
            sym.push_back(static_cast<T_sym_>(c));
            off.push_back(sym.size());
        }
    }
    else
        for(uint64_t i = 0; i < q; ++i)
        {
            const uint64_t len = (from_text ? std::min<uint64_t>(lens[rng() % lens.size()], T.size()) : lens[rng() % lens.size()]);
            if(from_text)
            {
                const uint64_t pos = rng() % (T.size() - len + 1);
                sym.insert(sym.end(), T.begin() + pos, T.begin() + pos + len);
                if(kind == "mutated" && len > 0 && vocab > 1)  // One symbol replaced by a different one.
                {
                    T_sym_& x = sym[sym.size() - len + rng() % len];
                    const T_sym_ old = x;
                    while((x = symbol()) == old) {}
                }
            }
            else
                for(uint64_t j = 0; j < len; ++j)
                    sym.push_back(symbol());

            off.push_back(sym.size());
        }

    if(!query_file::write(a.get("out"), off, sym, seed, err))
    {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    std::printf("out=%s\nq=%" PRIu64 "\nsym_c=%zu\nkind=%s\nsource=%s\nvocab=%" PRIu64 "\nseed=%" PRIu64 "\n", a.get("out").c_str(), q,
                sym.size(), kind.c_str(), from_text ? source.c_str() : "-", vocab, seed);
    return 0;
}

}


int main(const int argc, char** const argv)
{
    Args a;
    std::string err;
    if(!parse(argc, argv, a, err) || !a.has("index") || (a.cmd == "find" && !a.has("queries")))
    {
        std::fprintf(stderr, "%s%s\n%s", err.empty() ? "" : "error: ", err.c_str(), USAGE);
        return 2;
    }

    if(a.cmd == "info")
        return run_info(a);

    if(a.cmd != "find" && a.cmd != "gen")
    {
        std::fprintf(stderr, "error: unknown command `%s`\n%s", a.cmd.c_str(), USAGE);
        return 2;
    }

    Index_Meta m;
    if(!index_meta::read(a.get("index"), m, err))
    {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    switch(m.sym_bytes)
    {
    case 1: return a.cmd == "find" ? run_find<uint8_t>(a) : run_gen<uint8_t>(a);
    case 2: return a.cmd == "find" ? run_find<uint16_t>(a) : run_gen<uint16_t>(a);
    default:    return a.cmd == "find" ? run_find<uint32_t>(a) : run_gen<uint32_t>(a);
    }
}
