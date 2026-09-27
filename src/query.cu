
// Command-line interface of the query engine. Output is `key=value` lines on
// stdout; diagnostics go to stderr. Exits `0` on success.

#include "Query_Engine.hpp"
#include "Index_Meta.hpp"
#include "Suffix_Index.hpp"
#include "query-file.hpp"
#include "utility.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
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
             [--spot N] [--reps K] [--itv-out F] [--cnt-out F] [--cnt-dir D] [--validate]
      Loads the index once, and finds the SA-intervals of the first N (default:
      all) queries of every query file, K times (default 3); reports the load,
      transfer, and search times and the counts' hash. R is `device` (default)
      or `pinned`; --spot sets the suffix-array order spot-checks at load
      (default 100000). --validate checks every interval with the O(m)
      certificate on the device. The counts of a single query file go to
      --cnt-out, or those of each to D/<stem>.cnt.
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
    idx_t hits = 0, occ = 0;
    for(const auto& I : out)
        hits += (I.hi > I.lo), occ += I.hi - I.lo;

    std::printf("reps=%" PRIu64 "\nt_h2d=%.6f\nt_find_min=%.6f\nt_find_median=%.6f\nt_d2h=%.6f\nqps=%.0f\nhits=%" PRIu64 "\noccurrences=%" PRIu64
                "\ncounts_hash=0x%016" PRIx64 "\n",
                reps, t_h2d, sorted.front(), sorted[sorted.size() / 2], t_d2h, q / sorted.front(), hits, occ, query_file::counts_hash(out.data(), q));

    int rc = 0;
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
    uint64_t q_max = std::numeric_limits<uint64_t>::max(), spot = 100000, reps = 3;
    Query_Options opt;
    if(!get_u64(a, "q", q_max, err) || !get_u64(a, "spot", spot, err) || !get_u64(a, "reps", reps, err)
        || !parse_residency(a.get("text-res", "device"), opt.text_res) || !parse_residency(a.get("sa-res", "device"), opt.sa_res))
    {
        std::fprintf(stderr, "error: %s\n", err.empty() ? "unknown residency" : err.c_str());
        return 2;
    }

    std::vector<std::string> files;
    const std::string list = a.get("queries");
    for(std::size_t b = 0; b < list.size(); )
    {
        const auto e = std::min(list.find(',', b), list.size());
        if(e > b)
            files.push_back(list.substr(b, e - b));
        b = e + 1;
    }

    if(files.empty() || (files.size() > 1 && (a.has("cnt-out") || a.has("itv-out"))))
    {
        std::fprintf(stderr, "error: %s\n", files.empty() ? "no query file" : "--cnt-out and --itv-out take a single query file; use --cnt-dir");
        return 2;
    }

    opt.spot = spot;
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

    // Counts go to `--cnt-out`, or to `<stem>.cnt` of every query file under `--cnt-dir`.
    int rc = 0;
    for(const auto& f : files)
    {
        const std::string cnt = a.has("cnt-dir") ? (std::filesystem::path(a.get("cnt-dir")) / std::filesystem::path(f).stem()).string() + ".cnt"
                                                 : a.get("cnt-out");
        rc |= find_batch(eng, f, q_max, reps, a.has("validate"), a.get("itv-out"), cnt);
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
