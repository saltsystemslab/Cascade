
// Test runner of the query engine: fixtures across symbol widths, edge cases,
// and oracles, host-side and on the device. Exits `0` iff every test passes;
// the device tests are skipped without a device.

#include "CPU_Reference.hpp"
#include "Index_Meta.hpp"
#include "Query_Engine.hpp"
#include "Suffix_Index.hpp"
#include "query-file.hpp"
#include "query_tests.hpp"
#include "sa40.hpp"
#include "utility.cuh"

#include <cuda_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>


namespace
{

using namespace gpu_saca;


// Reports the failure of `cond`, and fails the current test.
#define CHECK(cond, ...) \
do \
{ \
    if(!(cond)) \
    { \
        std::fprintf(stderr, "    check failed at %s:%d: %s. ", __FILE__, __LINE__, #cond); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        return false; \
    } \
} \
while(0)


// Returns a fresh scratch directory for a test.
std::filesystem::path scratch_dir(const std::string& name)
{
    const auto dir = std::filesystem::temp_directory_path() / ("gsa-test-query-" + std::to_string(::getpid())) / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}


// Returns `true` iff the intervals `a` and `b` are identical.
bool same(const SA_Interval& a, const SA_Interval& b)
{
    return a.lo == b.lo && a.hi == b.hi;
}


bool test_sa40_round_trip()
{
    std::vector<idx_t> v = {0, 1, 255, 256, 65535, 65536, (idx_t(1) << 32) - 1, idx_t(1) << 32, (idx_t(1) << 32) + 1,
                            SA_MAX_N - 2, SA_MAX_N - 1};
    std::mt19937_64 rng(1);
    for(int i = 0; i < 10000; ++i)
        v.push_back(rng() & SA_ENTRY_MASK);

    std::vector<uint8_t> buf(SA_WIDTH * v.size() + SA_PAD_BYTES, 0xAB);
    for(idx_t r = 0; r < v.size(); ++r)
        sa40::store(buf.data(), r, v[r]);

    for(idx_t r = 0; r < v.size(); ++r)
        CHECK(sa40::load(buf.data(), r) == v[r], "rank %" PRIu64 ": got %" PRIu64 ", expected %" PRIu64, r, sa40::load(buf.data(), r), v[r]);

    return true;
}


bool test_query_file_round_trip()
{
    const auto dir = scratch_dir("qry");
    const std::string path = (dir / "q.qry").string();

    std::mt19937_64 rng(2);
    std::vector<uint64_t> off = {0};
    std::vector<uint16_t> sym;
    for(int i = 0; i < 1000; ++i)
    {
        const auto len = rng() % 50;    // Includes empty queries.
        for(uint64_t k = 0; k < len; ++k)
            sym.push_back(static_cast<uint16_t>(rng()));
        off.push_back(sym.size());
    }

    std::string err;
    CHECK(query_file::write(path, off, sym, 7, err), "%s", err.c_str());

    std::vector<uint64_t> off_r;
    std::vector<uint16_t> sym_r;
    Query_File_Header h;
    CHECK(query_file::read(path, UINT64_MAX, off_r, sym_r, h, err), "%s", err.c_str());
    CHECK(off_r == off && sym_r == sym && h.q_c == 1000 && h.seed == 7, "full read differs");

    // A batch of the first `q` queries is a prefix of the file.
    CHECK(query_file::read(path, 17, off_r, sym_r, h, err), "%s", err.c_str());
    CHECK(off_r.size() == 18 && std::equal(off_r.begin(), off_r.end(), off.begin()), "prefix offsets differ");
    CHECK(std::equal(sym_r.begin(), sym_r.end(), sym.begin()) && sym_r.size() == off[17], "prefix symbols differ");

    // The symbol width is checked.
    std::vector<uint32_t> sym_w;
    CHECK(!query_file::read(path, UINT64_MAX, off_r, sym_w, h, err), "a width mismatch must be rejected");

    // A corrupt magic is rejected.
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(0);
        f.put('X');
    }
    CHECK(!query_file::read(path, UINT64_MAX, off_r, sym_r, h, err), "a bad magic must be rejected");

    // Counts and their hash.
    const std::vector<SA_Interval> I = {{0, 3}, {5, 5}, {7, 100}};
    const std::string cnt = (dir / "c.cnt").string();
    CHECK(query_file::write_counts(cnt, I.data(), I.size()), "cannot write counts");
    std::ifstream is(cnt, std::ios::binary);
    std::vector<uint64_t> c(3);
    is.read(reinterpret_cast<char*>(c.data()), 24);
    CHECK(is && c[0] == 3 && c[1] == 0 && c[2] == 93, "counts file content differs");

    // FNV-1a over the bytes of the `.cnt` content.
    uint64_t h_ref = 0xcbf29ce484222325ull;
    for(const auto x : c)
        for(int b = 0; b < 8; ++b)
            h_ref = (h_ref ^ ((x >> (8 * b)) & 0xFF)) * 0x100000001b3ull;
    CHECK(query_file::counts_hash(I.data(), I.size()) == h_ref, "counts hash differs from the reference");

    return true;
}


bool test_index_meta_round_trip()
{
    const auto dir = scratch_dir("meta");

    Index_Meta m;
    m.n = 12345;
    m.sym_bytes = 2;
    m.text = (dir / "text.u16").string();
    m.sa_part_off = index_meta::part_offsets(m.sa_bytes(), 2);
    m.sa_part = {(dir / "sa.u40.part0").string(), (dir / "sa.u40.part1").string()};
    m.sep = 65535;
    m.sa_source = "unit test";

    CHECK(m.sa_part_off[0] == 0 && m.sa_part_off[1] % SA_PAGE_BYTES == 0 && m.sa_part_off[1] < m.sa_bytes(), "bad part offsets");

    std::string err;
    CHECK(index_meta::write(dir.string(), m, err), "%s", err.c_str());

    Index_Meta r;
    CHECK(index_meta::read(dir.string(), r, err), "%s", err.c_str());
    CHECK(r.n == m.n && r.sym_bytes == 2 && r.text == m.text && r.sa_part == m.sa_part && r.sa_part_off == m.sa_part_off
          && r.sep == 65535 && r.sa_source == m.sa_source, "metadata differs after the round-trip");

    // The files are checked against the metadata.
    CHECK(!index_meta::validate_files(r, err), "missing files must be rejected");
    for(const auto& p : {r.text, r.sa_part[0], r.sa_part[1]})
        std::ofstream(p, std::ios::binary).close();
    std::filesystem::resize_file(r.text, m.n * 2);
    for(std::size_t i = 0; i < 2; ++i)
        std::filesystem::resize_file(r.sa_part[i], r.part_bytes(i));
    CHECK(index_meta::validate_files(r, err), "%s", err.c_str());
    std::filesystem::resize_file(r.sa_part[1], r.part_bytes(1) + 1);
    CHECK(!index_meta::validate_files(r, err), "a mis-sized part must be rejected");

    // Inconsistent metadata is rejected.
    Index_Meta bad = m;
    bad.sa_part_off[1] += 1;
    CHECK(!index_meta::validate(bad, err), "unaligned part offsets must be rejected");
    bad = m;
    bad.sa_order = "bytes";
    CHECK(!index_meta::validate(bad, err), "an unsupported order must be rejected");

    return true;
}


// Returns a random text of `n` symbols over the symbols `alpha`, with the
// separator `sep` placed every ~`doc_len` symbols if `doc_len > 0`.
template <typename T_sym_>
std::vector<T_sym_> random_text(const idx_t n, const std::vector<T_sym_>& alpha, const T_sym_ sep, const idx_t doc_len, std::mt19937_64& rng)
{
    std::vector<T_sym_> T(n);
    for(idx_t i = 0; i < n; ++i)
        T[i] = (doc_len > 0 && rng() % doc_len == 0) ? sep : alpha[rng() % alpha.size()];

    return T;
}


// Returns in `off` and `sym` a batch of `count` fixture queries over the text `T`
// of the symbols `alpha`: substrings, possibly running past the text's end,
// random patterns, mutated substrings, and zero-tails.
template <typename T_sym_>
void fixture_queries(const std::vector<T_sym_>& T, const std::vector<T_sym_>& alpha, std::mt19937_64& rng, const int count,
                     std::vector<uint64_t>& off, std::vector<T_sym_>& sym)
{
    const idx_t n = T.size();
    off.assign(1, 0);
    sym.clear();
    const auto push = [&](const std::vector<T_sym_>& P) { sym.insert(sym.end(), P.begin(), P.end()); off.push_back(sym.size()); };

    const idx_t lens[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100};
    for(int k = 0; k < count; ++k)
    {
        const idx_t m = lens[rng() % (sizeof(lens) / sizeof(lens[0]))];
        const idx_t pos = rng() % n;
        std::vector<T_sym_> P;

        switch(k % 5)
        {
        case 0: // A substring; may run past the text's end, then it is padded with random symbols.
        case 1:
            for(idx_t i = 0; i < m; ++i)
                P.push_back(pos + i < n ? T[pos + i] : alpha[rng() % alpha.size()]);
            break;

        case 2: // A random pattern.
            for(idx_t i = 0; i < m; ++i)
                P.push_back(alpha[rng() % alpha.size()]);
            break;

        case 3: // A mutated substring.
            for(idx_t i = 0; i < m; ++i)
                P.push_back(pos + i < n ? T[pos + i] : alpha[rng() % alpha.size()]);
            if(m > 0)
                P[rng() % m] = alpha[rng() % alpha.size()];
            break;

        default:    // A suffix of the text extended with zeros: the zero-tail shape.
            {
                const idx_t tail = std::min<idx_t>(n, 1 + rng() % 9);
                const idx_t zeros = 1 + rng() % 9;
                for(idx_t i = n - tail; i < n; ++i)
                    P.push_back(T[i]);
                for(idx_t z = 0; z < zeros; ++z)
                    P.push_back(0);
            }
        }

        push(P);
    }
}


// Checks the oracle `ref::find_one`, the certificate, and the batch validator on
// a fixture text `T`, against the suffix-array-free `brute_find`.
template <typename T_sym_>
bool check_fixture(const std::vector<T_sym_>& T, const std::vector<T_sym_>& alpha, const uint64_t seed, const char* const what)
{
    const idx_t n = T.size();
    const auto SA = test::build_sa40(T);

    std::string err;
    const auto sa = [&](const idx_t r) { return sa40::load(SA.data(), r); };
    CHECK(ref::spot_check_sa_order(T.data(), n, sa, n, n, seed, err), "%s: %s", what, err.c_str());

    std::mt19937_64 rng(seed);
    std::vector<uint64_t> off;
    std::vector<T_sym_> sym;
    fixture_queries(T, alpha, rng, 400, off, sym);

    const idx_t q = off.size() - 1;
    std::vector<SA_Interval> I(q);
    for(idx_t i = 0; i < q; ++i)
    {
        const T_sym_* const P = sym.data() + off[i];
        const idx_t m = off[i + 1] - off[i];
        I[i] = ref::find_one(T.data(), n, SA.data(), P, m);
        const auto B = test::brute_find(T.data(), n, P, m);
        CHECK(same(I[i], B), "%s: query %" PRIu64 " (m = %" PRIu64 "): oracle [%" PRIu64 ", %" PRIu64 ") vs brute [%" PRIu64 ", %" PRIu64 ")",
              what, i, m, I[i].lo, I[i].hi, B.lo, B.hi);
        CHECK(ref::certify(T.data(), n, SA.data(), P, m, I[i]), "%s: query %" PRIu64 ": certificate rejects the true interval", what, i);

        // Any other interval must be rejected.
        if(I[i].lo > 0)
            CHECK(!ref::certify(T.data(), n, SA.data(), P, m, SA_Interval{I[i].lo - 1, I[i].hi}), "%s: query %" PRIu64 ": lo - 1 accepted", what, i);
        if(I[i].hi < n)
            CHECK(!ref::certify(T.data(), n, SA.data(), P, m, SA_Interval{I[i].lo, I[i].hi + 1}), "%s: query %" PRIu64 ": hi + 1 accepted", what, i);
        if(I[i].lo < I[i].hi)
            CHECK(!ref::certify(T.data(), n, SA.data(), P, m, SA_Interval{I[i].lo + 1, I[i].hi}), "%s: query %" PRIu64 ": lo + 1 accepted", what, i);
    }

    CHECK(test::validate_intervals(T.data(), n, SA.data(), off.data(), sym.data(), q, I.data(), err), "%s: %s", what, err.c_str());

    return true;
}


bool test_oracle_random()
{
    std::mt19937_64 rng(3);
    const idx_t sizes[] = {1, 2, 3, 15, 16, 17, 1000, 20000};

    for(const idx_t n : sizes)
    {
        // `uint16_t` over alphabets of 2 to 65536 symbols, including 0, the maximum, and the separator.
        for(const idx_t sigma : {2, 3, 4, 16, 256, 65536})
        {
            std::vector<uint16_t> alpha;
            for(idx_t s = 0; s < sigma; ++s)
                alpha.push_back(static_cast<uint16_t>(sigma == 65536 ? s : (s * 65535) / (sigma > 1 ? sigma - 1 : 1)));
            const auto T = random_text<uint16_t>(n, alpha, 0xFFFF, 50, rng);
            const std::string what = "u16 n=" + std::to_string(n) + " sigma=" + std::to_string(sigma);
            if(!check_fixture(T, alpha, rng(), what.c_str()))   return false;
        }

        // `uint8_t` and `uint32_t` with high-bit symbols.
        {
            const std::vector<uint8_t> alpha = {0, 1, 2, 0x7F, 0x80, 0xFE, 0xFF};
            const auto T = random_text<uint8_t>(n, alpha, 0xFF, 40, rng);
            if(!check_fixture(T, alpha, rng(), ("u8 n=" + std::to_string(n)).c_str()))  return false;
        }
        {
            const std::vector<uint32_t> alpha = {0, 1, 0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFEu};
            const auto T = random_text<uint32_t>(n, alpha, 0xFFFFFFFFu, 40, rng);
            if(!check_fixture(T, alpha, rng(), ("u32 n=" + std::to_string(n)).c_str()))  return false;
        }
    }

    // Periodic texts: deep common prefixes.
    {
        std::vector<uint16_t> T;
        for(int i = 0; i < 3000; ++i)
            T.push_back(static_cast<uint16_t>(i % 2 ? 7 : 3));
        if(!check_fixture(T, {3, 7}, 11, "u16 periodic (ab)^k"))    return false;

        std::vector<uint16_t> fib = {1}, prev = {0};    // A Fibonacci word.
        while(fib.size() < 2500)
        {
            auto next = fib;
            next.insert(next.end(), prev.begin(), prev.end());
            prev = fib;
            fib = next;
        }
        if(!check_fixture(fib, {0, 1}, 12, "u16 Fibonacci word"))   return false;
    }

    return true;
}


bool test_edge_cases()
{
    // Each case: text, pattern, and the expected count.
    struct Case { std::vector<uint16_t> T, P; idx_t count; const char* what; };
    const std::vector<Case> cases =
    {
        {{0, 0, 0, 0},  {},             4, "empty pattern"},
        {{0, 0, 0, 0},  {0},            4, "all-zero text"},
        {{0, 0, 0, 0},  {0, 0, 0, 0},   1, "all-zero, full text"},
        {{0, 0, 0, 0},  {0, 0, 0, 0, 0},0, "pattern longer than the text"},
        {{5, 3, 5, 0},  {0},            1, "trailing zero"},
        {{5, 3, 5, 0},  {0, 0},         0, "zero-tail must not match padding"},
        {{5, 3, 5, 0},  {5, 0},         1, "zero-tail inner match"},
        {{5, 3, 5, 0},  {5, 0, 0},      0, "zero-tail past the end"},
        {{9, 8, 7},     {7, 0},         0, "a text-end suffix with a zero-tail"},
        {{3, 4, 5},     {1},            0, "sorts before SA[0]"},
        {{3, 4, 5},     {3, 1},         0, "shares SA[0]'s first symbol, sorts before it"},
        {{3, 4, 5},     {9},            0, "sorts after SA[n - 1]"},
        {{3, 4, 5},     {5, 5},         0, "runs past the end of the last suffix"},
        {{65535, 1, 65535, 1, 65535},   {65535},    3, "separators"},
        {{65535, 1, 65535, 1, 65535},   {1, 65535}, 2, "pattern with a separator"},
        {{65535, 65535},                {65535, 65535, 65535},  0, "all-max text"},
    };

    for(const auto& c : cases)
    {
        const idx_t n = c.T.size();
        const auto SA = test::build_sa40(c.T);
        const auto I = ref::find_one(c.T.data(), n, SA.data(), c.P.data(), c.P.size());
        const auto B = test::brute_find(c.T.data(), n, c.P.data(), c.P.size());
        CHECK(same(I, B), "%s: oracle [%" PRIu64 ", %" PRIu64 ") vs brute [%" PRIu64 ", %" PRIu64 ")", c.what, I.lo, I.hi, B.lo, B.hi);
        CHECK(I.hi - I.lo == c.count, "%s: count %" PRIu64 ", expected %" PRIu64, c.what, I.hi - I.lo, c.count);
        CHECK(ref::certify(c.T.data(), n, SA.data(), c.P.data(), c.P.size(), I), "%s: certificate rejects", c.what);
    }

    // `uint32_t` high-bit order is numeric.
    {
        const std::vector<uint32_t> T = {0x80000000u, 0x7FFFFFFFu, 0xFFFFFFFFu, 0};
        const auto SA = test::build_sa40(T);
        CHECK(sa40::load(SA.data(), 0) == 3 && sa40::load(SA.data(), 1) == 1 && sa40::load(SA.data(), 2) == 0 && sa40::load(SA.data(), 3) == 2,
              "u32 suffixes are not in numeric order");
    }

    return true;
}


bool test_spot_check()
{
    std::string err;

    // A little-endian `uint16_t` text on which byte order and numeric order disagree.
    std::mt19937_64 rng(5);
    std::vector<uint16_t> T(4000);
    for(auto& x : T)
        x = static_cast<uint16_t>(rng() % 4 == 0 ? 0x0100 : (rng() % 2 ? 0x0001 : 0x0201));

    const idx_t n = T.size();
    const auto SA = test::build_sa40(T);
    const auto SB = test::build_sa40_byte_order(T);
    const auto at = [](const std::vector<uint8_t>& S) { return [&S](const idx_t r) { return sa40::load(S.data(), r); }; };

    CHECK(ref::spot_check_sa_order(T.data(), n, at(SA), n, n, 1, err), "%s", err.c_str());
    CHECK(!ref::spot_check_sa_order(T.data(), n, at(SB), n, n, 1, err), "a byte-order suffix array must be rejected");

    auto swapped = SA;   // Swap an adjacent pair.
    const idx_t x = sa40::load(SA.data(), 100), y = sa40::load(SA.data(), 101);
    sa40::store(swapped.data(), 100, y);
    sa40::store(swapped.data(), 101, x);
    CHECK(!ref::spot_check_sa_order(T.data(), n, at(swapped), n, n, 1, err), "a swapped pair must be rejected");

    auto oob = SA;
    sa40::store(oob.data(), 7, n + 3);
    CHECK(!ref::spot_check_sa_order(T.data(), n, at(oob), n, n, 1, err), "an out-of-range entry must be rejected");

    return true;
}


// Returns `true` iff a CUDA device is available.
bool gpu_available()
{
    int c = 0;
    return cudaGetDeviceCount(&c) == cudaSuccess && c > 0;
}


// Writes into `dir` a query index of the text `T` with the padded suffix array
// `SA`, the suffix array split into `parts` parts. Returns `false` with a reason
// in `err` on failure.
template <typename T_sym_>
bool write_index(const std::filesystem::path& dir, const std::vector<T_sym_>& T, const std::vector<uint8_t>& SA, const idx_t parts,
                 std::string& err)
{
    Index_Meta m;
    m.n = T.size();
    m.sym_bytes = sizeof(T_sym_);
    m.text = (dir / "text.bin").string();
    m.sa_part_off = index_meta::part_offsets(m.sa_bytes(), parts);
    m.sa_source = "test_query";
    std::ofstream(m.text, std::ios::binary).write(reinterpret_cast<const char*>(T.data()), m.n * sizeof(T_sym_));
    for(idx_t i = 0; i < parts; ++i)
    {
        m.sa_part.push_back((dir / ("sa.u40.part" + std::to_string(i))).string());
        std::ofstream(m.sa_part[i], std::ios::binary).write(reinterpret_cast<const char*>(SA.data() + m.sa_part_off[i]), m.part_bytes(i));
    }

    return index_meta::write(dir.string(), m, err);
}


// Returns the count of the intervals `I` of the batch `off`, `sym` that fail the
// certificate on the device.
template <typename T_sym_>
unsigned long long device_certify(const Query_Engine<T_sym_>& eng, const std::vector<uint64_t>& off, const std::vector<T_sym_>& sym,
                                  const std::vector<SA_Interval>& I)
{
    const idx_t q = I.size();
    uint64_t* off_d = nullptr;
    T_sym_* sym_d = nullptr;
    SA_Interval* I_d = nullptr;
    unsigned long long* flags_d = nullptr;
    unsigned long long flags[2] = {0, q};
    CUDA_CHECK(cudaMalloc(&off_d, off.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&sym_d, std::max<std::size_t>(sym.size(), 1) * sizeof(T_sym_)));
    CUDA_CHECK(cudaMalloc(&I_d, std::max<idx_t>(q, 1) * sizeof(SA_Interval)));
    CUDA_CHECK(cudaMalloc(&flags_d, sizeof(flags)));
    CUDA_CHECK(cudaMemcpy(off_d, off.data(), off.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sym_d, sym.data(), sym.size() * sizeof(T_sym_), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(I_d, I.data(), q * sizeof(SA_Interval), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(flags_d, flags, sizeof(flags), cudaMemcpyHostToDevice));

    eng.certify_async(off_d, sym_d, q, I_d, flags_d, flags_d + 1, 0);
    CUDA_CHECK(cudaMemcpy(flags, flags_d, sizeof(flags), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(off_d));
    CUDA_CHECK(cudaFree(sym_d));
    CUDA_CHECK(cudaFree(I_d));
    CUDA_CHECK(cudaFree(flags_d));
    return flags[0];
}


// Checks the device engine against the oracle on the fixture text `T` of the
// symbols `alpha`, in every device-accessible placement, with the suffix array
// in `parts` parts; and that the device certificate accepts its intervals and
// rejects perturbed ones.
template <typename T_sym_>
bool check_gpu_fixture(const std::vector<T_sym_>& T, const std::vector<T_sym_>& alpha, const uint64_t seed, const idx_t parts,
                       const char* const what)
{
    const auto dir = scratch_dir("gpu");
    const auto SA = test::build_sa40(T);
    std::string err;
    CHECK(write_index(dir, T, SA, parts, err), "%s: %s", what, err.c_str());

    std::mt19937_64 rng(seed);
    std::vector<uint64_t> off;
    std::vector<T_sym_> sym;
    fixture_queries(T, alpha, rng, 400, off, sym);
    const idx_t q = off.size() - 1, n = T.size();
    std::vector<SA_Interval> want(q), got(q);
    for(idx_t i = 0; i < q; ++i)
        want[i] = ref::find_one(T.data(), n, SA.data(), sym.data() + off[i], off[i + 1] - off[i]);

    for(const auto text_res : {Residency::device, Residency::pinned})
        for(const auto sa_res : {Residency::device, Residency::pinned})
        {
            const std::string where = std::string(what) + " (text " + residency_name(text_res) + ", suffix array " + residency_name(sa_res) + ")";
            Query_Engine<T_sym_> eng;
            CHECK(eng.load(dir.string(), Query_Options{text_res, sa_res, 100000}, err), "%s: %s", where.c_str(), err.c_str());
            std::fill(got.begin(), got.end(), SA_Interval{~uint64_t(0), ~uint64_t(0)});
            eng.find(off.data(), sym.data(), q, got.data());
            for(idx_t i = 0; i < q; ++i)
                CHECK(same(got[i], want[i]), "%s: query %" PRIu64 ": device [%" PRIu64 ", %" PRIu64 ") vs oracle [%" PRIu64 ", %" PRIu64 ")",
                      where.c_str(), i, got[i].lo, got[i].hi, want[i].lo, want[i].hi);

            CHECK(device_certify(eng, off, sym, got) == 0, "%s: the device certificate rejects correct intervals", where.c_str());
            auto wrong = got;
            idx_t perturbed = 0;
            for(auto& I : wrong)
                if(I.hi < n)
                    ++I.hi, ++perturbed;
            CHECK(device_certify(eng, off, sym, wrong) == perturbed, "%s: the device certificate accepts perturbed intervals", where.c_str());
        }

    return true;
}


bool test_gpu_find()
{
    std::mt19937_64 rng(7);
    for(const idx_t n : {1, 2, 3, 17, 1000, 20000})
    {
        const idx_t parts = (SA_WIDTH * n >= 2 * SA_PAGE_BYTES ? 2 : 1);   // Parts are page-aligned.
        for(const idx_t sigma : {2, 4, 65536})
        {
            std::vector<uint16_t> alpha;
            for(idx_t s = 0; s < sigma; ++s)
                alpha.push_back(static_cast<uint16_t>(sigma == 65536 ? s : (s * 65535) / (sigma - 1)));
            const auto T = random_text<uint16_t>(n, alpha, 0xFFFF, 50, rng);
            const std::string what = "u16 n=" + std::to_string(n) + " sigma=" + std::to_string(sigma);
            if(!check_gpu_fixture(T, alpha, rng(), parts, what.c_str()))   return false;
        }

        {
            const std::vector<uint8_t> alpha = {0, 1, 2, 0x7F, 0x80, 0xFE, 0xFF};
            const auto T = random_text<uint8_t>(n, alpha, 0xFF, 40, rng);
            if(!check_gpu_fixture(T, alpha, rng(), parts, ("u8 n=" + std::to_string(n)).c_str()))   return false;
        }
        {
            const std::vector<uint32_t> alpha = {0, 1, 0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFEu};
            const auto T = random_text<uint32_t>(n, alpha, 0xFFFFFFFFu, 40, rng);
            if(!check_gpu_fixture(T, alpha, rng(), parts, ("u32 n=" + std::to_string(n)).c_str()))   return false;
        }
    }

    // Periodic text: deep common prefixes.
    std::vector<uint16_t> T;
    for(int i = 0; i < 3000; ++i)
        T.push_back(static_cast<uint16_t>(i % 2 ? 7 : 3));

    return check_gpu_fixture(T, {3, 7}, 13, 2, "u16 periodic (ab)^k");
}


bool test_gpu_rejects()
{
    // A little-endian `uint16_t` text on which byte order and numeric order disagree.
    std::mt19937_64 rng(5);
    std::vector<uint16_t> T(4000);
    for(auto& x : T)
        x = static_cast<uint16_t>(rng() % 4 == 0 ? 0x0100 : (rng() % 2 ? 0x0001 : 0x0201));

    const auto SA = test::build_sa40(T);
    const auto dir = scratch_dir("gpu-rejects");
    std::string err;
    Query_Engine<uint16_t> eng;

    CHECK(write_index(dir, T, test::build_sa40_byte_order(T), 1, err), "%s", err.c_str());
    CHECK(!eng.load(dir.string(), Query_Options{}, err), "a byte-order suffix array must be rejected at load");

    auto swapped = SA;
    const idx_t x = sa40::load(SA.data(), 100), y = sa40::load(SA.data(), 101);
    sa40::store(swapped.data(), 100, y);
    sa40::store(swapped.data(), 101, x);
    CHECK(write_index(dir, T, swapped, 2, err), "%s", err.c_str());
    CHECK(!eng.load(dir.string(), Query_Options{Residency::pinned, Residency::device, 100000}, err), "a swapped pair must be rejected at load");

    CHECK(write_index(dir, T, SA, 1, err), "%s", err.c_str());
    CHECK(eng.load(dir.string(), Query_Options{}, err), "%s", err.c_str());
    Query_Engine<uint8_t> narrow;
    CHECK(!narrow.load(dir.string(), Query_Options{}, err), "a symbol-width mismatch must be rejected");
    CHECK(!eng.load(dir.string(), Query_Options{Residency::pageable, Residency::device, 0}, err), "a host-only placement must be rejected");

    std::filesystem::resize_file(dir / "sa.u40.part0", SA_WIDTH * T.size() - 1);
    CHECK(!eng.load(dir.string(), Query_Options{}, err), "a truncated suffix array must be rejected");

    return true;
}


bool test_index_host_tiers()
{
    // `uint32_t` symbols on whose little-endian bytes byte order and numeric order disagree.
    std::mt19937_64 rng(9);
    const uint32_t alpha[] = {0x00000001u, 0x00000100u, 0x00010000u, 0xFFFFFFFFu};
    std::vector<uint32_t> T(5000);
    for(auto& x : T)
        x = alpha[rng() % 4];

    const auto SA = test::build_sa40(T);
    const auto dir = scratch_dir("host-tiers");
    std::string err;
    CHECK(write_index(dir, T, SA, 2, err), "%s", err.c_str());

    Suffix_Index idx;
    CHECK(idx.load(dir.string(), Residency::pageable, Residency::pageable, 100000, err), "%s", err.c_str());
    CHECK(idx.host_accessible() && !idx.device_accessible() && !idx.stats().spot_on_device, "unexpected placement of a host-only index");
    CHECK(std::memcmp(idx.host_text<uint32_t>(), T.data(), T.size() * sizeof(uint32_t)) == 0, "text differs after loading");
    CHECK(std::memcmp(idx.host_sa(), SA.data(), SA_WIDTH * T.size()) == 0, "suffix array differs after loading");
    for(idx_t b = 0; b < TEXT_PAD_BYTES; ++b)
        CHECK(reinterpret_cast<const uint8_t*>(idx.host_text<uint32_t>())[T.size() * sizeof(uint32_t) + b] == 0, "text padding is not zero");
    for(idx_t b = 0; b < SA_PAD_BYTES; ++b)
        CHECK(idx.host_sa()[SA_WIDTH * T.size() + b] == 0, "suffix-array padding is not zero");

    CHECK(write_index(dir, T, test::build_sa40_byte_order(T), 1, err), "%s", err.c_str());
    CHECK(!idx.load(dir.string(), Residency::pageable, Residency::pageable, T.size(), err) && !idx.loaded(),
          "a misordered suffix array must be rejected by the host spot-check");

    return true;
}

}


int main()
{
    if(!gpu_saca::sa40::host_is_little_endian())
    {
        std::fprintf(stderr, "the query index requires a little-endian host\n");
        return 1;
    }

    struct Test { const char* name; std::function<bool()> fn; bool gpu; };
    const std::vector<Test> tests =
    {
        {"sa40 round-trip",         test_sa40_round_trip,       false},
        {"query-file round-trip",   test_query_file_round_trip, false},
        {"index-meta round-trip",   test_index_meta_round_trip, false},
        {"oracle vs brute force",   test_oracle_random,         false},
        {"edge cases",              test_edge_cases,            false},
        {"order spot-check",        test_spot_check,            false},
        {"index load, host tiers",  test_index_host_tiers,      false},
        {"device find vs oracle",   test_gpu_find,              true},
        {"device load rejects",     test_gpu_rejects,           true},
    };

    const bool gpu = gpu_available();
    int failed = 0, skipped = 0;
    for(const auto& t : tests)
    {
        if(t.gpu && !gpu)
        {
            ++skipped;
            std::fprintf(stderr, "[test_query] %-24s SKIP (no device)\n", t.name);
            continue;
        }

        const bool ok = t.fn();
        failed += !ok;
        std::fprintf(stderr, "[test_query] %-24s %s\n", t.name, ok ? "PASS" : "FAIL");
    }

    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::temp_directory_path() / ("gsa-test-query-" + std::to_string(::getpid())), ec);

    std::fprintf(stderr, "[test_query] %d / %zu tests failed, %d skipped\n", failed, tests.size(), skipped);
    return failed == 0 ? 0 : 1;
}
