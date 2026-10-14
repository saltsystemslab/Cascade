
// Test runner of the query engine: fixtures across symbol widths, edge cases,
// and oracles, host-side and on the device. Exits `0` iff every test passes;
// the device tests are skipped without a device.

#include "CPU_Reference.hpp"
#include "Index_Meta.hpp"
#include "Packed_Prefix.hpp"
#include "Query_Engine.hpp"
#include "Suffix_Index.hpp"
#include "query-file.hpp"
#include "query_tests.hpp"
#include "sa40.hpp"
#include "utility.cuh"

#include <cuda_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
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

    // A repeated entry of a suffix longer than the comparison limit: undecidable by comparison.
    auto rep = SA;
    idx_t r = 0;
    while(sa40::load(SA.data(), r) > n - 2000)
        ++r;
    sa40::store(rep.data(), r + 1, sa40::load(SA.data(), r));
    CHECK(!ref::spot_check_sa_order(T.data(), n, at(rep), n, 1000, 1, err), "a repeated entry must be rejected");

    return true;
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


// Returns the most parts, up to `k`, that `index_meta::part_offsets` splits a
// suffix array of `n` entries into: parts are page-aligned.
idx_t parts_for(const idx_t n, const idx_t k)
{
    idx_t p = k;
    while(p > 1 && SA_WIDTH * n < p * SA_PAGE_BYTES)
        --p;

    return p;
}


// Checks the host's search against the oracle on the fixture text `T` of the
// symbols `alpha`, with some queries repeated: over the whole suffix array, and
// within brackets of random widths around the true boundaries, some wider than
// a window of entries; in the batch's order, and sorted with identical queries
// adjacent; over several counts of lanes and threads; with the suffix array in
// memory, and mapped from its files in parts, the brackets' pages asked for
// ahead. Checks also that the host's certificate accepts the intervals and
// rejects perturbed ones.
template <typename T_sym_>
bool check_host_search(const std::vector<T_sym_>& T, const std::vector<T_sym_>& alpha, const uint64_t seed, const char* const what)
{
    const idx_t n = T.size();
    const auto SA = test::build_sa40(T);
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> off;
    std::vector<T_sym_> sym;
    fixture_queries(T, alpha, rng, 400, off, sym);
    for(int k = 0; k < 200; ++k)
    {
        const idx_t i = rng() % 400;
        const std::vector<T_sym_> P(sym.begin() + off[i], sym.begin() + off[i + 1]);
        sym.insert(sym.end(), P.begin(), P.end());
        off.push_back(sym.size());
    }

    const idx_t q = off.size() - 1;
    std::vector<SA_Interval> want(q);
    for(idx_t i = 0; i < q; ++i)
        want[i] = ref::find_one(T.data(), n, SA.data(), sym.data() + off[i], off[i + 1] - off[i]);

    const auto around = [&](const idx_t x)
    {
        const idx_t w = (rng() % 4 == 0 ? 1000 : 40);
        return Rank_Range{x - std::min<idx_t>(x, rng() % w), x + std::min<idx_t>(n - x, rng() % w)};
    };

    std::vector<uint32_t> ids(q);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](const uint32_t i, const uint32_t j)
    {
        return std::lexicographical_compare(sym.begin() + off[i], sym.begin() + off[i + 1], sym.begin() + off[j], sym.begin() + off[j + 1]);
    });

    std::vector<Rank_Range> br(2 * q), br_sorted(2 * q);
    for(idx_t i = 0; i < q; ++i)
        br[2 * i] = around(want[i].lo), br[2 * i + 1] = around(want[i].hi);
    for(idx_t g = 0; g < q; ++g)
        br_sorted[2 * g] = br[2 * ids[g]], br_sorted[2 * g + 1] = br[2 * ids[g] + 1];

    // The suffix array in memory, and mapped from its files.
    const auto dir = scratch_dir("host-search");
    std::string err;
    Suffix_Index idx;
    CHECK(write_index(dir, T, SA, parts_for(n, 2), err), "%s: %s", what, err.c_str());
    CHECK(idx.load(dir.string(), Residency::pageable, Residency::mapped, 1000, err), "%s: %s", what, err.c_str());
    const uint8_t* const source[] = {SA.data(), idx.host_sa()};
    const char* const where[] = {"in memory", "mapped"};

    // The pages of the mapped suffix array asked for ahead: those that the brackets' searches probe first, of brackets as wide as the
    // array too, each marked once, and none of the others; none refused.
    host_search::SA_Files files;
    CHECK(files.open(idx.meta().sa_part, idx.meta().sa_part_off, 3, err), "%s: %s", what, err.c_str());
    const std::vector<Rank_Range> whole(2 * q, Rank_Range{0, n});
    const idx_t page = static_cast<idx_t>(::sysconf(_SC_PAGESIZE)), pages = (SA_WIDTH * n + page - 1) / page;
    const Rank_Range* const bracket_sets[] = {whole.data(), br.data(), br_sorted.data()};
    for(const Rank_Range* const b : bracket_sets)
    {
        std::vector<uint8_t> want_page(pages, 0);
        for(idx_t k = 0; k < 2 * q; ++k)
            if(b[k].last > b[k].first)
            {
                idx_t p0 = SA_WIDTH * b[k].first / page, p1 = std::min(pages - 1, (SA_WIDTH * b[k].last + 2) / page);
                if(p1 - p0 + 1 > host_search::PREFETCH_PAGES)
                {
                    const idx_t mid = b[k].first + (b[k].last - b[k].first) / 2;
                    p0 = SA_WIDTH * mid / page, p1 = std::min(pages - 1, (SA_WIDTH * mid + 7) / page);
                }

                std::fill(want_page.begin() + p0, want_page.begin() + p1 + 1, 1);
            }

        const std::unique_ptr<std::atomic<uint8_t>[]> asked(new std::atomic<uint8_t>[pages]());
        idx_t refused = 0;
        const idx_t c = host_search::prefetch(files, n, b, q, asked.get(), refused);
        CHECK(refused == 0 && c >= static_cast<idx_t>(std::count(want_page.begin(), want_page.end(), 1)), "%s: %" PRIu64 " pages asked for ahead, "
              "%" PRIu64 " requests refused", what, c, refused);
        for(idx_t g = 0; g < pages; ++g)
            CHECK(asked[g].load() == want_page[g], "%s: page %" PRIu64 " of %" PRIu64 " %s asked for ahead", what, g, pages,
                  want_page[g] ? "not" : "wrongly");
        CHECK(host_search::prefetch(files, n, b, q, asked.get(), refused) == 0 && refused == 0, "%s: pages asked for again", what);
    }

    const char* const how[] = {"over the whole suffix array", "within brackets", "sorted, within brackets"};
    for(int s = 0; s < 2; ++s)
        for(const unsigned lanes : {1u, 2u, 5u, 16u, 64u})
            for(const unsigned threads : {1u, 3u})
                for(int k = 0; k < 3; ++k)
                {
                    const Rank_Range* const b = (k == 0 ? nullptr : (k == 1 ? br.data() : br_sorted.data()));
                    if(s == 1 && b != nullptr)
                    {
                        idx_t refused = 0;
                        host_search::prefetch(files, n, b, q, nullptr, refused);
                        CHECK(refused == 0, "%s: the kernel refused to read the brackets' pages ahead", what);
                    }

                    std::vector<SA_Interval> got(q, SA_Interval{~uint64_t(0), ~uint64_t(0)});
                    host_search::find(T.data(), n, source[s], off.data(), sym.data(), q, b, k == 2 ? ids.data() : nullptr, got.data(), threads, lanes);
                    for(idx_t i = 0; i < q; ++i)
                        CHECK(same(got[i], want[i]), "%s, suffix array %s, %u lanes, %u threads, %s: query %" PRIu64 " (m = %" PRIu64 "): [%" PRIu64
                              ", %" PRIu64 ") vs oracle [%" PRIu64 ", %" PRIu64 ")", what, where[s], lanes, threads, how[k], i, off[i + 1] - off[i],
                              got[i].lo, got[i].hi, want[i].lo, want[i].hi);
                }

    unsigned long long fail_c = 0, first = q;
    host_search::certify(T.data(), n, SA.data(), off.data(), sym.data(), q, want.data(), fail_c, first, 3);
    CHECK(fail_c == 0 && first == q, "%s: the host certificate rejects correct intervals", what);

    auto wrong = want;
    idx_t perturbed = 0, first_wrong = q;
    for(idx_t i = 0; i < q; ++i)
        if(wrong[i].hi < n)
        {
            ++wrong[i].hi, ++perturbed;
            first_wrong = std::min(first_wrong, i);
        }

    host_search::certify(T.data(), n, SA.data(), off.data(), sym.data(), q, wrong.data(), fail_c, first, 3);
    CHECK(fail_c == perturbed && first == first_wrong, "%s: the host certificate accepts perturbed intervals", what);

    return true;
}


bool test_host_search()
{
    std::mt19937_64 rng(19);
    for(const idx_t n : {1, 2, 3, 15, 16, 17, 1000, 20000})
    {
        for(const idx_t sigma : {2, 4, 65536})
        {
            std::vector<uint16_t> alpha;
            for(idx_t s = 0; s < sigma; ++s)
                alpha.push_back(static_cast<uint16_t>(sigma == 65536 ? s : (s * 65535) / (sigma - 1)));
            const auto T = random_text<uint16_t>(n, alpha, 0xFFFF, 50, rng);
            const std::string what = "u16 n=" + std::to_string(n) + " sigma=" + std::to_string(sigma);
            if(!check_host_search(T, alpha, rng(), what.c_str()))  return false;
        }

        {
            const std::vector<uint8_t> alpha = {0, 1, 2, 0x7F, 0x80, 0xFE, 0xFF};
            const auto T = random_text<uint8_t>(n, alpha, 0xFF, 40, rng);
            if(!check_host_search(T, alpha, rng(), ("u8 n=" + std::to_string(n)).c_str()))  return false;
        }
        {
            const std::vector<uint32_t> alpha = {0, 1, 0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFEu};
            const auto T = random_text<uint32_t>(n, alpha, 0xFFFFFFFFu, 40, rng);
            if(!check_host_search(T, alpha, rng(), ("u32 n=" + std::to_string(n)).c_str()))  return false;
        }
    }

    // Periodic text: deep common prefixes, and long matches.
    std::vector<uint16_t> T;
    for(int i = 0; i < 3000; ++i)
        T.push_back(static_cast<uint16_t>(i % 2 ? 7 : 3));

    return check_host_search(T, {3, 7}, 23, "u16 periodic (ab)^k");
}


// Returns `true` iff a CUDA device is available.
bool gpu_available()
{
    int c = 0;
    return cudaGetDeviceCount(&c) == cudaSuccess && c > 0;
}


// Returns the metadata of an index of `n` symbols of `sym_bytes` bytes, whose
// files are all at `path`, for the planner, which reads none of them.
Index_Meta sized_meta(const idx_t n, const uint32_t sym_bytes, const std::string& path)
{
    Index_Meta m;
    m.n = n;
    m.sym_bytes = sym_bytes;
    m.text = path;
    m.sa_part = {path};
    m.sa_part_off = {0};
    return m;
}


// Returns options that leave every pick to the planner, within budgets of
// `device` and `host` bytes; `0` measures one.
Query_Options auto_options(const idx_t device = 0, const idx_t host = 0)
{
    Query_Options opt;
    opt.text_res = opt.sa_res = Residency::automatic;
    opt.stride = AUTO_STRIDE;
    opt.key_words = AUTO_KEY_WORDS;
    opt.device_budget = device;
    opt.host_budget = host;
    return opt;
}


// Returns the options `opt` with the tiers named `text_res` and `sa_res`.
Query_Options with_tiers(Query_Options opt, const Residency text_res, const Residency sa_res)
{
    opt.text_res = text_res;
    opt.sa_res = sa_res;
    return opt;
}


// Checks the planner on indexes of synthetic sizes, with set budgets: the
// campaigns' indexes on the benchmark node pick its plans and tables; over
// shrinking budgets, every plan, the first that fits at the least stride that
// fits, and none past the last; the edges of a fit; gathered brackets that
// would no longer fit their windows; named tiers, tables, gathers, and chunks
// mixed with the planner's picks; symbols of every width; and budgets
// measured.
bool test_planner()
{
    using P = Query_Engine<uint16_t>;
    const std::string path = scratch_dir("planner").string();   // On a local file system.
    const auto dev = Residency::device, pin = Residency::pinned, pg = Residency::pageable, map = Residency::mapped;
    std::string err;
    Placement p;

    // The benchmark node: the device's free memory at load, and the host's available memory.
    const idx_t node_dev = 101388779520ull, node_host = idx_t(993) << 30;
    struct Case { const char* what; idx_t n; Plan plan; Residency text_res, sa_res; idx_t stride; uint32_t key_words; idx_t table_bytes; };
    const Case cases[] =
    {
        {"dolmasample", 8039098124ull, Plan::A, dev, dev, 32, 2, 4019549072ull},
        {"arxiv", 30214423492ull, Plan::C, pg, pg, 32, 4, 30214423520ull},
        {"github", 63944091504ull, Plan::C, pg, pg, 32, 4, 63944091520ull},
        {"c4", 198079545568ull, Plan::D, pg, map, 128, 4, 49519886400ull},
        {"pile", 238429547834ull, Plan::D, pg, map, 128, 4, 59607386976ull},
    };

    for(const auto& c : cases)
    {
        CHECK(P::plan(sized_meta(c.n, 2, path), auto_options(node_dev, node_host), p, err), "%s: %s", c.what, err.c_str());
        CHECK(p.plan == c.plan && p.text_res == c.text_res && p.sa_res == c.sa_res && p.stride == c.stride && p.key_words == c.key_words
              && p.table_bytes == c.table_bytes, "%s: plan %s, text %s, suffix array %s, stride %" PRIu64 ", %u key words, %" PRIu64 " table bytes",
              c.what, plan_name(p.plan), residency_name(p.text_res), residency_name(p.sa_res), p.stride, p.key_words, p.table_bytes);
        CHECK(p.fits && p.auto_text_res && p.auto_sa_res && p.auto_stride && p.auto_key_words && p.device_budget == node_dev
              && p.host_budget == node_host && p.device_margin == node_dev / 16 && p.host_margin == node_host / 16
              && p.device_bytes + p.device_margin <= node_dev && p.host_bytes + p.host_margin <= node_host
              && p.host_search == (c.plan == Plan::C || c.plan == Plan::D) && p.gather == (c.plan == Plan::B) && p.sa_local,
              "%s: unexpected placement", c.what);
        CHECK(p.chunk_queries == (idx_t(1) << (p.gather ? 24 : 18)) && p.chunk_symbols == (idx_t(1) << (p.gather ? 28 : 24)),
              "%s: chunks of the placement's default sizes", c.what);
    }

    // Over shrinking budgets: the first plan that fits, its tiers named alone fitting, at the least stride that fits; no earlier plan fits,
    // nor the pick at half its stride; and every plan picked, and none past the last.
    struct Tiers { Plan plan; Residency text_res, sa_res; };
    const Tiers plans[] = {{Plan::A, dev, dev}, {Plan::C, pg, pg}, {Plan::B, dev, pin}, {Plan::D, pg, map}};
    const Index_Meta m = sized_meta(4000000000ull, 2, path);    // A text of 8 GB, a suffix array of 20 GB.
    std::set<Plan> seen;
    int none = 0;
    for(idx_t d = idx_t(1) << 30; d <= idx_t(48) << 30; d += idx_t(1) << 29)
        for(const idx_t h : {idx_t(8) << 30, idx_t(16) << 30, idx_t(24) << 30, idx_t(27) << 30, idx_t(30) << 30, idx_t(40) << 30})    // 27 GiB: B's.
        {
            const bool ok = P::plan(m, auto_options(d, h), p, err);
            const std::string what = "device " + std::to_string(d) + ", host " + std::to_string(h);
            for(const auto& t : plans)
            {
                Placement q;
                const bool fits = P::plan(m, with_tiers(auto_options(d, h), t.text_res, t.sa_res), q, err) && q.fits;
                if(ok && t.plan == p.plan)
                {
                    CHECK(fits && q.stride == p.stride && q.key_words == p.key_words && q.device_bytes == p.device_bytes && !q.auto_text_res
                          && !q.auto_sa_res, "%s: plan %s picked, but not alone", what.c_str(), plan_name(p.plan));
                    break;
                }

                CHECK(!fits, "%s: plan %s fits, before the pick %s", what.c_str(), plan_name(t.plan), ok ? plan_name(p.plan) : "of none");
            }

            if(!ok)
            {
                ++none;
                continue;
            }

            seen.insert(p.plan);
            CHECK(p.fits && p.device_bytes + p.device_margin <= d && p.host_bytes + p.host_margin <= h && p.stride >= 32 && p.stride <= 4096
                  && (p.stride & (p.stride - 1)) == 0, "%s: plan %s does not fit at stride %" PRIu64, what.c_str(), plan_name(p.plan), p.stride);
            if(p.stride > 32)
            {
                Query_Options o = with_tiers(auto_options(d, h), p.text_res, p.sa_res);
                o.stride = p.stride / 2;
                o.key_words = p.key_words;
                Placement q;
                CHECK(P::plan(m, o, q, err) && !q.fits && !q.auto_stride, "%s: plan %s fits at half its stride %" PRIu64, what.c_str(),
                      plan_name(p.plan), p.stride);
            }
        }

    CHECK(seen.size() == 4 && none > 0, "budgets over every plan picked %zu of them, and fit none %d times", seen.size(), none);

    // The edge of a fit: the least device budget for A at stride 32 holds its bytes and its margin; a byte less does not, and picks the
    // next stride. Likewise with the tiers and the table named.
    const auto least = [&](const Index_Meta& x, const std::function<bool(const Placement&)>& want, idx_t lo, idx_t hi,
                           const Query_Options& o = auto_options(), const idx_t host = 0)  // `0`: the node's host memory.
    {
        while(lo < hi)
        {
            const idx_t mid = lo + (hi - lo) / 2;
            Query_Options om = o;
            om.device_budget = mid;
            om.host_budget = (host != 0 ? host : node_host);
            Placement q;
            if(P::plan(x, om, q, err) && want(q))
                hi = mid;
            else
                lo = mid + 1;
        }

        return lo;
    };

    const idx_t edge = least(m, [](const Placement& q) { return q.plan == Plan::A && q.stride == 32; }, idx_t(1) << 30, idx_t(1) << 40);
    CHECK(P::plan(m, auto_options(edge, node_host), p, err) && p.plan == Plan::A && p.stride == 32 && p.device_bytes + p.device_margin <= edge,
          "the edge of A at stride 32: %s", err.c_str());
    const idx_t a32 = p.device_bytes;
    CHECK(P::plan(m, auto_options(edge - 1, node_host), p, err) && p.plan == Plan::A && p.stride == 64 && a32 + std::max<idx_t>(idx_t(1) << 30, (edge - 1) / 16) > edge - 1,
          "a byte below the edge of A at stride 32, plan %s at stride %" PRIu64, plan_name(p.plan), p.stride);
    {
        Query_Options o = with_tiers(auto_options(), dev, dev);
        o.stride = 32;
        o.key_words = 2;
        const idx_t named_edge = least(m, [](const Placement& q) { return q.fits; }, idx_t(1) << 30, idx_t(1) << 40, o);
        CHECK(named_edge == edge, "the edge of A at stride 32 named, %" PRIu64 ", differs from the planner's, %" PRIu64, named_edge, edge);
        o.device_budget = edge - 1;
        o.host_budget = node_host;
        CHECK(P::plan(m, o, p, err) && !p.fits && p.device_bytes == a32, "a byte below the edge of A at stride 32 named fits");
    }

    // B, on a host whose memory holds the suffix array but not the text beside it, so that C does not fit. Gathered brackets fit their
    // windows at strides up to 128: past the least device budget for B, the pick is D, though B at stride 256 would fit.
    const Index_Meta arxiv = sized_meta(30214423492ull, 2, path);
    const idx_t b_host = idx_t(200) << 30;
    CHECK(P::plan(arxiv, auto_options(node_dev, b_host), p, err) && p.plan == Plan::B && p.stride == 32 && p.key_words == 2,
          "arxiv on a host of 200 GiB: plan %s at stride %" PRIu64, plan_name(p.plan), p.stride);
    const idx_t b_edge = least(arxiv, [](const Placement& q) { return q.plan == Plan::A || q.plan == Plan::B; }, idx_t(1) << 30, idx_t(1) << 40,
                               auto_options(), b_host);
    CHECK(P::plan(arxiv, auto_options(b_edge, b_host), p, err) && p.plan == Plan::B && p.stride == 128, "the edge of B: plan %s at stride %" PRIu64,
          plan_name(p.plan), p.stride);
    CHECK(P::plan(arxiv, auto_options(b_edge - 1, b_host), p, err) && p.plan == Plan::D, "below the edge of B: plan %s", plan_name(p.plan));
    Placement q;
    CHECK(P::plan(arxiv, with_tiers(auto_options(b_edge - 1, node_host), dev, pin), q, err) == false
          && P::plan(arxiv, [&] { auto o = with_tiers(auto_options(b_edge - 1, node_host), dev, pin); o.stride = 128; o.key_words = 2; return o; }(), q, err)
          && q.table_room >= Sample_Table::bytes_of(arxiv.n, 256, 2) + 4 * (2 * 1024 * (idx_t(1) << 18) + 16) + 2 * DEVICE_PAGE_BYTES,
          "below the edge of B, its tiers named fit no table, though one of stride 256 would fit their room");

    // Named tiers, with the planner's other picks, and named tables, gathers, and chunks.
    const Index_Meta dolma = sized_meta(8039098124ull, 2, path), github = sized_meta(63944091504ull, 2, path), c4 = sized_meta(198079545568ull, 2, path);
    const Query_Options node = auto_options(node_dev, node_host);
    CHECK(!P::plan(github, with_tiers(node, dev, Residency::automatic), p, err), "github fits no plan with its text on the device");
    CHECK(P::plan(github, with_tiers(node, pg, Residency::automatic), p, err) && p.plan == Plan::C && !p.auto_text_res && p.auto_sa_res,
          "github, its text pageable: %s", err.c_str());
    CHECK(P::plan(c4, with_tiers(node, pg, Residency::automatic), p, err) && p.plan == Plan::D, "c4, its text pageable: %s", err.c_str());
    CHECK(P::plan(c4, with_tiers(node, Residency::automatic, map), p, err) && p.plan == Plan::D && p.stride == 128 && p.key_words == 4,
          "c4, its suffix array mapped: %s", err.c_str());
    CHECK(P::plan(dolma, with_tiers(node, Residency::automatic, dev), p, err) && p.plan == Plan::A, "dolmasample, its suffix array on the device");
    CHECK(!P::plan(dolma, with_tiers(node, pin, Residency::automatic), p, err), "no plan has the text pinned");
    CHECK(P::plan(c4, with_tiers(node, pg, map), p, err) && p.fits && p.plan == Plan::D && p.stride == 128 && p.key_words == 4 && p.auto_stride
          && !p.auto_text_res, "c4 with its tiers named and its table automatic: %s", err.c_str());
    {
        Query_Options o = node;
        o.stride = 64;
        CHECK(!P::plan(c4, o, p, err), "c4 fits no plan with a table of stride 64 and 4-word keys");
        o.key_words = 2;
        CHECK(P::plan(c4, o, p, err) && p.plan == Plan::D && p.stride == 64 && p.key_words == 2 && !p.auto_stride && !p.auto_key_words,
              "c4 with a table of stride 64 and 2-word keys: %s", err.c_str());
        o.stride = AUTO_STRIDE;
        CHECK(P::plan(c4, o, p, err) && p.plan == Plan::D && p.stride == 64 && p.key_words == 2, "c4 with 2-word keys: %s", err.c_str());
        o.stride = 0;
        CHECK(P::plan(c4, o, p, err) && p.plan == Plan::D && p.stride == 0 && p.key_words == 0 && p.table_bytes == 0, "c4 in the direct mode: %s",
              err.c_str());
    }
    {
        Query_Options o = with_tiers(node, dev, dev);  // Named tiers and table, as before the planner: placed as named, fitting or not.
        o.stride = 32;
        o.key_words = 0;
        CHECK(P::plan(c4, o, p, err) && p.plan == Plan::A && !p.fits && p.stride == 32 && p.key_words == 2, "c4, named on the device: %s", err.c_str());
        CHECK(P::plan(c4, with_tiers(o, pin, pin), p, err) && p.plan == Plan::custom && p.stride == 32, "c4, named pinned: %s", err.c_str());
    }
    {
        Query_Options o = node;
        o.gather = Gather::always;
        CHECK(!P::plan(github, o, p, err), "github fits no plan that gathers its brackets");
        o.gather = Gather::never;
        CHECK(P::plan(arxiv, o, p, err) && p.plan == Plan::C && !p.gather, "arxiv, not gathering: %s", err.c_str());
        CHECK(P::plan(arxiv, with_tiers(o, dev, pin), p, err) && p.plan == Plan::B && !p.gather && p.chunk_queries == (idx_t(1) << 18),
              "arxiv in B, not gathering: %s", err.c_str());
        const idx_t ungathered = p.device_bytes;
        CHECK(P::plan(arxiv, with_tiers(node, dev, pin), p, err) && p.gather && p.device_bytes > ungathered, "gathers take device memory");
        o = node;
        o.chunk_queries = idx_t(1) << 20;
        CHECK(P::plan(dolma, o, p, err) && p.plan == Plan::A && p.chunk_queries == (idx_t(1) << 20), "dolmasample, larger chunks: %s", err.c_str());
        const idx_t larger = p.device_bytes;
        CHECK(P::plan(dolma, node, p, err) && larger > p.device_bytes, "larger chunks take more device memory");
    }

    // Symbols of 4 bytes take keys of 4 words on the device too; of 1 byte, of 2. And of the engine's width only.
    {
        Placement r;
        CHECK(Query_Engine<uint32_t>::plan(sized_meta(1000000000ull, 4, path), node, r, err) && r.plan == Plan::A && r.key_words == 4,
              "4-byte symbols on the device: %s", err.c_str());
        CHECK(Query_Engine<uint8_t>::plan(sized_meta(1000000000ull, 1, path), node, r, err) && r.plan == Plan::A && r.key_words == 2,
              "1-byte symbols on the device: %s", err.c_str());
        CHECK(Query_Engine<uint32_t>::plan(sized_meta(1000000000ull, 4, path), with_tiers(node, pg, pg), r, err) && r.plan == Plan::C && r.key_words == 4,
              "4-byte symbols on the host: %s", err.c_str());
        CHECK(!Query_Engine<uint8_t>::plan(dolma, node, r, err), "a symbol-width mismatch must be rejected");
    }

    // Budgets unset are measured: the device's free memory, if there is a device, and the host's available memory.
    const bool device = gpu_available();
    CHECK(P::plan(dolma, auto_options(0, node_host), p, err) == device && (!device || p.device_budget > 0), "the device budget, measured: %s",
          device ? err.c_str() : "measured without a device");
    CHECK(P::plan(dolma, auto_options(node_dev, 0), p, err) && p.host_budget > 0 && p.device_budget == node_dev, "the host budget, measured: %s",
          err.c_str());

    return true;
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


// Returns the intervals of the batch `off`, `sym` that the engine finds from
// device memory, through `find_async`.
template <typename T_sym_>
std::vector<SA_Interval> device_find(const Query_Engine<T_sym_>& eng, const std::vector<uint64_t>& off, const std::vector<T_sym_>& sym)
{
    const idx_t q = off.size() - 1;
    uint64_t* off_d = nullptr;
    T_sym_* sym_d = nullptr;
    SA_Interval* I_d = nullptr;
    CUDA_CHECK(cudaMalloc(&off_d, off.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&sym_d, std::max<std::size_t>(sym.size(), 1) * sizeof(T_sym_)));
    CUDA_CHECK(cudaMalloc(&I_d, std::max<idx_t>(q, 1) * sizeof(SA_Interval)));
    CUDA_CHECK(cudaMemcpy(off_d, off.data(), off.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sym_d, sym.data(), sym.size() * sizeof(T_sym_), cudaMemcpyHostToDevice));

    eng.find_async(off_d, sym_d, q, I_d, 0);
    std::vector<SA_Interval> I(q);
    CUDA_CHECK(cudaMemcpy(I.data(), I_d, q * sizeof(SA_Interval), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(off_d));
    CUDA_CHECK(cudaFree(sym_d));
    CUDA_CHECK(cudaFree(I_d));
    return I;
}


// Pinned host copies of a batch and room for its intervals, for the pipeline's
// zero-copy path.
template <typename T_sym_>
struct Pinned_Batch
{
    uint64_t* off = nullptr;
    T_sym_* sym = nullptr;
    SA_Interval* out = nullptr;

    Pinned_Batch(const std::vector<uint64_t>& o, const std::vector<T_sym_>& y)
    {
        CUDA_CHECK(pinned_allocate_host(off, o.size()));
        CUDA_CHECK(pinned_allocate_host(sym, std::max<std::size_t>(y.size(), 1)));
        CUDA_CHECK(pinned_allocate_host(out, std::max<std::size_t>(o.size() - 1, 1)));
        std::copy(o.begin(), o.end(), off);
        std::copy(y.begin(), y.end(), sym);
    }

    ~Pinned_Batch()
    {
        CUDA_CHECK(deallocate_host(off));
        CUDA_CHECK(deallocate_host(sym));
        CUDA_CHECK(deallocate_host(out));
    }

    Pinned_Batch(const Pinned_Batch&) = delete;
    Pinned_Batch& operator=(const Pinned_Batch&) = delete;
};


// Checks that the engine finds the intervals `want` of the batch `off`, `sym`
// every way: through the pipeline from pageable memory, from pinned memory,
// and from pinned queries into pageable intervals; and from device memory.
template <typename T_sym_>
bool check_every_way(const Query_Engine<T_sym_>& eng, const std::vector<uint64_t>& off, const std::vector<T_sym_>& sym,
                     const std::vector<SA_Interval>& want, const std::string& what)
{
    const idx_t q = off.size() - 1;
    std::vector<SA_Interval> got(q, SA_Interval{~uint64_t(0), ~uint64_t(0)});
    eng.find(off.data(), sym.data(), q, got.data());
    Pinned_Batch<T_sym_> pb(off, sym);
    std::fill(pb.out, pb.out + q, SA_Interval{~uint64_t(0), ~uint64_t(0)});
    eng.find(pb.off, pb.sym, q, pb.out);
    std::vector<SA_Interval> mixed(q, SA_Interval{~uint64_t(0), ~uint64_t(0)});
    eng.find(pb.off, pb.sym, q, mixed.data());
    const auto dev = device_find(eng, off, sym);

    for(idx_t i = 0; i < q; ++i)
    {
        CHECK(same(got[i], want[i]), "%s: query %" PRIu64 " (m = %" PRIu64 "): pageable [%" PRIu64 ", %" PRIu64 ") vs oracle [%" PRIu64 ", %" PRIu64 ")",
              what.c_str(), i, off[i + 1] - off[i], got[i].lo, got[i].hi, want[i].lo, want[i].hi);
        CHECK(same(pb.out[i], want[i]), "%s: query %" PRIu64 " (m = %" PRIu64 "): pinned [%" PRIu64 ", %" PRIu64 ") vs oracle [%" PRIu64 ", %" PRIu64 ")",
              what.c_str(), i, off[i + 1] - off[i], pb.out[i].lo, pb.out[i].hi, want[i].lo, want[i].hi);
        CHECK(same(mixed[i], want[i]), "%s: query %" PRIu64 " (m = %" PRIu64 "): pinned into pageable [%" PRIu64 ", %" PRIu64 ") vs oracle [%" PRIu64
              ", %" PRIu64 ")", what.c_str(), i, off[i + 1] - off[i], mixed[i].lo, mixed[i].hi, want[i].lo, want[i].hi);
        CHECK(same(dev[i], want[i]), "%s: query %" PRIu64 " (m = %" PRIu64 "): device [%" PRIu64 ", %" PRIu64 ") vs oracle [%" PRIu64 ", %" PRIu64 ")",
              what.c_str(), i, off[i + 1] - off[i], dev[i].lo, dev[i].hi, want[i].lo, want[i].hi);
    }

    return true;
}


// Checks the engine against the oracle on the fixture text `T` of the symbols
// `alpha`, in every placement, with the suffix array in `parts` parts: in the
// direct mode, and in two phases over every stride of `{1, 2, 3, 16, 64, n +
// 5}`, key width, batch order, and bracket source the placement allows; every
// way the engine finds, with chunks of a few queries, so that the pipeline
// crosses chunks and keeps long queries alone, and gathers of fewer still; and
// the host's search over several counts of lanes. Checks also that the
// placement picks the searcher and the bracket source, and that the
// certificate accepts the intervals and rejects perturbed ones.
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

    struct Mode { idx_t stride; uint32_t key_words; bool sort; Gather gather; };
    std::vector<Mode> modes = {{0, 0, false, Gather::automatic}};
    for(const idx_t stride : {idx_t(1), idx_t(2), idx_t(3), idx_t(16), idx_t(64), n + 5})
        for(const uint32_t key_words : {1u, 2u, 4u})
            for(const bool sort : {false, true})
                for(const Gather gather : {Gather::never, Gather::always})
                    modes.push_back({stride, key_words, sort, gather});

    // Placements, and the lanes of each thread of the host's search, if the host searches, and whether it reads a mapped suffix array's
    // pages ahead.
    struct Placement { Residency text_res, sa_res; unsigned lanes; bool prefetch; };
    for(const Placement pl : {Placement{Residency::device, Residency::device, 16, true}, Placement{Residency::device, Residency::pinned, 16, true},
                              Placement{Residency::pinned, Residency::device, 16, true}, Placement{Residency::pinned, Residency::pinned, 16, true},
                              Placement{Residency::pageable, Residency::pageable, 3, true}, Placement{Residency::pinned, Residency::pageable, 16, true},
                              Placement{Residency::pageable, Residency::pinned, 1, true}, Placement{Residency::pageable, Residency::mapped, 16, true},
                              Placement{Residency::pinned, Residency::mapped, 5, true}, Placement{Residency::pageable, Residency::mapped, 3, false}})
    {
        const std::string where = std::string(what) + " (text " + residency_name(pl.text_res) + ", suffix array " + residency_name(pl.sa_res)
                                  + (pl.prefetch ? "" : ", no prefetching") + ")";
        const bool host = (pl.text_res == Residency::pageable || pl.sa_res == Residency::pageable || pl.sa_res == Residency::mapped);
        Query_Engine<T_sym_> eng;
        Query_Options opt{pl.text_res, pl.sa_res, 100000, 0};
        opt.chunk_queries = 7;
        opt.chunk_symbols = 23;
        opt.gather_queries = 5;
        opt.host_threads = 4;
        opt.host_lanes = pl.lanes;
        opt.host_prefetch = pl.prefetch;
        CHECK(eng.load(dir.string(), opt, err), "%s: %s", where.c_str(), err.c_str());
        CHECK(eng.host_searches() == host, "%s: the placement must pick the searcher", where.c_str());
        CHECK(eng.options().gather == (pl.sa_res == Residency::pinned && !host ? Gather::always : Gather::never),
              "%s: the placement must pick the bracket source", where.c_str());
        for(const auto& md : modes)
        {
            if(host && md.gather == Gather::always)
            {
                CHECK(!eng.configure(md.stride, md.key_words, md.sort, md.gather, err), "%s: gathering must be rejected if the host searches",
                      where.c_str());
                continue;
            }

            CHECK(eng.configure(md.stride, md.key_words, md.sort, md.gather, err), "%s: %s", where.c_str(), err.c_str());
            const std::string mode = where + ", stride " + std::to_string(md.stride) + ", key words " + std::to_string(md.key_words)
                                     + ", sort " + std::to_string(md.sort) + ", gather " + std::to_string(md.gather == Gather::always);
            if(!check_every_way(eng, off, sym, want, mode))
                return false;
        }

        // A mapped suffix array reads its pages anew once they are dropped.
        if(pl.sa_res == Residency::mapped)
        {
            CHECK(eng.drop_cached_sa(err), "%s: %s", where.c_str(), err.c_str());
            if(!check_every_way(eng, off, sym, want, where + ", its pages dropped"))
                return false;
        }

        eng.find(off.data(), sym.data(), q, got.data());

        CHECK(device_certify(eng, off, sym, got) == 0, "%s: the certificate rejects correct intervals", where.c_str());
        auto wrong = got;
        idx_t perturbed = 0;
        for(auto& I : wrong)
            if(I.hi < n)
                ++I.hi, ++perturbed;
        CHECK(device_certify(eng, off, sym, wrong) == perturbed, "%s: the certificate accepts perturbed intervals", where.c_str());
    }

    return true;
}


// Checks that the sample table the engine builds for the fixture text `T` holds
// the zero-padded packed prefixes of every `stride`-th suffix, for keys of `W_`
// words, with the suffix array in `parts` parts: built on the device, on the
// host from either of its tiers, and streamed from a mapped suffix array.
template <typename T_sym_, idx_t W_>
bool check_sample_table(const std::vector<T_sym_>& T, const idx_t stride, const idx_t parts, const char* const what)
{
    const auto dir = scratch_dir("table");
    const auto SA = test::build_sa40(T);
    std::string err;
    CHECK(write_index(dir, T, SA, parts, err), "%s: %s", what, err.c_str());

    const idx_t n = T.size(), J = (n + stride - 1) / stride;
    std::vector<T_sym_> padded(T);
    padded.resize(n + W_ * 8 / sizeof(T_sym_), 0);
    struct Placement { Residency text_res, sa_res; };
    for(const Placement pl : {Placement{Residency::device, Residency::device}, Placement{Residency::pinned, Residency::pinned},
                              Placement{Residency::pageable, Residency::pageable}, Placement{Residency::pageable, Residency::mapped},
                              Placement{Residency::pinned, Residency::mapped}})
    {
        Query_Engine<T_sym_> eng;
        CHECK(eng.load(dir.string(), Query_Options{pl.text_res, pl.sa_res, 0, stride, static_cast<uint32_t>(W_)}, err), "%s: %s", what, err.c_str());
        const auto& tab = eng.table();
        CHECK(tab.size() == J && tab.stride() == stride && tab.key_words() == W_ && tab.key_symbols() == W_ * 8 / sizeof(T_sym_),
              "%s, text %s, suffix array %s: unexpected table shape", what, residency_name(pl.text_res), residency_name(pl.sa_res));

        std::vector<Packed_Prefix<W_>> got(J);
        CUDA_CHECK(cudaMemcpy(got.data(), tab.template keys<W_>(), J * sizeof(Packed_Prefix<W_>), cudaMemcpyDeviceToHost));
        for(idx_t j = 0; j < J; ++j)
            CHECK(got[j] == Packed_Prefix<W_>::pack_prefix(padded.data() + sa40::load(SA.data(), j * stride)),
                  "%s, %" PRIu64 " parts, text %s, suffix array %s: key %" PRIu64 " of stride %" PRIu64 " differs from the host's", what, parts,
                  residency_name(pl.text_res), residency_name(pl.sa_res), j, stride);
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


// Checks the pipeline on batches of many chunks, in the direct mode and in two
// phases, sorted and not, and gathered in tiles that cross chunks, with the
// device searching and with the host: chunks bounded by their queries and by
// their symbols, as the most for a chunk and as its share of a batch split
// across the streams, or whole if gathering; the most's defaults; queries
// longer than a chunk, empty queries, runs of identical queries and their near
// twins, offsets that start past the symbols' first, an empty batch, and
// concurrent calls.
bool test_pipeline()
{
    std::mt19937_64 rng(17);
    const std::vector<uint16_t> alpha = {1, 2, 3, 4, 0xFFFE};
    const auto T = random_text<uint16_t>(50000, alpha, 0xFFFF, 200, rng);
    const idx_t n = T.size();
    const auto SA = test::build_sa40(T);
    const auto dir = scratch_dir("pipeline");
    std::string err;
    CHECK(write_index(dir, T, SA, 2, err), "%s", err.c_str());

    std::vector<uint64_t> off;
    std::vector<uint16_t> sym;
    fixture_queries(T, alpha, rng, 30000, off, sym);
    for(const idx_t m : {idx_t(6000), idx_t(3000)})    // Longer than some chunks, and than a whole chunk's symbols.
    {
        const idx_t pos = rng() % (n - m);
        sym.insert(sym.end(), T.begin() + pos, T.begin() + pos + m);
        off.push_back(sym.size());
    }

    const auto oracle = [&](const std::vector<uint64_t>& o, const std::vector<uint16_t>& y)
    {
        std::vector<SA_Interval> I(o.size() - 1);
        for(idx_t i = 0; i < I.size(); ++i)
            I[i] = ref::find_one(T.data(), n, SA.data(), y.data() + o[i], o[i + 1] - o[i]);
        return I;
    };

    const auto want = oracle(off, sym);
    std::vector<uint64_t> off5(off);    // The same queries, after 5 other symbols.
    for(auto& x : off5)
        x += 5;
    std::vector<uint16_t> sym5(5, 3);
    sym5.insert(sym5.end(), sym.begin(), sym.end());
    const std::vector<uint64_t> off_e(1001, 0); // Empty queries.
    const std::vector<SA_Interval> want_e(1000, SA_Interval{0, n});

    std::vector<uint64_t> off_l;    // Enough queries to split across the streams with the default least shares.
    std::vector<uint16_t> sym_l;
    fixture_queries(T, alpha, rng, 200000, off_l, sym_l);
    const auto want_l = oracle(off_l, sym_l);

    // Runs of identical queries, shuffled, whose sorted gathers cross tiles and chunks: some of the queries above, and twins that share
    // a packed prefix but not a length or the symbols past it, many times over; the longest few times.
    std::vector<uint64_t> off_d(1, 0);
    std::vector<uint16_t> sym_d;
    {
        std::vector<std::vector<uint16_t>> distinct;
        for(idx_t i = 0; i < 40; ++i)
            distinct.emplace_back(sym.begin() + off[i], sym.begin() + off[i + 1]);
        const idx_t pos = rng() % (n - 20);
        std::vector<uint16_t> P(T.begin() + pos, T.begin() + pos + 20);   // Longer than a key.
        distinct.push_back(P);
        P.back() = (P.back() == 1 ? 2 : 1);
        distinct.push_back(P);
        P.push_back(0);
        distinct.push_back(P);
        distinct.push_back({P[0], P[1], P[2]});
        distinct.push_back({P[0], P[1], P[2], 0});

        std::vector<const std::vector<uint16_t>*> runs;
        for(const auto& x : distinct)
            runs.insert(runs.end(), 400, &x);
        const std::vector<uint16_t> long_a(sym.begin() + off[off.size() - 3], sym.begin() + off[off.size() - 2]);
        const std::vector<uint16_t> long_b(sym.begin() + off[off.size() - 2], sym.end());
        runs.insert(runs.end(), 3, &long_a);
        runs.insert(runs.end(), 3, &long_b);
        std::shuffle(runs.begin(), runs.end(), rng);
        for(const auto* x : runs)
        {
            sym_d.insert(sym_d.end(), x->begin(), x->end());
            off_d.push_back(sym_d.size());
        }
    }

    const auto want_d = oracle(off_d, sym_d);

    // The most for a chunk picked by the brackets' source.
    for(const Gather g : {Gather::never, Gather::always})
    {
        Query_Options opt;
        opt.gather = g;
        Query_Engine<uint16_t> eng;
        CHECK(eng.load(dir.string(), opt, err), "%s", err.c_str());
        const bool big = (g == Gather::always);
        CHECK(eng.options().chunk_queries == (idx_t(1) << (big ? 24 : 18)) && eng.options().chunk_symbols == (idx_t(1) << (big ? 28 : 24)),
              "chunks of gathered brackets must default to 2^24 queries and 2^28 symbols, others to 2^18 and 2^24");
    }

    {
        Query_Options opt;
        opt.text_res = opt.sa_res = Residency::pageable;
        Query_Engine<uint16_t> eng;
        CHECK(eng.load(dir.string(), opt, err), "%s", err.c_str());
        CHECK(eng.host_searches() && eng.options().chunk_queries == (idx_t(1) << 18) && eng.options().chunk_symbols == (idx_t(1) << 24),
              "chunks of the host's search must default to 2^18 queries and 2^24 symbols");
    }

    // The most for a chunk, and the least of its share of a batch; with the device searching, and the host, over a suffix array in memory
    // and mapped.
    struct Chunk { idx_t queries, symbols, min_queries, min_symbols; };
    struct Mode { idx_t stride; uint32_t key_words; bool sort; Gather gather; };
    struct Placement { Residency text_res, sa_res; };
    const Query_Options dflt;
    for(const Placement pl : {Placement{Residency::device, Residency::device}, Placement{Residency::pageable, Residency::pageable},
                              Placement{Residency::pageable, Residency::mapped}})
    {
        for(const Chunk ch : {Chunk{dflt.chunk_queries, dflt.chunk_symbols, dflt.min_chunk_queries, dflt.min_chunk_symbols}, Chunk{1000, 5000, 1000, 5000},
                              Chunk{3, 2, 3, 2}, Chunk{dflt.chunk_queries, dflt.chunk_symbols, 1000, 5000}, Chunk{dflt.chunk_queries, dflt.chunk_symbols, 1, 1},
                              Chunk{700, 3000, 1, 1}})
        {
            Query_Options opt;
            opt.text_res = pl.text_res;
            opt.sa_res = pl.sa_res;
            opt.chunk_queries = ch.queries;
            opt.chunk_symbols = ch.symbols;
            opt.min_chunk_queries = ch.min_queries;
            opt.min_chunk_symbols = ch.min_symbols;
            opt.gather_queries = 777;
            Query_Engine<uint16_t> eng;
            CHECK(eng.load(dir.string(), opt, err), "%s", err.c_str());
            for(const Mode md : {Mode{0, 0, false, Gather::never}, Mode{16, 2, false, Gather::never}, Mode{16, 2, true, Gather::never},
                                 Mode{16, 2, true, Gather::always}})
            {
                if(eng.host_searches() && md.gather == Gather::always)  // Rejected; see `check_gpu_fixture`.
                    continue;

                CHECK(eng.configure(md.stride, md.key_words, md.sort, md.gather, err), "%s", err.c_str());
                const std::string what = std::string("text ") + residency_name(pl.text_res) + ", suffix array " + residency_name(pl.sa_res)
                                         + ", chunks of up to " + std::to_string(eng.options().chunk_queries)
                                         + " queries and " + std::to_string(eng.options().chunk_symbols) + " symbols, of at least "
                                         + std::to_string(ch.min_queries) + " and " + std::to_string(ch.min_symbols) + ", stride " + std::to_string(md.stride)
                                         + ", sort " + std::to_string(md.sort) + ", gather " + std::to_string(md.gather == Gather::always);
                if(!check_every_way(eng, off, sym, want, what) || !check_every_way(eng, off5, sym5, want, what + ", offsets from 5")
                    || !check_every_way(eng, off_e, std::vector<uint16_t>(), want_e, what + ", empty queries")
                    || !check_every_way(eng, off_d, sym_d, want_d, what + ", runs of identical queries")
                    || (ch.min_queries == dflt.min_chunk_queries && !check_every_way(eng, off_l, sym_l, want_l, what + ", 200000 queries")))
                    return false;

                eng.find(off.data(), sym.data(), 0, nullptr);   // No queries: nothing to touch.

                // Concurrent calls take their turns.
                std::vector<SA_Interval> got_a(want.size()), got_b(want.size());
                std::thread other([&] { eng.find(off5.data(), sym5.data(), want.size(), got_b.data()); });
                eng.find(off.data(), sym.data(), want.size(), got_a.data());
                other.join();
                for(idx_t i = 0; i < want.size(); ++i)
                    CHECK(same(got_a[i], want[i]) && same(got_b[i], want[i]), "%s: concurrent calls: query %" PRIu64 " differs", what.c_str(), i);
            }
        }
    }

    return true;
}


// Checks the timing of the phases of `find`, and the engine's account of its
// memory, in placements of every path through the pipeline: the device
// searching, gathering its brackets or not, and the host, over a suffix array
// in memory and mapped; in the direct mode and in two phases, from pageable
// memory and from pinned. Timed or not, the intervals must be the oracle's;
// untimed, the phases nil. Timed, every path's own phases must take time and
// the others none, a stream's device phases no more than the call's wall time,
// and the calling thread's no more either.
bool test_phase_times()
{
    std::mt19937_64 rng(29);
    const std::vector<uint16_t> alpha = {1, 2, 3, 4, 0xFFFE};
    const auto T = random_text<uint16_t>(20000, alpha, 0xFFFF, 200, rng);
    const idx_t n = T.size();
    const auto SA = test::build_sa40(T);
    const auto dir = scratch_dir("phases");
    std::string err;
    CHECK(write_index(dir, T, SA, parts_for(n, 2), err), "%s", err.c_str());

    std::vector<uint64_t> off;
    std::vector<uint16_t> sym;
    fixture_queries(T, alpha, rng, 20000, off, sym);
    const idx_t q = off.size() - 1;
    std::vector<SA_Interval> want(q);
    for(idx_t i = 0; i < q; ++i)
        want[i] = ref::find_one(T.data(), n, SA.data(), sym.data() + off[i], off[i + 1] - off[i]);

    Pinned_Batch<uint16_t> pb(off, sym);
    constexpr double SLACK = 1e-4; // Seconds: the resolution of the device's timer, and the clocks' drift.
    struct Placement { Residency text_res, sa_res; };
    for(const Placement pl : {Placement{Residency::device, Residency::device}, Placement{Residency::device, Residency::pinned},
                              Placement{Residency::pageable, Residency::pageable}, Placement{Residency::pageable, Residency::mapped}})
        for(const bool timed : {false, true})
        {
            const std::string where = std::string("text ") + residency_name(pl.text_res) + ", suffix array " + residency_name(pl.sa_res)
                                      + (timed ? ", timed" : ", untimed");
            Query_Options opt;
            opt.text_res = pl.text_res;
            opt.sa_res = pl.sa_res;
            opt.chunk_queries = 3000;
            opt.chunk_symbols = 50000;
            opt.min_chunk_queries = 1000;
            opt.min_chunk_symbols = 5000;
            opt.host_threads = 4;
            opt.phase_times = timed;
            Query_Engine<uint16_t> eng;
            CHECK(eng.load(dir.string(), opt, err), "%s: %s", where.c_str(), err.c_str());
            const bool host = eng.host_searches(), mapped = (pl.sa_res == Residency::mapped);

            // The arrays in their tiers, the table, and the lanes.
            const auto& idx = eng.index();
            const Engine_Memory mem = eng.memory();
            const auto in = [](const Residency r, const Residency tier, const std::size_t b) { return r == tier ? b : std::size_t(0); };
            CHECK(idx.text_alloc_bytes() >= n * sizeof(uint16_t) && idx.sa_alloc_bytes() >= SA_WIDTH * n
                  && mem.text_device == in(pl.text_res, Residency::device, idx.text_alloc_bytes())
                  && mem.sa_device == in(pl.sa_res, Residency::device, idx.sa_alloc_bytes())
                  && mem.text_pinned == in(pl.text_res, Residency::pinned, idx.text_alloc_bytes())
                  && mem.sa_pinned == in(pl.sa_res, Residency::pinned, idx.sa_alloc_bytes()), "%s: the arrays' memory is misreported", where.c_str());
            CHECK(mem.table_device == eng.table().bytes() && mem.table_device > 0 && mem.lanes_device > 0 && mem.lanes_pinned > 0
                  && mem.pool_used_high == 0 && mem.device() == mem.text_device + mem.sa_device + mem.table_device + mem.lanes_device
                  && mem.pinned() == mem.text_pinned + mem.sa_pinned + mem.lanes_pinned, "%s: the engine's memory is misreported", where.c_str());

            for(const idx_t stride : {idx_t(0), idx_t(16)})
            {
                CHECK(eng.configure(stride, 2, false, Gather::automatic, err), "%s: %s", where.c_str(), err.c_str());
                const bool direct_host = (host && stride == 0);
                for(const bool pinned : {false, true})
                {
                    const std::string what = where + ", stride " + std::to_string(stride) + (pinned ? ", pinned" : ", pageable");
                    std::vector<SA_Interval> got(q, SA_Interval{~uint64_t(0), ~uint64_t(0)});
                    std::fill(pb.out, pb.out + q, SA_Interval{~uint64_t(0), ~uint64_t(0)});
                    SA_Interval* const out = (pinned ? pb.out : got.data());
                    eng.find(pinned ? pb.off : off.data(), pinned ? pb.sym : sym.data(), q, out);
                    for(idx_t i = 0; i < q; ++i)
                        CHECK(same(out[i], want[i]), "%s: query %" PRIu64 " differs from the oracle", what.c_str(), i);

                    const Phase_Times ph = eng.phases();
                    const auto t = [&](const Phase p) { return ph[p]; };
                    if(!timed)
                    {
                        CHECK(ph.wall == 0 && ph.chunks == 0 && std::all_of(ph.t.begin(), ph.t.end(), [](const double x) { return x == 0; }),
                              "%s: untimed phases must be nil", what.c_str());
                        continue;
                    }

                    CHECK(ph.wall > 0 && std::all_of(ph.t.begin(), ph.t.end(), [](const double x) { return x >= 0; }),
                          "%s: a phase's time is negative, or the wall time nil", what.c_str());
                    CHECK(direct_host ? ph.chunks == 0 : ph.chunks >= 2, "%s: %" PRIu64 " chunks", what.c_str(), ph.chunks);
                    CHECK(direct_host || (t(Phase::h2d) > 0 && t(Phase::d2h) > 0), "%s: the copies took no time", what.c_str());
                    CHECK((t(Phase::rank) > 0) == (stride > 0), "%s: ranking took %.9f s", what.c_str(), t(Phase::rank));
                    CHECK((t(Phase::search) > 0) == !host && (t(Phase::host_search) > 0) == host, "%s: the device's search took %.9f s, the host's"
                          " %.9f s", what.c_str(), t(Phase::search), t(Phase::host_search));
                    CHECK((t(Phase::prefetch) > 0) == (mapped && stride > 0), "%s: asking for pages took %.9f s", what.c_str(), t(Phase::prefetch));
                    CHECK((t(Phase::stage) > 0) == (!pinned && !direct_host), "%s: staging took %.9f s", what.c_str(), t(Phase::stage));

                    // A stream's device phases run one after another within the call, as do the calling thread's host phases. Staged, the
                    // device's search runs a dispatching thread per stream, each its chunks' staging and waits.
                    const double dev = t(Phase::h2d) + t(Phase::rank) + t(Phase::search) + t(Phase::d2h);
                    const double own = t(Phase::prefetch) + t(Phase::host_search) + t(Phase::wait) + t(Phase::stage);
                    const double threads = (!host && !pinned ? double(Query_Engine<uint16_t>::streams()) : 1.0);
                    CHECK(dev <= Query_Engine<uint16_t>::streams() * ph.wall + SLACK && own <= threads * ph.wall + SLACK, "%s: the device's phases"
                          " took %.6f s, the host's %.6f s, in %.6f s", what.c_str(), dev, own, ph.wall);
                }

                // From device memory: the device's search takes the pool's scratch, and the host's search is timed as a call.
                const auto dev = device_find(eng, off, sym);
                for(idx_t i = 0; i < q; ++i)
                    CHECK(same(dev[i], want[i]), "%s, stride %" PRIu64 ", from device memory: query %" PRIu64 " differs from the oracle",
                          where.c_str(), stride, i);
                CHECK(host || stride == 0 || eng.memory().pool_used_high > 0, "%s: the pool's scratch is unaccounted for", where.c_str());
                if(timed && host)
                    CHECK(eng.phases().wall > 0 && eng.phases()[Phase::host_search] > 0, "%s: the host's search from device memory is untimed",
                          where.c_str());
            }
        }

    return true;
}


bool test_sample_table()
{
    std::mt19937_64 rng(11);
    const std::vector<uint8_t> a8 = {0, 1, 2, 0x7F, 0x80, 0xFF};
    const std::vector<uint16_t> a16 = {0, 1, 0x00FF, 0x0100, 0xFFFE, 0xFFFF};
    const std::vector<uint32_t> a32 = {0, 1, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu};
    for(const idx_t n : {1, 5, 1000, 20000})
    {
        const auto T8 = random_text<uint8_t>(n, a8, 0xFF, 30, rng);
        const auto T16 = random_text<uint16_t>(n, a16, 0xFFFF, 30, rng);
        const auto T32 = random_text<uint32_t>(n, a32, 0xFFFFFFFFu, 30, rng);
        const idx_t p = parts_for(n, 3);    // At n = 20000, entries straddle the parts' ends.
        for(const idx_t stride : {idx_t(1), idx_t(3), idx_t(64)})
            if(!check_sample_table<uint8_t, 1>(T8, stride, p, "u8 W=1") || !check_sample_table<uint8_t, 2>(T8, stride, p, "u8 W=2")
                || !check_sample_table<uint8_t, 4>(T8, stride, p, "u8 W=4") || !check_sample_table<uint16_t, 1>(T16, stride, p, "u16 W=1")
                || !check_sample_table<uint16_t, 2>(T16, stride, p, "u16 W=2") || !check_sample_table<uint16_t, 4>(T16, stride, p, "u16 W=4")
                || !check_sample_table<uint32_t, 1>(T32, stride, p, "u32 W=1") || !check_sample_table<uint32_t, 2>(T32, stride, p, "u32 W=2")
                || !check_sample_table<uint32_t, 4>(T32, stride, p, "u32 W=4"))
                return false;
    }

    return true;
}


// Checks the engine's automatic placement of a fixture index, against the
// oracle every way the engine finds: every pick made, within the measured
// budgets, and with the device's budget too small for all but the host's
// search; tiers named with the other picked, and with the table picked; the
// table picked again when switching modes; the device memory of a load and
// its finds, within the planned; and a budget that fits no plan.
bool test_gpu_auto()
{
    std::mt19937_64 rng(31);
    const std::vector<uint16_t> alpha = {1, 2, 3, 4, 0xFFFE};
    const auto T = random_text<uint16_t>(20000, alpha, 0xFFFF, 100, rng);
    const auto SA = test::build_sa40(T);
    const auto dir = scratch_dir("auto");
    std::string err;
    Index_Meta m;
    CHECK(write_index(dir, T, SA, parts_for(T.size(), 2), err) && index_meta::read(dir.string(), m, err), "%s", err.c_str());

    std::vector<uint64_t> off;
    std::vector<uint16_t> sym;
    fixture_queries(T, alpha, rng, 400, off, sym);
    const idx_t q = off.size() - 1;
    std::vector<SA_Interval> want(q);
    for(idx_t i = 0; i < q; ++i)
        want[i] = ref::find_one(T.data(), T.size(), SA.data(), sym.data() + off[i], off[i + 1] - off[i]);

    // Loads with `opt`, and checks the placement's plan, tiers, and table, and the device memory that the load and its finds take; and
    // the intervals.
    const auto check = [&](const Query_Options& opt, const Plan plan, const Residency text_res, const Residency sa_res, const idx_t stride,
                           const uint32_t key_words, const std::string& what)
    {
        Query_Engine<uint16_t> eng;
        std::size_t free0 = 0, free1 = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free0, &total));
        CHECK(eng.load(dir.string(), opt, err), "%s: %s", what.c_str(), err.c_str());
        CUDA_CHECK(cudaMemGetInfo(&free1, &total));
        const auto& p = eng.placement();
        CHECK(p.plan == plan && p.text_res == text_res && p.sa_res == sa_res && p.stride == stride && p.key_words == key_words,
              "%s: plan %s, text %s, suffix array %s, stride %" PRIu64 ", %u key words", what.c_str(), plan_name(p.plan), residency_name(p.text_res),
              residency_name(p.sa_res), p.stride, p.key_words);
        CHECK(eng.index().text_residency() == text_res && eng.index().sa_residency() == sa_res && eng.options().text_res == text_res
              && eng.options().sa_res == sa_res && eng.table().stride() == stride && eng.table().key_words() == key_words
              && eng.host_searches() == p.host_search && (eng.options().gather == Gather::always) == p.gather
              && eng.options().chunk_queries == p.chunk_queries && eng.options().chunk_symbols == p.chunk_symbols,
              "%s: the engine is not loaded as placed", what.c_str());

        const idx_t slack = idx_t(64) << 20;   // The device's own: the code of kernels loaded on first use.
        CHECK(free0 - free1 <= p.device_bytes + slack, "%s: the load takes %zu bytes of device memory; %" PRIu64 " planned", what.c_str(),
              free0 - free1, p.device_bytes);
        if(!check_every_way(eng, off, sym, want, what))
            return false;

        CUDA_CHECK(cudaMemGetInfo(&free1, &total));
        CHECK(free0 - free1 <= p.device_bytes + slack, "%s: the load and its finds take %zu bytes of device memory; %" PRIu64 " planned",
              what.c_str(), free0 - free1, p.device_bytes);
        return true;
    };

    const auto dev = Residency::device, pin = Residency::pinned, pg = Residency::pageable, map = Residency::mapped, aut = Residency::automatic;

    // Every pick made, within the measured budgets: the fixture fits any device. The suffix array named pinned: gathered, in large chunks.
    if(!check(auto_options(), Plan::A, dev, dev, 32, 2, "every pick made")
        || !check(with_tiers(auto_options(), aut, pin), Plan::B, dev, pin, 32, 2, "the suffix array named pinned")
        || !check(with_tiers(auto_options(), aut, map), Plan::D, pg, map, 32, 4, "the suffix array named mapped")
        || !check(with_tiers(auto_options(), pg, pg), Plan::C, pg, pg, 32, 4, "the tiers named pageable"))
        return false;

    // The device's budget too small for the text and the lanes: the host searches, with a table of 4-word keys.
    Placement c;
    CHECK(Query_Engine<uint16_t>::plan(m, with_tiers(auto_options(), pg, pg), c, err), "%s", err.c_str());
    if(!check(auto_options(c.device_bytes + (idx_t(1) << 30)), Plan::C, pg, pg, 32, 4, "a device budget for plan C only"))
        return false;

    // Named tiers and table, with the table picked anew when switching modes: the planner's, whatever the load's.
    {
        Query_Engine<uint16_t> eng;
        Query_Options opt{dev, dev, 1000, 16, 1};
        CHECK(eng.load(dir.string(), opt, err) && eng.placement().plan == Plan::A && !eng.placement().auto_stride && eng.table().stride() == 16
              && eng.table().key_words() == 1, "named tiers and table: %s", err.c_str());
        struct Mode { idx_t stride; uint32_t key_words; idx_t want_stride; uint32_t want_key_words; };
        for(const Mode md : {Mode{AUTO_STRIDE, AUTO_KEY_WORDS, 32, 2}, Mode{AUTO_STRIDE, 4, 32, 4}, Mode{64, AUTO_KEY_WORDS, 64, 2}})
        {
            CHECK(eng.configure(md.stride, md.key_words, false, Gather::automatic, err) && eng.table().stride() == md.want_stride
                  && eng.table().key_words() == md.want_key_words && eng.options().stride == md.want_stride, "an automatic table: %s", err.c_str());
            if(!check_every_way(eng, off, sym, want, "an automatic table of stride " + std::to_string(md.want_stride)))
                return false;
        }
    }

    // No plan fits a device budget of no more than its margin; nor do the named tiers an automatic table.
    {
        Query_Engine<uint16_t> eng;
        CHECK(!eng.load(dir.string(), auto_options(idx_t(1) << 30), err) && !eng.index().loaded() && eng.placement().plan == Plan::custom,
              "a device budget of its margin fits no plan");
        CHECK(!eng.load(dir.string(), with_tiers(auto_options(idx_t(1) << 30), dev, dev), err) && !eng.index().loaded(),
              "named tiers in a device budget of its margin fit no automatic table");
        Suffix_Index idx;
        CHECK(!idx.load(dir.string(), aut, dev, 0, err) && !idx.load(dir.string(), dev, aut, 0, err), "an index loads into named tiers only");
    }

    return true;
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

    auto zeros = SA;   // All entries 0: every pair repeats a suffix longer than the comparison limit.
    std::fill(zeros.begin(), zeros.end(), 0);
    CHECK(write_index(dir, T, zeros, 1, err), "%s", err.c_str());
    CHECK(!eng.load(dir.string(), Query_Options{}, err), "a suffix array of repeated entries must be rejected at load");

    CHECK(write_index(dir, T, SA, 1, err), "%s", err.c_str());
    CHECK(eng.load(dir.string(), Query_Options{}, err), "%s", err.c_str());
    Query_Engine<uint8_t> narrow;
    CHECK(!narrow.load(dir.string(), Query_Options{}, err), "a symbol-width mismatch must be rejected");
    CHECK(!eng.load(dir.string(), Query_Options{Residency::pageable, Residency::device, 0}, err),
          "a placement split between device and pageable memory must be rejected");
    CHECK(!eng.load(dir.string(), Query_Options{Residency::device, Residency::mapped, 0}, err),
          "a placement split between device and mapped memory must be rejected");
    CHECK(!eng.load(dir.string(), Query_Options{Residency::mapped, Residency::mapped, 0}, err), "a mapped text must be rejected");
    for(const int k : {0, 1, 2})
    {
        Query_Options opt;
        if(k == 0)
            opt.chunk_queries = idx_t(INT_MAX) + 1;
        else
            (k == 1 ? opt.min_chunk_queries : opt.min_chunk_symbols) = 0;
        CHECK(!eng.load(dir.string(), opt, err), "chunks of over INT_MAX queries, and least shares of 0, must be rejected");
    }

    {
        Query_Options opt;
        opt.gather_queries = 0;
        CHECK(!eng.load(dir.string(), opt, err), "gathers of 0 queries must be rejected");
    }

    for(const unsigned lanes : {0u, host_search::MAX_LANES + 1})
    {
        Query_Options opt;
        opt.host_lanes = lanes;
        CHECK(!eng.load(dir.string(), opt, err), "host searches of %u lanes must be rejected", lanes);
    }

    {
        Query_Options opt{Residency::pageable, Residency::pageable, 0};
        opt.gather = Gather::always;
        CHECK(!eng.load(dir.string(), opt, err), "gathering must be rejected if the host searches");
    }

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

    // A mapped suffix array, its parts in one range of addresses.
    std::string drop_err;
    CHECK(!idx.drop_cached_sa(drop_err), "only a mapped suffix array has pages to drop");
    CHECK(write_index(dir, T, SA, 3, err), "%s", err.c_str());
    {
        Suffix_Index m;
        CHECK(m.load(dir.string(), Residency::pageable, Residency::mapped, 100000, err), "%s", err.c_str());
        CHECK(m.sa_residency() == Residency::mapped && m.host_accessible() && !m.device_accessible() && !m.stats().spot_on_device,
              "unexpected placement of a mapped suffix array");
        CHECK(reinterpret_cast<std::uintptr_t>(m.host_sa()) % SA_PAGE_BYTES == 0, "a mapped suffix array must start at a page");
        CHECK(std::memcmp(m.host_sa(), SA.data(), SA_WIDTH * T.size()) == 0, "mapped suffix array differs from its files");
        for(idx_t b = 0; b < SA_PAD_BYTES; ++b)
            CHECK(m.host_sa()[SA_WIDTH * T.size() + b] == 0, "padding of a mapped suffix array is not zero");

        CHECK(m.drop_cached_sa(err), "%s", err.c_str());
        CHECK(std::memcmp(m.host_sa(), SA.data(), SA_WIDTH * T.size()) == 0, "mapped suffix array differs once its pages are dropped");
    }

    Suffix_Index m;
    CHECK(!m.load(dir.string(), Residency::mapped, Residency::pageable, 0, err), "a mapped text must be rejected");
    CHECK(!m.load(dir.string(), Residency::device, Residency::mapped, 0, err), "a placement split between device and mapped memory must be rejected");

    CHECK(write_index(dir, T, test::build_sa40_byte_order(T), 1, err), "%s", err.c_str());
    CHECK(!idx.load(dir.string(), Residency::pageable, Residency::pageable, T.size(), err) && !idx.loaded(),
          "a misordered suffix array must be rejected by the host spot-check");
    CHECK(!m.load(dir.string(), Residency::pageable, Residency::mapped, T.size(), err) && !m.loaded(),
          "a misordered mapped suffix array must be rejected by the host spot-check");

    return true;
}


// Checks that streaming a suffix array from its parts covers its bytes once, in
// pieces within the parts that end at multiples of the alignment or at the
// parts' ends, each followed by zero padding; for several counts of parts and
// alignments. Checks also that alignments of partial blocks are rejected.
bool test_scan_sa()
{
    std::mt19937_64 rng(29);
    std::string err;
    for(const idx_t n : {idx_t(1), idx_t(1000), idx_t(20000), idx_t(100003)})
        for(const idx_t k : {idx_t(1), idx_t(2), idx_t(3), idx_t(7)})
        {
            const idx_t parts = parts_for(n, k);
            std::vector<uint8_t> SA(SA_WIDTH * n);  // Any bytes: streaming does not decode them.
            for(auto& b : SA)
                b = static_cast<uint8_t>(rng());

            const auto dir = scratch_dir("scan");
            Index_Meta m;
            CHECK(write_index(dir, std::vector<uint16_t>(n), SA, parts, err) && index_meta::read(dir.string(), m, err), "%s", err.c_str());
            for(const std::size_t align : {std::size_t(4096), std::size_t(20480), std::size_t(1) << 20})
            {
                std::vector<uint8_t> got(SA.size());
                std::vector<int> seen(SA.size(), 0);
                std::mutex mtx;
                std::string bad;
                const auto take = [&](const idx_t off, const uint8_t* const p, const std::size_t len)
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    const std::size_t part = std::upper_bound(m.sa_part_off.begin(), m.sa_part_off.end(), off) - m.sa_part_off.begin() - 1;
                    const idx_t end = off + len, part_end = m.sa_part_off[part] + m.part_bytes(part);
                    if(len == 0 || len > align || end > part_end || (end % align != 0 && end != part_end))
                        bad = "piece [" + std::to_string(off) + ", " + std::to_string(end) + ") is not within a part, aligned";
                    for(idx_t b = 0; b < SA_PAD_BYTES; ++b)
                        if(p[len + b] != 0)
                            bad = "piece [" + std::to_string(off) + ", " + std::to_string(end) + ") is not padded with zeros";
                    for(std::size_t b = 0; b < len; ++b)
                        got[off + b] = p[b], ++seen[off + b];
                };

                const std::string what = "n=" + std::to_string(n) + ", " + std::to_string(parts) + " parts, pieces of " + std::to_string(align);
                CHECK(index_io::scan_sa(m, align, take, err), "%s: %s", what.c_str(), err.c_str());
                CHECK(bad.empty(), "%s: %s", what.c_str(), bad.c_str());
                CHECK(got == SA && std::all_of(seen.begin(), seen.end(), [](const int c) { return c == 1; }),
                      "%s: the pieces do not cover the suffix array once", what.c_str());
            }

            const auto ignore = [](idx_t, const uint8_t*, std::size_t) {};
            CHECK(!index_io::scan_sa(m, 0, ignore, err) && !index_io::scan_sa(m, 1000, ignore, err), "pieces of partial blocks must be rejected");
        }

    return true;
}


// The host search's teams: every item of a loop taken once, as threads `t` of
// the team, over loops of sizes from none to many per thread, back to back so
// that threads late for a loop meet the next; with a thread held up in the
// middle of an item, and from callers running loops at once.
bool test_host_team()
{
    for(const unsigned threads : {1u, 3u, 8u})
    {
        host_search::Team& team = host_search::team(threads);
        CHECK(team.threads() == threads && &host_search::team(threads) == &team, "%u threads: a team of %u threads", threads, team.threads());
        std::vector<std::atomic<uint32_t>> taken(4096);
        std::atomic<unsigned> bad_t(0);
        for(int r = 0; r < 3000; ++r)
        {
            const idx_t items = (r % 7 == 0 ? 0 : (r % 5 == 0 ? 4096 : r % 23));
            for(idx_t k = 0; k < items; ++k)
                taken[k].store(0, std::memory_order_relaxed);
            const auto take = [&](const unsigned t, host_search::Team::Claim& c)
            {
                if(t >= threads)
                    bad_t.fetch_add(1);
                for(idx_t k; (k = c.next()) != host_search::Team::Claim::NONE; )
                    taken[k].fetch_add(1, std::memory_order_relaxed);
            };

            team.run(items, take);
            for(idx_t k = 0; k < items; ++k)
                CHECK(taken[k].load() == 1, "%u threads, loop %d of %" PRIu64 " items: item %" PRIu64 " taken %u times", threads, r, items, k,
                      taken[k].load());
        }

        CHECK(bad_t.load() == 0, "%u threads: a thread outside the team", threads);

        // A thread held up in an item: the loop is done with it, the other items taken by the others meanwhile.
        std::atomic<idx_t> count(0);
        const auto slow = [&](unsigned, host_search::Team::Claim& c)
        {
            for(idx_t k; (k = c.next()) != host_search::Team::Claim::NONE; count.fetch_add(1))
                if(k == 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
        };
        team.run(1000, slow);
        CHECK(count.load() == 1000, "%u threads: %" PRIu64 " of 1000 items done with a thread held up", threads, count.load());
    }

    // Callers at once: their loops run one after another.
    std::atomic<idx_t> sum(0);
    std::vector<std::thread> callers;
    for(int c = 0; c < 4; ++c)
        callers.emplace_back([&]
        {
            const auto add = [&](unsigned, host_search::Team::Claim& cl)
            {
                for(idx_t k; (k = cl.next()) != host_search::Team::Claim::NONE; )
                    sum.fetch_add(k + 1, std::memory_order_relaxed);
            };
            for(int r = 0; r < 200; ++r)
                host_search::team(3).run(100, add);
        });
    for(auto& c : callers)
        c.join();
    CHECK(sum.load() == idx_t(4) * 200 * (100 * 101 / 2), "concurrent callers: sum %" PRIu64, sum.load());

    return true;
}


// Checks the pages of a mapped suffix array that the host asks for ahead, on
// brackets picked for their edges: one whose searches read past its page, one
// within a page, one as wide as the array, one at its end, empty ones, two that
// share pages, and one across the parts' border; with each page asked for once,
// and each of a part from its own file. As a batch of their few queries, which
// the calling thread asks for alone, and of many copies, which the threads share.
bool test_prefetch_pages()
{
    std::string err;
    const idx_t n = 50000, page = static_cast<idx_t>(::sysconf(_SC_PAGESIZE));
    const idx_t last = (SA_WIDTH * n - 1) / page;   // The suffix array's last page.
    const auto dir = scratch_dir("prefetch");
    Index_Meta m;
    CHECK(write_index(dir, std::vector<uint16_t>(n), std::vector<uint8_t>(SA_WIDTH * n), 2, err) && index_meta::read(dir.string(), m, err),
          "%s", err.c_str());
    const idx_t border = m.sa_part_off[1] / page;   // The second part's first page.
    const idx_t r = border * page / SA_WIDTH;   // The rank whose entry ends past the border, or the one before it.

    // A query's brackets, and the bytes its searches read first: `[5 first, 5 (last - 1) + 8)`, or those of the first probe if wider.
    const std::vector<Rank_Range> br =
    {
        {0, 819}, {7, 7},
        {4100, 4105}, {4100, 4105},
        {0, n}, {n, n},
        {n - 3, n}, {n, n},
        {10000, 10010}, {10010, 10830},
        {r - 2, r + 2}, {r - 2, r + 2},
        {5, 5}, {9, 9},
    };
    const std::vector<std::pair<idx_t, idx_t>> bytes =    // The bytes read first, by bracket.
    {
        {0, SA_WIDTH * 818 + 7},
        {SA_WIDTH * 4100, SA_WIDTH * 4104 + 7},
        {SA_WIDTH * (n / 2), SA_WIDTH * (n / 2) + 7},
        {SA_WIDTH * (n - 3), SA_WIDTH * (n - 1) + 7},
        {SA_WIDTH * 10000, SA_WIDTH * 10009 + 7}, {SA_WIDTH * 10010, SA_WIDTH * 10829 + 7},
        {SA_WIDTH * (r - 2), SA_WIDTH * (r + 1) + 7},
    };
    std::vector<uint8_t> want(last + 1, 0);
    for(const auto& [b0, b1] : bytes)
        for(idx_t g = b0 / page; g <= std::min(last, b1 / page); ++g)
            want[g] = 1;
    CHECK(want[border - 1] && want[border], "the brackets must cross the parts' border");

    const idx_t want_c = std::count(want.begin(), want.end(), 1);
    for(const idx_t copies : {idx_t(1), idx_t(100)})
    {
        std::vector<Rank_Range> batch;
        for(idx_t k = 0; k < copies; ++k)
            batch.insert(batch.end(), br.begin(), br.end());

        const idx_t q = batch.size() / 2;
        for(const unsigned threads : {1u, 3u})
        {
            host_search::SA_Files f;
            CHECK(f.open(m.sa_part, m.sa_part_off, threads, err), "%s", err.c_str());
            const std::unique_ptr<std::atomic<uint8_t>[]> asked(new std::atomic<uint8_t>[last + 1]());
            idx_t refused = 0;
            const idx_t c = host_search::prefetch(f, n, batch.data(), q, asked.get(), refused);
            CHECK(refused == 0 && (threads == 1 ? c == want_c : c >= want_c), "%" PRIu64 " queries, %u threads: %" PRIu64 " pages asked for, not %"
                  PRIu64 "; %" PRIu64 " requests refused", q, threads, c, want_c, refused);
            for(idx_t g = 0; g <= last; ++g)
                CHECK(asked[g].load() == want[g], "%" PRIu64 " queries, %u threads: page %" PRIu64 " %s asked for", q, threads, g,
                      want[g] ? "not" : "wrongly");
            CHECK(host_search::prefetch(f, n, batch.data(), q, asked.get(), refused) == 0 && refused == 0,
                  "%" PRIu64 " queries, %u threads: pages asked for again", q, threads);
        }
    }

    return true;
}
}


// Runs the tests whose names contain any of the arguments, or every test
// without arguments; e.g. `test_query pipeline`.
int main(const int argc, char** const argv)
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
        {"host search vs oracle",   test_host_search,           false},
        {"index load, host tiers",  test_index_host_tiers,      false},
        {"suffix-array streaming",  test_scan_sa,               false},
        {"pages read ahead",        test_prefetch_pages,        false},
        {"host search's teams",     test_host_team,             false},
        {"planner",                 test_planner,               false},
        {"sample table",            test_sample_table,          true},
        {"device find vs oracle",   test_gpu_find,              true},
        {"pipeline",                test_pipeline,              true},
        {"phase times and memory",  test_phase_times,           true},
        {"device load rejects",     test_gpu_rejects,           true},
        {"automatic placement",     test_gpu_auto,              true},
    };

    const auto selected = [&](const Test& t)
    {
        return argc < 2 || std::any_of(argv + 1, argv + argc, [&](const char* const a) { return std::strstr(t.name, a) != nullptr; });
    };

    const bool gpu = gpu_available();
    int failed = 0, skipped = 0, run = 0;
    for(const auto& t : tests)
    {
        if(!selected(t))
            continue;

        ++run;
        if(t.gpu && !gpu)
        {
            ++skipped;
            std::fprintf(stderr, "[test_query] %-24s SKIP (no device)\n", t.name);
            continue;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = t.fn();
        failed += !ok;
        std::fprintf(stderr, "[test_query] %-24s %s (%.1f s)\n", t.name, ok ? "PASS" : "FAIL",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::temp_directory_path() / ("gsa-test-query-" + std::to_string(::getpid())), ec);

    std::fprintf(stderr, "[test_query] %d / %d tests failed, %d skipped\n", failed, run, skipped);
    return failed == 0 && run > 0 ? 0 : 1;
}
