
#ifndef QUERY_FILE_HPP
#define QUERY_FILE_HPP



// On-disk formats of query batches (`.qry`) and of their results (intervals:
// `.itv`, counts: `.cnt`). Header-only and CUDA-free, so that the baseline
// drivers can include it. All multi-byte fields are little-endian.
//
// `.qry` layout:
//   [0, 64)                     `Query_File_Header`
//   [64, 64 + 8 (q_c + 1))      `uint64_t off[q_c + 1]`, absolute, `off[0] = 0`
//   then                        `T sym[sym_c]`, the queries' symbols back-to-back
// Query `i` is `sym[off[i], off[i + 1])`. A batch of the first `q` queries is a
// prefix of the file: `off[0, q]` and `sym[0, off[q])`.
//
// `.itv`: `q x {uint64_t lo, hi}`, headerless.
// `.cnt`: `q x uint64_t (hi - lo)`, headerless.


#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>


namespace gpu_saca
{

// SA-interval `[lo, hi)` of a pattern: the ranks of the suffixes having the
// pattern as a prefix. Its count is `hi - lo`.
struct SA_Interval
{
    uint64_t lo;
    uint64_t hi;
};

static_assert(sizeof(SA_Interval) == 16, "`.itv` records must be 16 bytes.");


// Header of a `.qry` batch-file.
struct Query_File_Header
{
    char magic[8];  // `QRY_MAGIC`.
    uint32_t version;   // `QRY_VERSION`.
    uint32_t sym_bytes; // Width of a symbol.
    uint64_t q_c;   // Number of queries.
    uint64_t sym_c; // Total number of symbols across the queries.
    uint64_t min_len;   // Length of the shortest query.
    uint64_t max_len;   // Length of the longest query.
    uint64_t seed;  // Seed of the generator, for provenance.
    uint64_t reserved;
};

static_assert(sizeof(Query_File_Header) == 64, "`.qry` header must be 64 bytes.");

inline constexpr char QRY_MAGIC[8] = {'G', 'S', 'A', 'Q', 'R', 'Y', '0', '1'};
inline constexpr uint32_t QRY_VERSION = 1;


namespace query_file
{

// Reads the first `min(q_max, q_c)` queries of the `.qry` file at `path` into
// the absolute offsets `off` and the symbols `sym`, and its header into `h`.
// Returns `false` with a reason in `err` on failure.
template <typename T_sym_>
bool read(const std::string& path, const uint64_t q_max, std::vector<uint64_t>& off, std::vector<T_sym_>& sym, Query_File_Header& h, std::string& err)
{
    auto* const f = std::fopen(path.c_str(), "rb");
    if(f == nullptr)
    {
        err = "cannot open query file: " + path;
        return false;
    }

    const auto fail = [&](const std::string& why)
    {
        std::fclose(f);
        err = why + ": " + path;
        return false;
    };

    if(std::fread(&h, sizeof(h), 1, f) != 1)
        return fail("truncated query-file header");

    if(std::memcmp(h.magic, QRY_MAGIC, sizeof(QRY_MAGIC)) != 0)
        return fail("not a query file (bad magic)");

    if(h.version != QRY_VERSION)
        return fail("unsupported query-file version " + std::to_string(h.version));

    if(h.sym_bytes != sizeof(T_sym_))
        return fail("query-file symbol width " + std::to_string(h.sym_bytes) + " != expected " + std::to_string(sizeof(T_sym_)));

    const uint64_t q = (q_max < h.q_c ? q_max : h.q_c);
    off.resize(q + 1);
    if(std::fread(off.data(), sizeof(uint64_t), q + 1, f) != q + 1)
        return fail("truncated query offsets");

    if(off[0] != 0)
        return fail("query offsets must start at 0");

    for(uint64_t i = 0; i < q; ++i)
        if(off[i + 1] < off[i])
            return fail("query offsets are not non-decreasing");

    if(off[q] > h.sym_c)
        return fail("query offsets exceed the symbol count");

    // Skip the offsets of the queries not requested.
    const auto skip = static_cast<long>((h.q_c - q) * sizeof(uint64_t));
    if(skip > 0 && std::fseek(f, skip, SEEK_CUR) != 0)
        return fail("cannot seek past the query offsets");

    sym.resize(off[q]);
    if(!sym.empty() && std::fread(sym.data(), sizeof(T_sym_), sym.size(), f) != sym.size())
        return fail("truncated query symbols");

    std::fclose(f);
    return true;
}


// Writes the queries `sym[off[i], off[i + 1])` for `i < off.size() - 1` as a
// `.qry` file at `path`, stamped with `seed`. Returns `false` with a reason in
// `err` on failure.
template <typename T_sym_>
bool write(const std::string& path, const std::vector<uint64_t>& off, const std::vector<T_sym_>& sym, const uint64_t seed, std::string& err)
{
    if(off.empty() || off[0] != 0 || off.back() != sym.size())
    {
        err = "malformed query offsets";
        return false;
    }

    Query_File_Header h{};
    std::memcpy(h.magic, QRY_MAGIC, sizeof(QRY_MAGIC));
    h.version   = QRY_VERSION;
    h.sym_bytes = sizeof(T_sym_);
    h.q_c       = off.size() - 1;
    h.sym_c     = sym.size();
    h.min_len   = (h.q_c == 0 ? 0 : UINT64_MAX);
    h.max_len   = 0;
    for(uint64_t i = 0; i < h.q_c; ++i)
    {
        const auto len = off[i + 1] - off[i];
        h.min_len = (len < h.min_len ? len : h.min_len);
        h.max_len = (len > h.max_len ? len : h.max_len);
    }

    h.seed = seed;

    // Temporary plus `rename`, so a present file is a complete file.
    const std::string tmp = path + ".tmp";
    auto* const f = std::fopen(tmp.c_str(), "wb");
    if(f == nullptr)
    {
        err = "cannot create query file: " + tmp;
        return false;
    }

    const bool ok = (std::fwrite(&h, sizeof(h), 1, f) == 1
                        && std::fwrite(off.data(), sizeof(uint64_t), off.size(), f) == off.size()
                        && (sym.empty() || std::fwrite(sym.data(), sizeof(T_sym_), sym.size(), f) == sym.size()));
    if(std::fclose(f) != 0 || !ok || std::rename(tmp.c_str(), path.c_str()) != 0)
    {
        err = "cannot write query file: " + path;
        return false;
    }

    return true;
}


// Writes the intervals `I[0, q)` as a `.itv` file at `path`. Returns `false` on
// failure.
inline bool write_intervals(const std::string& path, const SA_Interval* const I, const uint64_t q)
{
    auto* const f = std::fopen(path.c_str(), "wb");
    if(f == nullptr)
        return false;

    const bool ok = (q == 0 || std::fwrite(I, sizeof(SA_Interval), q, f) == q);
    return std::fclose(f) == 0 && ok;
}


// Writes the counts of the intervals `I[0, q)` as a `.cnt` file at `path`.
// Returns `false` on failure.
inline bool write_counts(const std::string& path, const SA_Interval* const I, const uint64_t q)
{
    auto* const f = std::fopen(path.c_str(), "wb");
    if(f == nullptr)
        return false;

    constexpr uint64_t buf_c = 1 << 16;
    std::vector<uint64_t> buf(buf_c);
    bool ok = true;
    for(uint64_t b = 0; b < q && ok; b += buf_c)
    {
        const uint64_t e = (b + buf_c < q ? b + buf_c : q);
        for(uint64_t i = b; i < e; ++i)
            buf[i - b] = I[i].hi - I[i].lo;

        ok = std::fwrite(buf.data(), sizeof(uint64_t), e - b, f) == e - b;
    }

    return std::fclose(f) == 0 && ok;
}


// Returns the 64-bit FNV-1a hash of the bytes of the `.cnt` representation of
// the intervals `I[0, q)`, i.e. of the little-endian `uint64_t` counts. Equals
// the FNV-1a hash of the `.cnt` file's content, so that engines' results can be
// compared through records alone.
inline uint64_t counts_hash(const SA_Interval* const I, const uint64_t q)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for(uint64_t i = 0; i < q; ++i)
    {
        const uint64_t c = I[i].hi - I[i].lo;
        for(int b = 0; b < 8; ++b)
        {
            h ^= (c >> (8 * b)) & 0xFF;
            h *= 0x100000001b3ull;
        }
    }

    return h;
}

}

}



#endif
