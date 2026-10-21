
#ifndef INDEX_META_HPP
#define INDEX_META_HPP



#include "types.hpp"
#include "sa40.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <fstream>
#include <cstdio>
#include <ctime>


// Metadata of a query index: the `index.meta` sidecar of an index-directory,
// holding `key=value` lines. The arrays themselves are raw and headerless.

namespace gpu_saca
{

// Symbol order of the suffix arrays the query engine supports: numeric symbol
// order, with the end-of-text smaller than every symbol.
inline constexpr const char* SA_ORDER_NUMERIC = "numeric_eot_smallest";

// Name of the metadata file in an index-directory.
inline constexpr const char* INDEX_META_FILE = "index.meta";

// Page size of the on-disk suffix-array parts: part boundaries are aligned to
// it, so that no page straddles two parts.
constexpr idx_t SA_PAGE_BYTES = 4096;


// Metadata of a query index.
struct Index_Meta
{
    idx_t n = 0;    // Length of the text, in symbols.
    uint32_t sym_bytes = 0; // Width of a symbol.
    uint32_t sa_width = SA_WIDTH;   // Width of a suffix-array entry.
    std::string sa_order = SA_ORDER_NUMERIC;    // Order of the suffixes in the array.
    std::string text;   // Path of the text.
    std::vector<std::string> sa_part;   // Paths of the suffix array's parts, in byte order.
    std::vector<idx_t> sa_part_off; // Byte-offset of each part into the suffix array's byte-stream.
    int64_t sep = -1;   // Document-separator symbol, if any; `-1` otherwise.
    std::string sa_source;  // Provenance of the suffix array.
    std::string created_utc;    // Creation timestamp.

    // Returns the size of the suffix array in bytes.
    idx_t sa_bytes() const { return sa_width * n; }

    // Returns the size of the `i`'th part of the suffix array in bytes.
    idx_t part_bytes(const std::size_t i) const { return (i + 1 < sa_part_off.size() ? sa_part_off[i + 1] : sa_bytes()) - sa_part_off[i]; }
};


namespace index_meta
{

// Returns the current time in UTC, formatted in ISO 8601.
inline std::string now_utc()
{
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}


// Returns the page-aligned byte-offsets at which a suffix-array byte-stream of
// `sa_bytes` bytes is split into `k` near-equal parts. The first offset is `0`.
inline std::vector<idx_t> part_offsets(const idx_t sa_bytes, const idx_t k)
{
    std::vector<idx_t> off(k);
    for(idx_t i = 0; i < k; ++i)
    {
        const idx_t cut = static_cast<idx_t>((static_cast<unsigned __int128>(sa_bytes) * i) / k);
        off[i] = cut - cut % SA_PAGE_BYTES;
    }

    return off;
}


// Checks the internal consistency of `m`. Returns `false` with a reason in `err`
// if inconsistent.
inline bool validate(const Index_Meta& m, std::string& err)
{
    if(m.n == 0)    { err = "index has an empty text"; return false; }
    if(m.n >= SA_MAX_N) { err = "text too long for 40-bit suffix-array entries"; return false; }
    if(m.sym_bytes != 1 && m.sym_bytes != 2 && m.sym_bytes != 4)    { err = "unsupported symbol width " + std::to_string(m.sym_bytes); return false; }
    if(m.sa_width != SA_WIDTH)  { err = "unsupported suffix-array width " + std::to_string(m.sa_width); return false; }
    if(m.sa_order != SA_ORDER_NUMERIC)  { err = "unsupported suffix-array order `" + m.sa_order + "`"; return false; }
    if(m.text.empty())  { err = "index has no text path"; return false; }
    if(m.sa_part.empty() || m.sa_part.size() != m.sa_part_off.size())   { err = "malformed suffix-array parts"; return false; }
    if(m.sa_part_off[0] != 0)   { err = "the first suffix-array part must start at byte 0"; return false; }

    for(std::size_t i = 1; i < m.sa_part_off.size(); ++i)
        if(m.sa_part_off[i] <= m.sa_part_off[i - 1] || m.sa_part_off[i] % SA_PAGE_BYTES != 0 || m.sa_part_off[i] >= m.sa_bytes())
        {
            err = "suffix-array part offsets must be increasing, page-aligned, and in-range";
            return false;
        }

    return true;
}


// Checks that the files of the index `m` exist and have the sizes it implies.
// Returns `false` with a reason in `err` otherwise.
inline bool validate_files(const Index_Meta& m, std::string& err)
{
    std::error_code ec;
    const auto text_bytes = std::filesystem::file_size(m.text, ec);
    if(ec)
    {
        err = "cannot stat text: " + m.text;
        return false;
    }

    if(text_bytes != m.n * m.sym_bytes)
    {
        err = "text " + m.text + " has " + std::to_string(text_bytes) + " bytes; expected " + std::to_string(m.n * m.sym_bytes);
        return false;
    }

    for(std::size_t i = 0; i < m.sa_part.size(); ++i)
    {
        const auto part_bytes = std::filesystem::file_size(m.sa_part[i], ec);
        if(ec)  { err = "cannot stat suffix-array part: " + m.sa_part[i]; return false; }
        if(part_bytes != m.part_bytes(i))
        {
            err = "suffix-array part " + m.sa_part[i] + " has " + std::to_string(part_bytes) + " bytes; expected " + std::to_string(m.part_bytes(i));
            return false;
        }
    }

    return true;
}


// Reads the metadata of the index-directory `dir` into `m`, resolving relative
// paths against `dir`. Returns `false` with a reason in `err` on failure.
inline bool read(const std::string& dir, Index_Meta& m, std::string& err)
{
    const std::string path = (std::filesystem::path(dir) / INDEX_META_FILE).string();
    std::ifstream is(path);
    if(!is)
    {
        err = "cannot open index metadata: " + path;
        return false;
    }

    std::map<std::string, std::string> kv;
    for(std::string line; std::getline(is, line); )
    {
        if(line.empty() || line[0] == '#')
            continue;

        const auto eq = line.find('=');
        if(eq == std::string::npos) { err = "malformed line in " + path + ": `" + line + "`"; return false; }
        kv[line.substr(0, eq)] = line.substr(eq + 1);
    }

    const auto resolve = [&](const std::string& p) { return std::filesystem::path(p).is_absolute() ? p : (std::filesystem::path(dir) / p).string(); };

    try
    {
        if(kv["format"] != "gpu_saca_query_index")  { err = "not a query-index metadata file: " + path; return false; }
        if(std::stoul(kv.at("format_version")) != 1)    { err = "unsupported index-metadata version in " + path; return false; }

        m.n = std::stoull(kv.at("n"));
        m.sym_bytes = static_cast<uint32_t>(std::stoul(kv.at("sym_bytes")));
        m.sa_width = static_cast<uint32_t>(std::stoul(kv.at("sa_width")));
        m.sa_order = kv.at("sa_order");
        m.text = resolve(kv.at("text"));

        const auto part_c = std::stoull(kv.at("sa_part_c"));
        m.sa_part.clear();
        m.sa_part_off.clear();
        for(uint64_t i = 0; i < part_c; ++i)
        {
            m.sa_part.push_back(resolve(kv.at("sa_part_" + std::to_string(i))));
            m.sa_part_off.push_back(std::stoull(kv.at("sa_part_off_" + std::to_string(i))));
        }

        m.sep = (kv.count("sep") ? std::stoll(kv["sep"]) : -1);
        m.sa_source = kv["sa_source"];
        m.created_utc = kv["created_utc"];
    }
    catch(const std::exception&)
    {
        err = "missing or malformed field in " + path;
        return false;
    }

    return validate(m, err);
}


// Writes the metadata `m` into the index-directory `dir` atomically. Returns
// `false` with a reason in `err` on failure.
inline bool write(const std::string& dir, const Index_Meta& m, std::string& err)
{
    if(!validate(m, err))
        return false;

    std::ostringstream o;
    o << "format=gpu_saca_query_index\n" << "format_version=1\n"
      << "n=" << m.n << "\n" << "sym_bytes=" << m.sym_bytes << "\n" << "sa_width=" << m.sa_width << "\n"
      << "sa_order=" << m.sa_order << "\n" << "text=" << m.text << "\n" << "sa_part_c=" << m.sa_part.size() << "\n";

    for(std::size_t i = 0; i < m.sa_part.size(); ++i)
        o << "sa_part_" << i << "=" << m.sa_part[i] << "\n" << "sa_part_off_" << i << "=" << m.sa_part_off[i] << "\n";

    o << "sep=" << m.sep << "\n" << "sa_source=" << m.sa_source << "\n"
      << "created_utc=" << (m.created_utc.empty() ? now_utc() : m.created_utc) << "\n";

    const std::string path = (std::filesystem::path(dir) / INDEX_META_FILE).string(), tmp = path + ".tmp";
    {
        std::ofstream os(tmp);
        os << o.str();
        if(!os)
        {
            err = "cannot write index metadata: " + tmp;
            return false;
        }
    }

    if(std::rename(tmp.c_str(), path.c_str()) != 0)
    {
        err = "cannot rename index metadata into place: " + path;
        return false;
    }

    return true;
}

}

}



#endif
