
#ifndef SUFFIX_INDEX_HPP
#define SUFFIX_INDEX_HPP



#include "types.hpp"
#include "Index_Meta.hpp"

#include <cstdint>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>


// A query index loaded into memory: its text and its 40-bit suffix array, each
// resident in one memory tier, and padded as the query kernels expect.

namespace gpu_saca
{

// Memory tier of an array of a loaded query index.
enum class Residency
{
    device,     // Device memory.
    pinned,     // Pinned host memory mapped into the device's address space: zero-copy from the device.
    pageable,   // Pageable host memory, with transparent huge pages: host-side access only.
    mapped,     // Its files mapped into host memory, read on demand through the page cache: host-side access only; the suffix array only.
    automatic,  // Any of the above, as the query engine's planner picks it at load; never the tier of a loaded array.
};


// Granularity of the device's allocations: a large one takes its bytes rounded
// up to it.
constexpr idx_t DEVICE_PAGE_BYTES = idx_t(2) << 20;


// Returns the name of the memory tier `r`.
const char* residency_name(Residency r);

// Parses the memory-tier name `name` into `r`. Returns `false` if unknown.
bool parse_residency(const std::string& name, Residency& r);


namespace index_io
{

// Reads the bytes `[off, off + len)` of the file at `path` into `dst` with
// `GPU_SACA_QUERY_IO_THREADS` parallel readers (default 16). The reads bypass
// the page cache if the file system allows it and `off` and `dst` are 4 KB-
// aligned, in which case `dst` must have room for `len` rounded up to 4 KB.
// The range must end at or before the file's end. Returns `false` with a reason
// in `err` on failure.
bool read_range(const std::string& path, std::size_t off, std::size_t len, uint8_t* dst, std::string& err);

// Streams the suffix array of the index `m` from its part files, in pieces that
// end at multiples of `align` bytes into the array, or at a part's end: calls
// `fn(off, p, len)` for every piece, with the array's bytes `[off, off + len)`
// at `p`, followed by `SA_PAD_BYTES` zero bytes. The parts are read at once,
// each by `GPU_SACA_QUERY_IO_THREADS` (default 16) threads, which also make the
// calls; the reads bypass the page cache if the file system allows it. `align`
// must be a positive multiple of 4 KB. Returns `false` with a reason in `err`
// on failure.
bool scan_sa(const Index_Meta& m, std::size_t align, const std::function<void(idx_t, const uint8_t*, std::size_t)>& fn, std::string& err);

// Returns the bytes of host memory that `scan_sa` takes for its readers'
// buffers, streaming the suffix array of the index `m` in pieces of `align`
// bytes, at most.
idx_t scan_sa_bytes(const Index_Meta& m, std::size_t align);

}


// Timings and counts of loading an index.
struct Index_Load_Stats
{
    double t_text = -1; // Time to read and place the text; `-1` if unmeasured.
    double t_sa = -1;   // Time to read and place the suffix array.
    double t_spot = -1; // Time of the suffix-array order spot-check.
    idx_t spot_pairs = 0;   // Adjacent pairs spot-checked.
    bool spot_on_device = false;    // Whether the spot-check ran on the device.
};


// A query index loaded into memory. The text is followed by `TEXT_PAD_BYTES`
// zero bytes, and the suffix array by `SA_PAD_BYTES` zero bytes.
class Suffix_Index
{
private:

    // A loaded array.
    struct Array
    {
        void* base = nullptr;   // Base of the allocation.
        std::size_t alloc_bytes = 0;    // Bytes allocated.
        Residency res = Residency::device;
        void* dev = nullptr;    // Device address of the content, if device-accessible.
        void* host = nullptr;   // Host address of the content, if host-accessible.
        std::size_t registered_bytes = 0;   // Bytes registered with the device, for `pinned`.
        std::size_t mapped_bytes = 0;   // Bytes of the files' pages, for `mapped`.
        std::vector<int> fds;   // The open files, for `mapped`.
    };

    // A file contributing the bytes `[off, off + bytes)` of an array.
    struct Source
    {
        std::string path;
        std::size_t off;
        std::size_t bytes;
    };

    Index_Meta meta_;   // Metadata of the index.
    Array text_;    // The padded text.
    Array sa_;  // The padded suffix array.
    Index_Load_Stats stats_;    // Timings and counts of the load.


    // Allocates an array of `bytes` bytes plus `pad` zero bytes in the tier
    // `res` into `a`, and reads its content from the files `src`, staging
    // device-bound reads through the two `stage_bytes`-sized buffers `stage`.
    // Returns `false` with a reason in `err` on failure, leaving `a` for
    // `free_array`.
    static bool place(Array& a, Residency res, const std::vector<Source>& src, std::size_t bytes, std::size_t pad,
                      uint8_t* const (&stage)[2], std::size_t stage_bytes, std::string& err);

    // Maps the files `src` of an array of `bytes` bytes read-only into one range
    // of host addresses, each at its offset, followed by `pad` zero bytes, into
    // `a`: nothing is read until accessed, and then a page at a time, with no
    // readahead. Returns `false` with a reason in `err` on failure, leaving `a`
    // for `free_array`.
    static bool map_files(Array& a, const std::vector<Source>& src, std::size_t bytes, std::size_t pad, std::string& err);

    // Spot-checks the order of `spot` random adjacent pairs of the suffix array.
    // Returns `false` with a reason in `err` on a violation.
    bool spot_check(idx_t spot, std::string& err);

    // Frees the array `a`.
    static void free_array(Array& a);


public:

    Suffix_Index() = default;
    ~Suffix_Index() { release(); }
    Suffix_Index(const Suffix_Index&) = delete;
    Suffix_Index& operator=(const Suffix_Index&) = delete;
    Suffix_Index(Suffix_Index&& o) noexcept;
    Suffix_Index& operator=(Suffix_Index&& o) noexcept;

    // Loads the index-directory `dir`, placing the text in the tier `text_res`
    // and the suffix array in `sa_res`, and spot-checks the order of `spot`
    // random adjacent pairs of the suffix array. The two arrays must both be
    // accessible either from the device (`device`, `pinned`) or from the host
    // (`pinned`, `pageable`, and `mapped` for the suffix array). Returns `false`
    // with a reason in `err` on failure, leaving the index empty.
    bool load(const std::string& dir, Residency text_res, Residency sa_res, idx_t spot, std::string& err);

    // Returns in `device` and `host` the bytes of device and of host memory that
    // `load` takes for the index of metadata `m` with the text in the tier
    // `text_res` and the suffix array in `sa_res`, at most: each array as
    // allocated in its tier, a mapped one in the page cache only, and the host's
    // staging of device-bound reads, while it reads them.
    static void footprint(const Index_Meta& m, Residency text_res, Residency sa_res, idx_t& device, idx_t& host);

    // Frees the arrays, leaving the index empty.
    void release();

    // Drops the pages of a mapped suffix array from the page cache, so that its
    // next accesses read them from the drives; the pages that another process
    // maps stay. Returns `false` with a reason in `err` on failure, or if the
    // suffix array is not mapped.
    bool drop_cached_sa(std::string& err) const;

    // Returns `true` iff an index is loaded.
    bool loaded() const { return text_.base != nullptr; }

    const Index_Meta& meta() const { return meta_; }
    idx_t n() const { return meta_.n; }
    uint32_t sym_bytes() const { return meta_.sym_bytes; }
    Residency text_residency() const { return text_.res; }
    Residency sa_residency() const { return sa_.res; }
    const Index_Load_Stats& stats() const { return stats_; }

    // Returns `true` iff both arrays are accessible from the device.
    bool device_accessible() const { return text_.dev != nullptr && sa_.dev != nullptr; }

    // Returns `true` iff both arrays are accessible from the host.
    bool host_accessible() const { return text_.host != nullptr && sa_.host != nullptr; }

    // Returns the device address of the padded text; `nullptr` if `pageable`.
    template <typename T_sym_>
    const T_sym_* text() const { return static_cast<const T_sym_*>(text_.dev); }

    // Returns the device address of the padded suffix array; `nullptr` if `pageable`.
    const uint8_t* sa() const { return static_cast<const uint8_t*>(sa_.dev); }

    // Returns the host address of the padded text; `nullptr` if `device`.
    template <typename T_sym_>
    const T_sym_* host_text() const { return static_cast<const T_sym_*>(text_.host); }

    // Returns the host address of the padded suffix array; `nullptr` if `device`.
    // A mapped suffix array starts at a page.
    const uint8_t* host_sa() const { return static_cast<const uint8_t*>(sa_.host); }
};

}



#endif
