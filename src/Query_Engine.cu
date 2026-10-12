
#include "Query_Engine.hpp"
#include "CPU_Reference.hpp"
#include "Index_Meta.hpp"
#include "Packed_Prefix_Decomposer.cuh"
#include "sa40.hpp"
#include "utility.cuh"

#include <cub/cub.cuh>

#include <sys/vfs.h>
#include <unistd.h>

#include <atomic>
#include <climits>
#include <type_traits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <algorithm>


namespace gpu_saca
{

namespace
{

constexpr int BLOCK = 256;
constexpr idx_t WINDOW_MAX_BYTES = 1024;    // Bytes of a bracket's slot of gathered suffix-array entries, at most.
constexpr idx_t WINDOW_PAD_BYTES = 16;  // Bytes past the last slot: the decoding of an entry reads the word past its last byte.
constexpr idx_t NO_SLOT = ~idx_t(0);
constexpr idx_t MARKS_PER_ITEM = idx_t(1) << 20;  // Page marks that a thread clears at a time.
constexpr idx_t AUTO_STRIDE_MIN = 32;   // The first stride the planner tries for a table,
constexpr idx_t AUTO_STRIDE_MAX = 4096; // and the last: brackets of wider gaps are not much narrower than the whole suffix array.
constexpr idx_t MARGIN_PARTS = 16;  // Of a memory budget, the planner leaves a part this small free,
constexpr idx_t MARGIN_MIN_BYTES = idx_t(1) << 30;  // and no less than this.
constexpr idx_t SORT_SCRATCH_PER_KEY = 8;   // Bytes of a radix sort's scratch per key, at most: a tile's 256 4-byte digit counts, per tile of 128 keys or more.
constexpr idx_t SCAN_SCRATCH_PER_KEY = 1;   // Bytes of a scan's scratch per item, at most: its tiles' states.
constexpr idx_t SCRATCH_BASE_BYTES = idx_t(1) << 20;    // Bytes of either's scratch beyond, at most.


// Records on the stream `s` the start of a device phase on the clock `c`,
// unless null.
inline void phase_start(Phase_Clock* const c, const cudaStream_t s)
{
    if(c != nullptr)
        c->start(s);
}


// Records on the stream `s` the end of the device phase `p` on the clock `c`,
// unless null.
inline void phase_lap(Phase_Clock* const c, const Phase p, const cudaStream_t s)
{
    if(c != nullptr)
        c->lap(p, s);
}


// Returns the time now if the clock `c` is not null: the start of a host's
// phase on it.
inline std::chrono::steady_clock::time_point host_start(const Phase_Clock* const c)
{
    return c != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
}


// Adds the host's seconds since `t0` to the phase `p` of the clock `c`, unless
// null.
inline void host_end(Phase_Clock* const c, const Phase p, const std::chrono::steady_clock::time_point t0)
{
    if(c != nullptr)
        c->add(p, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}


// Gathered suffix-array bytes of a bracket: the bytes `[b0, b1)` of the suffix
// array, at byte `dst` of its tile's windows; `dst` is `NO_SLOT` for a bracket
// searched in place.
struct Window
{
    idx_t b0;
    idx_t b1;
    idx_t dst;
};


// Returns the window of the bracket `r` of the query at position `g` of a tile,
// its upper one if `upper`, whose other bracket is `o`. A bracket's search
// probes the entries at ranks `[r.first, r.last)`, which the window holds in
// whole 16-byte words: in the query's pair of `wb`-byte slots, shared by its two
// brackets, if their union fits in the pair; else in the bracket's own slot if
// it fits in that; else in none.
__device__ inline Window window_of(const Rank_Range r, const Rank_Range o, const bool upper, const idx_t g, const idx_t wb)
{
    const idx_t u0 = (SA_WIDTH * (r.first < o.first ? r.first : o.first)) & ~idx_t(15);
    const idx_t u1 = (SA_WIDTH * (r.last > o.last ? r.last : o.last) + 15) & ~idx_t(15);
    if(u1 - u0 <= 2 * wb)
        return Window{u0, u1, 2 * wb * g};

    const idx_t b0 = (SA_WIDTH * r.first) & ~idx_t(15), b1 = (SA_WIDTH * r.last + 15) & ~idx_t(15);
    return (b1 - b0 <= wb ? Window{b0, b1, 2 * wb * g + (upper ? wb : 0)} : Window{0, 0, NO_SLOT});
}


// Returns the bytes of a slot of gathered suffix-array entries that holds any
// bracket of a table of stride `stride` with keys of `K` symbols. A bracket
// probes at most `stride + K - 2` ranks: those of a sample gap, and the `K - 1`
// shorter suffixes that match the key of a pattern ending in `0`. Their bytes,
// rounded out to 16-byte words, gain at most 15 at each end.
inline idx_t bracket_window_bytes(const idx_t stride, const idx_t K)
{
    return (SA_WIDTH * (stride + K - 2) + 2 * 15 + 15) & ~idx_t(15);
}


// Returns the bytes `b` of a device allocation in whole device pages.
inline idx_t device_pages(const idx_t b)
{
    return (std::max<idx_t>(b, 1) + DEVICE_PAGE_BYTES - 1) / DEVICE_PAGE_BYTES * DEVICE_PAGE_BYTES;
}


// Returns the bytes `b` as decimal gigabytes, for messages.
std::string gigabytes(const idx_t b)
{
    char s[32];
    std::snprintf(s, sizeof(s), "%.1f GB", b / 1e9);
    return s;
}


// Returns the name of the type of the file system that holds the file at
// `path`, and in `remote` whether it is over a network; `?` if unknown.
std::string file_system(const std::string& path, bool& remote)
{
    struct Kind { uint32_t magic; const char* name; bool remote; };
    static const Kind kinds[] =
    {
        {0xEF53, "ext4", false}, {0x58465342, "xfs", false}, {0x9123683E, "btrfs", false}, {0x01021994, "tmpfs", false},
        {0x794C7630, "overlay", false}, {0x2FC12FC1, "zfs", false}, {0xF2F52010, "f2fs", false}, {0xCA451A4E, "bcachefs", false},
        {0x858458F6, "ramfs", false}, {0x6A656A63, "virtiofs", false},
        {0x6969, "nfs", true}, {0x517B, "smb", true}, {0xFF534D42, "cifs", true}, {0xFE534D42, "smb2", true}, {0x00C36400, "ceph", true},
        {0x0BD00BD0, "lustre", true}, {0x47504653, "gpfs", true}, {0x19830326, "beegfs", true}, {0xAAD7AAEA, "panfs", true},
        {0x01021997, "9p", true}, {0x65735546, "fuse", true}, {0x5346414F, "afs", true}, {0x01161970, "gfs2", true}, {0x7461636F, "ocfs2", true},
    };

    remote = false;
    struct statfs st;
    if(::statfs(path.c_str(), &st) != 0)
        return "?";

    const auto magic = static_cast<uint32_t>(st.f_type);
    for(const auto& k : kinds)
        if(k.magic == magic)
        {
            remote = k.remote;
            return k.name;
        }

    char s[16];
    std::snprintf(s, sizeof(s), "0x%x", magic);
    return s;
}


// Writes into `keys[i]` the first `min(m, K)` symbols of the query
// `sym[off[i] - off0, off[i + 1] - off0)` of length `m`, packed as a table key
// is, zero-padded to the key's `K` symbols; and `i` into `ids[i]` if `ids` is
// not null.
template <typename T_sym_, idx_t W_>
__global__ void pack_kernel(const uint64_t* const off, const T_sym_* const sym, const uint64_t off0, const idx_t q,
                            Packed_Prefix<W_>* const keys, uint32_t* const ids)
{
    constexpr idx_t spw = sizeof(uint64_t) / sizeof(T_sym_);   // Symbols per word.
    for(idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; i < q; i += idx_t(gridDim.x) * blockDim.x)
    {
        const T_sym_* const P = sym + (off[i] - off0);
        const idx_t m = off[i + 1] - off[i];
        Packed_Prefix<W_> pk;
        #pragma unroll
        for(idx_t w = 0; w < W_; ++w)
        {
            uint64_t word = 0;
            #pragma unroll
            for(idx_t k = 0; k < spw; ++k)
                if(w * spw + k < m)
                    word |= static_cast<uint64_t>(P[w * spw + k]) << ((spw - 1 - k) * 8 * sizeof(T_sym_));

            pk.word[w] = word;
        }

        keys[i] = pk;
        if(ids != nullptr)
            ids[i] = static_cast<uint32_t>(i);
    }
}


// Ranks the packed prefixes `keys[g]` of a batch's queries among the `J` keys
// of the sample table `tab` of stride `stride` over a text of length `n`, and
// writes the ranks that contain their SA-intervals' boundaries into `br`. The
// batch is the queries `sym[off[i] - off0, off[i + 1] - off0)`; position `g`
// of it is the query `ids[g]`, or `g` if `ids` is null.
// Adjacent threads pair up on a query: thread `2g` counts the keys smaller than
// the query's prefix, `a`, and thread `2g + 1` those not greater, `b`, with the
// keys cut to the query's length; they swap their counts, and then write the
// ranges of the interval's `lo` and `hi` into `br[2g]` and `br[2g + 1]`:
//   pattern not longer than a key:  lo in [(a - 1) s + 1, a s + e],  hi in [(b - 1) s + 1, b s];
//   longer:                         both in [(a - 1) s + 1, b s];
// with `e = m - 1` for a pattern of length `m` ending in the symbol `0`, and
// `e = 0` otherwise: suffixes shorter than such a pattern match its key while
// sorting before it. The ranges are clipped to `[0, n]`.
template <typename T_sym_, idx_t W_>
__global__ void table_kernel(const Packed_Prefix<W_>* const tab, const idx_t J, const idx_t stride, const idx_t n,
                             const uint64_t* const off, const T_sym_* const sym, const uint64_t off0, const Packed_Prefix<W_>* const keys,
                             const uint32_t* const ids, const idx_t q, Rank_Range* const br)
{
    constexpr idx_t spw = sizeof(uint64_t) / sizeof(T_sym_);   // Symbols per word.
    constexpr idx_t K = W_ * spw;   // Symbols per key.
    const idx_t t = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const idx_t g = ((t >> 1) < q ? (t >> 1) : q - 1);  // Threads past the batch shadow its last query, to keep every lane in step.
    const bool upper = t & 1;
    const idx_t i = (ids != nullptr ? ids[g] : g);
    const idx_t m = off[i + 1] - off[i];
    const idx_t m_key = (m < K ? m : K);
    const Packed_Prefix<W_> pk = keys[g];

    uint64_t mask[W_];  // Keeps the first `m_key` symbols of a key.
    #pragma unroll
    for(idx_t w = 0; w < W_; ++w)
    {
        const idx_t keep = (m_key > w * spw ? (m_key - w * spw < spw ? m_key - w * spw : spw) : 0);
        mask[w] = (keep == spw ? ~uint64_t(0) : (keep == 0 ? 0 : ~uint64_t(0) << ((spw - keep) * 8 * sizeof(T_sym_))));
    }

    idx_t low = 0, len = J;
    while(len > 0)
    {
        const idx_t half = len >> 1, mid = low + half;
        const Packed_Prefix<W_> key = tab[mid];
        int c = 0;
        #pragma unroll
        for(idx_t w = 0; w < W_; ++w)
            if(c == 0 && (key.word[w] & mask[w]) != pk.word[w])
                c = ((key.word[w] & mask[w]) < pk.word[w] ? -1 : 1);

        if(upper ? c <= 0 : c < 0)
        {
            low = mid + 1;
            len -= half + 1;
        }
        else
            len = half;
    }

    const idx_t other = __shfl_xor_sync(0xFFFFFFFF, low, 1);
    const idx_t a = (upper ? other : low), b = (upper ? low : other);
    idx_t first, last;
    if(m <= K)
    {
        const idx_t c = (upper ? b : a);
        first = (c > 0 ? (c - 1) * stride + 1 : 0);
        last = c * stride + (!upper && m > 0 && sym[off[i] - off0 + m - 1] == 0 ? m - 1 : 0);
    }
    else
    {
        first = (a > 0 ? (a - 1) * stride + 1 : 0);
        last = b * stride;
    }

    if(t < 2 * q)
        br[t] = Rank_Range{first, (last < n ? last : n)};
}


// Writes into `heads[g]` the position `g` of a batch sorted by the packed
// prefixes `keys` if its query differs from the one before it, and `0` if it
// does not: in its length, its prefix, or its symbols past the prefix's `K`.
// The queries are `sym[off[i] - off0, off[i + 1] - off0)`, position `g` of the
// batch the query `ids[g]`. A maximum-scan of `heads` then gives every
// position the first of its run of identical queries.
template <typename T_sym_, idx_t W_>
__global__ void head_kernel(const uint64_t* const off, const T_sym_* const sym, const uint64_t off0, const Packed_Prefix<W_>* const keys,
                            const uint32_t* const ids, const idx_t q, uint32_t* const heads)
{
    constexpr idx_t K = W_ * sizeof(uint64_t) / sizeof(T_sym_);   // Symbols per key.
    for(idx_t g = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; g < q; g += idx_t(gridDim.x) * blockDim.x)
    {
        bool same = false;
        if(g > 0)
        {
            const idx_t i = ids[g], j = ids[g - 1];
            const idx_t m = off[i + 1] - off[i];
            same = (off[j + 1] - off[j] == m && keys[g] == keys[g - 1]);
            for(idx_t x = K; same && x < m; ++x)
                same = (sym[off[i] - off0 + x] == sym[off[j] - off0 + x]);
        }

        heads[g] = (same ? 0 : static_cast<uint32_t>(g));
    }
}


// The maximum of two positions.
struct Max_Position
{
    __device__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return a > b ? a : b; }
};


// Returns the position whose gathered brackets serve the position `g` of a
// batch, in the tile starting at `g0`: the first of `g`'s run of identical
// queries, `heads[g]`, if in the tile, else the tile's first; `g` if `heads` is
// null.
__device__ inline idx_t gathered_at(const uint32_t* const heads, const idx_t g0, const idx_t g)
{
    return heads == nullptr ? g : (heads[g] > g0 ? heads[g] : g0);
}


// Gathers into the windows `win` of the tile of the positions `[g0, g0 + q)` of
// a batch the suffix-array bytes that the searches of their brackets `br[2g]`
// and `br[2g + 1]` probe, from the suffix array `SA`: each bracket's as
// `window_of` places it, in slots of `wb` bytes. With the runs' first positions
// `heads`, a run of identical queries gathers once in the tile. A warp gathers
// a query's two brackets, 16 bytes a lane: all of it their union, or half of
// it each bracket.
__global__ void gather_kernel(const uint8_t* const SA, const Rank_Range* const br, const uint32_t* const heads, const idx_t g0, const idx_t q,
                              const idx_t wb, uint8_t* const win)
{
    const idx_t t = (idx_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const unsigned lane = threadIdx.x & 31;
    if(t >= q)
        return;

    const idx_t g = g0 + t;
    if(gathered_at(heads, g0, g) != g)  // An earlier copy of the query in the tile gathers for it.
        return;

    const Rank_Range lo = br[2 * g], hi = br[2 * g + 1];
    const Window w_lo = window_of(lo, hi, false, t, wb), w_hi = window_of(hi, lo, true, t, wb);
    const bool whole = (w_lo.dst == w_hi.dst);  // The union's slots, or no slot.
    const Window& w = (whole || lane < 16 ? w_lo : w_hi);
    const idx_t first = 16 * (whole ? lane : lane & 15), step = (whole ? 32 * 16 : 16 * 16);
    if(w.dst != NO_SLOT)
        for(idx_t b = w.b0 + first; b < w.b1; b += step)
            *reinterpret_cast<uint4*>(win + w.dst + (b - w.b0)) = *reinterpret_cast<const uint4*>(SA + b);
}


// Searches the suffix array `SA` over the text `T[0, n)` for the queries
// `sym[off[i] - off0, off[i + 1] - off0)` at the positions `[g0, g0 + q)` of a
// batch, into `out`. Position `g` of the batch is the query `ids[g]`, or `g` if
// `ids` is null. Adjacent threads pair up on a query: thread `2(g - g0)` finds
// its interval's `lo`, the first rank whose truncated suffix is not smaller
// than the pattern, and thread `2(g - g0) + 1` its `hi`, the first rank whose
// truncated suffix is greater. A thread searches the ranks `br[2g + lane]` if
// `br` is not null, and all of `[0, n]` otherwise; the pair's searches then
// share their early path. If `win` is not null, it holds the positions'
// gathered brackets, in slots of `wb` bytes, which the searches read instead of
// the suffix array: a position's own, or those of the first of its run of
// identical queries in the tile per `heads`, if not null.
template <typename T_sym_>
__global__ void search_kernel(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off,
                              const T_sym_* const sym, const uint64_t off0, const uint32_t* const ids, const Rank_Range* const br,
                              const uint32_t* const heads, const idx_t g0, const idx_t q, const uint8_t* const win, const idx_t wb,
                              SA_Interval* const out)
{
    const idx_t t = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(t >= 2 * q)
        return;

    const idx_t g = g0 + (t >> 1);
    const bool upper = t & 1;
    const idx_t i = (ids != nullptr ? ids[g] : g);
    const T_sym_* const P = sym + (off[i] - off0);
    const idx_t m = off[i + 1] - off[i];

    // The answer is in `[low, low + len]`: the ranks before `low` go right, those from `low + len` do not.
    idx_t low = 0, len = n;
    const uint8_t* src = SA;    // The probed bytes, holding the suffix array's from its byte `b0`.
    idx_t b0 = 0;
    if(br != nullptr)
    {
        const Rank_Range r = br[2 * g + upper];
        low = r.first, len = r.last - r.first;
        if(win != nullptr)  // An identical query's brackets are the same, and so are their windows.
        {
            const Window w = window_of(r, br[2 * g + !upper], upper, gathered_at(heads, g0, g) - g0, wb);
            if(w.dst != NO_SLOT)
                src = win + w.dst, b0 = w.b0;
        }
    }

    while(len > 0)
    {
        const idx_t half = len >> 1, mid = low + half;
        const int c = ref::cmp_suffix_pattern(T, n, sa40::load_at(src, SA_WIDTH * mid - b0), P, m);
        if(upper ? c <= 0 : c < 0)
        {
            low = mid + 1;
            len -= half + 1;
        }
        else
            len = half;
    }

    if(upper)
        out[i].hi = low;
    else
        out[i].lo = low;
}


// Returns the scratch bytes of sorting `q` keys of `W_` words with their batch
// positions.
template <idx_t W_>
std::size_t sort_bytes(const idx_t q)
{
    std::size_t bytes = 0;
    cub::DoubleBuffer<Packed_Prefix<W_>> keys(nullptr, nullptr);
    cub::DoubleBuffer<uint32_t> ids(nullptr, nullptr);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, ids, static_cast<int>(q), Packed_Prefix_Decomposer<W_>{}, 0, static_cast<int>(W_ * 64)));
    return bytes;
}


// Returns the scratch bytes of sorting `q` keys of `w` words with their batch
// positions.
std::size_t sort_bytes(const idx_t q, const uint32_t w)
{
    return w == 1 ? sort_bytes<1>(q) : (w == 2 ? sort_bytes<2>(q) : sort_bytes<4>(q));
}


// Returns the scratch bytes of finding the first positions of the runs of
// identical queries in a sorted batch of `q`.
std::size_t scan_bytes(const idx_t q)
{
    std::size_t bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        nullptr, bytes,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        Max_Position{},
        static_cast<int>(q)));
    return bytes;
}


// Returns the bytes of device memory of the buffers of a two-phase search of
// `cap` queries with keys of `w` words, sorted if `sort`, as `allocate`
// allocates them, at most: in whole device pages, and the scratch of the sort
// and the scan at most as CUB sizes it for any device.
idx_t search_buffer_bytes(const idx_t cap, const uint32_t w, const bool sort)
{
    idx_t b = device_pages(cap * w * sizeof(uint64_t)) + device_pages(2 * cap * sizeof(Rank_Range));
    if(sort)
        b += device_pages(cap * w * sizeof(uint64_t)) + 3 * device_pages(cap * sizeof(uint32_t))
           + device_pages(SORT_SCRATCH_PER_KEY * cap + SCRATCH_BASE_BYTES) + device_pages(SCAN_SCRATCH_PER_KEY * cap + SCRATCH_BASE_BYTES);

    return b;
}


// Allocates into `sb` the buffers of a two-phase search of `cap` queries with
// keys of `w` words, and of sorting them and finding their runs if `sort`: from
// the pool `pool` on the stream `s` if the pool is not null, and directly
// otherwise. Returns the first error.
cudaError_t allocate(Search_Buffers& sb, const idx_t cap, const uint32_t w, const bool sort, const cudaMemPool_t pool, const cudaStream_t s)
{
    sb = Search_Buffers();
    sb.cap = cap;
    cudaError_t e = cudaSuccess;
    const auto alloc = [&](auto*& p, const std::size_t bytes)
    {
        void* v = nullptr;
        if(e == cudaSuccess)
            e = (pool != nullptr ? cudaMallocFromPoolAsync(&v, std::max<std::size_t>(bytes, 1), pool, s)
                                 : cudaMalloc(&v, std::max<std::size_t>(bytes, 1)));
        p = static_cast<std::remove_reference_t<decltype(p)>>(v);
        sb.bytes += std::max<std::size_t>(bytes, 1);
    };

    alloc(sb.keys[0], cap * w * sizeof(uint64_t));
    alloc(sb.brackets, 2 * cap * sizeof(Rank_Range));
    if(sort)
    {
        sb.sort_bytes = sort_bytes(cap, w);
        alloc(sb.keys[1], cap * w * sizeof(uint64_t));
        alloc(sb.ids[0], cap * sizeof(uint32_t));
        alloc(sb.ids[1], cap * sizeof(uint32_t));
        alloc(sb.sort_space, sb.sort_bytes);
        sb.scan_bytes = scan_bytes(cap);
        alloc(sb.heads, cap * sizeof(uint32_t));
        alloc(sb.scan_space, sb.scan_bytes);
    }

    return e;
}


// Frees the buffers `sb`: into the pool on the stream `s` if `pooled`, and
// directly otherwise.
void release(Search_Buffers& sb, const bool pooled, const cudaStream_t s)
{
    for(void* const p : {sb.keys[0], sb.keys[1], static_cast<void*>(sb.ids[0]), static_cast<void*>(sb.ids[1]), sb.brackets, sb.sort_space,
                         static_cast<void*>(sb.heads), sb.scan_space})
        if(p != nullptr)
            CUDA_CHECK(pooled ? cudaFreeAsync(p, s) : cudaFree(p));

    sb = Search_Buffers();
}


// The order of a batch of a two-phase search: its packed prefixes, and the
// batch position at each place, or null in the batch's own order.
template <idx_t W_>
struct Ranked
{
    const Packed_Prefix<W_>* keys;
    const uint32_t* ids;
};


// Ranks the device-side batch `sym[off[i] - off0, off[i + 1] - off0)`, `i < q`,
// over a text of length `n` in the sample table `tab` of keys of `W_` words,
// with the buffers `sb`: packs the queries' prefixes, sorts the batch by them
// first if `sort`, and writes the brackets of the queries' boundaries into
// `sb.brackets`. Asynchronous on the stream `s`. Returns the batch's order.
template <typename T_sym_, idx_t W_>
Ranked<W_> rank(const Sample_Table& tab, const idx_t n, const bool sort, const uint64_t* const off, const T_sym_* const sym, const uint64_t off0,
                const idx_t q, const Search_Buffers& sb, const cudaStream_t s)
{
    if(q > sb.cap || q > INT_MAX)
    {
        std::cerr << "Batch of " << q << " queries exceeds its search buffers' " << sb.cap << ", or the supported " << INT_MAX << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    auto* const keys = static_cast<Packed_Prefix<W_>*>(sb.keys[0]);
    const auto grid_q = static_cast<unsigned>(std::min<idx_t>((q + BLOCK - 1) / BLOCK, 65536));
    pack_kernel<T_sym_, W_><<<grid_q, BLOCK, 0, s>>>(off, sym, off0, q, keys, sort ? sb.ids[0] : nullptr);
    CUDA_CHECK(cudaGetLastError());

    const Packed_Prefix<W_>* k = keys;
    const uint32_t* id = nullptr;
    if(sort)
    {
        cub::DoubleBuffer<Packed_Prefix<W_>> key_buf(keys, static_cast<Packed_Prefix<W_>*>(sb.keys[1]));
        cub::DoubleBuffer<uint32_t> id_buf(sb.ids[0], sb.ids[1]);
        std::size_t bytes = sb.sort_bytes;
        if(sort_bytes<W_>(q) > bytes)
        {
            std::cerr << "Sort scratch of " << bytes << " bytes is short for " << q << " queries. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sb.sort_space, bytes, key_buf, id_buf, static_cast<int>(q), Packed_Prefix_Decomposer<W_>{},
                                                   0, static_cast<int>(W_ * 64), s));
        k = key_buf.Current();
        id = id_buf.Current();
    }

    const auto grid_2q = static_cast<unsigned>((2 * q + BLOCK - 1) / BLOCK);
    table_kernel<T_sym_, W_><<<grid_2q, BLOCK, 0, s>>>(tab.keys<W_>(), tab.size(), tab.stride(), n, off, sym, off0, k, id, q,
                                                       static_cast<Rank_Range*>(sb.brackets));
    CUDA_CHECK(cudaGetLastError());

    return Ranked<W_>{k, id};
}


// Finds the SA-intervals of the device-side batch `sym[off[i] - off0, off[i + 1]
// - off0)`, `i < q`, over the index `idx` in two phases, with the sample table
// `tab` of keys of `W_` words, and the buffers `sb`; sorting the batch by its
// packed prefixes first if `sort`. If `win` is not null, the brackets of every
// `tile` positions of the batch are gathered into it, in slots of `wb` bytes,
// and searched there; sorted, a run of identical queries gathers once in a
// tile. Asynchronous on the stream `s`; the ranking and the exact search timed
// on the stream's clock `clock`, unless null.
template <typename T_sym_, idx_t W_>
void find_two_phase(const Suffix_Index& idx, const Sample_Table& tab, const bool sort, const uint64_t* const off, const T_sym_* const sym,
                    const uint64_t off0, const idx_t q, SA_Interval* const out, const Search_Buffers& sb, uint8_t* const win,
                    const idx_t tile, const idx_t wb, const cudaStream_t s, Phase_Clock* const clock)
{
    const auto r = rank<T_sym_, W_>(tab, idx.n(), sort, off, sym, off0, q, sb, s);
    phase_lap(clock, Phase::rank, s);
    const uint32_t* const id = r.ids;
    const auto* const br = static_cast<const Rank_Range*>(sb.brackets);

    // The runs of identical queries of a sorted batch, whose gathers are the same. Probed in place, they coalesce anyway.
    const uint32_t* heads = nullptr;
    if(sort && win != nullptr)
    {
        const auto grid_q = static_cast<unsigned>(std::min<idx_t>((q + BLOCK - 1) / BLOCK, 65536));
        head_kernel<T_sym_, W_><<<grid_q, BLOCK, 0, s>>>(off, sym, off0, r.keys, id, q, sb.heads);
        CUDA_CHECK(cudaGetLastError());
        std::size_t bytes = sb.scan_bytes;
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(
            sb.scan_space, bytes,
            sb.heads, sb.heads,
            Max_Position{},
            static_cast<int>(q), s)
        );
        heads = sb.heads;
    }

    // A tile at a time: in the order of the batch, which sorted is that of the suffix array.
    const idx_t tile_q = (win != nullptr ? tile : q);
    for(idx_t g0 = 0; g0 < q; g0 += tile_q)
    {
        const idx_t tq = std::min(tile_q, q - g0);
        if(win != nullptr)
        {
            gather_kernel<<<static_cast<unsigned>((32 * tq + BLOCK - 1) / BLOCK), BLOCK, 0, s>>>(idx.sa(), br, heads, g0, tq, wb, win);
            CUDA_CHECK(cudaGetLastError());
        }

        search_kernel<<<static_cast<unsigned>((2 * tq + BLOCK - 1) / BLOCK), BLOCK, 0, s>>>(idx.text<T_sym_>(), idx.n(), idx.sa(), off, sym,
                                                                                          off0, id, br, heads, g0, tq, win, wb, out);
        CUDA_CHECK(cudaGetLastError());
    }

    phase_lap(clock, Phase::search, s);
}


// Returns the bytes of memory available per `/proc/meminfo`, without swapping;
// `0` if unknown.
idx_t available_bytes()
{
    std::ifstream is("/proc/meminfo");
    for(std::string line; std::getline(is, line); )
        if(line.rfind("MemAvailable:", 0) == 0)
            return std::stoull(line.substr(13)) * 1024;    // In kB.

    return 0;
}


// Returns `true` iff the host address `p` is in pinned memory: allocated or
// registered with the device.
bool is_pinned(const void* const p)
{
    cudaPointerAttributes a;
    if(p == nullptr || cudaPointerGetAttributes(&a, p) != cudaSuccess)
    {
        cudaGetLastError(); // Not an error to keep.
        return false;
    }

    return a.type == cudaMemoryTypeHost;
}


// Checks the intervals `I[0, q)` of the queries `sym[off[i], off[i + 1])` with
// the certificate of `ref::certify`: counts the failing queries into `fail_c`,
// and lowers `first` to the first failing one.
template <typename T_sym_>
__global__ void certify_kernel(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off,
                               const T_sym_* const sym, const idx_t q, const SA_Interval* const I, unsigned long long* const fail_c,
                               unsigned long long* const first)
{
    for(idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; i < q; i += idx_t(gridDim.x) * blockDim.x)
        if(!ref::certify(T, n, SA, sym + off[i], off[i + 1] - off[i], I[i]))
        {
            atomicAdd(fail_c, 1ull);
            atomicMin(first, static_cast<unsigned long long>(i));
        }
}


// Adds the count `c` of queries failing the certificate on the host to
// `fail_c`, and lowers `first` to the first of them, `f`.
__global__ void tally_kernel(unsigned long long* const fail_c, unsigned long long* const first, const unsigned long long c,
                             const unsigned long long f)
{
    atomicAdd(fail_c, c);
    atomicMin(first, f);
}


// Copies the device-side batch `sym[off[i], off[i + 1])`, `i < q`, into `off_h`
// and `sym_h`, once the earlier work of the stream `s` is done; synchronously.
template <typename T_sym_>
void fetch_batch(const uint64_t* const off, const T_sym_* const sym, const idx_t q, const cudaStream_t s, std::vector<uint64_t>& off_h,
                 std::vector<T_sym_>& sym_h)
{
    off_h.resize(q + 1);
    CUDA_CHECK(cudaMemcpyAsync(off_h.data(), off, (q + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    sym_h.resize(off_h[q]);
    if(off_h[q] > 0)
        CUDA_CHECK(cudaMemcpyAsync(sym_h.data(), sym, off_h[q] * sizeof(T_sym_), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

}


const char* plan_name(const Plan p)
{
    switch(p)
    {
    case Plan::A:   return "A";
    case Plan::B:   return "B";
    case Plan::C:   return "C";
    case Plan::D:   return "D";
    case Plan::custom:  return "custom";
    }

    return "?";
}


template <typename T_sym_>
idx_t Query_Engine<T_sym_>::lane_device_bytes(const idx_t cq, const idx_t cs)
{
    // As `allocate_lanes` allocates a lane: its chunk's offsets, symbols, and intervals, and the buffers of any mode's search.
    return STREAM_C * (device_pages((cq + 1) * sizeof(uint64_t)) + device_pages(cs * sizeof(T_sym_)) + device_pages(cq * sizeof(SA_Interval))
                       + search_buffer_bytes(cq, 4, true));
}


template <typename T_sym_>
idx_t Query_Engine<T_sym_>::lane_host_bytes(const idx_t cq, const idx_t cs, const bool host)
{
    // As `allocate_lanes` allocates a lane's pinned buffers: a chunk's staging, and its brackets and order if the host searches.
    return STREAM_C * ((cq + 1) * sizeof(uint64_t) + cs * sizeof(T_sym_) + cq * sizeof(SA_Interval)
                       + (host ? 2 * cq * sizeof(Rank_Range) + cq * sizeof(uint32_t) : 0));
}


template <typename T_sym_>
idx_t Query_Engine<T_sym_>::gather_bytes(const idx_t stride, const idx_t K, const idx_t gather_queries)
{
    // As `search` takes them from the pool: the windows of a search on every stream of `find`, and of one by `find_async`.
    const idx_t wb = std::min(WINDOW_MAX_BYTES, bracket_window_bytes(stride, K));
    return device_pages((STREAM_C + 1) * (2 * wb * gather_queries + WINDOW_PAD_BYTES));
}


template <typename T_sym_>
idx_t Query_Engine<T_sym_>::fit_stride(const idx_t n, const uint32_t w, const bool gather, const idx_t gather_queries, const idx_t room)
{
    const idx_t K = w * sizeof(uint64_t) / sizeof(T_sym_);
    for(idx_t s = AUTO_STRIDE_MIN; s <= AUTO_STRIDE_MAX; s *= 2)
    {
        if(gather && bracket_window_bytes(s, K) > WINDOW_MAX_BYTES) // Wider brackets would be searched in place, across the bus.
            break;

        if(device_pages(Sample_Table::bytes_of(n, s, w)) + (gather ? gather_bytes(s, K, gather_queries) : 0) <= room)
            return s;
    }

    return 0;
}


const char* phase_name(const Phase p)
{
    static const char* const name[PHASE_C] = {"stage", "h2d", "rank", "search", "d2h", "prefetch", "host_search", "wait"};
    return name[static_cast<std::size_t>(p)];
}


void Phase_Clock::record(const int8_t p, const cudaStream_t s)
{
    if(ev_c_ == ev_.size())
    {
        cudaEvent_t e;
        CUDA_CHECK(cudaEventCreate(&e));
        ev_.push_back(e);
        end_.push_back(NO_PHASE);
    }

    CUDA_CHECK(cudaEventRecord(ev_[ev_c_], s));
    end_[ev_c_++] = p;
}


void Phase_Clock::read(std::array<double, PHASE_C>& t) const
{
    for(std::size_t p = 0; p < PHASE_C; ++p)
        t[p] += host_[p];

    for(std::size_t k = 1; k < ev_c_; ++k)  // The first event starts a phase.
        if(end_[k] != NO_PHASE)
        {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_[k - 1], ev_[k]));
            t[static_cast<std::size_t>(end_[k])] += ms / 1e3;
        }
}


void Phase_Clock::clear()
{
    ev_c_ = 0;
    host_.fill(0);
}


void Phase_Clock::release()
{
    // Errors are not checked: a failed release leaves nothing to recover.
    for(const cudaEvent_t e : ev_)
        cudaEventDestroy(e);

    ev_.clear();
    end_.clear();
    clear();
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::fit(const Index_Meta& m, const Query_Options& opt, Placement& p, std::string& why)
{
    // The searcher, the brackets' source, the chunks, and the keys, as `load` and `configure` pick them.
    const auto device_side = [](const Residency r) { return r == Residency::device || r == Residency::pinned; };
    p.host_search = !(device_side(p.text_res) && device_side(p.sa_res));
    p.gather = (opt.gather != Gather::automatic ? opt.gather == Gather::always : p.sa_res == Residency::pinned && !p.host_search);
    p.chunk_queries = (opt.chunk_queries != 0 ? opt.chunk_queries : idx_t(1) << (p.gather ? 24 : 18));
    p.chunk_symbols = (opt.chunk_symbols != 0 ? opt.chunk_symbols : idx_t(1) << (p.gather ? 28 : 24));
    const uint32_t w_dflt = (sizeof(T_sym_) == 4 ? 4 : 2);
    const uint32_t w = (opt.key_words == AUTO_KEY_WORDS ? (p.host_search ? 4 : w_dflt) : (opt.key_words != 0 ? opt.key_words : w_dflt));
    if(p.gather && p.host_search)
    {
        why = "gathers no brackets: the host searches it";
        return false;
    }

    // The arrays in their tiers, the lanes and their pinned buffers, and a mapped suffix array's marks of the pages asked for.
    idx_t dev = 0, host = 0;
    Suffix_Index::footprint(m, p.text_res, p.sa_res, dev, host);
    dev += lane_device_bytes(p.chunk_queries, p.chunk_symbols);
    host += lane_host_bytes(p.chunk_queries, p.chunk_symbols, p.host_search);
    if(p.sa_res == Residency::mapped && opt.host_prefetch)
        host += (m.sa_bytes() + SA_PAGE_BYTES - 1) / SA_PAGE_BYTES;

    // The table, in the device memory that is left, with its gathers.
    const idx_t cap = (p.device_budget > p.device_margin ? p.device_budget - p.device_margin : 0);
    p.table_room = (cap > dev ? cap - dev : 0);
    p.device_bytes = dev;
    p.host_bytes = host;
    const idx_t s = (opt.stride == AUTO_STRIDE ? fit_stride(m.n, w, p.gather, opt.gather_queries, p.table_room) : opt.stride);
    if(s == 0 && opt.stride == AUTO_STRIDE)
    {
        why = "takes " + gigabytes(dev) + " of device memory without a table, which leaves " + gigabytes(p.table_room)
            + " of the budget less its margin: too little for a table of stride " + std::to_string(AUTO_STRIDE_MIN) + " to "
            + std::to_string(AUTO_STRIDE_MAX);
        return false;
    }

    p.stride = s;
    p.key_words = (s != 0 ? w : 0);
    p.table_bytes = Sample_Table::bytes_of(m.n, s, w);
    if(s != 0)
        p.device_bytes += device_pages(p.table_bytes) + (p.gather ? gather_bytes(s, w * sizeof(uint64_t) / sizeof(T_sym_), opt.gather_queries) : 0);
    p.host_bytes += Sample_Table::build_host_bytes(m, p.text_res, p.sa_res, s, w);

    if(p.device_bytes > cap)
    {
        why = "takes " + gigabytes(p.device_bytes) + " of device memory, past the budget of " + gigabytes(p.device_budget) + " less its margin";
        return false;
    }

    if(p.host_bytes + p.host_margin > p.host_budget)
    {
        why = "takes " + gigabytes(p.host_bytes) + " of host memory, past the budget of " + gigabytes(p.host_budget) + " less its margin";
        return false;
    }

    return true;
}


template <typename T_sym_>
Query_Engine<T_sym_>::Call_Timer::Call_Timer(const Query_Engine& eng):
      eng_(eng)
    , t0_(eng.opt_.phase_times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point())
{
    if(!eng_.opt_.phase_times)
        return;

    for(const auto& l : eng_.lane_)
        l.clock.clear();

    eng_.timed_ = Phase_Times();
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::plan(const Index_Meta& m, const Query_Options& opt, Placement& p, std::string& err)
{
    p = Placement();
    if(m.sym_bytes != sizeof(T_sym_))
    {
        err = "index has " + std::to_string(m.sym_bytes) + "-byte symbols; the engine expects " + std::to_string(sizeof(T_sym_));
        return false;
    }

    p.auto_text_res = (opt.text_res == Residency::automatic);
    p.auto_sa_res = (opt.sa_res == Residency::automatic);
    p.auto_stride = (opt.stride == AUTO_STRIDE);
    p.auto_key_words = (opt.key_words == AUTO_KEY_WORDS);
    const bool named = (!p.auto_text_res && !p.auto_sa_res);   // Named tiers are placed as named, fitting or not.

    // The budgets as set, or else measured; a failed measure fails only a pick of a tier or a stride.
    p.device_budget = opt.device_budget;
    if(p.device_budget == 0)
    {
        std::size_t free_b = 0, total_b = 0;
        const cudaError_t e = cudaMemGetInfo(&free_b, &total_b);
        if(e == cudaSuccess)
            p.device_budget = free_b;
        else
        {
            cudaGetLastError(); // Not an error to keep.
            if(!named || p.auto_stride)
            {
                err = std::string("cannot measure the device's free memory: ") + cudaGetErrorString(e) + "; set a device budget";
                return false;
            }
        }
    }

    p.host_budget = (opt.host_budget != 0 ? opt.host_budget : available_bytes());
    if(p.host_budget == 0 && (!named || p.auto_stride))
    {
        err = "cannot measure the host's available memory; set a host budget";
        return false;
    }

    p.device_margin = std::max(MARGIN_MIN_BYTES, p.device_budget / MARGIN_PARTS);
    p.host_margin = std::max(MARGIN_MIN_BYTES, p.host_budget / MARGIN_PARTS);

    // The plans whose tiers agree with those named, in order; or the tiers named, a plan's or not.
    struct Tiers { Plan plan; Residency text_res, sa_res; };
    static constexpr Tiers plans[] = {{Plan::A, Residency::device, Residency::device}, {Plan::B, Residency::device, Residency::pinned},
                                      {Plan::C, Residency::pageable, Residency::pageable}, {Plan::D, Residency::pageable, Residency::mapped}};
    std::vector<Tiers> cand;
    for(const auto& t : plans)
        if((p.auto_text_res || opt.text_res == t.text_res) && (p.auto_sa_res || opt.sa_res == t.sa_res))
            cand.push_back(t);

    if(named && cand.empty())
        cand.push_back(Tiers{Plan::custom, opt.text_res, opt.sa_res});

    if(cand.empty())
    {
        err = std::string("no plan places the ") + (p.auto_text_res ? "suffix array" : "text") + " in " + residency_name(p.auto_text_res ? opt.sa_res : opt.text_res)
            + " memory; name both tiers to place it so";
        return false;
    }

    std::string why;
    for(const auto& t : cand)
    {
        Placement c = p;
        c.plan = t.plan;
        c.text_res = t.text_res;
        c.sa_res = t.sa_res;
        std::string no;
        c.fits = fit(m, opt, c, no);
        if(c.fits || (named && (!c.auto_stride || c.stride != 0)))
        {
            p = c;
            why.clear();
            break;
        }

        why += (why.empty() ? "" : "; ") + std::string(named ? "" : std::string(plan_name(t.plan)) + " ") + no;
    }

    if(!why.empty())
    {
        err = (named ? std::string("the index in the tiers named ") : "no plan fits: ") + why;
        return false;
    }

    // The file systems of the suffix array's files: mapped, it is read from them on demand.
    std::vector<std::string> fs;
    for(const auto& path : m.sa_part)
    {
        bool remote = false;
        const std::string f = file_system(path, remote);
        if(std::find(fs.begin(), fs.end(), f) == fs.end())
            fs.push_back(f);
        p.sa_local = p.sa_local && !remote;
    }

    for(const auto& f : fs)
        p.sa_fs += (p.sa_fs.empty() ? "" : "+") + f;

    return true;
}


template <typename T_sym_>
Query_Engine<T_sym_>::Call_Timer::~Call_Timer()
{
    if(eng_.opt_.phase_times)
        eng_.timed_.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::load(const std::string& dir, const Query_Options& opt, std::string& err)
{
    if(opt.chunk_queries > INT_MAX || opt.min_chunk_queries == 0 || opt.min_chunk_symbols == 0)
    {
        err = "a chunk takes at most " + std::to_string(INT_MAX) + " queries, and its least share a query and a symbol";
        return false;
    }

    if(opt.gather_queries == 0)
    {
        err = "a gather needs room for a query";
        return false;
    }

    if(opt.host_lanes == 0 || opt.host_lanes > host_search::MAX_LANES)
    {
        err = "a thread of the host's search interleaves from 1 to " + std::to_string(host_search::MAX_LANES) + " searches";
        return false;
    }

    Index_Meta m;
    if(!index_meta::read(dir, m, err))
        return false;

    if(m.sym_bytes != sizeof(T_sym_))
    {
        err = "index has " + std::to_string(m.sym_bytes) + "-byte symbols; the engine expects " + std::to_string(sizeof(T_sym_));
        return false;
    }

    release_lanes();
    tab_.release();
    sa_files_.close();
    asked_.reset();
    idx_.release(); // The planner measures the memory free of the previous load.
    placement_ = Placement();
    Placement pl;
    if(!plan(m, opt, pl, err) || !idx_.load(dir, pl.text_res, pl.sa_res, opt.spot, err))
        return false;

    placement_ = pl;
    opt_ = opt;
    opt_.text_res = pl.text_res;
    opt_.sa_res = pl.sa_res;
    opt_.host_threads = host_search::thread_count(opt.host_threads);
    const bool configured = configure(pl.stride, pl.key_words, opt.sort, opt.gather, err);
    if(opt_.chunk_queries == 0) // Gathers gain from sorting large chunks.
        opt_.chunk_queries = idx_t(1) << (opt_.gather == Gather::always ? 24 : 18);
    if(opt_.chunk_symbols == 0)
        opt_.chunk_symbols = idx_t(1) << (opt_.gather == Gather::always ? 28 : 24);

    if(!configured || !allocate_lanes(err) || !open_sa_files(err))
    {
        release_lanes();
        tab_.release();
        sa_files_.close();
        asked_.reset();
        idx_.release();
        placement_ = Placement();
        return false;
    }

    if(host_searches())
        host_search::team(opt_.host_threads);   // Its threads start now, rather than in the first batch.

    return true;
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::configure(const idx_t stride, const uint32_t key_words, const bool sort, const Gather gather, std::string& err)
{
    tab_.release();
    opt_.stride = 0;
    opt_.key_words = 0;
    opt_.sort = sort;
    opt_.gather = (gather != Gather::automatic ? gather
                                               : (idx_.sa_residency() == Residency::pinned && !host_searches() ? Gather::always : Gather::never));
    window_bytes_ = 0;
    if(pool_ != nullptr)    // The pool keeps its memory; the new table may need it.
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemPoolTrimTo(pool_, 0));
    }

    if(opt_.gather == Gather::always && host_searches())
    {
        opt_.gather = Gather::never;
        err = "gathering needs the device to access the index; the host searches it";
        return false;
    }

    if(stride == 0)
        return true;

    // Automatic keys as the planner picks them, and an automatic stride the least whose table fits the room the load left for one.
    const uint32_t w_dflt = (sizeof(T_sym_) == 4 ? 4 : 2);
    const uint32_t w = (key_words == AUTO_KEY_WORDS ? (host_searches() ? 4 : w_dflt) : (key_words != 0 ? key_words : w_dflt));
    const idx_t s = (stride == AUTO_STRIDE ? fit_stride(idx_.n(), w, opt_.gather == Gather::always, opt_.gather_queries, placement_.table_room)
                                           : stride);
    if(s == 0)
    {
        err = "no sample table of stride " + std::to_string(AUTO_STRIDE_MIN) + " to " + std::to_string(AUTO_STRIDE_MAX) + " with keys of " + std::to_string(w)
            + " words fits the load's " + gigabytes(placement_.table_room) + " of device memory for one";
        return false;
    }

    if(!tab_.build(idx_, s, w, err))
        return false;

    opt_.stride = s;
    opt_.key_words = w;
    if(opt_.gather == Gather::always)
        window_bytes_ = std::min(WINDOW_MAX_BYTES, bracket_window_bytes(s, tab_.key_symbols()));

    return true;
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::allocate_lanes(std::string& err)
{
    const idx_t cq = opt_.chunk_queries, cs = opt_.chunk_symbols;
    cudaError_t e = cudaGetDevice(&device_);

    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device_;
    uint64_t keep = UINT64_MAX; // Returning the memory at every synchronization costs a batch more than its search.
    if(e == cudaSuccess)    e = cudaMemPoolCreate(&pool_, &props);
    if(e == cudaSuccess)    e = cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &keep);

    // The lanes' buffers, their bytes counted.
    lanes_device_bytes_ = lanes_pinned_bytes_ = 0;
    const auto on_device = [&](auto*& p, const std::size_t bytes)
    {
        if(e == cudaSuccess)    e = cudaMalloc(&p, bytes);
        lanes_device_bytes_ += bytes;
    };
    const auto pinned = [&](auto*& p, const std::size_t bytes)
    {
        if(e == cudaSuccess)    e = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
        lanes_pinned_bytes_ += bytes;
    };

    for(auto& l : lane_)
    {
        if(e == cudaSuccess)    e = cudaStreamCreateWithFlags(&l.stream, cudaStreamNonBlocking);
        on_device(l.off_d, (cq + 1) * sizeof(uint64_t));
        on_device(l.sym_d, cs * sizeof(T_sym_));
        on_device(l.out_d, cq * sizeof(SA_Interval));
        pinned(l.off_h, (cq + 1) * sizeof(uint64_t));
        pinned(l.sym_h, cs * sizeof(T_sym_));
        pinned(l.out_h, cq * sizeof(SA_Interval));
        if(e == cudaSuccess)    e = allocate(l.space, cq, 4, true, nullptr, nullptr);  // For any mode: keys of up to 4 words, sorted.
        lanes_device_bytes_ += l.space.bytes;
        if(host_searches())
        {
            pinned(l.br_h, 2 * cq * sizeof(Rank_Range));
            pinned(l.ids_h, cq * sizeof(uint32_t));
        }
    }

    if(e != cudaSuccess)
    {
        err = std::string("allocating the query pipeline failed: ") + cudaGetErrorString(e);
        return false;
    }

    return true;
}


template <typename T_sym_>
void Query_Engine<T_sym_>::release_lanes()
{
    // Errors are not checked: a failed release leaves nothing to recover.
    for(auto& l : lane_)
    {
        if(l.stream != nullptr)
            cudaStreamSynchronize(l.stream);

        for(void* const p : {l.space.keys[0], l.space.keys[1], static_cast<void*>(l.space.ids[0]), static_cast<void*>(l.space.ids[1]),
                             l.space.brackets, l.space.sort_space, static_cast<void*>(l.space.heads), l.space.scan_space,
                             static_cast<void*>(l.off_d), static_cast<void*>(l.sym_d), static_cast<void*>(l.out_d)})
            cudaFree(p);

        cudaFreeHost(l.off_h);
        cudaFreeHost(l.sym_h);
        cudaFreeHost(l.out_h);
        cudaFreeHost(l.br_h);
        cudaFreeHost(l.ids_h);
        l.clock.release();
        if(l.stream != nullptr)
            cudaStreamDestroy(l.stream);

        l = Lane();
    }

    lanes_device_bytes_ = lanes_pinned_bytes_ = 0;

    if(pool_ != nullptr)
    {
        cudaDeviceSynchronize();
        cudaMemPoolDestroy(pool_);
        pool_ = nullptr;
    }
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::open_sa_files(std::string& err)
{
    sa_files_.close();
    asked_.reset();
    asked_pages_ = asked_c_ = 0;
    if(idx_.sa_residency() != Residency::mapped || !opt_.host_prefetch)
        return true;

    const auto& m = idx_.meta();
    if(!sa_files_.open(m.sa_part, m.sa_part_off, opt_.host_threads, err))
        return false;

    const idx_t page = static_cast<idx_t>(::sysconf(_SC_PAGESIZE));
    asked_pages_ = (m.sa_bytes() + page - 1) / page;
    asked_.reset(new std::atomic<uint8_t>[asked_pages_]());
    const idx_t avail = available_bytes();
    asked_cap_ = (avail > 0 ? avail / page : ~idx_t(0));
    return true;
}


template <typename T_sym_>
void Query_Engine<T_sym_>::forget_asked() const
{
    const auto clear = [&](unsigned, host_search::Team::Claim& claim)  // On the host search's team; see `host_search::Team`.
    {
        for(idx_t k; (k = claim.next()) != host_search::Team::Claim::NONE; )
            for(idx_t p = k * MARKS_PER_ITEM; p < std::min(asked_pages_, (k + 1) * MARKS_PER_ITEM); ++p)
                asked_[p].store(0, std::memory_order_relaxed);
    };

    host_search::team(opt_.host_threads).run((asked_pages_ + MARKS_PER_ITEM - 1) / MARKS_PER_ITEM, clear);
    asked_c_ = 0;
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::drop_cached_sa(std::string& err) const
{
    std::lock_guard<std::mutex> lock(find_lock_);
    if(!idx_.drop_cached_sa(err))
        return false;

    if(asked_ != nullptr)
        forget_asked();

    return true;
}


template <typename T_sym_>
void Query_Engine<T_sym_>::search(const uint64_t* const off, const T_sym_* const sym, const uint64_t off0, const idx_t q,
                                  SA_Interval* const out, const Search_Buffers& space, const cudaStream_t s, Phase_Clock* const clock) const
{
    if(q == 0)
        return;

    if(!tab_.built())
    {
        const auto grid = static_cast<unsigned>((2 * q + BLOCK - 1) / BLOCK);
        search_kernel<<<grid, BLOCK, 0, s>>>(idx_.text<T_sym_>(), idx_.n(), idx_.sa(), off, sym, off0, nullptr, nullptr, nullptr, 0, q, nullptr, 0, out);
        CUDA_CHECK(cudaGetLastError());
        phase_lap(clock, Phase::search, s);
        return;
    }

    uint8_t* win = nullptr; // A tile's gathered brackets.
    const idx_t tile = std::min(q, opt_.gather_queries);
    if(window_bytes_ > 0)
        CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&win), 2 * window_bytes_ * tile + WINDOW_PAD_BYTES, pool_, s));

    switch(tab_.key_words())
    {
    case 1: find_two_phase<T_sym_, 1>(idx_, tab_, opt_.sort, off, sym, off0, q, out, space, win, tile, window_bytes_, s, clock); break;
    case 2: find_two_phase<T_sym_, 2>(idx_, tab_, opt_.sort, off, sym, off0, q, out, space, win, tile, window_bytes_, s, clock); break;
    default:    find_two_phase<T_sym_, 4>(idx_, tab_, opt_.sort, off, sym, off0, q, out, space, win, tile, window_bytes_, s, clock);
    }

    if(win != nullptr)
        CUDA_CHECK(cudaFreeAsync(win, s));
}


template <typename T_sym_>
void Query_Engine<T_sym_>::find_async(const uint64_t* const off, const T_sym_* const sym, const idx_t q, SA_Interval* const out,
                                      const cudaStream_t s) const
{
    if(q == 0)
        return;

    if(host_searches())
    {
        std::vector<uint64_t> off_h;
        std::vector<T_sym_> sym_h;
        fetch_batch(off, sym, q, s, off_h, sym_h);
        std::vector<SA_Interval> out_h(q);
        {
            std::lock_guard<std::mutex> guard(find_lock_);
            const Call_Timer timer(*this);
            find_host(off_h.data(), sym_h.data(), q, out_h.data());
        }

        CUDA_CHECK(cudaMemcpyAsync(out, out_h.data(), q * sizeof(SA_Interval), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        return;
    }

    Search_Buffers space;   // The direct mode needs none.
    if(tab_.built())
    {
        if(q > INT_MAX)
        {
            std::cerr << "Batch of " << q << " queries exceeds the supported " << INT_MAX << ". Aborting.\n";
            std::exit(EXIT_FAILURE);
        }

        CUDA_CHECK(allocate(space, q, tab_.key_words(), opt_.sort, pool_, s));
    }

    search(off, sym, 0, q, out, space, s);
    release(space, true, s);
}


template <typename T_sym_>
void Query_Engine<T_sym_>::find(const uint64_t* const off, const T_sym_* const sym, const idx_t q, SA_Interval* const out) const
{
    if(q == 0)
        return;

    std::lock_guard<std::mutex> guard(find_lock_);
    const Call_Timer timer(*this);
    if(host_searches())
    {
        find_host(off, sym, q, out);
        return;
    }

    const auto chunk = split(off, q);
    if(opt_.phase_times)
        timed_.chunks = chunk.size();

    const Pinned_Arrays pinned{is_pinned(off), is_pinned(sym), is_pinned(out)};
    if(pinned.off && pinned.sym && pinned.out)
    {
        // Nothing to stage: the chunks are enqueued in turn over the lanes, whose streams keep a lane's buffers until its previous
        // chunk is done with them. No thread waits on a chunk, so a chunk's search overlaps the next ones' transfers.
        for(std::size_t c = 0; c < chunk.size(); ++c)
            find_chunk(lane_[c % STREAM_C], off, sym, chunk[c].first, chunk[c].second, out, pinned);

        for(std::size_t l = 0; l < std::min(STREAM_C, chunk.size()); ++l)
        {
            const auto t0 = host_start(clock_of(lane_[l]));
            CUDA_CHECK(cudaStreamSynchronize(lane_[l].stream));
            host_end(clock_of(lane_[l]), Phase::wait, t0);
        }

        return;
    }

    std::atomic<std::size_t> next(0);   // The first chunk not claimed yet.
    const auto dispatch = [&](const std::size_t l)
    {
        for(std::size_t c; (c = next.fetch_add(1, std::memory_order_relaxed)) < chunk.size(); )
            find_chunk(lane_[l], off, sym, chunk[c].first, chunk[c].second, out, pinned);
    };

    if(chunk.size() == 1)   // No overlap to gain; and no thread to start.
    {
        dispatch(0);
        return;
    }

    // A dispatcher per lane, staging its chunks, as `GPU_SACA::compute_partition_IDs` runs them.
    std::vector<std::thread> dispatchers;
    for(std::size_t l = 0; l < std::min(STREAM_C, chunk.size()); ++l)
        dispatchers.emplace_back([&, l] { CUDA_CHECK(cudaSetDevice(device_)); dispatch(l); });

    for(auto& d : dispatchers)
        d.join();
}


template <typename T_sym_>
std::vector<std::pair<idx_t, idx_t>> Query_Engine<T_sym_>::split(const uint64_t* const off, const idx_t q) const
{
    // A chunk ends at the first of its share of the batch's queries and of its symbols, a share being at least the options' least for a
    // chunk, and at most their most; a query longer than a chunk's share is a chunk of its own. Gathered brackets take the most: sorted
    // together, more queries gather denser.
    const idx_t chunk_c = CHUNKS_PER_STREAM * STREAM_C, sym_c = off[q] - off[0];
    const bool whole = (window_bytes_ > 0);
    const idx_t share_q = (whole ? opt_.chunk_queries : std::min(opt_.chunk_queries, std::max(opt_.min_chunk_queries, (q + chunk_c - 1) / chunk_c)));
    const idx_t share_s = (whole ? opt_.chunk_symbols : std::min(opt_.chunk_symbols, std::max(opt_.min_chunk_symbols, (sym_c + chunk_c - 1) / chunk_c)));
    std::vector<std::pair<idx_t, idx_t>> chunk;
    for(idx_t a = 0; a < q; )
    {
        const idx_t b_max = std::min(q, a + share_q);
        const idx_t b = std::upper_bound(off + a + 1, off + b_max + 1, off[a] + share_s) - off - 1;
        chunk.emplace_back(a, std::max(b, a + 1));
        a = chunk.back().second;
    }

    return chunk;
}


template <typename T_sym_>
T_sym_* Query_Engine<T_sym_>::upload(const Lane& l, const uint64_t* const off, const T_sym_* const sym, const idx_t a, const idx_t b,
                                     const Pinned_Arrays& pinned, const bool parallel) const
{
    const idx_t q = b - a;
    const uint64_t off0 = off[a], sym_c = off[b] - off[a];
    const cudaStream_t s = l.stream;
    Phase_Clock* const clock = clock_of(l);
    const auto stage = [&](void* const dst, const void* const src, const std::size_t bytes)
    {
        const auto t0 = host_start(clock);
        if(parallel)
            host_search::copy(dst, src, bytes, opt_.host_threads);
        else
            std::memcpy(dst, src, bytes);
        host_end(clock, Phase::stage, t0);
    };

    const uint64_t* off_src = off + a;
    if(!pinned.off)
    {
        stage(l.off_h, off + a, (q + 1) * sizeof(uint64_t));
        off_src = l.off_h;
    }

    phase_start(clock, s);
    CUDA_CHECK(cudaMemcpyAsync(l.off_d, off_src, (q + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, s));

    // A query longer than a chunk has buffers of its own, filled straight from the batch.
    const bool alone = (sym_c > opt_.chunk_symbols);
    T_sym_* sym_d = l.sym_d;
    if(alone)
        CUDA_CHECK(cudaMallocFromPoolAsync(&sym_d, sym_c * sizeof(T_sym_), pool_, s));

    if(sym_c > 0)
    {
        const T_sym_* sym_src = sym + off0;
        if(!pinned.sym && !alone)
        {
            phase_lap(clock, Phase::h2d, s);   // The device's copying pauses for the host's staging.
            stage(l.sym_h, sym + off0, sym_c * sizeof(T_sym_));
            phase_start(clock, s);
            sym_src = l.sym_h;
        }

        CUDA_CHECK(cudaMemcpyAsync(sym_d, sym_src, sym_c * sizeof(T_sym_), cudaMemcpyHostToDevice, s));
    }

    phase_lap(clock, Phase::h2d, s);
    return sym_d;
}


template <typename T_sym_>
void Query_Engine<T_sym_>::find_chunk(const Lane& l, const uint64_t* const off, const T_sym_* const sym, const idx_t a, const idx_t b,
                                      SA_Interval* const out, const Pinned_Arrays& pinned) const
{
    const idx_t q = b - a;
    const cudaStream_t s = l.stream;
    Phase_Clock* const clock = clock_of(l);
    T_sym_* const sym_d = upload(l, off, sym, a, b, pinned, false);
    search(l.off_d, sym_d, off[a], q, l.out_d, l.space, s, clock);
    CUDA_CHECK(cudaMemcpyAsync(pinned.out ? out + a : l.out_h, l.out_d, q * sizeof(SA_Interval), cudaMemcpyDeviceToHost, s));
    phase_lap(clock, Phase::d2h, s);
    if(sym_d != l.sym_d)
        CUDA_CHECK(cudaFreeAsync(sym_d, s));

    // Staging reuses the lane's buffers, and staged intervals must reach `out`: the chunk must be done. Else, the stream orders it.
    if(!pinned.off || !pinned.sym || !pinned.out)
    {
        auto t0 = host_start(clock);
        CUDA_CHECK(cudaStreamSynchronize(s));
        host_end(clock, Phase::wait, t0);
        if(!pinned.out)
        {
            t0 = host_start(clock);
            std::memcpy(out + a, l.out_h, q * sizeof(SA_Interval));
            host_end(clock, Phase::stage, t0);
        }
    }
}


template <typename T_sym_>
void Query_Engine<T_sym_>::rank_chunk(const Lane& l, const uint64_t* const off, const T_sym_* const sym, const idx_t a, const idx_t b,
                                      const Pinned_Arrays& pinned) const
{
    const idx_t q = b - a;
    const cudaStream_t s = l.stream;
    T_sym_* const sym_d = upload(l, off, sym, a, b, pinned, true);
    const uint32_t* ids = nullptr;
    switch(tab_.key_words())
    {
    case 1: ids = rank<T_sym_, 1>(tab_, idx_.n(), opt_.sort, l.off_d, sym_d, off[a], q, l.space, s).ids; break;
    case 2: ids = rank<T_sym_, 2>(tab_, idx_.n(), opt_.sort, l.off_d, sym_d, off[a], q, l.space, s).ids; break;
    default:    ids = rank<T_sym_, 4>(tab_, idx_.n(), opt_.sort, l.off_d, sym_d, off[a], q, l.space, s).ids;
    }

    Phase_Clock* const clock = clock_of(l);
    phase_lap(clock, Phase::rank, s);
    CUDA_CHECK(cudaMemcpyAsync(l.br_h, l.space.brackets, 2 * q * sizeof(Rank_Range), cudaMemcpyDeviceToHost, s));
    if(ids != nullptr)
        CUDA_CHECK(cudaMemcpyAsync(l.ids_h, ids, q * sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    phase_lap(clock, Phase::d2h, s);
    if(sym_d != l.sym_d)
        CUDA_CHECK(cudaFreeAsync(sym_d, s));
}


template <typename T_sym_>
void Query_Engine<T_sym_>::find_host(const uint64_t* const off, const T_sym_* const sym, const idx_t q, SA_Interval* const out) const
{
    const T_sym_* const T = idx_.host_text<T_sym_>();
    const uint8_t* const SA = idx_.host_sa();
    if(!tab_.built())   // The direct mode: nothing for the device to rank.
    {
        const auto t0 = host_start(clock_of(lane_[0]));
        host_search::find(T, idx_.n(), SA, off, sym, q, nullptr, nullptr, out, opt_.host_threads, opt_.host_lanes);
        host_end(clock_of(lane_[0]), Phase::host_search, t0);
        return;
    }

    // The device ranks the chunks ahead over the lanes, while the host searches them in turn; a lane takes its next chunk once the host
    // is done with the brackets of its last. The host writes the intervals itself.
    const auto chunk = split(off, q);
    if(opt_.phase_times)
        timed_.chunks = chunk.size();

    const Pinned_Arrays pinned{is_pinned(off), is_pinned(sym), true};
    for(std::size_t c = 0; c < std::min(STREAM_C, chunk.size()); ++c)
        rank_chunk(lane_[c], off, sym, chunk[c].first, chunk[c].second, pinned);

    // A mapped suffix array's pages come from the drives: those of a chunk's brackets are asked for once the device ranks it, so that the
    // kernel reads them while the host searches the chunk before.
    const bool ahead = (sa_files_.threads() > 0 && !chunk.empty());
    const auto prefetch = [&](const std::size_t c)
    {
        const Lane& l = lane_[c % STREAM_C];
        Phase_Clock* const clock = clock_of(l);
        auto t0 = host_start(clock);
        CUDA_CHECK(cudaStreamSynchronize(l.stream));
        host_end(clock, Phase::wait, t0);
        t0 = host_start(clock);
        idx_t refused = 0;
        asked_c_ += host_search::prefetch(sa_files_, idx_.n(), l.br_h, chunk[c].second - chunk[c].first, asked_.get(), refused);
        if(asked_c_ > asked_cap_)   // The page cache may have evicted the pages marked first: they are asked for anew.
            forget_asked();
        host_end(clock, Phase::prefetch, t0);
    };

    if(ahead)
        prefetch(0);

    for(std::size_t c = 0; c < chunk.size(); ++c)
    {
        const Lane& l = lane_[c % STREAM_C];
        const auto [a, b] = chunk[c];
        if(ahead && c + 1 < chunk.size())
            prefetch(c + 1);

        Phase_Clock* const clock = clock_of(l);
        auto t0 = host_start(clock);
        CUDA_CHECK(cudaStreamSynchronize(l.stream));
        host_end(clock, Phase::wait, t0);
        t0 = host_start(clock);
        host_search::find(T, idx_.n(), SA, off + a, sym, b - a, l.br_h, opt_.sort ? l.ids_h : nullptr, out + a, opt_.host_threads,
                          opt_.host_lanes);
        host_end(clock, Phase::host_search, t0);
        if(c + STREAM_C < chunk.size())
            rank_chunk(l, off, sym, chunk[c + STREAM_C].first, chunk[c + STREAM_C].second, pinned);
    }
}


template <typename T_sym_>
Phase_Times Query_Engine<T_sym_>::phases() const
{
    std::lock_guard<std::mutex> lock(find_lock_);
    Phase_Times p = timed_;
    if(opt_.phase_times)
        for(const auto& l : lane_)
            l.clock.read(p.t);

    return p;
}


template <typename T_sym_>
Engine_Memory Query_Engine<T_sym_>::memory() const
{
    const auto in = [](const Residency r, const Residency tier, const std::size_t bytes) { return r == tier ? bytes : std::size_t(0); };
    Engine_Memory m;
    m.text_device = in(idx_.text_residency(), Residency::device, idx_.text_alloc_bytes());
    m.sa_device = in(idx_.sa_residency(), Residency::device, idx_.sa_alloc_bytes());
    m.text_pinned = in(idx_.text_residency(), Residency::pinned, idx_.text_alloc_bytes());
    m.sa_pinned = in(idx_.sa_residency(), Residency::pinned, idx_.sa_alloc_bytes());
    m.table_device = tab_.bytes();
    m.lanes_device = lanes_device_bytes_;
    m.lanes_pinned = lanes_pinned_bytes_;
    if(pool_ != nullptr)
    {
        uint64_t b = 0;
        CUDA_CHECK(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemHigh, &b));
        m.pool_used_high = b;
        CUDA_CHECK(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemHigh, &b));
        m.pool_reserved_high = b;
    }

    return m;
}


template <typename T_sym_>
void Query_Engine<T_sym_>::certify_async(const uint64_t* const off, const T_sym_* const sym, const idx_t q, const SA_Interval* const I,
                                         unsigned long long* const fail_c, unsigned long long* const first, const cudaStream_t s) const
{
    if(q == 0)
        return;

    if(host_searches())
    {
        std::vector<uint64_t> off_h;
        std::vector<T_sym_> sym_h;
        fetch_batch(off, sym, q, s, off_h, sym_h);
        std::vector<SA_Interval> I_h(q);
        CUDA_CHECK(cudaMemcpyAsync(I_h.data(), I, q * sizeof(SA_Interval), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        unsigned long long c = 0, f = ULLONG_MAX;
        host_search::certify(idx_.host_text<T_sym_>(), idx_.n(), idx_.host_sa(), off_h.data(), sym_h.data(), q, I_h.data(), c, f,
                             opt_.host_threads);
        if(c > 0)
        {
            tally_kernel<<<1, 1, 0, s>>>(fail_c, first, c, f);
            CUDA_CHECK(cudaGetLastError());
        }

        return;
    }

    const auto grid = static_cast<unsigned>(std::min<idx_t>((q + BLOCK - 1) / BLOCK, 65536));
    certify_kernel<<<grid, BLOCK, 0, s>>>(idx_.text<T_sym_>(), idx_.n(), idx_.sa(), off, sym, q, I, fail_c, first);
    CUDA_CHECK(cudaGetLastError());
}



// Template instantiations for the required instances.
template class Query_Engine<uint8_t>;
template class Query_Engine<uint16_t>;
template class Query_Engine<uint32_t>;

}
