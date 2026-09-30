
#include "Query_Engine.hpp"
#include "CPU_Reference.hpp"
#include "Index_Meta.hpp"
#include "Packed_Prefix_Decomposer.cuh"
#include "sa40.hpp"
#include "utility.cuh"

#include <cub/cub.cuh>

#include <atomic>
#include <climits>
#include <type_traits>
#include <cstring>
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


// The ranks `[first, last]` that contain a boundary of an SA-interval.
struct Rank_Range
{
    idx_t first;
    idx_t last;
};


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


// Gathers into the windows `win` of a tile the suffix-array bytes that the
// searches of its brackets `br[0, 2q)` probe, from the suffix array `SA`: each
// bracket's as `window_of` places it, in slots of `wb` bytes. A warp gathers a
// query's two brackets, 16 bytes a lane: all of it their union, or half of it
// each bracket.
__global__ void gather_kernel(const uint8_t* const SA, const Rank_Range* const br, const idx_t q, const idx_t wb, uint8_t* const win)
{
    const idx_t g = (idx_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const unsigned lane = threadIdx.x & 31;
    if(g >= q)
        return;

    const Rank_Range lo = br[2 * g], hi = br[2 * g + 1];
    const Window w_lo = window_of(lo, hi, false, g, wb), w_hi = window_of(hi, lo, true, g, wb);
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
// the suffix array.
template <typename T_sym_>
__global__ void search_kernel(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off,
                              const T_sym_* const sym, const uint64_t off0, const uint32_t* const ids, const Rank_Range* const br,
                              const idx_t g0, const idx_t q, const uint8_t* const win, const idx_t wb, SA_Interval* const out)
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
        if(win != nullptr)
        {
            const Window w = window_of(r, br[2 * g + !upper], upper, t >> 1, wb);
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


// Allocates into `sb` the buffers of a two-phase search of `cap` queries with
// keys of `w` words, and of sorting them if `sort`: from the pool `pool` on the
// stream `s` if the pool is not null, and directly otherwise. Returns the first
// error.
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
    }

    return e;
}


// Frees the buffers `sb`: into the pool on the stream `s` if `pooled`, and
// directly otherwise.
void release(Search_Buffers& sb, const bool pooled, const cudaStream_t s)
{
    for(void* const p : {sb.keys[0], sb.keys[1], static_cast<void*>(sb.ids[0]), static_cast<void*>(sb.ids[1]), sb.brackets, sb.sort_space})
        if(p != nullptr)
            CUDA_CHECK(pooled ? cudaFreeAsync(p, s) : cudaFree(p));

    sb = Search_Buffers();
}


// Finds the SA-intervals of the device-side batch `sym[off[i] - off0, off[i + 1]
// - off0)`, `i < q`, over the index `idx` in two phases, with the sample table
// `tab` of keys of `W_` words, and the buffers `sb`; sorting the batch by its
// packed prefixes first if `sort`. If `win` is not null, the brackets of every
// `tile` positions of the batch are gathered into it, in slots of `wb` bytes,
// and searched there. Asynchronous on the stream `s`.
template <typename T_sym_, idx_t W_>
void find_two_phase(const Suffix_Index& idx, const Sample_Table& tab, const bool sort, const uint64_t* const off, const T_sym_* const sym,
                    const uint64_t off0, const idx_t q, SA_Interval* const out, const Search_Buffers& sb, uint8_t* const win,
                    const idx_t tile, const idx_t wb, const cudaStream_t s)
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

    auto* const br = static_cast<Rank_Range*>(sb.brackets);
    const auto grid_2q = static_cast<unsigned>((2 * q + BLOCK - 1) / BLOCK);
    table_kernel<T_sym_, W_><<<grid_2q, BLOCK, 0, s>>>(tab.keys<W_>(), tab.size(), tab.stride(), idx.n(), off, sym, off0, k, id, q, br);
    CUDA_CHECK(cudaGetLastError());

    // A tile at a time: in the order of the batch, which sorted is that of the suffix array.
    const idx_t tile_q = (win != nullptr ? tile : q);
    for(idx_t g0 = 0; g0 < q; g0 += tile_q)
    {
        const idx_t tq = std::min(tile_q, q - g0);
        if(win != nullptr)
        {
            gather_kernel<<<static_cast<unsigned>((32 * tq + BLOCK - 1) / BLOCK), BLOCK, 0, s>>>(idx.sa(), br + 2 * g0, tq, wb, win);
            CUDA_CHECK(cudaGetLastError());
        }

        search_kernel<<<static_cast<unsigned>((2 * tq + BLOCK - 1) / BLOCK), BLOCK, 0, s>>>(idx.text<T_sym_>(), idx.n(), idx.sa(), off, sym,
                                                                                          off0, id, br, g0, tq, win, wb, out);
        CUDA_CHECK(cudaGetLastError());
    }
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

}


template <typename T_sym_>
bool Query_Engine<T_sym_>::load(const std::string& dir, const Query_Options& opt, std::string& err)
{
    if(opt.text_res == Residency::pageable || opt.sa_res == Residency::pageable)
    {
        err = "the device engine needs a device-accessible placement: device or pinned";
        return false;
    }

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
    if(!idx_.load(dir, opt.text_res, opt.sa_res, opt.spot, err))
        return false;

    opt_ = opt;
    const bool configured = configure(opt.stride, opt.key_words, opt.sort, opt.gather, err);
    if(opt_.chunk_queries == 0) // Gathers gain from sorting large chunks.
        opt_.chunk_queries = idx_t(1) << (opt_.gather == Gather::always ? 24 : 18);
    if(opt_.chunk_symbols == 0)
        opt_.chunk_symbols = idx_t(1) << (opt_.gather == Gather::always ? 28 : 24);

    if(!configured || !allocate_lanes(err))
    {
        release_lanes();
        tab_.release();
        idx_.release();
        return false;
    }

    return true;
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::configure(const idx_t stride, const uint32_t key_words, const bool sort, const Gather gather, std::string& err)
{
    tab_.release();
    opt_.stride = 0;
    opt_.key_words = 0;
    opt_.sort = sort;
    opt_.gather = (gather != Gather::automatic ? gather : (idx_.sa_residency() == Residency::pinned ? Gather::always : Gather::never));
    window_bytes_ = 0;
    if(pool_ != nullptr)    // The pool keeps its memory; the new table may need it.
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemPoolTrimTo(pool_, 0));
    }

    if(stride == 0)
        return true;

    const uint32_t w = (key_words != 0 ? key_words : (sizeof(T_sym_) == 4 ? 4 : 2));
    if(!tab_.build(idx_, stride, w, err))
        return false;

    opt_.stride = stride;
    opt_.key_words = w;

    // A bracket probes at most `stride + K - 2` ranks: those of a sample gap, and the `K - 1` shorter suffixes that match the key of a
    // pattern ending in `0`. Their bytes, rounded out to 16-byte words, gain at most 15 at each end.
    if(opt_.gather == Gather::always)
        window_bytes_ = std::min(WINDOW_MAX_BYTES, (SA_WIDTH * (stride + tab_.key_symbols() - 2) + 2 * 15 + 15) & ~idx_t(15));

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

    for(auto& l : lane_)
    {
        if(e == cudaSuccess)    e = cudaStreamCreateWithFlags(&l.stream, cudaStreamNonBlocking);
        if(e == cudaSuccess)    e = cudaMalloc(&l.off_d, (cq + 1) * sizeof(uint64_t));
        if(e == cudaSuccess)    e = cudaMalloc(&l.sym_d, cs * sizeof(T_sym_));
        if(e == cudaSuccess)    e = cudaMalloc(&l.out_d, cq * sizeof(SA_Interval));
        if(e == cudaSuccess)    e = cudaHostAlloc(&l.off_h, (cq + 1) * sizeof(uint64_t), cudaHostAllocDefault);
        if(e == cudaSuccess)    e = cudaHostAlloc(&l.sym_h, cs * sizeof(T_sym_), cudaHostAllocDefault);
        if(e == cudaSuccess)    e = cudaHostAlloc(&l.out_h, cq * sizeof(SA_Interval), cudaHostAllocDefault);
        if(e == cudaSuccess)    e = allocate(l.space, cq, 4, true, nullptr, nullptr);  // For any mode: keys of up to 4 words, sorted.
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
                             l.space.brackets, l.space.sort_space, static_cast<void*>(l.off_d), static_cast<void*>(l.sym_d),
                             static_cast<void*>(l.out_d)})
            cudaFree(p);

        cudaFreeHost(l.off_h);
        cudaFreeHost(l.sym_h);
        cudaFreeHost(l.out_h);
        if(l.stream != nullptr)
            cudaStreamDestroy(l.stream);

        l = Lane();
    }

    if(pool_ != nullptr)
    {
        cudaDeviceSynchronize();
        cudaMemPoolDestroy(pool_);
        pool_ = nullptr;
    }
}


template <typename T_sym_>
void Query_Engine<T_sym_>::search(const uint64_t* const off, const T_sym_* const sym, const uint64_t off0, const idx_t q,
                                  SA_Interval* const out, const Search_Buffers& space, const cudaStream_t s) const
{
    if(q == 0)
        return;

    if(!tab_.built())
    {
        const auto grid = static_cast<unsigned>((2 * q + BLOCK - 1) / BLOCK);
        search_kernel<<<grid, BLOCK, 0, s>>>(idx_.text<T_sym_>(), idx_.n(), idx_.sa(), off, sym, off0, nullptr, nullptr, 0, q, nullptr, 0, out);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    uint8_t* win = nullptr; // A tile's gathered brackets.
    const idx_t tile = std::min(q, opt_.gather_queries);
    if(window_bytes_ > 0)
        CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&win), 2 * window_bytes_ * tile + WINDOW_PAD_BYTES, pool_, s));

    switch(tab_.key_words())
    {
    case 1: find_two_phase<T_sym_, 1>(idx_, tab_, opt_.sort, off, sym, off0, q, out, space, win, tile, window_bytes_, s); break;
    case 2: find_two_phase<T_sym_, 2>(idx_, tab_, opt_.sort, off, sym, off0, q, out, space, win, tile, window_bytes_, s); break;
    default:    find_two_phase<T_sym_, 4>(idx_, tab_, opt_.sort, off, sym, off0, q, out, space, win, tile, window_bytes_, s);
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

    // The chunks: a chunk ends at the first of its share of the batch's queries and of its symbols, a share being at least the
    // options' least for a chunk, and at most their most; a query longer than a chunk's share is a chunk of its own. Gathered
    // brackets take the most: sorted together, more queries gather denser.
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

    const Pinned_Arrays pinned{is_pinned(off), is_pinned(sym), is_pinned(out)};
    if(pinned.off && pinned.sym && pinned.out)
    {
        // Nothing to stage: the chunks are enqueued in turn over the lanes, whose streams keep a lane's buffers until its previous
        // chunk is done with them. No thread waits on a chunk, so a chunk's search overlaps the next ones' transfers.
        for(std::size_t c = 0; c < chunk.size(); ++c)
            find_chunk(lane_[c % STREAM_C], off, sym, chunk[c].first, chunk[c].second, out, pinned);

        for(std::size_t l = 0; l < std::min(STREAM_C, chunk.size()); ++l)
            CUDA_CHECK(cudaStreamSynchronize(lane_[l].stream));

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
void Query_Engine<T_sym_>::find_chunk(const Lane& l, const uint64_t* const off, const T_sym_* const sym, const idx_t a, const idx_t b,
                                      SA_Interval* const out, const Pinned_Arrays& pinned) const
{
    const idx_t q = b - a;
    const uint64_t off0 = off[a], sym_c = off[b] - off[a];
    const cudaStream_t s = l.stream;

    const uint64_t* off_src = off + a;
    if(!pinned.off)
    {
        std::memcpy(l.off_h, off + a, (q + 1) * sizeof(uint64_t));
        off_src = l.off_h;
    }

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
            std::memcpy(l.sym_h, sym + off0, sym_c * sizeof(T_sym_));
            sym_src = l.sym_h;
        }

        CUDA_CHECK(cudaMemcpyAsync(sym_d, sym_src, sym_c * sizeof(T_sym_), cudaMemcpyHostToDevice, s));
    }

    search(l.off_d, sym_d, off0, q, l.out_d, l.space, s);
    CUDA_CHECK(cudaMemcpyAsync(pinned.out ? out + a : l.out_h, l.out_d, q * sizeof(SA_Interval), cudaMemcpyDeviceToHost, s));
    if(alone)
        CUDA_CHECK(cudaFreeAsync(sym_d, s));

    // Staging reuses the lane's buffers, and staged intervals must reach `out`: the chunk must be done. Else, the stream orders it.
    if(!pinned.off || !pinned.sym || !pinned.out)
    {
        CUDA_CHECK(cudaStreamSynchronize(s));
        if(!pinned.out)
            std::memcpy(out + a, l.out_h, q * sizeof(SA_Interval));
    }
}


template <typename T_sym_>
void Query_Engine<T_sym_>::certify_async(const uint64_t* const off, const T_sym_* const sym, const idx_t q, const SA_Interval* const I,
                                         unsigned long long* const fail_c, unsigned long long* const first, const cudaStream_t s) const
{
    if(q == 0)
        return;

    const auto grid = static_cast<unsigned>(std::min<idx_t>((q + BLOCK - 1) / BLOCK, 65536));
    certify_kernel<<<grid, BLOCK, 0, s>>>(idx_.text<T_sym_>(), idx_.n(), idx_.sa(), off, sym, q, I, fail_c, first);
    CUDA_CHECK(cudaGetLastError());
}



// Template instantiations for the required instances.
template class Query_Engine<uint8_t>;
template class Query_Engine<uint16_t>;
template class Query_Engine<uint32_t>;

}
