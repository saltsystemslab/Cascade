
#include "Query_Engine.hpp"
#include "CPU_Reference.hpp"
#include "Index_Meta.hpp"
#include "Packed_Prefix_Decomposer.cuh"
#include "sa40.hpp"
#include "utility.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <climits>


namespace gpu_saca
{

namespace
{

constexpr int BLOCK = 256;


// The ranks `[first, last]` that contain a boundary of an SA-interval.
struct Rank_Range
{
    idx_t first;
    idx_t last;
};


// Writes into `keys[i]` the first `min(m, K)` symbols of the query
// `sym[off[i], off[i + 1])` of length `m`, packed as a table key is, zero-padded
// to the key's `K` symbols; and `i` into `ids[i]` if `ids` is not null.
template <typename T_sym_, idx_t W_>
__global__ void pack_kernel(const uint64_t* const off, const T_sym_* const sym, const idx_t q, Packed_Prefix<W_>* const keys,
                            uint32_t* const ids)
{
    constexpr idx_t spw = sizeof(uint64_t) / sizeof(T_sym_);   // Symbols per word.
    for(idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; i < q; i += idx_t(gridDim.x) * blockDim.x)
    {
        const T_sym_* const P = sym + off[i];
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
// writes the ranks that contain their SA-intervals' boundaries into `br`.
// Position `g` of the batch is the query `ids[g]`, or `g` if `ids` is null.
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
                             const uint64_t* const off, const T_sym_* const sym, const Packed_Prefix<W_>* const keys,
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
        last = c * stride + (!upper && m > 0 && sym[off[i] + m - 1] == 0 ? m - 1 : 0);
    }
    else
    {
        first = (a > 0 ? (a - 1) * stride + 1 : 0);
        last = b * stride;
    }

    if(t < 2 * q)
        br[t] = Rank_Range{first, (last < n ? last : n)};
}


// Searches the suffix array `SA` over the text `T[0, n)` for the queries
// `sym[off[i], off[i + 1])` of a batch, into `out`. Position `g` of the batch is
// the query `ids[g]`, or `g` if `ids` is null. Adjacent threads pair up on a
// query: thread `2g` finds its interval's `lo`, the first rank whose truncated
// suffix is not smaller than the pattern, and thread `2g + 1` its `hi`, the
// first rank whose truncated suffix is greater. A thread searches the ranks
// `br[2g + lane]` if `br` is not null, and all of `[0, n]` otherwise; the pair's
// searches then share their early path.
template <typename T_sym_>
__global__ void search_kernel(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off,
                              const T_sym_* const sym, const uint32_t* const ids, const Rank_Range* const br, const idx_t q,
                              SA_Interval* const out)
{
    const idx_t t = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(t >= 2 * q)
        return;

    const idx_t g = t >> 1;
    const bool upper = t & 1;
    const idx_t i = (ids != nullptr ? ids[g] : g);
    const T_sym_* const P = sym + off[i];
    const idx_t m = off[i + 1] - off[i];

    // The answer is in `[low, low + len]`: the ranks before `low` go right, those from `low + len` do not.
    idx_t low = 0, len = n;
    if(br != nullptr)
        low = br[t].first, len = br[t].last - br[t].first;

    while(len > 0)
    {
        const idx_t half = len >> 1, mid = low + half;
        const int c = ref::cmp_suffix_pattern(T, n, sa40::load(SA, mid), P, m);
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


// Finds the SA-intervals of the device-side batch `sym[off[i], off[i + 1])`,
// `i < q`, over the index `idx` in two phases, with the sample table `tab` of
// keys of `W_` words; sorting the batch by its packed prefixes first if `sort`.
// Asynchronous on the stream `s`.
template <typename T_sym_, idx_t W_>
void find_two_phase(const Suffix_Index& idx, const Sample_Table& tab, const bool sort, const uint64_t* const off, const T_sym_* const sym,
                    const idx_t q, SA_Interval* const out, const cudaStream_t s)
{
    if(q > INT_MAX)
    {
        std::cerr << "Batch of " << q << " queries exceeds the supported " << INT_MAX << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    Packed_Prefix<W_>* keys[2] = {nullptr, nullptr};
    uint32_t* ids[2] = {nullptr, nullptr};
    Rank_Range* br = nullptr;
    void* sort_space = nullptr;
    CUDA_CHECK(cudaMallocAsync(&keys[0], q * sizeof(Packed_Prefix<W_>), s));
    CUDA_CHECK(cudaMallocAsync(&br, 2 * q * sizeof(Rank_Range), s));
    if(sort)
    {
        CUDA_CHECK(cudaMallocAsync(&keys[1], q * sizeof(Packed_Prefix<W_>), s));
        CUDA_CHECK(cudaMallocAsync(&ids[0], q * sizeof(uint32_t), s));
        CUDA_CHECK(cudaMallocAsync(&ids[1], q * sizeof(uint32_t), s));
    }

    const auto grid_q = static_cast<unsigned>(std::min<idx_t>((q + BLOCK - 1) / BLOCK, 65536));
    pack_kernel<T_sym_, W_><<<grid_q, BLOCK, 0, s>>>(off, sym, q, keys[0], ids[0]);
    CUDA_CHECK(cudaGetLastError());

    const Packed_Prefix<W_>* k = keys[0];
    const uint32_t* id = nullptr;
    if(sort)
    {
        cub::DoubleBuffer<Packed_Prefix<W_>> key_buf(keys[0], keys[1]);
        cub::DoubleBuffer<uint32_t> id_buf(ids[0], ids[1]);
        std::size_t sort_bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, key_buf, id_buf, static_cast<int>(q), Packed_Prefix_Decomposer<W_>{},
                                                   0, static_cast<int>(W_ * 64), s));
        CUDA_CHECK(cudaMallocAsync(&sort_space, sort_bytes, s));
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sort_space, sort_bytes, key_buf, id_buf, static_cast<int>(q), Packed_Prefix_Decomposer<W_>{},
                                                   0, static_cast<int>(W_ * 64), s));
        k = key_buf.Current();
        id = id_buf.Current();
    }

    const auto grid_2q = static_cast<unsigned>((2 * q + BLOCK - 1) / BLOCK);
    table_kernel<T_sym_, W_><<<grid_2q, BLOCK, 0, s>>>(tab.keys<W_>(), tab.size(), tab.stride(), idx.n(), off, sym, k, id, q, br);
    CUDA_CHECK(cudaGetLastError());
    search_kernel<<<grid_2q, BLOCK, 0, s>>>(idx.text<T_sym_>(), idx.n(), idx.sa(), off, sym, id, br, q, out);
    CUDA_CHECK(cudaGetLastError());

    for(void* const p : {static_cast<void*>(keys[0]), static_cast<void*>(keys[1]), static_cast<void*>(ids[0]), static_cast<void*>(ids[1]),
                         static_cast<void*>(br), sort_space})
        if(p != nullptr)
            CUDA_CHECK(cudaFreeAsync(p, s));
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

    Index_Meta m;
    if(!index_meta::read(dir, m, err))
        return false;

    if(m.sym_bytes != sizeof(T_sym_))
    {
        err = "index has " + std::to_string(m.sym_bytes) + "-byte symbols; the engine expects " + std::to_string(sizeof(T_sym_));
        return false;
    }

    tab_.release();
    if(!idx_.load(dir, opt.text_res, opt.sa_res, opt.spot, err))
        return false;

    opt_ = opt;
    if(!configure(opt.stride, opt.key_words, opt.sort, err))
    {
        idx_.release();
        return false;
    }

    return true;
}


template <typename T_sym_>
bool Query_Engine<T_sym_>::configure(const idx_t stride, const uint32_t key_words, const bool sort, std::string& err)
{
    tab_.release();
    opt_.stride = 0;
    opt_.key_words = 0;
    opt_.sort = sort;
    if(stride == 0)
        return true;

    const uint32_t w = (key_words != 0 ? key_words : (sizeof(T_sym_) == 4 ? 4 : 2));
    if(!tab_.build(idx_, stride, w, err))
        return false;

    opt_.stride = stride;
    opt_.key_words = w;
    return true;
}


template <typename T_sym_>
void Query_Engine<T_sym_>::find_async(const uint64_t* const off, const T_sym_* const sym, const idx_t q, SA_Interval* const out,
                                      const cudaStream_t s) const
{
    if(q == 0)
        return;

    if(!tab_.built())
    {
        const auto grid = static_cast<unsigned>((2 * q + BLOCK - 1) / BLOCK);
        search_kernel<<<grid, BLOCK, 0, s>>>(idx_.text<T_sym_>(), idx_.n(), idx_.sa(), off, sym, nullptr, nullptr, q, out);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    switch(tab_.key_words())
    {
    case 1: find_two_phase<T_sym_, 1>(idx_, tab_, opt_.sort, off, sym, q, out, s); break;
    case 2: find_two_phase<T_sym_, 2>(idx_, tab_, opt_.sort, off, sym, q, out, s); break;
    default:    find_two_phase<T_sym_, 4>(idx_, tab_, opt_.sort, off, sym, q, out, s);
    }
}


template <typename T_sym_>
void Query_Engine<T_sym_>::find(const uint64_t* const off, const T_sym_* const sym, const idx_t q, SA_Interval* const out) const
{
    if(q == 0)
        return;

    uint64_t* off_d = nullptr;
    T_sym_* sym_d = nullptr;
    SA_Interval* out_d = nullptr;
    const idx_t sym_c = off[q];
    CUDA_CHECK(cudaMalloc(&off_d, (q + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&sym_d, std::max<idx_t>(sym_c, 1) * sizeof(T_sym_)));
    CUDA_CHECK(cudaMalloc(&out_d, q * sizeof(SA_Interval)));
    CUDA_CHECK(cudaMemcpy(off_d, off, (q + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sym_d, sym, sym_c * sizeof(T_sym_), cudaMemcpyHostToDevice));

    find_async(off_d, sym_d, q, out_d, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaMemcpy(out, out_d, q * sizeof(SA_Interval), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(off_d));
    CUDA_CHECK(cudaFree(sym_d));
    CUDA_CHECK(cudaFree(out_d));
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
