
#include "Sample_Table.hpp"
#include "sa40.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>


namespace gpu_saca
{

namespace
{

constexpr idx_t HOST_PIECE_KEYS = idx_t(1) << 22;   // Keys the host packs at a time, into a pinned buffer.
constexpr std::size_t SCAN_PIECE_BYTES = SA_WIDTH * 4096 * 400; // Bytes of a piece of a streamed suffix array, ~8 MB: whole entries, and whole 4 KB blocks.


// Writes into `tab[0, J)` the packed prefixes of the suffixes at ranks
// `0, stride, 2 stride, ...` of the suffix array `SA` over the text `T`; the
// text's zero padding pads the prefixes of its last suffixes.
template <typename T_sym_, idx_t W_>
__global__ void build_kernel(const T_sym_* const T, const uint8_t* const SA, const idx_t stride, const idx_t J, Packed_Prefix<W_>* const tab)
{
    for(idx_t j = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; j < J; j += idx_t(gridDim.x) * blockDim.x)
        tab[j] = Packed_Prefix<W_>::pack_prefix(T + sa40::load(SA, j * stride));
}


// Launches `build_kernel` for keys of `W_` words over symbols of `T_sym_`.
template <typename T_sym_, idx_t W_>
cudaError_t build(const Suffix_Index& idx, const idx_t stride, const idx_t J, void* const tab)
{
    static_assert(W_ * sizeof(uint64_t) <= TEXT_PAD_BYTES, "a key must not read past the text's padding");

    const int block = 256;
    const auto grid = static_cast<unsigned>(std::min<idx_t>((J + block - 1) / block, 65536));
    build_kernel<T_sym_, W_><<<grid, block>>>(idx.text<T_sym_>(), idx.sa(), stride, J, static_cast<Packed_Prefix<W_>*>(tab));
    const auto e = cudaGetLastError();
    return e != cudaSuccess ? e : cudaDeviceSynchronize();
}


// Dispatches `build` over the key width `W`.
template <typename T_sym_>
cudaError_t build(const Suffix_Index& idx, const idx_t stride, const idx_t J, const uint32_t W, void* const tab)
{
    switch(W)
    {
    case 1: return build<T_sym_, 1>(idx, stride, J, tab);
    case 2: return build<T_sym_, 2>(idx, stride, J, tab);
    default:    return build<T_sym_, 4>(idx, stride, J, tab);
    }
}


// Writes into `tab[0, j1 - j0)` the packed prefixes of the suffixes at ranks
// `j stride`, `j0 <= j < j1`, of the host-side suffix array `SA` over the text
// `T`, with the host's threads.
template <typename T_sym_, idx_t W_>
void pack_on_host(const T_sym_* const T, const uint8_t* const SA, const idx_t stride, const idx_t j0, const idx_t j1, Packed_Prefix<W_>* const tab)
{
    #pragma omp parallel for schedule(static)
    for(idx_t j = j0; j < j1; ++j)
        tab[j - j0] = Packed_Prefix<W_>::pack_prefix(T + sa40::load(SA, j * stride));
}


// Builds the `J` keys of `W_` words of the table of stride `stride` over the
// host-accessible index `idx` into the device's `tab`, on the host, a piece at
// a time through two pinned buffers: packing a piece overlaps copying the one
// before.
template <typename T_sym_, idx_t W_>
cudaError_t build_on_host(const Suffix_Index& idx, const idx_t stride, const idx_t J, void* const tab)
{
    static_assert(W_ * sizeof(uint64_t) <= TEXT_PAD_BYTES, "a key must not read past the text's padding");

    const idx_t piece = std::min(J, HOST_PIECE_KEYS);
    Packed_Prefix<W_>* stage = nullptr;
    cudaStream_t s = nullptr;
    cudaEvent_t copied[2] = {nullptr, nullptr};
    cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(&stage), 2 * piece * sizeof(Packed_Prefix<W_>), cudaHostAllocDefault);
    if(e == cudaSuccess)    e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    for(auto& ev : copied)
        if(e == cudaSuccess)    e = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);

    for(idx_t j0 = 0, k = 0; e == cudaSuccess && j0 < J; j0 += piece, ++k)
    {
        const idx_t j1 = std::min(J, j0 + piece);
        Packed_Prefix<W_>* const buf = stage + (k & 1) * piece;
        e = cudaEventSynchronize(copied[k & 1]);    // The buffer's previous piece is copied; returns at once before its first.
        if(e != cudaSuccess)
            break;

        pack_on_host<T_sym_, W_>(idx.host_text<T_sym_>(), idx.host_sa(), stride, j0, j1, buf);
        e = cudaMemcpyAsync(static_cast<Packed_Prefix<W_>*>(tab) + j0, buf, (j1 - j0) * sizeof(Packed_Prefix<W_>), cudaMemcpyHostToDevice, s);
        if(e == cudaSuccess)    e = cudaEventRecord(copied[k & 1], s);
    }

    const cudaError_t done = (s != nullptr ? cudaStreamSynchronize(s) : cudaSuccess);
    for(const auto ev : copied)
        if(ev != nullptr)
            cudaEventDestroy(ev);
    if(s != nullptr)
        cudaStreamDestroy(s);
    cudaFreeHost(stage);
    return e != cudaSuccess ? e : done;
}


// Dispatches `build_on_host` over the key width `W`.
template <typename T_sym_>
cudaError_t build_on_host(const Suffix_Index& idx, const idx_t stride, const idx_t J, const uint32_t W, void* const tab)
{
    switch(W)
    {
    case 1: return build_on_host<T_sym_, 1>(idx, stride, J, tab);
    case 2: return build_on_host<T_sym_, 2>(idx, stride, J, tab);
    default:    return build_on_host<T_sym_, 4>(idx, stride, J, tab);
    }
}


// Builds the `J` keys of `W_` words of the table of stride `stride` over the
// index `idx`, whose suffix array is mapped, into the device's `tab`, on the
// host: in one pass over the suffix array's files, which streams them from the
// drives in large reads, all at once, rather than a page at a time through the
// mapping, and leaves the page cache as it was. A sample whose entry straddles
// two parts is read through the mapping. Returns `false` with a reason in `err`
// on failure.
template <typename T_sym_, idx_t W_>
bool build_streamed(const Suffix_Index& idx, const idx_t stride, const idx_t J, void* const tab, std::string& err)
{
    static_assert(W_ * sizeof(uint64_t) <= TEXT_PAD_BYTES, "a key must not read past the text's padding");

    const T_sym_* const T = idx.host_text<T_sym_>();
    const idx_t span = SA_WIDTH * stride;   // Bytes from a sample's entry to the next's.
    const std::unique_ptr<Packed_Prefix<W_>[]> keys(new Packed_Prefix<W_>[J]);

    // The samples whose entries a piece holds whole; the pieces end at whole entries but at the parts' ends.
    const auto pack = [&](const idx_t off, const uint8_t* const p, const std::size_t len)
    {
        for(idx_t j = (off + span - 1) / span; j < J && j * span + SA_WIDTH <= off + len; ++j)
            keys[j] = Packed_Prefix<W_>::pack_prefix(T + sa40::load_at(p, j * span - off));
    };

    if(!index_io::scan_sa(idx.meta(), SCAN_PIECE_BYTES, pack, err))
        return false;

    const auto& part_off = idx.meta().sa_part_off;
    for(std::size_t i = 1; i < part_off.size(); ++i)
    {
        const idx_t j = (part_off[i] - 1) / span;   // The last sample to start before the part.
        if(j < J && j * span + SA_WIDTH > part_off[i])
            keys[j] = Packed_Prefix<W_>::pack_prefix(T + sa40::load(idx.host_sa(), j * stride));
    }

    // A copy from pageable memory may return before its last transfer lands, and the searches' streams do not wait for the default one.
    cudaError_t e = cudaMemcpy(tab, keys.get(), J * sizeof(Packed_Prefix<W_>), cudaMemcpyHostToDevice);
    if(e == cudaSuccess)
        e = cudaDeviceSynchronize();
    if(e != cudaSuccess)
    {
        err = std::string("copying the sample table to the device failed: ") + cudaGetErrorString(e);
        return false;
    }

    return true;
}


// Dispatches `build_streamed` over the key width `W`.
template <typename T_sym_>
bool build_streamed(const Suffix_Index& idx, const idx_t stride, const idx_t J, const uint32_t W, void* const tab, std::string& err)
{
    switch(W)
    {
    case 1: return build_streamed<T_sym_, 1>(idx, stride, J, tab, err);
    case 2: return build_streamed<T_sym_, 2>(idx, stride, J, tab, err);
    default:    return build_streamed<T_sym_, 4>(idx, stride, J, tab, err);
    }
}

}


Sample_Table::Sample_Table(Sample_Table&& o) noexcept:
      keys_(std::exchange(o.keys_, nullptr))
    , size_(std::exchange(o.size_, 0))
    , stride_(std::exchange(o.stride_, 0))
    , key_words_(std::exchange(o.key_words_, 0))
    , key_symbols_(std::exchange(o.key_symbols_, 0))
    , t_build_(std::exchange(o.t_build_, -1))
{}


Sample_Table& Sample_Table::operator=(Sample_Table&& o) noexcept
{
    if(this != &o)
    {
        release();
        keys_ = std::exchange(o.keys_, nullptr);
        size_ = std::exchange(o.size_, 0);
        stride_ = std::exchange(o.stride_, 0);
        key_words_ = std::exchange(o.key_words_, 0);
        key_symbols_ = std::exchange(o.key_symbols_, 0);
        t_build_ = std::exchange(o.t_build_, -1);
    }

    return *this;
}


bool Sample_Table::build(const Suffix_Index& idx, const idx_t stride, const uint32_t key_words, std::string& err)
{
    release();
    if(!idx.loaded() || (!idx.device_accessible() && !idx.host_accessible()))
    {
        err = "the sample table needs a loaded index that the device or the host can access";
        return false;
    }

    if(stride == 0 || (key_words != 1 && key_words != 2 && key_words != 4))
    {
        err = "the sample table needs a positive stride and keys of 1, 2, or 4 words";
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const idx_t J = (idx.n() + stride - 1) / stride;
    void* keys = nullptr;
    auto e = cudaMalloc(&keys, J * key_words * sizeof(uint64_t));
    if(e != cudaSuccess)
    {
        err = std::string("allocating the sample table failed: ") + cudaGetErrorString(e);
        return false;
    }

    // A mapped suffix array streams from its files. Else, built on the host if it can: its cores read the text faster than the device's
    // random reads of host memory do.
    if(idx.sa_residency() == Residency::mapped)
    {
        bool ok = false;
        switch(idx.sym_bytes())
        {
        case 1: ok = build_streamed<uint8_t>(idx, stride, J, key_words, keys, err); break;
        case 2: ok = build_streamed<uint16_t>(idx, stride, J, key_words, keys, err); break;
        default:    ok = build_streamed<uint32_t>(idx, stride, J, key_words, keys, err);
        }

        if(!ok)
        {
            err = "building the sample table failed: " + err;
            cudaFree(keys);
            return false;
        }
    }
    else
    {
        const bool on_host = idx.host_accessible();
        switch(idx.sym_bytes())
        {
        case 1: e = (on_host ? build_on_host<uint8_t>(idx, stride, J, key_words, keys) : gpu_saca::build<uint8_t>(idx, stride, J, key_words, keys)); break;
        case 2: e = (on_host ? build_on_host<uint16_t>(idx, stride, J, key_words, keys) : gpu_saca::build<uint16_t>(idx, stride, J, key_words, keys)); break;
        default:    e = (on_host ? build_on_host<uint32_t>(idx, stride, J, key_words, keys) : gpu_saca::build<uint32_t>(idx, stride, J, key_words, keys));
        }

        if(e != cudaSuccess)
        {
            err = std::string("building the sample table failed: ") + cudaGetErrorString(e);
            cudaFree(keys);
            return false;
        }
    }

    keys_ = keys;

    size_ = J;
    stride_ = stride;
    key_words_ = key_words;
    key_symbols_ = key_words * sizeof(uint64_t) / idx.sym_bytes();
    t_build_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}


void Sample_Table::release()
{
    if(keys_ != nullptr)
        cudaFree(keys_);

    keys_ = nullptr;
    size_ = stride_ = key_symbols_ = 0;
    key_words_ = 0;
    t_build_ = -1;
}

}
