
#include "Sample_Table.hpp"
#include "sa40.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <utility>


namespace gpu_saca
{

namespace
{

constexpr idx_t HOST_PIECE_KEYS = idx_t(1) << 22;   // Keys the host packs at a time, into a pinned buffer.


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

    // Built on the host if it can: its cores read the text faster than the device's random reads of host memory do.
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
