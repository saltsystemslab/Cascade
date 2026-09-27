
#include "Query_Engine.hpp"
#include "CPU_Reference.hpp"
#include "Index_Meta.hpp"
#include "sa40.hpp"
#include "utility.cuh"

#include <algorithm>


namespace gpu_saca
{

namespace
{

constexpr int BLOCK = 256;


// Searches the whole suffix array `SA` over the text `T[0, n)` for the queries
// `sym[off[i], off[i + 1])`, `i < q`, into `out[0, q)`. Adjacent threads pair up
// on a query: thread `2i` finds its interval's `lo`, the first rank whose
// truncated suffix is not smaller than the pattern, and thread `2i + 1` its `hi`,
// the first rank whose truncated suffix is greater; the pair's searches share
// their early path.
template <typename T_sym_>
__global__ void find_direct_kernel(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off,
                                   const T_sym_* const sym, const idx_t q, SA_Interval* const out)
{
    const idx_t t = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(t >= 2 * q)
        return;

    const idx_t i = t >> 1;
    const bool upper = t & 1;
    const T_sym_* const P = sym + off[i];
    const idx_t m = off[i + 1] - off[i];

    // The answer is in `[low, low + len]`: the ranks before `low` go right, those from `low + len` do not.
    idx_t low = 0, len = n;
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

    return idx_.load(dir, opt.text_res, opt.sa_res, opt.spot, err);
}


template <typename T_sym_>
void Query_Engine<T_sym_>::find_async(const uint64_t* const off, const T_sym_* const sym, const idx_t q, SA_Interval* const out,
                                      const cudaStream_t s) const
{
    if(q == 0)
        return;

    const auto grid = static_cast<unsigned>((2 * q + BLOCK - 1) / BLOCK);
    find_direct_kernel<<<grid, BLOCK, 0, s>>>(idx_.text<T_sym_>(), idx_.n(), idx_.sa(), off, sym, q, out);
    CUDA_CHECK(cudaGetLastError());
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
