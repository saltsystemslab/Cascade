
#ifndef GPU_SACA_HPP
#define GPU_SACA_HPP


#include <cstddef>
#include <limits>


// TODO: name algorithm.
namespace gpu_saca
{


// The suffix array constructor class in GPUs.
// TODO:
//      1. Templatize alphabet
//      2. Batch out data to host memory; currently everything will be in device memory
class GPU_SACA
{
    typedef char T_alpha_;  // Type of the alphabet—this needs to be templatized.

private:

    T_alpha_* T_d_; // The text in device.
    const std::size_t n_;   // Size of the text.

    const std::size_t p_;   // Number of partitions for the SA.


public:

    // Constructs a GPU-SACA object for the input text `T_h` of size `n`.
    GPU_SACA(const T_alpha_* T_h, std::size_t n);

    ~GPU_SACA();

    // Constructs the suffix array at `SA`.
    void build_SA(std::size_t* SA_h);
};


// Returns `1` if `x` is in the upper half of `T_sym_`'s range, `0` otherwise.
template <typename T_sym_>
__device__ __forceinline__
uint8_t binarize(const T_sym_ x)
{
    // Maximum element of the lower half: exactly `2^(N - 1)` values at each half.
    constexpr T_sym_ mid = std::numeric_limits<T_sym_>::min() / 2 + std::numeric_limits<T_sym_>::max() / 2;
    return x <= mid ? 0 : 1;
}


}



#endif
