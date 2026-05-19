
#ifndef GPU_SACA_HPP
#define GPU_SACA_HPP


#include <cstddef>


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


public:

    // Constructs a GPU-SACA object for the input text `T_h` of size `n`.
    GPU_SACA(const T_alpha_* T_h, std::size_t n);

    ~GPU_SACA();
};


}



#endif
