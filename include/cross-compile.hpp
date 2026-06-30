
#ifndef CROSS_COMPILE_HPP
#define CROSS_COMPILE_HPP



// `__CUDACC__` is defined by `nvcc` when it is compiling CUDA source.
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#compilation-phases
#ifdef __CUDACC__
    #define CUDA_SPECIFIER __host__ __device__
#else
    #define CUDA_SPECIFIER
#endif



#endif
