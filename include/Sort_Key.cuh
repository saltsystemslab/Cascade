
#ifndef SORT_KEY_CUH
#define SORT_KEY_CUH


#include "Sort_Key.hpp"
#include "types.hpp"

#include "cuda/std/tuple"



namespace gpu_saca
{

// Decomposer of a 16-byte sort-key to its primary and secondary keys.
struct Sort_Key_Decomposer
{
    __device__ __host__
    cuda::std::tuple<idx_t&, idx_t&> operator()(Sort_Key& key) const
    {
        return {key.primary, key.secondary};
    }
};

}



#endif
