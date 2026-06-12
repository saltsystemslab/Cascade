
#include "Partition.hpp"
#include "utility.cuh"

#include "omp.h"


namespace gpu_saca
{

Partition::Partition(const idx_t sz):
      suf_(nullptr)
    , key_(nullptr)
    , sz_(sz)
{
    CUDA_CHECK(pinned_allocate_host(suf_, sz_));
    CUDA_CHECK(pinned_allocate_host(key_, sz_));
}


Partition::~Partition()
{
    CUDA_CHECK(deallocate_host(suf_));
    CUDA_CHECK(deallocate_host(key_));
}

}
