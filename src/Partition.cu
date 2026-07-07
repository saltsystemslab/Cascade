
#include "Partition.hpp"
#include "utility.cuh"

#include "omp.h"


namespace gpu_saca
{

Partition::Partition(const idx_t cap):
      suf_(nullptr)
    , key_(nullptr)
    , cap_(cap)
    , sz_(cap)
{
    CUDA_CHECK(pinned_allocate_host(suf_, cap_));
    CUDA_CHECK(pinned_allocate_host(key_, cap_));
}


Partition::~Partition()
{
    CUDA_CHECK(deallocate_host(suf_));
    CUDA_CHECK(deallocate_host(key_));
}

}
