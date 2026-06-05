
#include "Partition.cuh"
#include "utility.hpp"
#include "utility.cuh"

#include "omp.h"


namespace gpu_saca
{

Partition::Partition(const idx_t cap):
      suf(nullptr)
    , cap(cap)
{
    CUDA_CHECK(pinned_allocate_host(suf, cap));
}


Partition::~Partition()
{
    CUDA_CHECK(deallocate_host(suf));
}

}
