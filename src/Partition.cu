
#include "Partition.cuh"
#include "utility.hpp"

#include "omp.h"


namespace gpu_saca
{

Partition::Partition(const idx_t cap):
      suf(util::allocate<Suffix>(cap))
    , cap(cap)
{}


Partition::~Partition()
{
    util::deallocate(suf);
}

}
