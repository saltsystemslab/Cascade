
#include "Partition.cuh"
#include "utility.hpp"


namespace gpu_saca
{

Partition::Partition(const idx_t cap):
      suf(util::allocate<Suffix>(cap))
    , cap(cap)
    , sz(0)
{}


Partition::~Partition()
{
    util::deallocate(suf);
}

}
