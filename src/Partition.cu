
#include "Partition.hpp"
#include "utility.hpp"

#include "omp.h"


namespace gpu_saca
{

Partition::Partition(const idx_t cap):
      suf_(nullptr)
    , key_(nullptr)
    , cap_(cap)
{
    suf_ = util::allocate<idx_t>(cap_);
    key_ = util::allocate<idx_t>(cap_);
}


Partition::~Partition()
{
    util::deallocate(suf_);
    util::deallocate(key_);
}

}
