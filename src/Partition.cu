
#include "Partition.cuh"
#include "utility.hpp"

#include "omp.h"


namespace gpu_saca
{

Partition::Partition(const idx_t cap):
      suf(util::allocate<Suffix>(cap))
    , cap(cap)
    , sz(0)
    , w_buf(omp_get_max_threads())
{
    for(auto& buf : w_buf)
        buf.reserve(w_buf_cap);
}


Partition::~Partition()
{
    util::deallocate(suf);
}

}
