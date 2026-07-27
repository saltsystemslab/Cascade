
#include "Partition.hpp"

namespace gpu_saca
{

Partition::Partition(const idx_t cap):
      suf_(nullptr)
    , key_(nullptr)
    , cap_(cap)
{}


Partition::~Partition()
{}

}
