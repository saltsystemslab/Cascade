
#ifndef PARTITION_CUH
#define PARTITION_CUH



#include "types.hpp"
#include "Suffix.hpp"


namespace gpu_saca
{

// A partition of suffixes.
class Partition
{
private:

    Suffix* const suf;  // The suffixes (along with sort-keys) in this partition.
    const idx_t cap;    // Capacity of the partition.
    idx_t sz;   // Size of the partition.

public:

    // Allocates this partition for `cap` many suffixes.
    Partition(idx_t cap);

    ~Partition();
};

}



#endif