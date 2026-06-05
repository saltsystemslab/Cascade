
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

public:

    // Allocates this partition for `cap` many suffixes.
    Partition(idx_t cap);

    ~Partition();

    // Returns pointer to the (mutable) data of the partition.
    auto data() { return suf; }
};

}



#endif