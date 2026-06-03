
#ifndef PARTITION_CUH
#define PARTITION_CUH



#include "types.hpp"
#include "Suffix.hpp"
#include "Spin_Lock.hpp"
#include "utility.hpp"

#include <cstddef>
#include <vector>


namespace gpu_saca
{

// A partition of suffixes.
class Partition
{
private:

    Suffix* const suf;  // The suffixes (along with sort-keys) in this partition.
    const idx_t cap;    // Capacity of the partition.
    idx_t sz;   // Size of the partition.

    util::Spin_Lock lock;   // Lock to update the partition in concurrent settings.

    using w_buf_t = std::vector<Suffix>;    // Type of worker-local buffers.
    std::vector<util::Padded<w_buf_t>> w_buf;   // Worker-local buffers.

    static constexpr std::size_t w_buf_cap = 4 * 1024 * 1024llu / sizeof(Suffix);   // Worker-local buffers overflow to the partitions at size: 4KB.

public:

    // Allocates this partition for `cap` many suffixes.
    Partition(idx_t cap);

    ~Partition();
};

}



#endif