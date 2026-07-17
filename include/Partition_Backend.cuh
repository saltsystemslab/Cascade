
#ifndef PARTITION_BACKEND_CUH
#define PARTITION_BACKEND_CUH



#include "types.hpp"
#include "Staging_Slab.cuh"
#include "Partition.hpp"
#include "Prefix_Key.cuh"

#include <vector>


namespace gpu_saca
{

// Partition storage backend abstraction.
class Partition_Backend
{
public:

    // Reads the suffixes and their initial prefixes from the `p_id`'th
    // partition (of size `sz`) into `slab`.
    virtual void read_prefixes(p_id_t p_id, idx_t sz, Staging_Slab& s) const = 0;

    // Reads the suffixes and their ranks from the `p_id`'th partition (of size
    // `sz`) into `slab`.
    virtual void read_keys(p_id_t p_id, idx_t sz, Staging_Slab& s) const = 0;

    // Writes the suffixes and their ranks of the `p_id`'th partition (of size
    // `sz`) from `slab` to the backend.
    virtual void write(p_id_t p_id, idx_t sz, const Staging_Slab& s) = 0;
};


// RAM-backend for partitions: resides in unpinned host-space.
class RAM_Backend: public Partition_Backend
{
private:

    std::vector<Partition>& P_; // The partitions.
    const Prefix_Key* const prefix_;    // Arena for the initial prefixes of all the suffixes from all the partitions.
    const std::vector<idx_t>& P_off_;   // Per-partition offsets into flat collections of partition data (e.g. `prefix_` here).

public:

    RAM_Backend(std::vector<Partition>& P, const Prefix_Key* const prefix, const std::vector<idx_t>& P_off);

    void read_prefixes(p_id_t p_id, idx_t sz, Staging_Slab& s) const override;

    void read_keys(p_id_t p_id, idx_t sz, Staging_Slab& s) const override;

    void write(p_id_t p_id, idx_t sz, const Staging_Slab& s) override;
};

}



#endif
