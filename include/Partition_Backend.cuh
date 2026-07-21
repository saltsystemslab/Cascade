
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

    // Reads into `slab` the suffixes and their initial prefixes from the chunk
    // of the partition currently tagged in `slab`.
    virtual void read_prefixes(Staging_Slab& s) const = 0;

    // Reads into `slab` the suffixes and their ranks from the chunk of the
    // partition currently in `slab`.
    virtual void read_keys(Staging_Slab& s) const = 0;

    // Writes the chunk currently in `slab` into the backend.
    virtual void write(const Staging_Slab& s) = 0;
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

    void read_prefixes(Staging_Slab& s) const override;

    void read_keys(Staging_Slab& s) const override;

    void write(const Staging_Slab& s) override;
};

}



#endif
