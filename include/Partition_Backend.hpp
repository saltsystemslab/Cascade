
#ifndef PARTITION_BACKEND_HPP
#define PARTITION_BACKEND_HPP



#include "types.hpp"
#include "RAM_Backend.hpp"
#include "Disk_Backend.hpp"
#include "Staging_Slab.hpp"
#include "Spill_Store.hpp"
#include "Direct_File.hpp"
#include "Residency_Plan.hpp"

#include <cstddef>
#include <vector>


namespace gpu_saca
{

// Partitions' storage backend.
class Partition_Backend
{
private:

    const Residency_Plan plan_; // Host-residency plan for the partitions.
    RAM_Backend ram_;   // Storage for the resident partition-units.
    Disk_Backend disk_; // Storage for the spilled partition-units.

    idx_t* rho_;    // Global ranks of the suffixes.
    const idx_t n_; // Text-size.
    const idx_t secondary_bits_;    // Bit-width of the secondary-key of a suffix: `\ceil(\log_2(n))`.
    idx_t k_;   // Current stride (half of the context this generation is resolving) of the suffixes in the algorithm execution.
#ifdef TESTS
    const p_id_t* const P_id;   // Partition-IDs.
#endif


public:

    // Constructs a backend for partitions with capacities `P_cap`. It spills to
    // spill-store `spill` whenever the memory budget of the host (less its
    // fixed allocation cost `fixed_bytes) cannot hold some data. The global-
    // ranks array `rho` of size `n` and the secondary-key bit-width
    // `secondary_bits` are used to extend the ranks of the suffixes while they
    // are read off the backend.
#ifndef TESTS
    Partition_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, std::size_t fixed_bytes, idx_t* rho, idx_t n, idx_t secondary_bits);
#else
    Partition_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, std::size_t fixed_bytes, idx_t* rho, idx_t n, idx_t secondary_bits, const p_id_t* P_id);
#endif

    Partition_Backend(const Partition_Backend&) = delete;
    Partition_Backend& operator=(const Partition_Backend&) = delete;

    // Sets the stride to `k` for the rank-gather operations during partition-
    // reading.
    void set_stride(const idx_t k) { k_ = k; }

    // Returns the `p_id`'th partition's resident suffixes, if any.
    idx_t* resident_suf(const p_id_t p_id) { return plan_.data(p_id) ? ram_.suf(p_id) : nullptr; }

    // Returns the `p_id`'th partition's resident keys, if any.
    idx_t* resident_key(const p_id_t p_id) { return plan_.data(p_id) ? ram_.key(p_id) : nullptr; }

    // Returns the `p_id`'th partition's resident prefix-keys, if any.
    Prefix_Key* resident_prefix(const p_id_t p_id) { return plan_.prefix(p_id) ? ram_.prefix(p_id) : nullptr; }

    // Returns the `p_id`'th partition's spilled suffixes' file, if any.
    Direct_File* spilled_suf(const p_id_t p_id) { return plan_.data(p_id) ? nullptr : &disk_.suf(p_id); }

    // Returns the `p_id`'th partition's spilled prefix-keys' file, if any.
    Direct_File* spilled_prefix(const p_id_t p_id) { return plan_.prefix(p_id) ? nullptr : &disk_.prefix(p_id); }

    // Reads the staging slab `s`'s chunk of suffixes and their corresponding
    // prefix-keys from the backend into it.
    void read_prefixes(Staging_Slab& s) const;

    // Reads the staging slab `s`'s chunk of suffixes and their corresponding
    // ranks from the backend into it, and folds in their secondary keys into
    // the ranks, transforming the ranks into full keys.
    void read_keys(Staging_Slab& s) const;

    // Reads `sz` suffixes and their corresponding ranks of the `p_id`'th
    // partition, from its offset `off`, into `suf` and `key` respectively.
    void read_chunk(p_id_t p_id, idx_t off, idx_t sz, idx_t* suf, idx_t* key) const;

    // Reads `sz` suffixes and their corresponding initial prefix-keys of the
    // `p_id`'th partition, from its offset `off`, into `suf` and `key`
    // respectively.
    void read_prefix_chunk(p_id_t p_id, idx_t off, idx_t sz, idx_t* suf, Prefix_Key* key) const;

    // Writes the staging slab `s`'s chunk to the backend.
    void write(const Staging_Slab& s);

    // Drops the prefix-key files from the backend.
    void drop_prefixes();
};

}



#endif
