
#ifndef DISK_BACKEND_HPP
#define DISK_BACKEND_HPP



#include "types.hpp"
#include "Direct_File.hpp"
#include "Spill_Store.hpp"
#include "Staging_Slab.hpp"

#include <vector>

namespace gpu_saca
{

// Partitions' storage backend in disk.
class Disk_Backend
{

private:

    Spill_Store& spill_;    // The partitions' spill-files manager.

    std::vector<Direct_File> suf_;  // Suffix-files of the partitions.
    std::vector<Direct_File> key_;  // Key-files of the partitions.
    std::vector<Direct_File> prefix_;   // Prefix-key files of the partitions.

    idx_t* rho_;    // Global ranks of the suffixes.
    const idx_t n_; // Text-size.
    const idx_t secondary_bits_;    // Bit-width of the secondary-key of a suffix: `\ceil(\log_2(n))`.
    idx_t k_;   // Current prefix-context of the suffixes in the algorithm execution.

public:

    // Constructs a disk-backend for `p`-many partitions with their sizes in
    // `P_sz`, by opening their spill-files in `spill`; the global-ranks array
    // `rho` of size `n` and the secondary-key bit-width `secondary_bits` are
    // used to extend the ranks of the suffixes while they are read off the
    // backend.
    Disk_Backend(Spill_Store& spill, p_id_t p, const std::vector<idx_t>& P_sz, idx_t* rho, idx_t n, idx_t secondary_bits);

    Disk_Backend(const Disk_Backend&) = delete;
    Disk_Backend& operator=(const Disk_Backend&) = delete;

    // Sets the prefix-context to `k` for the rank-gather operations during
    // partition-reading.
    void set_prefix_context(const idx_t k) { k_ = k; }

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

    // Writes the staging slab `s`'s chunk to the backend.
    void write(const Staging_Slab& s);

    // Drops the prefix-key files from the backend.
    void drop_prefixes();
};

}



#endif
