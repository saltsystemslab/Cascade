
#ifndef DISK_BACKEND_HPP
#define DISK_BACKEND_HPP



#include "types.hpp"
#include "Direct_File.hpp"
#include "Packed_Prefix.hpp"
#include "Spill_Store.hpp"
#include "Residency_Plan.hpp"

#include <vector>

namespace gpu_saca
{

// Partitions' storage backend in disk.
class Disk_Backend
{

private:

    std::vector<Direct_File> suf_;  // Suffix-files of the partitions.
    std::vector<Direct_File> key_;  // Key-files of the partitions.
    std::vector<Direct_File> prefix_;   // Prefix-key files of the partitions.

public:

    // Constructs a disk-backend for partitions with capacities `P_cap`, by
    // opening their spill-files (if required) in `spill`, as per the residency
    // plan `plan`.
    Disk_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const Residency_Plan& plan);

    Disk_Backend(const Disk_Backend&) = delete;
    Disk_Backend& operator=(const Disk_Backend&) = delete;

    // Returns the `p_id`'th partition's suffix-file.
    Direct_File& suf(const p_id_t p_id) { return suf_[p_id]; }

    // Returns the `p_id`'th partition's prefix-keys-file.
    Direct_File& prefix(const p_id_t p_id) { return prefix_[p_id]; }

    // Reads `sz`-many initial prefixes of the `p_id`'th partition, from its
    // offset `off`, into `dst`.
    void read_prefix(p_id_t p_id, idx_t off, idx_t sz, Prefix_Key* dst) const;

    // Reads `sz`-many suffixes of the `p_id`'th partition, from its offset
    // `off`, into `dst`.
    void read_suf(p_id_t p_id, idx_t off, idx_t sz, idx_t* dst) const;

    // Reads `sz`-many keys of the `p_id`'th partition, from its offset `off`,
    // into `dst`.
    void read_key(p_id_t p_id, idx_t off, idx_t sz, idx_t* dst) const;

    // Writes `sz`-many suffixes and keys of the `p_id`'th partition, at its
    // offset `off`, from `suf` and `key`.
    void write(p_id_t p_id, idx_t off, idx_t sz, const idx_t* suf, const idx_t* key);

    // Drops the prefix-key files from the backend.
    void drop_prefixes();
};

}



#endif
