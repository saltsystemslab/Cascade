
#ifndef DISK_BACKEND_HPP
#define DISK_BACKEND_HPP



#include "types.hpp"
#include "Direct_File.hpp"
#include "Packed_Prefix.hpp"
#include "Spill_Store.hpp"
#include "Residency_Plan.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace gpu_saca
{

class Partition_Backend;


// Partitions' storage backend in disk.
class Disk_Backend
{
    friend class Partition_Backend;

private:

    std::vector<Direct_File> suf_;  // Suffix-files of the partitions.
    std::vector<Direct_File> key_;  // Key-files of the partitions.
    std::vector<Direct_File> prefix_;   // Prefix-key files of the partitions.

    std::vector<Direct_File> msg_tgt_tag_[2];   // Message-target files of the partitions' inboxes, per generation-parity.
    std::vector<Direct_File> msg_rank_[2];  // Message-rank files of the partitions' inboxes, per generation-parity.

    Spill_Store* const spill_;  // Spill-store to open the deferred files from.
    const std::vector<idx_t> P_cap_;    // Partition capacities.
    const std::vector<uint8_t> data_resident_;  // Whether each partition's data is host-resident.

    // TODO: all the following generation-parity logic need to move upwards in the hierarchy: to partition-backend or somewhere else.

    std::size_t write_parity_;  // Generation-parity currently being written to.


    // Returns the inbox-parity a generation with index `gen` writes to.
    static constexpr std::size_t write_parity(const idx_t gen) { return (gen & 1) ^ 1; }

    // Returns the `p_id`'th partition's inbox message-target file for the
    // generation-parity currently being written.
    auto& msg_tgt_tag(const p_id_t p_id) { return msg_tgt_tag_[write_parity_][p_id]; }

    // Returns the `p_id`'th partition's inbox message-rank file for the
    // generation-parity currently being written.
    auto& msg_rank(const p_id_t p_id) { return msg_rank_[write_parity_][p_id]; }

    // Sets the current generation's index for the backend.
    void set_generation(const idx_t gen) { write_parity_ = write_parity(gen); }

public:

    // Constructs a disk-backend for partitions with capacities `P_cap`, by
    // opening their spill-files (if required) in `spill`, as per the residency
    // plan `plan`.
    Disk_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const Residency_Plan& plan);

    Disk_Backend(const Disk_Backend&) = delete;
    Disk_Backend& operator=(const Disk_Backend&) = delete;

    // Opens the inbox-files of the parity first written at generation 1.
    void open_deferred_inboxes();

    // Retires the `p_id`'th partition's prefix-file, and opens its key-file.
    void retire_prefix_and_open_key(p_id_t p_id);

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
