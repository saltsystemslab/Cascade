
#ifndef DISK_BACKEND_HPP
#define DISK_BACKEND_HPP



#include "types.hpp"
#include "Direct_File.hpp"
#include "Packed_Prefix.hpp"
#include "Spill_Store.hpp"
#include "Residency_Plan.hpp"
#include "message.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace gpu_saca
{

class Partition_Backend;


// Partitions' storage backend in disk.
class Disk_Backend
{
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
    std::size_t read_parity_;   // Generation-parity currently being read from.

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
    auto& suf(const p_id_t p_id) { return suf_[p_id]; }

    // Returns the `p_id`'th partition's prefix-keys-file.
    auto& prefix(const p_id_t p_id) { return prefix_[p_id]; }

    // Returns the `p_id`'th partition's inbox message-target file for the
    // generation-parity currently being written.
    auto& msg_tgt_tag(const p_id_t p_id) { return msg_tgt_tag_[write_parity_][p_id]; }

    // Returns the `p_id`'th partition's inbox message-rank file for the
    // generation-parity currently being written.
    auto& msg_rank(const p_id_t p_id) { return msg_rank_[write_parity_][p_id]; }

    // Sets the current generation's index for the backend.
    void set_generation(const idx_t gen) { write_parity_ = write_parity(gen), read_parity_ = read_parity(gen); }

    // Reads `sz`-many initial prefixes of the `p_id`'th partition, from its
    // offset `off`, into `dst`.
    void read_prefix(p_id_t p_id, idx_t off, idx_t sz, Prefix_Key* dst) const;

    // Reads `sz`-many suffixes of the `p_id`'th partition, from its offset
    // `off`, into `dst`.
    void read_suf(p_id_t p_id, idx_t off, idx_t sz, idx_t* dst) const;

    // Reads `sz`-many keys of the `p_id`'th partition, from its offset `off`,
    // into `dst`.
    void read_key(p_id_t p_id, idx_t off, idx_t sz, p_sz_t* dst) const;

    // Reads `sz`-many messages of the `p_id`'th partition's inbox for the
    // current generation, from its offset `off`, into `tgt` and `rank`.
    void read_messages(p_id_t p_id, p_sz_t off, p_sz_t sz, idx_t* tgt, msg_rank_t* rank) const;

    // Writes `sz`-many suffixes and keys of the `p_id`'th partition, at its
    // offset `off`, from `suf` and `key`.
    void write(p_id_t p_id, idx_t off, idx_t sz, const idx_t* suf, const p_sz_t* key);

    // Retires the `p_id`'th partition's suffix- and key-files.
    void retire_data(p_id_t p_id);

    // Drops the prefix-key files from the backend.
    void drop_prefixes();
};

}



#endif
