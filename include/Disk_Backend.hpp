
#ifndef DISK_BACKEND_HPP
#define DISK_BACKEND_HPP



#include "types.hpp"
#include "Direct_File.hpp"
#include "Spill_Store.hpp"
#include "Residency_Plan.hpp"
#include "message.hpp"

#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>

namespace gpu_saca
{

class Partition_Backend;


// Partitions' storage backend in disk.
class Disk_Backend
{
private:

    std::vector<Direct_File> suf_;  // Suffix-files of the partitions.
    std::vector<Direct_File> key_;  // Key-files of the partitions.
    std::unique_ptr<std::once_flag[]> suf_open_;    // Opening-flags of the partitions' suffix-files.
    std::unique_ptr<std::once_flag[]> key_open_;    // Opening-flags of the partitions' key-files.

    std::vector<Direct_File> msg_tgt_[2];   // Message-target files of the partitions' inboxes, per generation-parity.
    std::vector<Direct_File> msg_rank_[2];  // Message-rank files of the partitions' inboxes, per generation-parity.
    std::unique_ptr<std::once_flag[]> inbox_open_[2];   // Opening-flags of the partitions' inbox-files, per generation-parity.

    Spill_Store* const spill_;  // Spill-store to open the deferred files from.
    const std::vector<idx_t> P_cap_;    // Partition capacities.

    // TODO: all the following generation-parity logic need to move upwards in the hierarchy: to partition-backend or somewhere else.

    std::size_t write_parity_;  // Generation-parity currently being written to.
    std::size_t read_parity_;   // Generation-parity currently being read from.


    // Opens the `p_id`'th partition's suffix-file, unless already open. Thread-
    // safe.
    void ensure_suf(p_id_t p_id);

    // Opens the `p_id`'th partition's key-file, unless already open. Thread-
    // safe.
    void ensure_key(p_id_t p_id);

public:

    // Constructs a disk-backend for partitions with capacities `P_cap`, in the
    // spill-store `spill`. The suffix-files of the partitions with suffixes
    // spilled per the residency plan `plan` open upfront; the rest open lazily,
    // at their first write.
    Disk_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const Residency_Plan& plan);

    Disk_Backend(const Disk_Backend&) = delete;
    Disk_Backend& operator=(const Disk_Backend&) = delete;

    // Opens the `p_id`'th partition's inbox-files for the generation-parity
    // currently being written, unless already open. Thread-safe.
    void ensure_inbox(p_id_t p_id);

    // Retires the partitions' inbox-files of both the generation-parities.
    void retire_inboxes();

    // Returns the `p_id`'th partition's inbox message-target file for the
    // generation-parity currently being written.
    auto& msg_tgt(const p_id_t p_id) { return msg_tgt_[write_parity_][p_id]; }

    // Returns the `p_id`'th partition's inbox message-rank file for the
    // generation-parity currently being written.
    auto& msg_rank(const p_id_t p_id) { return msg_rank_[write_parity_][p_id]; }

    // Sets the current generation's index for the backend.
    void set_generation(const idx_t gen) { write_parity_ = write_parity(gen), read_parity_ = read_parity(gen); }

    // Reads `sz`-many suffixes of the `p_id`'th partition, from its offset
    // `off`, into `dst`.
    void read_suf(p_id_t p_id, idx_t off, idx_t sz, idx_t* dst) const;

    // Reads `sz`-many keys of the `p_id`'th partition, from its offset `off`,
    // into `dst`.
    void read_key(p_id_t p_id, idx_t off, idx_t sz, p_sz_t* dst) const;

    // Reads `sz`-many messages of the `p_id`'th partition's inbox for the
    // current generation, from its offset `off`, into `tgt` and `rank`.
    void read_messages(p_id_t p_id, p_sz_t off, p_sz_t sz, idx_t* tgt, msg_rank_t* rank) const;

    // Writes `sz`-many suffixes of the `p_id`'th partition, at its offset
    // `off`, from `src`.
    void write_suf(p_id_t p_id, idx_t off, idx_t sz, const idx_t* src);

    // Writes `sz`-many keys of the `p_id`'th partition, at its offset `off`,
    // from `src`.
    void write_key(p_id_t p_id, idx_t off, idx_t sz, const p_sz_t* src);

    // Retires the `p_id`'th partition's suffix- and key-files.
    void retire_data(p_id_t p_id);
};

}



#endif
