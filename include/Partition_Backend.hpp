
#ifndef PARTITION_BACKEND_HPP
#define PARTITION_BACKEND_HPP



#include "types.hpp"
#include "RAM_Backend.hpp"
#include "Disk_Backend.hpp"
#include "Staging_Slab.hpp"
#include "Spill_Store.hpp"
#include "Residency_Plan.hpp"
#include "Spin_Lock.hpp"
#include "message.hpp"

#include <cstddef>
#include <atomic>
#include <vector>


namespace gpu_saca
{

// Partitions' storage backend.
class Partition_Backend
{
private:

    const p_id_t p_;    // Count of partitions.
    const Residency_Plan plan_; // Host-residency plan for the partitions.
    RAM_Backend ram_;   // Storage for the resident partition-units.
    Disk_Backend disk_; // Storage for the spilled partition-units.

    p_sz_t* msg_off_;   // Storage for the exclusive-ending offsets of the destination-partitions into the messages produced by a partition.
                        // All partitions are packed flat together, so has `p x p` entries.

    // Per-destination-partition inbox state.
    // {
        static constexpr p_sz_t OV_CAP = 64 * MSG_ALIGN;        // Capacity of a destination-partition's block-overflow buffer, in messages.

        std::size_t inbox_write_parity_;    // Inbox-parity this generation writes to.
        std::size_t inbox_read_parity_; // Inbox-parity this generation reads from.

        std::vector<std::atomic<p_sz_t>> msg_count_[2]; // Size of each destination-partition's inbox, including the buffered messages; per generation-parity.
        std::vector<std::atomic<p_sz_t>> msg_written_[2];   // Messages actually written to the backend, per destination; per generation-parity.

        std::vector<idx_t*> ov_tgt_;    // Block-overflow buffers for message-targets, per destination.
        std::vector<msg_rank_t*> ov_rank_;  // Rank-buffers, parallel to `ov_tgt_`.
        std::vector<p_sz_t> ov_sz_; // Messages currently buffered (due to block-alignment overhanging), per destination.
        std::vector<util::Spin_Lock> ov_lock_;  // Mutually-exclusive lock to the overflow buffers, per destination.
    // }


    // Flushes the `p_id`'th inbox's block-overflow buffer. `final` denotes
    // whether this is the last flush invoked on the buffer or not.
    void flush_overflow(p_id_t p_id, bool final);

    // Buffers the index-range `[l, r)` of slab `s`'s messages into the
    // `p_id`'th inbox's block-overflow buffer, flushing when required.
    void buffer_overhang(p_id_t p_id, const Staging_Slab& s, p_sz_t l, p_sz_t r);

    // Appends the `[l, r)` slot-range of the slab `s`'s messages, destined for
    // the `p_id`'th inbox.
    void append_messages(p_id_t p_id, const Staging_Slab& s, p_sz_t l, p_sz_t r);


public:

    // Constructs a backend for partitions with capacities `P_cap`. It spills to
    // spill-store `spill` whenever the memory `budget` of the host, less its
    // fixed allocation cost `fixed_bytes` cannot hold some data. The global-
    // ranks array `rho` of size `n` is used to extend the ranks of the suffixes
    // while they are read off the backend. Partition-ID map `P_id` feeds the routing tags.
    Partition_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, std::size_t budget, std::size_t fixed_bytes, p_id_t p);

    Partition_Backend(const Partition_Backend&) = delete;
    Partition_Backend& operator=(const Partition_Backend&) = delete;

    ~Partition_Backend();

    // Returns the fixed RAM-budget (in bytes) required by the backend for `p`
    // partitions. Currently: the block-overflow buffers for the messages.
    static std::size_t fixed_budget(p_id_t p);

    // Returns the count of partitions.
    auto p() const { return p_; }

    /// Opens the inbox-files of the parity first written at generation 1.
    void open_deferred_inboxes();

    // Opens the `p_id`'th partition's key-file.
    void open_key(p_id_t p_id);

    // Returns the (exclusive) ending-offsets of the destination-partitions into
    // the messages produced by the `p_id`'th partition.
    auto msg_off(const p_id_t p_id) const { return msg_off_ + p_id * p_; }

    // Returns the size of the `q`'th inbox at the generation last written to.
    p_sz_t msg_count(const p_id_t q) const { return msg_count_[inbox_write_parity_][q]; }

    // Returns the actual count of message-entries written to the `q`'th inbox's
    // backend; this can exceed `msg_count(q)` due to disk-block alignment
    // requirements.
    p_sz_t msg_written(const p_id_t q) const { return msg_written_[inbox_write_parity_][q]; }

    // Returns the `p_id`'th partition's resident suffixes, if any.
    auto* resident_suf(const p_id_t p_id) { return plan_.data(p_id) ? ram_.suf(p_id) : nullptr; }

    // Returns the `p_id`'th partition's resident keys, if any.
    auto* resident_key(const p_id_t p_id) { return plan_.data(p_id) ? ram_.key(p_id) : nullptr; }

    // Returns the `p_id`'th partition's spilled suffixes' file, if any.
    auto* spilled_suf(const p_id_t p_id) { return plan_.data(p_id) ? nullptr : &disk_.suf(p_id); }

    // Reads the staging slab `s`'s chunk of suffixes from the backend into it.
    void read_suffixes(Staging_Slab& s) const;

    // Reads the staging slab `s`'s chunk of suffixes and their corresponding
    // ranks from the backend into it.
    void read_keys(Staging_Slab& s) const;

    // Reads `sz` suffixes of the `p_id`'th partition, from its offset `off`,
    // into `suf`.
    void read_suf_chunk(p_id_t p_id, idx_t off, idx_t sz, idx_t* suf) const;

    // Reads `sz` suffixes and their corresponding ranks of the `p_id`'th
    // partition, from its offset `off`, into `suf` and `key` respectively.
    void read_chunk(p_id_t p_id, idx_t off, idx_t sz, idx_t* suf, p_sz_t* key) const;

    // Reads into the staging slab `s` its chunk of its partition's inbox
    // messages for the current generation.
    void read_messages(Staging_Slab& s) const;

    // Writes the staging slab `s`'s chunk to the backend.
    void write(const Staging_Slab& s);

    // Writes the messages staged in slab `s` to their destination inboxes.
    void write_messages(const Staging_Slab& s);

    // Prepares the inboxes for generation `gen`.
    void open_generation(idx_t gen);

    // Flushes every destination partition's block-overflow buffer. To be
    // invoked once at a generation's ending barrier.
    void close_generation();

    // Returns the message-count in the `p_id'`th partition's inbox for the
    // current generation (written to at the preceding generation).
    p_sz_t inbox_size(p_id_t p_id) const;

    // Returns the total size of the inboxes written to at the last generation.
    idx_t inbox_mass_last_gen() const;
};

}



#endif
