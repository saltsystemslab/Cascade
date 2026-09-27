
#ifndef PARTITION_BACKEND_HPP
#define PARTITION_BACKEND_HPP



#include "types.hpp"
#include "RAM_Backend.hpp"
#include "Disk_Backend.hpp"
#include "Staging_Slab.hpp"
#include "Spill_Store.hpp"
#include "Residency_Plan.hpp"
#include "Host_Budget.hpp"
#include "Message_Pool.hpp"
#include "Spin_Lock.hpp"
#include "message.hpp"
#include "utility.hpp"

#include <cstddef>
#include <atomic>
#include <memory>
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

    Host_Budget budget_;    // Host-memory budget of the elastic RAM-tier.
    const p_sz_t msg_chunk_cap_;    // Messages per RAM message-chunk: a slab's inbox-range spans `MSG_CHUNKS_PER_SLAB` of them.
    std::unique_ptr<Message_Pool> pool_;    // Pool of the RAM message-chunks.

    p_sz_t* msg_off_;   // Storage for the exclusive-ending offsets of the destination-partitions into the messages produced by a partition.
                        // All partitions are packed flat together, so has `p x p` entries.

    // Per-destination-partition inbox state. An inbox is its RAM-part followed by
    // its disk-part: `[0, n_r)` in RAM message-chunks, the `k`'th chunk holding
    // `[k * msg_chunk_cap_, (k + 1) * msg_chunk_cap_)`, then `[n_r, n_r + n_d)`
    // in the inbox-files, at file-offsets `[0, n_d)`. With a disk-part present,
    // `n_r` is `MSG_ALIGN`-aligned, so that the disk-part's reads stay aligned.
    // {
        static constexpr p_sz_t OV_CAP = 64 * MSG_ALIGN;        // Capacity of a destination-partition's block-overflow buffer, in messages.

        std::size_t inbox_write_parity_;    // Inbox-parity this generation writes to.
        std::size_t inbox_read_parity_; // Inbox-parity this generation reads from.

        // RAM-part of a destination-partition's inbox, for a generation-parity.
        struct RAM_Inbox
        {
            util::Spin_Lock lock;   // Mutually-exclusive lock to `count` and `chunk_c`; no I/O or allocation happens under it.
            p_sz_t count = 0;   // Messages reserved in the RAM-part: `n_r`.
            p_sz_t chunk_c = 0; // Message-chunks installed.
            std::vector<Message_Pool::chunk_t> chunk;   // Message-chunks of the RAM-part; sized for the partition's capacity, and never resized.
        };

        std::vector<util::Padded<RAM_Inbox>> ram_inbox_[2]; // RAM-parts of the destination-partitions' inboxes, per generation-parity.

        std::vector<std::atomic<p_sz_t>> msg_count_[2]; // Size of each destination-partition's inbox's disk-part, including the buffered messages; per generation-parity.
        std::vector<std::atomic<p_sz_t>> msg_written_[2];   // Messages actually written to the backend, per destination; per generation-parity.
        std::vector<std::atomic<uint64_t>> msg_sum_[2]; // Order-independent checksum of each destination-partition's inbox, per generation-parity.

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
    // the `p_id`'th inbox, to the inbox's disk-part.
    void append_messages(p_id_t p_id, const Staging_Slab& s, p_sz_t l, p_sz_t r);

    // Appends a prefix of the `[l, r)` slot-range of the slab `s`'s messages,
    // destined for the `p_id`'th inbox, to the inbox's RAM-part—as long as the
    // budget admits. `spare` holds a message-chunk acquired but not installed
    // yet, if any. Returns the length of the prefix appended.
    p_sz_t append_ram(p_id_t p_id, const Staging_Slab& s, p_sz_t l, p_sz_t r, Message_Pool::chunk_t& spare);

    // Copies the `[from, from + sz)` slot-range of the slab `s`'s messages into
    // the RAM-part `in` at its offset `pos`, reserved beforehand.
    void copy_in(const RAM_Inbox& in, p_sz_t pos, const Staging_Slab& s, p_sz_t from, p_sz_t sz);

    // Returns a message-chunk: a free one still resident if any, else a fresh
    // one if the budget admits it, else `Message_Pool::NIL`.
    Message_Pool::chunk_t acquire_msg_chunk();

    // Returns the message-chunk `c` to the pool; its memory stays charged.
    void release_msg_chunk(Message_Pool::chunk_t c);

    // Returns the RAM-part `in`'s message-chunks to the pool, and empties it.
    void release_inbox(RAM_Inbox& in);


public:

    // Constructs a backend for partitions with capacities `P_cap`, staged in
    // slabs of capacity `chunk_cap`. It spills to spill-store `spill` whenever
    // the memory `budget` of the host, less its fixed allocation cost
    // `fixed_bytes` cannot hold some data.
    Partition_Backend(const std::vector<idx_t>& P_cap, idx_t chunk_cap, Spill_Store& spill, std::size_t budget, std::size_t fixed_bytes, p_id_t p);

    Partition_Backend(const Partition_Backend&) = delete;
    Partition_Backend& operator=(const Partition_Backend&) = delete;

    ~Partition_Backend();

    // Returns the fixed RAM-budget (in bytes) required by the backend for `p`
    // partitions. Currently: the block-overflow buffers for the messages.
    static std::size_t fixed_budget(p_id_t p);

    // Returns the count of partitions.
    auto p() const { return p_; }

    // Retires the inboxes of both the generation-parities. To be invoked once
    // past the last generation, as no inbox is read or written onward.
    void retire_messages();

    // Opens the `p_id`'th partition's key-file.
    void open_key(p_id_t p_id);

    // Retires the `p_id`'th partition's suffixes and keys.
    void retire_partition(p_id_t p_id);

    // Returns the (exclusive) ending-offsets of the destination-partitions into
    // the messages produced by the `p_id`'th partition.
    auto msg_off(const p_id_t p_id) const { return msg_off_ + p_id * p_; }

    // Returns the size of the `q`'th inbox at the generation last written to.
    p_sz_t msg_count(const p_id_t q) const { return ram_inbox_[inbox_write_parity_][q].unwrap().count + disk_count(q); }

    // Returns the size of the `q`'th inbox's disk-part at the generation last
    // written to.
    p_sz_t disk_count(const p_id_t q) const { return msg_count_[inbox_write_parity_][q]; }

    // Returns the actual count of message-entries written to the `q`'th inbox's
    // disk-part; this can exceed `disk_count(q)` due to disk-block alignment
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
    // messages for the current generation. The RAM message-chunks read are
    // returned to the pool, as no other slab spans them.
    void read_messages(Staging_Slab& s);

    // Writes the staging slab `s`'s chunk to the backend.
    void write(const Staging_Slab& s);

    // Writes the messages staged in slab `s` to their destination inboxes.
    void write_messages(const Staging_Slab& s);

    // Prepares the inboxes for generation `gen`, where the partitions have
    // active-set sizes `Active_sz`.
    void open_generation(idx_t gen, const std::vector<idx_t>& Active_sz);

    // Completes every destination partition's inbox written at this generation:
    // aligns its RAM-part if it has a disk-part, and flushes its block-overflow
    // buffer. To be invoked once at a generation's ending barrier.
    void close_generation();

    // Returns the message-count in the `p_id'`th partition's inbox for the
    // current generation (written to at the preceding generation).
    p_sz_t inbox_size(p_id_t p_id) const;

    // Returns the order-independent checksum of the messages in the `p_id`'th
    // partition's inbox for the current generation (written to at the preceding
    // generation).
    uint64_t inbox_checksum(const p_id_t p_id) const { return msg_sum_[inbox_read_parity_][p_id]; }

    // Returns the total size of the inboxes written to at the last generation.
    idx_t inbox_mass_last_gen() const;

    // Returns the total size of the inboxes' disk-parts written to at the last
    // generation.
    idx_t disk_mass_last_gen() const;
};

}



#endif
