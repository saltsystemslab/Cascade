
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
#include "env.hpp"

#include <cstddef>
#include <random>
#include <atomic>
#include <memory>
#include <vector>
#include <tuple>
#include <algorithm>


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

    // Per-chunk residency of the partitions' content. A partition's `c`'th chunk
    // is its range `[c * chunk_cap_, (c + 1) * chunk_cap_)`, i.e. its `c`'th
    // staging-slab's; each field of each chunk resides in its own tier.
    // {
        // Fields of a partition's content.
        enum class Field : std::size_t { suf = 0, key, field_c, };

        static constexpr std::size_t FIELD_C = static_cast<std::size_t>(Field::field_c);

        // Residency tiers of a chunk's field. A RAM-chunk past its partition's
        // active-set is a retired tail: it is never in a cohort again. An
        // evicting chunk is a retired tail on its way to the disk, claimed by
        // one evictor.
        enum class Tier : uint8_t { disk = 0, ram, ram_tail, evicting, };

        const idx_t chunk_cap_; // Elements per content-chunk: a staging-slab's capacity.
        const std::vector<idx_t> P_cap_;    // Partition capacities.
        std::vector<idx_t> chunk_base_; // Index of each partition's first chunk, in the partitions' flattened chunk-sequence; `p + 1` entries.
        std::unique_ptr<std::atomic<uint8_t>[]> tier_;  // Tier of each chunk's each field, field-major: `FIELD_C x chunk_base_[p]` entries.
        idx_t* active_slot_;    // Pinned slots of the partitions' active-set sizes past their device-runs, for their writers; `p` entries.
        std::atomic<std::size_t> content_ram_;  // Bytes of the chunks' fields resident in RAM: the content's charge on the host-memory.
        std::atomic<std::size_t> tail_ram_; // Of `content_ram_`, the bytes in retired tails.
        std::atomic<std::size_t> evict_cursor_; // Rotating start of the evictors' scans over the tier-table.
    // }

    // Kinds of claims on the host-memory budget: of a cohort's chunk-field at
    // its writeback, which may reclaim room from the pooled message-chunks and
    // the retired tails; of a chunk-field retiring at its writeback, which
    // takes free room only; and of a message-chunk, which may reclaim room from
    // the retired tails.
    enum class Claim { content, retiring, message, };

    // Whether the content's chunks change tiers within the generations:
    // admitted to RAM at their writebacks, and evicted to the disk for the
    // claims on a full budget. Set by `GPU_SACA_DYNAMIC_CONTENT`; else iff the
    // residency-plan is chunk-wise.
    const bool dynamic_;

    bool frozen_;   // Whether the claims are all refused: past the last generation, as the concatenation claims no memory.

    std::size_t fixed_; // Bytes of the fixed host-allocation still held: `fixed_bytes` at construction, less the cap's raises.
    Host_Budget budget_;    // Host-memory budget of the RAM-tiers of the content and of the messages.
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

    // Statistics for the barriers' planner-log.
    // {
        // Kinds of transitions of the content's chunk-fields: writebacks that
        // keep a tier, markings of retired tails, the tests' tier-shuffles,
        // admissions to RAM of active and of retiring chunks, and evictions.
        enum class Transition : std::size_t { ram_write = 0, disk_write, tail_mark, shuffle, admit, admit_tail, evict, transition_c, };

        static constexpr std::size_t TRANSITION_C = static_cast<std::size_t>(Transition::transition_c);

        std::atomic<uint64_t> transition_c_[TRANSITION_C];  // Transitions since the last barrier, per kind.
        mutable std::atomic<uint64_t> disk_read_;   // Bytes the pipeline's readers read from the disk since the last barrier.
        uint64_t read_prediction_;  // Bytes predicted at the last barrier for `disk_read_` at the next.
    // }

    // Chaos of the tests: the tier-shuffle, with whole-chunk moves between the
    // tiers at the barriers, which mix the tiers within partitions; and the
    // refusals of claims at random.
    // {
        inline static const double TIER_SHUFFLE = env::get_f64("GPU_SACA_TIER_SHUFFLE").value_or(0.0);  // Probability of a cohort's chunk-field moving, per barrier.
        std::mt19937_64 shuffle_rng_;   // Source of the moves; seeded by `GPU_SACA_TIER_SHUFFLE_SEED`.
        inline static const double CHARGE_REFUSAL = env::get_f64("GPU_SACA_CHARGE_REFUSAL").value_or(0.0);  // Probability of a claim's refusal.
        const uint64_t refusal_seed_;   // Seed of the refusals: `GPU_SACA_CHARGE_REFUSAL_SEED`.
        std::atomic<uint64_t> refusal_draw_;    // Count of the draws for refusals so far.
    // }


    // Returns the width, in bytes, of a content-element's field `f`.
    static constexpr std::size_t width(const Field f) { return f == Field::suf ? sizeof(idx_t) : sizeof(p_sz_t); }

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

    // Returns the index of field `f` of the `p_id`'th partition's `c`'th chunk
    // into the tier-table.
    std::size_t tier_idx(Field f, p_id_t p_id, idx_t c) const;

    // Returns the tier of field `f` of the `p_id`'th partition's `c`'th chunk.
    Tier tier(Field f, p_id_t p_id, idx_t c) const;

    // Returns whether field `f` of the `p_id`'th partition's `c`'th chunk
    // resides in RAM.
    bool in_ram(const Field f, const p_id_t p_id, const idx_t c) const { return tier(f, p_id, c) != Tier::disk; }

    // Marks the RAM-resident fields of the `p_id`'th partition's `c`'th chunk
    // as retired tails; the disk-resident ones stay as they are. To be invoked
    // once the chunk's writeback completes.
    void mark_tail(p_id_t p_id, idx_t c);

    // Returns the count of elements in the `p_id`'th partition's `c`'th chunk.
    idx_t chunk_len(p_id_t p_id, idx_t c) const;

    // Returns the bytes of field `f` of the `p_id`'th partition's `c`'th chunk.
    std::size_t chunk_bytes(Field f, p_id_t p_id, idx_t c) const;

    // Moves field `f` of the `p_id`'th partition's `c`'th chunk from RAM to the
    // disk in whole: written out, then its tier set, then its memory released.
    void move_to_disk(Field f, p_id_t p_id, idx_t c);

    // Moves field `f` of the `p_id`'th partition's `c`'th chunk from the disk to
    // RAM in whole.
    void move_to_ram(Field f, p_id_t p_id, idx_t c);

    // Returns the field, the partition, and the chunk of the tier-table's
    // `x`'th entry.
    std::tuple<Field, p_id_t, idx_t> tier_entry(std::size_t x) const;

    // Charges `b` bytes of the host-memory budget for a claim of kind `claim`.
    // On a full cap, the claim reclaims room as far as its kind permits, and
    // takes the reclaimed charge directly, so that no competing claimant takes
    // the room freed. Returns whether charged. Thread-safe.
    bool charge(std::size_t b, Claim claim);

    // Returns whether to refuse a claim at random, with probability
    // `GPU_SACA_CHARGE_REFUSAL`, for the tests. Thread-safe.
    bool refused();

    // Evicts a retired tail's chunk-field to the disk, and returns its bytes,
    // still charged; `0` if no tail is left to evict. Thread-safe.
    std::size_t evict_one();

    // Admits field `f` of the `p_id`'th partition's `c`'th chunk to RAM iff its
    // claim of kind `claim` is charged; the chunk's writeback then goes to RAM,
    // whole. Returns whether admitted.
    bool admit(Field f, p_id_t p_id, idx_t c, Claim claim);

    // Aborts unless the RAM-chunks retired as tails are exactly the ones past
    // their partitions' active-sets, of sizes `Active_sz`, the RAM-chunks'
    // charges agree with their tiers, and the budget's charges are the
    // content's and the resident message-chunks'. To be invoked between the
    // generations.
    void check_tiers(const std::vector<idx_t>& Active_sz) const;

    // Counts a transition of kind `t`.
    void count(const Transition t) { transition_c_[static_cast<std::size_t>(t)].fetch_add(1, std::memory_order_relaxed); }

    // Applies `f(c, from, len)` to each piece `[from, from + len)` of a
    // partition's range `[off, off + sz)` that lies in the partition's `c`'th
    // chunk, in the range's order. An empty range touches no chunk.
    template <typename F_>
    void for_each_chunk(idx_t off, idx_t sz, F_ f) const;

    // Returns the bytes of field `f` in the `p_id`'th partition's range
    // `[off, off + sz)` that reside in disk.
    std::size_t disk_bytes(Field f, p_id_t p_id, idx_t off, idx_t sz) const;


public:

    // Constructs a backend for partitions with capacities `P_cap`, staged in
    // slabs of capacity `chunk_cap`. It spills to spill-store `spill` whenever
    // the memory `budget` of the host, less its fixed allocation cost
    // `fixed_bytes` cannot hold some data. The content that the partitions'
    // residency-plan places in RAM is charged to the budget first, and the
    // messages' RAM-tier takes the rest.
    Partition_Backend(const std::vector<idx_t>& P_cap, idx_t chunk_cap, Spill_Store& spill, std::size_t budget, std::size_t fixed_bytes, p_id_t p);

    Partition_Backend(const Partition_Backend&) = delete;
    Partition_Backend& operator=(const Partition_Backend&) = delete;

    ~Partition_Backend();

    // Returns the fixed RAM-budget (in bytes) required by the backend for `p`
    // partitions. Currently: the block-overflow buffers for the messages.
    static std::size_t fixed_budget(p_id_t p);

    // Raises the cap of the elastic RAM-tier by `b` bytes, freed from some
    // fixed allocation.
    void raise_cap(const std::size_t b) { fixed_ -= b, budget_.raise_cap(b); }

    // Returns the host-memory budget of the elastic RAM-tier.
    const auto& budget() const { return budget_; }

    // Returns the count of partitions.
    auto p() const { return p_; }

    // Retires the inboxes of both the generation-parities, and freezes the
    // claims on the budget. To be invoked once past the last generation, as no
    // inbox is read or written onward, and no memory is claimed.
    void retire_messages();

    // Retires the `p_id`'th partition's suffixes and keys.
    void retire_partition(p_id_t p_id);

    // Returns the (exclusive) ending-offsets of the destination-partitions into
    // the messages produced by the `p_id`'th partition.
    auto msg_off(const p_id_t p_id) const { return msg_off_ + p_id * p_; }

    // Returns the slot of the `p_id`'th partition's active-set size past its
    // device-run at the current generation; its writers read it.
    idx_t* active_slot(const p_id_t p_id) { return active_slot_ + p_id; }

    // Returns the size of the `q`'th inbox at the generation last written to.
    p_sz_t msg_count(const p_id_t q) const { return ram_inbox_[inbox_write_parity_][q].unwrap().count + disk_count(q); }

    // Returns the size of the `q`'th inbox's disk-part at the generation last
    // written to.
    p_sz_t disk_count(const p_id_t q) const { return msg_count_[inbox_write_parity_][q]; }

    // Returns the actual count of message-entries written to the `q`'th inbox's
    // disk-part; this can exceed `disk_count(q)` due to disk-block alignment
    // requirements.
    p_sz_t msg_written(const p_id_t q) const { return msg_written_[inbox_write_parity_][q]; }

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

    // Writes `sz` suffixes of the `p_id`'th partition, at its offset `off`, from
    // `suf`. The disk-resident pieces round up to whole disk-blocks: `suf` must
    // have that capacity, and the write must end at a disk-block or at the
    // partition's end. Thread-safe for disjoint ranges.
    void write_suffixes(p_id_t p_id, idx_t off, idx_t sz, const idx_t* suf);

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

    // Logs the backend's state at the barrier opening generation `gen`, where
    // the partitions have active-set sizes `Active_sz`: the memory, the content
    // and the inboxes by tier, the transitions since the last barrier, and the
    // generation's disk-reads—predicted here, and checked against the actual at
    // the next barrier.
    void log_barrier(idx_t gen, const std::vector<idx_t>& Active_sz);

    // Moves each field of each chunk in the cohorts to its other tier, with
    // probability `GPU_SACA_TIER_SHUFFLE`, at the barrier opening generation
    // `gen`, where the partitions have active-set sizes `Active_sz`. Seeded and
    // single-threaded; only the suffixes move before the first generation, as
    // the keys hold no content yet. The retired tails are evicted likewise,
    // through the evictor. For tests purposes only.
    void shuffle_tiers(idx_t gen, const std::vector<idx_t>& Active_sz);

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


inline std::size_t Partition_Backend::tier_idx(const Field f, const p_id_t p_id, const idx_t c) const
{
    return static_cast<std::size_t>(f) * chunk_base_[p_] + chunk_base_[p_id] + c;
}


inline idx_t Partition_Backend::chunk_len(const p_id_t p_id, const idx_t c) const
{
    return std::min(chunk_cap_, P_cap_[p_id] - c * chunk_cap_);
}


inline std::size_t Partition_Backend::chunk_bytes(const Field f, const p_id_t p_id, const idx_t c) const
{
    return chunk_len(p_id, c) * width(f);
}


inline Partition_Backend::Tier Partition_Backend::tier(const Field f, const p_id_t p_id, const idx_t c) const
{
    return static_cast<Tier>(tier_[tier_idx(f, p_id, c)].load(std::memory_order_relaxed));
}

}



#endif
