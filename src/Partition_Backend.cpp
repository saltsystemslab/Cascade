
#include "Partition_Backend.hpp"
#include "utility.hpp"
#include "utility.cuh"
#include "Pipeline_Probe.hpp"
#include "tests.hpp"

#include <cstring>
#include <mutex>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Partition_Backend::Partition_Backend(const std::vector<idx_t>& P_cap, const idx_t chunk_cap, Spill_Store& spill, const std::size_t budget, const std::size_t fixed_bytes, const p_id_t p):
      p_(p)
    , plan_(P_cap, chunk_cap, budget, fixed_bytes)
    , ram_(P_cap)
    , disk_(P_cap, spill, plan_)
    , chunk_cap_(chunk_cap)
    , P_cap_(P_cap)
    , chunk_base_(p_ + 1, 0)
    , active_slot_(nullptr)
    , content_ram_(plan_.data_resident())
    , tail_ram_(0)
    , evict_cursor_(0)
    , dynamic_(env::get_u64("GPU_SACA_DYNAMIC_CONTENT").value_or(plan_.chunk_wise()) != 0)
    , frozen_(false)
    , fixed_(fixed_bytes)
    , budget_(budget - fixed_bytes, plan_.force_spill())
    , msg_chunk_cap_(chunk_cap / MSG_CHUNKS_PER_SLAB)
    , pool_(std::make_unique<Message_Pool>(msg_chunk_cap_, budget_.pinned() ? 0 : budget))
    , msg_off_(nullptr)
    , inbox_write_parity_(write_parity(0))
    , inbox_read_parity_(read_parity(0))
    , ov_tgt_(p_, nullptr)
    , ov_rank_(p_, nullptr)
    , ov_sz_(p_, 0)
    , ov_lock_(p_)
    , transition_c_{}
    , disk_read_(0)
    , read_prediction_(0)
    , shuffle_rng_(env::get_u64("GPU_SACA_TIER_SHUFFLE_SEED").value_or(1))
    , refusal_seed_(env::get_u64("GPU_SACA_CHARGE_REFUSAL_SEED").value_or(1))
    , refusal_draw_(0)
{
    CUDA_CHECK(pinned_allocate_host(msg_off_, p_ * p_));
    CUDA_CHECK(pinned_allocate_host(active_slot_, p_));

    // The chunks' tiers follow the residency-plan: per field, a leading run of each partition's chunks starts in RAM.
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        chunk_base_[p_id + 1] = chunk_base_[p_id] + plan_.chunk_c(p_id);

    tier_ = std::make_unique<std::atomic<uint8_t>[]>(FIELD_C * chunk_base_[p_]);
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        for(idx_t c = 0; c < plan_.chunk_c(p_id); ++c)
            tier_[tier_idx(Field::suf, p_id, c)].store(static_cast<uint8_t>(c < plan_.suf_ram(p_id) ? Tier::ram : Tier::disk), std::memory_order_relaxed),
            tier_[tier_idx(Field::key, p_id, c)].store(static_cast<uint8_t>(c < plan_.key_ram(p_id) ? Tier::ram : Tier::disk), std::memory_order_relaxed);

    std::copy(P_cap.cbegin(), P_cap.cend(), active_slot_);  // No chunk is a retired tail before the first generation.

    // The content that the residency-plan places in RAM is charged to the budget first; the messages get the rest.
    if(!budget_.try_charge(plan_.data_resident()))
    {
        std::cerr << "The residency-plan's " << plan_.data_resident() << " bytes of RAM-content exceed the host-memory budget's cap of " << budget_.cap() << " bytes. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    for(idx_t parity = 0; parity < 2; ++parity)
    {
        ram_inbox_[parity] = std::vector<util::Padded<RAM_Inbox>>(p_);
        for(p_id_t p_id = 0; p_id < p_; ++p_id) // An inbox holds at most one message per suffix of its partition.
            ram_inbox_[parity][p_id].unwrap().chunk.assign((P_cap[p_id] + msg_chunk_cap_ - 1) / msg_chunk_cap_, Message_Pool::NIL);

        msg_count_[parity] = std::vector<std::atomic<p_sz_t>>(p_),
        msg_written_[parity] = std::vector<std::atomic<p_sz_t>>(p_),
        msg_sum_[parity] = std::vector<std::atomic<uint64_t>>(p_);
    }

    for(idx_t p_id = 0; p_id < p_; ++p_id)
        ov_tgt_[p_id] = util::aligned_allocate<idx_t>(OV_CAP, Direct_File::DISK_BLOCK),
        ov_rank_[p_id] = util::aligned_allocate<msg_rank_t>(OV_CAP, Direct_File::DISK_BLOCK);
}


Partition_Backend::~Partition_Backend()
{
    CUDA_CHECK(deallocate_host(msg_off_));
    CUDA_CHECK(deallocate_host(active_slot_));
    for(idx_t p_id = 0; p_id < p_; ++p_id)
        util::deallocate(ov_tgt_[p_id]),
        util::deallocate(ov_rank_[p_id]);
}


std::size_t Partition_Backend::fixed_budget(const p_id_t p)
{
    return p * OV_CAP * (sizeof(idx_t) + sizeof(msg_rank_t));
}


void Partition_Backend::retire_messages()
{
    frozen_ = true; // No memory is claimed onward either, so nothing is evicted, nor admitted.

    // No inbox is read or written onward: return the RAM-parts' memory, and the disk-parts' scratch-space.
    for(std::size_t parity = 0; parity < 2; ++parity)
        for(auto& in : ram_inbox_[parity])
            release_inbox(in.unwrap());

    const auto pool_bytes = std::size_t(pool_->carved() - pool_->released()) * pool_->chunk_bytes();    // All free, and charged but the released.
    budget_.unpool(pool_bytes), budget_.uncharge(pool_bytes);
    pool_.reset();

    disk_.retire_inboxes();
}


void Partition_Backend::retire_chunk(const p_id_t p_id, const idx_t c)
{
#ifdef TESTS
    if(!frozen_)
    {
        std::cerr << "Chunk " << c << " of partition " << p_id << " is retired before the claims freeze. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }
#endif

    // Its RAM-fields' memory goes, and their charges return. Their tiers turn to the disk first, so that no RAM-tier stands over
    // released memory, and a repeated retirement finds none; the chunk being dead onward, the disk need not hold it.
    const auto off = c * chunk_cap_, len = chunk_len(p_id, c);
    std::size_t ram = 0, tail = 0;
    for(std::size_t f = 0; f < FIELD_C; ++f)
    {
        const auto field = static_cast<Field>(f);
        const auto t = tier(field, p_id, c);
        if(t == Tier::disk)
            continue;

        tier_[tier_idx(field, p_id, c)].store(static_cast<uint8_t>(Tier::disk), std::memory_order_relaxed);
        field == Field::suf ?   ram_.release_suf(p_id, off, len) :
                                ram_.release_key(p_id, off, len);
        ram += chunk_bytes(field, p_id, c), tail += (t == Tier::ram_tail ? chunk_bytes(field, p_id, c) : 0);
    }

    content_ram_.fetch_sub(ram, std::memory_order_relaxed), tail_ram_.fetch_sub(tail, std::memory_order_relaxed);
    budget_.uncharge(ram);
}


void Partition_Backend::retire_partition(const p_id_t p_id)
{
    for(idx_t c = 0; c < chunk_base_[p_id + 1] - chunk_base_[p_id]; ++c)
        retire_chunk(p_id, c);

    disk_.retire_data(p_id);
}


template <typename F_>
void Partition_Backend::for_each_chunk(const idx_t off, const idx_t sz, F_ f) const
{
    for(idx_t from = off; from < off + sz;)
    {
        const auto c = from / chunk_cap_;
        const auto to = std::min(off + sz, (c + 1) * chunk_cap_);
        f(c, from, to - from);
        from = to;
    }
}


std::size_t Partition_Backend::disk_bytes(const Field f, const p_id_t p_id, const idx_t off, const idx_t sz) const
{
    std::size_t bytes = 0;
    for_each_chunk(off, sz, [&](const idx_t c, idx_t, const idx_t len){ bytes += (tier(f, p_id, c) == Tier::disk ? len * width(f) : 0); });

    return bytes;
}


void Partition_Backend::read_suffixes(Staging_Slab& s) const
{
    INSTRUMENT_TIMER(probe::P().me().io_ns);
    INSTRUMENT(probe::P().me().bytes += s.size() * sizeof(idx_t));
    INSTRUMENT(probe::P().me().disk_bytes += disk_bytes(Field::suf, s.p_id(), s.off(), s.size()));
#ifdef STATS
    disk_read_.fetch_add(disk_bytes(Field::suf, s.p_id(), s.off(), s.size()), std::memory_order_relaxed);
#endif

    read_suf_chunk(s.p_id(), s.off(), s.size(), s.suf());
}


void Partition_Backend::read_keys(Staging_Slab& s) const
{
    auto* const suf = s.suf();
    auto* const key = reinterpret_cast<p_sz_t*>(s.key());

    {
        INSTRUMENT_TIMER(probe::P().me().io_ns);
        INSTRUMENT(probe::P().me().bytes += s.size() * (sizeof(idx_t) + sizeof(p_sz_t)));
        INSTRUMENT(probe::P().me().disk_bytes += disk_bytes(Field::suf, s.p_id(), s.off(), s.size()) + disk_bytes(Field::key, s.p_id(), s.off(), s.size()));
#ifdef STATS
        disk_read_.fetch_add(disk_bytes(Field::suf, s.p_id(), s.off(), s.size()) + disk_bytes(Field::key, s.p_id(), s.off(), s.size()), std::memory_order_relaxed);
#endif
        read_chunk(s.p_id(), s.off(), s.size(), suf, key);
    }
}


void Partition_Backend::read_suf_chunk(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const suf) const
{
    // The pieces go in the range's order, so that a disk-piece's block-rounding spills only over the pieces yet to be read.
    for_each_chunk(off, sz,
        [&](const idx_t c, const idx_t from, const idx_t len)
        {
            in_ram(Field::suf, p_id, c) ?   ram_.read_suf(p_id, from, len, suf + (from - off)) :
                                            disk_.read_suf(p_id, from, len, suf + (from - off));
        });
}


void Partition_Backend::read_chunk(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const suf, p_sz_t* const key) const
{
    // The pieces go in the range's order, so that a disk-piece's block-rounding spills only over the pieces yet to be read.
    for_each_chunk(off, sz,
        [&](const idx_t c, const idx_t from, const idx_t len)
        {
            in_ram(Field::suf, p_id, c) ?   ram_.read_suf(p_id, from, len, suf + (from - off)) :
                                            disk_.read_suf(p_id, from, len, suf + (from - off));
            in_ram(Field::key, p_id, c) ?   ram_.read_key(p_id, from, len, key + (from - off)) :
                                            disk_.read_key(p_id, from, len, key + (from - off));
        });
}


void Partition_Backend::read_messages(Staging_Slab& s)
{
    INSTRUMENT_TIMER(probe::P().me().io_ns);
    INSTRUMENT(probe::P().me().bytes += s.inbox_sz() * MESSAGE_BYTES);

    const auto p_id = s.p_id();
    auto& in = ram_inbox_[inbox_read_parity_][p_id].unwrap();
    const p_sz_t lo = s.inbox_off(), hi = lo + s.inbox_sz(), n_r = in.count; // This slab holds the inbox's range `[lo, hi)`.

    // The RAM-portion `[lo, min(hi, n_r))`: whole message-chunks, as `lo` is a multiple of the slab-capacity; only this slab spans them.
    for(p_sz_t from = lo; from < std::min(hi, n_r); from += msg_chunk_cap_)
    {
        const auto c = in.chunk[from / msg_chunk_cap_];
        const auto len = std::min<p_sz_t>(std::min(hi, n_r) - from, msg_chunk_cap_);
        std::memcpy(s.msg_target() + (from - lo), pool_->tgt(c), len * sizeof(idx_t));
        std::memcpy(s.msg_rank() + (from - lo), pool_->rank(c), len * sizeof(msg_rank_t));

        release_msg_chunk(c), in.chunk[from / msg_chunk_cap_] = Message_Pool::NIL;
    }

    // The disk-portion `[max(lo, n_r), hi)`. Its file-offset `from - n_r` and slab-offset `from - lo` are both `MSG_ALIGN`-aligned, as
    // is `lo`, and as is `n_r` whenever the disk-part exists. The disk-read's block-rounding spills past `hi`, never over the RAM-portion.
    if(hi > n_r)
    {
        const auto from = std::max(lo, n_r);
        INSTRUMENT(probe::P().me().disk_bytes += (hi - from) * MESSAGE_BYTES);
#ifdef STATS
        disk_read_.fetch_add((hi - from) * MESSAGE_BYTES, std::memory_order_relaxed);
#endif
        disk_.read_messages(p_id, from - n_r, hi - from, s.msg_target() + (from - lo), s.msg_rank() + (from - lo));
    }
}


void Partition_Backend::write(const Staging_Slab& s)
{
    INSTRUMENT_TIMER(probe::P().me().io_ns);
    INSTRUMENT(probe::P().me().bytes += s.size() * (sizeof(idx_t) + sizeof(p_sz_t)));

    const auto p_id = s.p_id();
    const auto off = s.off();
    const auto a_new = active_slot_[p_id];  // The partition's active-set size past this generation.
    const auto* const suf = s.suf();
    const auto* const key = reinterpret_cast<const p_sz_t*>(s.key());
    for_each_chunk(off, s.size(),
        [&](const idx_t c, const idx_t from, const idx_t len)
        {
#ifdef TESTS
            if(tier(Field::suf, p_id, c) == Tier::ram_tail || tier(Field::key, p_id, c) == Tier::ram_tail)
            {
                std::cerr << "Chunk " << c << " of partition " << p_id << " is a retired tail, yet in the cohort. Aborting.\n";
                std::exit(EXIT_FAILURE);
            }
#endif
            // A disk-chunk written back whole may move to RAM: an active one reclaiming room for itself, and a retiring one
            // taking free room only.
            const bool retiring = (c * chunk_cap_ >= a_new);    // The chunk lies past the partition's new active-set.
            const bool whole = (len == chunk_len(p_id, c));
            const auto claim = (retiring ? Claim::retiring : Claim::content);
            const bool suf_ram = in_ram(Field::suf, p_id, c) || (whole && admit(Field::suf, p_id, c, claim));
            const bool key_ram = in_ram(Field::key, p_id, c) || (whole && admit(Field::key, p_id, c, claim));
            INSTRUMENT(probe::P().me().disk_bytes += (suf_ram ? 0 : len * sizeof(idx_t)) + (key_ram ? 0 : len * sizeof(p_sz_t)));

            suf_ram ?   ram_.write_suf(p_id, from, len, suf + (from - off)) :
                        disk_.write_suf(p_id, from, len, suf + (from - off));
            key_ram ?   ram_.write_key(p_id, from, len, key + (from - off)) :
                        disk_.write_key(p_id, from, len, key + (from - off));
#ifdef STATS
            count(suf_ram ? Transition::ram_write : Transition::disk_write);
            count(key_ram ? Transition::ram_write : Transition::disk_write);
#endif

            if(retiring)
                mark_tail(p_id, c);
        });
}


void Partition_Backend::check_tiers(const std::vector<idx_t>& Active_sz) const
{
    std::size_t ram = 0, tail = 0;
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        for(idx_t c = 0; c < chunk_base_[p_id + 1] - chunk_base_[p_id]; ++c)
            for(std::size_t f = 0; f < FIELD_C; ++f)
            {
                const auto t = tier(static_cast<Field>(f), p_id, c);
                if(t == Tier::evicting)
                {
                    std::cerr << "Chunk " << c << " of partition " << p_id << " is mid-eviction at a barrier. Aborting.\n";
                    std::exit(EXIT_FAILURE);
                }

                if(t != Tier::disk && (t == Tier::ram_tail) != (c * chunk_cap_ >= Active_sz[p_id]))
                {
                    std::cerr << "Chunk " << c << " of partition " << p_id << " is " << (t == Tier::ram_tail ? "" : "not ") << "a retired tail, at active-set size " << Active_sz[p_id] << ". Aborting.\n";
                    std::exit(EXIT_FAILURE);
                }

                ram += (t != Tier::disk ? chunk_bytes(static_cast<Field>(f), p_id, c) : 0);
                tail += (t == Tier::ram_tail ? chunk_bytes(static_cast<Field>(f), p_id, c) : 0);
            }

    if(ram != content_ram_ || tail != tail_ram_)
    {
        std::cerr << "The content's RAM-chunks are charged " << content_ram_ << " bytes, " << tail_ram_ << " of them in retired tails; their tiers hold " << ram << " and " << tail << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    // Between the generations, the budget's charges are the content's and the resident message-chunks'.
    const std::size_t msg = (pool_ ? std::size_t(pool_->carved() - pool_->released()) * pool_->chunk_bytes() : 0);
    if(budget_.used() != content_ram_ + msg)
    {
        std::cerr << "The host-memory budget has " << budget_.used() << " bytes charged, against " << content_ram_ << " of content and " << msg << " of message-chunks. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }
}


void Partition_Backend::mark_tail(const p_id_t p_id, const idx_t c)
{
    for(std::size_t f = 0; f < FIELD_C; ++f)
    {
        auto& t = tier_[tier_idx(static_cast<Field>(f), p_id, c)];
        if(t.load(std::memory_order_relaxed) == static_cast<uint8_t>(Tier::ram))
        {
            t.store(static_cast<uint8_t>(Tier::ram_tail), std::memory_order_release);  // Publishes the chunk's completed writeback.
            tail_ram_.fetch_add(chunk_bytes(static_cast<Field>(f), p_id, c), std::memory_order_relaxed);
#ifdef STATS
            count(Transition::tail_mark);
#endif
        }
    }
}


void Partition_Backend::move_to_disk(const Field f, const p_id_t p_id, const idx_t c)
{
    // The whole chunk goes, so its block-rounding spills only past the partition's end, if at all.
    // The tier turns before the release, so that no RAM-tier stands over released memory.
    const auto off = c * chunk_cap_, len = chunk_len(p_id, c);
    f == Field::suf ?   disk_.write_suf(p_id, off, len, ram_.suf(p_id) + off) :
                        disk_.write_key(p_id, off, len, ram_.key(p_id) + off);
    tier_[tier_idx(f, p_id, c)].store(static_cast<uint8_t>(Tier::disk), std::memory_order_release);
    f == Field::suf ?   ram_.release_suf(p_id, off, len) :
                        ram_.release_key(p_id, off, len);

    content_ram_.fetch_sub(chunk_bytes(f, p_id, c), std::memory_order_relaxed);
}


void Partition_Backend::move_to_ram(const Field f, const p_id_t p_id, const idx_t c)
{
    const auto off = c * chunk_cap_, len = chunk_len(p_id, c);
    f == Field::suf ?   disk_.read_suf(p_id, off, len, ram_.suf(p_id) + off) :
                        disk_.read_key(p_id, off, len, ram_.key(p_id) + off);
    tier_[tier_idx(f, p_id, c)].store(static_cast<uint8_t>(Tier::ram), std::memory_order_release);

    content_ram_.fetch_add(chunk_bytes(f, p_id, c), std::memory_order_relaxed);
}


std::tuple<Partition_Backend::Field, p_id_t, idx_t> Partition_Backend::tier_entry(const std::size_t x) const
{
    const auto flat = static_cast<idx_t>(x % chunk_base_[p_]);  // The chunk's index in the partitions' flattened chunk-sequence.
    const auto p_id = static_cast<p_id_t>(std::upper_bound(chunk_base_.cbegin(), chunk_base_.cend(), flat) - chunk_base_.cbegin() - 1);
    return {static_cast<Field>(x / chunk_base_[p_]), p_id, flat - chunk_base_[p_id]};
}


bool Partition_Backend::charge(const std::size_t b, const Claim claim)
{
#ifdef TESTS
    if(refused())
        return false;
#endif
    if(frozen_)
        return false;

    if(!dynamic_)   // The content's tiers stay as planned, and the messages take free room only.
        return claim == Claim::message && budget_.try_charge(b);

    if(budget_.try_charge(b))
        return true;

    if(claim == Claim::retiring)
        return false;

    // Reclaim: the pooled message-chunks' memory goes to the kernel for content, and then the retired tails go to the disk;
    // their charges stay, for this claim.
    std::size_t got = 0;
    if(claim == Claim::content)
        got = pool_->release_free(b), budget_.unpool(got);

    while(got < b)
        if(const auto e = evict_one(); e > 0)
            got += e;
        else
            break;

    budget_.uncharge(got >= b ? got - b : got); // The claim keeps what it needs; the rest frees up.
    return got >= b;
}


bool Partition_Backend::refused()
{
    if(CHARGE_REFUSAL <= 0.0)
        return false;

    const auto mix = [](uint64_t z) // SplitMix64's finalizer, over the draws' sequence.
    {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };

    const auto x = mix(refusal_seed_ + refusal_draw_.fetch_add(1, std::memory_order_relaxed) * 0x9E3779B97F4A7C15ull);
    return static_cast<double>(x >> 11) * 0x1.0p-53 < CHARGE_REFUSAL;   // A uniform draw from `[0, 1)`.
}


std::size_t Partition_Backend::evict_one()
{
    if(tail_ram_.load(std::memory_order_relaxed) == 0)  // No tail to scan for.
        return 0;

    const std::size_t entry_c = FIELD_C * chunk_base_[p_];
    for(std::size_t i = 0; i < entry_c; ++i)
    {
        const auto x = evict_cursor_.fetch_add(1, std::memory_order_relaxed) % entry_c;
        auto t = static_cast<uint8_t>(Tier::ram_tail);
        if(tier_[x].load(std::memory_order_relaxed) != t ||
            !tier_[x].compare_exchange_strong(t, static_cast<uint8_t>(Tier::evicting), std::memory_order_acquire))  // One evictor claims the tail, and sees its writeback.
            continue;

        const auto [f, p_id, c] = tier_entry(x);
        const auto bytes = chunk_bytes(f, p_id, c);
        move_to_disk(f, p_id, c);
        tail_ram_.fetch_sub(bytes, std::memory_order_relaxed);
#ifdef STATS
        count(Transition::evict);
#endif
        return bytes;
    }

    return 0;
}


bool Partition_Backend::admit(const Field f, const p_id_t p_id, const idx_t c, const Claim claim)
{
    if(!charge(chunk_bytes(f, p_id, c), claim))
        return false;

    // The writeback following fills the chunk whole, before any read of it: the chunk's one reader of the generation has read it.
    tier_[tier_idx(f, p_id, c)].store(static_cast<uint8_t>(Tier::ram), std::memory_order_relaxed);
    content_ram_.fetch_add(chunk_bytes(f, p_id, c), std::memory_order_relaxed);
#ifdef STATS
    count(claim == Claim::retiring ? Transition::admit_tail : Transition::admit);
#endif

    return true;
}


void Partition_Backend::write_suffixes(const p_id_t p_id, const idx_t off, const idx_t sz, const idx_t* const suf)
{
    for_each_chunk(off, sz,
        [&](const idx_t c, const idx_t from, const idx_t len)
        {
            in_ram(Field::suf, p_id, c) ?   ram_.write_suf(p_id, from, len, suf + (from - off)) :
                                            disk_.write_suf(p_id, from, len, suf + (from - off));
        });
}


void Partition_Backend::write_messages(const Staging_Slab& s)
{
    INSTRUMENT_TIMER(probe::P().me().io_ns);
    INSTRUMENT(probe::P().me().bytes += s.outbox_sz() * MESSAGE_BYTES);

    const auto p_id = s.p_id();
    const auto* const msg_off = this->msg_off(p_id);

    // This slab contains the slice `[l_slab, r_slab)` of messages from `p_id`.
    const p_sz_t l_slab = s.outbox_off(), r_slab = std::min<idx_t>(s.outbox_off() + s.outbox_sz(), msg_off[p_ - 1]);
    if(l_slab >= r_slab)
        return;

    // Without a RAM-tier, all messages go to the inbox-files, in destination-order. Otherwise they go to the RAM-parts first, visiting
    // the destinations from a rotating start, so that the concurrent writers do not contend over them in lockstep.
    const bool to_ram = (budget_.cap() > 0);
    auto spare = Message_Pool::NIL; // Message-chunk acquired for this slab's messages, but not installed yet.
    for(p_id_t i = 0; i < p_; ++i)
    {
        const p_id_t q = (to_ram ? (p_id + i) % p_ : i);

        // `q` is the destination for the slice `[l_q, r_q)` of messages from `p_id`.
        const auto l_q = (q > 0 ? msg_off[q - 1] : 0), r_q = msg_off[q];
        const auto l = std::max(l_slab, l_q), r = std::min(r_slab, r_q); // Intersect the slab's slice with the `q`-destined slice.
        if(l < r)
        {
#ifdef TESTS
            msg_sum_[inbox_write_parity_][q].fetch_add(test::message_checksum(s.msg_target() + (l - l_slab), s.msg_rank() + (l - l_slab), r - l), std::memory_order_relaxed);
#endif
            const p_sz_t in_ram = (to_ram ? append_ram(q, s, l, r, spare) : 0);  // Prefix of the slice appended to the RAM-part.
            if(l + in_ram < r)
                append_messages(q, s, l + in_ram, r);
        }
    }

    if(spare != Message_Pool::NIL)
        release_msg_chunk(spare);
}


p_sz_t Partition_Backend::append_ram(const p_id_t p_id, const Staging_Slab& s, const p_sz_t l, const p_sz_t r, Message_Pool::chunk_t& spare)
{
    auto& in = ram_inbox_[inbox_write_parity_][p_id].unwrap();
    p_sz_t done = 0;    // Length of the prefix of `[l, r)` appended so far.
    while(l + done < r)
    {
        p_sz_t pos, take;   // Reserve the RAM-part's range `[pos, pos + take)` for the next messages.
        {
            const std::lock_guard<util::Spin_Lock> lock(in.lock);
            if(in.count == in.chunk_c * msg_chunk_cap_ && spare != Message_Pool::NIL && in.chunk_c < in.chunk.size())    // The open chunk is full: install the spare.
                in.chunk[in.chunk_c++] = spare, spare = Message_Pool::NIL;

            pos = in.count;
            take = std::min<p_sz_t>(r - (l + done), in.chunk_c * msg_chunk_cap_ - in.count);
            in.count += take;
        }

        if(take == 0)   // No room: acquire a spare message-chunk outside the lock, or leave the rest to the disk-part.
        {
            if(spare != Message_Pool::NIL)  // All the chunk-slots are installed and full; impossible by the injectivity of the messages.
            {
#ifdef TESTS
                std::cerr << "RAM-part of inbox " << p_id << " overflows its partition's capacity. Aborting.\n";
                std::exit(EXIT_FAILURE);
#endif
                break;
            }

            if((spare = acquire_msg_chunk()) == Message_Pool::NIL)  // The budget is exhausted.
                break;

            continue;
        }

        copy_in(in, pos, s, l + done, take);
        done += take;
    }

    return done;
}


void Partition_Backend::copy_in(const RAM_Inbox& in, p_sz_t pos, const Staging_Slab& s, p_sz_t from, p_sz_t sz)
{
    // The reserved range's chunks were installed under the lock that reserved it, hence visible here.
    const auto base = s.outbox_off();   // `s` contains the index-range `[base, base + s.outbox_sz())` of the source-partition's outbox.
    while(sz > 0)
    {
        const auto c = in.chunk[pos / msg_chunk_cap_];
        const p_sz_t o = pos % msg_chunk_cap_;
        const auto len = std::min<p_sz_t>(sz, msg_chunk_cap_ - o);
        std::memcpy(pool_->tgt(c) + o, s.msg_target() + (from - base), len * sizeof(idx_t));
        std::memcpy(pool_->rank(c) + o, s.msg_rank() + (from - base), len * sizeof(msg_rank_t));

        pos += len, from += len, sz -= len;
    }
}


Message_Pool::chunk_t Partition_Backend::acquire_msg_chunk()
{
    if(const auto c = pool_->take_free(); c != Message_Pool::NIL)  // Free chunks are resident, and charged already.
        return budget_.unpool(pool_->chunk_bytes()), c;

    if(!charge(pool_->chunk_bytes(), Claim::message))
        return Message_Pool::NIL;

    const auto c = pool_->take_new();
    if(c == Message_Pool::NIL)  // The pool's reservation is exhausted; impossible while it covers the cap.
        budget_.uncharge(pool_->chunk_bytes());

    return c;
}


void Partition_Backend::release_msg_chunk(const Message_Pool::chunk_t c)
{
    pool_->give_back(c);
    budget_.pool(pool_->chunk_bytes());
}


void Partition_Backend::release_inbox(RAM_Inbox& in)
{
    for(p_sz_t k = 0; k < in.chunk_c; ++k)
        if(in.chunk[k] != Message_Pool::NIL)
            release_msg_chunk(in.chunk[k]), in.chunk[k] = Message_Pool::NIL;

    in.count = 0, in.chunk_c = 0;
}


void Partition_Backend::append_messages(const p_id_t p_id, const Staging_Slab& s, const p_sz_t l, const p_sz_t r)
{
    INSTRUMENT(probe::P().me().disk_bytes += (r - l) * MESSAGE_BYTES);
    disk_.ensure_inbox(p_id);   // The only entry for messages to the inbox-files, the degenerate buffering-only case included.

    const auto parity = inbox_write_parity_;
    msg_count_[parity][p_id].fetch_add(r - l, std::memory_order_relaxed);

    const auto base = s.outbox_off();   // `s` contains the index-range `[base, base + s.outbox_sz())` of the source-partition's outbox.
    const p_sz_t a_l = ((l + MSG_ALIGN - 1) / MSG_ALIGN) * MSG_ALIGN;   // Leftmost `MSG_ALIGN`-ed endpoint of `[l, r)`.
    const p_sz_t a_r = (r / MSG_ALIGN) * MSG_ALIGN; // Rightmost `MSG_ALIGN`-ed (exclusive) endpoint of `[l, r)`.

    if(a_l < a_r)
    {
        const auto to_write = a_r - a_l;
        const auto write_off = msg_written_[parity][p_id].fetch_add(to_write, std::memory_order_relaxed);
        disk_.ensure_inbox_room(p_id, write_off + to_write);
        disk_.msg_tgt(p_id).write_at(s.msg_target() + (a_l - base), to_write * sizeof(idx_t), write_off * sizeof(idx_t));
        disk_.msg_rank(p_id).write_at(s.msg_rank() + (a_l - base), to_write * sizeof(msg_rank_t), write_off * sizeof(msg_rank_t));
    }
    else    // Degenerate case of too short message-block.
    {
        buffer_overhang(p_id, s, l, r);
        return;
    }

    if(l < a_l) // Buffer the overhang at the head of `[l, r)`.
        buffer_overhang(p_id, s, l, a_l);

    if(a_r < r) // Buffer the overhang at the tail of `[l, r)`.
        buffer_overhang(p_id, s, a_r, r);
}


void Partition_Backend::buffer_overhang(const p_id_t p_id, const Staging_Slab& s, const p_sz_t l, const p_sz_t r)
{
    const auto base = s.outbox_off();   // `s` contains the index-range `[base, base + s.outbox_sz())` of the source-partition's outbox.
    const std::lock_guard<util::Spin_Lock> lock(ov_lock_[p_id]);    // Lock the inbox's block-overflow buffer.

    std::memcpy(ov_tgt_[p_id] + ov_sz_[p_id], s.msg_target() + (l - base), (r - l) * sizeof(idx_t));
    std::memcpy(ov_rank_[p_id] + ov_sz_[p_id], s.msg_rank() + (l - base), (r - l) * sizeof(msg_rank_t));

    ov_sz_[p_id] += (r - l);
    if(ov_sz_[p_id] + 2 * MSG_ALIGN > OV_CAP)   // Whether the next buffer-addition may overflow or not.
        flush_overflow(p_id, false);
}


void Partition_Backend::flush_overflow(const p_id_t p_id, const bool final)
{
    const auto to_write = (!final ?
                            (ov_sz_[p_id] / MSG_ALIGN) * MSG_ALIGN :
                            ((ov_sz_[p_id] + MSG_ALIGN - 1) / MSG_ALIGN) * MSG_ALIGN);
    if(to_write == 0)
        return;

    const auto parity = inbox_write_parity_;
    const auto write_off = msg_written_[parity][p_id].fetch_add(to_write, std::memory_order_relaxed);
    disk_.ensure_inbox_room(p_id, write_off + to_write);
    disk_.msg_tgt(p_id).write_at(ov_tgt_[p_id], to_write * sizeof(idx_t), write_off * sizeof(idx_t));
    disk_.msg_rank(p_id).write_at(ov_rank_[p_id], to_write * sizeof(msg_rank_t), write_off * sizeof(msg_rank_t));

    const auto rem = (to_write < ov_sz_[p_id] ? ov_sz_[p_id] - to_write : 0);
    if(rem > 0)
    {
        std::memmove(ov_tgt_[p_id], ov_tgt_[p_id] + to_write, rem * sizeof(idx_t));
        std::memmove(ov_rank_[p_id], ov_rank_[p_id] + to_write, rem * sizeof(msg_rank_t));
    }

    ov_sz_[p_id] = rem;
}


void Partition_Backend::open_generation(const idx_t gen, const std::vector<idx_t>& Active_sz)
{
    inbox_write_parity_ = write_parity(gen), inbox_read_parity_ = read_parity(gen);
    disk_.set_generation(gen);

#ifdef TESTS
    check_tiers(Active_sz);
#endif

    const auto parity = write_parity(gen);
    std::memset(msg_off_, 0, p_ * p_* sizeof(p_sz_t));
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
    {
        // This parity was read at the preceding generation: consumed whole by its readers, or released at the generation's start.
        auto& in = ram_inbox_[parity][p_id].unwrap();
#ifdef TESTS
        if(std::any_of(in.chunk.cbegin(), in.chunk.cbegin() + in.chunk_c, [](const auto c){ return c != Message_Pool::NIL; }))
        {
            std::cerr << "RAM-part of inbox " << p_id << " was left unread at the preceding generation. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
#endif
        release_inbox(in);
        msg_count_[parity][p_id] = 0, msg_written_[parity][p_id] = 0, msg_sum_[parity][p_id] = 0;
        ov_sz_[p_id] = 0;

        if(Active_sz[p_id] == 0)    // The partition is not processed at this generation, so its inbox is never read.
            release_inbox(ram_inbox_[inbox_read_parity_][p_id].unwrap());
    }
}


void Partition_Backend::close_generation()
{
    const auto parity = inbox_write_parity_;
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
    {
        auto& in = ram_inbox_[parity][p_id].unwrap();
        const std::lock_guard<util::Spin_Lock> lock(ov_lock_[p_id]);

        // With a disk-part present, the RAM-part's trailing `count mod MSG_ALIGN` messages move to the disk-part, through its block-
        // overflow buffer, so that the disk-part's reads stay aligned. They all lie in the last chunk, as chunks hold whole `MSG_ALIGN`-
        // groups; and the buffer has room, as it keeps `2 * MSG_ALIGN` free past any addition.
        const p_sz_t trail = in.count % MSG_ALIGN;
        if(msg_count_[parity][p_id] > 0 && trail > 0)
        {
            const auto from = in.count - trail;
            const auto c = in.chunk[from / msg_chunk_cap_];
            const p_sz_t o = from % msg_chunk_cap_;
            std::memcpy(ov_tgt_[p_id] + ov_sz_[p_id], pool_->tgt(c) + o, trail * sizeof(idx_t));
            std::memcpy(ov_rank_[p_id] + ov_sz_[p_id], pool_->rank(c) + o, trail * sizeof(msg_rank_t));
            ov_sz_[p_id] += trail, msg_count_[parity][p_id] += trail, in.count = from;

            if(o == 0)  // The last chunk emptied.
                release_msg_chunk(c), in.chunk[--in.chunk_c] = Message_Pool::NIL;
        }

        flush_overflow(p_id, true);
    }
}


void Partition_Backend::log_barrier(const idx_t gen, const std::vector<idx_t>& Active_sz)
{
    constexpr auto GB = [](const std::size_t bytes){ return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); };

    // The cohort's content by tier, and the generation's disk-reads of it: of the suffixes alone at the first generation.
    p_id_t active_c = 0;
    std::size_t cohort_ram = 0, cohort_disk = 0;
    uint64_t predicted = 0;
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
    {
        active_c += (Active_sz[p_id] > 0);
        for_each_chunk(0, Active_sz[p_id],
            [&](const idx_t c, idx_t, const idx_t len)
            {
                for(std::size_t f = 0; f < FIELD_C; ++f)
                {
                    const auto field = static_cast<Field>(f);
                    const auto bytes = len * width(field);
                    if(in_ram(field, p_id, c))
                        cohort_ram += bytes;
                    else
                        cohort_disk += bytes, predicted += (gen > 0 || field == Field::suf ? bytes : 0);
                }
            });
    }

    // The cohort's inboxes, i.e. the ones the generation reads, and the generation's disk-reads of them.
    const auto parity = read_parity(gen);
    idx_t msg_ram = 0, msg_disk = 0;
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        if(Active_sz[p_id] > 0)
            msg_ram += ram_inbox_[parity][p_id].unwrap().count, msg_disk += msg_count_[parity][p_id];
    predicted += msg_disk * MESSAGE_BYTES;

    const auto content = content_ram_.load(std::memory_order_relaxed);
    const auto tail = tail_ram_.load(std::memory_order_relaxed);
    const auto [content_files, inbox_files] = disk_.allocated();
    const auto transitions = [&](const Transition t){ return transition_c_[static_cast<std::size_t>(t)].exchange(0, std::memory_order_relaxed); };
    std::cerr   << "Barrier " << gen << " (" << active_c << " partitions active):\n"
                << "\tmemory:  cap " << GB(budget_.cap()) << "GBs, fixed " << GB(fixed_) << "GBs; "
                    << "RSS " << GB(util::resident_memory()) << "GBs, against " << GB(fixed_ + budget_.used()) << "GBs accounted.\n"
                << "\tcontent: cohort " << GB(cohort_ram) << "GBs in RAM, " << GB(cohort_disk) << "GBs on disk; "
                    << "RAM-chunks " << GB(content) << "GBs, " << GB(tail) << "GBs of them retired tails.\n"
                << "\tinboxes: cohort's " << msg_ram << " messages in RAM, " << msg_disk << " on disk; "
                    << "message-chunks " << GB(budget_.used() - content) << "GBs charged, " << GB(budget_.pooled()) << "GBs of them pooled.\n"
                << "\tscratch: spill-files take up " << GB(content_files + inbox_files) << "GBs: "
                    << GB(content_files) << "GBs of content-files, " << GB(inbox_files) << "GBs of inbox-files.\n"
                << "\ttransitions since the last barrier: " << transitions(Transition::ram_write) << " RAM-writes, "
                    << transitions(Transition::disk_write) << " disk-writes, " << transitions(Transition::admit) << " admissions, "
                    << transitions(Transition::admit_tail) << " retiring admissions, " << transitions(Transition::tail_mark) << " tail-marks, "
                    << transitions(Transition::evict) << " evictions, " << transitions(Transition::shuffle) << " tier-shuffles.\n";

    // The preceding generation's disk-reads, against their prediction at the last barrier.
    const auto read = disk_read_.exchange(0, std::memory_order_relaxed);
    std::cerr << "\tdisk-reads: ";
    if(gen > 0)
        std::cerr << read << " bytes at generation " << (gen - 1) << ", predicted " << read_prediction_ << (read == read_prediction_ ? " (exact)" : " (MISMATCH)") << "; ";
    std::cerr << predicted << " bytes predicted for generation " << gen << ".\n";

#ifdef TESTS
    if(gen > 0 && read != read_prediction_)
    {
        std::cerr << "Disk-reads at generation " << (gen - 1) << " differ from their prediction. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    if(active_c == 0)   // No generation opens past the last barrier, to check the tiers at.
        check_tiers(Active_sz);
#endif

    read_prediction_ = predicted;
}


void Partition_Backend::shuffle_tiers(const idx_t gen, const std::vector<idx_t>& Active_sz)
{
    if(TIER_SHUFFLE <= 0.0)
        return;

    std::bernoulli_distribution move(TIER_SHUFFLE);
    const std::size_t field_c = (gen > 0 ? FIELD_C : 1);    // The suffixes alone, before the first generation.
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        for(idx_t c = 0; c * chunk_cap_ < Active_sz[p_id]; ++c) // The cohort's chunks: the retired tails stay.
            for(std::size_t f = 0; f < field_c; ++f)
                if(move(shuffle_rng_))
                {
                    const auto field = static_cast<Field>(f);
                    const auto bytes = chunk_bytes(field, p_id, c);
                    if(in_ram(field, p_id, c))
                        move_to_disk(field, p_id, c), budget_.uncharge(bytes);
                    else if(budget_.try_charge(bytes))  // A move to RAM takes free room only.
                        move_to_ram(field, p_id, c);
                    else
                        continue;
#ifdef STATS
                    count(Transition::shuffle);
#endif
                }

    // The retired tails leave likewise, through the evictor, and their charges return to the budget.
    std::size_t evict_c = 0;
    for(std::size_t x = 0; x < FIELD_C * chunk_base_[p_]; ++x)
        evict_c += (tier_[x].load(std::memory_order_relaxed) == static_cast<uint8_t>(Tier::ram_tail) && move(shuffle_rng_));

    for(std::size_t i = 0; i < evict_c; ++i)
        budget_.uncharge(evict_one());
}


p_sz_t Partition_Backend::inbox_size(const p_id_t p_id) const
{
    return ram_inbox_[inbox_read_parity_][p_id].unwrap().count + msg_count_[inbox_read_parity_][p_id];
}


idx_t Partition_Backend::inbox_mass_last_gen() const
{
    idx_t sz = 0;
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        sz += msg_count(p_id);

    return sz;
}


idx_t Partition_Backend::disk_mass_last_gen() const
{
    idx_t sz = 0;
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        sz += disk_count(p_id);

    return sz;
}

}
