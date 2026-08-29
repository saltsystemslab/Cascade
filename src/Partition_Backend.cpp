
#include "Partition_Backend.hpp"
#include "utility.hpp"
#include "utility.cuh"
#include "isa-instrument.hpp"

#include <cstring>
#include <mutex>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Partition_Backend::Partition_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const std::size_t fixed_bytes, const p_id_t p):
      p_(p)
    , plan_(P_cap, fixed_bytes, sizeof(Prefix_Key))
    , ram_(P_cap, plan_)
    , disk_(P_cap, spill, plan_)
    , msg_off_(nullptr)
    , inbox_write_parity_(write_parity(0))
    , inbox_read_parity_(read_parity(0))
    , ov_tgt_(p_, nullptr)
    , ov_rank_(p_, nullptr)
    , ov_sz_(p_, 0)
    , ov_lock_(p_)
{
    CUDA_CHECK(pinned_allocate_host(msg_off_, p_ * p_));

    for(idx_t parity = 0; parity < 2; ++parity)
        msg_count_[parity] = std::vector<std::atomic<p_sz_t>>(p_),
        msg_written_[parity] = std::vector<std::atomic<p_sz_t>>(p_);

    for(idx_t p_id = 0; p_id < p_; ++p_id)
        ov_tgt_[p_id] = util::aligned_allocate<idx_t>(OV_CAP, Direct_File::DISK_BLOCK),
        ov_rank_[p_id] = util::aligned_allocate<msg_rank_t>(OV_CAP, Direct_File::DISK_BLOCK);
}


Partition_Backend::~Partition_Backend()
{
    CUDA_CHECK(deallocate_host(msg_off_));
    for(idx_t p_id = 0; p_id < p_; ++p_id)
        util::deallocate(ov_tgt_[p_id]),
        util::deallocate(ov_rank_[p_id]);
}


void Partition_Backend::open_deferred_inboxes()
{
    disk_.open_deferred_inboxes();
}


void Partition_Backend::retire_prefix_and_open_key(const p_id_t p_id)
{
    disk_.retire_prefix_and_open_key(p_id);
}


void Partition_Backend::read_prefixes(Staging_Slab& s) const
{
    INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);

    read_prefix_chunk(s.p_id(), s.off(), s.size(), s.suf(), s.key());
}


void Partition_Backend::read_keys(Staging_Slab& s) const
{
    auto* const suf = s.suf();
    auto* const key = reinterpret_cast<p_sz_t*>(s.key());

    {
        INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);
        read_chunk(s.p_id(), s.off(), s.size(), suf, key);
    }
}


void Partition_Backend::read_chunk(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const suf, p_sz_t* const key) const
{
    plan_.data(p_id) ?  (ram_.read_suf(p_id, off, sz, suf), ram_.read_key(p_id, off, sz, key)) :
                        (disk_.read_suf(p_id, off, sz, suf), disk_.read_key(p_id, off, sz, key));
}


void Partition_Backend::read_prefix_chunk(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const suf, Prefix_Key* const key) const
{
    plan_.data(p_id) ?  ram_.read_suf(p_id, off, sz, suf) :
                        disk_.read_suf(p_id, off, sz, suf);

    plan_.prefix(p_id) ?    ram_.read_prefix(p_id, off, sz, key) :
                            disk_.read_prefix(p_id, off, sz, key);
}


void Partition_Backend::read_messages(Staging_Slab& s) const
{
    INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);

    disk_.read_messages(s.p_id(), s.inbox_off(), s.inbox_sz(), s.msg_target(), s.msg_rank());
}


void Partition_Backend::write(const Staging_Slab& s)
{
    const auto p_id = s.p_id();
    const auto* const key = reinterpret_cast<const p_sz_t*>(s.key());
    plan_.data(p_id) ?  ram_.write(p_id, s.off(), s.size(), s.suf(), key) :
                        disk_.write(p_id, s.off(), s.size(), s.suf(), key);
}


void Partition_Backend::write_messages(const Staging_Slab& s)
{
    const auto p_id = s.p_id();
    const auto* const msg_off = this->msg_off(p_id);

    // This slab contains the slice `[l_slab, r_slab)` of messages from `p_id`.
    const p_sz_t l_slab = s.outbox_off(), r_slab = std::min<idx_t>(s.outbox_off() + s.outbox_sz(), msg_off[p_ - 1]);
    if(l_slab >= r_slab)
        return;

    for(p_id_t q = 0; q < p_; ++q)
    {
        // `q` is the destination for the slice `[l_q, r_q)` of messages from `p_id`.
        const auto l_q = (q > 0 ? msg_off[q - 1] : 0), r_q = msg_off[q];
        const auto l = std::max(l_slab, l_q), r = std::min(r_slab, r_q); // Intersect the slab's slice with the `q`-destined slice.
        if(l < r)
            append_messages(q, s, l, r);
    }
}


void Partition_Backend::append_messages(const p_id_t p_id, const Staging_Slab& s, const p_sz_t l, const p_sz_t r)
{
    const auto parity = inbox_write_parity_;
    msg_count_[parity][p_id].fetch_add(r - l, std::memory_order_relaxed);

    const auto base = s.outbox_off();   // `s` contains the index-range `[base, base + s.outbox_sz())` of the source-partition's outbox.
    const p_sz_t a_l = ((l + MSG_ALIGN - 1) / MSG_ALIGN) * MSG_ALIGN;   // Leftmost `MSG_ALIGN`-ed endpoint of `[l, r)`.
    const p_sz_t a_r = (r / MSG_ALIGN) * MSG_ALIGN; // Rightmost `MSG_ALIGN`-ed (exclusive) endpoint of `[l, r)`.

    if(a_l < a_r)
    {
        const auto to_write = a_r - a_l;
        const auto write_off = msg_written_[parity][p_id].fetch_add(to_write, std::memory_order_relaxed);
        disk_.msg_tgt_tag(p_id).write_at(s.msg_target() + (a_l - base), to_write * sizeof(idx_t), write_off * sizeof(idx_t));
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
    disk_.msg_tgt_tag(p_id).write_at(ov_tgt_[p_id], to_write * sizeof(idx_t), write_off * sizeof(idx_t));
    disk_.msg_rank(p_id).write_at(ov_rank_[p_id], to_write * sizeof(msg_rank_t), write_off * sizeof(msg_rank_t));

    const auto rem = (to_write < ov_sz_[p_id] ? ov_sz_[p_id] - to_write : 0);
    if(rem > 0)
    {
        std::memmove(ov_tgt_[p_id], ov_tgt_[p_id] + to_write, rem * sizeof(idx_t));
        std::memmove(ov_rank_[p_id], ov_rank_[p_id] + to_write, rem * sizeof(msg_rank_t));
    }

    ov_sz_[p_id] = rem;
}


void Partition_Backend::open_generation(const idx_t gen)
{
    inbox_write_parity_ = write_parity(gen), inbox_read_parity_ = read_parity(gen);
    disk_.set_generation(gen);

    const auto parity = write_parity(gen);
    std::memset(msg_off_, 0, p_ * p_* sizeof(p_sz_t));
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
    {
        msg_count_[parity][p_id] = 0, msg_written_[parity][p_id] = 0;
        ov_sz_[p_id] = 0;
    }
}


void Partition_Backend::close_generation()
{
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
    {
        const std::lock_guard<util::Spin_Lock> lock(ov_lock_[p_id]);
        flush_overflow(p_id, true);
    }
}


p_sz_t Partition_Backend::inbox_size(const p_id_t p_id) const
{
    return msg_count_[inbox_read_parity_][p_id];
}


idx_t Partition_Backend::inbox_mass_last_gen() const
{
    const auto parity = inbox_write_parity_;
    idx_t sz = 0;
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        sz += msg_count_[parity][p_id];

    return sz;
}


void Partition_Backend::drop_prefixes()
{
    ram_.drop_prefixes(), disk_.drop_prefixes();
}

}
