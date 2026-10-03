
#include "Disk_Backend.hpp"
#include "lane.hpp"
#include "message.hpp"

#include <algorithm>


namespace gpu_saca
{

Disk_Backend::Disk_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const Residency_Plan& plan):
      suf_(P_cap.size())
    , key_(P_cap.size())
    , suf_open_(std::make_unique<std::once_flag[]>(P_cap.size()))
    , key_open_(std::make_unique<std::once_flag[]>(P_cap.size()))
    , spill_(&spill)
    , P_cap_(P_cap)
    , write_parity_(write_parity(0))
    , read_parity_(read_parity(0))
{
    const auto p = P_cap_.size();

    // The partitions' spill-files are striped across the scratch before any of them opens.
    std::vector<std::size_t> content(p), inbox(p);
    for(p_id_t p_id = 0; p_id < p; ++p_id)
        content[p_id] = content_footprint(P_cap_[p_id]), inbox[p_id] = inbox_footprint(P_cap_[p_id]);
    spill_->stripe(content, inbox, plan.cap());

    for(std::size_t parity = 0; parity < 2; ++parity)   // Inbox-files are opened lazily, at their first write, and grow with their messages.
        msg_tgt_[parity].resize(p), msg_rank_[parity].resize(p),
        inbox_open_[parity] = std::make_unique<std::once_flag[]>(p),
        inbox_room_[parity] = std::make_unique<std::atomic<p_sz_t>[]>(p),
        inbox_grow_[parity] = std::make_unique<std::mutex[]>(p);

    for(p_id_t p_id = 0; p_id < p; ++p_id)  // The suffix-files with chunks spilled open upfront, off the scatter's path.
        if(plan.suf_ram(p_id) < plan.chunk_c(p_id))
            ensure_suf(p_id);
}


std::size_t Disk_Backend::content_footprint(const idx_t cap)
{
    return Direct_File::block_up(lane::bytes(cap)) + Direct_File::block_up(cap * sizeof(p_sz_t));
}


std::size_t Disk_Backend::inbox_footprint(const idx_t cap)
{
    const auto inbox_cap = cap + MSG_ALIGN; // As far as the inbox-files grow.
    return Direct_File::block_up(lane::bytes(inbox_cap)) + Direct_File::block_up(inbox_cap * sizeof(msg_rank_t));
}


void Disk_Backend::ensure_suf(const p_id_t p_id)
{
    std::call_once(suf_open_[p_id], // A content-file is opened once, and never re-assigned: concurrent writes may be using it.
        [&]
        {
            suf_[p_id] = spill_->open(p_id, Spill_Role::suf, Direct_File::Mode::read_write);
            suf_[p_id].preallocate(lane::bytes(P_cap_[p_id]));
        }
    );
}


void Disk_Backend::ensure_key(const p_id_t p_id)
{
    std::call_once(key_open_[p_id], // A content-file is opened once, and never re-assigned: concurrent writes may be using it.
        [&]
        {
            key_[p_id] = spill_->open(p_id, Spill_Role::key, Direct_File::Mode::read_write);
            key_[p_id].preallocate(P_cap_[p_id] * sizeof(p_sz_t));
        }
    );
}


void Disk_Backend::ensure_inbox(const p_id_t p_id)
{
    const auto parity = write_parity_;
    std::call_once(inbox_open_[parity][p_id],   // An inbox-file is opened once, and never re-assigned: concurrent writes may be using it.
        [&]
        {
            msg_tgt_[parity][p_id] = spill_->open(p_id, msg_tgt_role(parity), Direct_File::Mode::read_write);
            msg_rank_[parity][p_id] = spill_->open(p_id, msg_rank_role(parity), Direct_File::Mode::read_write);
        }
    );
}


void Disk_Backend::ensure_inbox_room(const p_id_t p_id, const p_sz_t msg_c)
{
    const auto parity = write_parity_;
    auto& room = inbox_room_[parity][p_id];
    if(room.load(std::memory_order_acquire) >= msg_c)   // Sees the growth that made the room.
        return;

    const std::lock_guard<std::mutex> lock(inbox_grow_[parity][p_id]);
    const auto r = room.load(std::memory_order_relaxed);
    if(r >= msg_c)  // Grown by another writer meanwhile.
        return;

    // The files grow in steps, so that they are rarely extended under concurrent writes; and no further than an inbox holds:
    // a message per suffix of its partition, padded to `MSG_ALIGN`.
    const idx_t cap = P_cap_[p_id] + MSG_ALIGN;
    const auto to = std::max<idx_t>(msg_c, std::min<idx_t>(cap, idx_t(r) + std::max<idx_t>(r / 4, INBOX_GROWTH)));
    msg_tgt_[parity][p_id].preallocate(lane::bytes(to));
    msg_rank_[parity][p_id].preallocate(to * sizeof(msg_rank_t));
    room.store(static_cast<p_sz_t>(to), std::memory_order_release);
}


void Disk_Backend::retire_inbox(const p_id_t p_id)
{
    const auto parity = read_parity_;
    msg_tgt_[parity][p_id].truncate(), msg_rank_[parity][p_id].truncate();  // No-op for inbox-files never opened.
    inbox_room_[parity][p_id].store(0, std::memory_order_relaxed);  // Written again from empty, two generations on.
}


void Disk_Backend::retire_inboxes()
{
    for(std::size_t parity = 0; parity < 2; ++parity)
        for(std::size_t p_id = 0; p_id < P_cap_.size(); ++p_id)
            msg_tgt_[parity][p_id].unlink(), msg_rank_[parity][p_id].unlink(),  // No-op for inbox-files never opened.
            inbox_room_[parity][p_id].store(0, std::memory_order_relaxed);
}


std::pair<std::size_t, std::size_t> Disk_Backend::allocated() const
{
    std::size_t content = 0, inboxes = 0;
    for(std::size_t p_id = 0; p_id < P_cap_.size(); ++p_id)
    {
        content += (suf_[p_id].path().empty() ? 0 : Direct_File::block_up(lane::bytes(P_cap_[p_id])));  // Files unopened or retired have no path.
        content += (key_[p_id].path().empty() ? 0 : Direct_File::block_up(P_cap_[p_id] * sizeof(p_sz_t)));
        for(std::size_t parity = 0; parity < 2; ++parity)
        {
            const std::size_t room = inbox_room_[parity][p_id].load(std::memory_order_relaxed);
            inboxes += Direct_File::block_up(lane::bytes(room)) + Direct_File::block_up(room * sizeof(msg_rank_t));
        }
    }

    return {content, inboxes};
}


void Disk_Backend::read_suf(const p_id_t p_id, const idx_t off, const idx_t sz, uint8_t* const dst) const
{
    suf_[p_id].read_at(dst, lane::bytes(sz), lane::bytes(off));
}


void Disk_Backend::read_key(const p_id_t p_id, const idx_t off, const idx_t sz, p_sz_t* const dst) const
{
    key_[p_id].read_at(dst, sz * sizeof(p_sz_t), off * sizeof(p_sz_t));
}


void Disk_Backend::read_messages(const p_id_t p_id, const p_sz_t off, const p_sz_t sz, uint8_t* const tgt, msg_rank_t* const rank) const
{
    msg_tgt_[read_parity_][p_id].read_at(tgt, lane::bytes(sz), lane::bytes(off));
    msg_rank_[read_parity_][p_id].read_at(rank, sz * sizeof(msg_rank_t), off * sizeof(msg_rank_t));
}


void Disk_Backend::write_suf(const p_id_t p_id, const idx_t off, const idx_t sz, const uint8_t* const src)
{
    ensure_suf(p_id);
    suf_[p_id].write_at(src, lane::bytes(sz), lane::bytes(off));
}


void Disk_Backend::write_key(const p_id_t p_id, const idx_t off, const idx_t sz, const p_sz_t* const src)
{
    ensure_key(p_id);   // Key-files open at their first write, as partitions leave initial ranking.
    key_[p_id].write_at(src, sz * sizeof(p_sz_t), off * sizeof(p_sz_t));
}


void Disk_Backend::retire_data(const p_id_t p_id)
{
    suf_[p_id].unlink(), key_[p_id].unlink();
}

}
