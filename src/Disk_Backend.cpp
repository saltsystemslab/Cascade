
#include "Disk_Backend.hpp"
#include "message.hpp"


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
    for(std::size_t parity = 0; parity < 2; ++parity)   // Inbox-files are opened lazily, at their first write.
        msg_tgt_[parity].resize(p), msg_rank_[parity].resize(p),
        inbox_open_[parity] = std::make_unique<std::once_flag[]>(p);

    for(p_id_t p_id = 0; p_id < p; ++p_id)  // The suffix-files with chunks spilled open upfront, off the scatter's path.
        if(plan.suf_ram(p_id) < plan.chunk_c(p_id))
            ensure_suf(p_id);
}


void Disk_Backend::ensure_suf(const p_id_t p_id)
{
    std::call_once(suf_open_[p_id], // A content-file is opened once, and never re-assigned: concurrent writes may be using it.
        [&]
        {
            suf_[p_id] = spill_->open(p_id, Spill_Role::suf, Direct_File::Mode::read_write);
            suf_[p_id].preallocate(P_cap_[p_id] * sizeof(idx_t));
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

            const auto cap = P_cap_[p_id] + MSG_ALIGN;
            msg_tgt_[parity][p_id].preallocate(cap * sizeof(idx_t));
            msg_rank_[parity][p_id].preallocate(cap * sizeof(msg_rank_t));
        }
    );
}


void Disk_Backend::retire_inboxes()
{
    for(std::size_t parity = 0; parity < 2; ++parity)
        for(std::size_t p_id = 0; p_id < P_cap_.size(); ++p_id)
            msg_tgt_[parity][p_id].unlink(), msg_rank_[parity][p_id].unlink();  // No-op for inbox-files never opened.
}


void Disk_Backend::read_suf(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const dst) const
{
    suf_[p_id].read_at(dst, sz * sizeof(idx_t), off * sizeof(idx_t));
}


void Disk_Backend::read_key(const p_id_t p_id, const idx_t off, const idx_t sz, p_sz_t* const dst) const
{
    key_[p_id].read_at(dst, sz * sizeof(p_sz_t), off * sizeof(p_sz_t));
}


void Disk_Backend::read_messages(const p_id_t p_id, const p_sz_t off, const p_sz_t sz, idx_t* const tgt, msg_rank_t* const rank) const
{
    msg_tgt_[read_parity_][p_id].read_at(tgt, sz * sizeof(idx_t), off * sizeof(idx_t));
    msg_rank_[read_parity_][p_id].read_at(rank, sz * sizeof(msg_rank_t), off * sizeof(msg_rank_t));
}


void Disk_Backend::write_suf(const p_id_t p_id, const idx_t off, const idx_t sz, const idx_t* const src)
{
    ensure_suf(p_id);
    suf_[p_id].write_at(src, sz * sizeof(idx_t), off * sizeof(idx_t));
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
