
#include "Disk_Backend.hpp"
#include "message.hpp"


namespace gpu_saca
{

Disk_Backend::Disk_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const Residency_Plan& plan):
      spill_(&spill)
    , P_cap_(P_cap)
    , data_resident_(plan.data())
    , write_parity_(write_parity(0))
    , read_parity_(read_parity(0))
{
    const auto p = P_cap_.size();
    suf_.reserve(p), key_.reserve(p), prefix_.reserve(p);
    for(std::size_t parity = 0; parity < 2; ++parity)
        msg_tgt_tag_[parity].reserve(p), msg_rank_[parity].reserve(p);

    for(p_id_t p_id = 0; p_id < p; ++p_id)
    {
        // Open the required prefix- and the suffix-files; opening of required key-
        // files is deferred at least until the corresponding prefix-files retire.

        key_.emplace_back();
        if(plan.data(p_id))
            suf_.emplace_back();
        else
        {
            suf_.emplace_back(spill.open(p_id, Spill_Role::suf, Direct_File::Mode::read_write));
            suf_.back().preallocate(P_cap_[p_id] * sizeof(idx_t));
        }

        if(plan.prefix(p_id))
            prefix_.emplace_back();
        else
        {
            prefix_.emplace_back(spill.open(p_id, Spill_Role::prefix, Direct_File::Mode::read_write));
            prefix_.back().preallocate(P_cap_[p_id] * sizeof(Prefix_Key));
        }


        // Open the inbox-files with parity that generation 0 writes to; opening
        // of the inboxes with the other parity is deferred at least until the
        // corresponding prefix-files retire.
        {
            constexpr auto parity = write_parity(0);
            msg_tgt_tag_[parity].emplace_back(spill.open(p_id, msg_tgt_tag_role(parity), Direct_File::Mode::read_write));
            msg_rank_[parity].emplace_back(spill.open(p_id, msg_rank_role(parity), Direct_File::Mode::read_write));

            const auto cap = P_cap_[p_id] + MSG_ALIGN;
            msg_tgt_tag_[parity].back().preallocate(cap * sizeof(idx_t));
            msg_rank_[parity].back().preallocate(cap * sizeof(msg_rank_t));

            msg_tgt_tag_[parity ^ 1].emplace_back(), msg_rank_[parity ^ 1].emplace_back();
        }
    }
}


void Disk_Backend::open_deferred_inboxes()
{
    const auto p = P_cap_.size();
    constexpr auto parity = write_parity(1);

    for(p_id_t p_id = 0; p_id < p; ++p_id)
    {
        msg_tgt_tag_[parity][p_id] = spill_->open(p_id, msg_tgt_tag_role(parity), Direct_File::Mode::read_write);
        msg_rank_[parity][p_id] = spill_->open(p_id, msg_rank_role(parity), Direct_File::Mode::read_write);

        const auto cap = P_cap_[p_id] + MSG_ALIGN;
        msg_tgt_tag_[parity][p_id].preallocate(cap * sizeof(idx_t));
        msg_rank_[parity][p_id].preallocate(cap * sizeof(msg_rank_t));
    }
}


void Disk_Backend::read_prefix(const p_id_t p_id, const idx_t off, const idx_t sz, Prefix_Key* const dst) const
{
    prefix_[p_id].read_at(dst, sz * sizeof(Prefix_Key), off * sizeof(Prefix_Key));
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
    msg_tgt_tag_[read_parity_][p_id].read_at(tgt, sz * sizeof(idx_t), off * sizeof(idx_t));
    msg_rank_[read_parity_][p_id].read_at(rank, sz * sizeof(msg_rank_t), off * sizeof(msg_rank_t));
}


void Disk_Backend::write(const p_id_t p_id, const idx_t off, const idx_t sz, const idx_t* const suf, const p_sz_t* const key)
{
    suf_[p_id].write_at(suf, sz * sizeof(idx_t), off * sizeof(idx_t));
    key_[p_id].write_at(key, sz * sizeof(p_sz_t), off * sizeof(p_sz_t));
}


void Disk_Backend::retire_prefix_and_open_key(const p_id_t p_id)
{
    prefix_[p_id].unlink();

    if(!data_resident_[p_id])
    {
        key_[p_id] = spill_->open(p_id, Spill_Role::key, Direct_File::Mode::read_write);
        key_[p_id].preallocate(P_cap_[p_id] * sizeof(p_sz_t));
    }
}


void Disk_Backend::drop_prefixes()
{
    for(auto& pre : prefix_)
        pre.unlink();
}

}
