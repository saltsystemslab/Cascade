
#include "Disk_Backend.hpp"
#include "isa-instrument.hpp"


namespace gpu_saca
{

Disk_Backend::Disk_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const Residency_Plan& plan)
{
    const auto p = P_cap.size();
    suf_.reserve(p), key_.reserve(p), prefix_.reserve(p);
    for(p_id_t p_id = 0; p_id < p; ++p_id)
    {
        if(plan.data(p_id))
            suf_.emplace_back(), key_.emplace_back();
        else
        {
            suf_.emplace_back(spill.open(p_id, Spill_Role::suf, Direct_File::Mode::read_write));
            key_.emplace_back(spill.open(p_id, Spill_Role::key, Direct_File::Mode::read_write));

            suf_.back().preallocate(P_cap[p_id] * sizeof(idx_t));
            key_.back().preallocate(P_cap[p_id] * sizeof(idx_t));
        }

        if(plan.prefix(p_id))
            prefix_.emplace_back();
        else
        {
            prefix_.emplace_back(spill.open(p_id, Spill_Role::prefix, Direct_File::Mode::read_write));
            prefix_.back().preallocate(P_cap[p_id] * sizeof(Prefix_Key));
        }
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


void Disk_Backend::read_key(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const dst) const
{
    key_[p_id].read_at(dst, sz * sizeof(idx_t), off * sizeof(idx_t));
}


void Disk_Backend::write(const p_id_t p_id, const idx_t off, const idx_t sz, const idx_t* const suf, const idx_t* const key)
{
    suf_[p_id].write_at(suf, sz * sizeof(idx_t), off * sizeof(idx_t));
    key_[p_id].write_at(key, sz * sizeof(idx_t), off * sizeof(idx_t));
}


void Disk_Backend::drop_prefixes()
{
    for(auto& pre : prefix_)
        pre.unlink();
}

}
