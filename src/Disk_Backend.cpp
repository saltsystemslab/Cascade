
#include "Disk_Backend.hpp"


namespace gpu_saca
{

Disk_Backend::Disk_Backend(Spill_Store& spill, const p_id_t p, const std::vector<idx_t>& P_sz, idx_t* const rho, const idx_t n, const idx_t secondary_bits):
      spill_(spill)
    , rho_(rho)
    , n_(n)
    , secondary_bits_(secondary_bits)
    , k_(0)
{
    suf_.reserve(p), key_.reserve(p), prefix_.reserve(p);
    for(p_id_t p_id = 0; p_id < p; ++p_id)
    {
        suf_.emplace_back(spill_.open(p_id, Spill_Role::suf, Direct_File::Mode::read_write));
        key_.emplace_back(spill_.open(p_id, Spill_Role::key, Direct_File::Mode::read_write));
        prefix_.emplace_back(spill_.open(p_id, Spill_Role::prefix, Direct_File::Mode::read));

        // Suffix- and prefix-key-files have already been opened during partitioning.
        key_.back().preallocate(P_sz[p_id] * sizeof(idx_t));
    }
}


void Disk_Backend::read_prefixes(Staging_Slab& s) const
{
    const auto p_id = s.p_id();
    suf_[p_id].read_at(s.suf(), s.size() * sizeof(idx_t), s.off() * sizeof(idx_t));
    prefix_[p_id].read_at(s.key(), s.size() * sizeof(Prefix_Key), s.off() * sizeof(Prefix_Key));
}


void Disk_Backend::read_keys(Staging_Slab& s) const
{
    const auto p_id = s.p_id();
    suf_[p_id].read_at(s.suf(), s.size() * sizeof(idx_t), s.off() * sizeof(idx_t));
    key_[p_id].read_at(s.key(), s.size() * sizeof(idx_t), s.off() * sizeof(idx_t));

    // Gather the secondary keys of the read suffixes and fold those in to the keys.
    {
        const idx_t* const suf = s.suf();
        idx_t* const key = reinterpret_cast<idx_t*>(s.key());
        for(idx_t i = 0; i < s.size(); ++i)
        {
            const auto j = suf[i];  // Suffix.
            const idx_t secondary = (j + k_ < n_ ? rho_[j + k_] : 0);
            key[i] = (key[i] << secondary_bits_) | secondary;
        }
    }
}


void Disk_Backend::write(const Staging_Slab& s)
{
    const auto p_id = s.p_id();
    suf_[p_id].write_at(s.suf(), s.size() * sizeof(idx_t), s.off() * sizeof(idx_t));
    key_[p_id].write_at(s.key(), s.size() * sizeof(idx_t), s.off() * sizeof(idx_t));
}


void Disk_Backend::drop_prefixes()
{
    for(auto& pre : prefix_)
        pre.unlink();
}

}
