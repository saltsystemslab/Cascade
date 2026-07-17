
#include "Partition_Backend.cuh"

#include <cstring>


namespace gpu_saca
{

RAM_Backend::RAM_Backend(std::vector<Partition>& P, const Prefix_Key* const prefix, const std::vector<idx_t>& P_off):
      P_(P)
    , prefix_(prefix)
    , P_off_(P_off)
{}


void RAM_Backend::read_prefixes(const p_id_t p_id, const idx_t sz, Staging_Slab& s) const
{
    std::memcpy(s.key(), prefix_ + P_off_[p_id], sz * sizeof(Prefix_Key));
    std::memcpy(s.suf(), P_[p_id].suf(), sz * sizeof(idx_t));
}


void RAM_Backend::read_keys(const p_id_t p_id, const idx_t sz, Staging_Slab& s) const
{
    std::memcpy(s.key(), P_[p_id].key(), sz * sizeof(idx_t));
    std::memcpy(s.suf(), P_[p_id].suf(), sz * sizeof(idx_t));
}


void RAM_Backend::write(const p_id_t p_id, const idx_t sz, const Staging_Slab& s)
{
    std::memcpy(P_[p_id].key(), s.key(), sz * sizeof(idx_t));
    std::memcpy(P_[p_id].suf(), s.suf(), sz * sizeof(idx_t));
}

}
