
#include "Partition_Backend.hpp"

#include <cstring>


namespace gpu_saca
{

RAM_Backend::RAM_Backend(std::vector<Partition>& P, const Prefix_Key* const prefix, const std::vector<idx_t>& P_off):
      P_(P)
    , prefix_(prefix)
    , P_off_(P_off)
{}


void RAM_Backend::read_prefixes(Staging_Slab& s) const
{
    std::memcpy(s.key(), prefix_ + P_off_[s.p_id()] + s.off(), s.size() * sizeof(Prefix_Key));
    std::memcpy(s.suf(), P_[s.p_id()].suf() + s.off(), s.size() * sizeof(idx_t));
}


void RAM_Backend::read_keys(Staging_Slab& s) const
{
    std::memcpy(s.key(), P_[s.p_id()].key() + s.off(), s.size() * sizeof(idx_t));
    std::memcpy(s.suf(), P_[s.p_id()].suf() + s.off(), s.size() * sizeof(idx_t));
}


void RAM_Backend::write(const Staging_Slab& s)
{
    std::memcpy(P_[s.p_id()].key() + s.off(), s.key(), s.size() * sizeof(idx_t));
    std::memcpy(P_[s.p_id()].suf() + s.off(), s.suf(), s.size() * sizeof(idx_t));
}

}
