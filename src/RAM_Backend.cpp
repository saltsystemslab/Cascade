
#include "RAM_Backend.hpp"
#include "Direct_File.hpp"
#include "utility.hpp"

#include <cstring>


namespace gpu_saca
{

RAM_Backend::RAM_Backend(const std::vector<idx_t>& P_cap):
      P_cap_(P_cap)
    , suf_(P_cap.size(), nullptr)
    , key_(P_cap.size(), nullptr)
{
    for(p_id_t p_id = 0; p_id < P_cap_.size(); ++p_id)
        suf_[p_id] = static_cast<idx_t*>(util::reserve_memory(reservation(P_cap_[p_id], sizeof(idx_t)))),
        key_[p_id] = static_cast<p_sz_t*>(util::reserve_memory(reservation(P_cap_[p_id], sizeof(p_sz_t))));
}


RAM_Backend::~RAM_Backend()
{
    for(p_id_t p_id = 0; p_id < P_cap_.size(); ++p_id)
        util::unreserve_memory(suf_[p_id], reservation(P_cap_[p_id], sizeof(idx_t))),
        util::unreserve_memory(key_[p_id], reservation(P_cap_[p_id], sizeof(p_sz_t)));
}


std::size_t RAM_Backend::reservation(const idx_t n, const std::size_t w)
{
    return Direct_File::block_up(n * w);
}


void RAM_Backend::release_suf(const p_id_t p_id, const idx_t off, const idx_t sz)
{
    util::release_memory(suf_[p_id] + off, sz * sizeof(idx_t));
}


void RAM_Backend::release_key(const p_id_t p_id, const idx_t off, const idx_t sz)
{
    util::release_memory(key_[p_id] + off, sz * sizeof(p_sz_t));
}


void RAM_Backend::read_suf(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const dst) const
{
    std::memcpy(dst, suf_[p_id] + off, sz * sizeof(idx_t));
}


void RAM_Backend::read_key(const p_id_t p_id, const idx_t off, const idx_t sz, p_sz_t* const dst) const
{
    std::memcpy(dst, key_[p_id] + off, sz * sizeof(p_sz_t));
}


void RAM_Backend::write_suf(const p_id_t p_id, const idx_t off, const idx_t sz, const idx_t* const src)
{
    std::memcpy(suf_[p_id] + off, src, sz * sizeof(idx_t));
}


void RAM_Backend::write_key(const p_id_t p_id, const idx_t off, const idx_t sz, const p_sz_t* const src)
{
    std::memcpy(key_[p_id] + off, src, sz * sizeof(p_sz_t));
}

}
