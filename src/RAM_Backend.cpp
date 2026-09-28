
#include "RAM_Backend.hpp"
#include "utility.hpp"

#include <cstring>


namespace gpu_saca
{

RAM_Backend::RAM_Backend(const std::vector<idx_t>& P_cap, const Residency_Plan& plan):
      suf_(P_cap.size(), nullptr)
    , key_(P_cap.size(), nullptr)
{
    for(p_id_t p_id = 0; p_id < P_cap.size(); ++p_id)
        if(plan.data(p_id))
        {
            suf_[p_id] = util::checked_allocate<idx_t>(P_cap[p_id]);
            key_[p_id] = util::checked_allocate<p_sz_t>(P_cap[p_id]);
        }
}


RAM_Backend::~RAM_Backend()
{
    const auto p = suf_.size();
    for(p_id_t p_id = 0; p_id < p; ++p_id)
        util::deallocate(suf_[p_id]),
        util::deallocate(key_[p_id]);
}


void RAM_Backend::retire_data(const p_id_t p_id)
{
    util::deallocate(suf_[p_id]), suf_[p_id] = nullptr;
    util::deallocate(key_[p_id]), key_[p_id] = nullptr;
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
