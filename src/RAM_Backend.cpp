
#include "RAM_Backend.hpp"
#include "utility.hpp"

#include <cstring>


namespace gpu_saca
{

RAM_Backend::RAM_Backend(const std::vector<idx_t>& P_cap, const Residency_Plan& plan):
      suf_(P_cap.size(), nullptr)
    , key_(P_cap.size(), nullptr)
    , prefix_(P_cap.size(), nullptr)
{
    for(p_id_t p_id = 0; p_id < P_cap.size(); ++p_id)
    {
        if(plan.data(p_id))
        {
            suf_[p_id] = util::allocate<idx_t>(P_cap[p_id]);
            key_[p_id] = util::allocate<idx_t>(P_cap[p_id]);
        }

        if(plan.prefix(p_id))
            prefix_[p_id] = util::allocate<Prefix_Key>(P_cap[p_id]);
    }
}


RAM_Backend::~RAM_Backend()
{
    const auto p = suf_.size();
    for(p_id_t p_id = 0; p_id < p; ++p_id)
        util::deallocate(suf_[p_id]),
        util::deallocate(key_[p_id]),
        util::deallocate(prefix_[p_id]);
}


void RAM_Backend::drop_prefixes()
{
    for(auto*& p : prefix_)
        util::deallocate(p),
        p = nullptr;
}


void RAM_Backend::read_prefix(const p_id_t p_id, const idx_t off, const idx_t sz, Prefix_Key* const dst) const
{
    std::memcpy(dst, prefix_[p_id] + off, sz * sizeof(Prefix_Key));
}


void RAM_Backend::read_suf(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const dst) const
{
    std::memcpy(dst, suf_[p_id] + off, sz * sizeof(idx_t));
}


void RAM_Backend::read_key(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const dst) const
{
    std::memcpy(dst, key_[p_id] + off, sz * sizeof(idx_t));
}


void RAM_Backend::write(const p_id_t p_id, const  idx_t off, const  idx_t sz, const idx_t* const suf, const idx_t* const key)
{
    std::memcpy(suf_[p_id] + off, suf, sz * sizeof(idx_t));
    std::memcpy(key_[p_id] + off, key, sz * sizeof(idx_t));
}

}
