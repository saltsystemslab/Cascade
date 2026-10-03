
#include "Staging_Slab.hpp"
#include "lane.hpp"
#include "utility.cuh"


namespace gpu_saca
{

Staging_Slab::Staging_Slab():
      key_(nullptr)
    , suf_(nullptr)
    , beta_(nullptr)
    , msg_target_(nullptr)
    , msg_rank_(nullptr)
    , alloc_{}
{}


Staging_Slab::~Staging_Slab()
{
    for(auto* const alloc : alloc_)
        CUDA_CHECK(deallocate_host(alloc));
}


void Staging_Slab::allocate(idx_t sz)
{
    pinned_allocate_aligned_host(key_, alloc_[0], sz);
    pinned_allocate_aligned_host(suf_, alloc_[1], lane::bytes(sz));
    pinned_allocate_aligned_host(beta_, alloc_[2], sz);
    pinned_allocate_aligned_host(msg_target_, alloc_[3], 2 * sz);  // Outbox can have twice the size of active set, in the worst-case.
    pinned_allocate_aligned_host(msg_rank_, alloc_[4], 2 * sz);
}


void Staging_Slab::assign(const p_id_t p_id, const idx_t off, const idx_t sz, const p_sz_t msg_off, const p_sz_t inbox_sz)
{
    p_id_ = p_id, off_ = off, sz_ = sz, inbox_off_ = msg_off, inbox_sz_ = inbox_sz;
}

}
