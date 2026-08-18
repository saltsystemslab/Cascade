
#include "Staging_Slab.hpp"
#include "utility.cuh"


namespace gpu_saca
{

Staging_Slab::Staging_Slab():
      key_(nullptr)
    , suf_(nullptr)
    , beta_(nullptr)
    , msg_target_(nullptr)
    , msg_rank_(nullptr)
{}


Staging_Slab::~Staging_Slab()
{
    CUDA_CHECK(deallocate_host(key_));
    CUDA_CHECK(deallocate_host(suf_));
    CUDA_CHECK(deallocate_host(beta_));
    CUDA_CHECK(deallocate_host(msg_target_));
    CUDA_CHECK(deallocate_host(msg_rank_));
}


void Staging_Slab::allocate(idx_t sz)
{
    CUDA_CHECK(pinned_allocate_host(key_, sz));
    CUDA_CHECK(pinned_allocate_host(suf_, sz));
    CUDA_CHECK(pinned_allocate_host(beta_, sz));
    CUDA_CHECK(pinned_allocate_host(msg_target_, sz));
    CUDA_CHECK(pinned_allocate_host(msg_rank_, sz));
}


void Staging_Slab::assign(const p_id_t p_id, const idx_t off, const idx_t sz)
{
    p_id_ = p_id, off_ = off, sz_ = sz;
}

}
