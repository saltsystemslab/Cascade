
#include "Staging_Slab.hpp"
#include "Direct_File.hpp"
#include "utility.cuh"

#include <cstdlib>


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
    // TODO: fix the following allocations by disk-block-aligning them and omit the failure-mode exit.
    CUDA_CHECK(pinned_allocate_host(key_, sz));
    CUDA_CHECK(pinned_allocate_host(suf_, sz));
    CUDA_CHECK(pinned_allocate_host(beta_, sz));
    CUDA_CHECK(pinned_allocate_host(msg_target_, 2 * sz));  // Outbox can have twice the size of active set, in the worst-case.
    CUDA_CHECK(pinned_allocate_host(msg_rank_, 2 * sz));

    const auto aligned = [](const auto* const addr){ return Direct_File::is_block_aligned(addr); };
    if(!aligned(key_) || !aligned(suf_) || !aligned(msg_target_) || !aligned(msg_rank_))
    {
        std::cerr << "Staging slabs' allocations are not disk-block aligned. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }
}


void Staging_Slab::assign(const p_id_t p_id, const idx_t off, const idx_t sz, const p_sz_t msg_off, const p_sz_t inbox_sz)
{
    p_id_ = p_id, off_ = off, sz_ = sz, inbox_off_ = msg_off, inbox_sz_ = inbox_sz;
}

}
