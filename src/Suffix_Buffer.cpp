
#include "Suffix_Buffer.hpp"
#include "Partition_Backend.hpp"
#include "utility.hpp"

#include "omp.h"

#include <utility>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Suffix_Buffer::Suffix_Buffer(Partition_Backend& backend, const p_id_t p_id):
      backend_(&backend)
    , p_id_(p_id)
    , w_suf_base_(nullptr)
    , w_fill_(omp_get_max_threads())
    , flushed_(0)
    , worker_c_(omp_get_max_threads())
{
    w_suf_base_ = util::aligned_allocate<uint8_t>(lane::bytes(worker_c_ * w_buf_cap), Direct_File::DISK_BLOCK);
    if(w_suf_base_ == nullptr)
    {
        std::cerr << "Error in aligned-allocating scatter buffers of partitions. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    std::for_each(w_fill_.begin(), w_fill_.end(), [](auto& fill){ fill.unwrap() = 0; });
}


Suffix_Buffer::Suffix_Buffer(Suffix_Buffer&& rhs):
      backend_(rhs.backend_)
    , p_id_(rhs.p_id_)
    , w_suf_base_(rhs.w_suf_base_)
    , w_fill_(std::move(rhs.w_fill_))
    , flushed_(rhs.flushed_.load(std::memory_order_relaxed))
    , worker_c_(rhs.worker_c_)
{
    rhs.backend_ = nullptr;
    rhs.w_suf_base_ = nullptr;
}


Suffix_Buffer::~Suffix_Buffer()
{
    util::deallocate(w_suf_base_);
}


std::size_t Suffix_Buffer::scatter_buffer_memory(const std::size_t worker_c, const std::size_t p)
{
    return p * lane::bytes(worker_c * w_buf_cap);
}


void Suffix_Buffer::flush()
{
    idx_t total_tail = w_fill_[0].unwrap();
    w_fill_[0].unwrap() = 0;

    for(idx_t w_id = 1; w_id < worker_c_; ++w_id)
    {
        auto& tail_w = w_fill_[w_id].unwrap();
        std::memmove(w_suf_base_ + lane::bytes(total_tail), w_suf(w_id), lane::bytes(tail_w));

        total_tail += tail_w;
        tail_w = 0;
    }

    if(total_tail > 0)
    {
        const auto off = flushed_.fetch_add(total_tail, std::memory_order_relaxed);
        sink(off, total_tail, w_suf_base_);
    }
}


void Suffix_Buffer::sink(const idx_t off, const idx_t sz, const uint8_t* suf)
{
    // Full buffers land at multiples of `w_buf_cap`; only the tails' flush, the partition's last, ends past a disk-block.
    backend_->write_suffixes(p_id_, off, sz, suf);
}

}
