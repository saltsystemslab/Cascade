
#include "Suffix_Buffer.cuh"
#include "utility.hpp"

#include "omp.h"

#include <utility>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Suffix_Buffer::Suffix_Buffer(Direct_File suf_file, Direct_File key_file, idx_t cap):
      suf_file_(std::move(suf_file))
    , key_file_(std::move(key_file))
    , w_suf_base_(nullptr)
    , w_key_base_(nullptr)
    , w_fill_(omp_get_max_threads())
    , flushed_(0)
    , worker_c_(omp_get_max_threads())
{
    suf_file_.preallocate(cap * sizeof(idx_t));
    key_file_.preallocate(cap * sizeof(Prefix_Key));

    w_suf_base_ = util::aligned_allocate<idx_t>(worker_c_ * w_buf_cap, Direct_File::DISK_BLOCK);
    w_key_base_ = util::aligned_allocate<Prefix_Key>(worker_c_ * w_buf_cap, Direct_File::DISK_BLOCK);
    if(w_suf_base_ == nullptr || w_key_base_ == nullptr)
    {
        std::cerr << "Error in aligned-allocating scatter buffers of partitions. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    std::for_each(w_fill_.begin(), w_fill_.end(), [](auto& fill){ fill.unwrap() = 0; });
}


Suffix_Buffer::Suffix_Buffer(Suffix_Buffer&& rhs):
      suf_file_(std::move(rhs.suf_file_))
    , key_file_(std::move(rhs.key_file_))
    , w_suf_base_(rhs.w_suf_base_)
    , w_key_base_(rhs.w_key_base_)
    , w_fill_(std::move(rhs.w_fill_))
    , flushed_(rhs.flushed_.load(std::memory_order_relaxed))
    , worker_c_(rhs.worker_c_)
{
    rhs.w_suf_base_ = nullptr;
    rhs.w_key_base_ = nullptr;
}


Suffix_Buffer::~Suffix_Buffer()
{
    util::deallocate(w_suf_base_);
    util::deallocate(w_key_base_);
}


void Suffix_Buffer::flush()
{
    idx_t total_tail = w_fill_[0].unwrap();
    w_fill_[0].unwrap() = 0;

    for(idx_t w_id = 1; w_id < worker_c_; ++w_id)
    {
        auto& tail_w = w_fill_[w_id].unwrap();
        std::memmove(w_suf_base_ + total_tail, w_suf(w_id), tail_w * sizeof(idx_t));
        std::memmove(w_key_base_ + total_tail, w_key(w_id), tail_w * sizeof(Prefix_Key));

        total_tail += tail_w;
        tail_w = 0;
    }

    if(total_tail > 0)
    {
        const auto off = flushed_.fetch_add(total_tail, std::memory_order_relaxed);
        suf_file_.write_at(w_suf_base_, total_tail * sizeof(idx_t), off * sizeof(idx_t));
        key_file_.write_at(w_key_base_, total_tail * sizeof(Prefix_Key), off * sizeof(Prefix_Key));
    }
}

}
