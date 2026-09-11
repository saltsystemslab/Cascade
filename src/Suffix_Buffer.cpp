
#include "Suffix_Buffer.hpp"
#include "utility.hpp"

#include "omp.h"

#include <utility>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Suffix_Buffer::Suffix_Buffer(idx_t* const suf_mem, Direct_File* const suf_file):
      suf_mem_(suf_mem)
    , suf_file_(suf_file)
    , w_suf_base_(nullptr)
    , w_fill_(omp_get_max_threads())
    , flushed_(0)
    , worker_c_(omp_get_max_threads())
{
    w_suf_base_ = util::aligned_allocate<idx_t>(worker_c_ * w_buf_cap, Direct_File::DISK_BLOCK);
    if(w_suf_base_ == nullptr)
    {
        std::cerr << "Error in aligned-allocating scatter buffers of partitions. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    std::for_each(w_fill_.begin(), w_fill_.end(), [](auto& fill){ fill.unwrap() = 0; });
}


Suffix_Buffer::Suffix_Buffer(Suffix_Buffer&& rhs):
      suf_mem_(rhs.suf_mem_)
    , suf_file_(rhs.suf_file_)
    , w_suf_base_(rhs.w_suf_base_)
    , w_fill_(std::move(rhs.w_fill_))
    , flushed_(rhs.flushed_.load(std::memory_order_relaxed))
    , worker_c_(rhs.worker_c_)
{
    rhs.suf_mem_ = nullptr;
    rhs.suf_file_ = nullptr;
    rhs.w_suf_base_ = nullptr;
}


Suffix_Buffer::~Suffix_Buffer()
{
    util::deallocate(w_suf_base_);
}


std::size_t Suffix_Buffer::scatter_buffer_memory(const std::size_t worker_c, const std::size_t p)
{
    return p * (worker_c * w_buf_cap * sizeof(idx_t));
}


void Suffix_Buffer::flush()
{
    idx_t total_tail = w_fill_[0].unwrap();
    w_fill_[0].unwrap() = 0;

    for(idx_t w_id = 1; w_id < worker_c_; ++w_id)
    {
        auto& tail_w = w_fill_[w_id].unwrap();
        std::memmove(w_suf_base_ + total_tail, w_suf(w_id), tail_w * sizeof(idx_t));

        total_tail += tail_w;
        tail_w = 0;
    }

    if(total_tail > 0)
    {
        const auto off = flushed_.fetch_add(total_tail, std::memory_order_relaxed);
        sink(off, total_tail, w_suf_base_);
    }
}


void Suffix_Buffer::sink(const idx_t off, const idx_t sz, const idx_t* suf)
{
    suf_mem_ != nullptr ?   (void) std::memcpy(suf_mem_ + off, suf, sz * sizeof(idx_t)) :
                            suf_file_->write_at(suf, sz * sizeof(idx_t), off * sizeof(idx_t));
}

}
