
#include "Message_Pool.hpp"
#include "message.hpp"
#include "utility.hpp"

#include <mutex>
#include <algorithm>


namespace gpu_saca
{

Message_Pool::Message_Pool(const std::size_t chunk_cap, const std::size_t max_bytes):
      chunk_cap_(chunk_cap)
    , chunk_bytes_(chunk_cap * MESSAGE_BYTES)
    , chunk_c_(static_cast<chunk_t>(std::min<std::size_t>(max_bytes / chunk_bytes_, NIL)))  // `NIL` itself stays out of the index-range.
    , base_(static_cast<char*>(util::reserve_memory(chunk_c_ * chunk_bytes_)))
    , carved_(0)
{}


Message_Pool::~Message_Pool()
{
    util::unreserve_memory(base_, chunk_c_ * chunk_bytes_);
}


Message_Pool::chunk_t Message_Pool::take_free()
{
    const std::lock_guard<util::Spin_Lock> lock(free_lock_);
    if(free_.empty())
        return NIL;

    const auto c = free_.back();
    free_.pop_back();
    return c;
}


Message_Pool::chunk_t Message_Pool::take_new()
{
    auto c = carved_.load(std::memory_order_relaxed);
    do
        if(c >= chunk_c_)
            return NIL;
    while(!carved_.compare_exchange_weak(c, c + 1, std::memory_order_relaxed));

    return c;
}


void Message_Pool::give_back(const chunk_t c)
{
    const std::lock_guard<util::Spin_Lock> lock(free_lock_);
    free_.push_back(c);
}

}
