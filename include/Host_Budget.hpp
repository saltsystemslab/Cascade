
#ifndef HOST_BUDGET_HPP
#define HOST_BUDGET_HPP



#include <cstddef>
#include <atomic>


namespace gpu_saca
{

// Host-memory budget of the partition-backend's elastic claimants: the RAM-tier
// chunks of partition-content and of messages. Accounts in bytes charged, which
// never exceed the cap; and the cap only rises.
class Host_Budget
{
private:

    std::atomic<std::size_t> cap_;  // Cap on the charged bytes.
    std::atomic<std::size_t> used_; // Bytes charged: held by the claimants, including the free message-chunks still resident.
    std::atomic<std::size_t> pooled_;   // Bytes of the free message-chunks still resident, out of `used_`; used only as a statistic.
    const bool pinned_; // Whether the cap is pinned at zero, i.e. everything spills.

public:

    // Constructs a budget with cap `cap`, or with the cap pinned at zero if
    // `pinned` is `true`.
    Host_Budget(std::size_t cap, bool pinned);

    Host_Budget(const Host_Budget&) = delete;
    Host_Budget& operator=(const Host_Budget&) = delete;

    // Charges `b` bytes iff the cap has room for them. Returns `true` iff
    // the charging is successful.
    bool try_charge(std::size_t b);

    // Uncharges `b` bytes.
    void uncharge(std::size_t b);

    // Raises the cap by `b` bytes, unless the cap is pinned at zero.
    void raise_cap(std::size_t b);

    // Marks `b` of the charged bytes as held by free message-chunks still
    // resident.
    void pool(std::size_t b);

    // Marks `b` of the pooled bytes as held by claimants again.
    void unpool(std::size_t b);

    // Returns the current cap.
    auto cap() const { return cap_.load(std::memory_order_relaxed); }

    // Returns the bytes charged.
    auto used() const { return used_.load(std::memory_order_relaxed); }

    // Returns the bytes of the free message-chunks still resident.
    auto pooled() const { return pooled_.load(std::memory_order_relaxed); }

    // Returns whether the cap is pinned at zero.
    bool pinned() const { return pinned_; }
};


inline Host_Budget::Host_Budget(const std::size_t cap, const bool pinned):
      cap_(pinned ? 0 : cap)
    , used_(0)
    , pooled_(0)
    , pinned_(pinned)
{}


inline bool Host_Budget::try_charge(const std::size_t b)
{
    auto used = used_.load(std::memory_order_relaxed);
    do
        if(used + b > cap_.load(std::memory_order_relaxed))
            return false;
    while(!used_.compare_exchange_weak(used, used + b, std::memory_order_relaxed));

    return true;
}


inline void Host_Budget::uncharge(const std::size_t b)
{
    used_.fetch_sub(b, std::memory_order_relaxed);
}


inline void Host_Budget::raise_cap(const std::size_t b)
{
    if(!pinned_)
        cap_.fetch_add(b, std::memory_order_relaxed);
}


inline void Host_Budget::pool(const std::size_t b)
{
    pooled_.fetch_add(b, std::memory_order_relaxed);
}


inline void Host_Budget::unpool(const std::size_t b)
{
    pooled_.fetch_sub(b, std::memory_order_relaxed);
}

}



#endif
