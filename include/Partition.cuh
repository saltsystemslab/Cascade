
#ifndef PARTITION_CUH
#define PARTITION_CUH



#include "types.hpp"
#include "Suffix.hpp"
#include "Spin_Lock.hpp"
#include "utility.hpp"

#include <cstddef>
#include <vector>
#include <cstring>
#include <cassert>


namespace gpu_saca
{

// A partition of suffixes.
class Partition
{
private:

    Suffix* const suf;  // The suffixes (along with sort-keys) in this partition.
    const idx_t cap;    // Capacity of the partition.
    idx_t sz;   // Size of the partition.

    util::Spin_Lock lock;   // Lock to update the partition in concurrent settings.

    using w_buf_t = std::vector<Suffix>;    // Type of worker-local buffers.
    std::vector<util::Padded<w_buf_t>> w_buf;   // Worker-local buffers.

    static constexpr std::size_t w_buf_cap = 4 * 1024 * 1024llu / sizeof(Suffix);   // Worker-local buffers overflow to the partitions at size: 4KB.

public:

    // Allocates this partition for `cap` many suffixes.
    Partition(idx_t cap);

    ~Partition();

    // Adds a suffix to the partition with index `suf`, primary key `p_key`, and
    // secondary key `s_key`, from the `w_id`'th worker.
    void add(idx_t suf, idx_t p_key, idx_t s_key, idx_t w_id);
};


inline void Partition::add(const idx_t suf, const idx_t p_key, const idx_t s_key, const idx_t w_id)
{
    auto& buf = w_buf[w_id].unwrap();
    buf.emplace_back(suf, p_key, s_key);
    assert(buf.size() <= w_buf_cap);

    if(buf.size() == w_buf_cap)
    {
        assert(sz + w_buf <= cap);

        lock.lock();
        std::memcpy(reinterpret_cast<void*>(suf + sz), reinterpret_cast<void*>(buf.data()), w_buf_cap * sizeof(Suffix));
        sz += w_buf_cap;
        lock.unlock();

        buf.clear();
    }
}

}



#endif