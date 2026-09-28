
#ifndef SUFFIX_BUFFER_HPP
#define SUFFIX_BUFFER_HPP



#include "Direct_File.hpp"
#include "types.hpp"
#include "utility.hpp"

#include "omp.h"

#include <cstddef>
#include <atomic>
#include <vector>


namespace gpu_saca
{

class Partition_Backend;


// A buffer of suffixes, backed by a partition's storage in the backend.
class Suffix_Buffer
{
private:

    static constexpr std::size_t w_buf_cap = 32 * 1024llu / sizeof(idx_t);  // Worker-local buffers flush at reaching 32 KBs worth of suffixes.

    static_assert((w_buf_cap * sizeof(idx_t)) % Direct_File::DISK_BLOCK == 0);

    Partition_Backend* backend_;    // Backend sinking the suffixes.
    p_id_t p_id_;   // ID of the partition the suffixes belong to.

    idx_t* w_suf_base_; // Flattened collection of worker-local suffix buffers.
    std::vector<util::Padded<idx_t>> w_fill_;   // Fill count of the worker-local buffers.

    std::atomic<idx_t> flushed_;    // Number of suffixes reserved / written to the sink files.
    const idx_t worker_c_;  // Worker count.


    auto w_suf(const idx_t w) { return w_suf_base_ + w * w_buf_cap; }   // Returns the `w`'th worker's suffix-buffer.

    // Sinks `sz`-many suffixes from `suf`, at the partition-offset `off`.
    void sink(idx_t off, idx_t sz, const idx_t* suf);

public:

    // Constructs a buffer that overflows suffixes to the `p_id`'th partition's
    // storage in the backend `backend`.
    Suffix_Buffer(Partition_Backend& backend, p_id_t p_id);

    Suffix_Buffer(const Suffix_Buffer&) = delete;
    Suffix_Buffer& operator=(const Suffix_Buffer&) = delete;
    Suffix_Buffer(Suffix_Buffer&& rhs);
    Suffix_Buffer& operator=(Suffix_Buffer&&) = delete;

    ~Suffix_Buffer();

    // Returns the total memory of the worker-local scatter-buffers for `p`-many
    // partitions and `worker_c`-many workers.
    static std::size_t scatter_buffer_memory(std::size_t worker_c, std::size_t p);

    // Adds a suffix with index `suf`.
    void add(idx_t suf);

    // Flushes the buffer to the sinks. There must be no concurrent pending add
    // invocations.
    void flush();

    // Returns the number of suffixes flushed to the sinks.
    auto flushed() const { return flushed_.load(std::memory_order_relaxed); }
};


inline void Suffix_Buffer::add(const idx_t suf)
{
    const auto w_id = omp_get_thread_num();
    auto& fill = w_fill_[w_id].unwrap();
    auto suf_buf = w_suf(w_id);

    suf_buf[fill] = suf;
    if(++fill == w_buf_cap)
    {
        const auto off = flushed_.fetch_add(w_buf_cap, std::memory_order_relaxed);
        sink(off, w_buf_cap, suf_buf);
        fill = 0;
    }
}

}



#endif
