
#ifndef SUFFIX_BUFFER_HPP
#define SUFFIX_BUFFER_HPP



#include "Prefix_Key.hpp"
#include "Direct_File.hpp"
#include "types.hpp"
#include "utility.hpp"

#include "omp.h"

#include <cstddef>
#include <atomic>
#include <vector>


namespace gpu_saca
{

// A buffer of suffixes along with their associated prefix-keys, backed by
// direct-files.
class Suffix_Buffer
{
private:

    static constexpr std::size_t w_buf_cap = 256 * 1024llu / sizeof(idx_t); // Worker-local buffers overflow at reaching 256 KB worth of suffixes.

    static_assert((w_buf_cap * sizeof(idx_t)) % Direct_File::DISK_BLOCK == 0);
    static_assert((w_buf_cap * sizeof(Prefix_Key)) % Direct_File::DISK_BLOCK == 0);

    idx_t* suf_mem_;    // Resident sink for the suffixes.
    Prefix_Key* key_mem_;   // Resident sink for the prefix-keys.
    Direct_File* suf_file_; // Spill-file sink for the suffixes.
    Direct_File* key_file_; // Spill-file sink for the prefix-keys.

    idx_t* w_suf_base_;         // Flattened collection of worker-local suffix buffers.
    Prefix_Key* w_key_base_;    // Flattened collection of worker-local prefix-key-buffers.
    std::vector<util::Padded<idx_t>> w_fill_;   // Fill count of the worker-local buffers.

    std::atomic<idx_t> flushed_;    // Number of suffixes reserved / written to the sink files.
    const idx_t worker_c_;  // Worker count.


    auto w_suf(const idx_t w) { return w_suf_base_ + w * w_buf_cap; }   // Returns the `w`'th worker's suffix-buffer.
    auto w_key(const idx_t w) { return w_key_base_ + w * w_buf_cap; }   // Returns the `w`'th worker's prefix-key-buffer.

    // Sinks `sz`-many suffixes and their prefix-keys, from `suf` and `key`, at
    // the partition-offset `off`.
    void sink(idx_t off, idx_t sz, const idx_t* suf, const Prefix_Key* key);

public:

    // Constructs a buffer that overflows suffixes and their prefix-keys to
    // the partition's sinks: the resident allocations `suf_mem` and `key_mem`
    // when these are non-null, and the spill-files `suf_file` and `key_file`
    // otherwise.
    Suffix_Buffer(idx_t* suf_mem, Prefix_Key* key_mem, Direct_File* suf_file, Direct_File* key_file);

    Suffix_Buffer(const Suffix_Buffer&) = delete;
    Suffix_Buffer& operator=(const Suffix_Buffer&) = delete;
    Suffix_Buffer(Suffix_Buffer&& rhs);
    Suffix_Buffer& operator=(Suffix_Buffer&&) = delete;

    ~Suffix_Buffer();

    // Returns the total memory of the worker-local scatter-buffers for `p`-many
    // partitions and `worker_c`-many workers.
    static std::size_t scatter_buffer_memory(std::size_t worker_c, std::size_t p);

    // Adds a suffix with index `suf` and prefix-key `p_key`.
    void add(idx_t suf, const Prefix_Key& p_key);

    // Flushes the buffer to the sinks. There must be no concurrent pending add
    // invocations.
    void flush();

    // Returns the number of suffixes flushed to the sinks.
    auto flushed() const { return flushed_.load(std::memory_order_relaxed); }
};


inline void Suffix_Buffer::add(const idx_t suf, const Prefix_Key& p_key)
{
    const auto w_id = omp_get_thread_num();
    auto& fill = w_fill_[w_id].unwrap();
    auto suf_buf = w_suf(w_id);
    auto key_buf = w_key(w_id);

    suf_buf[fill] = suf, key_buf[fill] = p_key;
    if(++fill == w_buf_cap)
    {
        const auto off = flushed_.fetch_add(w_buf_cap, std::memory_order_relaxed);
        sink(off, w_buf_cap, suf_buf, key_buf);
        fill = 0;
    }
}

}



#endif
