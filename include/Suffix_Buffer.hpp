
#ifndef SUFFIX_BUFFER_HPP
#define SUFFIX_BUFFER_HPP



#include "Sort_Key.hpp"
#include "Spin_Lock.hpp"
#include "types.hpp"
#include "utility.hpp"

#include "omp.h"

#include <cstddef>
#include <vector>
#include <cstring>
#include <cassert>


namespace gpu_saca
{

// A buffer of suffixes, backed by a provided sink. Additions are thread-safe.
class Suffix_Buffer
{
private:

    using suf_buf_t = std::vector<idx_t>;       // Type of worker-local suffix buffers.
    using key_buf_t = std::vector<Sort_Key>;    // Type of worker-local sort-key (of suffixes) buffers.

    static constexpr std::size_t w_buf_cap = 4 * 1024llu / (sizeof(idx_t) + sizeof(Sort_Key));  // Worker-local buffers overflow at total size: 4KB.

    std::vector<util::Padded<suf_buf_t>> w_suf_buf; // Worker-local suffix buffers.
    std::vector<util::Padded<key_buf_t>> w_key_buf; // Worker-local sort-key buffers.

    idx_t* const suf_sink_;     // Sink to flush the worker-local suffix buffers.
    Sort_Key* const key_sink_;  // Sink to flush the worker-local sort-key buffers.

    std::size_t flushed_;   // Number of suffixes flushed to the sink.

    util::Spin_Lock lock;   // Mutually-exclusive access-lock to the sink.

public:

    // Constructs a buffer that overflows suffixes to `suf_sink` and their keys
    // to `key_sink`.
    Suffix_Buffer(idx_t* suf_sink, Sort_Key* key_sink);

    Suffix_Buffer(const Suffix_Buffer&) = delete;
    Suffix_Buffer& operator=(const Suffix_Buffer&) = delete;
    Suffix_Buffer(Suffix_Buffer&& rhs);
    Suffix_Buffer& operator=(Suffix_Buffer&&) = delete;

    // Adds a suffix to the buffer with index `suf`, primary key `p_key`, and
    // secondary key `s_key`.
    void add(idx_t suf, idx_t p_key, idx_t s_key);

    // Flushes the buffer to the sink. There must be no concurrent pending add
    // invocations.
    void flush();

    // Returns the number of suffixes flushed to the sinks.
    auto flushed() const { return flushed_; }
};


inline void Suffix_Buffer::add(const idx_t suf, const idx_t p_key, const idx_t s_key)
{
    const auto w_id = omp_get_thread_num();
    auto& suf_buf = w_suf_buf[w_id].unwrap();
    auto& key_buf = w_key_buf[w_id].unwrap();

    suf_buf.push_back(suf);
    key_buf.push_back({p_key, s_key});
    assert(suf_buf.size() <= w_buf_cap), assert(key_buf.size() <= w_buf_cap), assert(suf_buf.size() == key_buf.size());

    if(suf_buf.size() == w_buf_cap)
    {
        lock.lock();
        std::memcpy(reinterpret_cast<void*>(suf_sink_ + flushed_), reinterpret_cast<void*>(suf_buf.data()), w_buf_cap * sizeof(idx_t));
        std::memcpy(reinterpret_cast<void*>(key_sink_ + flushed_), reinterpret_cast<void*>(key_buf.data()), w_buf_cap * sizeof(Sort_Key));
        flushed_ += w_buf_cap;
        lock.unlock();

        suf_buf.clear();
        key_buf.clear();
    }
}

}



#endif
