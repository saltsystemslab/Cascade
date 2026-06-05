
#ifndef SUFFIX_BUFFER_HPP
#define SUFFIX_BUFFER_HPP



#include "Suffix.hpp"
#include "Spin_Lock.hpp"
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

    using w_buf_t = std::vector<Suffix>;    // Type of worker-local buffers.
    static constexpr std::size_t w_buf_cap = 4 * 1024llu / sizeof(Suffix);  // Worker-local buffers overflow at size: 4KB.

    std::vector<util::Padded<w_buf_t>> w_buf;   // Worker-local buffers.
    Suffix* const sink_; // Sink to flush the worker-local buffers.
    std::size_t flushed_;   // Number of suffixes flushed to the sink.

    util::Spin_Lock lock;   // Mutually-exclusive access-lock to the sink.

public:

    // Constructs a buffer that overflows to `sink`.
    Suffix_Buffer(Suffix* sink);

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

    // Returns the number of suffixes flushed to the sink.
    auto flushed() const { return flushed_; }
};


inline void Suffix_Buffer::add(const idx_t suf, const idx_t p_key, const idx_t s_key)
{
    const auto w_id = omp_get_thread_num();
    auto& buf = w_buf[w_id].unwrap();
    buf.emplace_back(suf, p_key, s_key);
    assert(buf.size() <= w_buf_cap);

    if(buf.size() == w_buf_cap)
    {
        lock.lock();
        std::memcpy(reinterpret_cast<void*>(sink_ + flushed_), reinterpret_cast<void*>(buf.data()), w_buf_cap * sizeof(Suffix));
        flushed_ += w_buf_cap;
        lock.unlock();

        buf.clear();
    }
}

}



#endif
