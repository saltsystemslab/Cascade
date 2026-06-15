
#include "Suffix_Buffer.hpp"

#include "omp.h"

#include <utility>


namespace gpu_saca
{

Suffix_Buffer::Suffix_Buffer(idx_t* const suf_sink, Sort_Key* const key_sink):
      w_suf_buf(omp_get_max_threads())
    , w_key_buf(omp_get_max_threads())
    , suf_sink_(suf_sink)
    , key_sink_(key_sink)
    , flushed_(0)
{
    for(std::size_t w_id = 0; w_id < omp_get_max_threads(); ++w_id)
        w_suf_buf[w_id].unwrap().reserve(w_buf_cap),
        w_key_buf[w_id].unwrap().reserve(w_buf_cap);
}


Suffix_Buffer::Suffix_Buffer(Suffix_Buffer&& rhs):
      w_suf_buf(std::move(rhs.w_suf_buf))
    , w_key_buf(std::move(rhs.w_key_buf))
    , suf_sink_(rhs.suf_sink_)
    , key_sink_(rhs.key_sink_)
    , flushed_(rhs.flushed_)
{}


void Suffix_Buffer::flush()
{
    for(std::size_t w_id = 0; w_id < w_suf_buf.size(); ++w_id)
    {
        auto& suf_buf = w_suf_buf[w_id].unwrap();
        auto& key_buf = w_key_buf[w_id].unwrap();

        const auto sz = suf_buf.size();
        assert(key_buf.size() == sz);
        std::memcpy(reinterpret_cast<void*>(suf_sink_ + flushed_), reinterpret_cast<void*>(suf_buf.data()), sz * sizeof(idx_t));
        std::memcpy(reinterpret_cast<void*>(key_sink_ + flushed_), reinterpret_cast<void*>(key_buf.data()), sz * sizeof(Sort_Key));
        flushed_ += sz;

        suf_buf.clear();
        key_buf.clear();
    }
}

}
