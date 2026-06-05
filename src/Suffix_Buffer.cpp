
#include "Suffix_Buffer.hpp"

#include "omp.h"
#include <utility>


namespace gpu_saca
{

Suffix_Buffer::Suffix_Buffer(Suffix* const sink):
      w_buf(omp_get_max_threads())
    , sink(sink)
    , flushed(0)
{
    for(auto& w : w_buf)
        w.unwrap().reserve(w_buf_cap);
}


Suffix_Buffer::Suffix_Buffer(Suffix_Buffer&& rhs):
      w_buf(std::move(rhs.w_buf))
    , sink(rhs.sink)
    , flushed(rhs.flushed)
{}


void Suffix_Buffer::flush()
{
    for(auto& w : w_buf)
    {
        auto& buf = w.unwrap();
        std::memcpy(reinterpret_cast<void*>(sink + flushed), reinterpret_cast<void*>(buf.data()), buf.size() * sizeof(Suffix));
        flushed += buf.size();

        buf.clear();
    }
}

}
