
#include "Partition_ID_Map.hpp"

#include <cstring>


namespace gpu_saca
{

Partition_ID_Map::Partition_ID_Map():
      n_(0)
    , bits_(0)
    , mask_(0)
    , word_c_(0)
    , word_(nullptr)
{}


Partition_ID_Map::Partition_ID_Map(const idx_t n, const p_id_t p):
      n_(n)
    , bits_(bits_for(p))
    , mask_((uint64_t(1) << bits_) - 1)
    , word_c_(word_c(n, bits_))
    , word_(util::checked_allocate<uint64_t>(word_c_))
{
    word_[word_c_ - 1] = 0; // The padding: only ever loaded past the last ID's bits.
}


Partition_ID_Map::Partition_ID_Map(Partition_ID_Map&& rhs):
      word_(nullptr)
{
    this->operator=(std::move(rhs));
}


Partition_ID_Map& Partition_ID_Map::operator=(Partition_ID_Map&& rhs)
{
    if(this != &rhs)
    {
        util::deallocate(word_);

        n_ = rhs.n_, bits_ = rhs.bits_, mask_ = rhs.mask_, word_c_ = rhs.word_c_, word_ = rhs.word_;
        rhs.n_ = 0, rhs.bits_ = 0, rhs.mask_ = 0, rhs.word_c_ = 0, rhs.word_ = nullptr;
    }

    return *this;
}


Partition_ID_Map::~Partition_ID_Map()
{
    util::deallocate(word_);
}


void Partition_ID_Map::store_packed(const idx_t off, const idx_t sz, const uint64_t* const src)
{
    std::memcpy(word_ + (off / GROUP_SZ) * bits_, src, packed_word_c(sz, bits_) * sizeof(uint64_t));
}


void Partition_ID_Map::prepage()
{
    util::prepage(word_, word_c_);
}

}
