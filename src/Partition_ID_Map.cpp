
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
    , mask_((uint64_t(1) << (bits_ - 1)) - 1)
    , word_c_(word_c(n, bits_))
    , word_(allocate_words(word_c_))
{
    word_[word_c_ - 1] = 0; // The padding: only ever loaded past the last ID's bits.
}


Partition_ID_Map::Partition_ID_Map(const idx_t n):
      n_(n)
    , bits_(1)
    , mask_(0)
    , word_c_(word_c(n, bits_))
    , word_(allocate_words(word_c_))
{
    #pragma omp parallel for schedule(static)
    for(std::size_t w = 0; w < word_c_; ++w)
        word_[w] = 0;
}


uint64_t* Partition_ID_Map::allocate_words(const std::size_t count)
{
    return util::checked_allocate<uint64_t>(
                util::math::block_up<util::PAGE_SIZE>(count * sizeof(uint64_t)) / sizeof(uint64_t), util::PAGE_SIZE);
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
    std::memcpy(word_ + word_of(off, bits_), src, packed_word_c(sz, bits_) * sizeof(uint64_t));
}


void Partition_ID_Map::write_packed(Direct_File& f, const idx_t off, const idx_t sz, const uint32_t bits, uint64_t* const src)
{
    const auto bytes = packed_word_c(sz, bits) * sizeof(uint64_t);
    std::memset(reinterpret_cast<char*>(src) + bytes, 0, Direct_File::block_up(bytes) - bytes);
    f.write_at(src, bytes, word_of(off, bits) * sizeof(uint64_t));
}


void Partition_ID_Map::read(const Direct_File& f, const idx_t off, const idx_t sz)
{
    f.read_at(word_, packed_word_c(sz, bits_) * sizeof(uint64_t), word_of(off, bits_) * sizeof(uint64_t));
}


void Partition_ID_Map::read(const std::string& path, const std::size_t thread_c)
{
    util::file::read_parallel(path, word_, word_c_ * sizeof(uint64_t), thread_c);
}


void Partition_ID_Map::retire(const Partition_ID_Map& flags)
{
    // A group's flags take up a word of `flags`, and its entries `bits_` words here; so the groups' writes are disjoint.
    #pragma omp parallel for schedule(static)
    for(idx_t g = 0; g < (n_ + GROUP_SZ - 1) / GROUP_SZ; ++g)
        for(auto f = flags.word_[g]; f != 0; f &= f - 1)
        {
            const auto bit = flag_bit(g * GROUP_SZ + __builtin_ctzll(f));
            word_[bit / 64] |= uint64_t(1) << (bit % 64);
        }
}


idx_t Partition_ID_Map::retired_count() const
{
    idx_t c = 0;

    #pragma omp parallel for schedule(static) reduction(+ : c)
    for(idx_t i = 0; i < n_; ++i)
        c += retired(i);

    return c;
}


void Partition_ID_Map::prepage()
{
    util::prepage(word_, word_c_);
}

}
