
#ifndef SAMPLE_TABLE_HPP
#define SAMPLE_TABLE_HPP



#include "types.hpp"
#include "Packed_Prefix.hpp"
#include "Suffix_Index.hpp"

#include <cstddef>
#include <cstdint>
#include <string>


namespace gpu_saca
{

// The sample table of a query index: the packed `K`-symbol prefix of every
// `s`-th suffix in suffix-array order, zero-padded past the text's end, in
// device memory. Its keys are non-decreasing, so that a query's rank among them
// brackets its SA-interval's boundaries within `s` ranks.
class Sample_Table
{
private:

    void* keys_ = nullptr;  // Device address of the keys.
    idx_t size_ = 0;    // Count of samples, `ceil(n / stride)`.
    idx_t stride_ = 0;
    uint32_t key_words_ = 0;
    idx_t key_symbols_ = 0; // `K`: symbols per key.
    double t_build_ = -1;   // Seconds to build; `-1` if unbuilt.

public:

    Sample_Table() = default;
    ~Sample_Table() { release(); }
    Sample_Table(const Sample_Table&) = delete;
    Sample_Table& operator=(const Sample_Table&) = delete;
    Sample_Table(Sample_Table&& o) noexcept;
    Sample_Table& operator=(Sample_Table&& o) noexcept;

    idx_t size() const { return size_; }
    idx_t stride() const { return stride_; }
    uint32_t key_words() const { return key_words_; }
    idx_t key_symbols() const { return key_symbols_; }
    std::size_t bytes() const { return size_ * key_words_ * sizeof(uint64_t); }
    double t_build() const { return t_build_; }

    // Builds the table of the suffixes at ranks `0, stride, 2 stride, ...` of the
    // index `idx`, with keys of `key_words` 64-bit words: 1, 2, or 4. The host
    // builds it if both arrays are in host memory, and the device otherwise.
    // Returns `false` with a reason in `err` on failure, leaving the table empty.
    bool build(const Suffix_Index& idx, idx_t stride, uint32_t key_words, std::string& err);

    // Frees the table.
    void release();

    // Returns `true` iff a table is built.
    bool built() const { return keys_ != nullptr; }

    // Returns the device address of the keys; `W_` must equal `key_words()`.
    template <idx_t W_>
    const Packed_Prefix<W_>* keys() const { return static_cast<const Packed_Prefix<W_>*>(keys_); }
};

}



#endif
