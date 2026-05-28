
#ifndef TESTS_HPP
#define TESTS_HPP



#include <cstdint>
#include <cstddef>
#include <vector>


namespace gpu_saca::test
{

// Computes the frequency histogram of the quantized `k`-grams of the text `T`
// of size `n` into `H`.
template <typename T_sym_>
void kgram_histogram(const T_sym_* T, std::size_t n, uint64_t* H, std::size_t k);

// Returns `true` iff the on-GPU frequency histogram `H_d` of quantized
// `k`-grams of the text `T` of size `n` matches to a CPU-side histogram.
template <typename T_sym_>
bool validate_kgram_histogram(const T_sym_* T, std::size_t n, std::size_t k, const uint64_t* H_d);

// Returns the partition sizes defined by the frequency histogram `H` and the
// pivot set `X`.
std::vector<uint64_t> get_partition_sizes(const std::vector<uint64_t>& H, const std::vector<uint32_t>& X);

}



#endif
