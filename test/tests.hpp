
#ifndef TESTS_HPP
#define TESTS_HPP



#include <cstdint>
#include <cstddef>


namespace gpu_saca::test
{

// Computes the frequency histogram of the binarized `k`-grams of the text `T`
// of size `n` into `H`.
template <typename T_sym_>
void kgram_histogram(const T_sym_* T, std::size_t n, uint64_t* H, std::size_t k);

}



#endif
