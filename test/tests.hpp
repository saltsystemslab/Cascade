
#ifndef TESTS_HPP
#define TESTS_HPP



#include "types.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>


namespace gpu_saca::test
{

// Computes the frequency histogram of the quantized `k`-grams of the text `T`
// of size `n` into `H`.
template <typename T_sym_>
void kgram_histogram(const T_sym_* T, idx_t n, freq_cnt_t* H, std::size_t k);

// Returns `true` iff the on-GPU frequency histogram `H_d` of quantized
// `k`-grams of the text `T` of size `n` matches to a CPU-side histogram.
template <typename T_sym_>
bool validate_kgram_histogram(const T_sym_* T, idx_t n, std::size_t k, const freq_cnt_t* H_d);

// Returns the partition sizes defined by the frequency histogram `H` and the
// pivot set `X`.
std::vector<freq_cnt_t> get_partition_sizes(const std::vector<freq_cnt_t>& H, const std::vector<pivot_t>& X);

// Computes the partition-IDs of the suffixes of the text `T` of size `n` into
// `P_id` based on their prefix `k`-grams, where the partitions are defined by
// the pivot `k`-grams in `X`.
template <typename  T_sym_>
void compute_partition_IDs(const T_sym_* T, idx_t n, p_id_t* P_id, std::size_t k, const std::vector<pivot_t>& X);

// Returns `true` iff the on-GPU partition-IDs `P_id_d` of quantized `k`-grams
// of the text `T` of size `n` matches to the partition IDs computed at the CPU-
// side.
template <typename T_sym_>
bool validate_partition_IDs(const T_sym_* T, idx_t n, std::size_t k, const std::vector<pivot_t>& X, const p_id_t* P_id_d);

}



#endif
