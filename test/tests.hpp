
#ifndef TESTS_HPP
#define TESTS_HPP



#include "types.hpp"
#include "Partition.hpp"
#include "Sort_Key.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>


namespace gpu_saca::test
{

// Computes the frequency histogram of the quantized `k`-grams of the text `T`
// of size `n` into `H`.
template <typename T_sym_>
void kgram_histogram(const T_sym_* T, idx_t n, freq_cnt_t* H, std::size_t k);

// Returns `true` iff the frequency histogram `H` of quantized `k`-grams of the
// text `T` of size `n` matches to the histogram computed at the CPU-side as
// reference.
template <typename T_sym_>
bool validate_kgram_histogram(const T_sym_* T, idx_t n, std::size_t k, const freq_cnt_t* H);

// Computes the partition-IDs of the suffixes of the text `T` of size `n` into
// `P_id` based on their prefix `k`-grams, where the partitions are defined by
// the pivot `k`-grams in `X`.
template <typename  T_sym_>
void compute_partition_IDs(const T_sym_* T, idx_t n, p_id_t* P_id, std::size_t k, const std::vector<pivot_t>& X);

// Returns `true` iff the partition-IDs `P_id` of quantized `k`-grams of the
// text `T` of size `n`, defined by the pivots `X`, match to the partition-IDs
// computed at the CPU-side as reference.
template <typename T_sym_>
bool validate_partition_IDs(const T_sym_* T, idx_t n, std::size_t k, const std::vector<pivot_t>& X, const p_id_t* P_id);

// Returns `true` iff the keys in `key`, corresponding to the partition `part`,
// are sorted, and `part` is properly ranked based on the sorted keys.
bool validate_local_ranking(const Partition& part, const Sort_Key* key);

// Returns `true` iff all the partitions in `P` are properly sorted-and-ranked,
// with their keys being packed flat at `key` after the device-side sort.
bool validate_local_rankings(const std::vector<Partition>& P, const Sort_Key* key);

// Returns the symbol `tau` such that roughly half of the symbol occurrences in
// the text `T`, of size `n`, fall into `[min, tau]` and the rest into
// `(tau, max]`.
template <typename T_sym_>
T_sym_ universe_split_threshold(const T_sym_* T, idx_t n);

// Returns `true` iff `SA` is a permutation of `[0, n)`.
bool is_permutation(const idx_t* SA, idx_t n);

// Returns `true` iff `SA` is the suffix array of the text `T` of size `n`.
template <typename T_sym_>
bool is_correct_SA(const idx_t* SA, const T_sym_* T, idx_t n);

}



#endif
