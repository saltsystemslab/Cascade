
#ifndef TESTS_HPP
#define TESTS_HPP



#include "types.hpp"
#include "Partition.hpp"
#include "Prefix_Key.cuh"
#include "Sort_Key.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>


namespace gpu_saca::test
{

// Returns `true` iff the partition-IDs `P_id` of the text `T` (of size `n`),
// defined by the pivots `X`, match to the partition-IDs computed at the CPU-
// side as reference.
template <typename T_sym_>
bool validate_partition_IDs(const T_sym_* T, idx_t n, const std::vector<Prefix_Key>& X, const p_id_t* P_id);

// Returns `true` iff the keys in `key` and their corresponding suffixes in
// `suf`, corresponding to the partition `part`, are sorted, and `part` is
// properly ranked based on the sorted keys.
bool validate_local_ranking(const Partition& part, const Sort_Key* key, const idx_t* suf);

// Returns `true` iff all the partitions in `P` are properly sorted-and-ranked,
// with their keys packed flat at `key` and the corresponding suffixes packed
// flat at `suf`, after the device-side sort.
bool validate_local_rankings(const std::vector<Partition>& P, const Sort_Key* key, const idx_t* suf);

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
