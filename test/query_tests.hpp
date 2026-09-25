
#ifndef QUERY_TESTS_HPP
#define QUERY_TESTS_HPP



#include "types.hpp"
#include "query-file.hpp"

#include <cstdint>
#include <string>
#include <vector>


namespace gpu_saca::test
{

// Returns `true` iff each interval `I[i]` is exactly the SA-interval of the
// query `sym[off[i], off[i + 1])` over the text `T[0, n)` with the 40-bit
// suffix array `SA`, through the `O(m)` certificate of `ref::certify`. Assumes
// that `SA` is sorted. On failure, `err` explains the first offending query.
template <typename T_sym_>
bool validate_intervals(const T_sym_* T, idx_t n, const uint8_t* SA, const uint64_t* off, const T_sym_* sym, idx_t q, const SA_Interval* I, std::string& err);

// Returns the 40-bit suffix array of the text `T`, in the numeric symbol order
// with shorter suffixes first, built by comparison-sorting. Applicable at small
// scale.
template <typename T_sym_>
std::vector<uint8_t> build_sa40(const std::vector<T_sym_>& T);

// Returns the 40-bit suffix array of the text `T` in the raw byte order of its
// little-endian byte-stream, i.e. infini-gram's order. For negative controls.
template <typename T_sym_>
std::vector<uint8_t> build_sa40_byte_order(const std::vector<T_sym_>& T);

// Returns the SA-interval of the pattern `P[0, m)` over the text `T[0, n)`
// computed without any suffix array: `lo` counts the suffixes smaller than `P`,
// `hi - lo` those having `P` as a prefix. `O(n m)`; for small fixtures.
template <typename T_sym_>
SA_Interval brute_find(const T_sym_* T, idx_t n, const T_sym_* P, idx_t m);

}



#endif
