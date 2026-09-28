
#ifndef CPU_REFERENCE_HPP
#define CPU_REFERENCE_HPP



#include "cross-compile.hpp"
#include "types.hpp"
#include "sa40.hpp"
#include "query-file.hpp"

#include <cstdint>
#include <random>
#include <string>


// Reference semantics of the query engine, written symbol-by-symbol and
// independent of the packed-key GPU path: the correctness oracle, the interval
// certificate, and the suffix-array order spot-check.
namespace gpu_saca::ref
{

// Returns the order of the suffix at `pos` of the text `T[0, n)`, truncated to
// `m` symbols, against the pattern `P[0, m)`: `-1`, `0`, or `1`. A suffix that
// ends before `m` symbols while still matching counts as smaller.
template <typename T_sym_>
CUDA_SPECIFIER
inline int cmp_suffix_pattern(const T_sym_* const T, const idx_t n, const idx_t pos, const T_sym_* const P, const idx_t m)
{
    const idx_t len = (n - pos < m ? n - pos : m);
    for(idx_t i = 0; i < len; ++i)
        if(T[pos + i] != P[i])
            return T[pos + i] < P[i] ? -1 : 1;

    return len < m ? -1 : 0;
}


// Returns the SA-interval of the pattern `P[0, m)` over the text `T[0, n)` with
// the 40-bit suffix array `SA`, through plain binary searches over `[0, n)`.
// The empty pattern's interval is `[0, n)`.
template <typename T_sym_>
inline SA_Interval find_one(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const T_sym_* const P, const idx_t m)
{
    if(m == 0)
        return {0, n};

    // `lo` is the first rank whose truncated suffix is not smaller than `P`.
    idx_t low = 0, high = n;
    while(low < high)
    {
        const idx_t mid = low + (high - low) / 2;
        if(cmp_suffix_pattern(T, n, sa40::load(SA, mid), P, m) < 0)
            low = mid + 1;
        else
            high = mid;
    }

    const idx_t lo = low;

    // `hi` is the first rank whose truncated suffix is greater than `P`.
    high = n;
    while(low < high)
    {
        const idx_t mid = low + (high - low) / 2;
        if(cmp_suffix_pattern(T, n, sa40::load(SA, mid), P, m) <= 0)
            low = mid + 1;
        else
            high = mid;
    }

    return {lo, low};
}


// Returns `true` iff `I` is exactly the SA-interval of the pattern `P[0, m)`
// over the text `T[0, n)`, provided that the suffix array `SA` is sorted: it
// checks the interval's boundary suffixes against `P`, in `O(m)` time.
template <typename T_sym_>
CUDA_SPECIFIER
inline bool certify(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const T_sym_* const P, const idx_t m, const SA_Interval& I)
{
    if(I.lo > I.hi || I.hi > n)
        return false;

    if(m == 0)
        return I.lo == 0 && I.hi == n;

    const auto cmp = [&](const idx_t r) { return cmp_suffix_pattern(T, n, sa40::load(SA, r), P, m); };

    if(I.lo > 0 && cmp(I.lo - 1) >= 0)  // The suffix right before must be smaller.
        return false;

    if(I.lo < I.hi && (cmp(I.lo) != 0 || cmp(I.hi - 1) != 0))   // The interval's ends must match.
        return false;

    if(I.hi < n && cmp(I.hi) <= 0)  // The suffix right after must be greater.
        return false;

    return true;
}


// Returns the order of the suffixes at `i` and `j` of the text `T[0, n)`,
// comparing at most `max_cmp` symbols: `-1`, `1`, or `0` if undecided within
// `max_cmp` symbols. An exhausted suffix is smaller than any extension of it.
template <typename T_sym_>
CUDA_SPECIFIER
inline int cmp_suffixes(const T_sym_* const T, const idx_t n, const idx_t i, const idx_t j, const idx_t max_cmp)
{
    const idx_t len_i = n - i, len_j = n - j;
    const idx_t len = (len_i < len_j ? len_i : len_j);
    const idx_t lim = (len < max_cmp ? len : max_cmp);

    for(idx_t k = 0; k < lim; ++k)
        if(T[i + k] != T[j + k])
            return T[i + k] < T[j + k] ? -1 : 1;

    if(lim == max_cmp && len > max_cmp)
        return 0;

    return len_i < len_j ? -1 : 1;
}


// Returns `true` iff `samples` random adjacent pairs of the suffix array,
// accessed through `sa(r)`, over the text `T[0, n)`, are in the numeric symbol
// order with shorter suffixes first, comparing at most `max_cmp` symbols per
// pair; the first and the last pair are always checked, and every pair is
// checked if `samples >= n - 1`. On failure, `err` explains the first
// offending pair. Undecided pairs count as consistent.
template <typename T_sym_, typename F_sa_>
bool spot_check_sa_order(const T_sym_* const T, const idx_t n, F_sa_ sa, const idx_t samples, const idx_t max_cmp, const uint64_t seed, std::string& err)
{
    if(n < 2)
        return true;

    const bool exhaustive = (samples >= n - 1);
    const idx_t checks = (exhaustive ? n - 1 : samples + 2);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<idx_t> dist(0, n - 2);

    for(idx_t k = 0; k < checks; ++k)
    {
        const idx_t r = (exhaustive ? k : (k == 0 ? 0 : (k == 1 ? n - 2 : dist(rng))));
        const idx_t i = sa(r), j = sa(r + 1);
        if(i >= n || j >= n)
        {
            err = "suffix-array entry out of range at rank " + std::to_string(i >= n ? r : r + 1);
            return false;
        }

        if(i == j)  // Undecidable within `max_cmp` symbols if long, so checked apart.
        {
            err = "suffix-array entry " + std::to_string(i) + " repeats at ranks " + std::to_string(r) + " and " + std::to_string(r + 1);
            return false;
        }

        if(cmp_suffixes(T, n, i, j, max_cmp) > 0)
        {
            err = "suffix-array order violated between ranks " + std::to_string(r) + " and " + std::to_string(r + 1)
                + " (suffixes " + std::to_string(i) + ", " + std::to_string(j) + "); expected numeric order with shorter suffixes first";
            return false;
        }
    }

    return true;
}

}



#endif
