
#include "tests.hpp"

#include "omp.h"

#include <limits>
#include <cstring>
#include <algorithm>


namespace gpu_saca::test
{

/*
template <typename  T_sym_>
void compute_partition_IDs(const T_sym_* T, idx_t n, p_id_t* P_id, std::size_t k, const std::vector<pivot_t>& X)
{
    // TODO: repeated code; have better policy for this.
    const auto quantize = [](const T_sym_ x)
    {
        // Maximum element of the lower half: exactly `2^(N - 1)` values at each half.
        constexpr T_sym_ mid = std::numeric_limits<T_sym_>::min() / 2 + std::numeric_limits<T_sym_>::max() / 2;
        return x <= mid ? 0 : 1;
    };

    const auto part_count = X.size() + 1;
    #pragma omp parallel for schedule(static)
    for(std::size_t i = 0; i < n; ++i)
    {
        kgram_t quant_kgram = 0;    // Quantized k-gram.
        for(std::size_t j = 0; j < k; ++j)
            quant_kgram |= (i + j < n ?
                                static_cast<kgram_t>(quantize(T[i + j])) << (k - 1 - j) :
                                0);

        // Binary search for `quant_kgram` into the pivots for the smallest `p_id` such that `quant_kgram <= pivots[p_id]`.
        p_id_t low = 0, high = part_count - 1;  // The solution lies in the range `[low, high]`.
        while(low < high)
        {
            const auto mid = low + (high - low) / 2;
            if(quant_kgram <= X[mid])   // The solution lies in the range `[low, mid]`.
                high = mid;
            else    // It lies in the range `[mid + 1, high]`.
                low = mid + 1;
        }

        P_id[i] = low;
    }
}


template <typename T_sym_>
bool validate_partition_IDs(const T_sym_* const T, const idx_t n, const std::size_t k, const std::vector<pivot_t>& X, const p_id_t* const P_id)
{
    std::vector<p_id_t> cpu_P_id(n);
    compute_partition_IDs(T, n, cpu_P_id.data(), k, X);
    return std::memcmp(P_id, cpu_P_id.data(), n * sizeof(p_id_t)) == 0;
}
*/


template <typename T_sym_>
T_sym_ universe_split_threshold(const T_sym_* T, const std::size_t n)
{
    static_assert(sizeof(T_sym_) <= 2,
        "Symbol-frequency counting requires a table of size the symbol-universe; only practical for <= 16-bit alphabets.");

    constexpr std::size_t universe_sz = 1llu << (8 * sizeof(T_sym_));

    // Per-thread local frequency tables.
    const auto n_threads = omp_get_max_threads();
    std::vector<std::vector<idx_t>> local_freq(n_threads, std::vector<idx_t>(universe_sz, 0));

    #pragma omp parallel
    {
        const auto t_id = omp_get_thread_num();
        auto& freq = local_freq[t_id];

        #pragma omp for schedule(static)
        for(idx_t i = 0; i < n; ++i)
            freq[T[i]]++;
    }

    // Reduce the per-thread tables into a single frequency table.
    std::vector<idx_t> freq(universe_sz, 0);
    #pragma omp parallel for schedule(static)
    for(std::size_t v = 0; v < universe_sz; ++v)
    {
        idx_t sum = 0;
        for(int t = 0; t < n_threads; ++t)
            sum += local_freq[t][v];

        freq[v] = sum;
    }

    // Walk the cumulative distribution to find the smallest value `tau` such
    // that the cumulative mass up to and including `tau` reaches half the total.
    const idx_t half = n / 2;
    idx_t cum_sum = 0;
    T_sym_ tau = 0;
    for(std::size_t v = 0; v < universe_sz; ++v)
    {
        cum_sum += freq[v];
        if(cum_sum >= half)
        {
            tau = static_cast<T_sym_>(v);
            break;
        }
    }

    return tau;
}



/*
// Template instantiations for the required instances.
template void compute_partition_IDs(const char*, idx_t, p_id_t*, std::size_t, const std::vector<pivot_t>&);
template void compute_partition_IDs(const uint8_t*, idx_t, p_id_t*, std::size_t, const std::vector<pivot_t>&);
template void compute_partition_IDs(const uint16_t*, idx_t, p_id_t*, std::size_t, const std::vector<pivot_t>&);
template void compute_partition_IDs(const uint32_t*, idx_t, p_id_t*, std::size_t, const std::vector<pivot_t>&);

template bool validate_partition_IDs(const char*, idx_t, std::size_t, const std::vector<pivot_t>&, const p_id_t*);
template bool validate_partition_IDs(const uint8_t*, idx_t, std::size_t, const std::vector<pivot_t>&, const p_id_t*);
template bool validate_partition_IDs(const uint16_t*, idx_t, std::size_t, const std::vector<pivot_t>&, const p_id_t*);
template bool validate_partition_IDs(const uint32_t*, idx_t, std::size_t, const std::vector<pivot_t>&, const p_id_t*);
*/

template char       universe_split_threshold(const char*, std::size_t);
template uint8_t    universe_split_threshold(const uint8_t*, std::size_t);
template uint16_t   universe_split_threshold(const uint16_t*, std::size_t);

}
