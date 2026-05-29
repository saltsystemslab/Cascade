
#include "tests.hpp"
#include "omp.h"

#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>
#include <cstring>
#include <cassert>


namespace gpu_saca::test
{

template <typename T_sym_>
void kgram_histogram(const T_sym_* const T, const idx_t n, freq_cnt_t* const H, const std::size_t k)
{
    // TODO: repeated code; have better policy for this.
    const auto quantize = [](const T_sym_ x)
    {
        // Maximum element of the lower half: exactly `2^(N - 1)` values at each half.
        constexpr T_sym_ mid = std::numeric_limits<T_sym_>::min() / 2 + std::numeric_limits<T_sym_>::max() / 2;
        return x <= mid ? 0 : 1;
    };

    const auto kgram_count = n;
    const auto hist_sz = 1llu << k;

    std::memset(H, 0, hist_sz * sizeof(freq_cnt_t));    // Clear out the histogram.

    #pragma omp parallel
    {
        std::vector<freq_cnt_t> local_hist(hist_sz, 0);

        #pragma omp for schedule(static)
        for(std::size_t i = 0; i < kgram_count; ++i)    // Thread-local counting.
        {
            kgram_t quant_kgram = 0;    // Quantized k-gram.
            for(std::size_t j = 0; j < k; ++j)
                quant_kgram |= (i + j < n ?
                                    static_cast<kgram_t>(quantize(T[i + j])) << (k - 1 - j) :
                                    0);

            assert(quant_kgram < hist_sz);
            local_hist[quant_kgram]++;
        }

        #pragma omp for schedule(static)    // Global reduction of the local counts.
        for(std::size_t b = 0; b < hist_sz; ++b)
            #pragma omp atomic
            H[b] += local_hist[b];
    }
}


std::vector<freq_cnt_t> get_partition_sizes(const std::vector<freq_cnt_t>& H, const std::vector<pivot_t>& X)
{
    const auto p = X.size() + 1;    // Number of partitions.
    const auto hist_sz = H.size();  // Histogram size.

    std::vector<freq_cnt_t> sz(p, 0);   // Partition sizes.
    p_id_t p_id = 0;    // Which partition we're at right now.
    for(std::size_t bin = 0; bin < hist_sz; ++bin)
    {
        if(p_id < X.size() && bin > X[p_id])    // Moving onto the next partition.
            p_id++;

        sz[p_id] += H[bin]; // The pivots are inclusive upper bounds of their partition.
    }

    return sz;
}


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



// Template instantiations for the required instances.
template void kgram_histogram(const char*, idx_t, freq_cnt_t*, std::size_t);
template void kgram_histogram(const uint8_t*, idx_t, freq_cnt_t*, std::size_t);
template void kgram_histogram(const uint16_t*, idx_t, freq_cnt_t*, std::size_t);
template void kgram_histogram(const uint32_t*, idx_t, freq_cnt_t*, std::size_t);

template void compute_partition_IDs(const char*, idx_t, p_id_t*, std::size_t, const std::vector<pivot_t>&);
template void compute_partition_IDs(const uint8_t*, idx_t, p_id_t*, std::size_t, const std::vector<pivot_t>&);
template void compute_partition_IDs(const uint16_t*, idx_t, p_id_t*, std::size_t, const std::vector<pivot_t>&);
template void compute_partition_IDs(const uint32_t*, idx_t, p_id_t*, std::size_t, const std::vector<pivot_t>&);

}
