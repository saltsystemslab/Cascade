
#include "tests.hpp"
#include "omp.h"

#include <limits>
#include <vector>
#include <cstring>
#include <algorithm>
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


template <typename T_sym_>
bool validate_kgram_histogram(const T_sym_* const T_h, const idx_t n, const std::size_t k, const freq_cnt_t* const H)
{
    const std::size_t hist_sz = 1llu << k;

    std::vector<freq_cnt_t> cpu_hist(hist_sz);
    kgram_histogram(T_h, n, cpu_hist.data(), k);
    return std::memcmp(H, cpu_hist.data(), hist_sz * sizeof(freq_cnt_t)) == 0;
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



// Template instantiations for the required instances.
template void kgram_histogram(const char*, idx_t, freq_cnt_t*, std::size_t);
template void kgram_histogram(const uint8_t*, idx_t, freq_cnt_t*, std::size_t);
template void kgram_histogram(const uint16_t*, idx_t, freq_cnt_t*, std::size_t);
template void kgram_histogram(const uint32_t*, idx_t, freq_cnt_t*, std::size_t);

template bool validate_kgram_histogram(const char*, idx_t, std::size_t, const freq_cnt_t*);
template bool validate_kgram_histogram(const uint8_t*, idx_t, std::size_t, const freq_cnt_t*);
template bool validate_kgram_histogram(const uint16_t*, idx_t, std::size_t, const freq_cnt_t*);
template bool validate_kgram_histogram(const uint32_t*, idx_t, std::size_t, const freq_cnt_t*);

}
