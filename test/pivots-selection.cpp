
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
void kgram_histogram(const T_sym_* const T, const std::size_t n, uint64_t* const H, const std::size_t k)
{
    // TODO: repeated code; have better policy for this.
    const auto binarize = [](const T_sym_ x)
    {
        // Maximum element of the lower half: exactly `2^(N - 1)` values at each half.
        constexpr T_sym_ mid = std::numeric_limits<T_sym_>::min() / 2 + std::numeric_limits<T_sym_>::max() / 2;
        return x <= mid ? 0 : 1;
    };

    const auto kgram_count = n - k + 1;
    const auto hist_sz = 1llu << k;

    std::memset(H, 0, hist_sz * sizeof(uint64_t));  // Clear out the histogram.

    #pragma omp parallel
    {
        std::vector<uint64_t> local_hist(hist_sz, 0);

        #pragma omp for schedule(static)
        for(std::size_t i = 0; i < kgram_count; ++i)    // Thread-local counting.
        {
            uint32_t bin_kgram = 0; // Binarized k-gram.
            for(std::size_t j = 0; j < k; ++j)
                bin_kgram |= binarize(T[i + j]) << (k - 1 - j);

            assert(bin_kgram < hist_sz);
            local_hist[bin_kgram]++;
        }

        #pragma omp for schedule(static)    // Global reduction of the local counts.
        for(std::size_t b = 0; b < hist_sz; ++b)
            #pragma omp atomic
            H[b] += local_hist[b];
    }
}



// Template instantiations for the required instances.
template void kgram_histogram<char>(const char*, std::size_t, uint64_t*, std::size_t);
template void kgram_histogram<uint8_t>(const uint8_t*, std::size_t, uint64_t*, std::size_t);
template void kgram_histogram<uint16_t>(const uint16_t*, std::size_t, uint64_t*, std::size_t);
template void kgram_histogram<uint32_t>(const uint32_t*, std::size_t, uint64_t*, std::size_t);

}
