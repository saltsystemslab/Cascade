
#include "tests.hpp"
#include "utility.hpp"

#include "omp.h"


namespace gpu_saca::test
{

bool is_permutation(const idx_t* const SA, const idx_t n)
{
    uint8_t* const is_present = util::allocate<uint8_t>(n);

    #pragma omp parallel for schedule(static)
    for(idx_t i = 0; i < n; ++i)
        is_present[i] = 0;

    bool is_perm = true;

    #pragma omp parallel for schedule(static) reduction(&& : is_perm)
    for(idx_t i = 0; i < n; ++i)
        if(SA[i] >= n)
            is_perm = false;
        else
            is_present[SA[i]] = 1;

    if(!is_perm)
    {
        util::deallocate(is_present);
        return false;
    }

    #pragma omp parallel for schedule(static) reduction(&& : is_perm)
    for(idx_t i = 0; i < n; ++i)
        if(!is_present[i])
            is_perm = false;

    util::deallocate(is_present);
    return is_perm;
}


template <typename T_sym_>
bool is_correct_SA(const idx_t* const SA, const T_sym_* const T, const idx_t n)
{
    if(!is_permutation(SA, n))
        return false;

    idx_t* const rank = gpu_saca::util::allocate<idx_t>(n);

    #pragma omp parallel for schedule(static)
    for(idx_t i = 0; i < n; ++i)
        rank[SA[i]] = i;

    bool is_sorted = true;

    #pragma omp parallel for schedule(static) reduction(&& : is_sorted)
    for(idx_t i = 1; i < n; ++i)
    {
        const auto x = SA[i - 1];
        const auto y = SA[i];
        if(T[x] < T[y])
            continue;
        else if(T[x] > T[y])
            is_sorted = false;
        else
        {
            const int64_t next_x = (x + 1 < n ? static_cast<int64_t>(rank[x + 1]) : -1);
            const int64_t next_y = (y + 1 < n ? static_cast<int64_t>(rank[y + 1]) : -1);
            if(next_x >= next_y)
                is_sorted = false;
        }
    }

    gpu_saca::util::deallocate(rank);
    return is_sorted;
}



// Template instantiations for the required instances.
template bool is_correct_SA(const idx_t*, const char*, idx_t);
template bool is_correct_SA(const idx_t*, const uint8_t*, idx_t);
template bool is_correct_SA(const idx_t*, const uint16_t*, idx_t);
template bool is_correct_SA(const idx_t*, const uint32_t*, idx_t);
template bool is_correct_SA(const idx_t*, const uint64_t*, idx_t);

}
