
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


template <typename T_sym_>
bool is_correct_SA(const std::string& text_path, const std::string& SA_path)
{
    const auto sym_count = util::file_size<T_sym_>(text_path);
    const auto n = sym_count;

    const auto SA_sz = util::file_size<idx_t>(SA_path);
    if(SA_sz != n)
    {
        std::cerr   << "The suffix array at " << SA_path << " has " << SA_sz << " entries whereas the text at "
                    << text_path << " has " << n << " suffixes.\n";
        return false;
    }

    auto* const T = util::checked_allocate<T_sym_>(n);
    util::read_file(text_path, T, sym_count);

    auto* const SA = util::checked_allocate<idx_t>(n);
    util::read_file(SA_path, SA, n);

    const auto correct = is_correct_SA(SA, T, n);

    util::deallocate(SA), util::deallocate(T);
    return correct;
}



// Template instantiations for the required instances.
template bool is_correct_SA(const idx_t*, const uint8_t*, idx_t);
template bool is_correct_SA(const idx_t*, const uint16_t*, idx_t);
template bool is_correct_SA(const idx_t*, const uint32_t*, idx_t);
template bool is_correct_SA(const idx_t*, const uint64_t*, idx_t);

template bool is_correct_SA<uint8_t>(const std::string&, const std::string&);
template bool is_correct_SA<uint16_t>(const std::string&, const std::string&);
template bool is_correct_SA<uint32_t>(const std::string&, const std::string&);
template bool is_correct_SA<uint64_t>(const std::string&, const std::string&);

}
