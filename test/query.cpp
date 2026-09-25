
#include "query_tests.hpp"
#include "CPU_Reference.hpp"
#include "sa40.hpp"

#include <limits>
#include <atomic>
#include <cstring>
#include <numeric>
#include <algorithm>


namespace gpu_saca::test
{

template <typename T_sym_>
bool validate_intervals(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off, const T_sym_* const sym, const idx_t q, const SA_Interval* const I, std::string& err)
{
    std::atomic<idx_t> bad{std::numeric_limits<uint64_t>::max()};

    #pragma omp parallel for schedule(dynamic, 4096)
    for(idx_t i = 0; i < q; ++i)
        if(!ref::certify(T, n, SA, sym + (off[i] - off[0]), off[i + 1] - off[i], I[i]))
        {
            idx_t cur = bad.load();
            while(i < cur && !bad.compare_exchange_weak(cur, i))
                ;
        }

    if(bad.load() != std::numeric_limits<uint64_t>::max())
    {
        const idx_t i = bad.load();
        err =   "interval certificate failed for query " + std::to_string(i) + " (length " + std::to_string(off[i + 1] - off[i])
                + "): [" + std::to_string(I[i].lo) + ", " + std::to_string(I[i].hi) + ")";
        return false;
    }

    return true;
}


template <typename T_sym_>
std::vector<uint8_t> build_sa40(const std::vector<T_sym_>& T)
{
    const idx_t n = T.size();
    std::vector<idx_t> sa(n);
    std::iota(sa.begin(), sa.end(), idx_t(0));
    std::sort(sa.begin(), sa.end(), [&](const idx_t i, const idx_t j) { return ref::cmp_suffixes(T.data(), n, i, j, n) < 0; });

    std::vector<uint8_t> SA(SA_WIDTH * n + SA_PAD_BYTES, 0);
    for(idx_t r = 0; r < n; ++r)
        sa40::store(SA.data(), r, sa[r]);

    return SA;
}


template <typename T_sym_>
std::vector<uint8_t> build_sa40_byte_order(const std::vector<T_sym_>& T)
{
    const idx_t n = T.size();
    const auto B = reinterpret_cast<const uint8_t*>(T.data());
    const idx_t nb = n * sizeof(T_sym_);
    std::vector<idx_t> sa(n);
    std::iota(sa.begin(), sa.end(), idx_t(0));
    std::sort(sa.begin(), sa.end(), [&](const idx_t i, const idx_t j)
    {
        const idx_t bi = i * sizeof(T_sym_), bj = j * sizeof(T_sym_);
        const idx_t len = std::min(nb - bi, nb - bj);
        const int c = std::memcmp(B + bi, B + bj, len);
        return c != 0 ? c < 0 : (nb - bi) < (nb - bj);
    });

    std::vector<uint8_t> SA(SA_WIDTH * n + SA_PAD_BYTES, 0);
    for(idx_t r = 0; r < n; ++r)
        sa40::store(SA.data(), r, sa[r]);

    return SA;
}


template <typename T_sym_>
SA_Interval brute_find(const T_sym_* const T, const idx_t n, const T_sym_* const P, const idx_t m)
{
    if(m == 0)
        return {0, n};

    idx_t less = 0, equal = 0;
    for(idx_t i = 0; i < n; ++i)
    {
        const int c = ref::cmp_suffix_pattern(T, n, i, P, m);
        less += (c < 0);
        equal += (c == 0);
    }

    return {less, less + equal};
}


// Template instantiations for the required instances.

template bool validate_intervals<uint8_t>(const uint8_t*, idx_t, const uint8_t*, const uint64_t*, const uint8_t*, idx_t, const SA_Interval*, std::string&);
template bool validate_intervals<uint16_t>(const uint16_t*, idx_t, const uint8_t*, const uint64_t*, const uint16_t*, idx_t, const SA_Interval*, std::string&);
template bool validate_intervals<uint32_t>(const uint32_t*, idx_t, const uint8_t*, const uint64_t*, const uint32_t*, idx_t, const SA_Interval*, std::string&);

template std::vector<uint8_t> build_sa40<uint8_t>(const std::vector<uint8_t>&);
template std::vector<uint8_t> build_sa40<uint16_t>(const std::vector<uint16_t>&);
template std::vector<uint8_t> build_sa40<uint32_t>(const std::vector<uint32_t>&);

template std::vector<uint8_t> build_sa40_byte_order<uint8_t>(const std::vector<uint8_t>&);
template std::vector<uint8_t> build_sa40_byte_order<uint16_t>(const std::vector<uint16_t>&);
template std::vector<uint8_t> build_sa40_byte_order<uint32_t>(const std::vector<uint32_t>&);

template SA_Interval brute_find<uint8_t>(const uint8_t*, idx_t, const uint8_t*, idx_t);
template SA_Interval brute_find<uint16_t>(const uint16_t*, idx_t, const uint16_t*, idx_t);
template SA_Interval brute_find<uint32_t>(const uint32_t*, idx_t, const uint32_t*, idx_t);

}
