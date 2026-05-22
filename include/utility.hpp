
#ifndef UTILITY_HPP
#define UTILITY_HPP



#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include <chrono>


namespace caps_sa::timer
{
    // Fields for profiling time.

    typedef std::chrono::high_resolution_clock::time_point time_point_t;
    const static auto now = std::chrono::high_resolution_clock::now;
    const static auto duration = [](const std::chrono::nanoseconds& d) { return std::chrono::duration_cast<std::chrono::duration<double>>(d).count(); };
}


namespace gpu_saca
{

// Generates a random text of symbols of type `T_sym_` and of length `n` at `T`.
template <typename T_sym_>
void generate_random_text(std::vector<T_sym_>& T, const std::size_t n)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<uint64_t> dist(std::numeric_limits<T_sym_>::min(), std::numeric_limits<T_sym_>::max());

    T.resize(n);
    for(std::size_t i = 0; i < n; ++i)
        T[i] = static_cast<T_sym_>(dist(rng));
}

}



#endif
