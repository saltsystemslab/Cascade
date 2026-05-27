
#ifndef UTILITY_HPP
#define UTILITY_HPP



#include <cstdint>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <cstdio>
#include <cmath>
#include <cstdlib>
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


// Reads an entire file of elements of type `T_` from `file_path` into `v` and
// returns the count of elements read. Terminates `v` by appending the least
// symbol of type `T_`.
template <typename T_>
std::size_t read_file_terminated(const std::string& file_path, std::vector<T_>& v)
{
    const auto file_bytes = std::filesystem::file_size(file_path);
    if(file_bytes % sizeof(T_) != 0)
    {
        std::cerr << "The file at path " << file_path << " has " << file_bytes << " many bytes, not matching alignment for type " << typeid(T_).name() << "\n";
        std::exit(EXIT_FAILURE);
    }

    v.resize(file_bytes / sizeof(T_) + 1);
    std::ifstream is(file_path, std::ios::binary);
    is.read(reinterpret_cast<char*>(v.data()), file_bytes);
    v.back() = std::numeric_limits<T_>::min();

    return v.size();
}


struct Collection_Stats
{
    uint64_t n;
    uint64_t total;
    uint64_t min;
    uint64_t max;
    double mean;
    double median;
    double std_dev;
    double imbalance;


    void print() const
    {
        const int w = 20;   // Label column width
        std::printf("%-*s %llu\n",    w, "Size:",           (unsigned long long)n);
        std::printf("%-*s %llu\n",    w, "Total mass:",     (unsigned long long)total);
        std::printf("%-*s %llu\n",    w, "Minimum:",        (unsigned long long)min);
        std::printf("%-*s %llu\n",    w, "Maximum:",        (unsigned long long)max);
        std::printf("%-*s %.2f\n",    w, "Mean:",           mean);
        std::printf("%-*s %.2f\n",    w, "Median:",         median);
        std::printf("%-*s %.2f\n",    w, "Std deviation:",  std_dev);
        std::printf("%-*s %.4f\n",    w, "Imbalance:",      imbalance);
    }


    template <typename T_>
    static Collection_Stats get_collection_stat(const std::vector<T_>& C)
    {
        const std::size_t n = C.size();

        uint64_t total = 0;
        uint64_t min = std::numeric_limits<uint64_t>::max();
        uint64_t max = 0;
        for(const auto x : C)
            total += x,
            min = std::min(min, x),
            max = std::max(max, x);


        const double mean = static_cast<double>(total) / n;

        double var = 0;
        for(const auto x : C)
        {
            const auto diff = static_cast<double>(x) - mean;
            var += diff * diff;
        }

        const double std_dev =std::sqrt(var / n);

        auto sorted_C = C;
        std::sort(sorted_C.begin(), sorted_C.end());
        const double median = (n % 2 ?  sorted_C[n / 2] :
                                        (sorted_C[n / 2 - 1] + sorted_C[n / 2]) / 2.0);

        const double imbalance = (mean > 0 ? static_cast<double>(max - min) / mean : 0);

        return Collection_Stats{n, total, min, max, mean, median, std_dev, imbalance};
    }
};

}



#endif
