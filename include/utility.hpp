
#ifndef UTILITY_HPP
#define UTILITY_HPP



#include "cross-compile.hpp"

#include <cstdint>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include <filesystem>
#include <iostream>
#include <typeinfo>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <chrono>


namespace gpu_saca::timer
{
    // Fields for profiling time.

    typedef std::chrono::high_resolution_clock::time_point time_point_t;
    const static auto now = std::chrono::high_resolution_clock::now;
    const static auto duration = [](const std::chrono::nanoseconds& d) { return std::chrono::duration_cast<std::chrono::duration<double>>(d).count(); };

    // Scope-timer, based on RAII mechanism.
    class Scope_Timer
    {
    private:

        uint64_t* const sink_;  // Where to accumulate the scope's timing.
        const uint64_t t0_; // Initial timestamp of the timer's scope.

        // Returns the current timestamp in nanoseconds.
        static uint64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    public:

        // Constructs a scope-timer accumulating its timing to `sink`.
        explicit Scope_Timer(uint64_t& sink):
              sink_(&sink)
            , t0_(now_ns())
        {}

        ~Scope_Timer() { *sink_ += (now_ns() - t0_); }

        Scope_Timer(const Scope_Timer&) = delete;
        Scope_Timer& operator=(const Scope_Timer&) = delete;
        Scope_Timer(Scope_Timer&&) = delete;
        Scope_Timer& operator=(Scope_Timer&&) = delete;
    };
}


namespace gpu_saca::util::type
{
    template <typename T>
    inline void print_type()
    {
        // Prints the full function signature containing the exact type name.
#if defined(__clang__) || defined(__GNUC__)
        std::cout << __PRETTY_FUNCTION__ << "\n";
#elif defined(_MSC_VER)
        std::cout << __FUNCSIG__ << "\n";
#endif
    }
}


namespace gpu_saca::util
{

constexpr std::size_t L1_CACHE_LINE_SIZE = 64;


// Wrapper class for a data-element of type `T_` to ensure that in a linear
// collection of `T_`'s, each element is aligned to a cache-line boundary.
template <typename T_>
class alignas(L1_CACHE_LINE_SIZE)
    Padded
{
private:

    T_ data_;

public:

    Padded()
    {}

    Padded(const T_& data):
      data_(data)
    {}

    Padded(T_&& data):
        data_(std::move(data))
    {}

    T_& unwrap() { return data_; }

    const T_& unwrap() const { return data_; }
};


namespace math
{
    // Returns the number of bits required to represent `x`, i.e. `ceil(log_2(x))`.
    inline uint64_t bits_for(const uint64_t x)
    {
        return x <= 1 ? 0 : 64 - __builtin_clzll(x - 1);
    }
}


// Returns the `N` symbols situated at the text `T` as a 64-bit word. The
// endian-ness of the word matches the lexicographical order of the text—the
// symbol at index `idx` corresponds to the most significant bits of the word.
// NB: the text must have been appropriately padded at the tail.
template <typename T_sym_, uint64_t N = (8 / sizeof(T_sym_))>
CUDA_SPECIFIER
uint64_t pack_word(const T_sym_* const T)
{
    if constexpr(N == 0)
        return 0;
    else
    {
        constexpr uint64_t shift = (N - 1) * sizeof(T_sym_) * 8;
        return (static_cast<uint64_t>(*T) << shift) | pack_word<T_sym_, N - 1>(T + 1);
    }
}


// Returns pointer to a memory-allocation for `size` elements of type `T_`.
template <typename T_>
T_* allocate(const std::size_t size)
{
    return static_cast<T_*>(std::malloc(size * sizeof(T_)));
}


// Returns pointer to a memory-allocation for `n` elements of type `T_`, whose
// alignment is specified by `alignment`.
template <typename T_>
T_* aligned_allocate(const std::size_t size, const std::size_t alignment)
{
    return static_cast<T_*>(std::aligned_alloc(alignment, size * sizeof(T_)));
}


// Returns pointer to a memory-allocation for `size` elements of type `T_`;
// aborts the execution if the allocation fails.
template<typename T_>
T_* checked_allocate(const std::size_t sz)
{
    auto* const p = allocate<T_>(sz);
    if(p == nullptr)
    {
        std::cerr << "Failed to allocate " << sz << "-many bytes for: ";
        type::print_type<T_>();
        std::cerr << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    return p;
}


// Deallocates the pointer `ptr`, allocated with `allocate`.
template <typename T_>
void deallocate(T_* const ptr)
{
    std::free(ptr);
}


// Returns the host's currently available memory, in bytes.
std::size_t available_memory();


// Generates a random text of symbols of type `T_sym_` and of length `n` at `T`.
// `T` must have enough space allocated beforehand.
template <typename T_sym_>
void generate_random_text(T_sym_* const T, const std::size_t n)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<uint64_t> dist(std::numeric_limits<T_sym_>::min(), std::numeric_limits<T_sym_>::max());

    for(std::size_t i = 0; i < n; ++i)
        T[i] = static_cast<T_sym_>(dist(rng));
}


// Returns the size of the file of elements of type `T_` at path `file_path`.
template <typename T_>
std::size_t file_size(const std::string& file_path)
{
    const auto file_bytes = std::filesystem::file_size(file_path);
    if(file_bytes % sizeof(T_) != 0)
    {
        std::cerr << "The file at path " << file_path << " has " << file_bytes << " many bytes, not matching alignment for type " << typeid(T_).name() << "\n";
        std::exit(EXIT_FAILURE);
    }

    return file_bytes / sizeof(T_);
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
        std::printf("\t%-*s %llu\n",    w, "Size:",           (unsigned long long)n);
        std::printf("\t%-*s %llu\n",    w, "Total mass:",     (unsigned long long)total);
        std::printf("\t%-*s %llu\n",    w, "Minimum:",        (unsigned long long)min);
        std::printf("\t%-*s %llu\n",    w, "Maximum:",        (unsigned long long)max);
        std::printf("\t%-*s %.2f\n",    w, "Mean:",           mean);
        std::printf("\t%-*s %.2f\n",    w, "Median:",         median);
        std::printf("\t%-*s %.2f\n",    w, "Std deviation:",  std_dev);
        std::printf("\t%-*s %.4f\n",    w, "Imbalance:",      imbalance);
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
