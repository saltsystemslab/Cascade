
#ifndef UTILITY_HPP
#define UTILITY_HPP



#include "cross-compile.hpp"

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <limits>
#include <utility>
#include <vector>
#include <filesystem>
#include <fstream>
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
    typedef std::chrono::steady_clock::time_point time_point_t;
    const static auto now = std::chrono::steady_clock::now;
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
constexpr std::size_t PAGE_SIZE = 4096;
constexpr std::size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;


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
    // A signed symbol would sign-extend across every higher symbol-slot of the word, so the word's numeric order would stop matching the text's
    // lexicographic order.
    static_assert(std::is_unsigned_v<T_sym_>, "the symbol type must be unsigned");

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


// Prepages `size` elements (of type `T_`) from address `ptr`, in parallel. The
// pages are supposed to be first-touched here; if data already exists, some of
// it will be destroyed.
template <typename T_>
void prepage(T_* const ptr, const std::size_t size)
{
    constexpr std::size_t stride = (PAGE_SIZE + sizeof(T_) - 1) / sizeof(T_);   // Elements spanning a page.

    #pragma omp parallel for schedule(static)
    for(std::size_t i = 0; i < size; i += stride)
        ptr[i] = T_();
}


// Returns the host's currently available memory, in bytes.
std::size_t available_memory();

// Returns a reservation of `bytes` bytes of address-space, `HUGE_PAGE_SIZE`-
// aligned and backed by physical memory only when touched; `nullptr` for empty
// reservations. The reservation opts into transparent huge pages iff the env
// var `GPU_SACA_MADV_HUGEPAGE` is non-zero.
void* reserve_memory(std::size_t bytes);

// Releases the reservation `ptr` of `bytes` bytes, made with `reserve_memory`.
void unreserve_memory(void* ptr, std::size_t bytes);

// Returns from `/proc/self/smaps` (pseudo-file with memory consumption metrics
// for each memory mapping), for the memory mapping containing the address `p`,
// `(RSS-bytes, THP-bytes)`. Both are `0` if not found.
std::pair<std::size_t, std::size_t> thp_coverage(const void* p);

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


// Reads `count` elements of type `T_` from the file at path `file_path` into
// `dst`, which must have enough space allocated beforehand.
template <typename T_>
void read_file(const std::string& file_path, T_* const dst, const std::size_t count)
{
    std::ifstream is(file_path, std::ios::binary);
    is.read(reinterpret_cast<char*>(dst), count * sizeof(T_));
    if(!is)
    {
        std::cerr << "Could not read " << (count * sizeof(T_)) << " bytes from the file at path " << file_path << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }
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


namespace gpu_saca::util::file
{

// Exits the executions due to reason `op` on the file at path `path`.
void exit_on_io_failure(const char* op, const std::string& path);

// Writes `bytes`-many bytes from `buf` to the file-offset `off` of the file
// `fd`, opened at path `path`. Additionally queues the bytes for writeback, so
// that dirty pages do not pile up from a long output-stream.
void write_and_flush(int fd, const std::string& path, const void* buf, std::size_t bytes, std::size_t off);

}



#endif
