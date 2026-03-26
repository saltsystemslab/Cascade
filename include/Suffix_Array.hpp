
#ifndef CAPS_SA_SUFFIX_ARRAY_HPP
#define CAPS_SA_SUFFIX_ARRAY_HPP



#include <cstdint>
#include <cstddef>


namespace caps_sa
{

typedef uint64_t idx_t;

constexpr std::size_t blk_sz = 256;

constexpr std::size_t chunk_sz = 65536;

void build_SA_naive(const char* T, std::size_t n, std::size_t* SA);

void build_SA_prefix_doubling(const char* T, std::size_t n, std::size_t* SA);


// Functor class to compare two suffixes.
struct Suffix_Comparator
{
    const char* T_d;
    const std::size_t n;

    __host__ __device__
    Suffix_Comparator(const char* T_d, const std::size_t n): T_d(T_d), n(n)
    {}

    __host__ __device__
    bool operator()(std::size_t x, std::size_t y)
    {
        for(; x < n && y < n; ++x, ++y)
            if(T_d[x] != T_d[y])
                return T_d[x] < T_d[y];

        return x > y;   // Shorter suffix precedes.
    }
};


// The suffix array constructor class.
class Suffix_Array
{
private:

    const char* const T_h;  // The input text in host.
    char* T_d;  // The input text in device.
    const idx_t n;  // Length of the input text.

    idx_t* SA_d;    // The suffix array on device.


    // Sorts chunks of the suffixes of the text where each chunk has size
    // `chunk_sz`, into the flat array `S_d`.
    void sort_chunks(idx_t* SA_d, idx_t chunk_sz) const;


public:

    // Constructs a suffix array constructor object for the input text `T` of
    // size `n`.
    Suffix_Array(const char* const T, const idx_t n);

    ~Suffix_Array();

    // Constructs the suffix array of the provided text at `SA`.
    void build_SA(idx_t* SA);
};

}



#endif
