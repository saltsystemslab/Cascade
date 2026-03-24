
#ifndef CAPS_SA_SUFFIX_ARRAY_HPP
#define CAPS_SA_SUFFIX_ARRAY_HPP



#include <cstdint>
#include <cstddef>


namespace caps_sa
{

void build_SA_naive(const char* T, std::size_t n, std::size_t* SA);

void build_SA_prefix_doubling(const char* T, std::size_t n, std::size_t* SA);


// The suffix array constructor class.
class Suffix_Array
{
private:

    typedef uint64_t idx_t;
    const char* const T_h;  // The input text.
    const idx_t n;  // Length of the input text.


public:

    // Constructs a suffix array constructor object for the input text `T` of
    // size `n`.
    Suffix_Array(const char* const T, const idx_t n):
          T_h(T)
        , n(n)
    {}

    // Constructs the suffix array of the provided text at `SA`.
    void build_SA(idx_t* SA);
};

}



#endif
