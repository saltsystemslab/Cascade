
#ifndef CAPS_SA_SUFFIX_ARRAY_HPP
#define CAPS_SA_SUFFIX_ARRAY_HPP



#include <cstddef>


namespace caps_sa
{

void build_SA_naive(const char* T, std::size_t n, std::size_t* SA);

void build_SA_prefix_doubling(const char* T, std::size_t n, std::size_t* SA);

}



#endif
