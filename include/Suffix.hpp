
#ifndef SUFFIX_HPP
#define SUFFIX_HPP



#include "types.hpp"


namespace gpu_saca
{

// A suffix index, with associated sort-keys.
class Suffix
{
private:

    idx_t suf;  // The suffix's index.
    idx_t key[2];   // The primary and the secondary sort-key for the suffix.

public:

    // Returns the suffix's index.
    auto suffix() const { return suf; }

    // Returns the primary sort-key of the suffix.
    auto primary_key() const { return key[0]; }

    // Returns the secondary sort-key of the suffix.
    auto secondary_key() const { return key[1]; }
};

}



#endif