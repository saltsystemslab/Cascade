
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

    // Constructs a suffix with index `suf`, primary key `p_key`, and secondary
    // key `s_key`.
    Suffix(const idx_t suf, const idx_t p_key, const idx_t s_key):
          suf(suf)
        , key{p_key, s_key}
    {}

    // Returns the suffix's index.
    auto suffix() const { return suf; }

    // Returns the primary sort-key of the suffix.
    auto primary_key() const { return key[0]; }

    // Returns the secondary sort-key of the suffix.
    auto secondary_key() const { return key[1]; }
};

}



#endif