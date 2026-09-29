
#ifndef RAM_BACKEND_HPP
#define RAM_BACKEND_HPP



#include "types.hpp"

#include <cstddef>
#include <vector>


namespace gpu_saca
{

// Partitions' storage backend in RAM.
class RAM_Backend
{

private:

    const std::vector<idx_t> P_cap_;    // Partition capacities.
    std::vector<idx_t*> suf_;   // Suffixes of the partitions.
    std::vector<p_sz_t*> key_;  // Keys of the partitions.

    // Returns the bytes reserved for `n` elements of `w` bytes each: whole disk-
    // blocks, as disk-block rounded I/O can move them in and out.
    static std::size_t reservation(idx_t n, std::size_t w);

public:

    // Reserves the RAM-storage for partitions with capacities `P_cap`: address-
    // space for each partition's each field, backed by memory only where written.
    RAM_Backend(const std::vector<idx_t>& P_cap);

    RAM_Backend(const RAM_Backend&) = delete;
    RAM_Backend& operator=(const RAM_Backend&) = delete;

    ~RAM_Backend();

    // Returns the `p_id`'th partition's suffixes.
    idx_t* suf(const p_id_t p_id) { return suf_[p_id]; }

    // Returns the `p_id`'th partition's keys.
    p_sz_t* key(const p_id_t p_id) { return key_[p_id]; }

    // Reads `sz`-many suffixes of the `p_id`'th partition, from its offset
    // `off`, into `dst`.
    void read_suf(p_id_t p_id, idx_t off, idx_t sz, idx_t* dst) const;

    // Reads `sz`-many keys of the `p_id`'th partition, from its offset `off`,
    // into `dst`.
    void read_key(p_id_t p_id, idx_t off, idx_t sz, p_sz_t* dst) const;

    // Writes `sz`-many suffixes of the `p_id`'th partition, at its offset
    // `off`, from `src`.
    void write_suf(p_id_t p_id, idx_t off, idx_t sz, const idx_t* src);

    // Writes `sz`-many keys of the `p_id`'th partition, at its offset `off`,
    // from `src`.
    void write_key(p_id_t p_id, idx_t off, idx_t sz, const p_sz_t* src);

    // Releases the memory of `sz`-many suffixes of the `p_id`'th partition, from
    // its page-aligned offset `off`; the range reads back as zeros.
    void release_suf(p_id_t p_id, idx_t off, idx_t sz);

    // Releases the memory of `sz`-many keys of the `p_id`'th partition, from its
    // page-aligned offset `off`; the range reads back as zeros.
    void release_key(p_id_t p_id, idx_t off, idx_t sz);

    // Releases the memory of the `p_id`'th partition's suffixes and keys; their
    // address-space stays reserved.
    void retire_data(p_id_t p_id);
};

}



#endif
