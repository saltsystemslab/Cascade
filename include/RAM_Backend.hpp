
#ifndef RAM_BACKEND_HPP
#define RAM_BACKEND_HPP



#include "Prefix_Key.hpp"
#include "Residency_Plan.hpp"
#include "types.hpp"

#include <vector>


namespace gpu_saca
{

// Partitions' storage backend in RAM.
class RAM_Backend
{

private:

    std::vector<idx_t*> suf_;   // Suffixes of the partitions.
    std::vector<idx_t*> key_;   // Keys of the partitions.
    std::vector<Prefix_Key*> prefix_;   // Prefix-keys of the partitions.

public:

    // Allocates the RAM-resident storage for partitions with capacities
    // `P_cap`, per the residency-plan `plan`.
    RAM_Backend(const std::vector<idx_t>& P_cap, const Residency_Plan& plan);

    RAM_Backend(const RAM_Backend&) = delete;
    RAM_Backend& operator=(const RAM_Backend&) = delete;

    ~RAM_Backend();

    // Reads `sz`-many initial prefixes of the `p_id`'th partition, from its
    // offset `off`, into `dst`.
    void read_prefix(p_id_t p_id, idx_t off, idx_t sz, Prefix_Key* dst) const;

    // Reads `sz`-many suffixes of the `p_id`'th partition, from its offset
    // `off`, into `dst`.
    void read_suf(p_id_t p_id, idx_t off, idx_t sz, idx_t* dst) const;

    // Reads `sz`-many keys of the `p_id`'th partition, from its offset `off`,
    // into `dst`.
    void read_key(p_id_t p_id, idx_t off, idx_t sz, idx_t* dst) const;

    // Writes `sz`-many suffixes and keys of the `p_id`'th partition, at its
    // offset `off`, from `suf` and `key`.
    void write_data(p_id_t p_id, idx_t off, idx_t sz, const idx_t* suf, const idx_t* key);

    // Drops the prefix-keys from the backend.
    void drop_prefixes();

    // Returns the `p_id`'th partition's resident suffixes.
    const idx_t* suf(const p_id_t p_id) const { return suf_[p_id]; }

    // Returns the `p_id`'th partition's resident prefix-keys.
    const Prefix_Key* prefix(const p_id_t p_id) const { return prefix_[p_id]; }
};

}



#endif
