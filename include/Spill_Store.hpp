
#ifndef SPILL_STORE_HPP
#define SPILL_STORE_HPP



#include "Direct_File.hpp"
#include "types.hpp"

#include <vector>
#include <string>

namespace gpu_saca
{

// Types / roles of per-partition spill files. A partition has three separate
// spill files: `suf`- / `key`-file holds the partition's suffix- / key-content.
enum class Spill_Role
{
    prefix, // Initial packed-prefixes.
    suf,    // Suffixes, parallel to `prefix` and `key`.
    key,    // Keys (local ranks), parallel to `suf`.
};


// Manager of a partition's spill-files.
class Spill_Store
{
private:

    std::vector<std::string> drive_root_;   // Scratch roots (ideally one per NVMe) to stripe partitions across.
    std::string run_dir_;   // Run-scoped sub-directory path, created per each root.


    // Returns the file-name suffix for spill-role `role`.
    static const char* role_name(Spill_Role role);


public:

    // Creates directories `<root>/<run_dir>` under each root in `drive_roots`,
    // for a spill-store.
    Spill_Store(std::vector<std::string> drive_roots, const std::string& run_dir);

    // Returns the path to the `p_id`'th partition's `role` file.
    std::string path(p_id_t p_id, Spill_Role role) const;

    // Opens the `p_id`'th partitions `role` file as a direct-file with access-
    // mode `mode`, and returns it.
    Direct_File open(p_id_t p_id, Spill_Role role, Direct_File::Mode mode) const;

    // Removes the opened run-directories.
    void cleanup() const;
};

}



#endif
