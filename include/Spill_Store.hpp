
#ifndef SPILL_STORE_HPP
#define SPILL_STORE_HPP



#include "Direct_File.hpp"
#include "types.hpp"

#include <cstddef>
#include <vector>
#include <string>

namespace gpu_saca
{

// Types / roles of per-partition spill files. A partition has the following
// spill files: `suf`- / `key`-file holds the partition's suffix- / key-content,
// and two pairs (for alternating generations) of message-inbox files (`tgt`
// holds the targets, and `rank` holds the senders' ranks).
enum class Spill_Role
{
    // Following are partition-content.
    suf,    // Suffixes, parallel to `key`.
    key,    // Keys (local ranks), parallel to `suf`.

    // Following are message-inboxes.
    msg_tgt_0,  // Message targets; even-parity inbox.
    msg_rank_0, // Senders' ranks, parallel to `msg_tgt_0`.
    msg_tgt_1,  // Message targets; odd-parity inbox.
    msg_rank_1, // Senders' ranks, parallel to `msg_tgt_1`.
};

// Count of spill-roles, i.e. the maximum count of spill-files a partition has
// open at once.
constexpr std::size_t SPILL_ROLE_C = static_cast<std::size_t>(Spill_Role::msg_rank_1) + 1;


// Returns the message-target inbox role a generation with parity `parity`
// writes to.
constexpr Spill_Role msg_tgt_role(const idx_t parity) { return parity ? Spill_Role::msg_tgt_1 : Spill_Role::msg_tgt_0;  }

// Returns the message-rank inbox role a generation with parity `parity` writes
// to.
constexpr Spill_Role msg_rank_role(const idx_t parity) { return parity ? Spill_Role::msg_rank_1 : Spill_Role::msg_rank_0; }


// Manager of a partition's spill-files.
class Spill_Store
{
private:

    std::vector<std::string> drive_root_;   // Scratch roots (ideally one per NVMe) to stripe partitions across.
    std::string run_dir_;   // Run-scoped sub-directory path, created per each root.
    std::string op_path_;   // Path to the run's output-file; empty for no output.
    std::size_t op_bytes_;  // Bytes of the run's output.
    std::vector<std::size_t> root_of_;  // Scratch root of each partition's spill-files, as an index into `drive_root_`.


    // Returns the file-name suffix for spill-role `role`.
    static const char* role_name(Spill_Role role);


public:

    // Creates directories `<root>/<run_dir>` under each root in `drive_roots`,
    // for a spill-store. The run writes its output of `op_bytes` bytes to the
    // file at `op_path` once the inbox-files retire, alongside the content-
    // files; or none, if `op_path` is empty.
    Spill_Store(std::vector<std::string> drive_roots, const std::string& run_dir, const std::string& op_path, std::size_t op_bytes);

    // Stripes the partitions' spill-files round-robin across the scratch roots.
    // The `p_id`'th partition's content-files take up `content[p_id]` bytes,
    // and its inbox-files at most `inbox[p_id]` bytes per generation-parity.
    // Aborts if the filesystems cannot hold the spill-files even with `ram`
    // bytes of the partitions' data held in RAM instead: past the initial
    // ranking, whose messages fill a generation-parity's inbox-files, or at the
    // output, which joins the content-files once the inbox-files retire. Warns
    // if they may not, without RAM: with the spill-files at their largest, or
    // at the output. To be invoked before any file opens.
    void stripe(const std::vector<std::size_t>& content, const std::vector<std::size_t>& inbox, std::size_t ram);

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
