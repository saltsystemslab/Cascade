
#include "Spill_Store.hpp"
#include "utility.hpp"

#include <sys/stat.h>
#include <sys/statvfs.h>

#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <system_error>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

const char* Spill_Store::role_name(const Spill_Role role)
{
    switch(role)
    {
    case Spill_Role::suf:       return "suf";
    case Spill_Role::key:       return "key";

    case Spill_Role::msg_tgt_0:     return "msg_tgt_0";
    case Spill_Role::msg_rank_0:    return "msg_rank_0";
    case Spill_Role::msg_tgt_1:     return "msg_tgt_1";
    case Spill_Role::msg_rank_1:    return "msg_rank_1";
    }

    return "undefined";
}


Spill_Store::Spill_Store(std::vector<std::string> drive_roots, const std::string& run_dir, const std::string& op_path, const std::size_t op_bytes):
      drive_root_(std::move(drive_roots))
    , run_dir_(run_dir)
    , op_path_(op_path)
    , op_bytes_(op_bytes)
{
    if(drive_root_.empty())
    {
        std::cerr << "No scratch drive-roots provided for spill-store. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    for(const auto& root : drive_root_)
    {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(root) / run_dir_, ec);
        if(ec)
        {
            std::cerr << "Error in creating run directory under " << root << ": " << ec.message() << ". Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
    }
}


void Spill_Store::stripe(const std::vector<std::size_t>& content, const std::vector<std::size_t>& inbox, const std::size_t ram)
{
    constexpr auto GB = [](const std::size_t bytes){ return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); };

    // Paths on the filesystems of the spill-files and of the output: the roots, and past them the output-file's directory.
    // The output truncates the output-file present, if any, freeing up its space.
    const auto root_c = drive_root_.size();
    const auto dump = !op_path_.empty();
    std::vector<std::string> at(drive_root_);
    std::size_t op_present = 0; // Bytes of the output-file present.
    if(dump)
    {
        const auto dir = std::filesystem::path(op_path_).parent_path();
        at.push_back(dir.empty() ? std::string(".") : dir.string());

        struct stat st;
        if(::stat(op_path_.c_str(), &st) == 0 && S_ISREG(st.st_mode))
            op_present = static_cast<std::size_t>(st.st_blocks) * 512;
    }

    // The paths' filesystems, each as the index of its first path, and their free space: paths on one filesystem share its
    // room.
    std::vector<dev_t> dev(at.size());  // Device of each path.
    std::vector<std::size_t> fs(at.size()); // Filesystem of each path.
    std::vector<std::size_t> room(at.size(), 0);    // Free bytes of each filesystem.
    for(std::size_t r = 0; r < at.size(); ++r)
    {
        struct stat st;
        struct statvfs vfs;
        if(::stat(at[r].c_str(), &st) != 0 || ::statvfs(at[r].c_str(), &vfs) != 0)
            util::file::exit_on_io_failure("statvfs", at[r]);

        dev[r] = st.st_dev;
        fs[r] = std::find(dev.cbegin(), dev.cbegin() + r, st.st_dev) - dev.cbegin();
        room[fs[r]] = static_cast<std::size_t>(vfs.f_bavail) * vfs.f_frsize;
    }

    // The partitions stripe round-robin across the roots.
    const auto p = content.size();
    root_of_.resize(p);
    std::vector<std::size_t> taken(at.size(), 0);   // Bytes striped to each filesystem, of the spill-files at their largest: with both the generation-parities' inbox-files.
    std::vector<std::size_t> content_taken(at.size(), 0);   // Bytes of content-files striped to each filesystem.
    std::vector<std::size_t> inbox_taken(at.size(), 0); // Bytes of a generation-parity's inbox-files striped to each filesystem.
    std::vector<std::size_t> root_part_c(root_c, 0);    // Partitions striped to each root.
    for(p_id_t p_id = 0; p_id < p; ++p_id)
    {
        const auto to = p_id % root_c;
        root_of_[p_id] = to, root_part_c[to]++;
        taken[fs[to]] += content[p_id] + 2 * inbox[p_id], content_taken[fs[to]] += content[p_id], inbox_taken[fs[to]] += inbox[p_id];
    }

    // The filesystems' needs at the moments that bound them: past the initial ranking, whose messages fill a generation-
    // parity's inbox-files; and at the output, with the content-files, and with the room of the output-file present. RAM
    // may hold up to `ram` bytes of the partitions' data in all instead; but not of the output.
    const auto op_fs = (dump ? fs[root_c] : at.size()); // Filesystem of the output, if any.
    const auto msg_need = [&](const std::size_t f){ return content_taken[f] + inbox_taken[f]; };
    const auto op_need = [&](const std::size_t f){ return content_taken[f] + (f == op_fs ? op_bytes_ : 0); };
    const auto op_room = [&](const std::size_t f){ return room[f] + (f == op_fs ? op_present : 0); };
    const auto short_of = [](const std::size_t need, const std::size_t have){ return need > have ? need - have : 0; };
    std::size_t msg_need_all = 0, msg_short = 0;    // Bytes needed of the filesystems past the initial ranking, and that they fall short by, in all.
    std::size_t op_need_all = 0, op_short = 0;  // Bytes needed of the filesystems at the output, and that they fall short by, in all.
    for(std::size_t f = 0; f < at.size(); ++f)
        if(fs[f] == f)
            msg_need_all += msg_need(f), msg_short += short_of(msg_need(f), room[f]),
            op_need_all += op_need(f), op_short += short_of(op_need(f), op_room(f));

#ifdef STATS
    std::cerr   << "Scratch striping, with the spill-files' needs past the initial ranking, at the output, and at their largest "
                << "(RAM may hold " << GB(ram) << "GBs of the partitions' data instead):\n";
    for(std::size_t f = 0; f < at.size(); ++f)
        if(fs[f] == f)
        {
            std::size_t part_c = 0;
            std::cerr << "\t";
            for(std::size_t r = f; r < at.size(); ++r)
                if(fs[r] == f)
                    std::cerr << (r > f ? ", " : "") << (r < root_c ? at[r] : op_path_), part_c += (r < root_c ? root_part_c[r] : 0);
            std::cerr << ": " << part_c << " partitions; " << GB(msg_need(f)) << "GBs, " << GB(op_need(f)) << "GBs, and " << GB(taken[f]) << "GBs, of " << GB(room[f]) << "GBs free";
            if(f == op_fs && op_present > 0)
                std::cerr << ", and " << GB(op_present) << "GBs of the output-file present";
            std::cerr << ".\n";
        }
#endif

    if(msg_short > ram)
    {
        std::cerr   << "The scratch falls " << GB(msg_short) << "GBs short of the " << GB(msg_need_all) << "GBs that the spill-files take up past the initial ranking, "
                    << "more than the " << GB(ram) << "GBs of the partitions' data that RAM may hold instead. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    if(dump && op_bytes_ > op_room(op_fs))
    {
        std::cerr << "The filesystem of the output-file " << op_path_ << " has " << GB(op_room(op_fs)) << "GBs free, short of the output's " << GB(op_bytes_) << "GBs. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    if(op_short > ram)
    {
        std::cerr   << "The scratch falls " << GB(op_short) << "GBs short of the " << GB(op_need_all) << "GBs that the output and the content-files take up, "
                    << "more than the " << GB(ram) << "GBs of the partitions' data that RAM may hold instead. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    for(std::size_t f = 0; f < root_c; ++f)
        if(fs[f] == f && taken[f] > room[f])
            std::cerr   << "Warning: the scratch filesystem of " << at[f] << " has " << GB(room[f]) << "GBs free, short of the " << GB(taken[f])
                        << "GBs that its partitions' spill-files may take up, at their largest; the run may abort for want of space.\n";

    if(dump && op_need(op_fs) > op_room(op_fs))
        std::cerr   << "Warning: the filesystem of the output-file " << op_path_ << " has " << GB(op_room(op_fs)) << "GBs free for the output, short of the " << GB(op_need(op_fs))
                    << "GBs that the output and its partitions' content-files may take up; the run may abort for want of space.\n";
}


std::string Spill_Store::path(const p_id_t p_id, const Spill_Role role) const
{
    const auto& root = drive_root_[root_of_[p_id]];
    return (std::filesystem::path(root) / run_dir_ / (std::to_string(p_id) + "." + role_name(role))).string();
}


Direct_File Spill_Store::open(p_id_t p_id, Spill_Role role, Direct_File::Mode mode) const
{
    return Direct_File(path(p_id, role), mode);
}


void Spill_Store::cleanup() const
{
    for(const auto& root : drive_root_)
    {
        std::error_code ec;
        std::filesystem::remove_all(std::filesystem::path(root) / run_dir_, ec);
        if(ec)
        {
            std::cerr << "Error in removing run directory under " << root << ": " << ec.message() << ". Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

}
