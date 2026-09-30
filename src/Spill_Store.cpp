
#include "Spill_Store.hpp"
#include "utility.hpp"

#include <sys/stat.h>
#include <sys/statvfs.h>

#include <numeric>
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


Spill_Store::Spill_Store(std::vector<std::string> drive_roots, const std::string& run_dir):
      drive_root_(std::move(drive_roots))
    , run_dir_(run_dir)
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


void Spill_Store::stripe(const std::vector<std::size_t>& footprint)
{
    constexpr auto GB = [](const std::size_t bytes){ return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); };

    // The roots' filesystems, each as the index of its first root, and their free space: roots on one filesystem share its room.
    const auto root_c = drive_root_.size();
    std::vector<dev_t> dev(root_c); // Device of each root.
    std::vector<std::size_t> fs(root_c);    // Filesystem of each root.
    std::vector<std::size_t> room(root_c, 0);   // Free bytes of each filesystem.
    for(std::size_t r = 0; r < root_c; ++r)
    {
        struct stat st;
        struct statvfs vfs;
        if(::stat(drive_root_[r].c_str(), &st) != 0 || ::statvfs(drive_root_[r].c_str(), &vfs) != 0)
            util::file::exit_on_io_failure("statvfs", drive_root_[r]);

        dev[r] = st.st_dev;
        fs[r] = std::find(dev.cbegin(), dev.cbegin() + r, st.st_dev) - dev.cbegin();
        room[fs[r]] = static_cast<std::size_t>(vfs.f_bavail) * vfs.f_frsize;
    }

    // Largest partitions first, each to the root whose filesystem has the most room left, and to the root least taken among
    // those of one filesystem.
    std::vector<p_id_t> order(footprint.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](const auto x, const auto y){ return footprint[x] > footprint[y]; });

    root_of_.assign(footprint.size(), 0);
    std::vector<std::size_t> taken(root_c, 0);  // Bytes striped to each filesystem.
    std::vector<std::size_t> root_taken(root_c, 0); // Bytes striped to each root.
    std::vector<std::size_t> root_part_c(root_c, 0);    // Partitions striped to each root.
    for(const auto p_id : order)
    {
        std::size_t to = 0;
        for(std::size_t r = 1; r < root_c; ++r)
        {
            // The rooms left compare with each side offset by the other's taken bytes, so that neither underflows.
            const auto left_r = room[fs[r]] + taken[fs[to]], left_to = room[fs[to]] + taken[fs[r]];
            if(left_r > left_to || (left_r == left_to && root_taken[r] < root_taken[to]))
                to = r;
        }

        root_of_[p_id] = to;
        taken[fs[to]] += footprint[p_id], root_taken[to] += footprint[p_id], root_part_c[to]++;
    }

#ifdef STATS
    std::cerr << "Scratch striping (spill-files at their largest):\n";
    for(std::size_t f = 0; f < root_c; ++f)
        if(fs[f] == f)
        {
            std::size_t part_c = 0;
            std::cerr << "\t";
            for(std::size_t r = f; r < root_c; ++r)
                if(fs[r] == f)
                    std::cerr << (r > f ? ", " : "") << drive_root_[r], part_c += root_part_c[r];
            std::cerr << ": " << part_c << " partitions, " << GB(taken[f]) << "GBs of " << GB(room[f]) << "GBs free.\n";
        }
#endif

    const auto taken_all = std::accumulate(taken.cbegin(), taken.cend(), std::size_t(0));
    const auto room_all = std::accumulate(room.cbegin(), room.cend(), std::size_t(0));
    for(std::size_t f = 0; f < root_c; ++f)
        if(fs[f] == f && taken[f] > room[f])
        {
            std::cerr   << "The scratch filesystem of " << drive_root_[f] << " has " << GB(room[f]) << "GBs free, short of the " << GB(taken[f])
                        << "GBs its partitions' spill-files may take up (" << GB(taken_all) << "GBs for all the partitions, against " << GB(room_all) << "GBs free). Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
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
