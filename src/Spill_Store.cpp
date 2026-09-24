
#include "Spill_Store.hpp"

#include <string>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <system_error>


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


std::string Spill_Store::path(const p_id_t p_id, const Spill_Role role) const
{
    const auto drive_c = drive_root_.size();
    const auto& root = drive_root_[p_id % drive_c]; // Striping happens here.
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
