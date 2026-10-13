
#include "Output_File.hpp"
#include "utility.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Output_File::Cut Output_File::cut(const std::size_t lo, const std::size_t hi, const std::size_t chunk_bytes, const std::size_t c, const std::size_t chunk_c)
{
    constexpr auto BLK = Direct_File::DISK_BLOCK;
    const auto base = util::math::block_down<BLK>(lo);  // Start of the disk-block holding the range's start.
    const auto mid_lo = util::math::block_up<BLK>(lo);  // Start of the range's block-aligned middle.
    const auto mid_hi = std::max<std::size_t>(mid_lo, util::math::block_down<BLK>(hi)); // End of the middle; empty if the range lies in one block.

    Cut cut{};
    cut.lo = std::min<std::size_t>(std::max<std::size_t>(mid_lo, base + c * chunk_bytes), mid_hi);
    cut.hi = std::max<std::size_t>(cut.lo, std::min<std::size_t>(mid_hi, base + (c + 1) * chunk_bytes));
    if(c == 0)
        cut.head_lo = lo, cut.head_hi = std::min<std::size_t>(hi, mid_lo);

    if(c + 1 == chunk_c)
        cut.tail_lo = std::max<std::size_t>(mid_lo, util::math::block_down<BLK>(hi)), cut.tail_hi = std::max(cut.tail_lo, hi);

    return cut;
}


Output_File::Output_File(const std::string& path, const std::size_t bytes):
      fd_(-1)
    , path_(path)
    , bytes_(bytes)
    , written_(0)
{
    fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);    // Only owner can read-write, others read-only.
    if(fd_ < 0)
        util::file::exit_on_io_failure("open", path_);

    if(::fallocate(fd_, 0, 0, bytes_) != 0)
        util::file::exit_on_io_failure("fallocate", path_);
}


Output_File::~Output_File()
{
    if(fd_ >= 0)
        ::close(fd_);
}


void Output_File::write_piece(const void* const buf, const std::size_t bytes, const std::size_t off)
{
    if(bytes == 0)  // `sync_file_range` would take an empty range to the end of the file.
        return;

    util::file::write_and_flush(fd_, path_, buf, bytes, off);
#ifdef TESTS
    written_.fetch_add(bytes, std::memory_order_relaxed);
#endif
}


void Output_File::write_edge(const void* const buf, const std::size_t bytes, const std::size_t off)
{
    if(bytes == 0)
        return;

    util::file::write_and_flush(fd_, path_, buf, bytes, off);
#ifdef TESTS
    constexpr auto BLK = Direct_File::DISK_BLOCK;
    if(off / BLK != (off + bytes - 1) / BLK)
    {
        std::cerr << "An edge of the output, [" << off << ", " << (off + bytes) << "), spans disk-blocks. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    written_.fetch_add(bytes, std::memory_order_relaxed);
    lock_.lock();
    edge_bytes_[off / BLK] += bytes;
    lock_.unlock();
#endif
}


void Output_File::close()
{
#ifdef TESTS
    // The pieces and the edges write every byte of the output once, and a disk-block with an edge is the edges' alone.
    if(written_ != bytes_)
    {
        std::cerr << "The output-file " << path_ << " had " << written_ << " bytes written, against its " << bytes_ << ". Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    constexpr auto BLK = Direct_File::DISK_BLOCK;
    for(const auto& blk : edge_bytes_)
        if(blk.second != std::min(BLK, bytes_ - blk.first * BLK))
        {
            std::cerr << "Disk-block " << blk.first << " of the output-file " << path_ << " had " << blk.second << " bytes of edges written. Aborting.\n";
            std::exit(EXIT_FAILURE);
        }
#endif

    // The earlier `sync_file_range`s only queued writebacks, with no guarantees on
    // actual completions. Therefore block until the writes physically complete.
    if(::fsync(fd_) != 0)
        util::file::exit_on_io_failure("fsync", path_);

    if(::close(fd_) != 0)
        util::file::exit_on_io_failure("close", path_);

    fd_ = -1;
}

}
