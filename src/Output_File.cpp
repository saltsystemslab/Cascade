
#include "Output_File.hpp"
#include "utility.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <cerrno>
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


Output_File::Output_File(const std::string& path, const std::size_t bytes, const bool direct):
      fd_(-1)
    , path_(path)
    , bytes_(bytes)
    , direct_(direct)
    , written_(0)
{
    constexpr auto flags = O_WRONLY | O_CREAT | O_TRUNC;
    if(direct_)
    {
        fd_ = ::open(path_.c_str(), flags | O_DIRECT, 0644);    // Only owner can read-write, others read-only.
        if(fd_ < 0 && errno == EINVAL)  // The filesystem does not support direct I/O.
        {
            std::cerr << "Warning: no direct writes at filesystem-path: " << path_ << "; writing through the page-cache.\n";
            direct_ = false;
        }
    }

    if(!direct_)
        fd_ = ::open(path_.c_str(), flags, 0644);

    if(fd_ < 0)
        util::file::exit_on_io_failure("open", path_);

    // Under `O_DIRECT`, the last disk-block is written whole, and the file sized to the output at closing.
    if(::fallocate(fd_, 0, 0, direct_ ? Direct_File::block_up(bytes_) : bytes_) != 0)
        util::file::exit_on_io_failure("fallocate", path_);

#ifdef STATS
    std::cerr << "Output-file " << path_ << " is written " << (direct_ ? "with O_DIRECT" : "through the page-cache") << ".\n";
#endif
}


Output_File::~Output_File()
{
    if(fd_ >= 0)
        ::close(fd_);
}


void Output_File::write_direct(const void* const buf, const std::size_t bytes, const std::size_t off)
{
    if(!Direct_File::is_block_aligned(buf) || off % Direct_File::DISK_BLOCK != 0 || bytes % Direct_File::DISK_BLOCK != 0)
        util::file::exit_on_io_failure("O_DIRECT write-buffer / file-block alignment", path_);

    std::size_t written = 0;
    while(written < bytes)  // `pwrite` writes at most ~2GB.
    {
        const auto written_bytes = ::pwrite(fd_, static_cast<const char*>(buf) + written, bytes - written, off + written);
        if(written_bytes <= 0)
            util::file::exit_on_io_failure("pwrite(O_DIRECT)", path_);

        written += written_bytes;
    }
}


void Output_File::write_piece(const void* const buf, const std::size_t bytes, const std::size_t off)
{
    if(bytes == 0)  // `sync_file_range` would take an empty range to the end of the file.
        return;

    if(direct_)
        write_direct(buf, bytes, off);
    else
        util::file::write_and_flush(fd_, path_, buf, bytes, off);

#ifdef TESTS
    written_.fetch_add(bytes, std::memory_order_relaxed);
#endif
}


void Output_File::write_edge(const void* const buf, const std::size_t bytes, const std::size_t off)
{
    if(bytes == 0)
        return;

    constexpr auto BLK = Direct_File::DISK_BLOCK;
    if(off / BLK != (off + bytes - 1) / BLK)
    {
        std::cerr << "An edge of the output, [" << off << ", " << (off + bytes) << "), spans disk-blocks. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    if(direct_)
    {
        lock_.lock();
        std::memcpy(edge_[off / BLK].byte + off % BLK, buf, bytes);
        lock_.unlock();
    }
    else
        util::file::write_and_flush(fd_, path_, buf, bytes, off);

#ifdef TESTS
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

    // Write the assembled edge-blocks, the last one whole, and trim the file to the output.
    if(direct_)
    {
        for(const auto& blk : edge_)
            write_direct(blk.second.byte, Direct_File::DISK_BLOCK, blk.first * Direct_File::DISK_BLOCK);

        if(::ftruncate(fd_, bytes_) != 0)
            util::file::exit_on_io_failure("ftruncate", path_);
    }

    // The earlier `sync_file_range`s only queued writebacks, with no guarantees on
    // actual completions; and the direct writes' extent-updates are not durable
    // yet. Therefore block until the writes physically complete.
    if(::fsync(fd_) != 0)
        util::file::exit_on_io_failure("fsync", path_);

    if(::close(fd_) != 0)
        util::file::exit_on_io_failure("close", path_);

    fd_ = -1;
}

}
