
#include "Direct_File.hpp"
#include "utility.hpp"

#include <cstdint>
#include <cstddef>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <cstdlib>


namespace gpu_saca
{

bool Direct_File::is_block_aligned(const void* const ptr)
{
    return (reinterpret_cast<std::uintptr_t>(ptr) & (DISK_BLOCK - 1)) == 0;
}


bool Direct_File::is_block_aligned(const std::size_t sz)
{
    return (sz & (DISK_BLOCK - 1)) == 0;
}


Direct_File::Direct_File():
      fd_(-1)
{}


Direct_File::Direct_File(const std::string& path, const Mode mode):
      fd_(-1)
    , path_(path)
{
    int flags = O_DIRECT | O_CREAT;
    switch(mode)
    {
        case Mode::read:        flags |= O_RDONLY;  break;
        case Mode::write:       flags |= O_WRONLY;  break;
        case Mode::read_write:  flags |= O_RDWR;    break;
    }

    fd_ = ::open(path_.c_str(), flags, 0644);   // Only owner can read-write, others read-only.
    if(fd_ < 0)
        util::file::exit_on_io_failure("open(O_DIRECT)", path_);
}


Direct_File::Direct_File(Direct_File&& rhs):
      fd_(-1)
{
    this->operator=(std::move(rhs));
}


Direct_File& Direct_File::operator=(Direct_File&& rhs)
{
    if(this != &rhs)
    {
        if(fd_ >= 0)
            ::close(fd_);

        fd_ = rhs.fd_;
        path_ = std::move(rhs.path_);
        rhs.fd_ = -1;
    }

    return *this;
}


Direct_File::~Direct_File()
{
    if(fd_ >= 0)
        ::close(fd_);
}


void Direct_File::preallocate(const std::size_t bytes)
{
    if(::fallocate(fd_, 0, 0, block_up(bytes)) != 0)
        util::file::exit_on_io_failure("fallocate", path_);
}


void Direct_File::read_at(void* const buf, const std::size_t bytes, const std::size_t off) const
{
    if(!is_block_aligned(buf) || !is_block_aligned(off))
        util::file::exit_on_io_failure("O_DIRECT read-buffer / file-block alignment", path_);

    const auto to_read = block_up(bytes);
    std::size_t read = 0;
    while(read < to_read)   // `pread` reads at most ~2GB.
    {
        const auto read_bytes = ::pread(fd_, static_cast<char*>(buf) + read, to_read - read, off + read);
        if(read_bytes < 0)  util::file::exit_on_io_failure("pread(O_DIRECT)", path_);
        if(read_bytes == 0) util::file::exit_on_io_failure("pread(O_DIRECT): unexpected EOF", path_);   // `errno` may be stale for the EOF case, and thus `strerror(errno)` is not meaningful then.
        read += read_bytes;
    }
}


void Direct_File::write_at(const void* const buf, const std::size_t bytes, const std::size_t off)
{
    if(!is_block_aligned(buf) || !is_block_aligned(off))
        util::file::exit_on_io_failure("O_DIRECT write-buffer / file-block alignment", path_);

    const auto to_write = block_up(bytes);
    std::size_t written = 0;
    while(written < to_write)   // `pwrite` writes at most ~2GB.
    {
        const auto written_bytes = ::pwrite(fd_, static_cast<const char*>(buf) + written, to_write - written, off + written);
        if(written_bytes <= 0)  util::file::exit_on_io_failure("pwrite(O_DIRECT)", path_);
        written += written_bytes;
    }
}


void Direct_File::unlink()
{
    if(fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }

    if(!path_.empty() && ::unlink(path_.c_str()) != 0)
        util::file::exit_on_io_failure("unlink", path_);

    path_.clear();
}

}
