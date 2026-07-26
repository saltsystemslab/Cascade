
#include "Direct_File.hpp"

#include <cstddef>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <cstdlib>
#include <iostream>


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
        exit("open(O_DIRECT)");
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


void Direct_File::exit(const char* const op) const
{
    std::cerr << op << " failed at filesystem-path: " << path_ << "; error: " << std::strerror(errno) << "\n";
    std::exit(EXIT_FAILURE);
}


void Direct_File::preallocate(const std::size_t bytes)
{
    if(::fallocate(fd_, 0, 0, block_up(bytes)) != 0)
        exit("fallocate");
}


void Direct_File::read_at(void* const buf, const std::size_t bytes, const std::size_t off) const
{
    if(!is_block_aligned(buf) || !is_block_aligned(off))
        exit("O_DIRECT read-buffer / file-block alignment");

    const auto to_read = block_up(bytes);
    const auto read_bytes = ::pread(fd_, buf, to_read, off);
    if(read_bytes < 0 || static_cast<std::size_t>(read_bytes) != to_read)
        exit("pread(O_DIRECT)");
}


void Direct_File::write_at(const void* const buf, const std::size_t bytes, const std::size_t off)
{
    if(!is_block_aligned(buf) || !is_block_aligned(off))
        exit("O_DIRECT write-buffer / file-block alignment");

    const auto to_write = block_up(bytes);
    const auto written_bytes = ::pwrite(fd_, buf, to_write, off);
    if(written_bytes < 0 || static_cast<std::size_t>(written_bytes) != to_write)
        exit("pwrite(O_DIRECT)");
}


void Direct_File::unlink()
{
    if(fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }

    if(::unlink(path_.c_str()) != 0)
        exit("unlink");
}

}
