
#include "corpus.hpp"

#include <string>
#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>


namespace gpu_saca::bench
{

using Clock = std::chrono::steady_clock;

inline auto since(const Clock::time_point& t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }


struct Args
{
    std::string mode;   // Run or verify.
    std::string input;  // Input path.
    std::string sa_out; // Output path.
    std::string corpus ;    // Label of the input.
    std::string cache = "as-is";    // What to do with the page-cache after the run: "warm", "cold", or "as-is".
    std::string results_dir = ".";  // Which directory to put the records in.
    std::string run_id; // User-supplied ID for the run, overloading the auto-generated one.

    int64_t n = 0;  // Number of tokens to read. Optional; reads whole file when absent or is `< 0`.
    int64_t threads = 0;    // Number of CPU-threads to use; `0` uses `OMP`-default.
    int64_t fs = 0; // `libsais` free-space. Unused currently.

    // Verify-only arguments.
    std::string sa; // SA file to verify.
    std::string against;    // Optional second SA file to crosscheck against.
};


inline bool write_SA(const std::string& path, const uint64_t* const SA, int64_t n, std::string& err)
{
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(fd < 0)
    {
        err = "cannot open for write: " + path;
        return false;
    }


    const auto bytes_to_write = static_cast<size_t>(n) * sizeof(int64_t);
    std::size_t bytes_written = 0;
    while(bytes_written < bytes_to_write)   // `pwrite` writes at most ~2GB.
    {
        const auto written_bytes = ::pwrite(fd, reinterpret_cast<const char*>(SA) + bytes_written, bytes_to_write - bytes_written, bytes_written);
        if(written_bytes <= 0)
        {
            err = "pwrite failed at offset " + std::to_string(bytes_written);
            ::close(fd);
            return false;
        }

        bytes_written += written_bytes;
    }

    const bool ok = (::close(fd) == 0);
    if(!ok)
        err = "close failed: " + path;

    return ok;
}


inline bool read_SA(const std::string& path, gpu_saca::bench::Buffer<uint64_t>& SA, const int64_t n_expect, std::string& err)
{
    const int fd = ::open(path.c_str(), O_RDONLY);
    if(fd < 0)
    {
        err = "cannot open: " + path;
        return false;
    }

    const off_t bytes = ::lseek(fd, 0, SEEK_END);
    const int64_t n = static_cast<int64_t>(bytes) / static_cast<int64_t>(sizeof(uint64_t));

    if(n_expect > 0 && n != n_expect)
    {
        err = "SA length " + std::to_string(n) + " != expected " + std::to_string(n_expect);
        ::close(fd);
        return false;
    }

    if(!SA.allocate(n))
    {
        err = "allocation failed for SA of " + std::to_string(n);
        ::close(fd);
        return false;
    }


    const std::size_t bytes_to_read = static_cast<std::size_t>(n) * sizeof(uint64_t);
    std::size_t bytes_read = 0;
    uint8_t* const dst = reinterpret_cast<uint8_t*>(SA.p);

    while(bytes_read < bytes_to_read)   // `pread` reads at most ~2GB.
    {
        const auto read_bytes = ::pread(fd, dst + bytes_read, bytes_to_read - bytes_read, static_cast<off_t>(bytes_read));
        if(read_bytes <= 0)
        {
            err = "pread failed at offset " + std::to_string(bytes_read);
            ::close(fd);
            return false;
        }

        bytes_read += read_bytes;
    }

    ::close(fd);
    SA.n = n;

    return true;
}

}
