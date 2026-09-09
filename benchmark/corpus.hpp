
#ifndef BENCH_CORPUS_HPP
#define BENCH_CORPUS_HPP



#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>


static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Corpora are little-endian arrays; a big-endian host would silently sort byte-swapped tokens.");


namespace gpu_saca::bench
{

// RAII buffer.
template <typename T_>
struct Buffer
{
    T_* p = nullptr;
    std::size_t n = 0;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    ~Buffer() { std::free(p); }

    bool allocate(const std::size_t count)
    {
        std::free(p);

        p = static_cast<T_*>(std::malloc(static_cast<size_t>(count) * sizeof(T_)));
        n = (p ? count : 0);

        return p != nullptr;
    }
};


// File-properties to cheaply decide whether file-bytes may have changed.
struct File_Prop
{
    int64_t size = 0;
    int64_t mtime = 0;
};


// Reads `n_req` many elements of type `T_sym_` from the file at path `path`, to
// the RAII-buffer `buf`, and fills in the file-properties `f_prop`. Fills in
// `err` with error message, if some error is encountered. If `drop_cache` is
// specified, then requests the OS to drop the page-cache entries for this file.
template <typename T_sym_>
inline bool load_prefix(const std::string& path, const int64_t n_req, Buffer<T_sym_>& buf, File_Prop& f_prop, std::string& err, const bool drop_cache)
{
    const int fd = ::open(path.c_str(), O_RDONLY);
    if(fd < 0)
    {
        err = "open failed: " + path;
        return false;
    }

    struct stat sb;
    if(::fstat(fd, &sb) != 0)
    {
        err = "fstat failed: " + path;
        ::close(fd);
        return false;
    }

    f_prop.size = sb.st_size;
    f_prop.mtime = sb.st_mtime;

    if(drop_cache)
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

    const int64_t n_avail = f_prop.size / sizeof(T_sym_);
    const int64_t n = (n_req <= 0) ? n_avail : std::min(n_req, n_avail);

    if(n <= 0)
    {
        err = "empty corpus: " + path;
        ::close(fd);
        return false;
    }

    if(!buf.allocate(n))
    {
        err = "allocation failed for " + std::to_string(n) + " tokens";
        ::close(fd);
        return false;
    }


    const std::size_t bytes_to_read = static_cast<std::size_t>(n) * sizeof(T_sym_);
    std::size_t bytes_read = 0;
    uint8_t* const dst = reinterpret_cast<uint8_t*>(buf.p);
    while(bytes_read < bytes_to_read)   // `pread` reads at most ~2GB.
    {
        const auto read_bytes = ::pread(fd, dst + bytes_read, bytes_to_read - bytes_read, static_cast<off_t>(bytes_read));
        if(read_bytes < 0)
        {
            err = "pread failed at offset " + std::to_string(bytes_read);
            ::close(fd);
            return false;
        }

        if(read_bytes == 0) // Shorter file than expected.
            break;

        bytes_read += read_bytes;
    }

    ::close(fd);
    buf.n = static_cast<int64_t>(bytes_read / sizeof(T_sym_));

    return buf.n > 0;
}

}



#endif
