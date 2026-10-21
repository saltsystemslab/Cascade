

#ifndef DIRECT_FILE_HPP
#define DIRECT_FILE_HPP



#include "utility.hpp"

#include <cstddef>
#include <string>


namespace gpu_saca
{

// A file-handle supporting block-aligned read-write with `O_DIRECT`.
class Direct_File
{

public:

    enum class Mode { read, write, read_write, };

    // `O_DIRECT` alignment granularity. 4 KBs of alignment covers common NVMe
    // logical-block size.
    static constexpr std::size_t DISK_BLOCK = 4096;

    // Returns `sz` rounded up to the closest multiple of `DISK_BLOCK`.
    static std::size_t block_up(const std::size_t sz) { return util::math::block_up<DISK_BLOCK>(sz); }

    // Returns `true` iff the address `ptr` is disk-block aligned.
    static bool is_block_aligned(const void* ptr);

private:

    int fd_;    // Open file-descriptor.
    std::string path_;  // Filesystem path of the file.

    // Returns `true` iff the size `sz` is disk-block aligned.
    static bool is_block_aligned(std::size_t sz);

public:

    // Constructs a null file-handle.
    Direct_File();

    // Opens a direct-file at `path` with access-mode `mode`.
    Direct_File(const std::string& path, Mode mode);

    Direct_File(const Direct_File&) = delete;
    Direct_File& operator=(const Direct_File&) = delete;

    Direct_File(Direct_File&& rhs);
    Direct_File& operator=(Direct_File&& rhs);

    ~Direct_File();

    // Returns the filesystem path of the file.
    const std::string& path() const { return path_; }

    // Preallocates the file to (block-rounded) `bytes`.
    void preallocate(std::size_t bytes);

    // Empties the file, releasing its space; the file stays open.
    void truncate();

    // Reads `bytes`-many bytes from the file-offset `off` into `buf`. `off` and
    // `buf` must be `DISK_BLOCK`-aligned, and `buf` must have `block_up(bytes)`
    // bytes of capacity, at least.
    void read_at(void* buf, std::size_t bytes, std::size_t off) const;

    // Writes `bytes`-many bytes from `buf` to the file-offset `off`. `off` and
    // `buf` must be `DISK_BLOCK`-aligned, and `buf` must have `block_up(bytes)`
    // bytes of capacity, at least.
    void write_at(const void* buf, std::size_t bytes, std::size_t off);

    // Unlinks the file from the filesystem.
    void unlink();
};

};



#endif