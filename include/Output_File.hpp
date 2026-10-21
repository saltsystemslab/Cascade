#ifndef OUTPUT_FILE_HPP
#define OUTPUT_FILE_HPP



#include "Direct_File.hpp"
#include "Spin_Lock.hpp"

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <string>
#include <map>


namespace gpu_saca
{

// The suffix array's output-file, written in pieces at their file-offsets: with
// `O_DIRECT` where the filesystem allows, else through the page-cache. A
// partition's output-range is cut at disk-block boundaries: into its chunks'
// pieces of the range's block-aligned middle, and the partial blocks at the
// range's two ends, its edges. Under `O_DIRECT`, the edges are assembled in
// memory into their disk-blocks, and written at closing.
class Output_File
{
public:

    // Cut of a chunk's share of its partition's output-range: the chunk's piece,
    // and the range's edges it carries—the first chunk the head, and the last
    // the tail. Absent parts are empty intervals.
    struct Cut
    {
        std::size_t lo, hi; // The piece, `[lo, hi)`.
        std::size_t head_lo, head_hi;   // The head, `[head_lo, head_hi)`.
        std::size_t tail_lo, tail_hi;   // The tail, `[tail_lo, tail_hi)`.
    };

private:

    // A disk-block of the output, assembled from the edges in it.
    struct alignas(Direct_File::DISK_BLOCK) Edge_Block
    {
        uint8_t byte[Direct_File::DISK_BLOCK];
    };

    int fd_;    // Open file-descriptor.
    std::string path_;  // Filesystem path of the file.
    std::size_t bytes_; // Size of the output.
    bool direct_;   // Whether the file is written with `O_DIRECT`.
    util::Spin_Lock lock_;  // Lock over `edge_` and `edge_bytes_`.
    std::map<std::size_t, Edge_Block> edge_;    // Disk-blocks with edges, by block-index, under `O_DIRECT`.
    std::map<std::size_t, std::size_t> edge_bytes_; // Bytes of the edges written to each disk-block, by block-index; for test purposes.
    std::atomic<std::size_t> written_;  // Bytes written in all; for test purposes.


    // Writes `bytes`-many bytes from `buf` to the file-offset `off`, with
    // `O_DIRECT`; `buf`, `bytes`, and `off` must be disk-block aligned.
    void write_direct(const void* buf, std::size_t bytes, std::size_t off);

public:

    // Returns the cut of the `c`'th of `chunk_c` chunks of the output-range
    // `[lo, hi)`, at `chunk_bytes` bytes per chunk, a multiple of
    // `Direct_File::DISK_BLOCK`.
    static Cut cut(std::size_t lo, std::size_t hi, std::size_t chunk_bytes, std::size_t c, std::size_t chunk_c);

    // Opens the file at `path` for an output of `bytes` bytes, truncating it,
    // and preallocates it; with `O_DIRECT` iff `direct` and the filesystem
    // allows.
    Output_File(const std::string& path, std::size_t bytes, bool direct);

    Output_File(const Output_File&) = delete;
    Output_File& operator=(const Output_File&) = delete;

    ~Output_File();

    // Writes the disk-block aligned piece `[off, off + bytes)` of the output
    // from `buf`, likewise aligned.
    void write_piece(const void* buf, std::size_t bytes, std::size_t off);

    // Writes the edge `[off, off + bytes)` of the output from `buf`; under
    // `O_DIRECT`, into its disk-block's assembly.
    void write_edge(const void* buf, std::size_t bytes, std::size_t off);

    // Writes the assembled edges, if any, blocks until the writes physically
    // complete, and closes the file.
    void close();
};

}



#endif
