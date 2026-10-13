#ifndef OUTPUT_FILE_HPP
#define OUTPUT_FILE_HPP



#include "Direct_File.hpp"
#include "Spin_Lock.hpp"

#include <cstddef>
#include <atomic>
#include <string>
#include <map>


namespace gpu_saca
{

// The suffix array's output-file, written in pieces at their file-offsets,
// through the page-cache. A partition's output-range is cut at disk-block
// boundaries: into its chunks' pieces of the range's block-aligned middle, and
// the partial blocks at the range's two ends, its edges.
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

    int fd_;    // Open file-descriptor.
    std::string path_;  // Filesystem path of the file.
    std::size_t bytes_; // Size of the output.
    util::Spin_Lock lock_;  // Lock over `edge_bytes_`.
    std::map<std::size_t, std::size_t> edge_bytes_; // Bytes of the edges written to each disk-block, by block-index; for test purposes.
    std::atomic<std::size_t> written_;  // Bytes written in all; for test purposes.

public:

    // Returns the cut of the `c`'th of `chunk_c` chunks of the output-range
    // `[lo, hi)`, at `chunk_bytes` bytes per chunk, a multiple of
    // `Direct_File::DISK_BLOCK`.
    static Cut cut(std::size_t lo, std::size_t hi, std::size_t chunk_bytes, std::size_t c, std::size_t chunk_c);

    // Opens the file at `path` for an output of `bytes` bytes, truncating it,
    // and preallocates it.
    Output_File(const std::string& path, std::size_t bytes);

    Output_File(const Output_File&) = delete;
    Output_File& operator=(const Output_File&) = delete;

    ~Output_File();

    // Writes the piece `[off, off + bytes)` of the output from `buf`.
    void write_piece(const void* buf, std::size_t bytes, std::size_t off);

    // Writes the edge `[off, off + bytes)` of the output from `buf`.
    void write_edge(const void* buf, std::size_t bytes, std::size_t off);

    // Blocks until the writes physically complete, and closes the file.
    void close();
};

}



#endif
