
#ifndef HOST_SEARCH_HPP
#define HOST_SEARCH_HPP



#include "types.hpp"
#include "query-file.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


// The host's search of a query index whose arrays are in host memory: the
// boundaries of every query's SA-interval, binary-searched over the suffix
// array, within the brackets that the device's sample-table search gives, or
// over the whole suffix array. A thread interleaves the searches of several
// boundaries, prefetching what each one's next probe reads before it probes for
// the others: the cache misses of independent searches overlap, where a single
// search waits on each of its own in turn.

namespace gpu_saca
{

// The ranks `[first, last]` that contain a boundary of an SA-interval.
struct Rank_Range
{
    idx_t first;
    idx_t last;
};


namespace host_search
{

// Most boundary searches a thread interleaves.
constexpr unsigned MAX_LANES = 1024;


// Returns the count of threads that `requested` selects: itself, or OpenMP's
// default if `0`.
unsigned thread_count(unsigned requested);

// Finds into `out[i]` the SA-interval of every query `sym[off[i], off[i + 1])`,
// `i < q`, over the text `T[0, n)` with the 40-bit suffix array `SA`, padded as
// `Suffix_Index` pads them. Position `g` of the batch is the query `ids[g]`, or
// `g` if `ids` is null; its interval's `lo` is in the ranks `br[2g]`, and its
// `hi` in `br[2g + 1]`, or both anywhere in `[0, n]` if `br` is null. With
// `ids`, a query identical to the one at the position before it takes that
// one's interval. `threads` threads search, `0` for OpenMP's default, each
// interleaving `lanes` boundary searches.
template <typename T_sym_>
void find(const T_sym_* T, idx_t n, const uint8_t* SA, const uint64_t* off, const T_sym_* sym, idx_t q, const Rank_Range* br,
          const uint32_t* ids, SA_Interval* out, unsigned threads, unsigned lanes);

// Checks the intervals `I[0, q)` of the queries `sym[off[i], off[i + 1])` with
// the certificate of `ref::certify`, over the same arrays as `find`: adds the
// count of failing queries to `fail_c`, and lowers `first` to the first failing
// one. `threads` threads check, `0` for OpenMP's default.
template <typename T_sym_>
void certify(const T_sym_* T, idx_t n, const uint8_t* SA, const uint64_t* off, const T_sym_* sym, idx_t q, const SA_Interval* I,
             unsigned long long& fail_c, unsigned long long& first, unsigned threads);

// The part files of a mapped suffix array, open for asking the kernel to read
// its pages ahead: a descriptor of every part for each of the threads that ask,
// so that no two threads share a file, and the parts' byte offsets into the
// array.
class SA_Files
{
private:

    std::vector<idx_t> part_off_;   // The parts' byte offsets into the array, in order.
    std::vector<int> fd_;   // `fd_[t * parts + i]`: thread `t`'s descriptor of part `i`.
    unsigned threads_ = 0;


public:

    SA_Files() = default;
    ~SA_Files() { close(); }
    SA_Files(const SA_Files&) = delete;
    SA_Files& operator=(const SA_Files&) = delete;

    // Opens the part files `path`, at the byte offsets `part_off` into the
    // array, read-only, once for each of `threads` threads. Returns `false` with
    // a reason in `err` on failure, leaving none open.
    bool open(const std::vector<std::string>& path, const std::vector<idx_t>& part_off, unsigned threads, std::string& err);

    // Closes the files.
    void close();

    // Returns the count of threads that the files are open for; `0` if closed.
    unsigned threads() const { return threads_; }

    // Returns the count of parts.
    std::size_t parts() const { return part_off_.size(); }

    // Returns thread `t`'s descriptor of part `i`.
    int fd(const unsigned t, const std::size_t i) const { return fd_[t * part_off_.size() + i]; }

    // Returns part `i`'s byte offset into the array.
    idx_t part_off(const std::size_t i) const { return part_off_[i]; }
};


// Most pages of a bracket that `prefetch` asks for whole.
constexpr idx_t PREFETCH_PAGES = 4;

// Asks the kernel to read ahead, without waiting for them, the pages of the
// mapped suffix array of `n` entries in the files `f` that `find` first probes
// in the brackets `br[0, 2q)`: all of a bracket's pages if they are at most
// `PREFETCH_PAGES`, else the page of its first probe. A page marked in `asked`,
// a byte per page, is not asked for again, and those asked for are marked; if
// `asked` is null, every page is asked for. As many threads ask as `f` is open
// for, with their own descriptors. Returns the count of pages asked for; adds
// the count of requests that the kernel refused to `refused`.
idx_t prefetch(const SA_Files& f, idx_t n, const Rank_Range* br, idx_t q, std::atomic<uint8_t>* asked, idx_t& refused);

// Copies `bytes` bytes from `src` to `dst` with `threads` threads, `0` for
// OpenMP's default.
void copy(void* dst, const void* src, std::size_t bytes, unsigned threads);

}

}



#endif
