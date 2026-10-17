
#ifndef HOST_SEARCH_HPP
#define HOST_SEARCH_HPP



#include "types.hpp"
#include "query-file.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
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


// A team of threads for the host's loops over items, the calling thread among
// them: the team's threads take a loop's items in turn, and the loop is done
// once its items are, however many of the threads took part. A thread with no
// item to take waits, spinning for `GPU_SACA_HOST_SPIN_RUNNING_US` microseconds
// (default 200) while the loop runs on, or for `GPU_SACA_HOST_SPIN_IDLE_US`
// (default 2000) once the team is idle, for the next loop, and then sleeps. So
// a thread that another task keeps from its core holds up no loop but one whose
// item it is in the middle of, and that one only until the kernel moves it to
// a core that the sleepers leave idle. A libgomp team instead waits for every
// one of its threads at every loop's end, the others spinning there for 300,000
// pauses, milliseconds, before they sleep, which keeps every core busy: a task
// beside a team of every core then holds up each of its loops by about as long,
// ten times and more a loop of the host's search.
class Team
{
public:

    class Claim;


private:

    using Body = void (*)(void* f, unsigned t, Claim& c);  // Runs the loop's body `f` as thread `t`, over the items it takes of `c`.

    std::vector<std::thread> worker_;   // The team's threads but the caller: thread `t` is `worker_[t - 1]`.
    std::mutex run_lock_;   // Held by `run`: a loop at a time.
    uint32_t last_ = 0; // The number of the last loop run; loops are numbered from 1, cyclically.

    // The current loop, as the threads take part in it: its number, in the bits of `next_` from `NUMBER_SHIFT`, and its next item to
    // take, in those below; the number of the loop that the description in `items_`, `body_`, and `f_` is of, `0` while it changes.
    alignas(64) std::atomic<uint64_t> next_{0};
    alignas(64) std::atomic<uint32_t> loop_{0};
    std::atomic<idx_t> items_{0};
    std::atomic<Body> body_{nullptr};
    std::atomic<void*> f_{nullptr};

    alignas(64) std::atomic<idx_t> done_{0};    // The current loop's items done.
    std::atomic<uint32_t> finished_{0}; // The number of the last loop done, which the caller waits for.
    std::atomic<uint32_t> caller_asleep_{0};    // Whether the caller sleeps, waiting.
    alignas(64) std::atomic<uint32_t> started_{0};  // The number of the last loop started, which the other threads wait for.
    std::atomic<uint32_t> asleep_{0};   // Those of them asleep, waiting.
    std::atomic<bool> stop_{false};

    // Starts the loop of the body `f`, run by `body`, over `items` items.
    // Returns its number.
    uint32_t start(idx_t items, Body body, void* f);

    // Takes part in the current loop as thread `t`, if an item of it is left.
    void take_part(unsigned t);

    // Waits for the loop numbered `s` to be done.
    void wait_for(uint32_t s);

    // Runs thread `t` of the team, loop after loop, until the team stops.
    void work(unsigned t);


public:

    // The items of a loop that a thread takes.
    class Claim
    {
        friend class Team;

    private:

        Team& team_;
        const idx_t items_; // The loop's items.
        idx_t first_;   // The first item taken, until `next` returns it.
        idx_t taken_ = 1;   // The items taken, the first among them.

        Claim(Team& team, const idx_t items, const idx_t first) : team_(team), items_(items), first_(first) {}


    public:

        static constexpr idx_t NONE = ~idx_t(0);

        // Returns the next item that the thread takes, or `NONE` once the
        // loop has none left.
        idx_t next();
    };


    // Starts a team of `threads` threads, the caller's among them.
    explicit Team(unsigned threads);
    ~Team();
    Team(const Team&) = delete;
    Team& operator=(const Team&) = delete;

    // Returns the count of the team's threads, the caller's among them.
    unsigned threads() const { return static_cast<unsigned>(worker_.size()) + 1; }

    // Runs `f(t, c)` as threads `t` of the team, the calling thread as thread
    // `0`, over the items `[0, items)`: a thread that takes part takes items
    // from `c.next()` until it returns `Claim::NONE`, and is done with them
    // once `f` returns. Returns once every item is done. The loops of
    // concurrent calls run one after another; `f` must not run one itself.
    template <typename F_>
    void run(const idx_t items, F_& f)
    {
        if(items == 0)
            return;

        const std::lock_guard<std::mutex> guard(run_lock_);
        const uint32_t s = start(items, [](void* const g, const unsigned t, Claim& c) { (*static_cast<F_*>(g))(t, c); },
                                 const_cast<void*>(static_cast<const void*>(&f)));
        take_part(0);
        wait_for(s);
    }
};


// Returns the process's team of `threads` threads, started by the first call
// for that count: the host's loops of the same count of threads all run on the
// same threads.
Team& team(unsigned threads);

// Finds into `out[i]` the SA-interval of every query `sym[off[i], off[i + 1])`,
// `i < q`, over the text `T[0, n)` with the 40-bit suffix array `SA`, padded as
// `Suffix_Index` pads them. Position `g` of the batch is the query `ids[g]`, or
// `g` if `ids` is null; its interval's `lo` is in the ranks `br[2g]`, and its
// `hi` in `br[2g + 1]`, or both anywhere in `[0, n]` if `br` is null. With
// `ids`, a query identical to the one at the position before it takes that
// one's interval. The process's team of `threads` threads searches, `0` for
// OpenMP's default count, each thread interleaving `lanes` boundary searches.
template <typename T_sym_>
void find(const T_sym_* T, idx_t n, const uint8_t* SA, const uint64_t* off, const T_sym_* sym, idx_t q, const Rank_Range* br,
          const uint32_t* ids, SA_Interval* out, unsigned threads, unsigned lanes);

// Checks the intervals `I[0, q)` of the queries `sym[off[i], off[i + 1])` with
// the certificate of `ref::certify`, over the same arrays as `find`: adds the
// count of failing queries to `fail_c`, and lowers `first` to the first failing
// one. The process's team of `threads` threads checks, `0` for OpenMP's
// default count.
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
// `asked` is null, every page is asked for. The process's team of as many
// threads as `f` is open for asks, each with its own descriptors. Returns the
// count of pages asked for; adds the count of requests that the kernel refused
// to `refused`.
idx_t prefetch(const SA_Files& f, idx_t n, const Rank_Range* br, idx_t q, std::atomic<uint8_t>* asked, idx_t& refused);

// Copies `bytes` bytes from `src` to `dst` with the process's team of
// `threads` threads, `0` for OpenMP's default count.
void copy(void* dst, const void* src, std::size_t bytes, unsigned threads);

}

}



#endif
