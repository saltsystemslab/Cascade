
#include "Host_Search.hpp"
#include "CPU_Reference.hpp"
#include "env.hpp"
#include "sa40.hpp"

#include <omp.h>

#include <immintrin.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>


namespace gpu_saca
{

namespace host_search
{

namespace
{

constexpr idx_t LINE_BYTES = 64;    // Bytes of a cache line.
constexpr idx_t WINDOW_BYTES = 512; // Suffix-array bytes of a search's remaining ranks that are prefetched together, at most.
constexpr idx_t BLOCK_MIN = 32; // Least positions of a block of a batch, a thread's unit of work.
constexpr idx_t BLOCK_MAX = 1024;   // Most positions of a block.
constexpr idx_t BLOCKS_PER_THREAD = 16; // Blocks a batch splits into per thread, within the bounds of a block.
constexpr std::size_t COPY_PIECE = std::size_t(1) << 16;    // Bytes a thread copies at a time: a chunk's few MB spread over every thread.

constexpr unsigned NUMBER_SHIFT = 40;   // The low bits of `Team::next_`, the next item to take; the loop's number is in the rest.
constexpr idx_t ITEM_MASK = (idx_t(1) << NUMBER_SHIFT) - 1;
constexpr uint32_t NUMBER_MAX = (uint32_t(1) << (64 - NUMBER_SHIFT)) - 1;  // The loops' numbers cycle through `[1, NUMBER_MAX]`.
// How long a team's thread spins, waiting, before it sleeps: while a loop is still running, for a thread busy with an item of it, which
// the kernel may have displaced from its core, and the sleepers' idle cores then take in; and once the team is idle, for the next loop.
const auto SPIN_RUNNING = std::chrono::microseconds(env::get_u64("GPU_SACA_HOST_SPIN_RUNNING_US").value_or(200));
const auto SPIN_IDLE = std::chrono::microseconds(env::get_u64("GPU_SACA_HOST_SPIN_IDLE_US").value_or(2000));


// Sleeps until the word `w` is no longer `v`, or a wake-up.
void futex_wait(std::atomic<uint32_t>& w, const uint32_t v)
{
    ::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&w), FUTEX_WAIT_PRIVATE, v, nullptr, nullptr, 0);
}


// Wakes the threads asleep on the word `w`.
void futex_wake(std::atomic<uint32_t>& w)
{
    ::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&w), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}


// Waits until `ready()`, which a change of the word `w` turns true: spinning
// for `SPIN_RUNNING` while `running()`, or for `SPIN_IDLE` from when it is not,
// and then asleep on `w`, counted in `asleep` meanwhile. A thread that changes
// `w` wakes the sleepers if `asleep` counts any.
template <typename F_, typename G_>
void await(std::atomic<uint32_t>& w, std::atomic<uint32_t>& asleep, const F_& ready, const G_& running)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto idle = clock::time_point::max();   // When the team was first seen idle.
    for(unsigned k = 1; !ready(); ++k)
    {
        _mm_pause();
        if(k % 64 == 0)
        {
            const auto t = clock::now();
            if(running() ? t - t0 >= SPIN_RUNNING : t - (idle = std::min(idle, t)) >= SPIN_IDLE)
            {
                for(uint32_t v = w.load(); !ready(); v = w.load())
                {
                    asleep.fetch_add(1);
                    if(!ready())    // A change past it has seen the count, or this sees the change.
                        futex_wait(w, v);
                    asleep.fetch_sub(1);
                }

                return;
            }
        }
    }
}


// Prefetches the cache line of the address `p`.
inline void prefetch(const void* const p)
{
    __builtin_prefetch(p, 0, 3);
}


// Returns the entry at rank `r` of the suffix array `SA`, with an 8-byte read:
// the suffix array's padding covers the read past its last entry.
inline idx_t entry(const uint8_t* const SA, const idx_t r)
{
    uint64_t v;
    std::memcpy(&v, SA + SA_WIDTH * r, sizeof(v));
    return v & SA_ENTRY_MASK;   // The host is little-endian: the entry is in the low bytes.
}


// Prefetches the lines of the suffix array `SA` that reading the entries at
// ranks `[r0, r1)` touches.
inline void prefetch_entries(const uint8_t* const SA, const idx_t r0, const idx_t r1)
{
    for(idx_t b = (SA_WIDTH * r0) & ~(LINE_BYTES - 1); b < SA_WIDTH * r1 + (sizeof(uint64_t) - SA_WIDTH); b += LINE_BYTES)
        prefetch(SA + b);
}


// Prefetches the text that comparing the suffix at `pos` of the text `T[0, n)`
// against a pattern of length `m` reads first: the line of the suffix's first
// symbol, and that of its last one compared within a line's span.
template <typename T_sym_>
inline void prefetch_suffix(const T_sym_* const T, const idx_t n, const idx_t pos, const idx_t m)
{
    constexpr idx_t spl = LINE_BYTES / sizeof(T_sym_);  // Symbols per line.
    const idx_t len = std::min(std::min(n - pos, m), spl);
    if(len > 0)
    {
        prefetch(T + pos);
        prefetch(T + pos + len - 1);
    }
}


// Returns the order of the suffix at `pos` of the text `T[0, n)`, truncated to
// `m` symbols, against the pattern `P[0, m)`, as `ref::cmp_suffix_pattern`
// does; comparing a word of symbols at a time.
template <typename T_sym_>
inline int cmp(const T_sym_* const T, const idx_t n, const idx_t pos, const T_sym_* const P, const idx_t m)
{
    constexpr idx_t spw = sizeof(uint64_t) / sizeof(T_sym_);   // Symbols per word.
    const T_sym_* const S = T + pos;
    const idx_t len = std::min(n - pos, m);
    idx_t k = 0;
    for(; k + spw <= len; k += spw)
    {
        uint64_t a, b;
        std::memcpy(&a, S + k, sizeof(a));
        std::memcpy(&b, P + k, sizeof(b));
        if(a != b)  // The host is little-endian: the lowest differing bit is in the first differing symbol.
        {
            const idx_t d = k + static_cast<idx_t>(__builtin_ctzll(a ^ b)) / (8 * sizeof(T_sym_));
            return S[d] < P[d] ? -1 : 1;
        }
    }

    for(; k < len; ++k)
        if(S[k] != P[k])
            return S[k] < P[k] ? -1 : 1;

    return len < m ? -1 : 0;
}


// A batch searched on the host, and the arrays it is searched over, as `find`
// takes them.
template <typename T_sym_>
struct Batch
{
    const T_sym_* T;
    idx_t n;
    const uint8_t* SA;
    const uint64_t* off;
    const T_sym_* sym;
    const Rank_Range* br;
    const uint32_t* ids;
    SA_Interval* out;
};


// What the next visit to a boundary search reads: nothing, as none is in
// flight; its probe's suffix-array entry; or its probe's suffix.
enum class Stage : uint8_t
{
    idle,
    entry,
    suffix,
};


// A boundary search in flight: the boundary is in the ranks `[low, low + len]`,
// and the search probes the rank `mid`.
template <typename T_sym_>
struct Lane
{
    const T_sym_* P;    // The pattern.
    idx_t m;    // Its length.
    idx_t low;
    idx_t len;
    idx_t mid;
    idx_t pos;  // The probe's suffix, once its entry is read.
    idx_t* dst; // The boundary's place in its interval.
    bool upper; // Whether the boundary is the interval's `hi`, the first rank whose truncated suffix is greater than the pattern; else its `lo`, the first not smaller.
    bool window;    // Whether the entries of the ranks left to probe are prefetched.
    Stage stage = Stage::idle;
};


// Prefetches the suffix-array entry of the probe of the lane `l`, or those of
// every rank it has left to probe once their bytes fit in a window; the lane
// reads the entry at its next visit.
template <typename T_sym_>
inline void aim(Lane<T_sym_>& l, const uint8_t* const SA)
{
    if(SA_WIDTH * l.len <= WINDOW_BYTES)
    {
        prefetch_entries(SA, l.low, l.low + l.len);
        l.window = true;
    }
    else
        prefetch_entries(SA, l.mid, l.mid + 1);

    l.stage = Stage::entry;
}


// Advances the search of the lane `l` over the batch `x` by a visit: reads its
// probe's entry and prefetches the probe's suffix, or compares the suffix and
// moves to the next probe, whose suffix it prefetches right away if the window
// holds its entry. Returns `false` once the search is done, its boundary
// written.
template <typename T_sym_>
inline bool step(Lane<T_sym_>& l, const Batch<T_sym_>& x)
{
    if(l.stage == Stage::entry)
    {
        l.pos = entry(x.SA, l.mid);
        prefetch_suffix(x.T, x.n, l.pos, l.m);
        l.stage = Stage::suffix;
        return true;
    }

    const int c = cmp(x.T, x.n, l.pos, l.P, l.m);
    const idx_t half = l.mid - l.low;
    if(l.upper ? c <= 0 : c < 0)
    {
        l.low = l.mid + 1;
        l.len -= half + 1;
    }
    else
        l.len = half;

    if(l.len == 0)
    {
        *l.dst = l.low;
        l.stage = Stage::idle;
        return false;
    }

    l.mid = l.low + (l.len >> 1);
    if(l.window)
    {
        l.pos = entry(x.SA, l.mid);
        prefetch_suffix(x.T, x.n, l.pos, l.m);
    }
    else
        aim(l, x.SA);

    return true;
}


// Returns `true` iff the queries `i` and `j` of the batch `x` are identical.
template <typename T_sym_>
inline bool same_query(const Batch<T_sym_>& x, const idx_t i, const idx_t j)
{
    const idx_t m = x.off[i + 1] - x.off[i];
    return x.off[j + 1] - x.off[j] == m && std::equal(x.sym + x.off[i], x.sym + x.off[i] + m, x.sym + x.off[j]);
}


// Searches blocks of `block` positions of the batch `x` of `q` positions over
// the lanes `lane`, taking the next block of `claim` whenever the current one
// has no boundary left to start, until none is left: a lane starts on the next
// boundary as soon as it is done with one, across blocks. With the positions
// sorted, `dup[g]` marks a position `g` whose query repeats the one before it
// in its block, and which then takes that one's interval once the thread's
// searches are done.
template <typename T_sym_>
void search_blocks(const Batch<T_sym_>& x, const idx_t q, const idx_t block, Team::Claim& claim, std::vector<Lane<T_sym_>>& lane,
                   uint8_t* const dup)
{
    idx_t g0 = 0;   // The current block's first position.
    idx_t next = 0; // The next boundary: the `hi` of the position `next / 2` if `next` is odd, else its `lo`.
    idx_t end = 0;  // The current block's end, in boundaries.
    bool left = true;   // Whether blocks may be left to take.
    std::vector<idx_t> taken;   // The blocks the thread searched.
    std::size_t busy = 0;   // Lanes with a search in flight.

    // Starts the lane `l` on the next boundary with ranks to probe; those without resolve on the way.
    const auto start = [&](Lane<T_sym_>& l)
    {
        for(;;)
        {
            if(next == end)
            {
                const idx_t k = (left ? claim.next() : Team::Claim::NONE);
                if(k == Team::Claim::NONE)
                {
                    left = false;
                    return;
                }

                taken.push_back(k);
                g0 = k * block;
                next = 2 * g0;
                end = 2 * std::min(q, g0 + block);
            }

            const idx_t b = next++;
            const idx_t g = b >> 1;
            const bool upper = b & 1;
            const idx_t i = (x.ids != nullptr ? x.ids[g] : g);
            if(x.ids != nullptr && !upper)
            {
                dup[g] = (g > g0 && same_query(x, i, x.ids[g - 1]));
                if(dup[g])
                {
                    ++next; // And its `hi`.
                    continue;
                }
            }

            const Rank_Range r = (x.br != nullptr ? x.br[b] : Rank_Range{0, x.n});
            idx_t* const dst = (upper ? &x.out[i].hi : &x.out[i].lo);
            if(r.last == r.first)
            {
                *dst = r.first;
                continue;
            }

            l.P = x.sym + x.off[i];
            l.m = x.off[i + 1] - x.off[i];
            l.low = r.first;
            l.len = r.last - r.first;
            l.mid = l.low + (l.len >> 1);
            l.dst = dst;
            l.upper = upper;
            l.window = false;
            aim(l, x.SA);
            ++busy;
            return;
        }
    };

    for(auto& l : lane)
        l.stage = Stage::idle;

    do
        for(auto& l : lane)
        {
            if(l.stage != Stage::idle && !step(l, x))
                --busy;
            if(l.stage == Stage::idle)
                start(l);
        }
    while(busy > 0);

    if(x.ids != nullptr)    // In order: the position before a repeat has its interval by then.
        for(const idx_t k : taken)
            for(idx_t g = k * block + 1; g < std::min(q, (k + 1) * block); ++g)
                if(dup[g])
                    x.out[x.ids[g]] = x.out[x.ids[g - 1]];
}

}


unsigned thread_count(const unsigned requested)
{
    return requested > 0 ? requested : static_cast<unsigned>(std::max(1, omp_get_max_threads()));
}


idx_t Team::Claim::next()
{
    if(first_ != NONE)
    {
        const idx_t k = first_;
        first_ = NONE;
        return k;
    }

    // The loop is not done while this thread holds its first item: the number of the loop that this takes from is its own.
    const idx_t k = team_.next_.fetch_add(1, std::memory_order_relaxed) & ITEM_MASK;
    if(k >= items_)
        return NONE;

    ++taken_;
    return k;
}


Team::Team(const unsigned threads)
{
    for(unsigned t = 1; t < threads; ++t)
        worker_.emplace_back([this, t] { work(t); });
}


Team::~Team()
{
    stop_.store(true);
    started_.fetch_add(1);
    futex_wake(started_);
    for(auto& w : worker_)
        w.join();
}


uint32_t Team::start(const idx_t items, const Body body, void* const f)
{
    if(items > ITEM_MASK)
    {
        std::fprintf(stderr, "A loop of %llu items exceeds a team's most, %llu. Aborting.\n", static_cast<unsigned long long>(items),
                     static_cast<unsigned long long>(ITEM_MASK));
        std::exit(EXIT_FAILURE);
    }

    // The description first, and then the items open, numbered with the loop: a thread late for the last loop may take an item of
    // this one, which then counts only if the description is of the same loop, read whole.
    const uint32_t s = last_ % NUMBER_MAX + 1;
    last_ = s;
    loop_.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    items_.store(items, std::memory_order_relaxed);
    body_.store(body, std::memory_order_relaxed);
    f_.store(f, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    loop_.store(s, std::memory_order_release);
    next_.store(uint64_t(s) << NUMBER_SHIFT, std::memory_order_release);

    started_.store(s);
    if(asleep_.load() > 0)
        futex_wake(started_);

    return s;
}


void Team::take_part(const unsigned t)
{
    // An item taken counts if it is of the loop that the description is of, read whole between two equal numbers, and one of its
    // items; else the loop it is of is done, or its description is changing for the next one, which this thread then leaves.
    const uint64_t c = next_.fetch_add(1, std::memory_order_acq_rel);
    const uint32_t s = static_cast<uint32_t>(c >> NUMBER_SHIFT);
    const idx_t k = c & ITEM_MASK;
    const uint32_t s0 = loop_.load(std::memory_order_acquire);
    const idx_t items = items_.load(std::memory_order_relaxed);
    const Body body = body_.load(std::memory_order_relaxed);
    void* const f = f_.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if(s0 != s || loop_.load(std::memory_order_relaxed) != s || k >= items)
        return;

    Claim claim(*this, items, k);
    body(f, t, claim);
    if(done_.fetch_add(claim.taken_, std::memory_order_acq_rel) + claim.taken_ == items)
    {
        finished_.store(s);
        if(caller_asleep_.load() > 0)
            futex_wake(finished_);
    }
}


void Team::wait_for(const uint32_t s)
{
    await(finished_, caller_asleep_, [&] { return finished_.load() == s; }, [] { return true; });
}


void Team::work(const unsigned t)
{
    for(uint32_t seen = 0; ; )
    {
        await(started_, asleep_, [&] { return started_.load() != seen; }, [&] { return finished_.load() != started_.load(); });
        seen = started_.load();
        if(stop_.load())
            return;

        take_part(t);
    }
}


Team& team(const unsigned threads)
{
    // The teams last as long as the process, their threads asleep through its exit: no static object outlives them.
    static std::mutex lock;
    static auto& teams = *new std::map<unsigned, std::unique_ptr<Team>>();
    const std::lock_guard<std::mutex> guard(lock);
    auto& t = teams[std::max(1u, threads)];
    if(t == nullptr)
        t = std::make_unique<Team>(std::max(1u, threads));

    return *t;
}


template <typename T_sym_>
void find(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off, const T_sym_* const sym, const idx_t q,
          const Rank_Range* const br, const uint32_t* const ids, SA_Interval* const out, const unsigned threads, const unsigned lanes)
{
    if(q == 0)
        return;

    // Blocks of positions, many per thread, which the threads take in turn: the last ones even out the threads' ends.
    const idx_t t = thread_count(threads);
    const idx_t block = std::min(BLOCK_MAX, std::max(BLOCK_MIN, q / (BLOCKS_PER_THREAD * t)));
    const Batch<T_sym_> x{T, n, SA, off, sym, br, ids, out};
    std::vector<uint8_t> dup(ids != nullptr ? q : 0);
    const auto search = [&](unsigned, Team::Claim& claim)
    {
        std::vector<Lane<T_sym_>> lane(std::max(1u, lanes));
        search_blocks(x, q, block, claim, lane, dup.data());
    };

    team(static_cast<unsigned>(t)).run((q + block - 1) / block, search);
}


template <typename T_sym_>
void certify(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off, const T_sym_* const sym, const idx_t q,
             const SA_Interval* const I, unsigned long long& fail_c, unsigned long long& first, const unsigned threads)
{
    const idx_t t = thread_count(threads);
    const idx_t block = std::min(BLOCK_MAX, std::max(BLOCK_MIN, q / (BLOCKS_PER_THREAD * t)));
    std::atomic<unsigned long long> c(0), f(ULLONG_MAX);
    const auto check = [&](unsigned, Team::Claim& claim)
    {
        unsigned long long c_t = 0, f_t = ULLONG_MAX;
        for(idx_t k; (k = claim.next()) != Team::Claim::NONE; )
            for(idx_t i = k * block; i < std::min(q, (k + 1) * block); ++i)
                if(!ref::certify(T, n, SA, sym + off[i], off[i + 1] - off[i], I[i]))
                {
                    ++c_t;
                    f_t = std::min<unsigned long long>(f_t, i);
                }

        c.fetch_add(c_t);
        for(unsigned long long f0 = f.load(); f_t < f0 && !f.compare_exchange_weak(f0, f_t); ) {}
    };

    team(static_cast<unsigned>(t)).run((q + block - 1) / block, check);
    fail_c += c.load();
    first = std::min(first, f.load());
}


bool SA_Files::open(const std::vector<std::string>& path, const std::vector<idx_t>& part_off, const unsigned threads, std::string& err)
{
    close();
    if(path.empty() || path.size() != part_off.size() || threads == 0)
    {
        err = "the files of a suffix array open for a thread or more, with an offset for each";
        return false;
    }

    fd_.reserve(std::size_t(threads) * path.size());
    for(unsigned t = 0; t < threads; ++t)
        for(const auto& p : path)
        {
            const int d = ::open(p.c_str(), O_RDONLY);
            if(d < 0)
            {
                err = "cannot open " + p + ": " + std::strerror(errno);
                close();
                return false;
            }

            fd_.push_back(d);
        }

    part_off_ = part_off;
    threads_ = threads;
    return true;
}


void SA_Files::close()
{
    for(const int d : fd_)
        ::close(d);

    fd_.clear();
    part_off_.clear();
    threads_ = 0;
}


idx_t prefetch(const SA_Files& f, const idx_t n, const Rank_Range* const br, const idx_t q, std::atomic<uint8_t>* const asked, idx_t& refused)
{
    if(q == 0 || n == 0 || f.threads() == 0)
        return 0;

    const idx_t page = static_cast<idx_t>(::sysconf(_SC_PAGESIZE));
    const idx_t last_page = (SA_WIDTH * n - 1) / page;  // The suffix array's last; the padding past it needs no reading.
    const std::size_t parts = f.parts();

    // Returns in `[p0, p1]` the pages that the search of the ranks `r` reads: the entries of `r.first` to `r.last - 1`, 8 bytes at a
    // time, or those of its first probe if more than `PREFETCH_PAGES`; `p0 > p1` if none.
    const auto pages = [&](const Rank_Range& r, idx_t& p0, idx_t& p1)
    {
        p0 = 1, p1 = 0;
        if(r.last <= r.first)
            return;

        p0 = SA_WIDTH * r.first / page;
        p1 = std::min(last_page, (SA_WIDTH * r.last + (sizeof(uint64_t) - SA_WIDTH) - 1) / page);
        if(p1 - p0 + 1 > PREFETCH_PAGES)
        {
            const idx_t mid = r.first + ((r.last - r.first) >> 1);  // As `find` probes first.
            p0 = SA_WIDTH * mid / page;
            p1 = std::min(last_page, (SA_WIDTH * mid + sizeof(uint64_t) - 1) / page);
        }
    };

    const auto unasked = [&](const idx_t p) { return asked == nullptr || asked[p].load(std::memory_order_relaxed) == 0; };
    const idx_t block = std::min(BLOCK_MAX, std::max(BLOCK_MIN, q / (BLOCKS_PER_THREAD * f.threads())));
    std::atomic<idx_t> asked_c(0), refused_c(0);
    const auto ask_blocks = [&](const unsigned t, Team::Claim& claim)   // The threads of `find`, each with its own descriptors.
    {
        idx_t asked_t = 0, refused_t = 0;

        // Asks for the pages of `[p0, p1]` not yet asked for, in runs within a part: pages do not straddle parts.
        const auto ask = [&](idx_t p0, const idx_t p1)
        {
            for(std::size_t i = 0; p0 <= p1; )
            {
                if(!unasked(p0))
                {
                    ++p0;
                    continue;
                }

                while(i + 1 < parts && f.part_off(i + 1) <= p0 * page)
                    ++i;

                const idx_t end = std::min(p1 + 1, i + 1 < parts ? f.part_off(i + 1) / page : last_page + 1);
                idx_t e = p0;
                for(; e < end && unasked(e); ++e)
                    if(asked != nullptr)
                        asked[e].store(1, std::memory_order_relaxed);

                if(e == p0) // Another thread asked for it meanwhile; an empty request would ask for the rest of the file.
                {
                    ++p0;
                    continue;
                }

                asked_t += e - p0;
                if(::posix_fadvise(f.fd(t, i), static_cast<off_t>(p0 * page - f.part_off(i)), static_cast<off_t>((e - p0) * page),
                                   POSIX_FADV_WILLNEED) != 0)
                    ++refused_t;

                p0 = e;
            }
        };

        for(idx_t k; (k = claim.next()) != Team::Claim::NONE; )
            for(idx_t g = k * block; g < std::min(q, (k + 1) * block); ++g)
            {
                idx_t a0, a1, b0, b1;
                pages(br[2 * g], a0, a1);
                pages(br[2 * g + 1], b0, b1);
                if(a0 <= a1 && b0 <= b1 && b0 <= a1 + 1 && a0 <= b1 + 1)   // The boundaries' pages overlap or abut: one request.
                {
                    a0 = std::min(a0, b0), a1 = std::max(a1, b1);
                    b0 = 1, b1 = 0;
                }

                if(a0 <= a1)
                    ask(a0, a1);
                if(b0 <= b1)
                    ask(b0, b1);
            }

        asked_c.fetch_add(asked_t, std::memory_order_relaxed);
        refused_c.fetch_add(refused_t, std::memory_order_relaxed);
    };

    team(f.threads()).run((q + block - 1) / block, ask_blocks);
    refused += refused_c.load();
    return asked_c.load();
}


void copy(void* const dst, const void* const src, const std::size_t bytes, const unsigned threads)
{
    const std::size_t pieces = (bytes + COPY_PIECE - 1) / COPY_PIECE;
    if(pieces <= 1)
    {
        if(bytes > 0)
            std::memcpy(dst, src, bytes);
        return;
    }

    const auto copy_pieces = [&](unsigned, Team::Claim& claim)
    {
        for(idx_t p; (p = claim.next()) != Team::Claim::NONE; )
        {
            const std::size_t b = p * COPY_PIECE;
            std::memcpy(static_cast<uint8_t*>(dst) + b, static_cast<const uint8_t*>(src) + b, std::min(COPY_PIECE, bytes - b));
        }
    };

    team(thread_count(threads)).run(pieces, copy_pieces);
}



// Template instantiations for the required instances.
template void find(const uint8_t*, idx_t, const uint8_t*, const uint64_t*, const uint8_t*, idx_t, const Rank_Range*, const uint32_t*, SA_Interval*,
                   unsigned, unsigned);
template void find(const uint16_t*, idx_t, const uint8_t*, const uint64_t*, const uint16_t*, idx_t, const Rank_Range*, const uint32_t*, SA_Interval*,
                   unsigned, unsigned);
template void find(const uint32_t*, idx_t, const uint8_t*, const uint64_t*, const uint32_t*, idx_t, const Rank_Range*, const uint32_t*, SA_Interval*,
                   unsigned, unsigned);

template void certify(const uint8_t*, idx_t, const uint8_t*, const uint64_t*, const uint8_t*, idx_t, const SA_Interval*, unsigned long long&,
                      unsigned long long&, unsigned);
template void certify(const uint16_t*, idx_t, const uint8_t*, const uint64_t*, const uint16_t*, idx_t, const SA_Interval*, unsigned long long&,
                      unsigned long long&, unsigned);
template void certify(const uint32_t*, idx_t, const uint8_t*, const uint64_t*, const uint32_t*, idx_t, const SA_Interval*, unsigned long long&,
                      unsigned long long&, unsigned);

}

}
