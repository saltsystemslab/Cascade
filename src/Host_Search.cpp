
#include "Host_Search.hpp"
#include "CPU_Reference.hpp"
#include "sa40.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
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
// the lanes `lane`, taking the block `next_block` whenever the current one has
// no boundary left to start, until none is left: a lane starts on the next
// boundary as soon as it is done with one, across blocks. With the positions
// sorted, `dup[g]` marks a position `g` whose query repeats the one before it
// in its block, and which then takes that one's interval once the thread's
// searches are done.
template <typename T_sym_>
void search_blocks(const Batch<T_sym_>& x, const idx_t q, const idx_t block, std::atomic<idx_t>& next_block, std::vector<Lane<T_sym_>>& lane,
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
                const idx_t k = (left ? next_block.fetch_add(1, std::memory_order_relaxed) : q);
                if(k * block >= q)
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
    std::atomic<idx_t> next_block(0);

    // Every team of the host's search is of the same threads, those without a block idling: libgomp ends the threads that a smaller
    // team leaves out, and a larger team then starts them anew.
    #pragma omp parallel num_threads(static_cast<int>(t))
    {
        std::vector<Lane<T_sym_>> lane(std::max(1u, lanes));
        search_blocks(x, q, block, next_block, lane, dup.data());
    }
}


template <typename T_sym_>
void certify(const T_sym_* const T, const idx_t n, const uint8_t* const SA, const uint64_t* const off, const T_sym_* const sym, const idx_t q,
             const SA_Interval* const I, unsigned long long& fail_c, unsigned long long& first, const unsigned threads)
{
    unsigned long long c = 0, f = ULLONG_MAX;

    #pragma omp parallel for num_threads(static_cast<int>(thread_count(threads))) schedule(static) reduction(+ : c) reduction(min : f)
    for(idx_t i = 0; i < q; ++i)
        if(!ref::certify(T, n, SA, sym + off[i], off[i + 1] - off[i], I[i]))
        {
            ++c;
            f = std::min<unsigned long long>(f, i);
        }

    fail_c += c;
    first = std::min(first, f);
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

    #pragma omp parallel for num_threads(static_cast<int>(thread_count(threads))) schedule(static)  // All the threads; see `find`.
    for(std::size_t p = 0; p < pieces; ++p)
    {
        const std::size_t b = p * COPY_PIECE;
        std::memcpy(static_cast<uint8_t*>(dst) + b, static_cast<const uint8_t*>(src) + b, std::min(COPY_PIECE, bytes - b));
    }
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
