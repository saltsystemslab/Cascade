
#ifndef QUERY_ENGINE_HPP
#define QUERY_ENGINE_HPP



#include "types.hpp"
#include "query-file.hpp"
#include "Suffix_Index.hpp"
#include "Sample_Table.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <mutex>
#include <string>


// Batched exact-match queries over a query index: `find` returns the SA-interval
// of every pattern of a batch. The two-phase search ranks each pattern's packed
// prefix among the keys of a sample table, which brackets the interval's
// boundaries within a stride of ranks, and then binary-searches the suffix
// array within the brackets. Without a table, the direct mode binary-searches
// the whole suffix array: the table-free baseline, correct for any placement
// the device can access. With the suffix array in pinned host memory, the
// search gathers the brackets' entries into device memory first, in whole
// words: a transfer per query, rather than one per probe. A host-side batch
// streams through the device in chunks, over several streams: the transfers of
// some chunks overlap the searches of others.

namespace gpu_saca
{

// Where the two-phase search reads the suffix-array entries of its brackets.
enum class Gather
{
    automatic,  // As `always` if the suffix array is in pinned host memory, else as `never`.
    never,  // From the suffix array, a probe at a time.
    always, // From copies in device memory, gathered per batch.
};


// Placement, load-time checks, search mode, and chunking of an engine.
struct Query_Options
{
    Residency text_res = Residency::device; // Tier of the text.
    Residency sa_res = Residency::device;   // Tier of the suffix array.
    idx_t spot = 100000;    // Random adjacent pairs of the suffix array spot-checked for order at load; `0` skips.
    idx_t stride = 32;  // Sample-table stride; `0` selects the direct mode.
    uint32_t key_words = 0; // 64-bit words per table key: 1, 2, or 4; `0` picks 2, or 4 for 4-byte symbols.
    bool sort = false;  // Whether to sort every batch by its packed prefixes before searching.
    Gather gather = Gather::automatic;  // Where the two-phase search reads its brackets' suffix-array entries.
    idx_t gather_queries = idx_t(1) << 18;  // Queries whose brackets are gathered at a time, bounding the copies' device memory.
    idx_t chunk_queries = 0;    // Queries per chunk of a host-side batch, at most; `0` picks 2^18, or 2^24 if gathering.
    idx_t chunk_symbols = 0;    // Symbols per chunk, at most; `0` picks 2^24, or 2^28 if gathering. A longer query is a chunk of its own.
    idx_t min_chunk_queries = idx_t(1) << 16;   // Least queries of a chunk's share of a host-side batch, which splits into two chunks per stream unless gathering.
    idx_t min_chunk_symbols = idx_t(1) << 22;   // Least symbols of a chunk's share.
};


// Device buffers of a two-phase search of up to `cap` queries.
struct Search_Buffers
{
    idx_t cap = 0;
    void* keys[2] = {nullptr, nullptr}; // The queries' packed prefixes; the second buffer for sorting them.
    uint32_t* ids[2] = {nullptr, nullptr};  // The queries' batch positions, in their keys' order when sorted.
    void* brackets = nullptr;   // The ranks that contain the queries' interval boundaries, two per query.
    void* sort_space = nullptr; // Scratch of the sort.
    std::size_t sort_bytes = 0;
};


// Query engine over indexes of symbols of type `T_sym_`.
template <typename T_sym_>
class Query_Engine
{
private:

    // A stream of the host-side pipeline, with the buffers of a chunk.
    struct Lane
    {
        cudaStream_t stream = nullptr;
        uint64_t* off_d = nullptr;  // The chunk's query offsets, on the device.
        T_sym_* sym_d = nullptr;    // Its symbols.
        SA_Interval* out_d = nullptr;   // Its intervals.
        uint64_t* off_h = nullptr;  // Pinned staging of the offsets, for a batch in pageable memory.
        T_sym_* sym_h = nullptr;    // Of the symbols.
        SA_Interval* out_h = nullptr;   // Of the intervals.
        Search_Buffers space;   // Of the chunk's two-phase search, in any mode.
    };

    // Which arrays of a host-side batch are in pinned memory, and so need no staging.
    struct Pinned_Arrays
    {
        bool off;
        bool sym;
        bool out;
    };

    static constexpr std::size_t STREAM_C = 3;  // Streams of the host-side pipeline.
    static constexpr std::size_t CHUNKS_PER_STREAM = 2; // Chunks per stream a batch splits into, when large enough.

    Suffix_Index idx_;  // The loaded index.
    Sample_Table tab_;  // Its sample table.
    Query_Options opt_; // The options of the load.
    idx_t window_bytes_ = 0;    // Bytes of a bracket's slot of gathered suffix-array entries; `0` unless gathering.
    int device_ = 0;    // The device of the load.
    std::array<Lane, STREAM_C> lane_;   // The streams of `find`.
    mutable std::mutex find_lock_;  // Held by `find`, which owns the lanes.
    cudaMemPool_t pool_ = nullptr;  // Scratch memory of `find_async`, kept across batches.


    // Allocates the lanes for chunks of the options' sizes, and the pool.
    // Returns `false` with a reason in `err` on failure.
    bool allocate_lanes(std::string& err);

    // Frees the lanes and the pool.
    void release_lanes();

    // Searches the device-side queries `sym[off[i] - off0, off[i + 1] - off0)`,
    // `i < q`, into `out[0, q)`, with the two-phase search's buffers `space` of
    // room for `q` queries, and the gathered brackets in scratch from the pool;
    // asynchronously on the stream `s`.
    void search(const uint64_t* off, const T_sym_* sym, uint64_t off0, idx_t q, SA_Interval* out, const Search_Buffers& space,
                cudaStream_t s) const;

    // Finds the SA-intervals of the queries `[a, b)` of the host-side batch
    // `off`, `sym` into `out[a, b)` through the lane `l`. The arrays not in
    // pinned memory, per `pinned`, go through the lane's staging, and then the
    // chunk is found synchronously; else it is only enqueued on the lane's
    // stream.
    void find_chunk(const Lane& l, const uint64_t* off, const T_sym_* sym, idx_t a, idx_t b, SA_Interval* out,
                    const Pinned_Arrays& pinned) const;


public:

    Query_Engine() = default;
    ~Query_Engine() { release_lanes(); }
    Query_Engine(const Query_Engine&) = delete;
    Query_Engine& operator=(const Query_Engine&) = delete;

    // Loads the index-directory `dir` as `opt` places it. The index must be of
    // symbols of `T_sym_`, placed device-accessibly. Returns `false` with a
    // reason in `err` on failure.
    bool load(const std::string& dir, const Query_Options& opt, std::string& err);

    // Switches the loaded engine's search mode to the `stride`, `key_words`,
    // `sort`, and `gather` settings of `Query_Options`, rebuilding the sample
    // table; the index stays loaded. Returns `false` with a reason in `err` on
    // failure, leaving the engine in the direct mode.
    bool configure(idx_t stride, uint32_t key_words, bool sort, Gather gather, std::string& err);

    // Finds the SA-intervals of the queries `sym[off[i], off[i + 1])`, `i < q`,
    // into `out[0, q)`; all device-side arrays, asynchronously on the stream `s`.
    void find_async(const uint64_t* off, const T_sym_* sym, idx_t q, SA_Interval* out, cudaStream_t s) const;

    // Finds the SA-intervals of the queries `sym[off[i], off[i + 1])`, `i < q`,
    // into `out[0, q)`; all host-side arrays, synchronously. The batch streams
    // through the device in chunks, `CHUNKS_PER_STREAM` per stream: a chunk
    // ends at its share of the batch's queries or of its symbols, whichever
    // comes first, a share being no less than the options' least for a chunk
    // and no more than their most. A batch whose brackets are gathered takes
    // chunks of the most instead: its gathers gain more from sorting more
    // queries together than from chunks overlapping. A batch in pinned memory
    // is transferred directly, its chunks enqueued ahead from the calling
    // thread; one with an array in pageable memory goes through pinned staging,
    // from a thread per stream. Concurrent calls run one after another.
    void find(const uint64_t* off, const T_sym_* sym, idx_t q, SA_Interval* out) const;

    // Checks the intervals `I[0, q)` of the queries `sym[off[i], off[i + 1])`
    // with the certificate of `ref::certify`: adds the count of failing queries
    // to `*fail_c` and lowers `*first` to the first failing one. All device-side
    // arrays, asynchronously on the stream `s`.
    void certify_async(const uint64_t* off, const T_sym_* sym, idx_t q, const SA_Interval* I, unsigned long long* fail_c,
                       unsigned long long* first, cudaStream_t s) const;

    // Returns the loaded index.
    const Suffix_Index& index() const { return idx_; }

    // Returns the sample table; empty in the direct mode.
    const Sample_Table& table() const { return tab_; }

    // Returns the options the engine was loaded with.
    const Query_Options& options() const { return opt_; }

    // Returns the count of streams of `find`.
    static constexpr std::size_t streams() { return STREAM_C; }
};

}



#endif
