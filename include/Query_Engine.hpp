
#ifndef QUERY_ENGINE_HPP
#define QUERY_ENGINE_HPP



#include "types.hpp"
#include "query-file.hpp"
#include "Suffix_Index.hpp"
#include "Sample_Table.hpp"
#include "Host_Search.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>


// Batched exact-match queries over a query index: `find` returns the SA-interval
// of every pattern of a batch. The two-phase search ranks each pattern's packed
// prefix among the keys of a sample table, which brackets the interval's
// boundaries within a stride of ranks, and then binary-searches the suffix
// array within the brackets. Without a table, the direct mode binary-searches
// the whole suffix array: the table-free baseline, correct for any placement.
// With the suffix array in pinned host memory, the search gathers the
// brackets' entries into device memory first, in whole words: a transfer per
// query, rather than one per probe. A host-side batch streams through the
// device in chunks, over several streams: the transfers of some chunks overlap
// the searches of others. With both arrays in host memory and either pageable,
// the host searches: the device ranks a chunk's patterns in the table, and the
// host's cores search the brackets over the arrays, while the device ranks the
// next chunks. A mapped suffix array is read from its drives on demand: the
// host asks for the pages of a chunk's brackets ahead, while it searches the
// chunk before, so that their reads overlap it and one another. The planner
// picks the tiers and the table that a load leaves automatic, from the index's
// sizes and the memory budgets: the first of the plans A to D that fits.

namespace gpu_saca
{

// Where the two-phase search reads the suffix-array entries of its brackets.
enum class Gather
{
    automatic,  // As `always` if the suffix array is in pinned host memory and the device searches, else as `never`.
    never,  // From the suffix array, a probe at a time.
    always, // From copies in device memory, gathered per batch.
};


// The memory plans of a query index, in the order the planner tries them.
enum class Plan
{
    A,  // Both arrays in device memory; the device searches.
    B,  // The text in device memory, and the suffix array in pinned host memory; the device searches, its brackets gathered.
    C,  // Both arrays in pageable host memory; the host searches.
    D,  // The text in pageable host memory, and the suffix array mapped from its files; the host searches.
    custom, // Other tiers, named.
};


// Returns the name of the plan `p`.
const char* plan_name(Plan p);


// The `Query_Options::stride` and `key_words` that leave them to the planner.
constexpr idx_t AUTO_STRIDE = ~idx_t(0);
constexpr uint32_t AUTO_KEY_WORDS = ~uint32_t(0);


// Placement, load-time checks, search mode, and chunking of an engine.
struct Query_Options
{
    Residency text_res = Residency::device; // Tier of the text; `automatic` lets the planner pick it.
    Residency sa_res = Residency::device;   // Tier of the suffix array; `automatic` likewise.
    idx_t spot = 100000;    // Random adjacent pairs of the suffix array spot-checked for order at load; `0` skips.
    idx_t stride = 32;  // Sample-table stride; `0` selects the direct mode, and `AUTO_STRIDE` lets the planner pick it.
    uint32_t key_words = 0; // 64-bit words per table key: 1, 2, or 4; `0` picks 2, or 4 for 4-byte symbols; `AUTO_KEY_WORDS`, the planner.
    bool sort = false;  // Whether to sort every batch by its packed prefixes before searching; gathering, identical queries then gather once.
    Gather gather = Gather::automatic;  // Where the two-phase search reads its brackets' suffix-array entries.
    idx_t gather_queries = idx_t(1) << 18;  // Queries whose brackets are gathered at a time, bounding the copies' device memory.
    idx_t chunk_queries = 0;    // Queries per chunk of a host-side batch, at most; `0` picks 2^18, or 2^24 if gathering.
    idx_t chunk_symbols = 0;    // Symbols per chunk, at most; `0` picks 2^24, or 2^28 if gathering. A longer query is a chunk of its own.
    idx_t min_chunk_queries = idx_t(1) << 16;   // Least queries of a chunk's share of a host-side batch, which splits into two chunks per stream unless gathering.
    idx_t min_chunk_symbols = idx_t(1) << 22;   // Least symbols of a chunk's share.
    unsigned host_threads = 0;  // Threads of the host's search, if it searches; `0` takes OpenMP's default, every core.
    unsigned host_lanes = 32;   // Boundary searches each such thread interleaves.
    bool host_prefetch = true;  // Whether the host asks the kernel ahead for the pages of a mapped suffix array that a chunk's brackets probe first.
    idx_t device_budget = 0;    // Bytes of device memory the planner places in; `0` takes the device's free memory at load.
    idx_t host_budget = 0;  // Bytes of host memory it places in; `0` takes the host's available memory at load.
    bool phase_times = false;   // Whether `find` times its phases, as `Query_Engine::phases` reports them.
};


// The placement of an index that an engine loads: its plan, its tiers, and its
// sample table, as named in the options or as the planner picks them, and the
// memory they take, as planned. The planner picks the first plan of A to D
// whose tiers agree with those named, and that fits both budgets, less their
// margins of a sixteenth, and at least 1 GiB; with an automatic table, of
// stride 32, doubled until it fits the device memory that is left, up to 4096,
// or while gathered brackets fit their windows, and of 2-word keys (4 for
// 4-byte symbols) if the device searches, else 4-word.
struct Placement
{
    Plan plan = Plan::custom;   // The plan of the tiers.
    Residency text_res = Residency::device; // The tier of the text.
    Residency sa_res = Residency::device;   // Of the suffix array.
    idx_t stride = 0;   // The sample table's stride at load; `0` in the direct mode.
    uint32_t key_words = 0; // Its keys' words; `0` in the direct mode.
    bool auto_text_res = false; // Whether the planner picked the text's tier,
    bool auto_sa_res = false;   // the suffix array's,
    bool auto_stride = false;   // the table's stride,
    bool auto_key_words = false;    // and its keys' words.
    bool host_search = false;   // Whether the host searches.
    bool gather = false;    // Whether the device gathers its brackets' suffix-array entries.
    idx_t chunk_queries = 0;    // The most queries of a chunk of a host-side batch, in effect.
    idx_t chunk_symbols = 0;    // The most symbols.
    bool fits = false;  // Whether the placement fits the budgets; always if the planner picked any tier.
    idx_t device_budget = 0;    // Bytes of device memory planned in: the device's free memory at load, unless set; `0` if unknown.
    idx_t host_budget = 0;  // Of host memory: its available memory at load, unless set; `0` if unknown.
    idx_t device_margin = 0;    // Bytes of the device budget left free: for the callers' batches, and the scratch of `find_async`.
    idx_t host_margin = 0;  // Of the host budget.
    idx_t device_bytes = 0; // Bytes the placement takes in device memory, at most: its arrays, its table, the lanes, and the gathers.
    idx_t host_bytes = 0;   // In host memory, at most: its arrays, the lanes' pinned buffers, and the scratch of loading them.
    idx_t table_bytes = 0;  // Bytes of the table's keys.
    idx_t table_room = 0;   // Bytes of device memory left to a table and its gathers: what an automatic table's stride fits.
    std::string sa_fs;  // The file systems of the suffix array's files.
    bool sa_local = true;   // Whether they are all local, rather than over a network; mapped, the others read slowly.
};


// Phases of a `find` call, as `Query_Engine::phases` times them.
enum class Phase
{
    stage,  // Host: copying the batch's pageable arrays into the lanes' pinned staging, and the intervals out of it.
    h2d,    // Device: copying the chunks' queries to the device.
    rank,   // Device: packing the chunks' queries, sorting them if sorting, and ranking them in the sample table.
    search, // Device: the exact search of the chunks' queries, with their brackets' gathers and their runs' heads if any.
    d2h,    // Device: copying the chunks' intervals to the host, or their brackets if the host searches.
    prefetch,   // Host: asking the kernel ahead for the pages of a mapped suffix array that the chunks' brackets probe.
    host_search,    // Host: the exact search of the chunks' queries, if the host searches.
    wait,   // Host: waiting on the device.
};

constexpr std::size_t PHASE_C = 8;  // Count of `Phase`s.

// Returns the name of the phase `p`.
const char* phase_name(Phase p);


// Times of the phases of a `find` call, in seconds: each phase's busy time,
// summed over the call's chunks, and over the threads that run them. The
// phases of chunks on different streams overlap, so the sum of the phases may
// exceed the call's wall time. A device phase runs from the stream's reaching
// its first operation to the end of its last, waiting on other streams' work
// meanwhile, if any.
struct Phase_Times
{
    std::array<double, PHASE_C> t{};    // Per `Phase`.
    double wall = 0;    // The call's wall time.
    idx_t chunks = 0;   // The chunks the call took, `0` in the direct mode on the host.

    double operator[](const Phase p) const { return t[static_cast<std::size_t>(p)]; }
};


// The timing of the phases of the chunks that a stream of `find` takes in a
// call: events recorded on the stream, between which the device's phases ran,
// read once the call's work is done; and the host's seconds in the chunks'
// phases. The events are kept across calls.
class Phase_Clock
{
private:

    static constexpr int8_t NO_PHASE = -1;

    std::vector<cudaEvent_t> ev_;   // The events; the first `ev_c_` recorded in the call, on the stream.
    std::vector<int8_t> end_;   // Per event, the phase that ran up to it from the event before; `NO_PHASE` if none.
    std::size_t ev_c_ = 0;
    std::array<double, PHASE_C> host_{};    // The host's seconds per phase.

    // Records the next event on the stream `s`, as the end of the phase `p`.
    void record(int8_t p, cudaStream_t s);


public:

    // Records on the stream `s` the start of a device phase.
    void start(const cudaStream_t s) { record(NO_PHASE, s); }

    // Records on the stream `s` the end of the device phase `p`, which ran
    // from the event recorded last; and so the start of the next.
    void lap(const Phase p, const cudaStream_t s) { record(static_cast<int8_t>(p), s); }

    // Adds `sec` seconds of the host to the phase `p`.
    void add(const Phase p, const double sec) { host_[static_cast<std::size_t>(p)] += sec; }

    // Adds the seconds of the call's phases to `t`; the work of the call's
    // events must be done.
    void read(std::array<double, PHASE_C>& t) const;

    // Clears the call's events and seconds, for a new call.
    void clear();

    // Destroys the events.
    void release();
};


// Memory of a loaded engine, in bytes.
struct Engine_Memory
{
    std::size_t text_device = 0;    // Of the text, in device memory.
    std::size_t sa_device = 0;  // Of the suffix array, in device memory.
    std::size_t table_device = 0;   // Of the sample table.
    std::size_t lanes_device = 0;   // Of the lanes' buffers.
    std::size_t pool_used_high = 0; // The highest of the memory pool's scratch in use at once, since the load.
    std::size_t pool_reserved_high = 0; // The most device memory that the pool held at once, since the load.
    std::size_t text_pinned = 0;    // Of the text, in pinned host memory.
    std::size_t sa_pinned = 0;  // Of the suffix array, in pinned host memory.
    std::size_t lanes_pinned = 0;   // Of the lanes' pinned staging.

    // Returns the engine's device memory, its pool's at the high-water mark of its use.
    std::size_t device() const { return text_device + sa_device + table_device + lanes_device + pool_used_high; }

    // Returns the engine's pinned host memory.
    std::size_t pinned() const { return text_pinned + sa_pinned + lanes_pinned; }
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
    uint32_t* heads = nullptr;  // Per position of the sorted queries, the first of its run of identical queries.
    void* scan_space = nullptr; // Scratch of finding the runs' first positions.
    std::size_t scan_bytes = 0;
    std::size_t bytes = 0;  // Bytes of the buffers.
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
        Rank_Range* br_h = nullptr; // Pinned: the chunk's brackets, if the host searches.
        uint32_t* ids_h = nullptr;  // Pinned: the batch position of each of its sorted places.
        Search_Buffers space;   // Of the chunk's two-phase search, in any mode.
        mutable Phase_Clock clock;  // Of its chunks' phases in a `find` call, if the options time them.
    };

    // Times the phases of a `find` call, if the engine's options say so, over
    // its life: clears the lanes' clocks when made, and sets the call's wall
    // time when destroyed.
    class Call_Timer
    {
    private:

        const Query_Engine& eng_;
        const std::chrono::steady_clock::time_point t0_;


    public:

        explicit Call_Timer(const Query_Engine& eng);
        ~Call_Timer();
        Call_Timer(const Call_Timer&) = delete;
        Call_Timer& operator=(const Call_Timer&) = delete;
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
    Query_Options opt_; // The options of the load, with its tiers.
    Placement placement_;   // The placement of the load.
    idx_t window_bytes_ = 0;    // Bytes of a bracket's slot of gathered suffix-array entries; `0` unless gathering.
    int device_ = 0;    // The device of the load.
    std::array<Lane, STREAM_C> lane_;   // The streams of `find`.
    mutable std::mutex find_lock_;  // Held by `find`, which owns the lanes.
    cudaMemPool_t pool_ = nullptr;  // Scratch memory of `find_async`, kept across batches.
    host_search::SA_Files sa_files_;    // A mapped suffix array's files, for the host to ask for its pages ahead, if it does.
    std::unique_ptr<std::atomic<uint8_t>[]> asked_; // Per page of the files, whether the host asked for it since the marks were last cleared.
    idx_t asked_pages_ = 0; // Pages of the marks.
    idx_t asked_cap_ = 0;   // Most pages marked before the marks are cleared: about the room of the page cache, which may evict them.
    mutable idx_t asked_c_ = 0; // Pages marked since the marks were last cleared.
    std::size_t lanes_device_bytes_ = 0;    // Device memory of the lanes' buffers.
    std::size_t lanes_pinned_bytes_ = 0;    // Pinned host memory of the lanes' staging.
    mutable Phase_Times timed_; // The wall time and the chunks of the last `find` call, if timing its phases; its phases are on the lanes' clocks.


    // Returns the clock of the lane `l`'s phases if the options time them; else
    // null.
    Phase_Clock* clock_of(const Lane& l) const { return opt_.phase_times ? &l.clock : nullptr; }

    // Allocates the lanes for chunks of the options' sizes, and the pool.
    // Returns `false` with a reason in `err` on failure.
    bool allocate_lanes(std::string& err);

    // Returns the bytes of device memory of the lanes for chunks of `cq` queries
    // and `cs` symbols, as `allocate_lanes` allocates them, at most.
    static idx_t lane_device_bytes(idx_t cq, idx_t cs);

    // Returns the bytes of their pinned host memory, with the buffers of the
    // host's search if `host`.
    static idx_t lane_host_bytes(idx_t cq, idx_t cs, bool host);

    // Returns the bytes of device memory of the gathered brackets of the searches
    // that run at once, at most, over a table of stride `stride` and keys of `K`
    // symbols, gathering for `gather_queries` queries at a time.
    static idx_t gather_bytes(idx_t stride, idx_t K, idx_t gather_queries);

    // Returns the least of the strides 32, 64, ... whose table, with keys of `w`
    // words over a text of `n` symbols, and its gathered brackets if `gather`,
    // fits `room` bytes of device memory; `0` if none does. Gathered, a stride's
    // brackets must fit their windows.
    static idx_t fit_stride(idx_t n, uint32_t w, bool gather, idx_t gather_queries, idx_t room);

    // Completes the placement `p` of the index of metadata `m` with the options
    // `opt`, its tiers and its budgets set: the searcher, the chunks, the table,
    // and the memory they take. Returns `true` iff it fits the budgets; else
    // gives the reason in `why`.
    static bool fit(const Index_Meta& m, const Query_Options& opt, Placement& p, std::string& why);

    // Frees the lanes and the pool.
    void release_lanes();

    // Opens a mapped suffix array's files for each of the host's threads, and
    // the marks of the pages asked for, if the host asks for pages ahead; else
    // closes them. Returns `false` with a reason in `err` on failure.
    bool open_sa_files(std::string& err);

    // Clears the marks of the pages asked for.
    void forget_asked() const;

    // Searches the device-side queries `sym[off[i] - off0, off[i + 1] - off0)`,
    // `i < q`, into `out[0, q)`, with the two-phase search's buffers `space` of
    // room for `q` queries, and the gathered brackets in scratch from the pool;
    // asynchronously on the stream `s`. Times the ranking and the exact search
    // on the clock `clock` of the stream, unless null.
    void search(const uint64_t* off, const T_sym_* sym, uint64_t off0, idx_t q, SA_Interval* out, const Search_Buffers& space,
                cudaStream_t s, Phase_Clock* clock = nullptr) const;

    // Returns the chunks `[a, b)` that the host-side batch of the `q` queries of
    // the offsets `off` streams through the device in, as `find` splits it.
    std::vector<std::pair<idx_t, idx_t>> split(const uint64_t* off, idx_t q) const;

    // Copies the queries `[a, b)` of the host-side batch `off`, `sym` into the
    // device buffers of the lane `l`, asynchronously on its stream; the arrays
    // not in pinned memory, per `pinned`, through the lane's staging, copied
    // into it with the host search's threads if `parallel`, and else with the
    // calling thread. Returns the device address of the chunk's symbols: the
    // lane's, or pool scratch for a query longer than a chunk, which the caller
    // frees on the stream.
    T_sym_* upload(const Lane& l, const uint64_t* off, const T_sym_* sym, idx_t a, idx_t b, const Pinned_Arrays& pinned, bool parallel) const;

    // Finds the SA-intervals of the queries `[a, b)` of the host-side batch
    // `off`, `sym` into `out[a, b)` through the lane `l`. The arrays not in
    // pinned memory, per `pinned`, go through the lane's staging, and then the
    // chunk is found synchronously; else it is only enqueued on the lane's
    // stream.
    void find_chunk(const Lane& l, const uint64_t* off, const T_sym_* sym, idx_t a, idx_t b, SA_Interval* out,
                    const Pinned_Arrays& pinned) const;

    // Enqueues on the lane `l` the device's part of the host's search of the
    // queries `[a, b)` of the host-side batch `off`, `sym`: ranks them in the
    // sample table, and copies their brackets, and their sorted order if
    // sorting, into the lane's pinned buffers. The arrays not in pinned memory,
    // per `pinned`, go through the lane's staging.
    void rank_chunk(const Lane& l, const uint64_t* off, const T_sym_* sym, idx_t a, idx_t b, const Pinned_Arrays& pinned) const;

    // Finds the SA-intervals of the host-side batch `off`, `sym` of `q` queries
    // into `out[0, q)`, with the host searching: in chunks, each ranked on the
    // device through a lane while the host searches an earlier one's brackets;
    // with a mapped suffix array, the pages of each chunk's brackets asked for
    // while the host searches the chunk before, if prefetching. The caller holds
    // `find_lock_`.
    void find_host(const uint64_t* off, const T_sym_* sym, idx_t q, SA_Interval* out) const;


public:

    Query_Engine() = default;
    ~Query_Engine() { release_lanes(); }
    Query_Engine(const Query_Engine&) = delete;
    Query_Engine& operator=(const Query_Engine&) = delete;

    // Loads the index-directory `dir` as `opt` places it, as `plan` completes
    // the placement. The index must be of symbols of `T_sym_`, with both arrays
    // accessible from the device, or both from the host. Returns `false` with a
    // reason in `err` on failure.
    bool load(const std::string& dir, const Query_Options& opt, std::string& err);

    // Plans the placement `p` of the index of metadata `m` with the options
    // `opt`, as `load` does, without loading it: the tiers and the table that
    // `opt` names, and the planner's picks of those it leaves automatic, within
    // the budgets of `opt`, or the device's free memory and the host's available
    // memory now; named tiers are not held to them. Returns `false` with a
    // reason in `err` if no plan fits, or on failure.
    static bool plan(const Index_Meta& m, const Query_Options& opt, Placement& p, std::string& err);

    // Switches the loaded engine's search mode to the `stride`, `key_words`,
    // `sort`, and `gather` settings of `Query_Options`, rebuilding the sample
    // table; the index stays loaded. An automatic stride is the least that fits
    // the load's room for a table, and automatic key words as the planner picks
    // them. Returns `false` with a reason in `err` on failure, leaving the engine
    // in the direct mode.
    bool configure(idx_t stride, uint32_t key_words, bool sort, Gather gather, std::string& err);

    // Finds the SA-intervals of the queries `sym[off[i], off[i + 1])`, `i < q`,
    // into `out[0, q)`; all device-side arrays, asynchronously on the stream `s`.
    // If the host searches, the batch goes there and back, synchronously.
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
    // from a thread per stream. If the host searches, it searches every chunk's
    // brackets from the calling thread, with its own threads; the direct mode
    // takes no chunks. Concurrent calls run one after another. If the options
    // time phases, the call times its own, for `phases`.
    void find(const uint64_t* off, const T_sym_* sym, idx_t q, SA_Interval* out) const;

    // Returns the times of the phases of the last `find` call with queries, or
    // of `find_async` if the host searches, if the options time phases; zeros
    // if not. Reads the device's from the events of the call, done by then.
    // Waits for a running `find`.
    Phase_Times phases() const;

    // Returns the memory of the loaded engine: its arrays', its table's, its
    // lanes', and its memory pool's, since the load. Not concurrently with
    // `load` or `configure`.
    Engine_Memory memory() const;

    // Checks the intervals `I[0, q)` of the queries `sym[off[i], off[i + 1])`
    // with the certificate of `ref::certify`: adds the count of failing queries
    // to `*fail_c` and lowers `*first` to the first failing one. All device-side
    // arrays, asynchronously on the stream `s`; if the host searches, it checks,
    // with the arrays copied to it synchronously.
    void certify_async(const uint64_t* off, const T_sym_* sym, idx_t q, const SA_Interval* I, unsigned long long* fail_c,
                       unsigned long long* first, cudaStream_t s) const;

    // Returns `true` iff the host searches the loaded index: its arrays are in
    // host memory, and the device cannot access either.
    bool host_searches() const { return idx_.loaded() && !idx_.device_accessible(); }

    // Drops the pages of a mapped suffix array from the page cache, as
    // `Suffix_Index::drop_cached_sa` does, so that they are asked for anew.
    // Waits for a running `find`. Returns `false` with a reason in `err` on
    // failure, or if the suffix array is not mapped.
    bool drop_cached_sa(std::string& err) const;

    // Returns the loaded index.
    const Suffix_Index& index() const { return idx_; }

    // Returns the sample table; empty in the direct mode.
    const Sample_Table& table() const { return tab_; }

    // Returns the options the engine was loaded with, with the tiers of its load.
    const Query_Options& options() const { return opt_; }

    // Returns the placement of the load.
    const Placement& placement() const { return placement_; }

    // Returns the count of streams of `find`.
    static constexpr std::size_t streams() { return STREAM_C; }
};

}



#endif
