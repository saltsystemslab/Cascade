
#ifndef PIPELINE_PROBE_HPP
#define PIPELINE_PROBE_HPP



#include "types.hpp"
#include "utility.hpp"
#include "env.hpp"

#include <cstdint>
#include <atomic>
#include <array>
#include <functional>
#include <vector>
#include <memory>
#include <thread>
#include <algorithm>


// Instrumentation scaffolding for the staging pipeline of the slab-pool between
// the partition-backend and the device.
// The pipeline is a chain of thread-pools connected by queues:
//  F --> [reader] --> C --> [stager] --> R --> [dispatcher] --> W --> [writer] --> F
//
// The instrumentation identifies per generation:
// 1.   The bottleneck pool: Per-role time is split into actual work (`io` /
//      `host` / `device`), blocked-on-input (`waitIn`), blocked-on-a-free-slab
//      (`waitF`), and unaccounted (`idle`). Each is normalized against
//      `pipeline_ns x role_count`, so the percentages are directly comparable
//      across pools of different sizes.
// 2.   The slabs' distribution: Queue depths are sampled periodically. `F`
//      sitting empty implies slab starvation; `C` backing up implies the
//      stagers not keeping pace with the readers; `R` empty implies the device
//      being starved of assemblies.
namespace gpu_saca::probe
{

// The staging pipeline's thread-pools.
enum class Role : std::size_t { reader = 0, stager, dispatcher, writer, role_c, };

constexpr std::size_t ROLE_C = static_cast<std::size_t>(Role::role_c);

inline const char* role_name(const Role r)
{
    switch(r)
    {
    case Role::reader:      return "reader";
    case Role::stager:      return "stager";
    case Role::dispatcher:  return "dispatcher";
    case Role::writer:      return "writer";
    default:                return "unbound";
    }
}


// The staging pipeline's queues.
enum class Queue : std::size_t { F = 0, C, R, W, queue_c, };

constexpr std::size_t QUEUE_C = static_cast<std::size_t>(Queue::queue_c);

// Depth reported for a queue that does not exist yet in this build.
constexpr uint64_t QUEUE_ABSENT = std::numeric_limits<uint64_t>::max();

inline const char* queue_name(const std::size_t q)
{
    constexpr const char* name[QUEUE_C] = {"F (free)", "C (to-stage)", "R (to-device)", "W (to-backend)"};
    return name[q];
}


// Per worker-thread stats accumulator.
struct alignas(2 * util::L1_CACHE_LINE_SIZE)
    Worker_Stat
{
    uint64_t io_ns = 0; // Time spent by: readers and writers; in: `pread` / `pwrite` / resident memcpy to / from the backend.
    uint64_t stage_ns = 0;  // Time spent by: stagers; in: host-side staging-compute.
    uint64_t device_ns = 0; // Time spent by: dispatchers; in: device-dispatch (H2D uploads, kernels, D2H downloads, and stream-syncs).
    uint64_t wait_in_ns = 0;    // Time spent in: blocking on the pool's input queues: `RT` / `C` / `R` / `W`.
    uint64_t wait_slab_ns = 0;  // Time spent by: readers; in: blocking on `F` for free slabs.
    uint64_t items = 0; // Units handled: slabs (for readers, writers, and stagers), and partitions (dispatchers).
    uint64_t bytes = 0; // Bytes moved: backend traffic (for readers and writers), and PCIe traffic (for dispatchers).
    uint64_t disk_bytes = 0;    // Of `bytes`, those moved to or from the disk: spilled content, the inboxes' disk-parts, and the output.

    // Time spent in actual work: in I/O, host-, or device-work.
    uint64_t busy_ns() const { return io_ns + stage_ns + device_ns; }

    // Resets the stats.
    void reset() { io_ns = stage_ns = device_ns = wait_in_ns = wait_slab_ns = items = bytes = disk_bytes = 0; }
};


// Sampled occupancy of a pipeline queue.
struct Queue_Stat
{
    uint64_t sum = 0;   // Sum of the sampled depths.
    uint64_t n = 0; // Number of samples.
    uint64_t zero_n = 0;    // Number of samples that found the queue empty.
    uint64_t min_n = std::numeric_limits<uint64_t>::max();  // Minimum sampled depth.
    uint64_t max_n = 0; // Maximum sampled depth.

    // Adds sampled depth `d` to the stats.
    void add(const uint64_t d)
    {
        sum += d, n++;
        zero_n += (d == 0);
        min_n = std::min(min_n, d), max_n = std::max(max_n, d);
    }

    // Resets the stats.
    void reset() { sum = n = zero_n = max_n = 0, min_n = std::numeric_limits<uint64_t>::max(); }

    // Returns the mean of the sampled depths.
    double mean() const { return n > 0 ? static_cast<double>(sum) / static_cast<double>(n) : 0.0; }

    // Returns the percentage of empty samples.
    double empty_pct() const { return n > 0 ? 100.0 * static_cast<double>(zero_n) / static_cast<double>(n) : 0.0; }
};


// Measurement probes for the staging pipeline.
class Pipeline_Probe
{
public:

    // Fills `d[0 .. QUEUE_C)` with the current depths of the pipeline queues,
    // in `Queue` order. A nonexistent queue reports `QUEUE_ABSENT`.
    using Depth_Fn = std::function<void(uint64_t*)>;

private:

    std::array<std::size_t, ROLE_C> role_c_{};  // Worker-count per role.
    std::array<std::size_t, ROLE_C> base_{};    // Slot-base per role into `w_`.
    std::size_t scratch_;   // Slot collecting the accumulations of threads that never bound themselves.
    std::vector<std::unique_ptr<Worker_Stat>> w_;   // Accumulator per worker-thread, plus the scratch slot at the tail.

    std::array<Queue_Stat, QUEUE_C> q_; // Sampled occupancy per pipeline queue.
    Depth_Fn depth_fn_; // Queue-depth reader provided by the pipeline.

    std::thread sampler_;   // Queue-occupancy sampler thread.
    std::atomic<bool> sampling_{false}; // Whether to sample queue depths.
    const uint64_t sample_ms_ = env::get_u64("GPU_SACA_PROBE_SAMPLE_MS").value_or(10);  // Queue-sampling period.

    std::atomic<int64_t> asm_partial_{0};   // Number of currently partial assemblies.
    std::atomic<uint64_t> asm_partial_max_{0};  // Maximum value of `asm_partial_`.
    std::atomic<int64_t> parked_slab_{0};   // Number of slabs currently parked in incomplete assemblies.
    std::atomic<uint64_t> parked_slab_max_{0};  // Maximum value of `parked_slab_`.
    std::size_t slab_c_ = 0;    // Slab-pool size.
    std::size_t admit_budget_ = 0;  // Slabs the pool reserves for assemblies.

    idx_t iter_ = 0;    // Generation index.
    idx_t ctx_ = 0; // Prefix-context size the current generation is resolving.
    idx_t active_ = 0;  // Active-set size entering the current generation.
    uint64_t pipeline_ns_ = 0;  // `drive_pipeline` wall-time.


    // Returns a static slot-ID storage for the caller thread.
    static std::size_t& slot() { thread_local std::size_t s = std::numeric_limits<std::size_t>::max(); return s; }

    // Raises `x` to `v` if `v` is larger.
    static void fetch_max(std::atomic<uint64_t>& x, const uint64_t v)
    {
        auto cur = x.load(std::memory_order_relaxed);
        while(v > cur && !x.compare_exchange_weak(cur, v, std::memory_order_relaxed));
    }

    // Prints the stats of the workers in role `r`, normalized against the
    // pool's aggregate thread-time.
    void report_role(Role r) const;

    // Prints the sampled queue-occupancies.
    void report_queues() const;

    // Prints the assembly gauge, and the slab-pool headroom it consumed.
    void report_assemblies() const;


public:

    // Initializes the accumulators for the pipeline's pool-sizes (`reader_c`,
    // `stager_c`, `dispatcher_c`, and `writer_c`), and records the slab-pool
    // layout: `slab_c` slabs, of which `admit_budget` is the budget for
    // assemblies.
    void init(std::size_t reader_c, std::size_t stager_c, std::size_t dispatcher_c, std::size_t writer_c, std::size_t slab_c, std::size_t admit_budget);

    // Binds the calling thread to the `i`'th worker-slot of role `r`.
    void bind(Role r, std::size_t i);

    // Returns the calling thread's stats accumulators.
    Worker_Stat& me() { const auto s = slot(); return *w_[s < scratch_ ? s : scratch_]; }

    // Installs `f` as the pipeline's queue-depth reader.
    void set_depth_reader(Depth_Fn f) { depth_fn_ = std::move(f); }

    // Starts sampling the queue-occupancies. Requires a depth-reader to have
    // been set.
    void start_sampler();

    // Stops sampling the queue-occupancies. Must be invoked before reporting.
    void stop_sampler();

    // Tracks an assembly whose pending-chunk counter went from `prev` to
    // `prev - 1`, out of `chunk_c` chunks in total.
    void track_assembly(idx_t prev, idx_t chunk_c);

    // Opens generation `gen` at prefix-context `ctx` over `active` suffixes.
    void begin_iteration(idx_t gen, idx_t ctx, idx_t active);

    // Adds `ns` nanoseconds to the pipeline wall-time.
    void accumulate_pipeline_ns(const uint64_t ns) { pipeline_ns_ += ns; }

    // Reports this generation's instrumented stats.
    void report_log() const;
};


// Returns the process-wide staging-pipeline probe.
inline Pipeline_Probe& P() { static Pipeline_Probe p; return p; }

}


#ifdef PROBE

#define PROBE_CONCAT_INNER(a, b) a##b
#define PROBE_CONCAT(a, b) PROBE_CONCAT_INNER(a, b)

// Executes `x` only in instrumented builds.
#define INSTRUMENT(x) do { x; } while(0)

// Scoped nanosecond-accumulator: adds the enclosing block's duration to `sink`.
#define INSTRUMENT_TIMER(sink) ::gpu_saca::timer::Scope_Timer PROBE_CONCAT(_probe_timer_, __LINE__){sink}

#else   // PROBE

#define INSTRUMENT(x) ((void)0)
#define INSTRUMENT_TIMER(sink) ((void)0)

#endif  // PROBE



#endif
