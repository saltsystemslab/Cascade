
#ifndef ISA_INSTRUMENT_HPP
#define ISA_INSTRUMENT_HPP



#include "types.hpp"
#include "utility.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <cstdio>
#include <cstring>


// ISA instrumentation scaffolding.
namespace gpu_saca::isa_instrument
{

// Per reader-thread stats accumulators.
struct alignas(2 * util::L1_CACHE_LINE_SIZE)
    Reader_Stat
{
    uint64_t read_ns = 0;   // Time spent in the backend `pread`.
    uint64_t stage_ns = 0;  // Time spent in staging slabs with host-side compute.
    uint64_t wait_ns = 0;   // Time spent in blocked-waiting for a free staging-slab.

    void reset() { read_ns = stage_ns = wait_ns = 0; }
};


// Statistics tracker for ISA-instrumentations and related operations.
class ISA_Instrumentation
{
private:

    std::vector<std::unique_ptr<Reader_Stat>> r_;    // Accumulators per reader-thread.

    idx_t iter_ = 0;    // Iteration index; 0 is initial-ranking.
    idx_t k_ = 0;   // Prefix-context size of the current iteration.
    idx_t active_ = 0;  // Active-set size entering the current iteration.

    uint64_t pipeline_ns_ = 0;  // `drive_pipeline` wall-time.
    uint64_t read_chunk_ns_ = 0;    // Time spent in `read_chunk` from backend, within `update_global_rank`. Disjoint from the reader-stats: they accumulate timings of the pool-based pipeline-backbone.
    uint64_t scatter_ns_ = 0;   // Time spent in ISA-scatters, within `update_global_rank`. Disjoint from the reader-stats: they accumulate timings of the pool-based pipeline-backbone.

    mutable uint64_t gather_total_ = 0; // Probes over the whole run.
    mutable uint64_t oob_total_ = 0;    // Out-of-bound probes over the whole run.


public:

    // Initializes the accumulators for `reader_c`-many reader-threads.
    void init(const std::size_t reader_c)
    {
        r_.clear();
        for(std::size_t i = 0; i < reader_c; ++i)
            r_.emplace_back(std::make_unique<Reader_Stat>());
    }

    // Returns a static slot-ID storage for the caller thread.
    static std::size_t& slot() { thread_local std::size_t s; return s; }

    // Binds the calling thread to accumulator slot `i`.
    void bind(const std::size_t i) { slot() = i; }

    // Returns the calling thread's read-stats accumulators.
    Reader_Stat& me() { return *r_[slot()]; }

    // Opens iteration `iter` at prefix-context `k` over `active` suffixes.
    void begin_iteration(const idx_t iter, const idx_t k, const idx_t active)
    {
        iter_ = iter, k_ = k, active_ = active;
        pipeline_ns_ = read_chunk_ns_ = scatter_ns_ = 0;
        for(auto& r : r_)
            r->reset();
    }

    // Adds `ns` nanoseconds to pipeline-time.
    void accumulate_pipeline_ns(const uint64_t ns) { pipeline_ns_ += ns; }

    // Adds `ns` nanoseconds to read-chunk-time.
    void accumulate_read_chunk_ns(const uint64_t ns) { read_chunk_ns_ += ns; }

    // Adds `ns` nanoseconds to scatter-time.
    void accumulate_scatter_ns(const uint64_t ns) { scatter_ns_ += ns; }

    // Reports the instrumented stats, and folds this iteration's probe
    // counters into the run-level totals.
    void report_log() const
    {
        uint64_t read_ns = 0, gather_ns = 0, wait_ns = 0;
        uint64_t gathers = 0, oob = 0;
        for(const auto& r : r_)
            read_ns += r->read_ns, gather_ns += r->stage_ns, wait_ns += r->wait_ns;

        std::cerr << "\n[ISA iter " << iter_ << ", context "<< k_ << ", active "<< active_ << "]\n";
        std::fprintf(stderr, "  drive_pipeline wall     %10.3f s\n", pipeline_ns_ / 1e9);

        const auto w_c = r_.size(); // Number of workers performing backend-read and the associated ISA-updates in the pipeline.
        const double denom = double(pipeline_ns_) * w_c;    // Normalization factor for each reader's timings.
        const auto percentage = [denom](const uint64_t ns) { return denom > 0.0 ? 100.0 * ns / denom : 0.0; };
        std::fprintf(stderr, "  readers (%zu, cpu-sum)  read %8.3f s (%5.1f%%) | gather %8.3f s (%5.1f%%) | wait %8.3f s (%5.1f%%) | other %5.1f%%\n",
                        r_.size(),
                        read_ns / 1e9,   percentage(read_ns),
                        gather_ns / 1e9, percentage(gather_ns),
                        wait_ns / 1e9,   percentage(wait_ns),
                        100.0 - percentage(read_ns) - percentage(gather_ns) - percentage(wait_ns));

        // std::fprintf(stderr, "  update_global_rank      read %8.3f s | scatter %8.3f s | total %8.3f s\n",
        //                 read_chunk_ns_ / 1e9, scatter_ns_ / 1e9, (read_chunk_ns_ + scatter_ns_) / 1e9);

        if(active_ > 0)
            std::fprintf(stderr, "  per-op                  gather %6.2f ns | scatter %6.2f ns\n",
                            double(gather_ns) / double(active_), double(scatter_ns_) / double(active_));

        gather_total_ += gathers, oob_total_ += oob;

        std::fflush(stderr);
    }
};


// Returns the process-wide ISA-instrumentation tracker.
inline ISA_Instrumentation& ISA_Instrument() { static ISA_Instrumentation s; return s; }

}


#ifdef ISA_INSTRUMENT

#define ISA_CONCAT_INNER(a, b) a##b
#define ISA_CONCAT(a, b) ISA_CONCAT_INNER(a, b)

// Executes `x` only in instrumented builds.
#define INSTRUMENT(x) do { x; } while(0)

// Scoped nanosecond-accumulator: adds the enclosing block's duration to `sink`.
#define INSTRUMENT_TIMER(sink) ::gpu_saca::timer::Scope_Timer ISA_CONCAT(_isa_instrument_timer_, __LINE__){sink}

#else   // ISA_INSTRUMENT

#define INSTRUMENT(x) ((void)0)
#define INSTRUMENT_TIMER(sink) ((void)0)

#endif  // ISA_INSTRUMENT



#endif
