
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

// Bit-layout of a `rho` entry while instrumented:
//   [63 .. 56]: generation stamp: the iteration at which the entry was last written;
//   [55 ..  0]: global rank.
// Requires `n < 2^56`.
namespace
{
    constexpr idx_t GEN_BITS = 8;   // Number of bits dedicated to generation-tracking.
    constexpr idx_t GEN_SHIFT = 64 - GEN_BITS;  // Position of the generation-bits in a `rho`-entry.
    constexpr idx_t RANK_MASK = (idx_t(1) << GEN_SHIFT) - 1;    // Mask to extract actual ranks from `rho`-entries.
    constexpr idx_t LOOK_BACK_BINS = idx_t(1) << GEN_BITS;  // Bin-count in look-back query histogram of generations.
}


// Per reader-thread stats accumulators.
struct alignas(2 * util::L1_CACHE_LINE_SIZE)
    Reader_Stat
{
    uint64_t read_ns = 0;   // Time spent in the backend `pread`.
    uint64_t gather_ns = 0; // Time spent in the `rho`-probe fold.
    uint64_t wait_ns = 0;   // Time spent in blocked-waiting for a free staging-slab.

    uint64_t gathers = 0;   // ISA-probes issued.
    uint64_t oob = 0;   // Probes short-circuited by `j + k >= n`.
    uint64_t look_back[LOOK_BACK_BINS] = {};    // Look-back histogram: generations elapsed since the probed entry was last written.

    void reset()
    {
        read_ns = gather_ns = wait_ns = 0;
        gathers = oob = 0;
        std::memset(look_back, 0, sizeof(look_back));
    }

    // Records a probe issued at generation `gen`. `v` is the raw `rho` entry
    // read, and is meaningful only when `in_range`.
    void observe(const idx_t gen, const bool in_bound, const idx_t v)
    {
        gathers++;
        if(!in_bound)
        {
            oob++;
            return;
        }

        look_back[(gen - (v >> GEN_SHIFT)) & (LOOK_BACK_BINS - 1)]++;
    }
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
    mutable uint64_t look_back_[LOOK_BACK_BINS] = {};   // Look-back histogram over the whole run.


    // Prints the look-back histogram `look_back` over `gathers` probes, of
    // which `oob` were out-of-bounds.
    static void report_look_back(const char* const pre, const uint64_t gathers, const uint64_t oob, const uint64_t* const look_back)
    {
        std::fprintf(stderr, "%sprobes %llu (out-of-bound %llu, %.5f%%)\n", pre, (unsigned long long)gathers, (unsigned long long)oob, 100.0 * oob / gathers);

        uint64_t served = 0, stale = 0;
        for(idx_t b = 0; b < LOOK_BACK_BINS; ++b)
            served += look_back[b], stale += (b >= 2 ? look_back[b] : 0);

        std::fprintf(stderr, "%slook-back (generations since target's last write):\n", pre);
        uint64_t total = 0;
        for(idx_t b = 0; b < LOOK_BACK_BINS; ++b)
            if(look_back[b])
            {
                total += look_back[b];
                std::fprintf(stderr, "%s    lambda %3llu : %18llu  (%7.4f%%, total %7.3f%%)%s\n",
                                pre, (unsigned long long)b, (unsigned long long)look_back[b],
                                100.0 * look_back[b] / served, 100.0 * total / served,
                                b == 1 ? "   <- coherent + relayed" : (b == 2 ? "   <- stale begins" : ""));
            }

        std::fprintf(stderr, "%sSTALE (lambda >= 2): %llu of %llu = %.4f%%\n", pre, (unsigned long long)stale, (unsigned long long)served, 100.0 * stale / served);
    }


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

    // Returns the current generation's index.
    idx_t generation() const { return iter_; }

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
        uint64_t look_back[LOOK_BACK_BINS] = {};
        for(const auto& r : r_)
        {
            read_ns += r->read_ns, gather_ns += r->gather_ns, wait_ns += r->wait_ns;
            gathers += r->gathers, oob += r->oob;
            for(idx_t b = 0; b < LOOK_BACK_BINS; ++b)
                look_back[b] += r->look_back[b];
        }

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

        std::fprintf(stderr, "  update_global_rank      read %8.3f s | scatter %8.3f s | total %8.3f s\n",
                        read_chunk_ns_ / 1e9, scatter_ns_ / 1e9, (read_chunk_ns_ + scatter_ns_) / 1e9);

        if(active_ > 0)
            std::fprintf(stderr, "  per-op                  gather %6.2f ns | scatter %6.2f ns\n",
                            double(gather_ns) / double(active_), double(scatter_ns_) / double(active_));

        // Look-back distribution.
        if(gathers > 0)
            report_look_back("  ", gathers, oob, look_back);

        gather_total_ += gathers, oob_total_ += oob;
        for(idx_t b = 0; b < LOOK_BACK_BINS; ++b)
            look_back_[b] += look_back[b];

        std::fflush(stderr);
    }

    // Reports the run-level look-back distribution.
    void report_summary() const
    {
        std::cerr << "\n[ISA run summary]\n";
        if(gather_total_ > 0)
            report_look_back("  ", gather_total_, oob_total_, look_back_);
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

// Stamps generation `g` into the high bits of global-rank `r`.
#define INSTRUMENT_STAMP(g, r) ((((::gpu_saca::idx_t)(g)) << ::gpu_saca::isa_instrument::GEN_SHIFT) | ((::gpu_saca::idx_t)(r)))

// Recovers the rank from a generation-stamped entry `v`.
#define INSTRUMENT_UNSTAMP(v) (((::gpu_saca::idx_t)(v)) & ::gpu_saca::isa_instrument::RANK_MASK)

#else   // ISA_INSTRUMENT

#define INSTRUMENT(x) ((void)0)
#define INSTRUMENT_TIMER(sink) ((void)0)
#define INSTRUMENT_STAMP(g, r) (r)
#define INSTRUMENT_UNSTAMP(v) (v)

#endif  // ISA_INSTRUMENT



#endif
