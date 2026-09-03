
#include "Pipeline_Probe.hpp"


#include <cstdio>


namespace gpu_saca::probe
{

void Pipeline_Probe::init(  const std::size_t reader_c, const std::size_t stager_c, const std::size_t dispatcher_c, const std::size_t writer_c,
                            const std::size_t slab_c, const std::size_t asm_reserve)
{
    role_c_ = {reader_c, stager_c, dispatcher_c, writer_c};
    slab_c_ = slab_c, admit_budget_ = asm_reserve;

    std::size_t off = 0;
    for(std::size_t r = 0; r < ROLE_C; ++r)
        base_[r] = off, off += role_c_[r];

    scratch_ = off++;

    w_.clear();
    w_.reserve(off);
    for(std::size_t i = 0; i < off; ++i)
        w_.emplace_back(std::make_unique<Worker_Stat>());
}


void Pipeline_Probe::bind(const Role r, const std::size_t i)
{
    slot() = base_[static_cast<std::size_t>(r)] + i;
}


void Pipeline_Probe::start_sampler()
{
    if(!depth_fn_ || sampling_.load(std::memory_order_relaxed))
        return;

    sampling_.store(true, std::memory_order_relaxed);
    sampler_ = std::thread([this]
    {
        uint64_t d[QUEUE_C];
        while(sampling_.load(std::memory_order_relaxed))
        {
            depth_fn_(d);
            for(std::size_t q = 0; q < QUEUE_C; ++q)
                if(d[q] != QUEUE_ABSENT)
                    q_[q].add(d[q]);

            std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms_));
        }
    });
}


void Pipeline_Probe::stop_sampler()
{
    if(!sampling_.exchange(false, std::memory_order_relaxed))
        return;

    if(sampler_.joinable())
        sampler_.join();
}


void Pipeline_Probe::track_assembly(const idx_t prev, const idx_t chunk_c)
{
    if(prev == chunk_c && chunk_c > 1)  // First slab in: the assembly is now partial.
        fetch_max(asm_partial_max_, asm_partial_.fetch_add(1, std::memory_order_relaxed) + 1);

    if(prev == 1)   // Last slab in: the assembly is complete and moves on to the device.
        asm_partial_.fetch_sub(chunk_c > 1 ? 1 : 0, std::memory_order_relaxed),
        parked_slab_.fetch_sub(static_cast<int64_t>(chunk_c) - 1, std::memory_order_relaxed);
    else
        fetch_max(parked_slab_max_, parked_slab_.fetch_add(1, std::memory_order_relaxed) + 1);
}


void Pipeline_Probe::begin_iteration(const idx_t gen, const idx_t ctx, const idx_t active)
{
    iter_ = gen, ctx_ = ctx, active_ = active;
    pipeline_ns_ = 0;

    for(auto& s : w_)
        s->reset();

    for(auto& q : q_)
        q.reset();

    asm_partial_.store(0, std::memory_order_relaxed), asm_partial_max_.store(0, std::memory_order_relaxed);
    parked_slab_.store(0, std::memory_order_relaxed), parked_slab_max_.store(0, std::memory_order_relaxed);
}


void Pipeline_Probe::report_role(const Role r) const
{
    const auto r_i = static_cast<std::size_t>(r);
    const auto n = role_c_[r_i];
    if(n == 0)
        return;

    Worker_Stat t;  // Role-aggregate.
    for(std::size_t i = 0; i < n; ++i)
    {
        const auto& s = *w_[base_[r_i] + i];
        t.io_ns += s.io_ns, t.stage_ns += s.stage_ns, t.device_ns += s.device_ns;
        t.wait_in_ns += s.wait_in_ns, t.wait_slab_ns += s.wait_slab_ns;
        t.items += s.items, t.bytes += s.bytes;
    }

    const double denom = static_cast<double>(pipeline_ns_) * n; // The pool's aggregate thread-time.
    const auto pct = [denom](const uint64_t ns) { return denom > 0.0 ? 100.0 * static_cast<double>(ns) / denom : 0.0; };
    const double idle = 100.0 - pct(t.busy_ns()) - pct(t.wait_in_ns) - pct(t.wait_slab_ns);
    const double sec = static_cast<double>(pipeline_ns_) / 1e9;

    std::fprintf(stderr, "  %-11s %4zu  %6.1f %6.1f %6.1f  %7.1f %6.1f %7.1f  %11llu %9.1f %7.2f\n",
                    role_name(r), n,
                    pct(t.io_ns), pct(t.stage_ns), pct(t.device_ns),
                    pct(t.wait_in_ns), pct(t.wait_slab_ns), idle,
                    static_cast<unsigned long long>(t.items),
                    static_cast<double>(t.bytes) / 1e9,
                    sec > 0.0 ? static_cast<double>(t.bytes) / 1e9 / sec : 0.0);
}


void Pipeline_Probe::report_queues() const
{
    std::fprintf(stderr, "  %-16s %9s %7s %7s %8s\n", "queue", "mean", "min", "max", "%empty");
    for(std::size_t q = 0; q < QUEUE_C; ++q)
    {
        if(q_[q].n == 0)
        {
            std::fprintf(stderr, "  %-16s %9s %7s %7s %8s\n", queue_name(q), "-", "-", "-", "-");
            continue;
        }

        std::fprintf(stderr, "  %-16s %9.1f %7llu %7llu %8.1f\n",
                        queue_name(q), q_[q].mean(),
                        static_cast<unsigned long long>(q_[q].min_n), static_cast<unsigned long long>(q_[q].max_n),
                        q_[q].empty_pct());
    }
}


void Pipeline_Probe::report_assemblies() const
{
    const auto parked = parked_slab_max_.load(std::memory_order_relaxed);
    const auto open = asm_partial_max_.load(std::memory_order_relaxed);
    const auto transit = (slab_c_ > admit_budget_ ? slab_c_ - admit_budget_ : 0);   // Slabs held back for reader- and writer-transit.

    std::fprintf(stderr, "  assemblies: open max %llu | slabs parked max %llu of %zu admissible (pool %zu, transit reserve %zu)%s\n",
                    (unsigned long long)open, (unsigned long long)parked,
                    admit_budget_, slab_c_, transit,
                    parked > admit_budget_ ? "  <-- ADMISSION VIOLATED" : "");
}


void Pipeline_Probe::report_log() const
{
    std::fprintf(stderr, "\n[probe gen %llu, context %llu, active %llu]\n",
                    static_cast<unsigned long long>(iter_), static_cast<unsigned long long>(ctx_),
                    static_cast<unsigned long long>(active_));
    std::fprintf(stderr, "  drive_pipeline wall  %10.3f s\n\n", double(pipeline_ns_) / 1e9);

    std::fprintf(stderr, "  %-11s %4s  %6s %6s %6s  %7s %6s %7s  %11s %9s %7s\n",
                    "role", "n", "io%", "stage%", "dev%", "waitIn%", "waitF%", "idle%", "items", "GB", "GB/s");
    for(std::size_t r = 0; r < ROLE_C; ++r)
        report_role(static_cast<Role>(r));

    // Name the pool with the highest productive occupancy: the bottleneck candidate.
    {
        Role top = Role::reader;
        double top_pct = -1.0;
        for(std::size_t r = 0; r < ROLE_C; ++r)
        {
            const auto n = role_c_[r];
            if(n == 0 || pipeline_ns_ == 0)
                continue;

            uint64_t busy = 0;
            for(std::size_t i = 0; i < n; ++i)
                busy += w_[base_[r] + i]->busy_ns();

            const double p = 100.0 * double(busy) / (double(pipeline_ns_) * double(n));
            if(p > top_pct)
                top_pct = p, top = static_cast<Role>(r);
        }

        if(top_pct >= 0.0)
            std::fprintf(stderr, "  ==> busiest pool: %s (%.1f%% productive)\n", role_name(top), top_pct);
    }

    std::fprintf(stderr, "\n");
    report_queues();
    std::fprintf(stderr, "\n");
    report_assemblies();

    if(!w_.empty() && w_[scratch_]->busy_ns() + w_[scratch_]->wait_in_ns + w_[scratch_]->wait_slab_ns > 0)
        std::fprintf(stderr, "  [warn] unbound thread(s) accumulated %.3f s into the scratch slot.\n",
                        double(w_[scratch_]->busy_ns() + w_[scratch_]->wait_in_ns + w_[scratch_]->wait_slab_ns) / 1e9);

    std::fflush(stderr);
}

}
