
#ifndef PIPELINE_PROBE_HPP
#define PIPELINE_PROBE_HPP



#include "utility.hpp"

#include <cstdint>
#include <cstdio>


// Instrumentation scaffolding for the staging pipeline between the partition-
// backend and the device.
// The pipeline is a chain of thread-pools connected by queues:
//  F --> [reader] --> C --> [stager] --> R --> [dispatcher] --> W --> [writer] --> F
//
// The probing identifies per generation:
// 1.   The bottleneck pool: Per-role time is split into actual work (`io` /
//      `host` / `device`), blocked-on-input (`waitIn`), blocked-on-a-free-slab
//      (`waitF`), and unaccounted (`idle`). Each is normalized against
//      `pipeline_ns x role_count`, so the percentages are directly comparable
//      across pools of different sizes.
// 2.   The slabs' distribution: Queue depths are sampled periodically. `F`
//      sitting empty implies slab starvation; `C` backing up implies the
//      stagers not keeping pace with the readers; `R` empty implies the device
//      being starved of assemblies.
//
// The assembly gauge additionally tracks the slabs parked in incomplete
// assemblies, which is the quantity the slab-pool sizing has to cover on top of
// the `STREAM_C` assemblies held by the dispatchers.
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
constexpr uint64_t QUEUE_ABSENT = ~uint64_t(0);

inline const char* queue_name(const std::size_t q)
{
    constexpr const char* name[QUEUE_C] = {"F (free)", "C (to-stage)", "R (to-device)", "W (to-backend)"};
    return name[q];
}


// Per worker-thread stats accumulators.
struct alignas(2 * util::L1_CACHE_LINE_SIZE)
    Worker_Stat
{
    uint64_t io_ns = 0; // Time spent by: readers and writers; in: `pread` / `pwrite` / resident memcpy to / from the backend.
    uint64_t gather_ns = 0; // Time spent by: stagers; in: host-side staging-compute.
    uint64_t device_ns = 0; // Time spent by: dispatchers; in: device-dispatch (H2D uploads, kernels, D2H downloads, and stream-syncs).
    uint64_t wait_in_ns = 0;    // Time spent in: blocking on the pool's input queues: `RT` / `C` / `R` / `W`.
    uint64_t wait_slab_ns = 0;  // Time spent by: readers; in: blocking on `F` for free slabs.
    uint64_t items = 0; // Units handled: slabs (for readers, writers, and stagers), and partitions (dispatchers).
    uint64_t bytes = 0; // Bytes moved: backend traffic (for readers and writers), and PCIe traffic (for dispatchers).

    // Time spent in actual work: in I/O, host-, or device-work.
    uint64_t busy_ns() const { return io_ns + gather_ns + device_ns; }

    // Resets the stats.
    void reset() { io_ns = gather_ns = device_ns = wait_in_ns = wait_slab_ns = items = bytes = 0; }
};

}



#endif

}



#endif
