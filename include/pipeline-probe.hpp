
#ifndef PIPELINE_PROBE_HPP
#define PIPELINE_PROBE_HPP



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

}



#endif
