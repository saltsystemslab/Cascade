#ifndef PROFILE_HPP
#define PROFILE_HPP


// NVTX-based profiling scopes. Guarded by `-DPROFILE`; expands to nothing when
// the flag is not set, so release builds pay zero cost.
//
// Usage:
//   PROFILE_SCOPE("phase_name");              // scoped range in current block
//   PROFILE_SCOPE_FMT("iter_%d", k);          // scoped range with formatted name
//
// Under the hood: uses the NVTX3 C API. Header-only (no link flag needed on
// CUDA 12.x — the API resolves through injected function pointers when a
// profiler like `nsys` is attached, and is a no-op otherwise).


#ifdef PROFILE

#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdarg>


namespace gpu_saca::profile
{

// RAII scoped NVTX range. The `nvtxRangePushA` / `nvtxRangePop` pair is
// per-thread, so nested scopes work correctly across host threads.
class scoped_range
{
public:
    explicit scoped_range(const char* name) { nvtxRangePushA(name); }
    ~scoped_range()                          { nvtxRangePop(); }

    scoped_range(const scoped_range&)            = delete;
    scoped_range& operator=(const scoped_range&) = delete;
    scoped_range(scoped_range&&)                 = delete;
    scoped_range& operator=(scoped_range&&)      = delete;
};


// Scoped NVTX range with a printf-style formatted name. Prefer `scoped_range`
// for hot loops with a static name.
class scoped_range_fmt
{
public:
    scoped_range_fmt(const char* fmt, ...)
    {
        char buf[128];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        nvtxRangePushA(buf);
    }

    ~scoped_range_fmt() { nvtxRangePop(); }

    scoped_range_fmt(const scoped_range_fmt&)            = delete;
    scoped_range_fmt& operator=(const scoped_range_fmt&) = delete;
    scoped_range_fmt(scoped_range_fmt&&)                 = delete;
    scoped_range_fmt& operator=(scoped_range_fmt&&)      = delete;
};

}   // namespace gpu_saca::profile


#define PROFILE_CONCAT_INNER(a, b) a##b
#define PROFILE_CONCAT(a, b) PROFILE_CONCAT_INNER(a, b)

#define PROFILE_SCOPE(name) \
    ::gpu_saca::profile::scoped_range PROFILE_CONCAT(_prof_range_, __LINE__){name}

#define PROFILE_SCOPE_FMT(fmt, ...) \
    ::gpu_saca::profile::scoped_range_fmt PROFILE_CONCAT(_prof_range_, __LINE__){fmt, __VA_ARGS__}

#else   // PROFILE

#define PROFILE_SCOPE(name)                 ((void)0)
#define PROFILE_SCOPE_FMT(fmt, ...)         ((void)0)

#endif  // PROFILE


#endif  // PROFILE_HPP