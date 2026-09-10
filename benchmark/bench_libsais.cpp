
// libsais baseline driver. Both alphabet widths from one binary: the two upstream
// entry points differ only in `T`'s type. Library facts: see baselines.md.

#include "bench.hpp"

#include "libsais16x64.h"
#include "libsais64.h"

#include <omp.h>


namespace
{

using namespace bench;


// The only difference between the widths. `freq` stays null: we never use the
// histogram, and supplying one would change what is measured.
inline int64_t sais_omp(const uint8_t* const T, int64_t* const SA, const int64_t n, const int64_t fs, const int64_t threads)
{
    return libsais64_omp(T, SA, n, fs, nullptr, threads);
}

inline int64_t sais_omp(const uint16_t* const T, int64_t* const SA, const int64_t n, const int64_t fs, const int64_t threads)
{
    return libsais16x64_omp(T, SA, n, fs, nullptr, threads);
}


template <typename T_sym_>
Outcome core(Ctx<T_sym_>& c)
{
    // With dynamic threads on, libsais silently runs serial and a thread sweep
    // comes back flat with no error anywhere (baselines.md). Refuse rather than
    // record a plausible-looking wrong number.
    if(omp_get_dynamic() != 0)
        return {st::BAD_PARAM, 0,
                "OMP_DYNAMIC is enabled; libsais would silently run serial. Set OMP_DYNAMIC=false."};

    Buffer<int64_t> SA;
    if(!SA.allocate(c.n + c.arg.fs))
        return {st::OOM_HOST, 0,
                "SA allocation failed: " + std::to_string(c.n + c.arg.fs) + " x 8 B"};

    const auto t0 = Clock::now();
    const int64_t r = sais_omp(c.T, SA.p, c.n, c.arg.fs, c.arg.threads);
    c.rec.t_core = since(t0);

    if(r != 0)
    {
        // -2 is an allocation failure, -1 is bad arguments -- which also covers
        // threads < 0, so the arguments go in `detail` to make -1 diagnosable.
        const char* const s = (r == -2) ? st::OOM_HOST : st::LIB_ERROR;

        return {s, r, "libsais returned " + std::to_string(r)
                    + " (n=" + std::to_string(c.n)
                    + " fs=" + std::to_string(c.arg.fs)
                    + " threads=" + std::to_string(c.arg.threads) + ")"};
    }

    if(!c.arg.sa_out.empty())
    {
        std::string err;
        const auto t1 = Clock::now();
        const bool ok = write_SA(c.arg.sa_out, SA.p, c.n, err);
        c.rec.t_write = since(t1);

        if(!ok)     return {st::LIB_ERROR, 0, err};
    }
    else
        c.rec.t_write = 0.0;

    if(c.arg.verify)
    {
        std::string detail;
        c.rec.verify = verify_SA(c.T, SA.p, c.n, detail);
        if(c.rec.verify != std::string("pass"))
            return {st::LIB_ERROR, 0, "verification " + c.rec.verify + ": " + detail};
    }

    return {st::OK, 0, ""};
}

}


int main(int argc, char** argv)
{
    bench::Args a;
    std::string err;

    if(!bench::parse(argc, argv, a, err))
    {
        // The one case where non-zero is right: without args we cannot even name
        // a record to write.
        std::fprintf(stderr, "%s\nUsage:\n%s", err.c_str(), bench::usage());
        return 2;
    }

    // One version macro per entry point; record the one matching the width used.
    return (a.alphabet == bench::alpha::TOKEN16)
            ? bench::bench_main<uint16_t>(a, "libsais16x64", LIBSAIS16X64_VERSION_STRING, core<uint16_t>)
            : bench::bench_main<uint8_t >(a, "libsais64",    LIBSAIS64_VERSION_STRING,    core<uint8_t >);
}
