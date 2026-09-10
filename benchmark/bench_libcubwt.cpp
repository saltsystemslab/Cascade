
// libcubwt baseline driver. A .cpp, not .cu: libcubwt.cuh is `extern "C"` and the
// cudart calls are plain C, so nvcc never compiles bench.hpp.
//
// uint8-only, uint32 SA. API caps n at 4,294,959,104, but the device budget binds
// far sooner (~30 B/symbol). Details: baselines.md.

#include "bench.hpp"

#include "libcubwt/libcubwt.cuh"

#include <cuda_runtime_api.h>


namespace
{

using namespace bench;


// Ported verbatim from the driver this replaces.
const char* libcubwt_strerror(const int64_t code)
{
    switch(code)
    {
        case LIBCUBWT_NO_ERROR:                 return "no error";
        case LIBCUBWT_BAD_PARAMETER:            return "bad parameter (check n >= 16 and pointers)";
        case LIBCUBWT_NOT_ENOUGH_MEMORY:        return "not enough host memory";
        case LIBCUBWT_GPU_ERROR:                return "CUDA error";
        case LIBCUBWT_GPU_NOT_SUPPORTED:        return "GPU not supported";
        case LIBCUBWT_GPU_NOT_ENOUGH_MEMORY:    return "not enough GPU memory";
        default:                                return "unknown error";
    }
}


Outcome core(Ctx<uint8_t>& c)
{
    if(c.n < 16)
        return {st::BAD_PARAM, 0, "libcubwt requires n >= 16, got " + std::to_string(c.n)};

    // Create the context before sampling free memory, else libcubwt is charged
    // for the context's few hundred MB.
    if(cudaFree(nullptr) != cudaSuccess)
        return {st::LIB_ERROR, 0, "no usable CUDA device"};

    std::size_t free_before = 0, total = 0;
    if(cudaMemGetInfo(&free_before, &total) != cudaSuccess)
        return {st::LIB_ERROR, 0, "cudaMemGetInfo failed"};

    // Device storage BEFORE the host SA, because the device is the first
    // binding constraint
    //
    // All of libcubwt's device allocation happens here; libcubwt_sa allocates
    // nothing. So this free-memory delta is the exact device peak.
    void* storage = nullptr;
    const int64_t alloc_r = libcubwt_allocate_device_storage(&storage, c.n);
    if(alloc_r != LIBCUBWT_NO_ERROR)
    {
        const char* const s = (alloc_r == LIBCUBWT_GPU_NOT_ENOUGH_MEMORY) ? st::OOM_DEVICE
                            : (alloc_r == LIBCUBWT_NOT_ENOUGH_MEMORY)     ? st::OOM_HOST
                            : (alloc_r == LIBCUBWT_BAD_PARAMETER)         ? st::BAD_PARAM
                                                                          : st::LIB_ERROR;

        c.rec.peak_device = static_cast<int64_t>(total);    // What it could not fit into.

        return {s, alloc_r, std::string("libcubwt_allocate_device_storage: ")
                          + libcubwt_strerror(alloc_r)
                          + " (n=" + std::to_string(c.n)
                          + ", device total=" + std::to_string(total) + " B)"};
    }

    std::size_t free_after = 0;
    if(cudaMemGetInfo(&free_after, &total) == cudaSuccess && free_before >= free_after)
        c.rec.peak_device = static_cast<int64_t>(free_before - free_after);

    Buffer<uint32_t> SA;
    if(!SA.allocate(c.n))
    {
        libcubwt_free_device_storage(storage);
        return {st::OOM_HOST, 0, "host SA allocation failed: " + std::to_string(c.n) + " x 4 B"};
    }

    const auto t0 = Clock::now();
    const int64_t r = libcubwt_sa(storage, c.T, SA.p, c.n);
    c.rec.t_core = since(t0);

    libcubwt_free_device_storage(storage);

    if(r != LIBCUBWT_NO_ERROR)
    {
        const char* const s = (r == LIBCUBWT_GPU_NOT_ENOUGH_MEMORY) ? st::OOM_DEVICE
                            : (r == LIBCUBWT_NOT_ENOUGH_MEMORY)     ? st::OOM_HOST
                            : (r == LIBCUBWT_BAD_PARAMETER)         ? st::BAD_PARAM
                                                                    : st::LIB_ERROR;

        return {s, r, std::string("libcubwt_sa: ") + libcubwt_strerror(r)
                    + " (n=" + std::to_string(c.n) + ")"};
    }

    if(!c.arg.sa_out.empty())
    {
        std::string err;
        const auto t1 = Clock::now();
        // Widened to the canonical on-disk uint64, so a cross-check against
        // libsais is a plain `cmp`.
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
        std::fprintf(stderr, "%s\nUsage:\n%s", err.c_str(), bench::usage());
        return 2;
    }

    if(a.alphabet == bench::alpha::TOKEN16)
    {
        std::fprintf(stderr,
            "libcubwt is uint8-only and cannot consume a uint16 token corpus.\n"
            "To run a token corpus through it, pass --alphabet byte_from_token16;\n"
            "that reads the file as raw bytes and is a reach-only comparison\n"
            "(n doubles, little-endian byte order is not token order, alphabet 256 not 65536).\n");
        return 2;
    }

    return bench::bench_main<uint8_t>(a, "libcubwt", LIBCUBWT_VERSION_STRING, core);
}
