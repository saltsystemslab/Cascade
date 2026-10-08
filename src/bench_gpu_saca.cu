
// Benchmark driver for our own method. Placement is with `core` rather than in
// the independent `benchmark/` so that our definitions do not leak to the
// baselines, and building this does not drag in the baselines' network fetches.

#include "GPU_SACA.hpp"

#include "../benchmark/bench.hpp"

#include <cuda_runtime_api.h>
#include <omp.h>

#include <cstdio>
#include <filesystem>
#include <string>


namespace
{

using namespace bench;


// Returns `false` if the core can be detected to definitely abort due to
// filesystem issues. The core can abort deep inside execution, as the scratch
// checks fire only after the text load and the partition-IDs computation.
// Early-checking of those failure condition detects a definite abort early.
bool preflight(const Args& arg, int64_t& n, std::string& err)
{
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(arg.input, ec);
    if(ec)  { err = "cannot stat input: " + arg.input; return false; }
    if(bytes == 0)  { err = "empty input: " + arg.input; return false; }
    if(bytes % sizeof(uint16_t))    { err = "input has an odd byte count, so it is not a uint16 token array: " + arg.input; return false; }

    n = static_cast<int64_t>(bytes / sizeof(uint16_t));

    const char* const scratch = std::getenv("GPU_SACA_SCRATCH");
    if(scratch == nullptr || *scratch == '\0')  { err = "GPU_SACA_SCRATCH is unset; set it to colon-separated scratch roots"; return false; }

    std::string roots(scratch);
    for(std::size_t b = 0; b <= roots.size(); )
    {
        const auto e = roots.find(':', b);
        const auto root = roots.substr(b, (e == std::string::npos ? roots.size() : e) - b);
        if(!root.empty())
        {
            if(!std::filesystem::is_directory(root, ec))    { err = "scratch root is not a directory: " + root; return false; }
            if(::access(root.c_str(), W_OK) != 0)   { err = "scratch root is not writable: " + root; return false; }
        }

        if(e == std::string::npos)  break;
        b = e + 1;
    }

    return true;
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

    bench::Record rec;
    rec.run_id          = a.run_id;
    rec.method          = "gpu_saca";
    rec.method_version  = a.method_version.empty() ? "unknown" : a.method_version;
    rec.input           = a.input;
    rec.corpus          = a.corpus;
    rec.tokenizer       = a.tokenizer;
    rec.alphabet        = a.alphabet;
    rec.symbol_bytes    = bench::alpha::symbol_bytes(a.alphabet);
    rec.placement       = a.placement;
    rec.threads         = a.threads;

    // The shared path to record-emission.
    const auto emit = [&](const char* const status, const int64_t code, const std::string& detail)
    { return bench::finalize(rec, a.results_dir, status, code, detail); };

    // The symbol type is fixed at compile time currently.
    if(a.alphabet != bench::alpha::TOKEN16)
        return emit(bench::st::BAD_PARAM, 0, "gpu_saca builds over uint16 tokens only; --alphabet must be token16");

    // There is no prefix-truncation path currently.
    if(a.n > 0)
        return emit(bench::st::BAD_PARAM, 0, "gpu_saca has no prefix mode; --n must be 0 (whole corpus)");

    int64_t n = 0;
    if(!preflight(a, n, err))
        return emit(bench::st::BAD_INPUT, 0, err);

    rec.n_symbols = n;
    rec.n_tokens  = n;

    // Probe the device before constructing, because the constructor creates the CUDA streams and a
    // failure there is a CUDA_CHECK abort. This turns common environmental failure (no GPU, driver
    // too old) into a record, matching what `bench_libcubwt` produces.
    int dev_c = 0;
    const auto cu = cudaGetDeviceCount(&dev_c);
    if(cu != cudaSuccess)
        return emit(bench::st::LIB_ERROR, cu, std::string("no usable CUDA device: ") + cudaGetErrorString(cu));
    if(dev_c == 0)
        return emit(bench::st::LIB_ERROR, 0, "no CUDA device found");

    if(a.threads > 0)
        omp_set_num_threads(static_cast<int>(a.threads));

    // What the runtime resolved, may not necessarily be what we requested.
    rec.threads_avail = omp_get_max_threads();

    // An empty --sa-out means no dump, which is the benchmarking default.
    gpu_saca::GPU_SACA<uint16_t> saca(a.input, a.sa_out);

    const auto t0 = bench::Clock::now();
    saca.build_SA();
    const auto t_build = bench::since(t0);

    const auto& bt = saca.timings();
    rec.t_load = bt.text_load;
    rec.t_core = bt.core();
    rec.t_write = a.sa_out.empty() ? 0.0 : -1.0;    // The dump is interleaved with the shuffle across a pool of writer threads,
                                                    // as such its cost cannot be separated out.

    // The phases as fields.
    rec.t_prepage_pid  = bt.prepage_pid;
    rec.t_pivots       = bt.pivots;
    rec.t_partition    = bt.partition;
    rec.t_device_setup = bt.device_setup;
    rec.t_initial_rank = bt.initial_rank;
    rec.t_reload_pid   = bt.reload_pid;
    rec.t_exponentiate = bt.exponentiate;
    rec.t_concatenate  = bt.concatenate;
    rec.n_partitions   = bt.p;
    rec.n_generations  = bt.gen_c;
    rec.pid_spill      = bt.pid_spill;

    // `t_build` is the driver's own wall time around build_SA(); t_load + t_core
    // should account for it, and a gap is worth digging.
    rec.notes = "build=" + std::to_string(t_build) + "s";

    return emit(bench::st::OK, 0, "");
}
