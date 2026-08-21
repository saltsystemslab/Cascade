
/*

Benchmark harness:

    bench_baseline run    --input P --n N [--threads T] [--fs F] [--sa-out P]
                          [--cache warm|cold|as-is] [--corpus NAME]
                          [--results-dir D] [--run-id ID]

    bench_baseline verify --input P --n N --sa P [--against P]
                          [--results-dir D] [--run-id ID]
--*/

#include "corpus.hpp"
#include "record.hpp"
#include "bench_common.hpp"

#include "libsais16x64.h"
#include "libsais.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef BENCH_REPO_COMMIT
    #define BENCH_REPO_COMMIT "unknown"
#endif


namespace
{

using namespace gpu_saca::bench;


bool parse(const int argc, char** const argv, Args& args, std::string& err)
{
    if(argc < 2)
    {
        err = "usage:   bench_baseline {run|verify} --input P --n N [--threads T] [--fs F] [--sa-out P]"
                            " [--cache warm|cold|as-is] [--corpus NAME]"
                            " [--results-dir D] [--run-id ID]";
        return false;
    }

    args.mode = argv[1];
    if(args.mode != "run" && args.mode != "verify")
    {
        err = "unknown mode: " + args.mode;
        return false;
    }

    for(int i = 2; i < argc; ++i)
    {
        const std::string k = argv[i];
        const bool has_val = (i + 1 < argc);
        const std::string v = has_val ? argv[i + 1] : "";

        if(!has_val)
        {
            err = "missing value for " + k;
            return false;
        }

        ++i;

        if     (k == "--input")         args.input       = v;
        else if(k == "--sa")            args.sa          = v;
        else if(k == "--against")       args.against     = v;
        else if(k == "--sa-out")        args.sa_out      = v;
        else if(k == "--corpus")        args.corpus      = v;
        else if(k == "--cache")         args.cache       = v;
        else if(k == "--results-dir")   args.results_dir = v;
        else if(k == "--run-id")        args.run_id      = v;
        else if(k == "--n")             args.n           = std::strtoll(v.c_str(), nullptr, 10);
        else if(k == "--threads")       args.threads     = std::strtoll(v.c_str(), nullptr, 10);
        else if(k == "--fs")            args.fs          = std::strtoll(v.c_str(), nullptr, 10);
        else
        {
            err = "unknown argument: " + k;
            return false;
        }
    }

    if(args.cache != "warm" && args.cache != "cold" && args.cache != "as-is")
    {
        err = "--cache must be warm, cold or as-is";
        return false;
    }

    if(args.input.empty() || args.corpus.empty())
    {
        err = "--input and --corpus are required";
        return false;
    }

    return true;
}


int run(const Args& arg)
{
    const auto t_start = Clock::now();

    gpu_saca::bench::Run_Record rec;
    rec.method              = "libsais16x64";
    rec.method_version      = LIBSAIS_VERSION_STRING;
    rec.commit              = BENCH_REPO_COMMIT;
    rec.input_path          = arg.input;
    rec.corpus              = arg.corpus;
    rec.threads             = arg.threads;
    rec.fs                  = arg.fs;
    rec.cache_policy        = arg.cache;
    rec.sa_element_bytes    = sizeof(int64_t);

    // libsais16x64 takes u16 natively: no alphabet conversion happens.
    rec.t_convert        = 0.0;

    rec.run_id = arg.run_id.empty()
                    ?   gpu_saca::bench::make_run_id("libsais16x64", rec.corpus, arg.n)
                        : arg.run_id;

    gpu_saca::bench::Buffer<uint16_t> T;
    gpu_saca::bench::File_Prop id;
    std::string err;

    const auto emit = [&](const char* const status, const int64_t code, const std::string& detail)
    {
        rec.status   = status;
        rec.code     = code;
        rec.detail   = detail;
        rec.t_e2e    = since(t_start);
        rec.peak_rss = gpu_saca::bench::peak_rss_bytes();

        if(!rec.write(arg.results_dir))
        {
            std::fprintf(stderr, "FATAL: could not write record to %s\n", arg.results_dir.c_str());
            return 1;
        }

        std::fprintf(stderr, "%s\n", rec.to_json().c_str());

        return 0;
    };


    const auto t0 = Clock::now();

    if(!gpu_saca::bench::load_prefix(arg.input, arg.n, T, id, err, arg.cache == "cold"))
        return emit(gpu_saca::bench::st::BAD_INPUT, 0, err);

    rec.t_load = since(t0);

    rec.n_tokens = T.n;
    rec.n_symbols = T.n;
    rec.file_size = id.size;
    rec.file_mtime = id.mtime;

    if(arg.n > 0 && static_cast<int64_t>(T.n) < arg.n)
        rec.notes = "short corpus: requested " + std::to_string(arg.n);


    gpu_saca::bench::Buffer<int64_t> SA;

    const auto t1 = Clock::now();
    if(!SA.allocate(T.n + arg.fs))
        return emit(gpu_saca::bench::st::OOM_HOST, 0,
                    "SA allocation failed: " + std::to_string(T.n + arg.fs) + " entries");
    rec.t_alloc = since(t1);


    const auto t2 = Clock::now();

    const int64_t r = libsais16x64_omp(T.p, SA.p, T.n, arg.fs, nullptr, arg.threads);
    rec.t_core = since(t2);

    if(r != 0)
    {
        // Confirmed from libsais 2.10.4 source: -2 is allocation failure, -1 is
        // bad arguments.
        const char* const s = (r == -2) ? gpu_saca::bench::st::OOM_HOST : gpu_saca::bench::st::LIB_ERROR;
        return emit(s, r, "libsais16x64 returned " + std::to_string(r));
    }


    if(!arg.sa_out.empty())
    {
        const auto t3 = Clock::now();
        if(!write_SA(arg.sa_out, reinterpret_cast<const uint64_t*>(SA.p), T.n, err))
            return emit(gpu_saca::bench::st::LIB_ERROR, 0, err);
        rec.t_write = since(t3);
    }
    else
        rec.t_write = 0.0;

    return emit(gpu_saca::bench::st::OK, 0, "");
}

}


int main(int argc, char ** argv)
{
    Args        a;
    std::string err;

    if(!parse(argc, argv, a, err))
    {
        // No record: we do not know enough to identify the run. This is the
        // one case where nonzero exit is correct.
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }

    return (a.mode == "run") ?
                run(a) :
                0;
}
