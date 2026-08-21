
#ifndef BENCH_RECORD_HPP
#define BENCH_RECORD_HPP



#include "bench_utility.hpp"


/*
Result schema for the benchmark harness.

Records go to <dir>/<run_id>.json, concatenated at analysis time. If
<dir>/<run_id>.json is absent after the job exits, the job was killed.
*/

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <unistd.h>


namespace gpu_saca::bench
{

static constexpr int SCHEMA_VERSION = 1;

/*
Terminal outcome. Failure is a *result*: budget-sweep experiments' outcome for
various methods will include terminations, so the termination statuses needs to
reach the store.
*/
namespace st
{
    constexpr const char* OK         = "ok";
    constexpr const char* OOM_HOST   = "oom_host";
    constexpr const char* OOM_DEVICE = "oom_device";
    constexpr const char* BAD_INPUT  = "bad_input";
    constexpr const char* LIB_ERROR  = "lib_error";
    constexpr const char* KILLED     = "killed";
    constexpr const char* TIMEOUT    = "timeout";
}


// Returns a unique run-ID for a method `method`'s run on corpus `corpus`, over
// its `n`-sized prefix.
inline std::string make_run_id(const std::string& method, const std::string& corpus, int64_t n)
{
    std::ostringstream o;
    o << method << "_" << corpus << "_n" << n << "_" << std::time(nullptr) << "_" << ::getpid();

    return o.str();
}


// Dumps the run-record `rec` inside the directory `dir`.
template <typename T_rec_>
inline bool dump_json_record(const T_rec_& rec, const std::string& dir)
{
    const auto path = dir + "/" + rec.run_id + T_rec_::FILE_EXT;

    std::ofstream f(path);
    if(!f)
        return false;

    f << rec.to_json() << "\n";

    return f.good();
}


// A benchmark run's record.
struct Run_Record
{
    // Metadata

    static constexpr const char* FILE_EXT = ".json";    // Run-record file-extension.
    int schema_version = SCHEMA_VERSION;    // Schema-generation.
    std::string run_id; // Unique ID of the run.
    std::string timestamp_utc = now_utc();  // Run-start timestamp.
    std::string host = host_name(); // Machine identifier.
    std::string method; // Method name.
    std::string method_version; // Method's version.
    std::string commit; // Method's commit used in the driver.

    // Input

    std::string input_path; // Input file.
    std::string corpus; // Label of the input.
    int64_t n_tokens = 0;   // Number of tokens actually present in the input.
    int64_t n_symbols = 0;  // Number of symbols sorted by the method.
    int64_t file_size = 0;  // Input file size.
    int64_t file_mtime = 0; // Input file's last modification time.
    std::string cache_policy = "as-is"; // What's done with the page-cache after the run: "warm", "cold", or "as-is".

    // Config

    int64_t threads = 0;    // Number of processor threads used in the run.
    int64_t fs = 0; // `libsais`'s free-space parameter.
    int device = -1;    // CUDA device ordinal.

    // Outcome

    std::string status = "unknown"; // Terminal outcome.
    int64_t code = 0; // Library return code.
    std::string detail; // Details on the outcome.

    // Timings in seconds; -1 is not measured, 0.0 is measured as zero

    double t_load = -1.0;   // Input loading time.
    double t_convert = -1.0;    // Alphabet conversion time.
    double t_alloc = -1.0;  // Output allocation time.
    double t_core = -1.0;   // Method execution time.
    double t_write = -1.0;  // Output writing time.
    double t_e2e = -1.0;    // Full end-to-end time.

    // Memory

    int64_t peak_rss = 0;   // Peak RSS.
    int64_t peak_device = 0;    // Peak device RAM usage.
    int64_t sa_element_bytes = 0;   // Bytes per entry of the produced SA.

    std::string notes;  // Misc notes.


    std::string to_json() const
    {
        std::ostringstream o;
        o.setf(std::ios::fixed);
        o.precision(6);

        o << "{"
          << "\"schema_version\":"      << schema_version
          << ",\"run_id\":\""           << esc(run_id)          << "\""
          << ",\"timestamp_utc\":\""    << esc(timestamp_utc)   << "\""
          << ",\"host\":\""             << esc(host)            << "\""
          << ",\"method\":\""           << esc(method)          << "\""
          << ",\"method_version\":\""   << esc(method_version)  << "\""
          << ",\"commit\":\""           << esc(commit)     << "\""
          << ",\"input_path\":\""       << esc(input_path)      << "\""
          << ",\"corpus\":\""           << esc(corpus)          << "\""
          << ",\"n_tokens\":"           << n_tokens
          << ",\"n_symbols\":"          << n_symbols
          << ",\"file_size\":"         << file_size
          << ",\"file_mtime\":"        << file_mtime
          << ",\"cache_policy\":\""     << esc(cache_policy)    << "\""
          << ",\"threads\":"            << threads
          << ",\"fs\":"                 << fs
          << ",\"device\":"             << device
          << ",\"status\":\""           << esc(status)          << "\""
          << ",\"code\":"               << code
          << ",\"detail\":\""           << esc(detail)          << "\""
          << ",\"t_load\":"             << t_load
          << ",\"t_convert\":"          << t_convert
          << ",\"t_alloc\":"            << t_alloc
          << ",\"t_core\":"             << t_core
          << ",\"t_write\":"            << t_write
          << ",\"t_e2e\":"              << t_e2e
          << ",\"peak_rss\":"           << peak_rss
          << ",\"peak_device\":"        << peak_device
          << ",\"sa_element_bytes\":"   << sa_element_bytes
          << ",\"notes\":\""            << esc(notes)           << "\""
          << "}";

        return o.str();
    }


    bool write(const std::string& dir) const { return dump_json_record(*this, dir); }
};


// A verification run's record.
struct Verify_Record
{
    // Metadata

    static constexpr const char* FILE_EXT = ".verify.json"; // Verify-run-record file-extension.
    int schema_version = SCHEMA_VERSION;    // Schema-generation.
    std::string run_id; // Can join to the corresponding `Run_Record`
    std::string timestamp_utc = now_utc();  // Run-start timestamp.
    std::string sa_path;    // Checked SA's path.
    int64_t n = 0;  // Entries checked.

    // "pass" | "fail" | "skip"
    std::string permutation = "skip";   // Permutation-check.
    std::string order = "skip"; // Ordering-check.
    std::string crossref = "skip";  // Exact equality against some other SA dump.
    std::string crossref_against;   // The other SA's path, crossref'ed against.
    int64_t first_mismatch = -1;    // Index of first mismatch.
    std::string detail; // Details.


    std::string to_json() const
    {
        std::ostringstream o;
        o << "{"
          << "\"schema_version\":"      << schema_version
          << ",\"run_id\":\""           << esc(run_id)          << "\""
          << ",\"timestamp_utc\":\""    << esc(timestamp_utc)   << "\""
          << ",\"sa_path\":\""          << esc(sa_path)         << "\""
          << ",\"n\":"                  << n
          << ",\"permutation\":\""      << esc(permutation)     << "\""
          << ",\"order\":\""            << esc(order)           << "\""
          << ",\"crossref\":\""         << esc(crossref)        << "\""
          << ",\"crossref_against\":\"" << esc(crossref_against)<< "\""
          << ",\"first_mismatch\":"     << first_mismatch
          << ",\"detail\":\""           << esc(detail)          << "\""
          << "}";

        return o.str();
    }


    bool write(const std::string& dir) const { return dump_json_record(*this, dir); }
};

}

#endif
