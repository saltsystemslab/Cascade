
#ifndef BENCH_UTILITY_HPP
#define BENCH_UTILITY_HPP



#include <string>
#include <fstream>
#include <ctime>
#include <unistd.h>


namespace gpu_saca::bench
{

// Returns `s`'s form with escape-symbols expanded to literal forms.
inline std::string esc(const std::string& s)
{
    std::string o;
    o.reserve(s.size() + 8);

    for(char c: s)
        switch(c)
        {
            case '"':   o += "\\\"";    break;
            case '\\':  o += "\\\\";    break;
            case '\n':  o += "\\n";     break;
            case '\r':  o += "\\r";     break;
            case '\t':  o += "\\t";     break;
            default:    o += c;
        }

    return o;
}


// Returns the current UTC-time.
inline std::string now_utc()
{
    const auto t = std::time(nullptr);
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));

    return std::string(b);
}


// Returns the name of the host machine.
inline std::string host_name()
{
    char b[256] = {0};
    if(::gethostname(b, sizeof(b) - 1) != 0)  return "unknown";

    return std::string(b);
}


// Returns the peak resident set size, from `VmHWM`. This is the high-water mark
// over the whole process lifetime.
inline int64_t peak_rss_bytes()
{
    std::ifstream f("/proc/self/status");
    std::string k;

    while(f >> k)
        if(k == "VmHWM:")
        {
            int64_t kb = 0;
            f >> kb;
            return kb * 1024;
        }

    return 0;
}

}



#endif
