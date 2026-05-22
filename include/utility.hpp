
#ifndef UTILITY_HPP
#define UTILITY_HPP



#include <chrono>


namespace caps_sa::timer
{
    // Fields for profiling time.

    typedef std::chrono::high_resolution_clock::time_point time_point_t;
    const static auto now = std::chrono::high_resolution_clock::now;
    const static auto duration = [](const std::chrono::nanoseconds& d) { return std::chrono::duration_cast<std::chrono::duration<double>>(d).count(); };
}



#endif
