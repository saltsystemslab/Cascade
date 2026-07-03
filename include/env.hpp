#ifndef ENV_HPP
#define ENV_HPP


// Minimal env-var accessor helpers for tuning knobs.

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>


namespace gpu_saca::env
{

inline std::optional<std::string> get(const char* const name)
{
    const char* const val = std::getenv(name);
    if(val == nullptr || *val == '\0')
        return std::nullopt;

    return std::string(val);
}


inline std::optional<uint64_t> get_u64(const char* const name)
{
    const auto s = get(name);
    if(!s)
        return std::nullopt;

    try
    {
        return std::stoull(*s);
    }
    catch(const std::exception&)
    {
        std::cerr << "Warning: env var " << name << " (`" << *s << "`) is not a valid integer; ignoring.\n";
        return std::nullopt;
    }
}


inline std::optional<double> get_f64(const char* const name)
{
    const auto s = get(name);
    if(!s)
        return std::nullopt;

    try
    {
        return std::stod(*s);
    }
    catch(const std::exception&)
    {
        std::cerr << "Warning: env var " << name << " (`" << *s << "`) is not a valid double; ignoring.\n";
        return std::nullopt;
    }
}

}


#endif
