
#include "GPU_SACA.hpp"
#include "utility.hpp"
#include "tests.hpp"

#include <string>
#include <cstdlib>
#include <iostream>
#include <cassert>


int main(int argc, char** argv)
{
    using T_sym_ = uint16_t;

    if(argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <text-file> <suffix-array-file>. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    const std::string ip_path(argv[1]);
    const std::string op_path(argv[2]);

    {
        gpu_saca::GPU_SACA<T_sym_> saca(ip_path, op_path);

        const auto t_0 = gpu_saca::timer::now();
        saca.build_SA();
        const auto t_1 = gpu_saca::timer::now();
        std::cerr << "Time taken: " << gpu_saca::timer::duration(t_1 - t_0) << "s.\n";
    }

#if defined(TESTS) || defined(VALIDATE)
    std::cerr << "Suffix array validation: " << (gpu_saca::test::is_correct_SA<T_sym_>(ip_path, op_path) ? "Passes" : "Fails") << ".\n";
#endif


    return 0;
}
