
#include "GPU_SACA.cuh"
#include "types.hpp"
#include "utility.hpp"
#include "utility.cuh"
#include "tests.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cassert>


int main(int argc, char** argv)
{
    (void)argc, (void)argv;

    // using T_sym_ = uint8_t;
    using T_sym_ = uint16_t;
    constexpr auto null_count = sizeof(gpu_saca::Prefix_Key) / sizeof(T_sym_);

    const std::string file_path(argv[1]);
    const auto sym_count = gpu_saca::util::file_size<T_sym_>(file_path);
    // const std::size_t sym_count = std::atoll(argv[1]);

    auto const T = gpu_saca::util::allocate<T_sym_>(sym_count + null_count);
    const auto n = sym_count + 1;

    {
        std::ifstream is(file_path, std::ios::binary);
        is.read(reinterpret_cast<char*>(T), sym_count * sizeof(T));
        // gpu_saca::util::generate_random_text(T, sym_count);

        for(std::size_t i = 0; i < null_count; ++i)
            T[sym_count + i] = std::numeric_limits<T_sym_>::min();
    }

    std::cerr << "Sequence length: " << n << "\n";

    {
        gpu_saca::idx_t* SA;
        {
            gpu_saca::GPU_SACA<T_sym_> saca(T, n);

            const auto t_0 = caps_sa::timer::now();
            saca.build_SA(SA);
            const auto t_1 = caps_sa::timer::now();
            std::cerr << "Time taken: " << caps_sa::timer::duration(t_1 - t_0) << "s.\n";
        }

#if defined(TESTS) || defined(VALIDATE)
        std::cerr << "Suffix array validation: " << (gpu_saca::test::is_correct_SA(SA, T, n) ? "Passes" : "Fails") << ".\n";
#endif
    }


    gpu_saca::util::deallocate(T);


    return 0;
}
