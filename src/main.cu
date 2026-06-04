
#include "Suffix_Array.hpp"
#include "GPU_SACA.cuh"
#include "Parser.hpp"
#include "utility.hpp"
#include "utility.cuh"

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
/*
    caps_sa::Parser input(argv[1]);
    input.read_next_seq();

    auto const T = input.seq();
    auto const n = input.seq_len();

*/
    // using T_sym_ = uint8_t;
    using T_sym_ = uint16_t;

    // const std::string file_path(argv[1]);
    // const auto sym_count = gpu_saca::util::file_size<T_sym_>(file_path);
    const std::size_t sym_count = std::atoll(argv[1]);
    const auto null_count = 16 / sizeof(T_sym_);

    auto const T = gpu_saca::util::allocate<T_sym_>(sym_count + null_count);
    const auto n = sym_count + 1;

    {
        // std::ifstream is(file_path, std::ios::binary);
        // is.read(reinterpret_cast<char*>(T), sym_count * sizeof(T));
        gpu_saca::util::generate_random_text(T, sym_count);

        for(std::size_t i = 0; i < null_count; ++i)
            T[sym_count + i] = std::numeric_limits<T_sym_>::min();
    }

    std::cerr << "Sequence length: " << n << "\n";

    {
        gpu_saca::GPU_SACA<T_sym_> saca(T, n);

        const auto t_0 = caps_sa::timer::now();
        saca.build_SA(nullptr);
        const auto t_1 = caps_sa::timer::now();
        std::cerr << "Time taken: " << caps_sa::timer::duration(t_1 - t_0) << "s.\n";
    }


    gpu_saca::util::deallocate(T);

/*
    {
        std::vector<std::size_t> SA(n + 1);
        // caps_sa::build_SA_naive(T, n, SA);
        // caps_sa::build_SA_prefix_doubling(T, n, SA);

        caps_sa::Suffix_Array Suf_Arr(T, n);

        const auto t_0 = caps_sa::now();
        Suf_Arr.build_SA(SA.data());
        const auto t_1 = caps_sa::now();

        std::cerr << "Built the SA using `cu-caps-sa`. Time taken: " << caps_sa::duration(t_1 - t_0) << " seconds.\n";

        std::cerr << "Suffix array is " << (Suf_Arr.validate() ? "correct" : "wrong") << "\n";
    }
*/


    return 0;
}
