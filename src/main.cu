
#include "Suffix_Array.hpp"
#include "GPU_SACA.cuh"
#include "Parser.hpp"
#include "utility.cuh"

#include <cstddef>
#include <string>
#include <cstdlib>
#include <iostream>


int main(int argc, char** argv)
{
    (void)argc, (void)argv;
    caps_sa::Parser input(argv[1]);
    input.read_next_seq();

    auto const T = input.seq();
    auto const n = input.seq_len();

    std::cerr << "Sequence length: " << n << "\n";

    {
        gpu_saca::GPU_SACA<char> saca(T, n);
    }

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
