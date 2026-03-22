
#include "Parser.hpp"
#include "Suffix_Array.hpp"

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

    std::size_t* const SA = reinterpret_cast<std::size_t*>(std::malloc((n + 1) * sizeof(std::size_t)));
    caps_sa::build_SA(T, n, SA);

    const std::size_t print_len = 10;
    std::cerr << "Lexicographically smallest suffix is at " << SA[0] << "; is " << std::string(T + SA[0], T + SA[0] + std::min((n + 1) - SA[0], print_len)) << "\n";
    std::cerr << "Lexicographically largest suffix is at  " << SA[n] << "; is " << std::string(T + SA[n], T + SA[n] + std::min((n + 1) - SA[n], print_len)) << "\n";

    std::free(SA);

    return 0;
}
