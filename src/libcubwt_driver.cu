
#include "Parser.hpp"
#include "utility.hpp"

#include "libcubwt/libcubwt.cuh"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <string_view>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char *libcubwt_strerror(int64_t code)
{
    switch (code)
    {
        case LIBCUBWT_NO_ERROR:              return "no error";
        case LIBCUBWT_BAD_PARAMETER:         return "bad parameter (check n >= 16 and pointers)";
        case LIBCUBWT_NOT_ENOUGH_MEMORY:     return "not enough host memory";
        case LIBCUBWT_GPU_ERROR:             return "CUDA error";
        case LIBCUBWT_GPU_NOT_SUPPORTED:     return "GPU not supported";
        case LIBCUBWT_GPU_NOT_ENOUGH_MEMORY: return "not enough GPU memory";
        default:                             return "unknown error";
    }
}

// Print SA with the corresponding suffix string (first 40 chars for readability)
static void print_sa(const uint8_t *T, int64_t n, const uint32_t *SA, int64_t len)
{
    printf("\n  %-6s  %-6s  Suffix\n", "Rank", "SA[i]");
    printf("  %-6s  %-6s  ------\n", "------", "------");
    for (int64_t i = 0; i < len; i++)
    {
        uint32_t pos = SA[i];
        // Print up to 40 chars of the suffix at T[pos..]
        char buf[41];
        int64_t suf_len = n - pos;
        int64_t show    = suf_len < 40 ? suf_len : 40;
        memcpy(buf, T + pos, (size_t)show);
        buf[show] = '\0';
        printf("  %-6lld  %-6u  \"%s\"%s\n",
               (long long)i, pos, buf,
               suf_len > 40 ? "..." : "");
    }
}

// ---------------------------------------------------------------------------
// CPU brute-force reference SA (correctness check only, O(n^2 log n))
// ---------------------------------------------------------------------------

static std::vector<uint32_t> reference_sa(const uint8_t *T, int64_t n)
{
    std::vector<uint32_t> sa((size_t)n);
    for (int64_t i = 0; i < n; i++) sa[(size_t)i] = (uint32_t)i;
    std::sort(sa.begin(), sa.end(), [&](uint32_t a, uint32_t b)
    {
        int64_t la = n - a, lb = n - b;
        int cmp = memcmp(T + a, T + b, (size_t)(la < lb ? la : lb));
        if (cmp != 0) return cmp < 0;
        return la < lb;
    });
    return sa;
}

// ISA from SA: ISA[SA[i]] = i
static std::vector<uint32_t> sa_to_isa(const std::vector<uint32_t> &sa)
{
    size_t n = sa.size();
    std::vector<uint32_t> isa(n);
    for (size_t i = 0; i < n; i++)
        isa[sa[i]] = (uint32_t)i;
    return isa;
}


// Example usage
int example()
{
    // -----------------------------------------------------------------------
    // Input — must be >= 16 bytes (library hard minimum)
    // -----------------------------------------------------------------------
    // const std::string input = "banana$panana$!!";   // exactly 16 bytes
    const std::string input = "abracadabra_and_more_padding____";

    const uint8_t *T = reinterpret_cast<const uint8_t *>(input.c_str());
    const int64_t  n = (int64_t)input.size();

    printf("=== libcubwt v1.0.0 — Suffix Array Example ===\n");
    printf("Input  : \"%s\"\n", input.c_str());
    printf("Length : %lld bytes\n\n", (long long)n);

    // -----------------------------------------------------------------------
    // 1. Allocate reusable GPU storage
    // -----------------------------------------------------------------------
    void *device_storage = nullptr;
    int64_t err = libcubwt_allocate_device_storage(&device_storage, n);
    if (err != LIBCUBWT_NO_ERROR)
    {
        fprintf(stderr, "libcubwt_allocate_device_storage failed: %s\n",
                libcubwt_strerror(err));
        return 1;
    }
    printf("[1/4] Device storage allocated for max_length=%lld\n", (long long)n);

    // -----------------------------------------------------------------------
    // 2. Build suffix array
    // -----------------------------------------------------------------------
    std::vector<uint32_t> SA((size_t)n);

    err = libcubwt_sa(device_storage, T, SA.data(), n);
    if (err != LIBCUBWT_NO_ERROR)
    {
        fprintf(stderr, "libcubwt_sa failed: %s\n", libcubwt_strerror(err));
        libcubwt_free_device_storage(device_storage);
        return 1;
    }
    printf("[2/4] Suffix array built\n");

    // -----------------------------------------------------------------------
    // 3. Build inverse suffix array
    // -----------------------------------------------------------------------
    std::vector<uint32_t> ISA((size_t)n);

    err = libcubwt_isa(device_storage, T, ISA.data(), n);
    if (err != LIBCUBWT_NO_ERROR)
    {
        fprintf(stderr, "libcubwt_isa failed: %s\n", libcubwt_strerror(err));
        libcubwt_free_device_storage(device_storage);
        return 1;
    }
    printf("[3/4] Inverse suffix array built\n");

    // -----------------------------------------------------------------------
    // 4. Free GPU storage
    // -----------------------------------------------------------------------
    err = libcubwt_free_device_storage(device_storage);
    if (err != LIBCUBWT_NO_ERROR)
    {
        fprintf(stderr, "libcubwt_free_device_storage failed: %s\n",
                libcubwt_strerror(err));
        return 1;
    }
    printf("[4/4] Device storage freed\n");

    // -----------------------------------------------------------------------
    // Print SA
    // -----------------------------------------------------------------------
    printf("\n--- Suffix Array (libcubwt) ---");
    print_sa(T, n, SA.data(), n);

    // -----------------------------------------------------------------------
    // Verify against CPU reference
    // -----------------------------------------------------------------------
    printf("\n--- Correctness check vs. CPU reference ---\n");
    std::vector<uint32_t> ref_sa  = reference_sa(T, n);
    std::vector<uint32_t> ref_isa = sa_to_isa(ref_sa);

    bool sa_ok  = (SA  == ref_sa);
    bool isa_ok = (ISA == ref_isa);

    printf("  SA  matches reference: %s\n", sa_ok  ? "YES ✓" : "NO ✗");
    printf("  ISA matches reference: %s\n", isa_ok ? "YES ✓" : "NO ✗");

    if (!sa_ok)
    {
        printf("\n  First mismatch in SA:\n");
        for (int64_t i = 0; i < n; i++)
        {
            if (SA[(size_t)i] != ref_sa[(size_t)i])
            {
                printf("    SA[%lld]: got %u, expected %u\n",
                       (long long)i, SA[(size_t)i], ref_sa[(size_t)i]);
                break;
            }
        }
    }
    if (!isa_ok)
    {
        printf("\n  First mismatch in ISA:\n");
        for (int64_t i = 0; i < n; i++)
        {
            if (ISA[(size_t)i] != ref_isa[(size_t)i])
            {
                printf("    ISA[%lld]: got %u, expected %u\n",
                       (long long)i, ISA[(size_t)i], ref_isa[(size_t)i]);
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Show ISA: rank of each position in the original string
    // -----------------------------------------------------------------------
    printf("\n--- Inverse Suffix Array (ISA) ---\n");
    printf("  ISA[i] = rank of suffix starting at position i\n");
    printf("  %-6s  %-6s  Char at T[i]\n", "i", "ISA[i]");
    printf("  %-6s  %-6s  -----------\n", "------", "------");
    for (int64_t i = 0; i < n; i++)
        printf("  %-6lld  %-6u  '%c'\n", (long long)i, ISA[(size_t)i], (char)T[i]);

    return (sa_ok && isa_ok) ? 0 : 1;
}


int bench_libcubwt(const char* const T, const std::size_t n, uint32_t* const SA)
{
    // Allocate GPU storage
    void* device_storage = nullptr;
    auto err = libcubwt_allocate_device_storage(&device_storage, n);
    if(err != LIBCUBWT_NO_ERROR)
    {
        fprintf(stderr, "libcubwt_allocate_device_storage failed: %s\n", libcubwt_strerror(err));
        return 1;
    }


    // Build suffix array
    err = libcubwt_sa(device_storage, reinterpret_cast<const uint8_t*>(T), SA, n);
    if(err != LIBCUBWT_NO_ERROR)
    {
        fprintf(stderr, "libcubwt_sa failed: %s\n", libcubwt_strerror(err));
        libcubwt_free_device_storage(device_storage);
        return 1;
    }


    // Free GPU storage
    err = libcubwt_free_device_storage(device_storage);
    if(err != LIBCUBWT_NO_ERROR)
    {
        fprintf(stderr, "libcubwt_free_device_storage failed: %s\n", libcubwt_strerror(err));
        return 1;
    }


    return 0;
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    // example();

    (void)argc, (void)argv;

    const std::string file_path(argv[1]);
    const auto sym_count = gpu_saca::util::file_size<uint8_t>(file_path);
    auto const T = gpu_saca::util::allocate<char>(sym_count);
    const auto n = sym_count;

    {
        std::ifstream is(file_path, std::ios::binary);
        is.read(reinterpret_cast<char*>(T), sym_count * sizeof(T));
    }

    std::cerr << "Sequence length: " << n << "\n";

    {
        std::vector<uint32_t> SA(n);

        const auto t_0 = gpu_saca::timer::now();
        bench_libcubwt(T, n, SA.data());
        const auto t_1 = gpu_saca::timer::now();

        std::cerr << "Built the SA using `libcubwt`. Time taken: " << gpu_saca::timer::duration(t_1 - t_0) << " seconds.\n";

        // std::cerr << "SA matches reference: " << (SA == reference_sa(reinterpret_cast<const uint8_t*>(T), n) ? "YES" : "NO") << "\n";
    }

    gpu_saca::util::deallocate(T);

    return 0;
}
