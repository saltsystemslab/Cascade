
#ifndef CAPS_SA_SUFFIX_ARRAY_HPP
#define CAPS_SA_SUFFIX_ARRAY_HPP



#include <cstdint>
#include <cstddef>


namespace caps_sa
{

typedef uint64_t idx_t;

constexpr idx_t pivot_flag = uint64_t(1) << (sizeof(idx_t) * 8 - 1);

constexpr std::size_t blk_sz = 256;

constexpr std::size_t chunk_sz = 65536;

void build_SA_naive(const char* T, std::size_t n, std::size_t* SA);

void build_SA_prefix_doubling(const char* T, std::size_t n, std::size_t* SA);


// Functor class to compare two suffixes.
struct Suffix_Comparator
{
    const char* T_d;
    const std::size_t n;

    __host__ __device__
    Suffix_Comparator(const char* T_d, const std::size_t n): T_d(T_d), n(n)
    {}

    __host__ __device__
    bool operator()(std::size_t x, std::size_t y)
    {
        x &= ~pivot_flag, y &= ~pivot_flag; // Clear the pivot-marker, if any.
        for(; x < n && y < n; ++x, ++y)
            if(T_d[x] != T_d[y])
                return T_d[x] < T_d[y];

        return x > y;   // Shorter suffix precedes.
    }
};


// Validates the current output present at `SA_d` over the text `T_d` of size
// `n` and populates the sorted-ness result of each index at `is_sorted`.
__global__ void validate_kernel(const char* const T_d, const idx_t* const SA_d, bool* const is_sorted, const idx_t n);

// Samples `k` suffixes from the chunk of size `chunk_sz` appropriate for this
// block from `input` and puts them successively into the appropriate position
// for this block at `output`.
__global__ void sample_from_chunk_with_replacement(const idx_t* input, idx_t* output, idx_t chunk_sz, idx_t k, uint64_t seed);

// Samples integers from the appropriate chunk of size `chunk_sz` for this block
// such that `k` integers are sampled by the block, and puts them successively
// into the appropriate position for this block at `output`.
__global__ void sample_with_replacement(idx_t chunk_sz, idx_t k, idx_t* output, uint64_t seed);

// Compares the suffixes `i` and `j` of the text `T_d` of length `n` and returns
// `-1 / 0 / +1` as per the `std::memcmp` return-policy.
__device__ __forceinline__ int sufcmp(const char* T_d, idx_t n, idx_t i, idx_t j);

// Returns the index `u_b` into the `m` sorted suffixes in `S` for the suffix
// `p` such that all suffixes in `S` at positions `>= u_b` compare larger to the
// suffix `p`. The suffixes are over the text `T_d` of length `n`.
__device__ __forceinline__ idx_t upper_bound(const char* T_d, idx_t n, const idx_t* S, idx_t m, idx_t p);

// Locates the `upper_bound` position of each of the `p_c` pivots from `P` at
// the the sorted arrays of suffixes each of size `chunk_sz` flattened at `SA_d`
// (only possible exception being the last chunk, which can be smaller), and
// places those positions / ranks at `R` in a flattened manner. The suffixes are
// over the text `T_d` of length `n`. The first and the last pivots are assumed
// to be flanking sentinel pivots, i.e. the pivot-placements `0` and `|X_i|`,
// for each subarray `X_i` are also placed. The all-vs-all search computations
// are structured as a 2D grid, the X-axis being the pivots and the Y-axis being
// the sorted chunks.
__global__ void locate_pivots_kernel(const char* T_d, idx_t n, const idx_t* SA_d, idx_t chunk_sz, const idx_t* P, idx_t p_c, idx_t* R);

// Computes the size of the `[i, j]`'th subchunk at `W[i, j]` in column-major
// order. There are `p` chunks and each is broken into `p` subchunks, delineated
// with `R`: `R[i][j]` stores where the `j`'th subchunk lives within the `i`'th
// chunk.
__global__ void get_subchunk_sizes_kernel(const idx_t* R, idx_t* W, idx_t p);

// Computes the size of each partition from the `p x p` grid `W` of subchunk
// sizes (which is in the column-major order). Also transforms the `W` matrix
// from containing subchunk-sizes to column-wide exclusive scans; `W[i]j[j]`
// denotes the index of the `i`'th subchunk in the `j`'th partition afterwards.
__global__ void get_partition_sizes_and_subchunk_indices_kernel(idx_t* W, idx_t* part_sz, idx_t p);

// Transforms the `p x p` matrix `W` of intra-partition subchunk offsets to
// subchunk-offsets into the flattened array of partitions. The partitions sizes
// are present at `part_sz`. `W` is in column-major order.
__global__ void get_absolute_subchunk_indices_kernel(idx_t* W, const idx_t* part_off, idx_t p);

// Scatters the `p` subchunks of the `p` chunks at `X` into `Y` appropriately.
// The chunks have size `chunk_sz` (except possibly the last one). `R[i][j]`
// stores the `j`'th subchunk's offset within the `i`'th chunk. `W[i][j]` stores
// the absolute index of the `i`'th subchunk from the `j`'th partition into the
// scatter output `Y`, in column-major order.
__global__ void scatter_subchunks_kernel(const idx_t* X, idx_t* Y, const idx_t* R, const idx_t* W, idx_t p, idx_t chunk_sz);

// The suffix array constructor class.
class Suffix_Array
{
private:

    const char* const T_h;  // The input text in host.
    char* T_d;  // The input text in device.
    const idx_t n;  // Length of the input text.

    const idx_t chunk_count;    // Number of chunks the suffix array and the text is broken into at the first phase.
    const idx_t partition_count;    // Number of final disjoint partitions of the suffix array.
    static constexpr idx_t c = 32;  // Over-sampling factor per chunk.
    const idx_t k;  // Sample per chunk: `c log n`.

    idx_t* SA_d;    // The suffix array on device.
    idx_t* SA;  // The final SA.

    idx_t* S_d; // The collection of samples.

    idx_t* P_h; // The pivots.
    idx_t* P_d; // The pivots at device.

    idx_t* P_rank;  // The ranks (upper-bound positions) of the pivots at each sorted subarray.


    // Sorts chunks of the suffixes of the text where each chunk has size
    // `chunk_sz`, into the flat array `S_d`.
    void sort_chunks(idx_t* SA_d, idx_t chunk_sz) const;

    // Collects the random sample collection of suffixes.
    void sample_suffixes();

    // Selects the pivots that demarcate the final partitions of the suffix array.
    void select_pivots();

    // Locates the upper-bound positions of the pivots in the sorted subarrays
    // and flattens them in `P_rank`. Besides these pivots, two flanking pivots,
    // `0` and `|X_i|`, for each subarray `X_i` are also placed.
    void locate_pivots();

    // Collates the subchunks delineated by the pivot locations in each sorted
    // chunk into appropriate partitions.
    void partition_sub_chunks();


public:

    // Constructs a suffix array constructor object for the input text `T` of
    // size `n`.
    Suffix_Array(const char* const T, const idx_t n);

    ~Suffix_Array();

    // Sorts chunks of the suffixes of the text with the final pivot set added
    // to the chunk. The chunks are of size `chunk_sz` and the sorted chunks are
    // placed into the flat device array `S_d`.
    void sort_chunks_with_pivots(idx_t* SA_d, idx_t chunk_sz) const;

    // Constructs the suffix array of the provided text at `SA`.
    void build_SA(idx_t* SA);

    // Validates the algorithm output.
    bool validate();
};

}



#endif
