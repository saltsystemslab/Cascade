
#ifndef QUERY_ENGINE_HPP
#define QUERY_ENGINE_HPP



#include "types.hpp"
#include "query-file.hpp"
#include "Suffix_Index.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>


// Batched exact-match queries over a query index: `find` returns the SA-interval
// of every pattern of a batch. This engine binary-searches the whole suffix
// array for every pattern (the direct mode): correct for any placement the
// device can access, and the table-free baseline of the two-phase search.

namespace gpu_saca
{

// Placement and load-time checks of an engine's index.
struct Query_Options
{
    Residency text_res = Residency::device; // Tier of the text.
    Residency sa_res = Residency::device;   // Tier of the suffix array.
    idx_t spot = 100000;    // Random adjacent pairs of the suffix array spot-checked for order at load; `0` skips.
};


// Query engine over indexes of symbols of type `T_sym_`.
template <typename T_sym_>
class Query_Engine
{
public:

    // Loads the index-directory `dir` as `opt` places it. The index must be of
    // symbols of `T_sym_`, placed device-accessibly. Returns `false` with a
    // reason in `err` on failure.
    bool load(const std::string& dir, const Query_Options& opt, std::string& err);

    // Finds the SA-intervals of the queries `sym[off[i], off[i + 1])`, `i < q`,
    // into `out[0, q)`; all device-side arrays, asynchronously on the stream `s`.
    void find_async(const uint64_t* off, const T_sym_* sym, idx_t q, SA_Interval* out, cudaStream_t s) const;

    // Finds the SA-intervals of the queries `sym[off[i], off[i + 1])`, `i < q`,
    // into `out[0, q)`; all host-side arrays, synchronously.
    void find(const uint64_t* off, const T_sym_* sym, idx_t q, SA_Interval* out) const;

    // Checks the intervals `I[0, q)` of the queries `sym[off[i], off[i + 1])`
    // with the certificate of `ref::certify`: adds the count of failing queries
    // to `*fail_c` and lowers `*first` to the first failing one. All device-side
    // arrays, asynchronously on the stream `s`.
    void certify_async(const uint64_t* off, const T_sym_* sym, idx_t q, const SA_Interval* I, unsigned long long* fail_c,
                       unsigned long long* first, cudaStream_t s) const;

    // Returns the loaded index.
    const Suffix_Index& index() const { return idx_; }


private:

    Suffix_Index idx_;  // The loaded index.
};

}



#endif
