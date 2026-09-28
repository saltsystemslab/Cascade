
#ifndef QUERY_ENGINE_HPP
#define QUERY_ENGINE_HPP



#include "types.hpp"
#include "query-file.hpp"
#include "Suffix_Index.hpp"
#include "Sample_Table.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>


// Batched exact-match queries over a query index: `find` returns the SA-interval
// of every pattern of a batch. The two-phase search ranks each pattern's packed
// prefix among the keys of a sample table, which brackets the interval's
// boundaries within a stride of ranks, and then binary-searches the suffix
// array within the brackets. Without a table, the direct mode binary-searches
// the whole suffix array: the table-free baseline, correct for any placement
// the device can access.

namespace gpu_saca
{

// Placement, load-time checks, and search mode of an engine.
struct Query_Options
{
    Residency text_res = Residency::device; // Tier of the text.
    Residency sa_res = Residency::device;   // Tier of the suffix array.
    idx_t spot = 100000;    // Random adjacent pairs of the suffix array spot-checked for order at load; `0` skips.
    idx_t stride = 32;  // Sample-table stride; `0` selects the direct mode.
    uint32_t key_words = 0; // 64-bit words per table key: 1, 2, or 4; `0` picks 2, or 4 for 4-byte symbols.
    bool sort = false;  // Whether to sort every batch by its packed prefixes before searching.
};


// Query engine over indexes of symbols of type `T_sym_`.
template <typename T_sym_>
class Query_Engine
{
private:

    Suffix_Index idx_;  // The loaded index.
    Sample_Table tab_;  // Its sample table.
    Query_Options opt_; // The options of the load.

public:

    // Loads the index-directory `dir` as `opt` places it. The index must be of
    // symbols of `T_sym_`, placed device-accessibly. Returns `false` with a
    // reason in `err` on failure.
    bool load(const std::string& dir, const Query_Options& opt, std::string& err);

    // Switches the loaded engine's search mode to the `stride`, `key_words`, and
    // `sort` settings of `Query_Options`, rebuilding the sample table; the
    // index stays loaded. Returns `false` with a reason in `err` on failure,
    // leaving the engine in the direct mode.
    bool configure(idx_t stride, uint32_t key_words, bool sort, std::string& err);

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

    // Returns the sample table; empty in the direct mode.
    const Sample_Table& table() const { return tab_; }

    // Returns the options the engine was loaded with.
    const Query_Options& options() const { return opt_; }
};

}



#endif
