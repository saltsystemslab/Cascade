
#ifndef TESTS_HPP
#define TESTS_HPP



#include "types.hpp"
#include "Packed_Prefix.hpp"
#include "Partition_Backend.hpp"
#include "Staging_Slab.hpp"
#include "bit-layout.hpp"
#include "cross-compile.hpp"

#include <vector>


// Forward declarations; specifically for constructs from CUDA-headers.

namespace gpu_saca
{
    template <typename T_sym_> class GPU_SACA;
}


namespace gpu_saca::test
{

// Returns `true` iff the partition-IDs `P_id` of the text `T` (of size `n`),
// defined by the pivots `X`, match to the partition-IDs computed at the CPU-
// side as reference.
template <typename T_sym_>
bool validate_partition_IDs(const T_sym_* T, idx_t n, const std::vector<Pivot_Key>& X, const p_id_t* P_id);

/*
// Returns `true` iff the keys in `key` and their corresponding suffixes in
// `suf`, corresponding to the partition `part`, are sorted, and `part` is
// properly ranked based on the sorted keys.
bool validate_local_ranking(const Partition& part, const Sort_Key* key, const idx_t* suf);

// Returns `true` iff all the partitions in `P` are properly sorted-and-ranked,
// with their keys packed flat at `key` and the corresponding suffixes packed
// flat at `suf`, after the device-side sort.
bool validate_local_rankings(const std::vector<Partition>& P, const Sort_Key* key, const idx_t* suf);
*/

// Returns the symbol `tau` such that roughly half of the symbol occurrences in
// the text `T`, of size `n`, fall into `[min, tau]` and the rest into
// `(tau, max]`.
template <typename T_sym_>
T_sym_ universe_split_threshold(const T_sym_* T, idx_t n);

// Returns `true` iff for the text `T` (of size `n`), the initial partition
// content defined by the spill-store `spill` (of files of the partitions with
// sizes `P_sz`) is correct: for each suffix `s` in the `p`'th partition, checks
// (1) whether it landed in the partition expected by `P_id`, ie `P_id[s] == p`;
// (2) that it is not duplicated anywhere; and (3) that its associated prefix
// matches to the actual prefix extracted from `T`.
template <typename T_sym_>
bool validate_partition_contents(const T_sym_* T, idx_t n, const p_id_t* P_id, const std::vector<idx_t>& P_sz, const Partition_Backend& backend);

// Returns `true` iff the suffix `suf` packed with its previous generation's
// forward-tag correctly matches to the ground truth available from partition-ID
// map `P_id`. There are `n` total suffixes and current generation's stride is
// `stride`.
bool validate_previous_gen_fwd_tag(idx_t suf, idx_t n, idx_t stride, const p_id_t* P_id);

// Returns `true` iff the target coherence invariant holds for the suffixes
// `j_1` and `j_2` with respective local ranks `r_1` and `r_2`. The suffixes
// must be from the same partition.
bool validate_coherent_tags(idx_t j_1, idx_t r_1, idx_t j_2, idx_t r_2);

// Returns `true` iff `SA` is a permutation of `[0, n)`.
bool is_permutation(const idx_t* SA, idx_t n);

// Returns `true` iff `SA` is the suffix array of the text `T` of size `n`.
template <typename T_sym_>
bool is_correct_SA(const idx_t* SA, const T_sym_* T, idx_t n);

// Returns `true` iff the messages present in the staging slab `s` are
// consistent according to the driver class `self`. `msg_off` contains the
// exclusive-end offsets (across the whole collection of messages produced from
// the corresponding partition) of groups of messages, collated by destination
// partitions.
template <typename T_sym_>
bool validate_messages(const Staging_Slab& s, const GPU_SACA<T_sym_>* self, const p_sz_t* msg_off);

// Returns `true` iff the sender-recipient message volumes agree at `backend`.
bool validate_inbox_mass(const Partition_Backend& backend);

// Returns `true` iff all the messages in the slab `s` are addressed to `s`'s
// corresponding partition, i.e. their destination partition-IDs, computed from
// `P_id`, equal to the slab's partition-ID. There are `n` possible recipient
// suffixes.
bool validate_inbox_routing(const Staging_Slab& s, const p_id_t* P_id, idx_t n);

// Returns `true` if the target coherence invariant holds for the packed keys
// (in device) `key_1` and `key_2`. The corresponding suffixes must be from the
// same partition.
CUDA_SPECIFIER
inline bool validate_coherent_pill_receipts(const idx_t key_1, const idx_t key_2)
{
    return  bit_layout::primary_of(key_1) != bit_layout::primary_of(key_2) ||
            bit_layout::secondary_of(key_1) != bit_layout::secondary_of(key_2) ||
            bit_layout::received_poison_pill(key_1) == bit_layout::received_poison_pill(key_2);
}

}



#endif
