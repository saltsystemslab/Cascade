
#ifndef GPU_SACA_CUH
#define GPU_SACA_CUH



#include "types.hpp"
#include "Packed_Prefix.hpp"
#include "bit-layout.hpp"


namespace gpu_saca
{

// Computes the partition-IDs of the suffixes of the text-chunk `T_chunk`, of
// type `T_sym_` and having `chunk_sz` many suffixes, against the `p - 1` pivots
// in `pivots`, into `P_id`. Also accumulates the per-partition suffix counts of
// this chunk into the global histogram `P_sz`.
// Grid organization: one thread per suffix of the chunk.
template <typename T_sym_, idx_t K, idx_t BLOCK_SZ>
__global__
void compute_partition_IDs_kernel(  const T_sym_* const __restrict__ T_chunk, const idx_t chunk_sz,
                                    const Pivot_Key* const __restrict__ pivots,
                                    p_id_t* const __restrict__ P_id, const p_id_t p, idx_t* const __restrict__ P_sz)
{
    constexpr auto TILE_SZ = BLOCK_SZ + K - 1;  // Size of the text-tile processed by a block.
    __shared__ T_sym_ tile[TILE_SZ];    // Text-tile.

    const auto t_id = threadIdx.x;
    const idx_t block_off = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ;  // Offset of this block into the text-chunk.

    // Load the corresponding text-tile cooperatively.
    {
        auto off = block_off + t_id;    // Offset into the text-chunk to load.
        tile[t_id] = off < chunk_sz ?
                        T_chunk[off] : std::numeric_limits<T_sym_>::min();

        if(BLOCK_SZ + t_id < TILE_SZ)   // Process the tile-segment of size `K - 1` overhanging from the block.
        {
            off = block_off + BLOCK_SZ + t_id;
            tile[BLOCK_SZ + t_id] = off < chunk_sz + K - 1 ?
                                        T_chunk[off] : std::numeric_limits<T_sym_>::min();
        }
    }

    __syncthreads();

    // Pack each suffix's `K`-prefix and compute its partition-ID, and count it.
    {
        const auto off = block_off + t_id;  // Offset into the text-chunk being processed.
        if(off < chunk_sz)
        {
            const auto pre_key = Pivot_Key::pack_prefix(tile + t_id);

            // Binary search for the smallest `p_id` such that `pre_key <= pivots[p_id]`.
            p_id_t low = 0, high = p - 1;   // The solution lies in the range `[low, high]`.
            while(low < high)
            {
                const auto mid = low + (high - low) / 2;
                if(pre_key <= pivots[mid])  // The solution lies in the range `[low, mid]`.
                    high = mid;
                else    // It lies in the range `[mid + 1, high]`.
                    low = mid + 1;
            }

            P_id[off] = low;
            atomicAdd(reinterpret_cast<unsigned long long*>(P_sz + low), 1ull);
        }
    }
}


// Fills `I[0, n)` with the identity permutation.
// Grid organization: one thread per index.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void iota_kernel(p_sz_t* const __restrict__ I, const idx_t n)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i < n)
        I[i] = i;
}


// Shuffles `src` into `dest` per the permutation `I`, all of size `n`.
// Grid organization: one thread per index.
// Shared memory layout: no usage.
template <typename T_, idx_t BLOCK_SZ>
__global__
void shuffle_kernel(T_* const __restrict__ dest, const T_* const __restrict__ src,
                    const p_sz_t* const __restrict__ I, const idx_t n)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i < n)
        dest[i] = src[I[i]];
}


// Initializes the ranks of the unique prefix-key groups from the sorted arrays
// `K` and `SA` (of size `n`) into `rank`, and marks the keys' activeness-flags
// into `is_active`. The first key of a group is ranked by its offset into `K`;
// the rest in the group are ranked `0`.
// Grid organization: one thread per key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void initialize_key_group_ranks_and_mark_active_flags_kernel(   const Prefix_Key* const __restrict__ K, const idx_t* const __restrict__ SA, const idx_t n,
                                                                idx_t* const __restrict__ rank,
                                                                uint8_t* const __restrict__ is_active)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    const auto tied = [&](const idx_t x, const idx_t y){ return K[x] == K[y] && bit_layout::tag_of(SA[x]) == bit_layout::tag_of(SA[y]); };

    rank[i] = (i == 0 ? 0 : (!tied(i, i - 1) ? i : 0));

    const bool pre_tie = (i == 0        ? false : tied(i, i - 1));
    const bool suc_tie = (i == n - 1    ? false : tied(i, i + 1));
    is_active[i] = (pre_tie || suc_tie);
}


// Ranks the distinct key groups and primary-key groups from the sorted arrays
// `K` and `SA` (of size `n`) into `key_group_rank` and `primary_key_group_rank`
// respectively, and marks the keys' activeness-flags into `is_active`. The low
// `secondary_bits` bits of a key are its secondary, and the rest are primary.
// The first entry of a group is ranked by its offset into `K`, whereas the rest
// of the entries in the group are ranked `0`.
// Grid organization: one thread per key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void rank_key_groups_and_primary_key_groups_and_mark_active_flags_kernel(   const idx_t* const __restrict__ K, const idx_t* const __restrict__ SA, const idx_t n,
                                                                            const idx_t secondary_bits,
                                                                            idx_t* const __restrict__ key_group_rank, idx_t* const __restrict__ primary_key_group_rank,
                                                                            uint8_t* const __restrict__ is_active)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    const auto tied = [&](const idx_t x, const idx_t y){ return K[x] == K[y] && bit_layout::tag_of(SA[x]) == bit_layout::tag_of(SA[y]); };

    if(i == 0)
    {
        // First group always starts at offset `0`.
        key_group_rank[0] = 0;
        primary_key_group_rank[0] = 0;
    }
    else
    {
        const bool key_tie = tied(i, i - 1);
        const bool primary_key_tie = ((K[i] >> secondary_bits) == (K[i - 1] >> secondary_bits));
        key_group_rank[i] = (!key_tie ? i : 0);
        primary_key_group_rank[i] = (!primary_key_tie ? i : 0);
    }

    const bool pre_tie = (i == 0        ? false : tied(i, i - 1));
    const bool suc_tie = (i == n - 1    ? false : tied(i, i + 1));
    is_active[i] = (pre_tie || suc_tie);
}


// Refines the primary-keys of the sorted array `K` (of size `n`) based on its
// key groups' ranks `key_group_rank` and its primary-key groups' ranks
// `primary_key_group_rank`, and writes back the refined *unpacked* primary-key.
// Grid organization: one thread per key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void refine_primary_keys_kernel(idx_t* const __restrict__ K, const idx_t n,
                                const idx_t secondary_bits,
                                const idx_t* const __restrict__ key_group_rank, const idx_t* const __restrict__ primary_key_group_rank)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    K[i] = (K[i] >> secondary_bits) + (key_group_rank[i] - primary_key_group_rank[i]);
}


// Counts the rank-propagation messages that each partition will receive at the
// next generation from the cohort `SA` (of size `n`) into `msg_count`. The
// current generation resolved context-size `ctx`. The backward routing tag for
// each sender from `SA` is in the corresponding entry in `beta`.
// Grid organization: one thread per suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void count_message_destination_distribution_kernel( const idx_t* const __restrict__ SA,
                                                    const p_id_t* const __restrict__ beta,
                                                    const idx_t n, const idx_t ctx,
                                                    p_sz_t* const __restrict__ msg_count)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    if(bit_layout::suffix_of(SA[i]) < ctx)  // No valid target recipient for next generation.
        return;

    atomicAdd(reinterpret_cast<unsigned int*>(msg_count) + beta[i], static_cast<unsigned int>(1));
}


// Emits the rank-propagation messages from cohort `SA` (of size `n`) into
// `msg_target` (target recipient, computed from the sender and the current
// generation's context-size `ctx`) and `msg_rank` (global rank, computed from
// local rank `K` and this partition's global offset `p_off` into the final SA).
// Backward routing tag for each sender from `SA` is in the corresponding entry
// in `beta`. Messages are collated by destination partitions using the per-
// partition append-offsets in `msg_off`.
// Grid organization: one thread per suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void emit_messages_kernel(  const idx_t* const __restrict__ K, const idx_t* const __restrict__ SA,
                            const p_id_t* const __restrict__ beta,
                            const idx_t n, const idx_t ctx, const idx_t p_off,
                            idx_t* const __restrict__ msg_target, idx_t* const __restrict__ msg_rank,
                            p_sz_t* const __restrict__ msg_off)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    const auto j = bit_layout::suffix_of((SA[i]));  // Suffix.
    if(j < ctx) // No valid target recipient for next generation.
        return;

    const auto slot = atomicAdd(reinterpret_cast<unsigned int*>(msg_off) + beta[i], static_cast<unsigned int>(1));  // Where to put the `beta[i]`'th partition's next incoming message.
    msg_target[slot] = bit_layout::pack(j - ctx, bit_layout::tag_of(SA[i]));   // Target recipient `j - ctx` will need `j`'s rank as its secondary and `j`'s forward-tag as its tertiary at next gen.
    msg_rank[slot] = p_off + K[i];  // `j`'s global rank.
}

}



#endif
