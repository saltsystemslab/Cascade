
#ifndef GPU_SACA_CUH
#define GPU_SACA_CUH



#include "types.hpp"
#include "Packed_Prefix.hpp"
#include "bit-layout.hpp"
#include "message.hpp"

#include "cub/cub.cuh"


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
// respectively, and marks the keys' activeness-flags into `is_active`. The
// first entry of a group is ranked by its offset into `K`, whereas the rest of
// the entries in the group are ranked `0`.
// Grid organization: one thread per key / suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void rank_key_groups_and_primary_key_groups_and_mark_active_flags_kernel(   const idx_t* const __restrict__ K, const idx_t* const __restrict__ SA, const idx_t n,
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
        const bool primary_key_tie = (bit_layout::primary_of(K[i]) == bit_layout::primary_of(K[i - 1]));
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
                                const idx_t* const __restrict__ key_group_rank, const idx_t* const __restrict__ primary_key_group_rank)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= n)
        return;

    K[i] = bit_layout::primary_of(K[i]) + (key_group_rank[i] - primary_key_group_rank[i]);
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
// generation's context-size `ctx`) and `msg_rank` (target's secondary key,
// computed from local rank `K`). Backward routing tag for each sender from `SA`
// is in the corresponding entry of `beta`. Messages are collated by destination
// partitions using the per- partition append-offsets in `msg_off`. `active_sz`
// delineates the active-retired set split of the cohort.
// Grid organization: one thread per suffix.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void emit_messages_kernel(  const idx_t* const __restrict__ K, const idx_t* const __restrict__ SA,
                            const p_id_t* const __restrict__ beta,
                            const idx_t n, const idx_t ctx, const idx_t* const __restrict__ active_sz,
                            idx_t* const __restrict__ msg_target, msg_rank_t* const __restrict__ msg_rank,
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
    msg_rank[slot] = (i < *active_sz ? K[i] : poison_pill(K[i]));  // `j`'s local rank.
}


// Locates each message's (target in `msg_tgt` and rank in `msg_rank`) recipient
// in the suffix-ordered active set `SA` (of size `n`). If found, fills the
// receipt-state of the recipient `recv_st`, and checks the message's payload
// against the rank-array `K` (supplied from global ISA).
// Grid organization: one thread per message.
// Shared memory layout: no usage.
template <idx_t BLOCK_SZ>
__global__
void join_messages_kernel(  const idx_t* const __restrict__ msg_tgt, const msg_rank_t* const __restrict__ msg_rank, const p_sz_t msg_c,
                            const idx_t* const __restrict__ SA, const idx_t* const __restrict__ K, const idx_t n,
                            const idx_t* const __restrict__ P_off, const p_id_t p,
                            uint8_t* const __restrict__ recv_st, idx_t* const __restrict__ mismatch_c)
{
    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    if(i >= msg_c)
        return;

    const auto tgt = bit_layout::suffix_of(msg_tgt[i]); // Recipient suffix of the message.
    idx_t low = 0, high = n;
    while(low < high)   // Locate `tgt` in the suffix-ordered active-set.
    {
        const auto mid = low + (high - low) / 2;
        const auto j = bit_layout::suffix_of(SA[mid]);
        if(j < tgt)
            low = mid + 1;
        else
            high = mid;
    }

    if(low == n || bit_layout::suffix_of(SA[low]) != tgt)   // Recipient has already retired.
        return;

    // Following are for temporary verification purposes.
    {
        recv_st[low] = (is_poison_pill(msg_rank[i]) ? 2 : 1);

        const auto secondary_global = bit_layout::secondary_of(K[low]);
        const p_id_t fwd_tag = bit_layout::tag_of(SA[low]); // Recipient's forward tag, filled in from the global ISA.
        p_id_t p_low = 0, p_high = p - 1;
        while(p_low < p_high)   // Locate the sender's global rank in the partitions' ranges.
        {
            const p_id_t mid = p_low + (p_high - p_low + 1) / 2;
            if(P_off[mid] <= secondary_global)
                p_low = mid;
            else
                p_high = mid - 1;
        }

        const auto exp_local_rank = secondary_global - P_off[p_low];  // Expected local rank of the sender in its partition.
        const p_id_t fwd_tag_rcvd = bit_layout::tag_of(msg_tgt[i]);   // Sender's forward tag from the preceding generation.
        if(exp_local_rank != extract_rank(msg_rank[i]) || fwd_tag_rcvd != fwd_tag)
            atomicAdd(reinterpret_cast<unsigned long long*>(mismatch_c), 1ull);
    }
}


// Counts from the receipt-states `recv_st` the number of suffixes: not having
// received any message into `unserved_c`, and having received poison-pills into
// `pill_recv_c`.
template <idx_t BLOCK_SZ>
__global__
void count_message_receipts_kernel(const uint8_t* const __restrict__ recv_st, const idx_t n, idx_t* const __restrict__ unserved_c, idx_t* const __restrict__ pill_recv_c)
{
    typedef cub::BlockReduce<idx_t, BLOCK_SZ> Block_Reduce;
    __shared__ typename Block_Reduce::TempStorage temp_unserved;
    __shared__ typename Block_Reduce::TempStorage temp_pill_recv;

    const idx_t i = static_cast<idx_t>(blockIdx.x) * BLOCK_SZ + threadIdx.x;
    const auto st = (i < n ? recv_st[i] : uint8_t(1));

    const auto block_unserved = Block_Reduce(temp_unserved).Sum(st == 0 ? idx_t(1) : idx_t(0));
    const auto block_pill_recv = Block_Reduce(temp_pill_recv).Sum(st == 2 ? idx_t(1) : idx_t(0));
    if(threadIdx.x == 0)
    {
        if(block_unserved > 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(unserved_c), static_cast<unsigned long long>(block_unserved));

        if(block_pill_recv > 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(pill_recv_c), static_cast<unsigned long long>(block_pill_recv));
    }
}

}



#endif
