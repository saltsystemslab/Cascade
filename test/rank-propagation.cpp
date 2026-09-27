
#include "tests.hpp"
#include "bit-layout.hpp"
#include "GPU_SACA.hpp"
#include "Staging_Slab.hpp"
#include "bit-layout.hpp"
#include "message.hpp"

#include "omp.h"

#include <algorithm>


namespace gpu_saca::test
{

bool validate_coherent_targets(const idx_t j_1, const idx_t r_1, const idx_t j_2, const idx_t r_2, const idx_t n, const idx_t stride, const idx_t ell, const p_id_t* const P_id)
{
    const auto src_1 = bit_layout::suffix_of(j_1) + stride; // Secondary-source of `j_1`.
    const auto src_2 = bit_layout::suffix_of(j_2) + stride; // Secondary-source of `j_2`.
    return r_1 != r_2 || bit_layout::fwd_tag(P_id, src_1, n, ell) == bit_layout::fwd_tag(P_id, src_2, n, ell);
}


template <typename T_sym_>
bool validate_messages(const Staging_Slab& s, const GPU_SACA<T_sym_>* const self, const p_sz_t* const msg_off)
{
    const auto p = self->p_;
    const auto msg_count = msg_off[p - 1];
    if(s.outbox_off() >= msg_count) // This slab's outbox chunk lies past the messages the partition produced.
        return true;

    const auto n = self->n_;
    const auto ctx = self->ctx_;
    const auto* const P_id = self->P_id;
    const auto p_id = s.p_id();
    const auto P_sz_max = *std::max_element(self->P_cap.cbegin(), self->P_cap.cend());

    const auto* const msg_target = s.msg_target();
    const auto* const msg_rank = s.msg_rank();
    const auto msgs_to_check = std::min<p_sz_t>(s.outbox_sz(), msg_count - s.outbox_off());

    bool is_valid = true;
    #pragma omp parallel
    {
        #pragma omp for schedule(static) reduction(&& : is_valid)
        for(p_sz_t i = 0; i < msgs_to_check; ++i)
        {
            #pragma omp cancellation point for

            // In the following checks, for a poison-pill message, it is impossible
            // to detect whether the pill was a direct push or a forward.

            const auto target = bit_layout::suffix_of(msg_target[i]);

            // Sender check: the sender of this message must belong to this partition.
            {
                const auto direct_sender = target + ctx;
                const auto pill_forwarder = target + ctx / 2;
                if((direct_sender >= n && pill_forwarder >= n) || (direct_sender >= n && pill_forwarder < n && !is_poison_pill(msg_rank[i])))
                {
                    is_valid = false;
                    std::cerr << "Incorrect off-bound senders in messages.";
                    #pragma omp cancel for
                }

                const bool sender_in_place =    (direct_sender < n && P_id[direct_sender] == p_id) ||
                                                (is_poison_pill(msg_rank[i]) && pill_forwarder < n && P_id[pill_forwarder] == p_id);
                if(!sender_in_place)
                {
                    is_valid = false;
                    std::cerr << "For some messages produced in a partition, their senders do not belong to that partition.\n";
                    #pragma omp cancel for
                }
            }

            // Message collation check: this message must lie in the group of messages going to the target's partition.
            {
                const auto dest_part = P_id[target];
                const auto msg_slot = s.outbox_off() + i;
                const auto msg_group_beg = (dest_part > 0 ? msg_off[dest_part - 1] : 0);
                const auto msg_group_end = msg_off[dest_part];
                if(!(msg_group_beg <= msg_slot && msg_slot < msg_group_end))
                {
                    is_valid = false;
                    std::cerr << "Some messages did not get collated into their destination partition's group.\n";
                    #pragma omp cancel for
                }
            }

            // Rank check: the rank carried in this message must fall in the sender's partition's (i.e. this one's) range in the final SA.
            {
                const auto r = extract_rank(msg_rank[i]);
                const auto& P_cap = self->P_cap;
                const auto rank_bound = (is_poison_pill(msg_rank[i]) ? P_sz_max : P_cap[p_id]); // A forwarded pill carries the rank from another partition.
                if(r >= rank_bound)
                {
                    is_valid = false;
                    std::cerr << "Messages carry incorrect out-of-range ranks from senders.\n";
                    #pragma omp cancel for
                }
            }
        }
    }

    return is_valid;
}


bool validate_inbox_mass(const Partition_Backend& backend)
{
    const auto p = backend.p();

    std::vector<idx_t> inbox_sz(p, 0);
    for(p_id_t s = 0; s < p; ++s)   // Source partition.
    {
        const auto* const msg_off = backend.msg_off(s);
        for(p_id_t t = 0; t < p; ++t)   // Sink partition.
        {
            const auto p2q = (t == 0 ? msg_off[t] : msg_off[t] - msg_off[t - 1]);
            inbox_sz[t] += p2q;
        }
    }


    for(p_id_t t = 0; t < p; ++t)
    {
        const auto msg_count = backend.msg_count(t);
        const auto disk_count = backend.disk_count(t);
        const auto msg_written = backend.msg_written(t);

        if(inbox_sz[t] != msg_count)
        {
            std::cerr << "For partition " << t << ", its senders sent " << inbox_sz[t] << " messages, whereas it received " << backend.msg_count(t) << ".\n";
            return false;
        }

        if(msg_written % MSG_ALIGN != 0 || msg_written - disk_count >= MSG_ALIGN)
        {
            std::cerr << "For partition " << t << ", size of written inbox violates disk-block alignment constraints.";
            return false;
        }

        if(disk_count > 0 && (msg_count - disk_count) % MSG_ALIGN != 0)   // The disk-part's reads need an aligned RAM-part before it.
        {
            std::cerr << "For partition " << t << ", size of its inbox's RAM-part is misaligned against its disk-part.";
            return false;
        }
    }

    return true;
}


uint64_t message_checksum(const idx_t* const tgt, const msg_rank_t* const rank, const idx_t n)
{
    const auto mix = [](uint64_t z) // SplitMix64's finalizer, as the per-message hash.
    {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };

    uint64_t sum = 0;   // Wrapping sum of the messages' hashes: independent of their order.
    for(idx_t i = 0; i < n; ++i)
        sum += mix(mix(tgt[i]) + rank[i]);

    return sum;
}


bool validate_inbox_routing(const Staging_Slab& s, const p_id_t* const P_id, const idx_t n)
{
    bool is_valid = true;

    #pragma omp parallel for schedule(static) reduction(&& : is_valid)
    for(p_sz_t i = 0; i < s.inbox_sz(); ++i)
    {
        const auto tgt = bit_layout::suffix_of(s.msg_target()[i]);
        if(tgt >= n || P_id[tgt] != s.p_id())
            is_valid = false;
    }

    return is_valid;
}



// Template instantiations for the required instances.

template bool validate_messages(const Staging_Slab&, const GPU_SACA<uint8_t>* self, const p_sz_t*);
template bool validate_messages(const Staging_Slab&, const GPU_SACA<uint16_t>* self, const p_sz_t*);
template bool validate_messages(const Staging_Slab&, const GPU_SACA<uint32_t>* self, const p_sz_t*);


}
