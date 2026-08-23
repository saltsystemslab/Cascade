
#include "tests.hpp"
#include "bit-layout.hpp"
#include "GPU_SACA.hpp"
#include "Staging_Slab.hpp"
#include "bit-layout.hpp"

#include "omp.h"

#include <cstdint>


namespace gpu_saca::test
{

bool validate_previous_gen_fwd_tag(const idx_t suf, const idx_t n, const idx_t stride, const p_id_t* const P_id)
{
    const auto j = bit_layout::suffix_of(suf);  // Suffix.
    const auto ctx_prev = stride;   // Last generation's context.
    const p_id_t phi_prev = (j + ctx_prev < n ? P_id[j + ctx_prev] : 0);
    return bit_layout::tag_of(suf) == phi_prev;
}


bool validate_target_coherence(const idx_t j_1, const idx_t r_1, const idx_t j_2, const idx_t r_2)
{
    return r_1 != r_2 || bit_layout::tag_of(j_1) == bit_layout::tag_of(j_2);
}


template <typename T_sym_>
bool validate_messages(const Staging_Slab& s, const GPU_SACA<T_sym_>* const self, const p_sz_t* const msg_off)
{
    const auto p = self->p_;
    const auto msg_count = msg_off[p - 1];
    if(s.off() >= msg_count)    // The partition produced fewer messages than the total capacity of the slabs across the assembly.
        return true;

    const auto n = self->n_;
    const auto ctx = self->ctx_;
    const auto* const P_id = self->P_id;
    const auto p_id = s.p_id();

    const auto* const msg_target = s.msg_target();
    const auto* const msg_rank = s.msg_rank();
    const auto msgs_to_check = std::min<idx_t>(s.size(), msg_count - s.off());

    bool is_valid = true;
    #pragma omp parallel
    {
        #pragma omp for schedule(static) reduction(&& : is_valid)
        for(p_sz_t i = 0; i < msgs_to_check; ++i)
        {
            #pragma omp cancellation point for

            const auto target = bit_layout::suffix_of(msg_target[i]);

            // Sender check: the sender of this message must belong to this partition.
            {
                const auto sender = target + ctx;
                if(sender >= n)
                {
                    is_valid = false;
                    std::cerr << "Incorrect off-bound senders in messages.";
                    #pragma omp cancel for
                }

                if(P_id[sender] != p_id)
                {
                    is_valid = false;
                    std::cerr << "For some messages produced in a partition, their senders do not belong to that partition.\n";
                    #pragma omp cancel for
                }
            }

            // Message collation check: this message must lie in the group of messages going to the target's partition.
            {
                const auto dest_part = P_id[target];
                const auto msg_slot = s.off() + i;
                const auto msg_group_beg = (dest_part > 0 ? msg_off[dest_part - 1] : 0);
                const auto msg_group_end = msg_off[dest_part];
                if(!(msg_group_beg <= msg_slot && msg_slot < msg_group_end))
                {
                    is_valid = false;
                    std::cerr << "Some messages did not get collated into their destination partition's group.\n";
                    #pragma omp cancel for
                }
            }

            // Forward tag check: the `phi`-tag carried in this message must be correct.
            // A sender that retired this generation can carry a stale forward tag: if its query for the
            // extension suffix has look-back depth `λ >= 2`, then the message it had received earlier had
            // a stale forward tag; this is benign per tertiary-irrelevance. Stale tags are impossible when
            // `λ < 2`, and this is guaranteed to hold for everyone up-to generation 1.
            {
                const p_id_t phi = bit_layout::tag_of(msg_target[i]);
                const p_id_t expected_phi = (target + 2 * ctx < n ? P_id[target + 2 * ctx] : p_id_t(0));
                if(ctx <= 2 * self->ell && phi != expected_phi)
                {
                    is_valid = false;
                    std::cerr << "Some messages carry incorrect forward tag.\n";
                    #pragma omp cancel for
                }
            }

            // Rank check: the rank carried in this message must fall in the sender's partition's (i.e. this one's) range in the final SA.
            {
                const auto r = msg_rank[i];
                const auto& P_cap = self->P_cap;
                if(r >= P_cap[p_id])
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



// Template instantiations for the required instances.

template bool validate_messages(const Staging_Slab&, const GPU_SACA<char>* self, const p_sz_t*);
template bool validate_messages(const Staging_Slab&, const GPU_SACA<uint8_t>* self, const p_sz_t*);
template bool validate_messages(const Staging_Slab&, const GPU_SACA<uint16_t>* self, const p_sz_t*);
template bool validate_messages(const Staging_Slab&, const GPU_SACA<uint32_t>* self, const p_sz_t*);


}
