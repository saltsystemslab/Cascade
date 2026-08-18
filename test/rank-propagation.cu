
#include "tests.hpp"
#include "GPU_SACA.cuh"
#include "Staging_Slab.hpp"

#include <cstdint>


namespace gpu_saca::test
{

template <typename T_sym_>
bool validate_messages(const Partition_Assembly& p_assembly, const GPU_SACA<T_sym_>* const self)
{
    if(p_assembly.slab.empty())
        return true;

    const auto n = self->n_;
    const auto p = self->p_;
    const auto ctx = self->ctx_;
    const auto p_id = p_assembly.slab[0]->p_id();
    const auto* const P_id = self->P_id;

    const auto& msg_off = p_assembly.msg_off;
    const auto msg_count = msg_off[p - 1];
    p_sz_t msgs_checked = 0;
    for(const auto* s : p_assembly.slab)
    {
        const auto* const msg_target = s->msg_target();
        const auto* const msg_rank = s->msg_rank();

        const auto msgs_to_check = std::min<p_sz_t>(s->size(), msg_count - msgs_checked);
        for(p_sz_t i = 0; i < msgs_to_check; ++i)
        {
            const auto target = bit_layout::suffix_of(msg_target[i]);

            // Sender check: the sender of this message must belong to this partition.
            {
                const auto sender = target + ctx;
                if(sender >= n)
                {
                    std::cerr << "Incorrect off-bound senders in messages.";
                    return false;
                }

                if(P_id[sender] != p_id)
                {
                    std::cerr << "For some messages produced in a partition, their senders do not belong to that partition.\n";
                    return false;
                }
            }

            // Message collation check: this message must lie in the group of messages going to the target's partition.
            {
                const auto dest_part = P_id[target];
                const auto msg_slot = msgs_checked + i;
                const auto msg_group_beg = (dest_part > 0 ? msg_off[dest_part - 1] : 0);
                const auto msg_group_end = msg_off[dest_part];
                if(!(msg_group_beg <= msg_slot && msg_slot < msg_group_end))
                {
                    std::cerr << "Some messages did not get collated into their destination partition's group.\n";
                    return false;
                }
            }

            // Forward tag check: the `phi`-tag carried in this message must be correct.
            {
                const p_id_t phi = bit_layout::tag_of(msg_target[i]);
                const p_id_t expected_phi = (target + 2 * ctx < n ? P_id[target + 2 * ctx] : p_id_t(0));
                if(phi != expected_phi)
                {
                    std::cerr << "Some messages carry incorrect forward tag.\n";
                    return false;
                }
            }

            // Rank check: the rank carried in this message must fall in the sender's partition's (i.e. this one's) range in the final SA.
            {
                const auto r = msg_rank[i];
                const auto& P_off = self->P_off;
                const auto& P_cap = self->P_cap;
                if(r < P_off[p_id] || r >= P_off[p_id] + P_cap[p_id])   // Looser check, applicable when `rho` gets eliminated but messages still carry global ranks.
                {
                    std::cerr << "Messages carry incorrect out-of-range ranks from senders.\n";
                    return false;
                }
            }
        }

        msgs_checked += msgs_to_check;
    }


    return true;
}



// Template instantiations for the required instances.
template bool validate_messages(const Partition_Assembly&, const GPU_SACA<char>*);
template bool validate_messages(const Partition_Assembly&, const GPU_SACA<uint8_t>*);
template bool validate_messages(const Partition_Assembly&, const GPU_SACA<uint16_t>*);
template bool validate_messages(const Partition_Assembly&, const GPU_SACA<uint32_t>*);

}
