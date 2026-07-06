
#include "tests.hpp"

#include "omp.h"


namespace gpu_saca::test
{

bool validate_local_ranking(const Partition& part, const Sort_Key* const key, const idx_t* const suf)
{
    const auto sz = part.size();
    if(sz == 0)
        return true;


    const auto* const rank = part.key();
    if(rank[0].primary != 0)
        return false;   // TODO: this is incorrect when partition-compaction will be complete.

    bool valid = true;

    #pragma diag_suppress set_but_not_used
    #pragma omp parallel for schedule(static) reduction(&& : valid)
    for(std::size_t i = 1; i < sz; ++i)
    {
        // Sorted-ness check
        if( (key[i - 1].primary > key[i].primary) ||
            (key[i - 1].primary == key[i].primary && key[i - 1].secondary > key[i].secondary))
        {
            valid = false;
            continue;
        }

        // Ranking-check
        const bool eq_keys = (key[i].primary == key[i - 1].primary && key[i].secondary == key[i - 1].secondary);
        const auto exp_rank = eq_keys ? rank[i - 1].primary : i;    // TODO: this is incorrect when partition-compaction will be complete.
        if(rank[i].primary != exp_rank)
            valid = false;
    }
    #pragma diag_default set_but_not_used

    return valid;
}


bool validate_local_rankings(const std::vector<Partition>& P, const Sort_Key* const key, const idx_t* const suf)
{
    idx_t p_off = 0;
    for(p_id_t p_id = 0; p_id < P.size(); ++p_id)
    {
        if(!validate_local_ranking(P[p_id], key + p_off, suf + p_off))
            return false;

        p_off += P[p_id].size();
    }

    return true;
}

}
