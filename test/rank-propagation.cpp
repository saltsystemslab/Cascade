
#include "tests.hpp"
#include "bit-layout.hpp"


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

}
