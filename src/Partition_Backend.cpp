
#include "Partition_Backend.hpp"
#include "bit-layout.hpp"
#include "isa-instrument.hpp"
#include "utility.cuh"
#include "tests.hpp"

#include <cstdlib>


namespace gpu_saca
{

Partition_Backend::Partition_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const std::size_t fixed_bytes, idx_t* const rho, idx_t n, const idx_t secondary_bits, const p_id_t* const P_id):
      p_(P_cap.size())
    , plan_(P_cap, fixed_bytes, sizeof(Prefix_Key))
    , ram_(P_cap, plan_)
    , disk_(P_cap, spill, plan_)
    , rho_(rho)
    , n_(n)
    , secondary_bits_(secondary_bits)
    , k_(0)
    , P_id(P_id)
{
    CUDA_CHECK(pinned_allocate_host(msg_off_, p_ * p_));
}


Partition_Backend::~Partition_Backend()
{
    CUDA_CHECK(deallocate_host(msg_off_));
}


void Partition_Backend::open_deferred_inboxes()
{
    disk_.open_deferred_inboxes();
}


void Partition_Backend::retire_prefix_and_open_key(const p_id_t p_id)
{
    disk_.retire_prefix_and_open_key(p_id);
}


void Partition_Backend::read_prefixes(Staging_Slab& s) const
{
    INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);

    read_prefix_chunk(s.p_id(), s.off(), s.size(), s.suf(), s.key());

    // Gather the backward routing tags of the read suffixes.
    {
        const idx_t* const suf = s.suf();
        p_id_t* const beta = s.beta();
        for(idx_t i = 0; i < s.size(); ++i)
        {
            const auto j = bit_layout::suffix_of(suf[i]);   // Suffix.
            beta[i] = (j >= ctx_ ? P_id[j - ctx_] : 0); // Backward tag.
        }
    }
}


void Partition_Backend::read_keys(Staging_Slab& s) const
{
    idx_t* const suf = s.suf();
    idx_t* const key = reinterpret_cast<idx_t*>(s.key());
    p_id_t* const beta = s.beta();

    {
        INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);
        read_chunk(s.p_id(), s.off(), s.size(), suf, key);
    }

    // Gather the secondary keys of the read suffixes and fold those in to the keys.
    {
        INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().gather_ns);

        for(idx_t i = 0; i < s.size(); ++i)
        {
#ifdef TESTS
            // This suffix survived generation `t - 1`. So it must have used the correct forward tag there.
            // Suffixes retired at generation `t - 1` could've had "incorrect" tags, but tertiary-irrelevance at their dying moments renders those stale tags unused.
            if(!test::validate_previous_gen_fwd_tag(suf[i], n_, k_, P_id))
            {
                std::cerr << "Incorrect forward tags used at last generation. Aborting.\n";
                std::exit(EXIT_FAILURE);
            }

            if(i > 0 && !test::validate_target_coherence(suf[i - 1], key[i - 1], suf[i], key[i]))
            {
                std::cerr << "Target coherence invariant broken. Aborting.\n";
                std::exit(EXIT_FAILURE);
            }
#endif

            const auto j = bit_layout::suffix_of(suf[i]);   // Suffix.
            const idx_t v = (j + k_ < n_ ? rho_[j + k_] : 0);   // Secondary rank, packed with previous generation's forward tag of the extension suffix.
            const auto phi = bit_layout::tag_of(v); // Forward tag of `j` for this generation.
            suf[i] = bit_layout::pack(j, phi);  // Writeback suffix to slab with its forward tag packed in.
            key[i] = (key[i] << secondary_bits_) | bit_layout::rank_of(v);
            beta[i] = (j >= ctx_ ? P_id[j - ctx_] : 0); // Backward tag.
        }
    }
}


void Partition_Backend::read_chunk(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const suf, idx_t* const key) const
{
    plan_.data(p_id) ?  (ram_.read_suf(p_id, off, sz, suf), ram_.read_key(p_id, off, sz, key)) :
                        (disk_.read_suf(p_id, off, sz, suf), disk_.read_key(p_id, off, sz, key));
}


void Partition_Backend::read_prefix_chunk(const p_id_t p_id, const idx_t off, const idx_t sz, idx_t* const suf, Prefix_Key* const key) const
{
    plan_.data(p_id) ?  ram_.read_suf(p_id, off, sz, suf) :
                        disk_.read_suf(p_id, off, sz, suf);

    plan_.prefix(p_id) ?    ram_.read_prefix(p_id, off, sz, key) :
                            disk_.read_prefix(p_id, off, sz, key);
}


void Partition_Backend::write(const Staging_Slab& s)
{
    const auto p_id = s.p_id();
    const idx_t* const key = reinterpret_cast<const idx_t*>(s.key());
    plan_.data(p_id) ?  ram_.write(p_id, s.off(), s.size(), s.suf(), key) :
                        disk_.write(p_id, s.off(), s.size(), s.suf(), key);
}


void Partition_Backend::drop_prefixes()
{
    ram_.drop_prefixes(), disk_.drop_prefixes();
}

}
