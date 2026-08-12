
#include "Partition_Backend.hpp"
#include "isa-instrument.hpp"


namespace gpu_saca
{

Partition_Backend::Partition_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const std::size_t fixed_bytes, idx_t* const rho, idx_t n, const idx_t secondary_bits):
      plan_(P_cap, fixed_bytes, sizeof(Prefix_Key))
    , ram_(P_cap, plan_)
    , disk_(P_cap, spill, plan_)
    , rho_(rho)
    , n_(n)
    , secondary_bits_(secondary_bits)
    , k_(0)
{}


void Partition_Backend::read_prefixes(Staging_Slab& s) const
{
    INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);

    read_prefix_chunk(s.p_id(), s.off(), s.size(), s.suf(), s.key());
}


void Partition_Backend::read_keys(Staging_Slab& s) const
{
    idx_t* const suf = s.suf();
    idx_t* const key = reinterpret_cast<idx_t*>(s.key());

    {
        INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);
        read_chunk(s.p_id(), s.off(), s.size(), suf, key);
    }

    // Gather the secondary keys of the read suffixes and fold those in to the keys.
    {
        INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().gather_ns);
#ifdef ISA_INSTRUMENT
        auto& stat = isa_instrument::ISA_Instrument().me();
        const auto gen = isa_instrument::ISA_Instrument().generation();
#endif

        for(idx_t i = 0; i < s.size(); ++i)
        {
            const auto j = suf[i];  // Suffix.
            const bool in_bound = (j + k_ < n_);
            const idx_t v = (in_bound ? rho_[j + k_] : 0);
            INSTRUMENT(stat.observe(gen, in_bound, v));
            key[i] = (key[i] << secondary_bits_) | INSTRUMENT_UNSTAMP(v);
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
