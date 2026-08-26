
#include "Partition_Backend.hpp"
#include "bit-layout.hpp"
#include "utility.hpp"
#include "utility.cuh"
#include "isa-instrument.hpp"
#include "tests.hpp"

#include <cstring>
#include <mutex>
#include <cstdlib>
#include <algorithm>


namespace gpu_saca
{

Partition_Backend::Partition_Backend(const std::vector<idx_t>& P_cap, Spill_Store& spill, const std::size_t fixed_bytes, idx_t* const rho, idx_t n, const p_id_t* const P_id):
      p_(P_cap.size())
    , plan_(P_cap, fixed_bytes, sizeof(Prefix_Key))
    , ram_(P_cap, plan_)
    , disk_(P_cap, spill, plan_)
    , msg_off_(nullptr)
    , inbox_write_parity_(write_parity(0))
    , inbox_read_parity_(read_parity(0))
    , ov_tgt_(p_, nullptr)
    , ov_rank_(p_, nullptr)
    , ov_sz_(p_, 0)
    , ov_lock_(p_)
    , rho_(rho)
    , n_(n)
    , k_(0)
    , P_id(P_id)
{
    CUDA_CHECK(pinned_allocate_host(msg_off_, p_ * p_));

    for(idx_t parity = 0; parity < 2; ++parity)
        msg_count_[parity] = std::vector<std::atomic<p_sz_t>>(p_),
        msg_written_[parity] = std::vector<std::atomic<p_sz_t>>(p_);

    for(idx_t p_id = 0; p_id < p_; ++p_id)
        ov_tgt_[p_id] = util::aligned_allocate<idx_t>(OV_CAP, Direct_File::DISK_BLOCK),
        ov_rank_[p_id] = util::aligned_allocate<msg_rank_t>(OV_CAP, Direct_File::DISK_BLOCK);
}


Partition_Backend::~Partition_Backend()
{
    CUDA_CHECK(deallocate_host(msg_off_));
    for(idx_t p_id = 0; p_id < p_; ++p_id)
        util::deallocate(ov_tgt_[p_id]),
        util::deallocate(ov_rank_[p_id]);
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
        auto* const beta = s.beta();
        for(idx_t i = 0; i < s.size(); ++i)
        {
            const auto j = bit_layout::suffix_of(suf[i]);   // Suffix.
            beta[i] = bit_layout::beta_pair(j >= ctx_ ? P_id[j - ctx_] : 0, 0); // Backward tag pair.
        }
    }
}


void Partition_Backend::read_keys(Staging_Slab& s, const std::vector<idx_t>& P_off) const
{
    idx_t* const suf = s.suf();
    idx_t* const key = reinterpret_cast<idx_t*>(s.key());
    auto* const beta = s.beta();

    {
        INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);
        read_chunk(s.p_id(), s.off(), s.size(), suf, key);
    }

    // Gather the secondary keys of the read suffixes and fold those in to the keys.
    {
        INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().gather_ns);

#ifdef TESTS
        idx_t prev_suf = 0, prev_key = 0;
#endif

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

            if(i > 0 && !test::validate_coherent_tags(prev_suf, prev_key, suf[i], key[i]))
            {
                std::cerr << "Target coherence invariant broken. Aborting.\n";
                std::exit(EXIT_FAILURE);
            }

            prev_suf = suf[i], prev_key = key[i];
#endif

            const auto j = bit_layout::suffix_of(suf[i]);   // Suffix.
            const idx_t v = (j + k_ < n_ ? rho_[j + k_] : 0);   // Secondary rank, packed with previous generation's forward tag of the extension suffix.
            const auto phi = bit_layout::tag_of(v); // Forward tag of `j` for this generation.
            const auto r_global = bit_layout::rank_of(v);   // Secondary value in global rank.
            p_id_t low = 0, high = p_ - 1;
            while(low < high)
            {
                const auto mid = low + (high - low + 1) / 2;
                if(P_off[mid] <= r_global)
                    low = mid;
                else
                    high = mid - 1;
            }

            const auto r_local = r_global - P_off[low]; // Secondary value in local rank.

            suf[i] = bit_layout::pack(j, phi);  // Writeback suffix to slab with its forward tag packed in.
            key[i] = bit_layout::key(key[i], r_local);
            beta[i] = bit_layout::beta_pair(j >= ctx_ ? P_id[j - ctx_] : 0, j >= k_ ? P_id[j - k_] : 0);    // Backward tag pair.
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


void Partition_Backend::read_messages(Staging_Slab& s) const
{
    INSTRUMENT_TIMER(isa_instrument::ISA_Instrument().me().read_ns);

    disk_.read_messages(s.p_id(), s.msg_off(), s.msg_sz(), s.msg_target(), s.msg_rank());
}


void Partition_Backend::write(const Staging_Slab& s)
{
    const auto p_id = s.p_id();
    const idx_t* const key = reinterpret_cast<const idx_t*>(s.key());
    plan_.data(p_id) ?  ram_.write(p_id, s.off(), s.size(), s.suf(), key) :
                        disk_.write(p_id, s.off(), s.size(), s.suf(), key);
}


void Partition_Backend::write_messages(const Staging_Slab& s)
{
    const auto p_id = s.p_id();
    const auto* const msg_off = this->msg_off(p_id);

    // This slab contains the slice `[l_slab, r_slab)` of messages from `p_id`.
    const p_sz_t l_slab = s.off(), r_slab = std::min<idx_t>(s.off() + s.size(), msg_off[p_ - 1]);
    if(l_slab >= r_slab)
        return;

    for(p_id_t q = 0; q < p_; ++q)
    {
        // `q` is the destination for the slice `[l_q, r_q)` of messages from `p_id`.
        const auto l_q = (q > 0 ? msg_off[q - 1] : 0), r_q = msg_off[q];
        const auto l = std::max(l_slab, l_q), r = std::min(r_slab, r_q); // Intersect the slab's slice with the `q`-destined slice.
        if(l < r)
            append_messages(q, s, l, r);
    }
}


void Partition_Backend::append_messages(const p_id_t p_id, const Staging_Slab& s, const p_sz_t l, const p_sz_t r)
{
    const auto parity = inbox_write_parity_;
    msg_count_[parity][p_id].fetch_add(r - l, std::memory_order_relaxed);

    const auto base = s.off();  // `s` contains the index-range `[base, base + s.size())` of the source-partition's data.
    const p_sz_t a_l = ((l + MSG_ALIGN - 1) / MSG_ALIGN) * MSG_ALIGN;   // Leftmost `MSG_ALIGN`-ed endpoint of `[l, r)`.
    const p_sz_t a_r = (r / MSG_ALIGN) * MSG_ALIGN; // Rightmost `MSG_ALIGN`-ed (exclusive) endpoint of `[l, r)`.

    if(a_l < a_r)
    {
        const auto to_write = a_r - a_l;
        const auto write_off = msg_written_[parity][p_id].fetch_add(to_write, std::memory_order_relaxed);
        disk_.msg_tgt_tag(p_id).write_at(s.msg_target() + (a_l - base), to_write * sizeof(idx_t), write_off * sizeof(idx_t));
        disk_.msg_rank(p_id).write_at(s.msg_rank() + (a_l - base), to_write * sizeof(msg_rank_t), write_off * sizeof(msg_rank_t));
    }
    else    // Degenerate case of too short message-block.
    {
        buffer_overhang(p_id, s, l, r);
        return;
    }

    if(l < a_l) // Buffer the overhang at the head of `[l, r)`.
        buffer_overhang(p_id, s, l, a_l);

    if(a_r < r) // Buffer the overhang at the tail of `[l, r)`.
        buffer_overhang(p_id, s, a_r, r);
}


void Partition_Backend::buffer_overhang(const p_id_t p_id, const Staging_Slab& s, const p_sz_t l, const p_sz_t r)
{
    const auto base = s.off();  // `s` contains the index-range `[base, base + s.size())` of the source-partition's data.
    const std::lock_guard<util::Spin_Lock> lock(ov_lock_[p_id]);    // Lock the inbox's block-overflow buffer.

    std::memcpy(ov_tgt_[p_id] + ov_sz_[p_id], s.msg_target() + (l - base), (r - l) * sizeof(idx_t));
    std::memcpy(ov_rank_[p_id] + ov_sz_[p_id], s.msg_rank() + (l - base), (r - l) * sizeof(msg_rank_t));

    ov_sz_[p_id] += (r - l);
    if(ov_sz_[p_id] + 2 * MSG_ALIGN > OV_CAP)   // Whether the next buffer-addition may overflow or not.
        flush_overflow(p_id, false);
}


void Partition_Backend::flush_overflow(const p_id_t p_id, const bool final)
{
    const auto to_write = (!final ?
                            (ov_sz_[p_id] / MSG_ALIGN) * MSG_ALIGN :
                            ((ov_sz_[p_id] + MSG_ALIGN - 1) / MSG_ALIGN) * MSG_ALIGN);
    if(to_write == 0)
        return;

    const auto parity = inbox_write_parity_;
    const auto write_off = msg_written_[parity][p_id].fetch_add(to_write, std::memory_order_relaxed);
    disk_.msg_tgt_tag(p_id).write_at(ov_tgt_[p_id], to_write * sizeof(idx_t), write_off * sizeof(idx_t));
    disk_.msg_rank(p_id).write_at(ov_rank_[p_id], to_write * sizeof(msg_rank_t), write_off * sizeof(msg_rank_t));

    const auto rem = (to_write < ov_sz_[p_id] ? ov_sz_[p_id] - to_write : 0);
    if(rem > 0)
    {
        std::memmove(ov_tgt_[p_id], ov_tgt_[p_id] + to_write, rem * sizeof(idx_t));
        std::memmove(ov_rank_[p_id], ov_rank_[p_id] + to_write, rem * sizeof(msg_rank_t));
    }

    ov_sz_[p_id] = rem;
}


void Partition_Backend::open_generation(const idx_t gen)
{
    inbox_write_parity_ = write_parity(gen), inbox_read_parity_ = read_parity(gen);
    disk_.set_generation(gen);

    const auto parity = write_parity(gen);
    std::memset(msg_off_, 0, p_ * p_* sizeof(p_sz_t));
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
    {
        msg_count_[parity][p_id] = 0, msg_written_[parity][p_id] = 0;
        ov_sz_[p_id] = 0;
    }
}


void Partition_Backend::close_generation()
{
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
    {
        const std::lock_guard<util::Spin_Lock> lock(ov_lock_[p_id]);
        flush_overflow(p_id, true);
    }
}


p_sz_t Partition_Backend::inbox_size(const p_id_t p_id) const
{
    return msg_count_[inbox_read_parity_][p_id];
}


idx_t Partition_Backend::inbox_mass_last_gen() const
{
    const auto parity = inbox_write_parity_;
    idx_t sz = 0;
    for(p_id_t p_id = 0; p_id < p_; ++p_id)
        sz += msg_count_[parity][p_id];

    return sz;
}


void Partition_Backend::drop_prefixes()
{
    ram_.drop_prefixes(), disk_.drop_prefixes();
}

}
