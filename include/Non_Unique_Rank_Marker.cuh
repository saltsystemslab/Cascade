
#ifndef NON_UNIQUE_RANK_MARKER_CUH
#define NON_UNIQUE_RANK_MARKER_CUH



namespace gpu_saca
{

class Non_Unique_Rank_Marker
{
private:

    static constexpr idx_t MSB = (idx_t(1) << (sizeof(idx_t) * 8 - 1)); // Most significant bit of suffix index words.


public:

    __host__ __device__
    static idx_t mark_non_unique(const idx_t r) { return r | MSB; }

    __host__ __device__
    static idx_t mark_unique(const idx_t r) { return r & ~MSB; }

    __host__ __device__
    static bool is_non_unique(const idx_t r) { return r & MSB; }

    __host__ __device__
    static bool is_unique(const idx_t r) { return !is_non_unique(r); }

    __host__ __device__
    bool operator()(const Sort_Key& key){ return is_non_unique(key.primary); }

    __host__ __device__
    bool operator()(const idx_t& suf){ return is_non_unique(suf); }
};

}



#endif
