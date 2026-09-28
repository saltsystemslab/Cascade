
// Structural statistics of a text, from its suffix array and the STATS log of a `caps_sa_gpu` run over it:
//
//     sa_structure <text-file> <suffix-array-file> <run-log> <output-json>
//
// The text has `uint16_t` symbols and the suffix array `uint64_t` indices, both little-endian, as `caps_sa_gpu` reads
// and writes them. The log supplies the run's `ell`, `K` and partition-sizes, and with those its partition-ID map `Π`.
//
// With `c_t = ell 2^t` and `K <= ell`, the suffix `j` is active after generation `t` iff some other suffix `j'` has
// `lcp(j, j') >= c_t`, `j + c_t < n`, `j' + c_t < n` and `Π(j + c_t) = Π(j' + c_t)`: the forward tag of generation 0
// refines the repeat-depth's superlevel set by `Π`. Such rank-classes are contiguous in the suffix array, so each
// SA-adjacent pair links its two suffixes up to some generation, and a suffix retires one generation past the deeper
// of its two links. That predicts every active-set size, inbox-size and look-back of the run exactly.
//
// The output JSON has:
//  - the repeat-depth histogram (capped at `DEPTH_CAP`), and the maximum repeat-depth;
//  - the retirement-generation histogram, i.e. the active-set sizes;
//  - per generation: the payloads and the forwarded poison-pills emitted, i.e. the next inbox-size, and the number of
//    runs of consecutive text-positions in the active set;
//  - per generation, the histogram of its queries' look-backs: the generations since the queried target retired.
// Along the way, it verifies the suffix array: that it is a permutation, and that each adjacent pair is in order.

#include "types.hpp"
#include "utility.hpp"

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


namespace
{

using gpu_saca::idx_t;
using sym_t = uint16_t; // Symbol type of the texts, as `caps_sa_gpu` reads them.

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the word-wise lcp and the file formats assume little-endian");

constexpr idx_t DEPTH_CAP = idx_t(1) << 20; // Repeat-depths at least this are histogrammed together.
constexpr idx_t BLOCK_SZ = idx_t(1) << 20;  // SA-entries per work-unit of the SA-order pass.
constexpr uint8_t UNSET = 0xFF; // Retirement generation of a text-position the suffix array hasn't covered yet.


// Aborts the execution after printing `msg`.
[[noreturn]] void die(const std::string& msg)
{
    std::cerr << msg << " Aborting.\n";
    std::exit(EXIT_FAILURE);
}


// Configuration of a `caps_sa_gpu` run, as its STATS log reports it.
struct Run_Config
{
    idx_t ell = 0;  // Context-size resolved at generation 0.
    idx_t K = 0;    // Pivot-width, in symbols.
    std::vector<idx_t> P_cap;   // Partition-sizes, in partition order.
};


// Returns the configuration of the run with its STATS log at `path`.
Run_Config parse_run_log(const std::string& path)
{
    std::ifstream is(path);
    if(!is)
        die("Cannot open the run-log at " + path + ".");

    Run_Config c;
    bool pivots_line = false, sizes_line = false;
    for(std::string line; std::getline(is, line);)
        if(const auto pos = line.find("Pivots selection: "); pos != std::string::npos)
        {
            if(pivots_line || std::sscanf(line.c_str() + pos, "Pivots selection: %*[0-9] partitions from %*[0-9] samples of %" SCNu64 "-symbol prefixes;"
                                                                " initial sort-keys of %" SCNu64 " symbols.", &c.K, &c.ell) != 2)
                die("Unexpected pivots-selection line in the run-log at " + path + ".");
            pivots_line = true;
        }
        else if(line.rfind("Partition sizes: ", 0) == 0)
        {
            if(sizes_line)
                die("More than one partition-sizes line in the run-log at " + path + ".");
            std::istringstream ss(line.substr(std::strlen("Partition sizes: ")));
            for(idx_t sz; ss >> sz;)
                c.P_cap.push_back(sz);
            sizes_line = true;
        }

    if(!pivots_line || !sizes_line || c.P_cap.empty())
        die("The run-log at " + path + " lacks the pivots-selection and the partition-sizes lines of a STATS build.");

    return c;
}


// Reads the entries `[beg, end)` of the suffix array from the file-descriptor `fd` into `dst`.
void read_SA(const int fd, const idx_t beg, const idx_t end, idx_t* const dst)
{
    auto* const p = reinterpret_cast<char*>(dst);
    const std::size_t bytes = (end - beg) * sizeof(idx_t);
    for(std::size_t got = 0; got < bytes;)
    {
        const auto r = ::pread(fd, p + got, bytes - got, static_cast<off_t>(beg * sizeof(idx_t) + got));
        if(r <= 0)
            die("Could not read the suffix array's entries [" + std::to_string(beg) + ", " + std::to_string(end) + ").");
        got += r;
    }
}


// Returns the length of the longest common prefix of the suffixes `a` and `b` of the `n`-length text `T`.
idx_t lcp(const sym_t* const T, const idx_t a, const idx_t b, const idx_t n)
{
    const idx_t m = n - std::max(a, b); // Length of the shorter suffix.
    idx_t l = 0;
    for(; l + 4 <= m; l += 4)   // Four symbols a time.
    {
        uint64_t x, y;
        std::memcpy(&x, T + a + l, sizeof(x));
        std::memcpy(&y, T + b + l, sizeof(y));
        if(x != y)
            return l + __builtin_ctzll(x ^ y) / (8 * sizeof(sym_t));    // The first mismatch sits in the lowest bits.
    }

    while(l < m && T[a + l] == T[b + l])
        l++;

    return l;
}


// Partition-ID map `Π` of a run, from its partition-sizes and the suffix array. The run's pivot `q` is an inclusive
// upper bound of partition `q`'s `K`-prefixes, and equal `K`-prefixes are never split across partitions, so the
// `K`-prefix of partition `q`'s last suffix is an equivalent pivot for every suffix of the text.
class Partition_Map
{
private:

    const sym_t* const T_;  // The text, padded with at least `K` null symbols.
    const idx_t K_; // Pivot-width.
    const idx_t p_; // Partition-count.
    std::vector<sym_t> pivots_; // Pivot `q` spans `[q K, (q + 1) K)`.

    // Returns whether the `K`-prefix `x` is at most the `K`-prefix `y`.
    bool leq(const sym_t* const x, const sym_t* const y) const
    {
        for(idx_t k = 0; k < K_; ++k)
            if(x[k] != y[k])
                return x[k] < y[k];

        return true;
    }


public:

    // Reconstructs the map for the text `T` of length `n`, from the partition-sizes `P_cap` and the suffix array at
    // the file-descriptor `fd`.
    Partition_Map(const sym_t* const T, const idx_t n, const idx_t K, const std::vector<idx_t>& P_cap, const int fd):
          T_(T)
        , K_(K)
        , p_(P_cap.size())
    {
        idx_t off = 0;  // Start of the next partition in the suffix array.
        for(idx_t q = 0; q < p_; ++q)
        {
            if(P_cap[q] == 0)
                die("Partition " + std::to_string(q) + " of the run is empty.");
            off += P_cap[q];
            if(q + 1 == p_)
                break;

            idx_t last_first[2];    // The last suffix of partition `q`, and the first one of partition `q + 1`.
            read_SA(fd, off - 1, off + 1, last_first);
            if(last_first[0] >= n || last_first[1] >= n)
                die("The suffix array has an out-of-range entry at " + std::to_string(off - 1) + ".");
            if(leq(T + last_first[1], T + last_first[0]))
                die("The " + std::to_string(K) + "-prefix does not increase across partitions " + std::to_string(q) + " and " + std::to_string(q + 1) +
                    ": the run-log and the suffix array differ in their texts, or in their partitioning.");

            pivots_.insert(pivots_.end(), T + last_first[0], T + last_first[0] + K);
        }

        if(off != n)
            die("The partition-sizes of the run sum to " + std::to_string(off) + ", not to the text-length " + std::to_string(n) + ".");
    }


    // Returns the partition-ID of the suffix `x`: the smallest `q` with its pivot bounding the `K`-prefix at `x`, and
    // `p - 1` if none does.
    idx_t operator()(const idx_t x) const
    {
        idx_t low = 0, high = p_ - 1;   // The solution lies in `[low, high]`.
        while(low < high)
        {
            const auto mid = low + (high - low) / 2;
            if(leq(T_ + x, pivots_.data() + mid * K_))
                high = mid;
            else
                low = mid + 1;
        }

        return low;
    }
};


// Returns the JSON array of the elements of `C`.
template <typename T_>
std::string json_array(const std::vector<T_>& C)
{
    std::string s = "[";
    for(std::size_t i = 0; i < C.size(); ++i)
        s += (i > 0 ? ", " : "") + std::to_string(C[i]);

    return s + "]";
}

}



int main(int argc, char** argv)
{
    using namespace gpu_saca::timer;

    if(argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <text-file> <suffix-array-file> <run-log> <output-json>. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }

    const std::string text_path(argv[1]), sa_path(argv[2]), log_path(argv[3]), op_path(argv[4]);
    const auto run = parse_run_log(log_path);
    const idx_t ell = run.ell, K = run.K, p = run.P_cap.size();
    if(K == 0 || K > ell)
        die("Unsupported widths ell = " + std::to_string(ell) + " and K = " + std::to_string(K) + ": the exact rule needs 0 < K <= ell.");

    const auto t_0 = now();
    const idx_t n = gpu_saca::util::file_size<sym_t>(text_path);
    auto* const T = gpu_saca::util::checked_allocate<sym_t>(n + K); // Null-padded for the `K`-prefixes at the tail.
    gpu_saca::util::read_file(text_path, T, n);
    std::fill(T + n, T + n + K, sym_t(0));
    std::cerr << "Text-read completed: " << n << " symbols. Time taken: " << duration(now() - t_0) << "s.\n";

    if(gpu_saca::util::file_size<idx_t>(sa_path) != n)
        die("The suffix array at " + sa_path + " does not have " + std::to_string(n) + " entries.");
    const int fd = ::open(sa_path.c_str(), O_RDONLY);
    if(fd < 0)
        die("Cannot open the suffix array at " + sa_path + ".");

    const Partition_Map Pi(T, n, K, run.P_cap, fd);
    std::cerr << "Partition-ID map reconstructed: " << p << " partitions, with " << K << "-symbol pivots; ell = " << ell << ".\n";


    // Returns the deepest generation that links the SA-adjacent suffixes `a` and `b`, with an lcp of `l`; -1 if none.
    const auto link_level = [&](const idx_t a, const idx_t b, const idx_t l)
    {
        if(l < ell)
            return -1;

        const int t = 63 - __builtin_clzll(l / ell);    // Deepest generation with its context `c_t = ell 2^t` at most `l`.
        const idx_t c = ell << t;
        if(l >= c + K || (a + c < n && b + c < n && Pi(a + c) == Pi(b + c)))    // The first implies the second.
            return t;

        return t - 1;   // Linked at `t - 1` regardless, as `c_{t - 1} >= K` makes the `K`-prefixes at `c_{t - 1}` equal.
    };


    // Pass over the suffix array: link the adjacent pairs, and retire each suffix one generation past its deeper link.
    auto* const d = gpu_saca::util::checked_allocate<uint8_t>(n);   // Retirement generation of each text-position.
    std::vector<idx_t> depth_hist(DEPTH_CAP + 1, 0);    // Histogram of the repeat-depths.
    idx_t depth_max = 0;    // Maximum repeat-depth.
    int d_max = 0;  // Maximum retirement generation.
    {
        const auto t_1 = now();

        #pragma omp parallel for schedule(static)
        for(idx_t i = 0; i < n; ++i)
            d[i] = UNSET;

        const idx_t block_c = (n + BLOCK_SZ - 1) / BLOCK_SZ;
        std::atomic<idx_t> bad_pair(n); // Smallest SA-index of a pair that is out-of-range or out-of-order.

        #pragma omp parallel
        {
            std::vector<idx_t> SA(BLOCK_SZ + 2);    // A block's SA-entries, with a neighbour on each side.
            std::vector<idx_t> lcp_v(BLOCK_SZ + 1); // lcp of the pairs `(r - 1, r)` of the block's `r`, and of its successor.
            std::vector<int8_t> L_v(BLOCK_SZ + 1);  // Link-generation of those pairs.
            std::vector<idx_t> hist(DEPTH_CAP + 1, 0);
            idx_t dep_max = 0;
            int gen_max = 0;

            #pragma omp for schedule(dynamic, 1)
            for(idx_t blk = 0; blk < block_c; ++blk)
            {
                const idx_t r_0 = blk * BLOCK_SZ, r_1 = std::min(n, r_0 + BLOCK_SZ); // The block's SA-range.
                const idx_t lo = (r_0 > 0 ? r_0 - 1 : 0), hi = std::min(n, r_1 + 1);  // With its neighbours.
                read_SA(fd, lo, hi, SA.data());
                const auto sa = [&](const idx_t r) { return SA[r - lo]; };

                for(idx_t r = r_0; r <= r_1; ++r)   // The pair `(r - 1, r)`; none exists at `r = 0` and `r = n`.
                {
                    auto& l = lcp_v[r - r_0];
                    auto& L = L_v[r - r_0];
                    l = 0, L = -1;
                    if(r == 0 || r == n)
                        continue;

                    const idx_t a = sa(r - 1), b = sa(r);
                    if(a >= n || b >= n)
                    {
                        idx_t cur = bad_pair.load();
                        while(r < cur && !bad_pair.compare_exchange_weak(cur, r));
                        continue;
                    }

                    l = lcp(T, a, b, n);
                    if(!(a + l == n || (b + l < n && T[a + l] < T[b + l]))) // Suffix `a` must precede suffix `b`.
                    {
                        idx_t cur = bad_pair.load();
                        while(r < cur && !bad_pair.compare_exchange_weak(cur, r));
                    }

                    L = static_cast<int8_t>(link_level(a, b, l));
                }

                for(idx_t r = r_0; r < r_1; ++r)
                {
                    const idx_t j = sa(r);
                    if(j >= n)
                        continue;

                    const int d_j = 1 + std::max(L_v[r - r_0], L_v[r - r_0 + 1]);
                    const idx_t depth = std::max(lcp_v[r - r_0], lcp_v[r - r_0 + 1]);
                    d[j] = static_cast<uint8_t>(d_j);
                    hist[std::min(depth, DEPTH_CAP)]++;
                    dep_max = std::max(dep_max, depth);
                    gen_max = std::max(gen_max, d_j);
                }
            }

            #pragma omp critical
            {
                for(idx_t x = 0; x <= DEPTH_CAP; ++x)
                    depth_hist[x] += hist[x];
                depth_max = std::max(depth_max, dep_max);
                d_max = std::max(d_max, gen_max);
            }
        }

        if(bad_pair < n)
            die("The suffix array is out of range or out of order at the pair ending at SA-index " + std::to_string(bad_pair.load()) + ".");

        idx_t unset = 0;
        #pragma omp parallel for schedule(static) reduction(+ : unset)
        for(idx_t i = 0; i < n; ++i)
            unset += (d[i] == UNSET);
        if(unset > 0)
            die("The suffix array misses " + std::to_string(unset) + " text-positions: it is not a permutation.");

        std::cerr << "SA-order pass completed. Maximum repeat-depth: " << depth_max << "; retirement by generation " << d_max << "."
                  << " Time taken: " << duration(now() - t_1) << "s.\n";
    }

    ::close(fd);
    gpu_saca::util::deallocate(T);


    // Pass over the text: the retirements, the runs of the active sets, and the queries of the cohorts.
    const idx_t D = d_max;  // Generations `1..D` have non-empty cohorts.
    std::vector<idx_t> retire_hist(D + 1, 0);   // Suffixes by retirement generation.
    std::vector<idx_t> runs(D + 1, 0);  // Runs of consecutive text-positions in `A^(t)`.
    std::vector<idx_t> payloads(D + 1, 0), forwards(D + 1, 0);  // Messages emitted at generation `t`, for `t + 1`.
    std::vector<idx_t> lookback((D + 1) * (D + 1), 0);  // Queries of generation `t` with look-back `λ` at `t (D + 1) + λ`.
    {
        const auto t_2 = now();
        std::atomic<bool> target_past_end(false);

        #pragma omp parallel
        {
            std::vector<idx_t> retire(D + 1, 0), run(D + 1, 0), pay(D + 1, 0), fwd(D + 1, 0), lb((D + 1) * (D + 1), 0);

            #pragma omp for schedule(static)
            for(idx_t i = 0; i < n; ++i)
            {
                const int d_i = d[i];
                retire[d_i]++;
                for(int t = (i > 0 ? d[i - 1] : 0); t < d_i; ++t)  // `i` starts a run of `A^(t)`: it is in `A^(t)`, and `i - 1` is not.
                    run[t]++;

                for(int t = 1; t <= d_i; ++t)   // `i` is in the cohort `P^(t) = A^(t - 1)`, and queries its target `i + c_{t - 1}`.
                {
                    const idx_t m = i + (ell << (t - 1));
                    if(m >= n)  // An active suffix always has its target within the text.
                    {
                        target_past_end = true;
                        break;
                    }

                    const int lambda = (d[m] < t ? t - d[m] : 0);  // Generations since the target retired.
                    lb[t * (D + 1) + lambda]++;
                    pay[t] += (i >= (ell << t));  // The payload to `i - c_t`.
                    fwd[t] += (lambda >= 1 && i >= (ell << (t - 1)));  // The received pill, forwarded to `i - c_{t - 1}`.
                }
            }

            #pragma omp critical
            {
                for(idx_t t = 0; t <= D; ++t)
                    retire_hist[t] += retire[t], runs[t] += run[t], payloads[t] += pay[t], forwards[t] += fwd[t];
                for(std::size_t x = 0; x < lb.size(); ++x)
                    lookback[x] += lb[x];
            }
        }

        if(target_past_end)
            die("An active suffix has its target past the text's end, contradicting the exact rule.");

        payloads[0] = (n > ell ? n - ell : 0);  // Generation 0's cohort is the whole text, and it forwards no pills.
        std::cerr << "Text-order pass completed. Time taken: " << duration(now() - t_2) << "s.\n";
    }

    gpu_saca::util::deallocate(d);


    std::vector<idx_t> active(D + 1, 0);    // `|A^(t)| = #{j : d(j) > t}`.
    for(idx_t t = D; t-- > 0;)
        active[t] = active[t + 1] + retire_hist[t + 1];
    for(idx_t t = 0; t < D; ++t)
        std::cerr << "Generation " << t << ": active-set " << active[t] << "; inbox-size for generation " << (t + 1) << ": " << (payloads[t] + forwards[t]) << ".\n";


    std::ofstream os(op_path);
    os << "{\n";
    os << "  \"text\": \"" << text_path << "\",\n";
    os << "  \"sa\": \"" << sa_path << "\",\n";
    os << "  \"log\": \"" << log_path << "\",\n";
    os << "  \"n\": " << n << ",\n";
    os << "  \"ell\": " << ell << ",\n";
    os << "  \"K\": " << K << ",\n";
    os << "  \"p\": " << p << ",\n";
    os << "  \"depth_cap\": " << DEPTH_CAP << ",\n";
    os << "  \"depth_max\": " << depth_max << ",\n";
    os << "  \"depth_hist\": [";
    for(idx_t x = 0, first = 1; x <= DEPTH_CAP; ++x)
        if(depth_hist[x] > 0)
            os << (first ? "" : ", ") << "[" << x << ", " << depth_hist[x] << "]", first = 0;
    os << "],\n";
    os << "  \"retire_hist\": " << json_array(retire_hist) << ",\n";
    os << "  \"payloads\": " << json_array(payloads) << ",\n";
    os << "  \"forwards\": " << json_array(forwards) << ",\n";
    os << "  \"runs\": " << json_array(runs) << ",\n";
    os << "  \"lookback\": [";
    for(idx_t t = 0; t <= D; ++t)
        os << (t > 0 ? ", " : "") << json_array(std::vector<idx_t>(lookback.begin() + t * (D + 1), lookback.begin() + t * (D + 1) + t + 1));
    os << "]\n";
    os << "}\n";
    if(!os)
        die("Could not write the output at " + op_path + ".");

    std::cerr << "Written the structural statistics to " << op_path << ".\n";

    return 0;
}
