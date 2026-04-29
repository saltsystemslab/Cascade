

#include "Suffix_Array.hpp"
#include "utility.hpp"

#include <cstdint>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>


namespace caps_sa
{

__host__
Suffix_Array::Suffix_Array(const char* const T, const idx_t n_):
      T_h(T)
    , n(n_ + 1)  // Account for the terminator symbol
    , chunk_count((n + chunk_sz - 1) / chunk_sz)
    , partition_count(chunk_count)  // TODO: number of final partitions is kept the same as the number of first-phase chunks now.
    , k(((c * std::log2(n) + 31 ) / 32) * 32)   // `c log n`
{
    // Copy over text to device
    const auto bytes = n * sizeof(char);
    cudaMalloc(reinterpret_cast<void**>(&T_d), bytes);
    cudaMemcpy(T_d, T_h, bytes, cudaMemcpyHostToDevice);

    // Allocate space for suffix array
    const auto pivot_slots = chunk_count * (partition_count - 1);   // Number of extra slots in the suffix array for additional placement of pivots.
    cudaMalloc(reinterpret_cast<void**>(&SA_d), (n + pivot_slots) * sizeof(idx_t));
}


__host__
Suffix_Array::~Suffix_Array()
{
    cudaFree(T_d), cudaFree(SA_d);
}


__host__
void Suffix_Array::build_SA(idx_t* const SA)
{
    // Get the sample suffix collection.
    sample_suffixes();
    std::cerr << "Sampled pivot candidates.\n";

    // Select the final pivots.
    select_pivots();
    std::cerr << "Selected the pivot set.\n";

    // Sort the chunks.
    sort_chunks(SA_d, chunk_sz);
    // sort_chunks_with_pivots(SA_d, chunk_sz;
    std::cerr << "Sorted the chunks.\n";

    // Locate the pivots at the sorted chunks.
    locate_pivots();
    std::cerr << "Located the pivots at the chunks.\n";

    // Route the subchunks to their final partitions.
    partition_sub_chunks();
    std::cerr << "Placed the subchunks at their final partitions.\n";

    // Copy back the chunked suffix array to host.
    cudaMemcpy(SA, SA_d, n * sizeof(idx_t), cudaMemcpyDeviceToHost);
}


__host__
void Suffix_Array::sort_chunks(idx_t* const SA_d, const idx_t chunk_sz) const
{
    for(std::size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
    {
        const auto chunk_begin = chunk_idx * chunk_sz;
        const auto chunk_len   = std::min(chunk_sz, n - chunk_begin);
        auto* const SA = SA_d + chunk_begin;

        // Populate the initial suffix array for this chunk.
        thrust::sequence(thrust::device, SA, SA + chunk_len, chunk_begin);

        // Sort each chunk successively.
        thrust::sort(thrust::device, SA, SA + chunk_len, Suffix_Comparator(T_d, n));
        cudaDeviceSynchronize();
    }
}


__host__
void Suffix_Array::sort_chunks_with_pivots(idx_t* const SA_d, const idx_t chunk_sz) const
{
    const auto pivot_count = partition_count - 1;

    for(std::size_t chunk_id = 0; chunk_id < chunk_count; ++chunk_id)
    {
        const auto chunk_begin = chunk_id * chunk_sz;   // At which suffix this chunk starts at.
        const auto chunk_end   = std::min(chunk_begin + chunk_sz, n);   // At which suffix this chunk ends at, exclusive.
        auto* const SA_op = SA_d + chunk_id * (chunk_sz + pivot_count); // Output location for this chunk, when sorted.
        const auto sz = chunk_end - chunk_begin;    // Size of this chunk.

        // Populate the chunk.
        thrust::tabulate(thrust::device, SA_op, SA_op + sz,
            [chunk_begin] __device__ (const auto i)
            {
                return chunk_begin + i;
            });
        cudaDeviceSynchronize();


        // Copy the pivots and mark them.
        thrust::tabulate(thrust::device, SA_op + sz, SA_op + sz + pivot_count,
            [P_d = this->P_d] __device__ (const idx_t i)
            {
                return P_d[i] | pivot_flag;
            });
        cudaDeviceSynchronize();


        thrust::sort(thrust::device, SA_op, SA_op + sz + pivot_count, Suffix_Comparator(T_d, n));
        cudaDeviceSynchronize();
    }
}


__host__
void Suffix_Array::sample_suffixes()
{
    const std::size_t k = ((c * std::log2(n) + 31 ) / 32) * 32;  // Sample per chunk: `c log n`

    cudaMalloc(reinterpret_cast<void**>(&S_d), k * chunk_count * sizeof(idx_t));

    sample_with_replacement<<<chunk_count, blk_sz>>>(chunk_sz, k, S_d, 0);
    cudaDeviceSynchronize();
}


__host__
void Suffix_Array::select_pivots()
{
    // Sort the collected samples.
    const auto sample_count = k * chunk_count;
    thrust::sort(thrust::device, S_d, S_d + sample_count, Suffix_Comparator(T_d, n));
    cudaDeviceSynchronize();

    std::vector<idx_t> S_h(sample_count);
    cudaMemcpy(S_h.data(), S_d, sample_count * sizeof(idx_t), cudaMemcpyDeviceToHost);

    const auto p = partition_count; // Number of final partitions.
    P_h = reinterpret_cast<idx_t*>(std::malloc((p - 1) * sizeof(idx_t)));

    const auto gap = sample_count / p;  // Distance-gap between pivots.
    const auto quant_err = sample_count % p;    // Numerator of the quantization error per partition.
    idx_t unaccounted = 0;  // Running total number of samples not properly accounted for due to quantization of partition sizes.
    std::size_t idx = 0;    // Index of the next pivot to choose from the samples.
    for(idx_t i = 0; i < p - 1; ++i)
    {
        idx += gap;
        unaccounted += quant_err;
        if(unaccounted >= p)    // Fix the cumulative error.
            idx++,
            unaccounted %= p;

        P_h[i] = S_h[idx - 1];
    }

    // Store a copy of the pivots at device.
    cudaMalloc(reinterpret_cast<void**>(&P_d), (p - 1) * sizeof(idx_t));
    cudaMemcpy(P_d, P_h, (p - 1) * sizeof(idx_t), cudaMemcpyHostToDevice);
}


__global__
void sample_from_chunk_with_replacement(const idx_t* const input, idx_t* const output, const idx_t chunk_sz, const idx_t k, const uint64_t seed)
{
    const auto chunk_id =  blockIdx.x;
    const auto thread_id = threadIdx.x;

    const uint64_t off = blockIdx.x * blockDim.x + threadIdx.x; // Unique sequence-offset per thread for random number generation.
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, off, 0, &rng);

    auto* const chunk_ip = input + chunk_id * chunk_sz;
    auto* const chunk_op = output + chunk_id * k;
    for(uint32_t i = thread_id; i < k; i += blockDim.x)  // A thread samples the `[t_id, t_id + dim.x, ...]`'th samples. A chunk can have a larger size than the block.
    {
        const auto sample_idx = curand(&rng) & (chunk_sz - 1);
        chunk_op[i] = chunk_ip[sample_idx];
    }
}


__global__
void sample_with_replacement(const idx_t chunk_sz, const idx_t k, idx_t* const output, const uint64_t seed)
{
    const auto chunk_id = blockIdx.x;
    const auto thread_id = threadIdx.x;

    const uint64_t off = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;  // Unique sequence-offset per thread for random number generation.
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, off, 0, &rng);

    assert((chunk_sz & (chunk_sz - 1)) == 0);   // `chunk_sz` must be a power-of-two.
    auto* const chunk_op = output + chunk_id * k;
    const auto shift = chunk_id * chunk_sz;
    for(uint32_t i = thread_id; i < k; i += blockDim.x) // A thread samples the `[t_id, t_id + dim.x, ...]`'th samples, since a chunk may have larger size than the thread-block.
        chunk_op[i] = (curand(&rng) & (chunk_sz - 1)) + shift;
}


__device__ __forceinline__
int sufcmp(const char* const T_d, const idx_t n, idx_t i, idx_t j)
{
    if(i == j)
        return 0;

    while(i < n && j < n && T_d[i] == T_d[j])
        i++, j++;

    // NB: Text must be terminated with some special character that is lexicographically the smallest.
    return T_d[i] < T_d[j] ? -1 : (T_d[i] > T_d[j] ? 1 : 0);
}


__device__ __forceinline__
idx_t upper_bound(const char* const T_d, const idx_t n, const idx_t* const S, const idx_t m, const idx_t p)
{
    idx_t low = 0, high = m;
    while(low < high)
    {
        const auto mid = low + (high - low) / 2;
        if(sufcmp(T_d, n, S[mid], p) <= 0)
            low = mid + 1;
        else
            high = mid;
    }

    return low;
}


__global__
void locate_pivots_kernel(const char* const T_d, const idx_t n, const idx_t* const SA_d, idx_t chunk_sz, const idx_t* const P, const idx_t p_c, idx_t* const R)
{
    const auto pivot_id = blockIdx.x * blockDim.x + threadIdx.x;   // Index of the pivot to locate.
    const auto chunk_idx = blockIdx.y;  // Which sorted chunk to locate the pivot into.
    if(pivot_id >= p_c)
        return;

    const auto chunk = SA_d + chunk_idx * chunk_sz; // The sorted chunk.
    const auto chunk_begin = chunk_idx * chunk_sz;
    const auto chunk_end = std::min(n, chunk_begin + chunk_sz);
    chunk_sz = chunk_end - chunk_begin;

    const auto rank_idx = chunk_idx * p_c + pivot_id;   // Output position for the search-result / rank.
    if(pivot_id == 0)
        R[rank_idx] = 0;
    else if(pivot_id == p_c - 1)
        R[rank_idx] = chunk_sz;
    else
        R[rank_idx] = upper_bound(T_d, n, chunk, chunk_sz, P[pivot_id - 1]);
}


void Suffix_Array::locate_pivots()
{
    const auto p_c = (partition_count - 1) + 2; // Number of pivots, including the two flanking sentinel pivots.
    cudaMalloc(reinterpret_cast<void**>(&P_rank), chunk_count * p_c * sizeof(idx_t));

    dim3 grid((p_c + blk_sz - 1) / blk_sz, chunk_count);    // 2D decomposition of the searches of all the pivots at all the chunks.
    locate_pivots_kernel<<<grid, blk_sz>>>(T_d, n, SA_d, chunk_sz, P_d, p_c, P_rank);
    cudaDeviceSynchronize();
}


__global__
void get_subchunk_sizes_kernel(const idx_t* const R, idx_t* const W, const idx_t p)
{
    const idx_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto subchunk_count = static_cast<uint64_t>(p) * p;
    if(thread_id >= subchunk_count)
        return;

    const auto i = thread_id / p;   // Chunk index.
    const auto j = thread_id % p;   // Subchunk index.
    const auto R_i = R + i * (p + 1);   // Delimiter array for the `i`'th chunk.
    const auto subchunk_sz = R_i[j + 1] - R_i[j];

    W[j * p + i] = subchunk_sz; // Stored in column-major order.
}


__global__
void get_partition_sizes_and_subchunk_indices_kernel(idx_t* const W, idx_t* const part_sz, const idx_t p)
{
    const auto j = blockIdx.x;  // Partition ID that this block will process.
    const auto thread_count = blockDim.x;

    extern __shared__ idx_t shared_mem[];   // Shared memory layout: local sum-sizes computed by the threads, followed by the column `W[][j]`.
    idx_t* local_sum = shared_mem;  // Local sum-sizes computed by the threads.
    idx_t* subchunk_sz = shared_mem + thread_count; // The subchunk sizes, i.e. the column `W[][j]`.

    const auto W_j_col = W + j * p; // The column `W[][j]`.


    // Load the `j`'th subchunk sizes column into shared memory.
    for(idx_t i = threadIdx.x; i < p; i += thread_count)
        subchunk_sz[i] = W_j_col[i];

    __syncthreads();


    // Local scan per thread to transform the subchunk sizes to thread-local exclusive scans.

    const auto stride = (p + thread_count - 1) / thread_count;  // Each thread processes `stride` consecutive sizes.
    const auto beg = threadIdx.x * stride;  // Starting offset for this thread.
    const auto end = std::min(beg + stride, p); // Ending (exclusive) offset for this thread.

    idx_t sum_sz = 0;
    for(idx_t i = beg; i < end; ++i)
    {
        const auto sz = subchunk_sz[i];
        subchunk_sz[i] = sum_sz;
        sum_sz += sz;
    }

    local_sum[threadIdx.x] = sum_sz;

    __syncthreads();


    // Kogge-Stone scan of local inclusive sum-sizes per thread to get global inclusive prefix-sum of those per-thread sums.

    for(idx_t s = 1; s < thread_count; s *= 2)
    {
        idx_t addend = 0;
        if(threadIdx.x >= s)
            addend = local_sum[threadIdx.x - s];

        __syncthreads();

        local_sum[threadIdx.x] += addend;
        __syncthreads();
    }

    // The size of the `j`'th partition is the last entry in the inclusive scan.
    if(threadIdx.x == thread_count - 1)
        part_sz[j] = local_sum[thread_count - 1];


    // Transform thread-local subchunk size exclusive scans to column-wide exclusive scan.

    const idx_t addend = (threadIdx.x > 0 ? local_sum[threadIdx.x - 1] : 0);
    for(idx_t idx = beg; idx < end; ++idx)
        subchunk_sz[idx] += addend;

    __syncthreads();

    // Copy back the column-wide subchunk size exclusive scan from shared memory to global.
    for(idx_t idx = threadIdx.x; idx < p; idx += thread_count)
        W_j_col[idx] = subchunk_sz[idx];

    __syncthreads();
}


void Suffix_Array::partition_sub_chunks()
{
    const auto p = (n + chunk_sz - 1) / chunk_sz;   // Number of partitions.
    idx_t* W;   // `W[i][j]` will denote where the `i`'th subchunk of the `j`'th partition will be stored at the final flattened array. It is also laid in column-major order.
    idx_t* part_sz; // Partition-sizes.

    // Get the  individual subchunk sizes.
    {
        cudaMalloc(reinterpret_cast<void**>(&W), p * p * sizeof(idx_t));
        get_subchunk_sizes_kernel<<<(p * p + blk_sz - 1) / blk_sz, blk_sz>>>(P_rank, W, p);
    }

    // Compute the partition sizes and transform the individual subchunk sizes to intra-partition subchunk indices.
    {
        cudaMalloc(reinterpret_cast<void**>(&part_sz), p * sizeof(idx_t));

        const auto thread_per_blk = std::min(p, idx_t(1024));
        const auto shared_mem_bytes = (thread_per_blk + p) * sizeof(idx_t);
        get_partition_sizes_and_subchunk_indices_kernel<<<p, thread_per_blk, shared_mem_bytes>>>(W, part_sz, p);
    }

    // Transform the partition sizes to exclusive scan, to get the index of each partition in the flattened SA.
    {
        std::size_t cub_tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr, cub_tmp_bytes,
            part_sz, part_sz,
            p
        );

        void* cub_tmp;
        cudaMalloc(&cub_tmp, cub_tmp_bytes);
        cub::DeviceScan::ExclusiveSum(
            cub_tmp, cub_tmp_bytes,
            part_sz, part_sz,
            p
        );
    }
}


__global__
void validate_kernel(const char* const T_d, const idx_t* const SA_d, bool* const is_sorted, const idx_t n)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        if((idx & (chunk_sz - 1)) != 0)
            is_sorted[idx] = Suffix_Comparator(T_d, n)(SA_d[idx - 1], SA_d[idx]);
        else
            is_sorted[idx] = true;
}


__host__
bool Suffix_Array::validate()
{
    bool* is_sorted_d;
    cudaMalloc(reinterpret_cast<void**>(&is_sorted_d), n * sizeof(bool));

    validate_kernel<<<(n + blk_sz - 1) / blk_sz, blk_sz>>>(T_d, SA_d, is_sorted_d, n);
    cudaDeviceSynchronize();

    bool* is_sorted_h = reinterpret_cast<bool*>(std::malloc(n * sizeof(bool)));
    cudaMemcpy(is_sorted_h, is_sorted_d, n * sizeof(bool), cudaMemcpyDeviceToHost);

    bool r = true;
    for(idx_t i = 0; i < n; ++i)
        r = r && is_sorted_h[i];

    cudaFree(is_sorted_d);
    std::free(is_sorted_h);

    return r;
}

}

