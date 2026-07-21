
#ifndef READ_TASK_CUH
#define READ_TASK_CUH



#include "types.hpp"
#include "Staging_Slab.cuh"


namespace gpu_saca
{

// A unit of task to read a chunk of a partition's content.
struct Read_Task
{
    p_id_t p_id;    // ID of the partition to read from.
    idx_t chunk_idx;    // ID of the chunk in the partition.
    idx_t off;  // Offset of the chunk into the partition.
    idx_t sz;   // Size of the chunk to read.
};

}



#endif
