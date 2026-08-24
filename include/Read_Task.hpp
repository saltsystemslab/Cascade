
#ifndef READ_TASK_HPP
#define READ_TASK_HPP



#include "types.hpp"


namespace gpu_saca
{

// A unit of task to read a chunk of a partition's content and a chunk of its
// inbox.
struct Read_Task
{
    p_id_t p_id;    // ID of the partition to read from.
    idx_t chunk_idx;    // ID of the content-chunk in the partition.
    idx_t off;  // Offset of the content-chunk into the partition.
    idx_t sz;   // Size of the content-chunk to read.
    p_sz_t msg_off; // Offset of the message-chunk into the inbox.
    p_sz_t msg_sz;  // Size of the message-chunk to read.
};

}



#endif
