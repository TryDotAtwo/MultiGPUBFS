#pragma once
#include <stdint.h>

// Single dispatcher, creating CUDA context only. Auxiliary streams must be
// distinct non-default streams. No allocation or payload ownership transfers.
extern "C" int mgbfs_batch_graph_create_v1(void** out);
extern "C" int mgbfs_batch_graph_begin_v1(void* handle, void* owner,
                                         void* generation, void* exchange);
extern "C" int mgbfs_batch_graph_submit_v1(void* handle, uint32_t batches);
extern "C" void mgbfs_batch_graph_cancel_v1(void* handle);
extern "C" int mgbfs_batch_graph_stats_v1(void* handle, uint64_t* launches,
    uint64_t* full_windows, uint64_t* batches, uint64_t* updates, uint64_t* rebuilds);
extern "C" void mgbfs_batch_graph_destroy_v1(void* handle);
