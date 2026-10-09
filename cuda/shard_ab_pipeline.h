#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Dense-state pipeline with persistent per-shard dedup tables.
 * S logical shards share K independent worker streams. Each shard belongs to
 * one worker for the whole layer. New routed portions enter preallocated A/B
 * input buffers; tables and accepted payload survive all portions of a layer.
 * No accumulated-buffer re-dedup or final key sort is performed.
 * History keys may be unordered: fixed-capacity hash indices are built at the
 * layer boundary, and the current index is rotated to previous when possible.
 * begin/push/finalize allocate nothing and read no device counts to the host.
 * Input keys/states have identical owner-sorted row order; device begin/rows
 * select one owner window. Input may be reused after producer-side copy;
 * workers retain their A/B staging leases until completion events.
 * Finalization seals shard-strided payload and calculates prefix offsets.
 * publish writes directly to the reserved live-ring extent; no compaction arena.
 * Finalize output is unordered, shard-strided (capacity rows per shard), and
 * aliases persistent storage until next begin; it is NOT a contiguous prefix.
 * Caller drains producer and all output readers before begin/destroy.
 */
/* Indexed live-ring path: no accepted state arena. Bind before begin. */
int mgbfs_shard_ab_indexed_query(uint32_t,uint32_t,uint32_t,uint32_t,uint64_t*);
int mgbfs_shard_ab_indexed_create(uint32_t,uint32_t,uint32_t,uint32_t,void*,void**);
int mgbfs_shard_ab_indexed_bind(void*,void*,void*,void*,uint8_t*,uint32_t*,uint32_t,uint32_t*,void*);
int mgbfs_shard_ab_pipeline_query(uint32_t maximum_shards,uint32_t capacity,
    uint32_t stride,uint32_t slots,uint64_t* bytes);
int mgbfs_shard_ab_pipeline_create(uint32_t maximum_shards,uint32_t capacity,
    uint32_t stride,uint32_t slots,void* producer_stream,void** output);
int mgbfs_shard_ab_pipeline_begin(void* pipeline,uint32_t shards,
    uint32_t logical_owner,uint32_t world,const void* previous,uint32_t pn,
    const void* current,uint32_t cn,uint32_t active_job_bound,uint32_t* fatal);
int mgbfs_shard_ab_pipeline_push(void* pipeline,const void* keys,
    const uint8_t* states,const uint32_t* begin,const uint32_t* rows,
    const uint32_t* source_rows,uint32_t input_capacity);
/* Deferred indexed selection: no state input is read. Device views alias the
 * pending lease; caller must complete materialize before push/finalize/begin.
 * Queries enqueue no work. Consumer streams must wait on producer completion. */
int mgbfs_shard_ab_pipeline_select(void*,const void*,const uint32_t*,const uint32_t*,const uint32_t*,uint32_t);
int mgbfs_shard_ab_pipeline_selection(void*,const uint32_t**,const uint32_t**,const uint32_t**,const uint32_t**,const uint32_t**);
int mgbfs_shard_ab_pipeline_materialize(void*,const uint8_t*,const uint32_t*,uint32_t);
/* Compact selected origins, then apply responses in the returned request order.
 * Output capacity covers the input job bound. The device count is checked before
 * any response write. All calls use producer stream; no host count readback. */
int mgbfs_shard_ab_pipeline_row_requests(void*,uint32_t*,uint32_t*,uint32_t);
int mgbfs_shard_ab_pipeline_requests(void*,const void*,void*,uint32_t*,uint32_t);
int mgbfs_shard_ab_pipeline_apply_responses(void*,const uint8_t*,const uint32_t*);
int mgbfs_shard_ab_pipeline_finalize(void* pipeline,void** keys,
    uint8_t** states,const uint32_t** count);
/* Call after finalize on the same producer stream. Capacity/reservation gates
 * precede all live-ring writes. Output hashes are unordered AoS; history hash indices handle next depth.
 * Caller validates sticky fatal before exposing the layer. */
int mgbfs_shard_ab_pipeline_prepare_publish(void* pipeline,void* ring,void* owner,void* extent,uint32_t* layer_count,uint32_t layer_capacity);
/* publish is valid only after prepare and all-rank admission succeeded. */
int mgbfs_shard_ab_pipeline_publish(void* pipeline,void* ring,void* owner,
    void* extent,uint8_t* output_states,void* output_keys,uint32_t* layer_count,
    uint32_t layer_capacity,uint32_t* next_count,void* next_extents,uint32_t* route_count);
/* Test/inspection metadata views; device pointers, no host readback. */
int mgbfs_shard_ab_pipeline_views(void* pipeline,const uint32_t** counts,const uint32_t** offsets);
/* Enqueue a response-lease drain; synchronize the producer before host inspection. */
int mgbfs_shard_ab_pipeline_drain(void* pipeline);
int mgbfs_shard_ab_pipeline_destroy(void* pipeline);
#ifdef __cplusplus
}
#endif

// Opt-in MGBFS_SHARD_AB_ASYNC_MATERIALIZE=1 (indexed HASH and SORT_MERGE): apply_responses
// leases the source and an independent metadata snapshot. Source must remain
// unchanged until next select or finalize drains the copy on the producer
// stream. Subsequent source writes must use that producer stream. Stable state
// slots are assigned before copying; SORT_MERGE carries these slots while
// relocating keys. Final publication drains every outstanding response copy.
