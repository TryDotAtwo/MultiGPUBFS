#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* One exclusive dedup slot. Keys are aligned AoS Hash128 residues; states use
 * aligned flat rows. Device count, keys and states belong to one physical A/B
 * buffer. Historical sorted keys remain borrowed through completion.
 * No allocation or count readback in run. Caller orders writers before run
 * and does not reuse the slot or physical buffer until completion.
 * Same baseline identity contract as the existing Hash128 owner. */
int mgbfs_shard_ab_job_query(uint32_t capacity,uint32_t stride,uint64_t* bytes);
int mgbfs_shard_ab_job_create(uint32_t capacity,uint32_t stride,void** output);
int mgbfs_shard_ab_job_run(void* job,void* keys,uint8_t* states,uint32_t* count,
    const void* previous,uint32_t previous_count,const void* current,
    uint32_t current_count,uint32_t* fatal,void* stream);
/* active_capacity is a host-known conservative bound on clean+dirty rows.
 * Scratch remains fixed at create capacity. Create queries and reserves CUB workspace for all powers of two up to its
 * capacity, plus the exact create capacity. Run accepts only these bounds. */
int mgbfs_shard_ab_job_run_bounded(void* job,uint32_t active_capacity,void* keys,
    uint8_t* states,uint32_t* count,const void* previous,uint32_t previous_count,
    const void* current,uint32_t current_count,uint32_t* fatal,void* stream);
int mgbfs_shard_ab_job_destroy(void* job);
#ifdef __cplusplus
}
#endif

extern "C" int mgbfs_shard_ab_job_run_added(void*,uint32_t,void*,uint8_t*,uint32_t*,const void*,uint32_t,const void*,uint32_t,uint32_t*,const uint32_t*,void*);

extern "C" int mgbfs_shard_ab_job_run_added_ready(void*,uint32_t,void*,uint8_t*,uint32_t*,const void*,uint32_t,const void*,uint32_t,uint32_t*,const uint32_t*,uint32_t,void*);
