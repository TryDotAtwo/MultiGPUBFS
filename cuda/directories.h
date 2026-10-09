#pragma once
#include "owner_job.h"
#ifdef __cplusplus
extern "C" {
#endif
int mgbfs_bucket_directory(const void* sorted,const uint32_t* count,uint32_t capacity,uint32_t buckets,
    MgbfsOwnerRange* directory,uint32_t* fatal,void* stream);
// Sorted input must belong to one logical owner (top bit). Output IDs are local.
int mgbfs_owner_bucket_directory(const void* sorted,const uint32_t* count,uint32_t capacity,uint32_t buckets,
    uint32_t owner,MgbfsOwnerRange* directory,uint32_t* fatal,void* stream);
// world is 1/2/4/8, buckets is LOCAL bucket count. Input belongs to owner.
int mgbfs_owner_bucket_directory_n(const void* sorted,const uint32_t* count,uint32_t capacity,
    uint32_t buckets,uint32_t owner,uint32_t world,MgbfsOwnerRange* directory,
    uint32_t* fatal,void* stream);
int mgbfs_bind_owner_jobs(MgbfsBucketJob* jobs,uint32_t count,const uint32_t* accepted_counts,uint32_t buckets,void* stream);
int mgbfs_compact_hash_layer(const void* accepted,const uint32_t* counts,uint32_t buckets,uint32_t k,
    void* output,uint32_t capacity,MgbfsOwnerRange* directory,uint32_t* total,uint32_t* fatal,void* stream);
/* Weighted FinalizeDepth companion: input StateRefs use the same fixed bucket
 * layout as accepted hashes. Both output planes have capacity records. Caller
 * retains StateRing/descriptor generations through all settlement readers.
 * Sticky fatal prevents BOTH output planes being written on capacity failure.
 * No allocation, count readback or host synchronization. */
int mgbfs_compact_hash_refs_layer(const void* accepted,const uint64_t* refs,
    const uint32_t* counts,uint32_t buckets,uint32_t k,void* output,uint64_t* output_refs,
    uint32_t capacity,MgbfsOwnerRange* directory,uint32_t* total,uint32_t* fatal,void* stream);
#ifdef __cplusplus
}
#endif
