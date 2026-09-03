#pragma once
#include "owner_job.h"
#ifdef __cplusplus
extern "C" {
#endif
int mgbfs_bucket_directory(const void* sorted,const uint32_t* count,uint32_t capacity,uint32_t buckets,
    MgbfsOwnerRange* directory,uint32_t* fatal,void* stream);
int mgbfs_bind_owner_jobs(MgbfsBucketJob* jobs,uint32_t count,const uint32_t* accepted_counts,uint32_t buckets,void* stream);
int mgbfs_compact_hash_layer(const void* accepted,const uint32_t* counts,uint32_t buckets,uint32_t k,
    void* output,uint32_t capacity,MgbfsOwnerRange* directory,uint32_t* total,uint32_t* fatal,void* stream);
#ifdef __cplusplus
}
#endif
