#ifndef MGBFS_OWNER_JOB_H
#define MGBFS_OWNER_JOB_H
#include <stdint.h>
/* Architecture v2 owner-job ABI, Linux x86_64. No owning pointers.
 * Incoming ranges index one packed bounded lane input; prev/curr index
 * immutable compact layer arenas. These descriptors do not grant commit
 * credits. One lane exclusively owns the shard until publication completes.
 */
#if defined(__cplusplus)
#define MGBFS_OWNER_ALIGN64 alignas(64)
#elif defined(_MSC_VER)
#define MGBFS_OWNER_ALIGN64 __declspec(align(64))
#else
#define MGBFS_OWNER_ALIGN64 __attribute__((aligned(64)))
#endif
typedef struct MgbfsOwnerRange {
  uint64_t begin, count;
} MgbfsOwnerRange;
typedef struct MGBFS_OWNER_ALIGN64 MgbfsBucketJob {
  uint32_t bucket, lane;
  MgbfsOwnerRange incoming, prev, curr;
  uint32_t accepted_count, generation;
} MgbfsBucketJob;
#undef MGBFS_OWNER_ALIGN64
#endif
