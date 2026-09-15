#ifndef MGBFS_LIBRARY_OWNER_ABI_H
#define MGBFS_LIBRARY_OWNER_ABI_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* V1, Linux x86_64. Every data pointer is device-resident on the owner device.
 * Counts/structs themselves are host control data. SoA conversion from the
 * native AoS Hash128 format is explicit and must be counted in time/memory.
 * This header defines the integration boundary; implementation is pending.
 */
typedef struct MgbfsLibraryKeysV1 {
  const uint32_t* words[4];
  uint32_t rows;
  uint32_t reserved; /* must be zero */
} MgbfsLibraryKeysV1;

typedef struct MgbfsLibraryCandidatesV1 {
  MgbfsLibraryKeysV1 keys;
  const uint32_t* source_indices;
} MgbfsLibraryCandidatesV1;

typedef struct MgbfsLibrarySurvivorsV1 {
  const uint32_t* source_indices;
  uint64_t epoch;
  uint32_t rows;
  uint32_t reserved;
} MgbfsLibrarySurvivorsV1;

/* Caller installs the fixed RMM resource before creation; it and immutable
 * history outlive all owners. Exactly one serialized writer/stream per owner.
 * Factory returns a null handle on error. Status 0 means success, never a
 * backend fallback. Exceptions must not cross the C boundary.
 */
int mgbfs_library_owner_create_v1(MgbfsLibraryKeysV1 history, uint32_t capacity,
                                 void* cuda_stream, void** owner);
/* Compare does not publish. Returned indices are borrowed until next compare
 * or destroy. Library count synchronization is included in this call's timing.
 */
int mgbfs_library_owner_compare_v1(void* owner, uint64_t epoch,
                                  MgbfsLibraryCandidatesV1 input,
                                  MgbfsLibrarySurvivorsV1* result);
/* granted_rows must be actual reserved state/archive credits, not a requested
 * capacity. Success only means commit was enqueued. Caller records/waits its
 * stream event before publishing host counts or releasing external leases.
 * Any error poisons the owner; the rank group must fail, not retry.
 */
int mgbfs_library_owner_commit_v1(void* owner, uint64_t epoch, uint32_t granted_rows);
/* Caller drains the owner's stream before destroy. No hidden global barrier. */
void mgbfs_library_owner_destroy_v1(void* owner);

#ifdef __cplusplus
}
static_assert(sizeof(MgbfsLibraryKeysV1) == 40, "keys ABI");
static_assert(sizeof(MgbfsLibraryCandidatesV1) == 48, "candidate ABI");
static_assert(sizeof(MgbfsLibrarySurvivorsV1) == 24, "survivor ABI");
#endif
#endif
