#ifndef MGBFS_CUDA_H
#define MGBFS_CUDA_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI V1, Linux x86_64. Status 0 means successful enqueue, not GPU completion.
 * All pointers except coefficients/generators, out-plan and error are device
 * pointers. Stream is cudaStream_t cast to void*. No pointer aliasing is allowed.
 * Create allocates fixed buffers before depth zero. Run never allocates device
 * memory. Plans are non-reentrant: use one per concurrent stream/route slot.
 * Caller must wait for outstanding stream work before destroying a plan.
 * Canonical matrix bytes are row-major; physical stride is align_up(n*n,16).
 * Parents and children are canonical u8 (<modulus); padding is zero. Hash128
 * is four little-endian u32 residues, numerical order word3 through word0.
 */
int mgbfs_hash_create(uint32_t bytes,uint32_t capacity,const uint8_t* limbs,const uint32_t* offsets,void** out,char* error,size_t error_capacity);
/* Pre-allocation queries. Host outputs, no cudaMalloc or CUDA launches.
 * Bytes exclude caller-owned input/output and allocator/runtime overhead.
 * Failed query zeroes output. Workspace is queried from the compiled GEMM.
 */
typedef struct MgbfsGenerateBytes {
  uint64_t generators, packed_parents, products_s32, workspace;
  uint32_t k, stride, rows, columns;
} MgbfsGenerateBytes;
typedef struct MgbfsHashBytes {
  uint64_t weights, offsets, partials_s32, workspace;
  uint32_t stride, reserved;
} MgbfsHashBytes;
int mgbfs_generate_query(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,uint32_t variant,MgbfsGenerateBytes* out);
int mgbfs_hash_query(uint32_t bytes,uint32_t capacity,MgbfsHashBytes* out);
int mgbfs_hash_run(void* plan,const uint8_t* input,uint32_t* output,uint32_t count,void* stream);
void mgbfs_hash_destroy(void* plan);
int mgbfs_generate_create(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,const uint8_t* generators,void** out,char* error,size_t error_capacity);
/* Explicit fixed generation variants: 0 legacy 64x32x64; 1 transposed
 * 64x32x32; 2 transposed 128x32x32; 3 transposed 64x32x64;
 * 4 transposed 64x32x32 with vector U4 output (requires n=4). No auto selection.
 * Same parent/child wire layout and arithmetic. Run status 7 rejects grid
 * overflow before any enqueue/output write. Existing create selects 0.
 */
int mgbfs_generate_create_variant(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,const uint8_t* generators,uint32_t variant,void** out,char* error,size_t error_capacity);
/* Macro transitions must be sorted by nondecreasing positive original-edge
 * weight. They are evaluated by one GEMM, but children are materialized in
 * move-major order: [move][parent][canonical padded state]. Thus each weight
 * run is already contiguous and needs no full-state transpose before routing.
 */
int mgbfs_generate_create_macro_variant(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,
  const uint8_t* generators,const uint32_t* weights,uint32_t variant,void** out,char* error,size_t error_capacity);
int mgbfs_generate_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* stream);
/* Measurement only: host array of four already-created CUDA timing events.
 * Records start, packed, GEMM done, children done. No synchronization/allocation.
 * Ordinary run does not record these events. count must be nonzero.
 */
int mgbfs_generate_profile_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* stream,void* const* marks);
void mgbfs_generate_destroy(void* plan);
int mgbfs_route_create(uint32_t capacity,void** out,char* error,size_t error_capacity);
/* CUB query uses the CURRENT CUDA device and compiled CUB policy. No kernels
 * or explicit device allocation. Query/create/run must use the same device.
 * sort/select share scratch sequentially: only max is allocated, not their sum.
 * Caller input/output/count banks and CUDA context overhead are excluded.
 */
typedef struct MgbfsRouteBytes {
  uint64_t sorted, refs, indices, selected, flags, scratch, sort_scratch, select_scratch;
} MgbfsRouteBytes;
int mgbfs_route_query(uint32_t capacity,MgbfsRouteBytes* out);
int mgbfs_route_run(void* plan,const void* hashes,const uint64_t* refs,void* sorted_hashes,uint64_t* sorted_refs,uint32_t* output_count,uint32_t count,int pre_dedup,void* stream);
void mgbfs_route_destroy(void* plan);
typedef struct MgbfsOwnerState {
  uint64_t last_epoch;
  uint32_t count, initialized, fatal, reserved;
} MgbfsOwnerState;
/* Owner bucket primitive, not the rank scheduler. prev/curr and candidates are
 * sorted Hash128 runs for ONE bucket. They may contain duplicate hashes.
 * accepted is a caller-owned flat bucket span of bucket_capacity records;
 * state must be zero-initialized before its first epoch. One scratch plan can
 * process different spans serially; concurrent jobs require separate plans.
 * candidate_count lives on device and is bounded by candidate_capacity.
 * All output spans have candidate_capacity entries; accepted has bucket_capacity.
 * state.fatal: 1 capacity, 2 epoch order, 3 unsorted input. Sticky until the
 * enclosing failed run is destroyed. Fatal preserves accepted and publishes
 * zero survivors. The host must treat any nonzero ABI return as group-fatal too.
 * Successful publication follows all dense writes in the supplied stream.
 */
int mgbfs_owner_create(uint32_t candidate_capacity,uint32_t bucket_capacity,void** out,char* error,size_t error_capacity);
int mgbfs_owner_run(void* plan,const void* prev,uint32_t prev_count,const void* curr,uint32_t curr_count,
  void* accepted,MgbfsOwnerState* state,const void* candidates,const uint64_t* refs,const uint32_t* candidate_count,
  void* survivors,uint64_t* survivor_refs,uint32_t* survivor_count,uint64_t epoch,void* stream);
void mgbfs_owner_destroy(void* plan);
typedef struct MgbfsFrontierState { uint32_t count, fatal; } MgbfsFrontierState;
/* Materialize one committed batch from a live source slot. References are local
 * source row indices, not global OriginRefs. Source stride is a multiple of 16.
 * Requests are sorted by source index; output states and hashes share that order.
 * All destination spans are disjoint from inputs. Append jobs serialize on one
 * stream. Device count may be zero. fatal: 1 capacity, 2 invalid source ref.
 * Failure is sticky and leaves destination bytes/count unchanged. The caller
 * retains source and request leases until stream completion and aborts the run
 * on any fatal (including a previous owner failure). No allocations in run.
 */
int mgbfs_materialize_create(uint32_t stride,uint32_t candidate_capacity,uint32_t frontier_capacity,void** out,char* error,size_t error_capacity);
int mgbfs_materialize_run(void* plan,const uint8_t* source,uint32_t source_count,const void* hashes,const uint64_t* refs,const uint32_t* count,uint8_t* states,void* out_hashes,MgbfsFrontierState* state,void* stream);
void mgbfs_materialize_destroy(void* plan);
#ifdef __cplusplus
}
#endif
#endif
