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
int mgbfs_hash_run(void* plan,const uint8_t* input,uint32_t* output,uint32_t count,void* stream);
void mgbfs_hash_destroy(void* plan);
int mgbfs_generate_create(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,const uint8_t* generators,void** out,char* error,size_t error_capacity);
int mgbfs_generate_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* stream);
void mgbfs_generate_destroy(void* plan);
int mgbfs_route_create(uint32_t capacity,void** out,char* error,size_t error_capacity);
int mgbfs_route_run(void* plan,const void* hashes,const uint64_t* refs,void* sorted_hashes,uint64_t* sorted_refs,uint32_t* output_count,uint32_t count,int pre_dedup,void* stream);
void mgbfs_route_destroy(void* plan);
#ifdef __cplusplus
}
#endif
#endif
