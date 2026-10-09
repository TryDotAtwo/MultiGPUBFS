#pragma once
#include <stdint.h>
// Host ABI, no allocations and no CPU readback of GPU counters.
struct GenericSortedNativeInput {
 uint32_t elements,world,rank,shard,shards,queue_capacity,state_bytes,transport;
 uint32_t kind,rows,cols,generators,parent_stride,chunk,hash_bits,reserved;
 uint64_t begin,seed;
 const void *local,*remote,*local_meta,*remote_meta,*local_counts,*remote_counts;
 const void *parents,*cursors,*frontiers,*permutations,*matrices,*moduli;
};
struct GenericSortedNativeWorkspace {
 void *hashes,*origins,*sorted,*flags,*prefix,*unique_hashes,*unique_origins,*unique_count,*temporary;
 uint64_t temporary_bytes;
};
struct GenericSortedNativeDestination {
 const void* snapshots;uint32_t snapshot_count,stride,base,capacity,rolling,reserved;
 void *pool,*run_hashes,*run_rows,*arena,*row_count,*frontier_count,*future,*reservation,*carry,*error;
};
extern "C" {
 int mgbfs_generic_sorted_native_shape(const GenericSortedNativeInput*,uint64_t*,uint64_t*,uint64_t*,void*);
 int mgbfs_generic_sorted_native_origin(const GenericSortedNativeInput*,const GenericSortedNativeWorkspace*,void*,void*);
 int mgbfs_generic_sorted_native_accept(const GenericSortedNativeInput*,const GenericSortedNativeWorkspace*,const GenericSortedNativeDestination*,void*);
 int mgbfs_generic_sorted_native_metadata_shape(uint64_t*,uint32_t);
 int mgbfs_generic_sorted_native_snapshot_acquire(const void*,const void*,void*,void*,void*,void*,void*);
 int mgbfs_generic_sorted_native_snapshot_release(const void*,void*,void*,void*);
 int mgbfs_generic_sorted_native_retire(const void*,void*,void*,void*);
 int mgbfs_generic_sorted_native_seed(const void*,void*,void*,void*,uint64_t,uint32_t,void*,void*);
}

// Cold graph construction; all buffers stay owned by the caller through destroy.
struct GenericSortedNativeCarry {
 const void* pool;
 void *tiers,*carry,*hashes,*rows,*arena,*error,*control,*handles;
 uint32_t stride,width,state_bytes,classes;
};
extern "C" {
 int mgbfs_generic_sorted_native_carry_shape(uint32_t,uint32_t,uint64_t*,uint64_t*);
 int mgbfs_generic_sorted_native_carry_create(const GenericSortedNativeCarry*,void*,void**);
 int mgbfs_generic_sorted_native_carry_launch(void*,void*);
 int mgbfs_generic_sorted_native_carry_destroy(void*);
}
