//! Generic successor ABI. Inputs and scratch are owned preallocated device arenas.
//! CPU graph validation happens before allocation; no CPU state generation here.
#[cfg(feature="cuda")]
extern "C" {
 /// Generate all children or regenerate only selected child origins into SoA.
 /// Caller provides valid tables, nonoverlapping arenas, capacities and stream.
 /// device_error is a zero-initialized u32 on the device, retained until its
 /// consumer event completes. Selected-child failures are reported there.
 pub fn mgbfs_generic_generate_i64(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,parent_count:u32,parent_stride:u32,permutation_tables:*const u32,
  matrix_tables:*const i64,moduli:*const u32,selected_children:*const u64,output_count:u32,
  output:*mut i64,output_stride:u32,device_error:*mut u32,stream:*mut std::ffi::c_void)->i32;
}

#[cfg(feature="cuda")]
extern "C" {
 /// All allocations remain stable until stream completion. Pending hash entries
 /// refer to immutable parent origins. On any device error, the future layer
 /// is invalid and this arena must not be resumed with another parent bank.
 pub fn mgbfs_generic_seed_i64(elements:u32,states:*const i64,state_stride:u32,state_count:u32,
  slots:*mut u64,slot_capacity:u32,seed:u64,hash_bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_seed_shared_i64(elements:u32,states:*const i64,state_stride:u32,state_count:u32,
  slots:*mut u64,slot_capacity:u32,seed:u64,hash_bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_expand_i64(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,parent_count:u32,parent_stride:u32,permutation_tables:*const u32,
  matrix_tables:*const i64,moduli:*const u32,slots:*mut u64,slot_capacity:u32,
  visited:*mut i64,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,
  future_capacity:u32,future_count:*mut u32,seed:u64,hash_bits:u32,error:*mut u32,
  stream:*mut std::ffi::c_void)->i32;
 /// Indices must be validated device row references, not unchecked user input.
 pub fn mgbfs_generic_gather_i64(elements:u32,source:*const i64,source_stride:u32,
  indices:*const u32,count:u32,output:*mut i64,output_stride:u32,stream:*mut std::ffi::c_void)->i32;
}

#[repr(C)]
#[derive(Clone,Copy,Debug,Default)]
pub struct GenericRouteRecord {pub hash:u64,pub parent:u64,pub source:u32,pub generator:u32,pub shard:u32,pub reserved:u32}
const _: [();32]=[();std::mem::size_of::<GenericRouteRecord>()];
#[cfg(feature="cuda")]
extern "C" {
 /// Owner map/cuts must pass startup validation; parent leases remain valid
 /// until destination consumers finish, not merely until routing completes.
 pub fn mgbfs_generic_route_i64(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,count:u32,stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,
  seed:u64,hash_bits:u32,world:u32,source:u32,local_shards:u32,queue_capacity:u32,parent_begin:u64,
  owner_to_rank:*const u32,owner_cuts:*const u64,queues:*mut GenericRouteRecord,counts:*mut u32,
  error:*mut u32,stream:*mut std::ffi::c_void)->i32;
}

#[cfg(feature="cuda")]
extern "C" {
 pub fn mgbfs_generic_regenerate_routes_i64(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,parent_count:u32,parent_stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,
  source:u32,parent_begin:u64,requests:*const GenericRouteRecord,request_count:u32,output:*mut i64,
  output_stride:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
}

#[cfg(feature="cuda")]
extern "C" {
 /// The descriptor and payload order must agree. One exclusive consumer
 /// lease per shard table; A/B producer overlap never reuses leased input.
 pub fn mgbfs_generic_accept_i64(elements:u32,incoming:*const i64,stride:u32,
  metadata:*const GenericRouteRecord,received:*const u32,bound:u32,slots:*mut u64,
  slot_capacity:u32,visited:*mut i64,visited_capacity:u32,visited_count:*mut u32,
  future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
}

#[cfg(feature="cuda")]
extern "C" {
 pub fn mgbfs_generic_regenerate_routes_count_i64(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,parent_count:u32,parent_stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,
  source:u32,parent_begin:u64,requests:*const GenericRouteRecord,request_capacity:u32,device_count:*const u32,
  output:*mut i64,output_stride:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
}

#[cfg(feature="cuda")]
extern "C" {
 pub fn mgbfs_generic_generate_u8(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,parents:*const u8,parent_count:u32,parent_stride:u32,permutation_tables:*const u32,matrix_tables:*const i64,moduli:*const u32,selected_children:*const u64,output_count:u32,output:*mut u8,output_stride:u32,device_error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_seed_u8(elements:u32,states:*const u8,state_stride:u32,state_count:u32,slots:*mut u64,slot_capacity:u32,seed:u64,hash_bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_seed_shared_u8(elements:u32,states:*const u8,state_stride:u32,state_count:u32,slots:*mut u64,slot_capacity:u32,seed:u64,hash_bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_expand_u8(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,parents:*const u8,parent_count:u32,parent_stride:u32,permutation_tables:*const u32,matrix_tables:*const i64,moduli:*const u32,slots:*mut u64,slot_capacity:u32,visited:*mut u8,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,seed:u64,hash_bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_gather_u8(elements:u32,source:*const u8,source_stride:u32,indices:*const u32,count:u32,output:*mut u8,output_stride:u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_route_u8(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,parents:*const u8,count:u32,stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,seed:u64,hash_bits:u32,world:u32,source:u32,local_shards:u32,queue_capacity:u32,parent_begin:u64,owner_to_rank:*const u32,owner_cuts:*const u64,queues:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_route_packed_u8(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,parents:*const u8,count:u32,stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,seed:u64,hash_bits:u32,world:u32,source:u32,local_shards:u32,queue_capacity:u32,parent_begin:u64,owner_to_rank:*const u32,owner_cuts:*const u64,queues:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_regenerate_routes_u8(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,parents:*const u8,parent_count:u32,parent_stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,source:u32,parent_begin:u64,requests:*const GenericRouteRecord,request_count:u32,output:*mut u8,output_stride:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_accept_u8(elements:u32,incoming:*const u8,stride:u32,metadata:*const GenericRouteRecord,received:*const u32,bound:u32,slots:*mut u64,slot_capacity:u32,visited:*mut u8,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_regenerate_routes_count_u8(kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,parents:*const u8,parent_count:u32,parent_stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,source:u32,parent_begin:u64,requests:*const GenericRouteRecord,request_capacity:u32,device_count:*const u32,output:*mut u8,output_stride:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
}

// Raw state pointers are never dereferenced in Rust; storage width chooses the typed ABI.
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_generate_storage(state_bytes:u32,kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,parent_count:u32,parent_stride:u32,permutation_tables:*const u32,
  matrix_tables:*const i64,moduli:*const u32,selected_children:*const u64,output_count:u32,
  output:*mut i64,output_stride:u32,device_error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_generate_u8(kind,elements,rows,cols,generators,parents.cast(),parent_count,parent_stride,permutation_tables,matrix_tables,moduli,selected_children,output_count,output.cast(),output_stride,device_error,stream),8=>mgbfs_generic_generate_i64(kind,elements,rows,cols,generators,parents,parent_count,parent_stride,permutation_tables,matrix_tables,moduli,selected_children,output_count,output,output_stride,device_error,stream),_=>1}}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_seed_storage(state_bytes:u32,elements:u32,states:*const i64,state_stride:u32,state_count:u32,
  slots:*mut u64,slot_capacity:u32,seed:u64,hash_bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_seed_u8(elements,states.cast(),state_stride,state_count,slots,slot_capacity,seed,hash_bits,error,stream),8=>mgbfs_generic_seed_i64(elements,states,state_stride,state_count,slots,slot_capacity,seed,hash_bits,error,stream),_=>1}}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_expand_storage(state_bytes:u32,kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,parent_count:u32,parent_stride:u32,permutation_tables:*const u32,
  matrix_tables:*const i64,moduli:*const u32,slots:*mut u64,slot_capacity:u32,
  visited:*mut i64,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,
  future_capacity:u32,future_count:*mut u32,seed:u64,hash_bits:u32,error:*mut u32,
  stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_expand_u8(kind,elements,rows,cols,generators,parents.cast(),parent_count,parent_stride,permutation_tables,matrix_tables,moduli,slots,slot_capacity,visited.cast(),visited_capacity,visited_count,future,future_capacity,future_count,seed,hash_bits,error,stream),8=>mgbfs_generic_expand_i64(kind,elements,rows,cols,generators,parents,parent_count,parent_stride,permutation_tables,matrix_tables,moduli,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,seed,hash_bits,error,stream),_=>1}}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_gather_storage(state_bytes:u32,elements:u32,source:*const i64,source_stride:u32,
  indices:*const u32,count:u32,output:*mut i64,output_stride:u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_gather_u8(elements,source.cast(),source_stride,indices,count,output.cast(),output_stride,stream),8=>mgbfs_generic_gather_i64(elements,source,source_stride,indices,count,output,output_stride,stream),_=>1}}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_route_storage(state_bytes:u32,kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,count:u32,stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,
  seed:u64,hash_bits:u32,world:u32,source:u32,local_shards:u32,queue_capacity:u32,parent_begin:u64,
  owner_to_rank:*const u32,owner_cuts:*const u64,queues:*mut GenericRouteRecord,counts:*mut u32,
  error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>(if elements<=24 {mgbfs_generic_route_packed_u8} else {mgbfs_generic_route_u8})(kind,elements,rows,cols,generators,parents.cast(),count,stride,permutations,matrices,moduli,seed,hash_bits,world,source,local_shards,queue_capacity,parent_begin,owner_to_rank,owner_cuts,queues,counts,error,stream),8=>mgbfs_generic_route_i64(kind,elements,rows,cols,generators,parents,count,stride,permutations,matrices,moduli,seed,hash_bits,world,source,local_shards,queue_capacity,parent_begin,owner_to_rank,owner_cuts,queues,counts,error,stream),_=>1}}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_regenerate_routes_storage(state_bytes:u32,kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,parent_count:u32,parent_stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,
  source:u32,parent_begin:u64,requests:*const GenericRouteRecord,request_count:u32,output:*mut i64,
  output_stride:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_regenerate_routes_u8(kind,elements,rows,cols,generators,parents.cast(),parent_count,parent_stride,permutations,matrices,moduli,source,parent_begin,requests,request_count,output.cast(),output_stride,error,stream),8=>mgbfs_generic_regenerate_routes_i64(kind,elements,rows,cols,generators,parents,parent_count,parent_stride,permutations,matrices,moduli,source,parent_begin,requests,request_count,output,output_stride,error,stream),_=>1}}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_accept_storage(state_bytes:u32,elements:u32,incoming:*const i64,stride:u32,
  metadata:*const GenericRouteRecord,received:*const u32,bound:u32,slots:*mut u64,
  slot_capacity:u32,visited:*mut i64,visited_capacity:u32,visited_count:*mut u32,
  future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_accept_u8(elements,incoming.cast(),stride,metadata,received,bound,slots,slot_capacity,visited.cast(),visited_capacity,visited_count,future,future_capacity,future_count,error,stream),8=>mgbfs_generic_accept_i64(elements,incoming,stride,metadata,received,bound,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream),_=>1}}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_regenerate_routes_count_storage(state_bytes:u32,kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const i64,parent_count:u32,parent_stride:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,
  source:u32,parent_begin:u64,requests:*const GenericRouteRecord,request_capacity:u32,device_count:*const u32,
  output:*mut i64,output_stride:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_regenerate_routes_count_u8(kind,elements,rows,cols,generators,parents.cast(),parent_count,parent_stride,permutations,matrices,moduli,source,parent_begin,requests,request_capacity,device_count,output.cast(),output_stride,error,stream),8=>mgbfs_generic_regenerate_routes_count_i64(kind,elements,rows,cols,generators,parents,parent_count,parent_stride,permutations,matrices,moduli,source,parent_begin,requests,request_capacity,device_count,output,output_stride,error,stream),_=>1}}

#[cfg(feature="cuda")]
extern "C" {
 pub fn mgbfs_generic_accept_all_i64(elements:u32,local:*const i64,remote:*const i64,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
 local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,stride:u32,slots:*mut u64,
 slot_capacity:u32,visited:*mut i64,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_accept_all_shared_i64(elements:u32,local:*const i64,remote:*const i64,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
 local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,stride:u32,slots:*mut u64,
 slot_capacity:u32,visited:*mut i64,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_accept_all_u8(elements:u32,local:*const u8,remote:*const u8,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
 local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,stride:u32,slots:*mut u64,
 slot_capacity:u32,visited:*mut u8,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_accept_all_shared_u8(elements:u32,local:*const u8,remote:*const u8,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
 local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,stride:u32,slots:*mut u64,
 slot_capacity:u32,visited:*mut u8,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_accept_all_packed_u8(elements:u32,local:*const u8,remote:*const u8,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
 local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,stride:u32,slots:*mut u64,
 slot_capacity:u32,visited:*mut u8,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_accept_all_shared_packed_u8(elements:u32,local:*const u8,remote:*const u8,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
 local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,stride:u32,slots:*mut u64,
 slot_capacity:u32,visited:*mut u8,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_accept_all_storage(state_bytes:u32,elements:u32,local:*const i64,remote:*const i64,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
 local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,stride:u32,slots:*mut u64,
 slot_capacity:u32,visited:*mut i64,visited_capacity:u32,visited_count:*mut u32,future:*mut u32,future_capacity:u32,future_count:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>(if elements<=24 {mgbfs_generic_accept_all_shared_packed_u8} else {mgbfs_generic_accept_all_shared_u8})(elements,local.cast(),remote.cast(),local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited.cast(),visited_capacity,visited_count,future,future_capacity,future_count,error,stream),8=>mgbfs_generic_accept_all_shared_i64(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream),_=>1}}

extern "C" {pub fn mgbfs_generic_route_retry_vote(source:*const u32,owner:*const u32,vote:*mut u32,stream:*mut std::ffi::c_void)->i32;}

#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_seed_shared_storage(state_bytes:u32,elements:u32,states:*const i64,state_stride:u32,state_count:u32,
  slots:*mut u64,slot_capacity:u32,seed:u64,hash_bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_seed_shared_u8(elements,states.cast(),state_stride,state_count,slots,slot_capacity,seed,hash_bits,error,stream),8=>mgbfs_generic_seed_shared_i64(elements,states,state_stride,state_count,slots,slot_capacity,seed,hash_bits,error,stream),_=>1}}

#[cfg(feature="cuda")]
extern "C" {
 pub fn mgbfs_generic_accept_rolling_i64(elements:u32,local:*const i64,remote:*const i64,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,slots:*mut u64,slot_capacity:u32,arena:*mut i64,stride:u32,base:u32,capacity:u32,accepted:*mut u32,future:*mut u32,positions:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_accept_rolling_u8(elements:u32,local:*const u8,remote:*const u8,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,slots:*mut u64,slot_capacity:u32,arena:*mut u8,stride:u32,base:u32,capacity:u32,accepted:*mut u32,future:*mut u32,positions:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_accept_rolling_packed_u8(elements:u32,local:*const u8,remote:*const u8,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,slots:*mut u64,slot_capacity:u32,arena:*mut u8,stride:u32,base:u32,capacity:u32,accepted:*mut u32,future:*mut u32,positions:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_reseed_rows_i64(elements:u32,arena:*const i64,stride:u32,base:u32,count:u32,slots:*mut u64,slot_capacity:u32,positions:*mut u32,seed:u64,bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_reseed_rows_u8(elements:u32,arena:*const u8,stride:u32,base:u32,count:u32,slots:*mut u64,slot_capacity:u32,positions:*mut u32,seed:u64,bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
 pub fn mgbfs_generic_retire_rows(slots:*mut u64,slot_capacity:u32,positions:*const u32,stride:u32,base:u32,count:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32;
}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_accept_rolling_storage(state_bytes:u32,elements:u32,local:*const i64,remote:*const i64,local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,slots:*mut u64,slot_capacity:u32,arena:*mut i64,stride:u32,base:u32,capacity:u32,accepted:*mut u32,future:*mut u32,positions:*mut u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>(if elements<=24{mgbfs_generic_accept_rolling_packed_u8}else{mgbfs_generic_accept_rolling_u8})(elements,local.cast(),remote.cast(),local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,q,slots,slot_capacity,arena.cast(),stride,base,capacity,accepted,future,positions,error,stream),8=>mgbfs_generic_accept_rolling_i64(elements,local.cast(),remote.cast(),local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,q,slots,slot_capacity,arena.cast(),stride,base,capacity,accepted,future,positions,error,stream),_=>1}}
#[cfg(feature="cuda")]
pub unsafe fn mgbfs_generic_reseed_rows_storage(state_bytes:u32,elements:u32,arena:*const i64,stride:u32,base:u32,count:u32,slots:*mut u64,slot_capacity:u32,positions:*mut u32,seed:u64,bits:u32,error:*mut u32,stream:*mut std::ffi::c_void)->i32{match state_bytes{1=>mgbfs_generic_reseed_rows_u8(elements,arena.cast(),stride,base,count,slots,slot_capacity,positions,seed,bits,error,stream),8=>mgbfs_generic_reseed_rows_i64(elements,arena,stride,base,count,slots,slot_capacity,positions,seed,bits,error,stream),_=>1}}
