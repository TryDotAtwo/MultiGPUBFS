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
