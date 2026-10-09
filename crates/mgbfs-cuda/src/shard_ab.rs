use std::ffi::c_void;
extern "C" {
 pub fn mgbfs_exchange_count_device_n(world:u32,capacity:u32,hashes:*const c_void,count:*const u32,owners:*mut u32,stream:*mut c_void)->i32;
 pub fn mgbfs_generate_selected_compact_device_base(plan:*mut c_void,parents:*const u8,parent_count:u32,refs:*const u64,base:*const u32,bound:u32,requests:*const u32,count:*const u32,capacity:u32,output:*mut u8,fatal:*mut u32,stream:*mut c_void)->i32;
 pub fn mgbfs_generate_selected_compact(plan:*mut c_void,parents:*const u8,parent_count:u32,refs:*const u64,base:u32,bound:u32,requests:*const u32,count:*const u32,capacity:u32,output:*mut u8,fatal:*mut u32,stream:*mut c_void)->i32;
 pub fn mgbfs_shard_ab_pipeline_select(p:*mut c_void,keys:*const c_void,begin:*const u32,rows:*const u32,source_rows:*const u32,capacity:u32)->i32;
 pub fn mgbfs_shard_ab_pipeline_row_requests(p:*mut c_void,requests:*mut u32,count:*mut u32,capacity:u32)->i32;
 pub fn mgbfs_shard_ab_pipeline_apply_responses(p:*mut c_void,responses:*const u8,count:*const u32)->i32;

 pub fn mgbfs_debug_hashes_equal(expected:*const c_void,actual:*const c_void,n:u32,fatal:*mut u32,stream:*mut c_void)->i32;
 pub fn mgbfs_shard_ab_indexed_query(s:u32,c:u32,w:u32,k:u32,bytes:*mut u64)->i32;
 pub fn mgbfs_shard_ab_indexed_create(s:u32,c:u32,w:u32,k:u32,stream:*mut c_void,out:*mut *mut c_void)->i32;
 pub fn mgbfs_shard_ab_indexed_bind(p:*mut c_void,ring:*mut c_void,owner:*mut c_void,extent:*mut c_void,states:*mut u8,layer:*mut u32,capacity:u32,next_count:*mut u32,next:*mut c_void)->i32;

 pub fn mgbfs_shard_ab_pipeline_query(maximum:u32,capacity:u32,stride:u32,slots:u32,bytes:*mut u64)->i32;
 pub fn mgbfs_shard_ab_pipeline_create(maximum:u32,capacity:u32,stride:u32,slots:u32,stream:*mut c_void,out:*mut *mut c_void)->i32;
 pub fn mgbfs_shard_ab_pipeline_begin(p:*mut c_void,shards:u32,owner:u32,world:u32,previous:*const c_void,pn:u32,current:*const c_void,cn:u32,bound:u32,fatal:*mut u32)->i32;
 pub fn mgbfs_shard_ab_pipeline_push(p:*mut c_void,keys:*const c_void,states:*const u8,begin:*const u32,rows:*const u32,source_rows:*const u32,capacity:u32)->i32;
 pub fn mgbfs_shard_ab_pipeline_finalize(p:*mut c_void,keys:*mut *mut c_void,states:*mut *mut u8,count:*mut *const u32)->i32;
    pub fn mgbfs_shard_ab_pipeline_prepare_publish(pipeline: *mut c_void, ring: *mut c_void, owner: *mut c_void, extent: *mut c_void, layer_count: *mut u32, layer_capacity: u32) -> i32;
 pub fn mgbfs_shard_ab_pipeline_publish(p:*mut c_void,ring:*mut c_void,owner:*mut c_void,extent:*mut c_void,states:*mut u8,keys:*mut c_void,layer_count:*mut u32,layer_capacity:u32,next_count:*mut u32,next_extents:*mut c_void,route_count:*mut u32)->i32;
 pub fn mgbfs_shard_ab_pipeline_destroy(p:*mut c_void)->i32;
}
