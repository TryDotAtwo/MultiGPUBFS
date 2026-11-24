//! Experimental independently segmented distributed exact BFS.
//! Not selected by the public runtime until multi-process and failure gates pass.
use std::{cell::Cell,ffi::c_void,mem::{size_of,size_of_val},ptr,rc::Rc};
use mgbfs_core::{graph_definition::GraphDefinitionV2,Result};
use mgbfs_cuda::{ffi::*,generic_graph::*};
use crate::{segmented_strategy::{SegmentStrategy,OwnerDecision,owner_decision,RawTileCursor},generic_distributed_memory::GenericDistributedMemoryPlan as Plan,
 generic_distributed_native::DistributedAdvance,generic_memory::GenericMemoryPlan,
 generic_native::{Buffer,GenericNativeBfs,Stream},segmented_history::{fit_segment_layout_compact,fit_packed_bitmap_segment_layout,fit_sorted_segment_layout_with_row_metadata,BitmapPack4Dictionary,HashLookupDirectorySeries,SegmentLayout,LOCAL_ROWS}};
mod hash_index;
mod layer;
mod nearest;
mod window;
pub(crate) mod packet;
use hash_index::{hash_directory_bytes, initial_hash_run_capacity};
pub(crate) use nearest::{RootSpec, PATH_PARENT_BYTES};
use nearest::{state_hash, NearestFns, NearestState, ParentLink, PathOwnership};
use window::WindowState;
use packet::{PacketCodec,PacketConfig,PacketOwnerShard,PacketRuntime,PACKET_ROUTE_STATUS_BYTES};
use crate::segmented_packet_transport::{PacketJobToken,PACKET_CANCELLED,PACKET_DEADLINE};

#[repr(C)] #[derive(Clone,Copy)]
struct SegmentDescriptor{slots:*mut u64,states:*mut c_void,positions:*mut u64,slots_count:u64,capacity:u64,banks:u32,position_item_bytes:u32}
#[repr(C)] #[derive(Clone,Copy)]
struct BatchHashSegment{states:*mut c_void,keys:*mut u64,rows:*mut u32,capacity:u64,banks:u32}
#[repr(C)] #[derive(Clone,Copy)]
struct BatchHashRun{segment:u32,bank:u32,begin:u64,length:u64,min_hash:u64,max_hash:u64,hash_fences:[u64;7]}
const _: [();96]=[();std::mem::size_of::<BatchHashRun>()];

#[repr(C)]
#[derive(Clone,Copy)]
struct HashSeriesView{
 keys:*const u64,references:*const c_void,prefix_offsets:*const u32,length:u32,
 replaced_runs:u32,reference_bytes:u32,row_bits:u32,
}
#[repr(C)]
struct HashSeriesDirectory{series:[HashSeriesView;3],count:u32,replaced_runs:u32}
const _: [();40]=[();std::mem::size_of::<HashSeriesView>()];
const _: [();128]=[();std::mem::size_of::<HashSeriesDirectory>()];
struct HashSeriesSnapshot{storage:RawBuffer,view:HashSeriesView,bank:u32}
extern "C" {
 fn mgbfs_segmented_abi_version()->u32;
 fn cudaGetLastError()->i32;
 fn cudaEventSynchronize(event:*mut c_void)->i32;
 fn cudaEventElapsedTime(ms:*mut f32,start:*mut c_void,end:*mut c_void)->i32;
 fn mgbfs_segmented_route_storage(state_bytes:u32,kind:u32,elements:u32,n:u32,m:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,matrices:*const i64,moduli:*const u32,seed:u64,bits:u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,cuts:*const u64,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_route_storage_grouped(state_bytes:u32,kind:u32,elements:u32,n:u32,m:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,matrices:*const i64,moduli:*const u32,seed:u64,bits:u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,cuts:*const u64,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_max_queue_count(counts:*const u32,n:u32,maximum:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_nccl_all_reduce_max_pair_u32(comm:*mut c_void,send:*const u32,recv:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_route(state_bytes:u32,domain:BitmapDeviceDomain,elements:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_route_grouped(state_bytes:u32,domain:BitmapDeviceDomain,elements:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_batch_route_scratch_bytes(children:u32,buckets:u32,bytes:*mut usize)->i32;
 fn mgbfs_segmented_batch_route(bitmap:u32,word_shard:u32,domain:BitmapDeviceDomain,state_bytes:u32,kind:u32,elements:u32,n:u32,m:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,matrices:*const i64,moduli:*const u32,seed:u64,bits:u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,cuts:*const u64,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,states:*mut c_void,scratch:*mut c_void,scratch_bytes:usize,stream:*mut c_void)->i32;
 fn mgbfs_hash_batch_scratch_bytes(n:u32,state_bytes:u32,packed:u32,bytes:*mut usize)->i32;
 fn mgbfs_hash_batch_seed_v2(state_bytes:u32,elements:u32,input:*const c_void,segment:BatchHashSegment,hash:u64,runs:*mut BatchHashRun,count:*mut u32,accepted:*mut u64,stream:*mut c_void)->i32;
 fn mgbfs_hash_batch_retire_bank_v2(runs:*mut BatchHashRun,count:*mut u32,capacity:u32,segments:*const BatchHashSegment,segment_count:u32,bank:u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_hash_batch_accept_v2(state_bytes:u32,packed:u32,elements:u32,local:*const c_void,remote:*const c_void,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,raw_begin:u32,segments:*const BatchHashSegment,segment_count:u32,active:u32,bank:u32,runs:*mut BatchHashRun,run_count:*mut u32,run_capacity:u32,accepted:*mut u64,spill:*mut u32,spill_count:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,scratch:*mut c_void,scratch_bytes:usize,stream:*mut c_void)->i32;
 fn mgbfs_hash_batch_accept_profiled_v6(state_bytes:u32,packed:u32,elements:u32,local:*const c_void,remote:*const c_void,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,raw_begin:u32,segments:*const BatchHashSegment,segment_count:u32,active:u32,bank:u32,runs:*mut BatchHashRun,run_count:*mut u32,run_capacity:u32,accepted:*mut u64,spill:*mut u32,spill_count:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,scratch:*mut c_void,scratch_bytes:usize,lookup_start:*mut c_void,lookup_end:*mut c_void,misses:*mut u64,stream:*mut c_void,view:*const HashSeriesDirectory,max_series:u32)->i32;
 fn mgbfs_hash_extent_remap_v1(refs:*mut c_void,rows:u32,ref_bytes:u32,row_bits:u32,remap:*const u32,segments:u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_hash_batch_compact_scratch_bytes(entries:u32,bytes:*mut usize)->i32;
 fn mgbfs_hash_batch_compact_tail_v2(state_bytes:u32,elements:u32,segments:*const BatchHashSegment,segment_count:u32,runs:*mut BatchHashRun,count:*mut u32,max_entries:u32,scratch:*mut c_void,scratch_bytes:usize,result:*mut u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_accept_batch_scratch_bytes(selected:u32,bytes:*mut usize)->i32;
 fn mgbfs_bitmap_accept_batch(state_bytes:u32,packed:u32,elements:u32,local:*const c_void,remote:*const c_void,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,owner:u32,total:u64,owner_len:u64,bits:*mut u64,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,raw_begin:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,spill:*mut u32,spill_count:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,scratch:*mut c_void,scratch_bytes:usize,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_seed(state_bytes:u32,input:*const c_void,elements:u32,active:SegmentDescriptor,bits:*mut u64,local:u64,accepted:*mut u64,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_accept(state_bytes:u32,packed:u32,elements:u32,local:*const c_void,remote:*const c_void,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,owner:u32,total:u64,owner_len:u64,bits:*mut u64,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,spill:*mut u32,spill_count:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_accept_grouped(state_bytes:u32,packed:u32,elements:u32,local:*const c_void,remote:*const c_void,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,owner:u32,total:u64,owner_len:u64,bits:*mut u64,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,spill:*mut u32,spill_count:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_generic_regenerate_all_routes_count_storage_wide_stride(state_bytes:u32,kind:u32,elements:u32,n:u32,m:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,matrices:*const i64,moduli:*const u32,source:u32,begin:u64,requests:*const GenericRouteRecord,counts:*const u32,capacity:u32,queues:u32,output:*mut c_void,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_nccl_exchange_bounded_triplets(comm:*mut c_void,rank:u32,world:u32,shards:u32,capacity:u32,used:u32,payload:u64,state_bytes:u32,counts:*const c_void,records:*const c_void,states:*const c_void,recv_counts:*mut c_void,recv_records:*mut c_void,recv_states:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_nccl_exchange_exact_payload(comm:*mut c_void,rank:u32,world:u32,shards:u32,capacity:u32,payload:u64,state_bytes:u32,send_counts:*const u32,recv_counts:*const u32,records:*const c_void,states:*const c_void,recv_records:*mut c_void,recv_states:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_u8(input:*const u8,hashes:*const u64,n:u32,elements:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_i64(input:*const i64,hashes:*const u64,n:u32,elements:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_routes_limited_u8(elements:u32,local:*const u8,remote:*const u8,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,visible:u32,packed:u32,selection:*const u32,selected:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,spill:*mut u32,sc:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_routes_limited_grouped_u8(elements:u32,local:*const u8,remote:*const u8,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,visible:u32,packed:u32,selection:*const u32,selected:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,spill:*mut u32,sc:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_routes_limited_i64(elements:u32,local:*const i64,remote:*const i64,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,spill:*mut u32,sc:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_routes_limited_grouped_i64(elements:u32,local:*const i64,remote:*const i64,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,spill:*mut u32,sc:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_segmented_retire(seg:SegmentDescriptor,bank:u32,count:u64,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_rehash_u8(width:u32,seg:SegmentDescriptor,c0:u64,c1:u64,c2:u64,seed:u64,bits:u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_rehash_i64(width:u32,seg:SegmentDescriptor,c0:u64,c1:u64,c2:u64,seed:u64,bits:u32,error:*mut u32,stream:*mut c_void)->i32;
}
struct RawBuffer{ptr:*mut c_void,bytes:usize,device:i32,host:*mut c_void}
impl RawBuffer{
 fn try_new(bytes:usize,device:i32)->Result<Option<Self>>{let mut p=ptr::null_mut();let bytes=bytes.max(1);let code=unsafe{cudaMalloc(&mut p,bytes)};if code==2{
  // Only classify allocation failure after prior stream work has been drained.
  // Clear the same runtime/thread's last error, but never hide a different one.
  let last=unsafe{cudaGetLastError()};if last!=0&&last!=2{return Err(format!("SEGMENTED_MALLOC_LAST_ERROR_{last}"))}return Ok(None)
 }crate::generic_native::checked!(cudaMalloc,code)?;Ok(Some(Self{ptr:p,bytes,device,host:ptr::null_mut()}))}
 fn try_mapped(bytes:usize,device:i32)->Result<Option<Self>>{
  use mgbfs_cuda::native_owner::{cudaFreeHost,cudaHostAlloc,cudaHostGetDevicePointer};
  let mut host=ptr::null_mut();
  // cudaHostAllocMapped. Keep the host address for the corresponding free.
  let code=unsafe{cudaHostAlloc(&mut host,bytes,2)};
  if code==2{let _=unsafe{cudaGetLastError()};return Ok(None)}
  crate::generic_native::checked!(cudaHostAlloc,code)?;
  let mut device_ptr=ptr::null_mut();
  let code=unsafe{cudaHostGetDevicePointer(&mut device_ptr,host,0)};
  if code!=0{
   unsafe{cudaFreeHost(host);}
   if matches!(code,1|801){let _=unsafe{cudaGetLastError()};return Ok(None)}
   crate::generic_native::checked!(cudaHostGetDevicePointer,code)?;
  }
  Ok(Some(Self{ptr:device_ptr,bytes,device,host}))
 }
 fn device_bytes(&self)->u64{if self.host.is_null(){self.bytes as u64}else{0}}
 fn mapped_bytes(&self)->u64{self.bytes as u64-self.device_bytes()}
 fn at<T>(&self,offset:usize)->*mut T{unsafe{self.ptr.cast::<u8>().add(offset).cast()}}
}
impl Drop for RawBuffer{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.device);if self.host.is_null(){cudaFree(self.ptr);}else{mgbfs_cuda::native_owner::cudaFreeHost(self.host);}}}}
const SEGMENT_COUNTER_BYTES: usize = 3 * std::mem::size_of::<u64>();

struct SegmentCounters { ptr: *mut c_void, bytes: usize }
impl SegmentCounters {
 fn at<T>(&self, offset: usize) -> *mut T {
  debug_assert!(offset < self.bytes);
  unsafe { self.ptr.cast::<u8>().add(offset).cast() }
 }
}

struct Segment{desc:SegmentDescriptor,state_bytes:u64,index_bytes:u64,_storage:RawBuffer,counts:SegmentCounters,parents:*mut ParentLink,banks:[u64;3],retired:u64,device_bytes:u64}
impl Segment{
 fn reset_hash_extent(&mut self,stream:*mut c_void)->Result<()>{
  debug_assert_eq!(self.banks,[0;3]);
  if self.desc.slots_count!=0{
   let slot_bytes=usize::try_from(self.desc.slots_count.checked_mul(8).ok_or("SEGMENT_ALLOCATION_RANGE")?)
    .map_err(|_|"SEGMENT_ALLOCATION_RANGE")?;
   crate::generic_native::checked!(cudaMemsetAsync,unsafe{
    cudaMemsetAsync(self.desc.slots.cast(),255,slot_bytes,stream)
   })?;
  }
  crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(self.counts.ptr,0,self.counts.bytes,stream)})?;
  self.retired=0;
  Ok(())
 }
 fn try_new(layout:SegmentLayout,device:i32,stream:*mut c_void,sorted:bool,path:bool)->Result<Option<Self>>{
  let total=usize::try_from(layout.device_bytes).map_err(|_|"SEGMENT_ALLOCATION_RANGE")?;
  let state_bytes=usize::try_from(layout.state_bytes).map_err(|_|"SEGMENT_ALLOCATION_RANGE")?;
  let position_bytes=usize::try_from(layout.position_bytes).map_err(|_|"SEGMENT_ALLOCATION_RANGE")?;
  let parent_bytes=if path{
   layout.rows_per_bank.checked_mul(u64::from(layout.banks))
    .and_then(|rows|rows.checked_mul(PATH_PARENT_BYTES))
    .and_then(|bytes|usize::try_from(bytes).ok()).ok_or("SEGMENT_ALLOCATION_RANGE")?
  }else{0};
  let slot_bytes=total.checked_sub(SEGMENT_COUNTER_BYTES)
   .and_then(|n|n.checked_sub(parent_bytes))
   .and_then(|n|n.checked_sub(state_bytes)).and_then(|n|n.checked_sub(position_bytes))
   .ok_or("SEGMENT_ALLOCATION_RANGE")?;
  let slot_offset=SEGMENT_COUNTER_BYTES+parent_bytes;
  let data_offset=slot_offset+slot_bytes;
  // Place i64/Pack4 states on the 8-byte boundary. A state extent with
  // an unaligned length contains only byte elements, so it can go last.
  let (state_offset,position_offset)=if state_bytes%8==0{
   (data_offset,data_offset+state_bytes)
  }else{
   (data_offset+position_bytes,data_offset)
  };
  let Some(storage)=RawBuffer::try_new(total,device)? else{return Ok(None)};
  let counts=SegmentCounters{ptr:storage.ptr,bytes:SEGMENT_COUNTER_BYTES};
  let slots: *mut u64=if slot_bytes==0{ptr::null_mut()}else{storage.at(slot_offset)};
  let parents=if path{storage.at(SEGMENT_COUNTER_BYTES)}else{ptr::null_mut()};
  let positions=if position_bytes==0{ptr::null_mut()}else{storage.at(position_offset)};
  if !sorted && slot_bytes!=0{
   crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(slots.cast(),255,slot_bytes,stream)})?;
  }
  crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(counts.ptr,0,counts.bytes,stream)})?;
  let desc=SegmentDescriptor{slots,states:storage.at(state_offset),positions,slots_count:layout.slots,capacity:layout.rows_per_bank,banks:layout.banks,position_item_bytes:layout.position_item_bytes};
  Ok(Some(Self{desc,state_bytes:layout.state_bytes,index_bytes:(slot_bytes+position_bytes) as u64,_storage:storage,counts,parents,banks:[0;3],retired:0,device_bytes:layout.device_bytes}))
 }
}
struct Queue{records:Buffer,counts:Buffer,states:Buffer}
impl Queue{fn new(p:&Plan,d:i32)->Result<Self>{let n=p.world as usize*p.shards as usize;let q=p.queue_capacity as usize;Ok(Self{records:Buffer::new(n*q*32,d)?,counts:Buffer::new(n*4,d)?,states:Buffer::new(n*q*p.queue_payload_bytes(),d)?})}}
impl Queue{
 fn control_only(counts:usize,device:i32)->Result<Self>{
  Ok(Self{records:Buffer::new(0,device)?,states:Buffer::new(0,device)?,counts:Buffer::new(counts,device)?})
 }
 fn side(&self)->QueueSide{QueueSide{records:self.records.ptr,states:self.states.ptr,counts:self.counts.ptr.cast()}}
 fn compact(plan:&Plan,rows:u32,device:i32)->Result<Self>{
  let rows=rows as usize;
  let counts=plan.world as usize*plan.shards as usize*std::mem::size_of::<u32>()+PACKET_ROUTE_STATUS_BYTES;
  let payload=rows.checked_mul(plan.queue_payload_bytes()).ok_or("SEGMENT_PACKET_PRODUCER_SIZE")?;
  Ok(Self{records:Buffer::new(rows.checked_mul(32).ok_or("SEGMENT_PACKET_PRODUCER_SIZE")?,device)?,
   states:Buffer::new(payload,device)?,counts:Buffer::new(counts,device)?})
 }
 fn slice(&self,queue:usize,row_offset:usize,plan:&Plan)->QueueSide{
  QueueSide{
   records:self.records.at(row_offset*32),
   states:self.states.at(row_offset*plan.queue_payload_bytes()),
   counts:self.counts.at(queue*4),
  }
 }
}

#[derive(Clone,Copy)]
struct QueueSide{records:*mut c_void,states:*mut c_void,counts:*mut u32}

enum OwnerInputs<'a>{
 Routed{local:QueueSide,remote:QueueSide,shards:&'a[usize],visible:u32},
 Compact{shards:&'a[PacketOwnerShard],config:PacketConfig},
}

type BitmapPacketOwnerFn=unsafe extern "C" fn(*const BitmapPacketOwnerArgs,u32,*mut c_void)->i32;
type BitmapPacketBatchFn=unsafe extern "C" fn(*const BitmapPacketOwnerArgs,*mut c_void)->i32;
type Pack4PacketOwnerFn=unsafe extern "C" fn(Pack4OwnerArgs,*mut c_void)->i32;
type CompactXorRouteFn=unsafe extern "C" fn(
 u32,u32,BitmapDeviceDomain,u32,u32,u32,u32,u32,u32,*const c_void,u32,u64,
 *const u32,*const i64,*const u32,u64,u32,u32,u32,u32,u32,u64,*const u32,*const u64,
 *mut GenericRouteRecord,*mut u32,*mut u32,*mut c_void,*mut c_void,usize,*mut c_void,u32,u64,u32,*const u64
)->i32;

#[repr(C)]
struct BitmapPacketOwnerArgs{
 state_bytes:u32,packed:u32,elements:u32,
 states:*const c_void,records:*const GenericRouteRecord,counts:*const u32,
 world:u32,owner:u32,shard:u32,shards:u32,total:u64,owner_len:u64,bits:*mut u64,
 capacity:u32,visible:u32,selection:*const u32,selected:u32,raw_begin:u32,
 active:SegmentDescriptor,bank:u32,accepted:*mut u64,spill:*mut u32,spill_count:*mut u32,
 error:*mut u32,global_count:*mut u64,limit:u64,scratch:*mut c_void,scratch_bytes:usize,
}

type ByteWireOwnerFn=unsafe extern "C" fn(u32,u32,u32,*const c_void,u32,*const u32,u32,
 *const SegmentDescriptor,u32,SegmentDescriptor,u32,*mut u64,*mut u32,*mut u32,*mut u32,*mut u64,u64,*mut c_void)->i32;
#[derive(Clone,Copy,Default,serde::Serialize)]
struct SyncCounts{
 nccl_reduce_votes:u64,
 nccl_round_gathers:u64,
 nccl_layer_count_gathers:u64,
 legacy_chunk_votes:u64,
 legacy_chunk_count_exchanges:u64,
 legacy_chunk_payload_submissions:u64,
 packet_hub_layer_commits:u64,
}

impl SyncCounts {
 fn since(self, start: Self) -> Self {
  Self {
   nccl_reduce_votes: self.nccl_reduce_votes - start.nccl_reduce_votes,
   nccl_round_gathers: self.nccl_round_gathers - start.nccl_round_gathers,
   nccl_layer_count_gathers: self.nccl_layer_count_gathers - start.nccl_layer_count_gathers,
   legacy_chunk_votes: self.legacy_chunk_votes - start.legacy_chunk_votes,
   legacy_chunk_count_exchanges: self.legacy_chunk_count_exchanges - start.legacy_chunk_count_exchanges,
   legacy_chunk_payload_submissions: self.legacy_chunk_payload_submissions - start.legacy_chunk_payload_submissions,
   packet_hub_layer_commits: self.packet_hub_layer_commits - start.packet_hub_layer_commits,
  }
 }
}

// Pinned metadata lets each lane enqueue its D2H copies before the host
// waits, avoiding synchronous default-stream copies while routing overlaps.
struct PinnedStatus{ptr:*mut u32}
impl PinnedStatus{
 fn try_new(bytes:usize)->Result<Option<Self>>{
  if std::env::var("MGBFS_SEGMENT_TEST_NO_PINNED").as_deref()==Ok("1"){return Ok(None)}
  let mut p=ptr::null_mut();
  let code=unsafe{mgbfs_cuda::native_owner::cudaHostAlloc(&mut p,bytes,0)};
  if code==2||code==801{
   let last=unsafe{cudaGetLastError()};if last!=0&&last!=code{return Err(format!("SEGMENTED_PIN_LAST_ERROR_{last}"))}
   eprintln!("SEGMENTED_PINNED_METADATA_UNAVAILABLE code={code}");return Ok(None)
  }
  crate::generic_native::checked!(cudaHostAlloc,code)?;Ok(Some(Self{ptr:p.cast()}))
 }
}
impl Drop for PinnedStatus{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaFreeHost(self.ptr.cast());}}}
// A lane owns its stream, control word and ping-pong spill queues until its
// current shard is fully accepted. Two lanes must never target the same shard.
// Optional CUDA-event diagnostics. Normal jobs allocate no timing events.
const HASH_MISS_BUCKETS: usize = 33;
const HASH_EXACT_MISS_LIMIT: usize = 64;
const HASH_EXACT_MISS_BUCKETS: usize = HASH_EXACT_MISS_LIMIT + 2;
type ProfiledHashAccept = unsafe extern "C" fn(
 u32,u32,u32,*const c_void,*const c_void,*const GenericRouteRecord,*const GenericRouteRecord,
 *const u32,*const u32,u32,u32,u32,u32,u32,u32,*const u32,u32,u32,*const BatchHashSegment,
 u32,u32,u32,*mut BatchHashRun,*mut u32,u32,*mut u64,*mut u32,*mut u32,*mut u32,*mut u64,u64,
 *mut c_void,usize,*mut c_void,*mut c_void,*mut u64,*mut c_void,*const HashSeriesDirectory,u32,
)->i32;

fn exact_hash_miss_kernel() -> Result<ProfiledHashAccept>{
 let symbol=unsafe{libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_hash_batch_accept_exact_miss_v1\0".as_ptr().cast())};
 if symbol.is_null(){return Err("SEGMENTED_HASH_EXACT_MISS_CUDA_ABI_UNAVAILABLE".into());}
 Ok(unsafe{std::mem::transmute::<*mut c_void,ProfiledHashAccept>(symbol)})
}

struct HashTiming{events:[*mut c_void;6],device:i32,misses:Buffer,exact:Option<ProfiledHashAccept>}
impl HashTiming{
 fn new(device:i32,exact:Option<ProfiledHashAccept>)->Result<Self>{
  let bins=if exact.is_some(){HASH_EXACT_MISS_BUCKETS}else{HASH_MISS_BUCKETS};
  let misses=Buffer::new(bins*size_of::<u64>(),device)?;
  misses.upload(&[0u64;HASH_EXACT_MISS_BUCKETS][..bins])?;
  let mut t=Self{events:[ptr::null_mut();6],device,misses,exact};
  for e in &mut t.events{crate::generic_native::checked!(cudaEventCreateWithFlags,unsafe{cudaEventCreateWithFlags(e,0)})?;}
  Ok(t)
 }
 fn record(&self,i:usize,stream:*mut c_void)->Result<()>{crate::generic_native::checked!(cudaEventRecord,unsafe{cudaEventRecord(self.events[i],stream)})}
 fn elapsed(&self,a:usize,b:usize)->Result<f64>{let mut ms=0f32;crate::generic_native::checked!(cudaEventElapsedTime,unsafe{cudaEventElapsedTime(&mut ms,self.events[a],self.events[b])})?;Ok(f64::from(ms))}
}
impl Drop for HashTiming{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.device);for e in self.events{if !e.is_null(){cudaEventDestroy(e);}}}}}
struct Lane{stream:Stream,control:Buffer,spill:[Buffer;2],host:Option<PinnedStatus>,scratch:Option<RawBuffer>,timing:Option<HashTiming>}
impl Lane{fn new(device:i32,bound:usize,scratch_bytes:usize,enable_hash_timing:bool,exact_misses:Option<ProfiledHashAccept>)->Result<Self>{
 let bytes=bound.checked_mul(4).ok_or("SEGMENTED_SPILL_SIZE")?;
 let scratch=if scratch_bytes==0{None}else{Some(RawBuffer::try_new(scratch_bytes,device)?.ok_or("SEGMENTED_BATCH_LANE_SCRATCH_OOM")?)};
 Ok(Self{stream:Stream::new(device)?,control:Buffer::new(16,device)?,spill:[Buffer::new(bytes,device)?,Buffer::new(bytes,device)?],host:PinnedStatus::try_new(24)?,scratch,timing:if enable_hash_timing{Some(HashTiming::new(device,exact_misses)?)}else{None}})
}}
struct OwnerProgress{shard:usize,pass:usize,round:usize,selected:u32,cap_probe:bool,grow_runs:bool,raw_begin:u32,raw:RawTileCursor}
#[repr(C)] #[derive(Clone,Copy)]
pub(crate) struct BitmapDeviceDomain{symbols:*const i64,counts:*const u32,distinct:u32,elements:u32,total:u64}
impl BitmapDeviceDomain {
 pub(crate) const EMPTY: Self=Self{symbols:ptr::null(),counts:ptr::null(),distinct:0,elements:0,total:0};
}
// The packed path is opt-in and capability-gated; no old library is allowed
// to silently interpret these structs. Sizes and ABI version are checked.
#[repr(C)] #[derive(Clone,Copy)]
struct Pack4RouteArgs {
 dictionary:BitmapPack4Dictionary,domain:BitmapDeviceDomain,
 parents:*const u64,permutations:*const u32,
 count:u32,generators:u32,world:u32,source:u32,shards:u32,capacity:u32,
 state_bytes:u32,grouped:u32,map:*const u32,records:*mut GenericRouteRecord,payload:*mut c_void,
 counts:*mut u32,error:*mut u32,owner_page:u32,owner_boundary:u64,
}
#[repr(C)] #[derive(Clone,Copy)]
struct Pack4OwnerArgs {
 dictionary:BitmapPack4Dictionary,elements:u32,state_bytes:u32,packed:u32,
 local:*const c_void,remote:*const c_void,
 local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
 local_counts:*const u32,remote_counts:*const u32,
 rank:u32,world:u32,owner:u32,total:u64,owner_len:u64,bits:*mut u64,
 shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,
 capacity:u64,banks:u32,bank:u32,rows:*mut u64,accepted:*mut u64,
 global_count:*mut u64,global_limit:u64,spill:*mut u32,spill_count:*mut u32,error:*mut u32,
}
#[repr(C)] #[derive(Clone,Copy)]
struct Pack4SeedArgs {
 dictionary:BitmapPack4Dictionary,input:*const i64,
 row:*mut u64,bits:*mut u64,accepted:*mut u64,error:*mut u32,local:u64,owner_len:u64,
}
type Pack4RouteFn=unsafe extern "C" fn(Pack4RouteArgs,*mut c_void)->i32;
type Pack4OwnerFn=unsafe extern "C" fn(Pack4OwnerArgs,*mut c_void)->i32;
type Pack4SeedFn=unsafe extern "C" fn(Pack4SeedArgs,*mut c_void)->i32;
#[derive(Clone,Copy)]
struct Pack4Fns{route:Pack4RouteFn,owner:Pack4OwnerFn,seed:Pack4SeedFn}
impl Pack4Fns{
 unsafe fn load()->Option<Self>{
  let query=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_pack4_bfs_abi_query\0".as_ptr().cast());
  let route=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_pack4_route\0".as_ptr().cast());
  let owner=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_pack4_owner_accept\0".as_ptr().cast());
  let seed=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_pack4_seed\0".as_ptr().cast());
  if query.is_null()||route.is_null()||owner.is_null()||seed.is_null(){return None}
  let query:unsafe extern "C" fn(u32)->u32=std::mem::transmute(query);
  let expected=[crate::segmented_history::BITMAP_PACK4_ABI_VERSION as usize,std::mem::size_of::<BitmapPack4Dictionary>(),std::mem::size_of::<Pack4RouteArgs>(),std::mem::size_of::<Pack4OwnerArgs>(),std::mem::size_of::<Pack4SeedArgs>()];
  if expected.iter().enumerate().any(|(i,&v)|query(i as u32) as usize!=v)||query(999)!=0{return None}
  Some(Self{route:std::mem::transmute(route),owner:std::mem::transmute(owner),seed:std::mem::transmute(seed)})
 }
}
#[derive(Clone,Copy)]
struct Pack4State{dictionary:BitmapPack4Dictionary,fns:Pack4Fns}
pub(crate) fn pack4_capability()->bool{unsafe{Pack4Fns::load().is_some()}}

pub(crate) fn nearest_abi_ready() -> bool {
    NearestFns::load().is_some()
}
pub(crate) fn bidirectional_nearest_abi_ready() -> bool {
    nearest::bidirectional_abi_ready()
}
fn decode_pack4(d:BitmapPack4Dictionary,word:u64)->Result<Vec<i64>>{
 let width=BitmapPack4Dictionary::width(d.elements,d.distinct);
 if d.elements==0||d.elements>64||d.distinct==0||d.distinct>16||d.distinct>d.elements||d.elements*width>64{return Err("BITMAP_PACK4_DICTIONARY".into())}
 if d.elements*width<64&&(word>>(width*d.elements))!=0{return Err("BITMAP_PACK4_NONZERO_PADDING".into())}
 let mut out=Vec::with_capacity(d.elements as usize);
 for e in 0..d.elements{let i=((word>>(width*e))&((1<<width)-1)) as usize;if i>=d.distinct as usize{return Err("BITMAP_PACK4_BAD_ORDINAL".into())}out.push(d.symbols[i]);}
 Ok(out)
}
type BitmapRouteV2=unsafe extern "C" fn(state_bytes:u32,domain:BitmapDeviceDomain,elements:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut c_void,layout_page:u32,layout_boundary:u64)->i32;
type BatchRouteV2=unsafe extern "C" fn(bitmap:u32,word_shard:u32,domain:BitmapDeviceDomain,state_bytes:u32,kind:u32,elements:u32,n:u32,m:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,matrices:*const i64,moduli:*const u32,seed:u64,bits:u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,cuts:*const u64,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,states:*mut c_void,scratch:*mut c_void,scratch_bytes:usize,stream:*mut c_void,layout_page:u32,layout_boundary:u64)->i32;
#[derive(Clone,Copy)]
struct BitmapRouteFns{normal:BitmapRouteV2,grouped:BitmapRouteV2,batch:BatchRouteV2}
impl BitmapRouteFns{unsafe fn load()->Option<Self>{
 let normal=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_bitmap_route_v2\0".as_ptr().cast());
 let grouped=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_bitmap_route_grouped_v2\0".as_ptr().cast());
 let batch=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_segmented_batch_route_v2\0".as_ptr().cast());
 if normal.is_null()||grouped.is_null()||batch.is_null(){return None}
 Some(Self{normal:std::mem::transmute(normal),grouped:std::mem::transmute(grouped),batch:std::mem::transmute(batch)})
}}
struct BitmapState{domain:BitmapDeviceDomain,_symbols:Buffer,_counts:Buffer,bits:RawBuffer,owner:u32,owner_len:u64,owner_page:u32,owner_boundary:u64,immutable_routes:Option<BitmapRouteFns>,xor:Option<(CompactXorRouteFn,Buffer)>}
struct DescriptorCache{data:RawBuffer,count:usize}
struct BatchHashShard{snapshots:Vec<HashSeriesSnapshot>,snapshot_rebuilds:u64,snapshot_retirements:u64,snapshot_evicted_for_rebuild:u64,snapshot_deferred:u64,snapshot_rebuild_seconds:f64,snapshot_deferred_seconds:f64,lookup_max_series:u32,dir:RawBuffer,capacity:u32,host_count:u32,view:Option<DescriptorCache>,max_runs:u32,run_histogram:std::collections::BTreeMap<u32,u64>,timed_max_runs:u32,timed_run_histogram:std::collections::BTreeMap<u32,u64>,deferred_merges:u64,merged_runs:u64}
impl BatchHashShard{
 fn replaced_runs(&self)->u32{self.snapshots.iter().map(|snapshot|snapshot.view.replaced_runs).sum()}
 fn searched_series(&self)->u32{self.host_count-self.replaced_runs()+self.snapshots.len() as u32}
 fn snapshot_directory(&self)->HashSeriesDirectory{
  let empty=HashSeriesView{keys:ptr::null(),references:ptr::null(),prefix_offsets:ptr::null(),
   length:0,replaced_runs:0,reference_bytes:0,row_bits:0};
  let mut directory=HashSeriesDirectory{series:[empty;3],count:self.snapshots.len() as u32,replaced_runs:self.replaced_runs()};
  for (dst,snapshot) in directory.series.iter_mut().zip(&self.snapshots){*dst=snapshot.view;}
  directory
 }
 fn count_ptr(&self)->*mut u32{self.dir.at(self.capacity as usize*std::mem::size_of::<BatchHashRun>())}
 fn observe(&mut self,n:u32,timed:bool){
  self.host_count=n;self.max_runs=self.max_runs.max(n);*self.run_histogram.entry(n).or_insert(0)+=1;
  if timed{self.timed_max_runs=self.timed_max_runs.max(n);*self.timed_run_histogram.entry(n).or_insert(0)+=1;}
 }
 fn median_of(hist:&std::collections::BTreeMap<u32,u64>)->Option<u32>{let count=hist.values().copied().sum::<u64>();if count==0{return None}let mut seen=0u64;for(&n,&v)in hist{seen+=v;if seen>=(count+1)/2{return Some(n)}}None}
 fn median(&self)->Option<u32>{Self::median_of(&self.run_histogram)}
 fn timed_median(&self)->Option<u32>{Self::median_of(&self.timed_run_histogram)}
}
impl Segment{fn sorted_view(&self)->BatchHashSegment{BatchHashSegment{states:self.desc.states,keys:self.desc.slots,rows:self.desc.positions.cast(),capacity:self.desc.capacity,banks:self.desc.banks}}}
struct OwnerPrepared{sealed:*const SegmentDescriptor,descriptor_count:u32,hash_segments:*const BatchHashSegment,hash_count:u32,hash_runs:*mut BatchHashRun,hash_run_count:*mut u32,hash_run_capacity:u32}
// Diagnostic only: events measure stream intervals around the route and first
// owner kernel. Normal execution creates no timing events and adds no waits.
struct OverlapProbe{events:[*mut c_void;5],device:i32,owner_recorded:bool}
impl OverlapProbe{
 fn new(device:i32)->Result<Self>{
  let mut v=Self{events:[ptr::null_mut();5],device,owner_recorded:false};
  for e in &mut v.events{crate::generic_native::checked!(cudaEventCreateWithFlags,unsafe{cudaEventCreateWithFlags(e,0)})?;}
  Ok(v)
 }
 fn start_route(&self,stream:*mut c_void)->Result<()>{
  for i in [0,1]{crate::generic_native::checked!(cudaEventRecord,unsafe{cudaEventRecord(self.events[i],stream)})?;}Ok(())
 }
 fn end_route(&self,stream:*mut c_void)->Result<()>{crate::generic_native::checked!(cudaEventRecord,unsafe{cudaEventRecord(self.events[2],stream)})}
 fn start_owner(&self,stream:*mut c_void)->Result<()>{
  crate::generic_native::checked!(cudaStreamWaitEvent,unsafe{cudaStreamWaitEvent(stream,self.events[0],0)})?;
  crate::generic_native::checked!(cudaEventRecord,unsafe{cudaEventRecord(self.events[3],stream)})
 }
 fn end_owner(&mut self,stream:*mut c_void)->Result<()>{
  crate::generic_native::checked!(cudaEventRecord,unsafe{cudaEventRecord(self.events[4],stream)})?;self.owner_recorded=true;Ok(())
 }
 fn report(&self,rank:u32,depth:u64)->Result<()>{
  if !self.owner_recorded{return Ok(())}
  for i in [2,4]{crate::generic_native::checked!(cudaEventSynchronize,unsafe{cudaEventSynchronize(self.events[i])})?;}
  let mut v=[0f32;4];for i in 0..4{crate::generic_native::checked!(cudaEventElapsedTime,unsafe{cudaEventElapsedTime(&mut v[i],self.events[0],self.events[i+1])})?;}
  let overlap=(v[1].min(v[3])-v[0].max(v[2])).max(0.);
  eprintln!("SEGMENT_OVERLAP rank={rank} depth={depth} route_start_ms={} route_end_ms={} owner_start_ms={} owner_end_ms={} overlap_ms={}",v[0],v[1],v[2],v[3],overlap);Ok(())
 }
}
impl Drop for OverlapProbe{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.device);for e in self.events{if !e.is_null(){cudaEventDestroy(e);}}}}}
struct Comm{ptr:*mut c_void,healthy:Cell<bool>,aborted:Cell<bool>}
impl Comm{
 fn abort_once(&self){if !self.aborted.get()&&!self.ptr.is_null(){crate::session_cache::permit_comm(false);let status=unsafe{mgbfs_nccl_abort(self.ptr)};self.aborted.set(true);if status!=0{eprintln!("SEGMENTED_COMM_ABORT_ERROR code={status}");}}}
}
impl Drop for Comm{fn drop(&mut self){unsafe{if !self.ptr.is_null(){if !self.healthy.get()||crate::native_fault::failed(){self.abort_once();}if self.aborted.get()||!crate::session_cache::comm_put(self.ptr){mgbfs_nccl_destroy(self.ptr);}}}}}
#[derive(Clone,Copy)]struct Range{shard:usize,segment:usize,bank:u32,begin:u64,end:u64,logical_start:u64}
fn mix(mut x:u64)->u64{x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^(x>>31)}

// Only quiescent, unclaimed session extents are released. Reclaim idle CUDA
// pool allocations before admitting segment geometry, so a previous job
// cannot make reusable memory appear unavailable to the next graph.
fn segment_sizing_memory_info()->Result<(usize,usize)>{
 crate::session_cache::release_unused_buffers();
 crate::generic_pool::trim_idle()?;
 let(mut free,mut total)=(0usize,0usize);
 crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
 Ok((free,total))
}

pub struct SegmentedDistributedBfs{
 nearest:Option<NearestState>,
 ownership:Option<PathOwnership>,
 window:Option<WindowState>,
 strategy:SegmentStrategy,compact_positions:bool,pack4:Option<Pack4State>,previous_snapshot:Option<Buffer>,previous_snapshot_rows:usize,previous_snapshot_valid:bool,previous_global:u64,base:GenericNativeBfs,plan:Plan,rank:u32,seed:u64,bits:u32,layout:SegmentLayout,bitmap:Option<BitmapState>,
 segments:Vec<Vec<Segment>>,reuse_hash_segments:bool,spare_hash_segment:Option<Segment>,hash_segment_reuses:u64,descriptor_cache:Vec<Option<DescriptorCache>>,hash_shards:Vec<Option<BatchHashShard>>,remaining_bytes:u64,storage_budget:u64,storage_peak_requested:u64,reserve_bytes:u64,configured_rows:u64,bank_totals:[u64;3],peak_live_rows:u64,last_resource:Option<serde_json::Value>,capacity_limit:Option<u64>,current:Vec<Range>,previous:Vec<Range>,
 counts:Vec<u64>,depth:u64,source_retries:u64,active_batch:u32,queue_batch_ceiling:u32,admitted_batch:u32,batch_recovery:bool,batch_recovery_policy:crate::segmented_strategy::QueueBatchRecovery,batch_recoveries:u64,layer_stats:bool,layer_stats_low:u64,layer_stats_high:u64,profile_spill:u64,profile_owner_passes:u64,peak_owner_lanes:u32,prefetched_chunks:u64,overlap_samples:u32,
 admitted_owner_cuts:Option<Vec<u64>>,
 initial_hash_extent_rows:Option<u64>,
 packet:Option<PacketRuntime>,
 global_sync:std::cell::Cell<SyncCounts>,
 last_layer_sync:SyncCounts,
 remote_payload_transfers:u64,
 hash_reclaimed_segments:u64,
 hash_reclaimed_device_bytes:u64,
 send:[Queue;2],recv:Queue,route_stream:Stream,route_scratch:Option<RawBuffer>,batch_offsets:Option<Buffer>,batch_offsets_host:Vec<u32>,batch_dense_origins:Option<crate::segmented_strategy::DenseOriginsFn>,batch_tile:u32,batch_merge_rows:u32,batch_merge_enabled:bool,hash_lookup_limit:Option<u32>,hash_eager_tail:Option<u32>,hash_timing_bounds:Option<(u64,u64)>,hash_owner_ms:f64,hash_lookup_ms:f64,hash_merge_ms:f64,hash_lookup_directory_runs:HashLookupDirectorySeries,hash_snapshot_timed_lookups:u64,hash_snapshot_installations:u64,hash_snapshot_mapped_host_installations:u64,hash_snapshot_rebuild_peak_bytes:u64,hash_snapshot_storage_evictions:u64,input_stats_bounds:Option<(u64,u64)>,input_valid:u64,input_raw_elements:u64,input_raw_tiles:u64,input_spill_elements:u64,input_spill_tiles:u64,input_empty_self:u64,input_empty_peer:u64,route_errors:[Buffer;2],control:Buffer,map:Buffer,cuts:Option<Buffer>,lanes:Vec<Lane>,comm:Rc<Comm>,test_oom_done:bool,bounds_trace:bool,overlap_bounds:Option<(u64,u64)>,exact_exchange:bool,transport_trace:bool,remote_records_sent:u64,remote_padded_records:u64,test_count_payload:bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArenaPurpose {
    Traversal,
    Nearest,
    Window,
}

impl SegmentedDistributedBfs{
 pub fn new(g:&GraphDefinitionV2,device:u32,rank:u32,p:Plan,id:&[u8;128],packet_job_token:Option<PacketJobToken>,seed:u64,bits:u32,cuts:Option<&[u64]>,ready:&crate::segmented_host_ready::HostAgreement,path:bool)->Result<Self>{
  Self::new_with_roots(g,device,rank,p,id,packet_job_token,seed,bits,cuts,ready,path,RootSpec::Single)
 }
 fn new_with_roots(g:&GraphDefinitionV2,device:u32,rank:u32,p:Plan,id:&[u8;128],packet_job_token:Option<PacketJobToken>,seed:u64,bits:u32,cuts:Option<&[u64]>,ready:&crate::segmented_host_ready::HostAgreement,path:bool,roots:RootSpec<'_>)->Result<Self>{
  let purpose=if path{ArenaPurpose::Nearest}else{ArenaPurpose::Traversal};
  Self::new_arena(g,device,rank,p,id,packet_job_token,seed,bits,cuts,ready,purpose,roots,false,None)
 }
 pub(crate) fn new_paired(
  graph:&GraphDefinitionV2,inverse:&GraphDefinitionV2,targets:&[Vec<i64>],
  device:u32,rank:u32,plan:Plan,id:&[u8;128],packet_job_token:Option<PacketJobToken>,
  seed:u64,bits:u32,cuts:Option<&[u64]>,ready:&crate::segmented_host_ready::HostAgreement,
 )->Result<(Self,Self)>{
  Self::new_path_pair(graph,inverse,targets,device,rank,plan,id,packet_job_token,
   seed,bits,cuts,ready,ArenaPurpose::Nearest)
 }
 fn new_path_pair(
  graph:&GraphDefinitionV2,inverse:&GraphDefinitionV2,targets:&[Vec<i64>],
  device:u32,rank:u32,plan:Plan,id:&[u8;128],packet_job_token:Option<PacketJobToken>,
  seed:u64,bits:u32,cuts:Option<&[u64]>,ready:&crate::segmented_host_ready::HostAgreement,
  purpose:ArenaPurpose,
 )->Result<(Self,Self)>{
  let uniform;
  let cuts=match cuts {
   Some(cuts)=>cuts,
   None=>{
    uniform=crate::segmented_history::weighted_owner_cuts(&vec![1;plan.world as usize]).map_err(str::to_owned)?;
    &uniform
   }
  };
  let forward=Self::new_arena(graph,device,rank,plan.clone(),id,packet_job_token,seed,bits,
   Some(cuts),ready,purpose,RootSpec::Single,true,None)?;
  let cuts=forward.admitted_owner_cuts.as_deref().unwrap_or(cuts);
  let reverse=Self::new_arena(inverse,device,rank,plan,id,packet_job_token,seed,bits,
   Some(cuts),&ready.reverse(),purpose,RootSpec::Targets(targets),true,Some(&forward))?;
  Ok((forward,reverse))
 }
 fn new_arena(g:&GraphDefinitionV2,device:u32,rank:u32,p:Plan,id:&[u8;128],packet_job_token:Option<PacketJobToken>,seed:u64,bits:u32,cuts:Option<&[u64]>,ready:&crate::segmented_host_ready::HostAgreement,purpose:ArenaPurpose,roots:RootSpec<'_>,paired:bool,primary:Option<&Self>)->Result<Self>{
  let path=purpose==ArenaPurpose::Nearest;
  let window=purpose==ArenaPurpose::Window;
  let route_provenance=purpose!=ArenaPurpose::Traversal;
  let target_roots=matches!(roots,RootSpec::Targets(_));
  let metadata_bytes=if path{PATH_PARENT_BYTES}else{0};
  let requested_path_cuts=if route_provenance{cuts.map(<[u64]>::to_vec)}else{None};
  let config:Result<_>=(||{
  p.validate(g.generator_count() as u32)?;g.validate()?;
  if unsafe{mgbfs_segmented_abi_version()}!=3{return Err("SEGMENTED_ABI_MISMATCH: expected 3".into())}
  let mut p=p;
  // Invalid values still participate in rank agreement so that one invalid
  // rank cannot leave its peers waiting inside the first data exchange.
  let strategy_code=match std::env::var("MGBFS_SEGMENT_STRATEGY"){
   Ok(v)=>SegmentStrategy::parse(Some(&v)).map(|v|v.code()).unwrap_or(3),
   Err(std::env::VarError::NotPresent)=>SegmentStrategy::Atomic.code(),
   Err(std::env::VarError::NotUnicode(_))=>3,
  };
  let recovery_code=match std::env::var("MGBFS_SEGMENT_BATCH_RECOVERY"){
   Err(std::env::VarError::NotPresent)=>u64::from(p.compact_transport),
   Ok(v) if v=="0"=>0,Ok(v) if v=="1"=>1,_=>2,
  };
  let bitmap_mode=std::env::var("MGBFS_EXPERIMENTAL_BITMAP").as_deref()==Ok("1");
  if target_roots&&!route_provenance{return Err("SEGMENTED_PATH_ROOT_LAYOUT".into())}
  if route_provenance&&(bitmap_mode||strategy_code!=SegmentStrategy::Batch.code()||p.packed_candidates()||p.history_layers!=1||!p.compact_transport||std::env::var("MGBFS_SEGMENT_EXACT_EXCHANGE").as_deref()==Ok("1")){
   return Err("SEGMENTED_PATH_OWNER_LAYOUT".into());
  }
  if window&&(p.world<2||p.compact_packet_slots!=3||p.compact_receive_slots!=3||!WindowState::abi_ready()){
   return Err("SEGMENTED_WINDOW_REQUIRES_RAW_CAROUSEL_AND_LINEAGE_ABI".into());
  }
  let pack4_requested=match std::env::var("MGBFS_SEGMENT_PACK4"){
   Err(std::env::VarError::NotPresent)=>false,Ok(v) if v=="0"=>false,Ok(v) if v=="1"=>true,
   _=>return Err("BITMAP_PACK4_INVALID".into()),
  };
  let pack4=if pack4_requested{
   if !bitmap_mode{return Err("BITMAP_PACK4_REQUIRES_BITMAP".into())}
   if strategy_code!=SegmentStrategy::Atomic.code(){return Err("BITMAP_PACK4_STRATEGY_UNSUPPORTED".into())}
   let dictionary=BitmapPack4Dictionary::from_root(&g.start).map_err(str::to_owned)?;
   let fns=unsafe{Pack4Fns::load()}.ok_or("BITMAP_PACK4_NATIVE_ABI_MISSING")?;
   Some(Pack4State{dictionary,fns})
  }else{None};
  let page_code=if !bitmap_mode{0}else{match std::env::var("MGBFS_SEGMENT_BITMAP_OWNER_PAGE"){
   Err(std::env::VarError::NotPresent)=>0u64,Ok(v) if v=="0"=>0,
   Ok(v)=>match v.parse::<u32>(){Ok(page) if page.is_power_of_two()&&page<=(1<<20)=>u64::from(page.trailing_zeros()+1),_=>63},
   Err(std::env::VarError::NotUnicode(_))=>63,
  }};
  let page_setter=crate::segmented_strategy::bitmap_owner_page_setter();
  let bitmap_orbit_abi=bitmap_mode&&crate::segmented_strategy::bitmap_orbit_abi_ready();
  let immutable_routes=if bitmap_mode&&crate::segmented_strategy::bitmap_immutable_abi_ready(){unsafe{BitmapRouteFns::load()}}else{None};
  let strategy_advertisement=strategy_code | (u64::from(crate::segmented_strategy::batch_abi_ready())<<8) | (recovery_code<<9)
    | (page_code<<11) | (u64::from(page_setter.is_some())<<17) | (u64::from(bitmap_mode)<<18) | (u64::from(bitmap_orbit_abi)<<19)
    | (u64::from(crate::segmented_strategy::warp_safe_abi_ready())<<20) | (u64::from(immutable_routes.is_some())<<21);
  let compact_positions=!bitmap_mode && std::env::var("MGBFS_SEGMENT_COMPACT_POS").as_deref()==Ok("1");
  let bitmap_banks=if bitmap_mode {match std::env::var("MGBFS_BITMAP_BANKS").as_deref(){Ok("2")=>2,Ok("3")|Err(_)=>3,_=>return Err("BITMAP_BANK_COUNT".into())}} else {3};
  if bitmap_mode{
   if !matches!(&g.action,mgbfs_core::graph_definition::GraphAction::Permutation{..}){return Err("BITMAP_REQUIRES_PERMUTATION_ACTION".into())}
   if cuts.is_some(){return Err("BITMAP_CUSTOM_OWNER_CUTS_UNSUPPORTED".into())}
   p.history_layers=3;
  }
  if let Ok(v)=std::env::var("MGBFS_SEGMENT_QUEUE_CAP"){
   let n=v.parse::<u32>().map_err(|_|"SEGMENTED_QUEUE_CAP")?;if n==0{return Err("SEGMENTED_QUEUE_CAP".into())}p.queue_capacity=p.queue_capacity.min(n);
  }
  if let Ok(v)=std::env::var("MGBFS_SEGMENT_TEST_QUEUE_CAP"){let n=v.parse::<u32>().map_err(|_|"SEGMENTED_TEST_QUEUE_CAP")?;if n==0||n>p.queue_capacity{return Err("SEGMENTED_TEST_QUEUE_CAP".into())}p.queue_capacity=n;}
  let strategy_advertisement=strategy_advertisement | (u64::from(p.queue_capacity)<<32);
  if p.history_algorithm!="HASH"||p.sort_candidates||p.parent_transport||p.generator_backend!="cuda"{return Err("SEGMENTED_UNSUPPORTED_PROFILE".into())}
  if rank>=p.world||bits>64||p.elements as usize!=g.start.len(){return Err("SEGMENTED_PLAN_SHAPE".into())}
  if !bitmap_mode&&p.history_layers==3&&!g.inverse_closed()?{return Err("SEGMENTED_ROLLING_REQUIRES_INVERSE_CLOSED".into())}
  if let Some(c)=cuts{if c.len()!=p.world as usize+1||c[0]!=0||c[c.len()-1]!=(1u64<<32)||c.windows(2).any(|x|x[0]>=x[1]){return Err("SEGMENTED_OWNER_CUTS".into())}}
  let tiny=GenericMemoryPlan::with_storage(p.elements,1,p.state_bytes)?;
   // Permutations preserve the root alphabet. Encode only a proven domain
   // and keep the entire state for the owner's collision comparison.
   let byte_packet_bits=if !route_provenance&&std::env::var("MGBFS_SEGMENT_PACKET_BINARY").as_deref()==Ok("1")
    && !bitmap_mode && p.state_bytes==1 && (25..=64).contains(&p.elements)
    && matches!(&g.action,mgbfs_core::graph_definition::GraphAction::Permutation{..}){
     if g.start.iter().all(|&value|(0..=1).contains(&value)){Some(1)}
     else if p.elements<=32&&g.start.iter().all(|&value|(0..=3).contains(&value)){Some(2)}
     else{None}
    }else{None};
   let range_base=if !route_provenance&&std::env::var("MGBFS_SEGMENT_PACKET_RANGE16").as_deref()!=Ok("0")
    && !bitmap_mode && p.state_bytes==8
    && matches!(&g.action,mgbfs_core::graph_definition::GraphAction::Permutation{..}){
    let low=*g.start.iter().min().ok_or("SEGMENTED_PLAN_SHAPE")?;
    let high=*g.start.iter().max().ok_or("SEGMENTED_PLAN_SHAPE")?;
    (high.abs_diff(low)<=u64::from(u16::MAX)).then_some(low)
   }else{None};
   let pack4_wire=pack4.is_some() && p.elements<=16 && p.packed_candidates()
    && g.start.iter().all(|&value|(0..=15).contains(&value));
   let byte_wire_requested=strategy_code<=SegmentStrategy::Warp.code()
    && match std::env::var("MGBFS_SEGMENT_PACKET_BYTE_DIRECT").as_deref(){
     Ok("1")=>true,
     Ok("0")=>false,
     _=>byte_packet_bits==Some(1),
    };
   let packet_config=PacketConfig::load(p.world,rank,bitmap_mode,pack4.is_some(),byte_packet_bits,byte_wire_requested,range_base,pack4_wire,packet_job_token)?;
   if p.compact_transport&&packet_config.is_none(){return Err("SEGMENT_PACKET_PLAN_REQUIRES_COMPACT_TRANSPORT".into())}
  let nearest_fns=if path{Some(NearestFns::load().ok_or("SEGMENTED_NEAREST_CUDA_ABI_UNAVAILABLE")?)}else{None};
  let descriptor=serde_json::json!({"path_parent_abi":nearest_fns.is_some(),"window_abi":window,"target_roots":target_roots,"world":p.world,"shards":p.shards,"elements":p.elements,"state_bytes":p.state_bytes,"batch":p.batch,"queue_capacity":p.queue_capacity,"bitmap":bitmap_mode,"bitmap_banks":bitmap_banks,"bitmap_pack4_symbols":pack4.map(|v|v.dictionary.symbols[..v.dictionary.distinct as usize].to_vec()),"strategy":strategy_advertisement,"compact_positions":compact_positions,"hash_bits":bits,"owner_cuts":cuts,"packet_route_lookahead":p.compact_route_lookahead,"packet_ring_slots":p.compact_packet_slots,"packet_receive_slots":p.compact_receive_slots,"packet_control":packet_config.map(|config|config.endpoint.advertised.to_string()),"packet_pack4":packet_config.is_some_and(|config|matches!(config.codec,Some(PacketCodec::Pack4(_)))),"packet_pack4_direct":packet_config.is_some_and(|config|config.pack4_wire_owner.is_some()),"packet_binary":packet_config.and_then(|config|match config.codec{Some(PacketCodec::Binary(bits,_))=>Some(bits+1),_=>None}),"packet_range16_base":packet_config.and_then(|config|match config.codec{Some(PacketCodec::Range16(base,_))=>Some(base),_=>None})});
   Ok((p,strategy_advertisement,bitmap_mode,page_code,page_setter,bitmap_orbit_abi,compact_positions,bitmap_banks,tiny,immutable_routes,pack4,packet_config,descriptor,nearest_fns))
  })();
   let descriptor=config.as_ref().ok().map(|x|&x.12);
  ready.agree("configuration",&config,descriptor)?;
   let(p,strategy_advertisement,bitmap_mode,page_code,page_setter,bitmap_orbit_abi,compact_positions,bitmap_banks,tiny,immutable_routes,pack4,packet_config,_,nearest_fns)=config?;
  let prepared:Result<_>=(||{crate::segmented_host_ready::injected("base")?;GenericNativeBfs::new_unseeded(g,device,tiny,seed,bits)})();
  ready.agree("base",&prepared,None)?;
  let base=prepared?;let d=base.device;let stream=base.stream.ptr;
  let comm=if let Some(primary)=primary{
   Rc::clone(&primary.comm)
  }else{
   let mut raw=crate::session_cache::comm_take().unwrap_or(ptr::null_mut());
   if raw.is_null(){let mut error=[0i8;512];crate::generic_native::checked!(mgbfs_nccl_create,unsafe{mgbfs_nccl_create(rank,p.world,device,id.as_ptr().cast(),&mut raw,error.as_mut_ptr(),error.len())})?;}
   Rc::new(Comm{ptr:raw,healthy:Cell::new(false),aborted:Cell::new(false)})
  };
  let using_packet=packet_config.is_some();
  let prepared:Result<_>=(||{
   crate::segmented_host_ready::injected("queue")?;
    let packet=if primary.is_some(){None}else{
     packet_config.map(|config|PacketRuntime::new(config,id,&p,d,base.generators)).transpose()?
    };
    let transport=packet.as_ref().or_else(||primary.and_then(|side|side.packet.as_ref()));
    let (send,recv)=if let Some(transport)=transport{
     let alternate=if transport.route_lookahead{Queue::compact(&p,transport.producer_rows,d)?}else{Queue::control_only(0,d)?};
     ([Queue::compact(&p,transport.producer_rows,d)?,alternate],Queue::control_only(p.world as usize*4,d)?)
    }else{
     let receive=if p.world==1{Queue::control_only(0,d)?}else{Queue::new(&p,d)?};
     ([Queue::new(&p,d)?,Queue::new(&p,d)?],receive)
    };
   let route_stream=Stream::new(d)?;let route_errors=[Buffer::new(12,d)?,Buffer::new(12,d)?];
   let control=Buffer::new(128+p.world as usize*16,d)?;
    let producer_rows=transport.map(|transport|transport.producer_rows);
    Ok((send,recv,route_stream,route_errors,control,packet,producer_rows))
  })();
  ready.agree("queues",&prepared,None)?;
   let(send,recv,route_stream,route_errors,control,packet,producer_rows)=prepared?;
  // Agree before seed materialization or the first graph-data exchange.
  let copied:Result<_>=(||{crate::segmented_host_ready::injected("strategy-copy")?;
   crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(control.at::<c_void>(32),(&strategy_advertisement as *const u64).cast(),8,1,stream)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;Ok(())})();
  ready.agree("strategy-copy",&copied,None)?;copied?;
  crate::generic_native::checked!(mgbfs_nccl_all_gather_u64,unsafe{mgbfs_nccl_all_gather_u64(comm.ptr,control.at(32),control.at(64),stream)})?;
  let mut strategy_codes=vec![0u64;p.world as usize];
  crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(strategy_codes.as_mut_ptr().cast(),control.at::<c_void>(64),strategy_codes.len()*8,2,stream)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
  let(strategy,batch_recovery,owner_page)=crate::segmented_strategy::agree_owner_page_geometry(&strategy_codes).map_err(str::to_owned)?;
  let bitmap_plan:Result<_>=(||{
  let bitmap_domain=if bitmap_mode{
   let generators=if let mgbfs_core::graph_definition::GraphAction::Permutation{generators,..}=&g.action{generators}else{return Err("BITMAP_REQUIRES_PERMUTATION_ACTION".into())};
   Some(if bitmap_orbit_abi{crate::permutation_bitmap::PositionOrbitDomain::new(&g.start,generators)}else{crate::permutation_bitmap::PositionOrbitDomain::full(&g.start)}.map_err(str::to_owned)?)
  }else{None};
  let page_map=if let Some(domain)=&bitmap_domain{
   Some(crate::permutation_bitmap::BitmapPageMap::new(domain.size(),p.world,owner_page).map_err(str::to_owned)?)
  }else{None};
  Ok((bitmap_domain,page_map))
  })();
  ready.agree("bitmap-plan",&bitmap_plan,None)?;
  let(bitmap_domain,page_map)=bitmap_plan?;
  if let Some(map)=page_map{
   if immutable_routes.is_none(){
    let setter_result:Result<_>=(||{
     let status=if let Some(set)=page_setter{unsafe{set(owner_page,map.boundary())}}else{0};
     if status!=0{return Err(format!("BITMAP_OWNER_PAGE_CONFIGURATION_FAILED rank={rank} code={status}"))}
     Ok(())
    })();
    ready.agree("bitmap-setter",&setter_result,None)?;
    setter_result?;
   }
  }
  let prepared:Result<_>=(||{
  crate::segmented_host_ready::injected("map")?;
  let sorted_hash=strategy==SegmentStrategy::Batch&&!bitmap_mode;
  let hash_series_setting=std::env::var("MGBFS_SEGMENT_HASH_SERIES");
  // Keep the explicitly requested unbounded path for historical A/B runs.
  // Regular sorted hash must never silently lose its lookup bound.
  let bounded_hash=sorted_hash&&hash_series_setting.as_deref()!=Ok("0");
  if route_provenance&&!bounded_hash{return Err("SEGMENTED_PATH_LOOKUP_BOUND".into())}
  if bounded_hash&&!crate::segmented_strategy::hash_series_abi_ready(){
   return Err("SEGMENT_HASH_SERIES_CUDA_ABI_UNAVAILABLE".into())
  }
  let hash_lookup_limit=if bounded_hash{
   let limit=if std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref()==Ok("1"){
    std::env::var("MGBFS_SEGMENT_TEST_MAX_LOOKUP_SERIES").ok()
     .map(|v|v.parse::<u32>().ok().filter(|&n|(4..=64).contains(&n)).ok_or("SEGMENTED_TEST_HASH_LOOKUP_LIMIT"))
     .transpose()?.unwrap_or(64)
   }else{64};
   Some(limit)
  }else{None};
  let hash_eager_tail=if let Some(limit)=hash_lookup_limit{
   let threshold=std::env::var("MGBFS_SEGMENT_HASH_EAGER_TAIL").ok()
    .map(|value|value.parse::<u32>().ok().filter(|&rows|rows==0||(2..=limit).contains(&rows))
      .ok_or("SEGMENTED_HASH_EAGER_TAIL"))
    .transpose()?;
   match threshold{
    Some(0)=>None,
    Some(rows)=>Some(rows),
    None if p.history_layers==1=>Some((limit/2).max(2)),
    None=>Some(16.min(limit)),
   }
  }else{None};
  // Allocate compact source offsets before measuring segment capacity. The
  // host table is reused across waves and is copied only after count exchange.
   let dense_input=!using_packet&&strategy==SegmentStrategy::Batch&&std::env::var("MGBFS_SEGMENT_TEST_RECTANGULAR").as_deref()!=Ok("1");
  let offset_count=(p.shards as usize).checked_mul(p.world as usize+1).ok_or("SEGMENTED_OFFSET_SIZE")?;
  let batch_offsets_host=if dense_input{vec![0u32;offset_count]}else{Vec::new()};
  let batch_offsets=if dense_input{Some(Buffer::new(offset_count.checked_mul(4).ok_or("SEGMENTED_OFFSET_SIZE")?,d)?)}else{None};
  let batch_dense_origins=if dense_input{Some(crate::segmented_strategy::dense_origins().ok_or("SEGMENT_STRATEGY_BATCH_NATIVE_ABI_MISSING")?)}else{None};
   let spill_sources=if using_packet{1}else{p.world};
   let bound=usize::try_from(u64::from(spill_sources)*u64::from(p.queue_capacity)).map_err(|_|"SEGMENTED_SPILL_SIZE")?;
  if bound>=0x80000000usize{return Err("SEGMENTED_SPILL_SIZE".into())}
  let mut h=state_hash(&g.start,seed,bits);
  if let Some(domain)=&bitmap_domain{h=page_map.ok_or("BITMAP_PAGE_MAP_MISSING")?.map(domain.rank(&g.start).map_err(str::to_owned)?).map_err(str::to_owned)?;}
  // A graph plan with an explicit legacy capacity need not have hash cuts.
  // Stage a cuts buffer anyway so external admission can replace its contents
  // with actual post-layout owner weights before any state is seeded.
  let provisional_cuts=if cuts.is_none()&&!bitmap_mode&&std::env::var("MGBFS_SEGMENT_EXTERNAL_LINK").as_deref()==Ok("1"){
   Some(crate::segmented_history::weighted_owner_cuts(&vec![1;p.world as usize]).map_err(str::to_owned)?)
  }else{None};
  let cuts=cuts.or(provisional_cuts.as_deref());
  let owner=if bitmap_mode{(h%u64::from(p.world)) as u32}else if let Some(c)=cuts{c.windows(2).position(|v|v[0]<=h>>32&&h>>32<v[1]).ok_or("SEGMENTED_ROOT_OWNER")? as u32}else{(((h>>32)*u64::from(p.world))>>32) as u32};
  let mut owner_map=(0..p.world).collect::<Vec<_>>();if cuts.is_none(){owner_map.swap(0,owner as usize);}let root=owner_map[owner as usize];
  let map=Buffer::new(owner_map.len()*4,d)?;map.upload(&owner_map)?;
  let cuts=if let Some(c)=cuts{let b=Buffer::new(c.len()*8,d)?;b.upload(c)?;Some(b)}else{None};
  // Configured segment rows are a maximum, not a demand to allocate
  // 2^32 entries even when the graph or GPU is smaller.
  let configured_rows=std::env::var("MGBFS_SEGMENT_MAX_ROWS").ok().map(|v|v.parse::<u64>().map_err(|_|"SEGMENTED_MAX_ROWS".to_owned())).transpose()?.unwrap_or(LOCAL_ROWS);
  if configured_rows==0||configured_rows>LOCAL_ROWS{return Err("SEGMENTED_MAX_ROWS".into())}
  let explicit_rows=std::env::var("MGBFS_SEGMENT_ROWS").ok().map(|v|v.parse::<u64>().map_err(|_|"SEGMENTED_ROWS".to_owned())).transpose()?;
  if explicit_rows.map_or(false,|x|x==0||x>configured_rows){return Err("SEGMENTED_ROWS".into())}
  let orbit=crate::generic_memory::state_space_bound(g).unwrap_or(LOCAL_ROWS);
  let divisor=u64::from(p.world)*u64::from(p.shards);
  let estimated=std::env::var("MGBFS_SEGMENT_EXPECTED_ROWS").ok().and_then(|v|v.parse::<u64>().ok()).filter(|&x|x>0).unwrap_or_else(||orbit.saturating_add(divisor-1)/divisor).max(32);
  Ok((sorted_hash,hash_lookup_limit,hash_eager_tail,batch_offsets_host,batch_offsets,batch_dense_origins,bound,h,owner_map,root,map,cuts,configured_rows,explicit_rows,estimated))
  })();
  ready.agree("map",&prepared,None)?;
  let(sorted_hash,hash_lookup_limit,hash_eager_tail,batch_offsets_host,batch_offsets,batch_dense_origins,bound,h,owner_map,mut root,map,cuts,configured_rows,explicit_rows,estimated)=prepared?;
  // Exercise actual peer communication before measuring free memory. NCCL
  // can lazily allocate transport state on its first collective/exchange.
  // Compact traffic needs one warmup word per peer, independent of shards.
  if primary.is_none(){
  let warmup_peer_bytes=if using_packet{4}else{u64::from(p.shards)*4};
  let cleared:Result<_>=(||{crate::segmented_host_ready::injected("warmup-clear")?;
   crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(send[0].counts.ptr,0,p.world as usize*warmup_peer_bytes as usize,stream)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;Ok(())})();
  ready.agree("warmup-clear",&cleared,None)?;cleared?;
  if p.world>1{
   crate::generic_native::checked!(mgbfs_nccl_exchange_triplets,unsafe{mgbfs_nccl_exchange_triplets(comm.ptr,rank,p.world,send[0].counts.ptr,warmup_peer_bytes,ptr::null(),0,ptr::null(),0,recv.counts.ptr,ptr::null_mut(),ptr::null_mut(),stream)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
  }
  }
  let prepared:Result<_>=(||{
  let reserve=crate::generic_memory::admission_reserve_bytes()?;
  let storage_banks=if bitmap_mode{bitmap_banks}else{p.history_layers};
  let fit_layout=|rows,budget|if sorted_hash{fit_sorted_segment_layout_with_row_metadata(p.elements,p.state_bytes,storage_banks,rows,budget,metadata_bytes)}else if pack4.is_some(){fit_packed_bitmap_segment_layout(p.elements,storage_banks,rows,budget)}else{fit_segment_layout_compact(bitmap_mode,p.elements,p.state_bytes,storage_banks,rows,budget,compact_positions)};
  let min_layout=fit_layout(1,u64::MAX).map_err(str::to_owned)?.ok_or("SEGMENTED_NO_CAPACITY")?;
  let hash_timing_bounds=if sorted_hash&&std::env::var("MGBFS_SEGMENT_HASH_TIMING").as_deref()==Ok("1"){
   let low=std::env::var("MGBFS_SEGMENT_HASH_TIMING_MIN_DEPTH").ok().map(|x|x.parse::<u64>().map_err(|_|"SEGMENTED_HASH_TIMING_DEPTH")).transpose()?.unwrap_or(0);
   let high=std::env::var("MGBFS_SEGMENT_HASH_TIMING_MAX_DEPTH").ok().map(|x|x.parse::<u64>().map_err(|_|"SEGMENTED_HASH_TIMING_DEPTH")).transpose()?.unwrap_or(u64::MAX);
   if low>high{return Err("SEGMENTED_HASH_TIMING_DEPTH".into())}
   Some((low,high))
  }else{None};
  let exact_hash_misses=if std::env::var("MGBFS_SEGMENT_HASH_EXACT_MISSES").as_deref()==Ok("1"){
   if hash_timing_bounds.is_none()
    || !hash_lookup_limit.is_some_and(|limit|limit as usize<=HASH_EXACT_MISS_LIMIT){
    return Err("SEGMENTED_HASH_EXACT_MISS_REQUIRES_BOUNDED_TIMING".into());
   }
   Some(exact_hash_miss_kernel()?)
  }else{None};
  let diagnostic_bytes=if exact_hash_misses.is_some(){(HASH_EXACT_MISS_BUCKETS*size_of::<u64>()) as u64}
   else if hash_timing_bounds.is_some(){(HASH_MISS_BUCKETS*size_of::<u64>()) as u64}else{0};
  let lane_fixed_bytes=16+diagnostic_bytes;
  let mut min_owner_scratch=0usize;
  if strategy==SegmentStrategy::Batch{
   if bitmap_mode{crate::generic_native::checked!(mgbfs_bitmap_accept_batch_scratch_bytes,unsafe{mgbfs_bitmap_accept_batch_scratch_bytes(1,&mut min_owner_scratch)})?;}
    else{crate::generic_native::checked!(mgbfs_hash_batch_scratch_bytes,unsafe{mgbfs_hash_batch_scratch_bytes(1,p.state_bytes,u32::from(p.packed_candidates()),&mut min_owner_scratch)})?;}
  }
  let lane_bytes=(if strategy==SegmentStrategy::Batch{1}else{bound} as u64).checked_mul(8).and_then(|v|v.checked_add(lane_fixed_bytes)).and_then(|v|v.checked_add(min_owner_scratch as u64)).ok_or("SEGMENTED_SPILL_SIZE")?;
  let bitmap=if let Some(domain)=&bitmap_domain{
   let logical=owner_map.iter().position(|&physical|physical==rank).ok_or("BITMAP_OWNER_MAP")? as u32;
   let partition=crate::permutation_bitmap::BitmapPartition::new(domain.size(),p.world).map_err(str::to_owned)?;
   let owner_len=partition.owner_len(logical).map_err(str::to_owned)?;
   let bytes=partition.owner_bytes(logical).map_err(str::to_owned)?.max(8);
   let(free,_)=segment_sizing_memory_info()?;
   if bytes>(free as u64).saturating_sub(reserve).saturating_sub(min_layout.device_bytes).saturating_sub(lane_bytes){return Err("BITMAP_DOMAIN_NO_VRAM".into())}
   let bit_storage=RawBuffer::try_new(usize::try_from(bytes).map_err(|_|"BITMAP_SIZE_OVERFLOW")?,d)?.ok_or("BITMAP_DOMAIN_NO_VRAM")?;
   crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(bit_storage.ptr,0,bit_storage.bytes,stream)})?;
   let xor=if packet_config.is_some()&&pack4.is_none()&&domain.size()<=1u64<<32
     &&std::env::var("MGBFS_SEGMENT_BITMAP_XOR_ROUTE").as_deref()==Ok("1"){
    let mgbfs_core::graph_definition::GraphAction::Permutation{generators,..}=&g.action
     else{return Err("BITMAP_REQUIRES_PERMUTATION_ACTION".into())};
    if let Some(masks)=domain.xor_generator_masks(generators){
     let symbol=unsafe{libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_segmented_compact_route_xor_v1\0".as_ptr().cast())};
     if symbol.is_null(){return Err("SEGMENT_PACKET_BITMAP_XOR_ABI_UNAVAILABLE".into())}
     let mask_storage=Buffer::new(masks.len()*8,d)?;
     mask_storage.upload(&masks)?;
     Some((unsafe{std::mem::transmute::<*mut c_void,CompactXorRouteFn>(symbol)},mask_storage))
    }else{None}
   }else{None};
   let(host_symbols,host_counts,distinct)=domain.device_layout();
   let symbols=Buffer::new(host_symbols.len()*8,d)?;symbols.upload(&host_symbols)?;
   let counts=Buffer::new(host_counts.len()*4,d)?;counts.upload(&host_counts)?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
   Some(BitmapState{domain:BitmapDeviceDomain{symbols:symbols.ptr.cast(),counts:counts.ptr.cast(),distinct,elements:p.elements,total:domain.size()},_symbols:symbols,_counts:counts,bits:bit_storage,owner:logical,owner_len,owner_page,owner_boundary:page_map.ok_or("BITMAP_PAGE_MAP_MISSING")?.boundary(),immutable_routes,xor})
  }else{None};
  // Preserve only the previous layer when its global size is <1000; the
  // next bank would otherwise overwrite it before a resource stop. This
  // bounded device snapshot is admitted before segment sizing.
  let previous_snapshot=if bitmap_mode&&bitmap_banks==2 {
   let row_bytes=if pack4.is_some(){8}else{(p.elements as usize).checked_mul(p.state_bytes as usize).ok_or("BITMAP_SNAPSHOT_SIZE")?};
   let bytes=1000usize.checked_mul(row_bytes).ok_or("BITMAP_SNAPSHOT_SIZE")?;
   Some(Buffer::new(bytes,d)?)
  } else {None};
  // One serial route workspace is separate from owner scratch, allowing the
  // next send slot to route while the previous slot is accepted.
  let mut admitted_batch=producer_rows.map_or(p.batch,|rows|rows/base.generators);
  let mut route_scratch=None;
  if strategy==SegmentStrategy::Batch||using_packet{
   admitted_batch=admitted_batch.min((0x7fff_fffeu32)/base.generators).max(1);
   loop{
    let children=admitted_batch.checked_mul(base.generators).ok_or("SEGMENTED_BATCH_ROUTE_RANGE")?;
    let buckets=p.world.checked_mul(p.shards).ok_or("SEGMENTED_QUEUE_EXTENT")?;
    let mut bytes=0usize;
    crate::generic_native::checked!(mgbfs_segmented_batch_route_scratch_bytes,unsafe{mgbfs_segmented_batch_route_scratch_bytes(children,buckets,&mut bytes)})?;
    let(free,_)=segment_sizing_memory_info()?;
    let available=(free as u64).saturating_sub(reserve).saturating_sub(min_layout.device_bytes).saturating_sub(lane_bytes);
    if bytes as u64<=available{
     if let Some(buf)=RawBuffer::try_new(bytes,d)?{route_scratch=Some(buf);break}
    }
    if admitted_batch==1{return Err(format!("SEGMENTED_BATCH_ROUTE_ADMISSION required_bytes={bytes} available_bytes={available} reserve_bytes={reserve} min_segment_bytes={}",min_layout.device_bytes))}
    admitted_batch=(admitted_batch/2).max(1);
   }
  }
  // Size each lane's complete scratch and spill before admitting segments.
  // Under pressure shrink the tile and then use as many requested lanes as fit.
  let requested=std::env::var("MGBFS_SEGMENT_OWNER_LANES").or_else(|_|std::env::var("MGBFS_SEGMENT_TEST_LANES")).ok().map(|v|v.parse::<u32>().map_err(|_|"SEGMENTED_OWNER_LANES".to_owned())).transpose()?.unwrap_or(4);
  if !(1..=64).contains(&requested){return Err("SEGMENTED_OWNER_LANES".into())}
  let mut batch_tile=if strategy==SegmentStrategy::Batch{
   std::env::var("MGBFS_SEGMENT_BATCH_TILE").ok().map(|v|v.parse::<u32>().map_err(|_|"SEGMENTED_BATCH_TILE".to_owned())).transpose()?.unwrap_or(65536).min(bound as u32)
  }else{bound as u32};
  if batch_tile==0{return Err("SEGMENTED_BATCH_TILE".into())}
  let (lane_count,owner_scratch)=loop{
   let mut scratch=0usize;
   if strategy==SegmentStrategy::Batch{
     if bitmap_mode{crate::generic_native::checked!(mgbfs_bitmap_accept_batch_scratch_bytes,unsafe{mgbfs_bitmap_accept_batch_scratch_bytes(batch_tile,&mut scratch)})?;}
     else{crate::generic_native::checked!(mgbfs_hash_batch_scratch_bytes,unsafe{mgbfs_hash_batch_scratch_bytes(batch_tile,p.state_bytes,u32::from(p.packed_candidates()),&mut scratch)})?;}
    }
   let lane_need=u64::from(batch_tile).checked_mul(8)
    .and_then(|v|v.checked_add(lane_fixed_bytes))
    .and_then(|v|v.checked_add(scratch as u64)).ok_or("SEGMENTED_SPILL_SIZE")?;
   let(free,_)=segment_sizing_memory_info()?;
   let affordable=(free as u64).saturating_sub(reserve).saturating_sub(min_layout.device_bytes)/lane_need;
   let count=(requested as u64).min(u64::from(p.shards)).min(affordable) as usize;
   if count>0{break(count,scratch)}
   if strategy!=SegmentStrategy::Batch||batch_tile==1{return Err(format!("SEGMENTED_LANE_ADMISSION_FAILED required_bytes={lane_need} free_bytes={free} reserve_bytes={reserve} min_segment_bytes={}",min_layout.device_bytes))}
   batch_tile=(batch_tile/2).max(1);
  };
  let(mut batch_merge_rows,mut batch_merge_enabled)=(0u32,false);
  if sorted_hash{
   let mut minimum=0usize;crate::generic_native::checked!(mgbfs_hash_batch_compact_scratch_bytes,unsafe{mgbfs_hash_batch_compact_scratch_bytes(0,&mut minimum)})?;
   if minimum<=owner_scratch{
    batch_merge_enabled=true;
    let(mut lo,mut hi)=(0u32,((owner_scratch/12).min(0x7fff_fffeusize)) as u32);
    while lo<hi{
     let mid=lo+(hi-lo+1)/2;let mut bytes=0usize;
     crate::generic_native::checked!(mgbfs_hash_batch_compact_scratch_bytes,unsafe{mgbfs_hash_batch_compact_scratch_bytes(mid,&mut bytes)})?;
     if bytes<=owner_scratch{lo=mid}else{hi=mid-1}
    }
    batch_merge_rows=lo;
    if let Ok(v)=std::env::var("MGBFS_SEGMENT_TEST_MERGE_ROWS"){
     let limit=v.parse::<u32>().map_err(|_|"SEGMENTED_TEST_MERGE_ROWS")?;
     batch_merge_rows=batch_merge_rows.min(limit);
    }
   }
  }
  let input_stats_bounds=if strategy==SegmentStrategy::Batch&&std::env::var("MGBFS_SEGMENT_TEST_INPUT_STATS").as_deref()==Ok("1"){
   let low=std::env::var("MGBFS_SEGMENT_TEST_INPUT_MIN_DEPTH").ok().map(|x|x.parse::<u64>().map_err(|_|"SEGMENTED_INPUT_STATS_DEPTH")).transpose()?.unwrap_or(0);
   let high=std::env::var("MGBFS_SEGMENT_TEST_INPUT_MAX_DEPTH").ok().map(|x|x.parse::<u64>().map_err(|_|"SEGMENTED_INPUT_STATS_DEPTH")).transpose()?.unwrap_or(u64::MAX);
   if low>high{return Err("SEGMENTED_INPUT_STATS_DEPTH".into())}
   Some((low,high))
  }else{None};
  let overlap_bounds=if std::env::var("MGBFS_SEGMENT_TEST_OVERLAP").as_deref()==Ok("1"){
   let low=std::env::var("MGBFS_SEGMENT_TEST_OVERLAP_MIN_DEPTH").ok().map(|x|x.parse::<u64>().map_err(|_|"SEGMENTED_OVERLAP_DEPTH")).transpose()?.unwrap_or(11);
   let high=std::env::var("MGBFS_SEGMENT_TEST_OVERLAP_MAX_DEPTH").ok().map(|x|x.parse::<u64>().map_err(|_|"SEGMENTED_OVERLAP_DEPTH")).transpose()?.unwrap_or(u64::MAX);
   if low>high{return Err("SEGMENTED_OVERLAP_DEPTH".into())}
   Some((low,high))
  }else{None};
  let mut lanes=Vec::with_capacity(lane_count);for _ in 0..lane_count{lanes.push(Lane::new(d,batch_tile as usize,owner_scratch,hash_timing_bounds.is_some(),exact_hash_misses)?);}
  eprintln!("SEGMENTED_OWNER_LANES rank={rank} lanes={lane_count} tile={batch_tile} owner_scratch={owner_scratch} route_scratch={}",route_scratch.as_ref().map_or(0,|x|x.bytes));
  Ok((reserve,storage_banks,bitmap,previous_snapshot,admitted_batch,route_scratch,batch_tile,batch_merge_rows,batch_merge_enabled,hash_timing_bounds,input_stats_bounds,overlap_bounds,lanes))
  })();
  ready.agree("memory",&prepared,None)?;
  let(reserve,storage_banks,bitmap,previous_snapshot,mut admitted_batch,route_scratch,batch_tile,batch_merge_rows,batch_merge_enabled,hash_timing_bounds,input_stats_bounds,overlap_bounds,lanes)=prepared?;
  let fit_layout=|rows,budget|if sorted_hash{fit_sorted_segment_layout_with_row_metadata(p.elements,p.state_bytes,storage_banks,rows,budget,metadata_bytes)}else if pack4.is_some(){fit_packed_bitmap_segment_layout(p.elements,storage_banks,rows,budget)}else{fit_segment_layout_compact(bitmap_mode,p.elements,p.state_bytes,storage_banks,rows,budget,compact_positions)};
  if strategy==SegmentStrategy::Batch&&!using_packet{
   // Only the legacy transport requires lockstep source rounds.
   // Compact receivers grant bytes independently of the sender's route tile.
   let proposed=u64::from(admitted_batch);
   let copied:Result<_>=(||{crate::segmented_host_ready::injected("batch-copy")?;
    crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(control.at::<c_void>(32),(&proposed as *const u64).cast(),8,1,stream)})?;
    crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;Ok(())})();
   ready.agree("batch-copy",&copied,None)?;copied?;
   crate::generic_native::checked!(mgbfs_nccl_all_gather_u64,unsafe{mgbfs_nccl_all_gather_u64(comm.ptr,control.at(32),control.at(64),stream)})?;
   let mut sizes=vec![0u64;p.world as usize];
   crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(sizes.as_mut_ptr().cast(),control.at::<c_void>(64),sizes.len()*8,2,stream)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
   admitted_batch=u32::try_from(*sizes.iter().min().ok_or("SEGMENTED_BATCH_AGREEMENT")?).map_err(|_|"SEGMENTED_BATCH_AGREEMENT")?;
   if admitted_batch==0{return Err("SEGMENTED_BATCH_AGREEMENT".into())}
  }
  let prepared:Result<_>=(||{
  let window_state=window.then(||WindowState::new(p.shards as usize,p.elements,d)).transpose()?;
  let path_workspace=if primary.is_none(){
   nearest_fns.map(|_|nearest::Workspace::new(p.elements,p.world,d)).transpose()?
  }else{None};
  let(free,_)=segment_sizing_memory_info()?;
  let mut remaining_bytes=(free as u64).saturating_sub(reserve);
  let capacity_limit=std::env::var("MGBFS_SEGMENT_MAX_RANK_STATES").ok().map(|v|v.parse::<u64>().map_err(|_|"SEGMENTED_CAPACITY_LIMIT".to_owned())).transpose()?;
  if capacity_limit==Some(0){return Err("SEGMENTED_CAPACITY_LIMIT".into())}
  let make_layout=|rows|if sorted_hash{SegmentLayout::sorted_with_row_metadata(p.elements,p.state_bytes,rows,storage_banks,metadata_bytes)}else if pack4.is_some(){SegmentLayout::bitmap_banks_pack4(p.elements,rows,storage_banks)}else if bitmap_mode{SegmentLayout::bitmap_banks(p.elements,p.state_bytes,rows,storage_banks)}else if compact_positions{SegmentLayout::new_compact(p.elements,p.state_bytes,rows,p.history_layers)}else{SegmentLayout::new(p.elements,p.state_bytes,rows,p.history_layers)};
  let layout=if let Some(exact)=explicit_rows{make_layout(exact).map_err(str::to_owned)?}else{
   // The preferred segment is one allocation quantum. A skewed shard can
   // grow by taking additional segments from the rank's shared budget.
   let shards=u64::from(p.shards);
   let capacity_rows=capacity_limit.map_or(LOCAL_ROWS,|limit|limit.div_ceil(shards));
   let upper=configured_rows.min(estimated).min(capacity_rows);
   // Path queries grow lazily. Window needs room for the next frontier
   // while the current frontier is still routed. Each extent allocation
   // consults live headroom, including memory used by the opposite side.
   let per_shard=remaining_bytes/u64::from(p.shards);
   let per_shard=if route_provenance{per_shard.min(64<<20)}else{per_shard};
   match fit_layout(upper,per_shard).map_err(str::to_owned)?{
    Some(v)=>v,
    None=>fit_layout(upper,remaining_bytes).map_err(str::to_owned)?.ok_or("SEGMENTED_NO_CAPACITY")?,
   }
  };
  if let Ok(v)=std::env::var("MGBFS_SEGMENT_TEST_LIMIT"){let count=v.parse::<u64>().map_err(|_|"SEGMENTED_TEST_LIMIT")?;remaining_bytes=remaining_bytes.min(count.saturating_mul(layout.device_bytes));}
  if let Ok(v)=std::env::var("MGBFS_SEGMENT_TEST_BYTES"){let bytes=v.parse::<u64>().map_err(|_|"SEGMENTED_TEST_BYTES")?;remaining_bytes=remaining_bytes.min(bytes);}
  let layer_stats=std::env::var("MGBFS_SEGMENT_LAYER_STATS").as_deref()==Ok("1");
  let layer_stats_low=std::env::var("MGBFS_SEGMENT_LAYER_STATS_MIN_DEPTH").ok().and_then(|v|v.parse().ok()).unwrap_or(0);
  let layer_stats_high=std::env::var("MGBFS_SEGMENT_LAYER_STATS_MAX_DEPTH").ok().and_then(|v|v.parse().ok()).unwrap_or(u64::MAX);
  Ok((remaining_bytes,layout,capacity_limit,layer_stats,layer_stats_low,layer_stats_high,path_workspace,window_state))
  })();
  ready.agree("layout",&prepared,None)?;
  let(remaining_bytes,layout,capacity_limit,layer_stats,layer_stats_low,layer_stats_high,path_workspace,window_state)=prepared?;
  // For wide rows, the admitted maximum can be a GiB-sized extent even
  // when only a few rows reach a shard. Grow from a small allocation.
  let wide_hash_rows = p.state_bytes == 8 || p.queue_payload_bytes() > 64
   || (p.elements > 24 && std::env::var("MGBFS_SEGMENT_RESIDENT_ALPHABET8").as_deref() == Ok("1"));
  let initial_hash_extent_rows=if !bitmap_mode&&!sorted_hash&&wide_hash_rows
   &&explicit_rows.is_none()&&layout.device_bytes>64<<20{
   let bytes_per_row=layout.device_bytes.div_ceil(layout.rows_per_bank);
   Some(((8<<20)/bytes_per_row).max(1))
  }else{None};
  let admitted_owner_cuts=if primary.is_none()&&std::env::var("MGBFS_SEGMENT_EXTERNAL_LINK").as_deref()==Ok("1")&&!bitmap_mode{
   let copied:Result<_>=(||{
    let rows=(remaining_bytes/layout.device_bytes).saturating_mul(layout.rows_per_bank).min(capacity_limit.unwrap_or(u64::MAX));
    if rows==0||cuts.is_none(){return Err("SEGMENTED_OWNER_ADMISSION_ROWS".into())}
    crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(control.at::<c_void>(32),(&rows as *const u64).cast(),8,1)})?;
    Ok(())
   })();
   ready.agree("owner-weight-copy",&copied,None)?;copied?;
   crate::generic_native::checked!(mgbfs_nccl_all_gather_u64,unsafe{mgbfs_nccl_all_gather_u64(comm.ptr,control.at(32),control.at(64),stream)})?;
   let ownership:Result<_>=(||{
    let mut weights=vec![0u64;p.world as usize];
    crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(weights.as_mut_ptr().cast(),control.at::<c_void>(64),weights.len()*8,2,stream)})?;
    crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
    let selected=crate::segmented_history::weighted_owner_cuts(&weights).map_err(str::to_owned)?;
    cuts.as_ref().ok_or("SEGMENTED_OWNER_CUTS")?.upload(&selected)?;
    let owner=selected.windows(2).position(|bounds|bounds[0]<=h>>32&&h>>32<bounds[1]).ok_or("SEGMENTED_ROOT_OWNER")?;
    Ok((selected,owner_map[owner]))
   })();
   let descriptor=ownership.as_ref().ok().map(|(selected,_)|serde_json::json!(selected));
   ready.agree("owner-weights",&ownership,descriptor.as_ref())?;
   let(selected,owner)=ownership?;root=owner;Some(selected)
  }else{None};
  let ownership=route_provenance.then(||PathOwnership::new(owner_map,admitted_owner_cuts.clone().or(requested_path_cuts)));
  let nearest=nearest_fns.map(|fns|NearestState::new(fns,path_workspace));
  let mut result=Self{nearest,ownership,window:window_state,packet,global_sync:Default::default(),last_layer_sync:Default::default(),remote_payload_transfers:0,hash_reclaimed_segments:0,hash_reclaimed_device_bytes:0,admitted_owner_cuts,initial_hash_extent_rows,strategy,compact_positions,pack4,previous_snapshot,previous_snapshot_rows:0,previous_snapshot_valid:false,previous_global:0,base,plan:p.clone(),rank,seed,bits,layout,bitmap,segments:(0..p.shards).map(|_|Vec::new()).collect(),reuse_hash_segments:std::env::var("MGBFS_SEGMENT_HASH_REUSE").as_deref()!=Ok("0"),spare_hash_segment:None,hash_segment_reuses:0,descriptor_cache:(0..p.shards).map(|_|None).collect(),hash_shards:(0..p.shards).map(|_|None).collect(),remaining_bytes,storage_budget:remaining_bytes,storage_peak_requested:0,reserve_bytes:reserve,configured_rows,bank_totals:[0;3],peak_live_rows:0,last_resource:None,capacity_limit,current:vec![],previous:vec![],counts:(0..p.world).map(|r|u64::from(r==root)).collect(),depth:0,source_retries:0,active_batch:admitted_batch,queue_batch_ceiling:admitted_batch,admitted_batch,batch_recovery,batch_recovery_policy:Default::default(),batch_recoveries:0,layer_stats,layer_stats_low,layer_stats_high,profile_spill:0,profile_owner_passes:0,peak_owner_lanes:0,prefetched_chunks:0,overlap_samples:0,send,recv,route_stream,route_scratch,batch_offsets,batch_offsets_host,batch_dense_origins,batch_tile,batch_merge_rows,batch_merge_enabled,hash_lookup_limit,hash_eager_tail,hash_timing_bounds,hash_owner_ms:0.,hash_lookup_ms:0.,hash_merge_ms:0.,hash_lookup_directory_runs:Default::default(),hash_snapshot_timed_lookups:0,hash_snapshot_installations:0,hash_snapshot_mapped_host_installations:0,hash_snapshot_rebuild_peak_bytes:0,hash_snapshot_storage_evictions:0,input_stats_bounds,input_valid:0,input_raw_elements:0,input_raw_tiles:0,input_spill_elements:0,input_spill_tiles:0,input_empty_self:0,input_empty_peer:0,route_errors,control,map,cuts,lanes,comm,test_oom_done:false,bounds_trace:std::env::var("MGBFS_SEGMENT_TEST_BOUNDS_TRACE").as_deref()==Ok("1"),overlap_bounds,exact_exchange:std::env::var("MGBFS_SEGMENT_EXACT_EXCHANGE").as_deref()==Ok("1"),transport_trace:std::env::var("MGBFS_SEGMENT_TRANSPORT_TRACE").as_deref()==Ok("1"),remote_records_sent:0,remote_padded_records:0,test_count_payload:std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref()==Ok("1") && std::env::var("MGBFS_SEGMENT_TEST_HOST_FAIL").as_deref()==Ok("count-payload")};
  let seed_result:Result<u32>=(||{
  if let RootSpec::Targets(targets)=roots{
   result.install_target_roots(targets)?;
   return Ok(0)
  }
  let mut failed=0;
  if rank==root{
   let shard_hash=if bitmap_mode{if strategy==SegmentStrategy::Batch{mix((h/u64::from(p.world))>>6)}else{mix(h)}}else{h};
   let shard=(((shard_hash as u32 as u64)*u64::from(p.shards))>>32) as usize;
   let seed_extent=if window{result.make_segment_prepared_bounded(shard,false,0,1)?}else{result.make_segment_prepared(shard,false,0)?};
   if seed_extent.is_none(){failed=1}else{
    let input_bytes=(p.elements as usize).checked_mul(if pack4.is_some(){8}else{p.state_bytes as usize}).ok_or("BITMAP_SEED_SIZE")?;
    let input=Buffer::new(input_bytes,d)?;
    if p.state_bytes==1&&pack4.is_none(){input.upload(&g.start.iter().map(|&v|v as u8).collect::<Vec<_>>())?;}else{input.upload(&g.start)?;}
    let hashes=Buffer::new(8,d)?;hashes.upload(&[h])?;let seg=&result.segments[shard][0];
    crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(result.control.at::<c_void>(0),0,4,stream)})?;
    let status=unsafe{if let Some(b)=&result.bitmap{
     if let Some(packed)=result.pack4{
      (packed.fns.seed)(Pack4SeedArgs{dictionary:packed.dictionary,input:input.ptr.cast(),row:seg.desc.states.cast(),bits:b.bits.ptr.cast(),accepted:seg.counts.ptr.cast(),error:result.control.at(0),local:h/u64::from(p.world),owner_len:b.owner_len},stream)
     }else{mgbfs_bitmap_seed(p.state_bytes,input.ptr,p.elements,seg.desc,b.bits.ptr.cast(),h/u64::from(p.world),seg.counts.ptr.cast(),stream)}
    }else if sorted_hash{let hs=result.hash_shards[shard].as_ref().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;mgbfs_hash_batch_seed_v2(p.state_bytes,p.elements,input.ptr,seg.sorted_view(),h,hs.dir.ptr.cast(),hs.count_ptr(),seg.counts.ptr.cast(),stream)
     }else if p.state_bytes==1{mgbfs_segmented_accept_u8(input.ptr.cast(),hashes.ptr.cast(),1,p.elements,ptr::null(),0,seg.desc,0,seg.counts.ptr.cast(),ptr::null_mut(),result.control.at(0),stream)}else{mgbfs_segmented_accept_i64(input.ptr.cast(),hashes.ptr.cast(),1,p.elements,ptr::null(),0,seg.desc,0,seg.counts.ptr.cast(),ptr::null_mut(),result.control.at(0),stream)}};
    crate::generic_native::checked!(mgbfs_segmented_accept_seed,status)?;crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
    let mut error=0u32;crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut error as *mut u32).cast(),result.control.ptr,4,2)})?;
    if error!=0{failed=error+1}else{if sorted_hash{result.hash_shards[shard].as_mut().unwrap().observe(1,false);}result.segments[shard][0].banks[0]=1;result.bank_totals[0]=1;result.peak_live_rows=1;result.current.push(Range{shard,segment:0,bank:0,begin:0,end:1,logical_start:0});}
   }
  }
  Ok(failed)
  })();
  if let Some(nearest)=result.nearest.as_mut(){nearest.publish(&mut result.current,&result.segments,0);}
  result.number_window_roots()?;
  ready.agree("seed",&seed_result,None)?;
  let failed=seed_result?;
  crate::segmented_host_ready::injected("after-seed")?;
  if target_roots{result.gather_target_root_counts()?;}
  else if result.vote(failed)?!=0{return Err("SEGMENTED_ROOT_ADMISSION_FAILED".into())}
  result.global_sync.set(SyncCounts::default());
  Ok(result)
 }
 fn release_spare_hash_segment(&mut self){
  if let Some(segment)=self.spare_hash_segment.take(){
   self.remaining_bytes+=segment.device_bytes;
  }
 }

 fn make_segment_bounded(&mut self,shard:usize,max_rows:u64)->Result<bool>{
  if max_rows==0{return Err("SEGMENTED_SEGMENT_RETRY_LIMIT".into())}
  let metadata_bytes=if self.nearest.is_some(){PATH_PARENT_BYTES}else{0};
  // A private test failpoint triggers a real, bounded, oversized cudaMalloc
  // on this rank only after several complete layers. It never fills VRAM.
  if !self.test_oom_done{
   if let Ok(v)=std::env::var("MGBFS_SEGMENT_TEST_REAL_OOM_AFTER_DEPTH"){
    let threshold=v.parse::<u64>().map_err(|_|"SEGMENTED_TEST_OOM_DEPTH")?;
    if self.depth>=threshold{
     self.test_oom_done=true;
     let(mut free,mut total)=(0usize,0usize);crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
     let requested=total.checked_add(1024*1024).ok_or("SEGMENTED_TEST_OOM_SIZE")?;
     match RawBuffer::try_new(requested,self.base.device)?{
      None=>{self.last_resource=Some(serde_json::json!({"reason":"cuda_allocation_test","rank":self.rank,"depth":self.depth,"shard":shard,"requested_bytes":requested,"device_free_bytes":free,"device_total_bytes":total}));eprintln!("SEGMENTED_TEST_REAL_OOM rank={} depth={} requested={} total={} code=2",self.rank,self.depth,requested,total);return Ok(false)},
      Some(v)=>{drop(v);return Err("SEGMENTED_TEST_OOM_INCONCLUSIVE_ALLOCATION_SUCCEEDED".into())}
     }
    }
   }
  }
  // Growing a descriptor temporarily holds both the old and new arrays.
  // The old array is already charged; reserve the entire new array here.
  let next_count=self.segments[shard].len().checked_add(1).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
  let sorted=self.strategy==SegmentStrategy::Batch&&self.bitmap.is_none();
   let descriptor_peak=if sorted{
    let bytes=next_count.checked_mul(std::mem::size_of::<BatchHashSegment>()).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
    if let Some(directory)=&self.hash_shards[shard]{
     // Reclaiming extents keeps the descriptor buffer for a nonempty shard.
     // hash_view rewrites it in place when that capacity still fits.
     if directory.view.as_ref().is_some_and(|view|view.data.bytes>=bytes){0}else{bytes as u64}
    }else{
     (bytes as u64).checked_add(hash_directory_bytes(initial_hash_run_capacity()?)? as u64).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?
    }
   }else if self.bitmap.is_none()&&next_count>1{
    let bytes=next_count.checked_mul(std::mem::size_of::<SegmentDescriptor>()).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
    if self.descriptor_cache[shard].as_ref().is_some_and(|cache|cache.data.bytes>=bytes){0}else{bytes as u64}
   }else{0};
  let(mut free,mut total)=(0usize,0usize);
  crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
  let preferred=self.layout.rows_per_bank.min(max_rows);
  if let Some(spare)=self.spare_hash_segment.take(){
   let headroom=self.remaining_bytes.min((free as u64).saturating_sub(self.reserve_bytes));
   if spare.desc.capacity==preferred && descriptor_peak<=headroom{
    let mut spare=spare;
    if let Err(error)=spare.reset_hash_extent(self.base.stream.ptr){
     self.remaining_bytes+=spare.device_bytes;
     return Err(error)
    }
    self.segments[shard].push(spare);
    self.hash_segment_reuses+=1;
    return Ok(true)
   }
   self.remaining_bytes+=spare.device_bytes;
   drop(spare);
   crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
  }
  if sorted {
   let preferred_bytes=SegmentLayout::sorted_with_row_metadata(self.plan.elements,self.plan.state_bytes,preferred,self.layout.banks,metadata_bytes)
    .map_err(str::to_owned)?.device_bytes;
   let available=self.remaining_bytes.min((free as u64).saturating_sub(self.reserve_bytes));
   if preferred_bytes.saturating_add(descriptor_peak)>available{
    let reclaimable:u64=self.hash_shards.iter().flatten()
     .flat_map(|item|&item.snapshots)
     .map(|snapshot|snapshot.storage.device_bytes()).sum();
    if reclaimable!=0{
     let after_reclaim=self.remaining_bytes.saturating_add(reclaimable)
      .min((free as u64).saturating_add(reclaimable).saturating_sub(self.reserve_bytes));
     let recovered_layout=fit_sorted_segment_layout_with_row_metadata(self.plan.elements,self.plan.state_bytes,
      self.layout.banks,preferred,after_reclaim.saturating_sub(descriptor_peak),metadata_bytes).map_err(str::to_owned)?;
     if let Some(layout)=recovered_layout{
      if self.reclaim_snapshots_for_storage(layout.device_bytes.saturating_add(descriptor_peak),free as u64)?{
       crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
      }
     }
    }
   }
  }
  let available=self.remaining_bytes.min((free as u64).saturating_sub(self.reserve_bytes));
  let budget=available.saturating_sub(descriptor_peak);
  let mut upper=self.layout.rows_per_bank.min(max_rows);let mut allocation_failed=false;
  loop{
   let sorted=self.strategy==SegmentStrategy::Batch&&self.bitmap.is_none();
    let fit=|rows,budget|if sorted{fit_sorted_segment_layout_with_row_metadata(self.plan.elements,self.plan.state_bytes,self.layout.banks,rows,budget,metadata_bytes)}else if self.pack4.is_some(){fit_packed_bitmap_segment_layout(self.plan.elements,self.layout.banks,rows,budget)}else{fit_segment_layout_compact(self.bitmap.is_some(),self.plan.elements,self.plan.state_bytes,self.layout.banks,rows,budget,self.compact_positions)};
    let layout=fit(upper,budget).map_err(str::to_owned)?;
   let Some(layout)=layout else {
    let minimum=fit(1,u64::MAX).map_err(str::to_owned)?.ok_or("SEGMENTED_LAYOUT_RANGE")?.device_bytes;
    self.last_resource=Some(serde_json::json!({"reason":if allocation_failed{"cuda_allocation"}else{"segment_budget"},"rank":self.rank,"completed_depth":self.depth,"target_depth":self.depth.saturating_add(1),"shard":shard,"requested_bytes":minimum.saturating_add(descriptor_peak),"preferred_segment_bytes":self.layout.device_bytes,"remaining_budget_bytes":self.remaining_bytes,"device_free_bytes":free,"device_total_bytes":total,"reserve_bytes":self.reserve_bytes,"descriptor_transient_bytes":descriptor_peak}));
    return Ok(false)
   };
   match Segment::try_new(layout,self.base.device,self.base.stream.ptr,sorted,self.nearest.is_some())?{
    Some(seg)=>{self.remaining_bytes-=layout.device_bytes;self.storage_peak_requested=self.storage_peak_requested.max(self.storage_budget-self.remaining_bytes);self.segments[shard].push(seg);return Ok(true)},
    None=>{allocation_failed=true;if layout.rows_per_bank==1{let minimum=layout.device_bytes.saturating_add(descriptor_peak);self.last_resource=Some(serde_json::json!({"reason":"cuda_allocation","rank":self.rank,"completed_depth":self.depth,"target_depth":self.depth.saturating_add(1),"shard":shard,"requested_bytes":minimum,"remaining_budget_bytes":self.remaining_bytes,"device_free_bytes":free,"device_total_bytes":total,"reserve_bytes":self.reserve_bytes}));return Ok(false)}upper=layout.rows_per_bank/2;}
   }
  }
 }
 // Called only between drained waves. All descriptors for a shard remain
 // immutable until a segment is appended; no per-chunk cudaMalloc/cudaFree.
 fn descriptor_view(&mut self,shard:usize)->Result<Option<(*const SegmentDescriptor,u32)>>{
  let count=self.segments[shard].len();
  if count<=1{return Ok(Some((ptr::null(),0)))}
  if self.descriptor_cache[shard].as_ref().map_or(true,|c|c.count!=count){
   let item_size=std::mem::size_of::<SegmentDescriptor>();
   let bytes=count.checked_mul(item_size).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
   let old=self.descriptor_cache[shard].as_ref().map_or(0,|cache|cache.data.bytes);
   if bytes<=old{
    let cache=self.descriptor_cache[shard].as_mut().ok_or("SEGMENTED_DESCRIPTOR_MISSING")?;
    // Reclamation invalidates the cache, so a one-entry increase is an
    // append. Keep the existing prefix resident on the device.
    if cache.count.checked_add(1)==Some(count){
     let descriptor=&self.segments[shard][count-1].desc;
     let dst=cache.data.at::<c_void>(bytes-item_size);
     crate::generic_native::checked!(cudaMemcpy,unsafe{
      cudaMemcpy(dst,(descriptor as *const SegmentDescriptor).cast(),item_size,1)
     })?;
    }else{
     let descriptors=self.segments[shard].iter().map(|s|s.desc).collect::<Vec<_>>();
     crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(cache.data.ptr,descriptors.as_ptr().cast(),bytes,1)})?;
    }
    cache.count=count;
   }else{
    let(mut free,mut total)=(0usize,0usize);
    crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
    let available=self.remaining_bytes.min((free as u64).saturating_sub(self.reserve_bytes));
    if bytes as u64>available{
     self.last_resource=Some(serde_json::json!({"reason":"descriptor_budget","rank":self.rank,"completed_depth":self.depth,"shard":shard,"requested_bytes":bytes,"remaining_budget_bytes":self.remaining_bytes,"device_free_bytes":free,"device_total_bytes":total,"reserve_bytes":self.reserve_bytes}));
     return Ok(None)
    }
    let grown=count.checked_next_power_of_two().and_then(|rows|rows.checked_mul(item_size)).unwrap_or(bytes);
    // Keep room for future descriptors when spare budget is available. Both arrays
    // coexist during replacement, so retry with the exact size if growth fails.
    let mut data=None;
    if grown>bytes && (grown as u64)<=available{
     data=RawBuffer::try_new(grown,self.base.device)?;
    }
    let data=match data{
     Some(buffer)=>Some(buffer),
     None=>RawBuffer::try_new(bytes,self.base.device)?,
    };
    let Some(data)=data else{
     self.last_resource=Some(serde_json::json!({"reason":"descriptor_cuda_allocation","rank":self.rank,"completed_depth":self.depth,"shard":shard,"requested_bytes":bytes,"remaining_budget_bytes":self.remaining_bytes,"device_free_bytes":free,"device_total_bytes":total}));
     return Ok(None)
    };
    let descriptors=self.segments[shard].iter().map(|s|s.desc).collect::<Vec<_>>();
    crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(data.ptr,descriptors.as_ptr().cast(),bytes,1)})?;
    self.storage_peak_requested=self.storage_peak_requested.max((self.storage_budget-self.remaining_bytes).checked_add(data.bytes as u64).ok_or("SEGMENTED_STORAGE_ACCOUNTING")?);
    self.remaining_bytes-=(data.bytes-old) as u64;
    self.descriptor_cache[shard]=Some(DescriptorCache{data,count});
   }
  }
  let c=self.descriptor_cache[shard].as_ref().ok_or("SEGMENTED_DESCRIPTOR_MISSING")?;
  Ok(Some((c.data.ptr.cast(),u32::try_from(c.count).map_err(|_|"SEGMENTED_DESCRIPTOR_SIZE")?)))
 }

 fn storage_allocation(&mut self,bytes:usize,shard:usize,reason:&str)->Result<Option<RawBuffer>>{
  let(mut free,mut total)=(0usize,0usize);
  crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
  if self.spare_hash_segment.is_some() && (bytes as u64)>self.remaining_bytes.min((free as u64).saturating_sub(self.reserve_bytes)){
   self.release_spare_hash_segment();
   crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
  }
  if self.reclaim_snapshots_for_storage(bytes as u64,free as u64)?{
   crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
  }
  let available=self.remaining_bytes.min((free as u64).saturating_sub(self.reserve_bytes));
  if bytes as u64>available{
   self.last_resource=Some(serde_json::json!({"reason":reason,"rank":self.rank,"completed_depth":self.depth,"shard":shard,"requested_bytes":bytes,"remaining_budget_bytes":self.remaining_bytes,"device_free_bytes":free,"device_total_bytes":total,"reserve_bytes":self.reserve_bytes}));
   return Ok(None)
  }
  let mut allocation=RawBuffer::try_new(bytes,self.base.device)?;
  // cudaMemGetInfo reports aggregate free bytes. Give CUDA another
  // chance after releasing the idle extent if fragmentation blocked it.
  if allocation.is_none()&&self.spare_hash_segment.is_some(){
   self.release_spare_hash_segment();
   crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
   allocation=RawBuffer::try_new(bytes,self.base.device)?;
  }
  match allocation{
   Some(v)=>{let live=self.storage_budget.checked_sub(self.remaining_bytes).ok_or("SEGMENTED_STORAGE_ACCOUNTING")?;let transient=live.checked_add(bytes as u64).ok_or("SEGMENTED_STORAGE_ACCOUNTING")?;self.storage_peak_requested=self.storage_peak_requested.max(transient);Ok(Some(v))},
   None=>{self.last_resource=Some(serde_json::json!({"reason":format!("{reason}_cuda_allocation"),"rank":self.rank,"completed_depth":self.depth,"shard":shard,"requested_bytes":bytes,"remaining_budget_bytes":self.remaining_bytes,"device_free_bytes":free,"device_total_bytes":total,"reserve_bytes":self.reserve_bytes}));Ok(None)}
  }
 }
 // Allocate a segment and its owner metadata as one transaction. If metadata
 // no longer fits after the segment allocation, undo only the new segment and
 // retry a smaller one before declaring this rank out of capacity.
 fn owner_prepared(&mut self,shard:usize,grow:bool)->Result<Option<OwnerPrepared>>{
  if self.strategy==SegmentStrategy::Batch&&self.bitmap.is_none(){
   self.hash_view(shard,grow)
  }else{
   let view=if self.bitmap.is_some(){Some((ptr::null(),0))}else{self.descriptor_view(shard)?};
   Ok(view.map(|(sealed,descriptor_count)|OwnerPrepared{sealed,descriptor_count,hash_segments:ptr::null(),hash_count:0,hash_runs:ptr::null_mut(),hash_run_count:ptr::null_mut(),hash_run_capacity:0}))
  }
 }
 fn make_segment_prepared(&mut self,shard:usize,grow:bool,spilled_rows:u32)->Result<Option<OwnerPrepared>>{
  self.make_segment_prepared_bounded(shard,grow,spilled_rows,self.layout.rows_per_bank)
 }
 fn make_segment_prepared_bounded(&mut self,shard:usize,mut grow:bool,spilled_rows:u32,mut max_rows:u64)->Result<Option<OwnerPrepared>>{
  if let Some(initial_rows)=self.initial_hash_extent_rows{
   let next_rows=self.segments[shard].last().map_or(initial_rows,|extent|extent.desc.capacity.saturating_mul(2).max(initial_rows));
   // A device-reported spill is already past the initial history lookup.
   // Let its size raise the next allocation quantum before replay.
   max_rows=max_rows.min(next_rows.max(u64::from(spilled_rows)));
  }
  loop{
   if !self.make_segment_bounded(shard,max_rows)?{return Ok(None)}
   // A directory may grow successfully even if allocating its segment view
   // then fails. Never grow the same directory twice during this retry.
   let before=if grow&&self.strategy==SegmentStrategy::Batch&&self.bitmap.is_none(){
    Some(self.hash_shards[shard].as_ref().ok_or("SEGMENTED_HASH_GROW_WITHOUT_DIRECTORY")?.capacity)
   }else{None};
   let prepared=self.owner_prepared(shard,grow)?;
   if let Some(capacity)=before{
    if self.hash_shards[shard].as_ref().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?.capacity>capacity{grow=false;}
   }
   if prepared.is_some(){self.last_resource=None;return Ok(prepared)}
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;
   let released=self.segments[shard].pop().ok_or("SEGMENTED_ROLLBACK_MISSING")?;
   let failed_rows=released.desc.capacity;
   self.remaining_bytes=self.remaining_bytes.checked_add(released.device_bytes).ok_or("SEGMENTED_ROLLBACK_OVERFLOW")?;
   drop(released);
   if failed_rows<=1{return Ok(None)}
   max_rows=failed_rows/2;
   eprintln!("SEGMENTED_SEGMENT_RETRY rank={} shard={} previous_rows={} next_max_rows={}",self.rank,shard,failed_rows,max_rows);
  }
 }
 fn note_sync(&self,update:impl FnOnce(&mut SyncCounts)){
  if self.plan.world>1{
   let mut counters=self.global_sync.get();update(&mut counters);self.global_sync.set(counters);
  }
 }
 fn vote(&mut self,value:u32)->Result<u32>{
  if self.plan.world==1{return Ok(value)}
  let s=self.base.stream.ptr;self.control.upload(&[value])?;
  self.note_sync(|counts|counts.nccl_reduce_votes+=1);
  crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32,unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm.ptr,self.control.at(0),self.control.at(4),s)})?;
  let mut v=0u32;crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync((&mut v as *mut u32).cast(),self.control.at::<c_void>(4),4,2,s)})?;
  self.wait_boundary_stream()?;Ok(v)
 }
 // One collective independently reduces the error vote and the actual
 // maximum queue size, so every rank uses identical bounded NCCL messages.
 fn vote_route_and_max(&self,value:u32,local_max:u32)->Result<(u32,u32)>{
  if self.plan.world==1{return Ok((value,local_max))}
  let st=self.base.stream.ptr;self.control.upload(&[value,local_max])?;
  self.note_sync(|counts|counts.nccl_reduce_votes+=1);
  crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_pair_u32,unsafe{mgbfs_nccl_all_reduce_max_pair_u32(self.comm.ptr,self.control.at(0),self.control.at(8),st)})?;
  let mut values=[0u32;2];crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(values.as_mut_ptr().cast(),self.control.at::<c_void>(8),8,2,st)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;Ok((values[0],values[1]))
 }
 // Agree on a bounded number of source chunks once per layer, instead of
 // voting for remaining work after every chunk. A queue retry recomputes it.
 fn global_rounds(&self,work:&[Range],index:usize,cursor:u64,batch:u32)->Result<u64>{
  if batch==0{return Err("SEGMENTED_BATCH_ZERO".into())}
  let mut local=0u64;
  for (i,r) in work.iter().enumerate().skip(index){
   let begin=if i==index{cursor.max(r.begin)}else{r.begin};
   if begin>r.end{return Err("SEGMENTED_RANGE_CURSOR".into())}
   let n=r.end-begin;let b=u64::from(batch);
   local=local.checked_add(n/b+u64::from(n%b!=0)).ok_or("SEGMENTED_ROUND_COUNT_OVERFLOW")?;
  }
  if self.plan.world==1{return Ok(local)}
  let st=self.base.stream.ptr;
  crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(self.control.at::<c_void>(32),(&local as *const u64).cast(),8,1,st)})?;
  self.note_sync(|counts|counts.nccl_round_gathers+=1);
  crate::generic_native::checked!(mgbfs_nccl_all_gather_u64,unsafe{mgbfs_nccl_all_gather_u64(self.comm.ptr,self.control.at(32),self.control.at(64),st)})?;
  let mut ranks=vec![0u64;self.plan.world as usize];
  crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(ranks.as_mut_ptr().cast(),self.control.at::<c_void>(64),ranks.len()*8,2,st)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
  Ok(ranks.into_iter().max().unwrap_or(0))
 }
 pub fn collective_stop(&mut self,v:u32)->Result<u32>{
  // Arm communicator cancellation before the root vote as well as layers.
  self.connect_packet_control()?;
  self.vote(v)
 }
 pub fn global_frontier(&self)->u64{self.counts.iter().sum()}
 pub fn frontier_bucket(&self)->u32{63-self.counts.iter().copied().max().unwrap_or(0).max(1).leading_zeros()}
 pub fn strategy_name(&self)->&'static str{self.strategy.name()}
 fn layer_stats_active(&self)->bool{
  let depth=self.depth.saturating_add(1);
  self.layer_stats&&self.layer_stats_low<=depth&&depth<=self.layer_stats_high
 }
 pub fn source_retries(&self)->u64{self.source_retries}
 pub fn peak_owner_lanes(&self)->u32{self.peak_owner_lanes}
 fn transport_totals(&self)->(u64,u64,u64,u64){
 let sync=self.global_sync.get();
 // Count sends at the NCCL API boundary, independently of NCCL's wire protocol.
 if let Some(packet)=&self.packet{
  packet.transport_totals()
 }else{
  let row_bytes=32+self.plan.queue_payload_bytes() as u64;
  let records=self.remote_records_sent;
  let submitted_records=records.saturating_add(self.remote_padded_records);
  let count_sends=sync.legacy_chunk_count_exchanges.saturating_mul(u64::from(self.plan.world-1));
  let count_bytes=count_sends.saturating_mul(u64::from(self.plan.shards)*4);
  let fields=1+(self.plan.queue_payload_bytes() as u64/u64::from(self.plan.state_bytes));
  (records.saturating_mul(row_bytes),
   submitted_records.saturating_mul(row_bytes).saturating_add(count_bytes),
   count_bytes,self.remote_payload_transfers.saturating_mul(fields).saturating_add(count_sends))
 }

 }
 fn hash_miss_summary(&self, exact:bool) -> serde_json::Value {
  let mut buckets = [0u64; HASH_EXACT_MISS_BUCKETS];
  let count=if exact{HASH_EXACT_MISS_BUCKETS}else{HASH_MISS_BUCKETS};
  let buckets=&mut buckets[..count];
  for timing in self.lanes.iter().filter_map(|lane| lane.timing.as_ref()) {
   let mut lane_buckets = [0u64; HASH_EXACT_MISS_BUCKETS];
   let lane_buckets=&mut lane_buckets[..count];
   let status = unsafe { cudaMemcpy(lane_buckets.as_mut_ptr().cast(),timing.misses.ptr,size_of_val(lane_buckets),2) };
   if status != 0 {
    return serde_json::json!({"status":"UNVERIFIED","cuda_error":status});
   }
   for (total, lane_count) in buckets.iter_mut().zip(lane_buckets) { *total += *lane_count; }
  }
  let samples: u64 = buckets.iter().sum();
  let quantile = |percent: u64| {
   if samples == 0 { return None; }
   let mut cumulative = 0;
   for (bin, count) in buckets.iter().enumerate() {
    cumulative += count;
    if u128::from(cumulative)*100 >= u128::from(samples)*u128::from(percent) {
     return Some(bin);
    }
   }
   None
  };
  if exact{
   if buckets[HASH_EXACT_MISS_LIMIT+1]!=0{
    return serde_json::json!({"status":"UNVERIFIED","above_supported_limit":buckets[HASH_EXACT_MISS_LIMIT+1]});
   }
   return serde_json::json!({
    "misses":samples,"p95":quantile(95),"p99":quantile(99),"max":buckets.iter().rposition(|&n|n!=0),
    "histogram_by_checked_series":&buckets[..=HASH_EXACT_MISS_LIMIT],
    "scope":"every visited miss for tile-unique candidates in timed launches, including spill/retry; snapshots and raw runs passing metadata filters"
   });
  }
  let range=|p|quantile(p).map(|bin|if bin==0{[0,0]}else{[1u64<<(bin-1),(1u64<<bin)-1]});
  serde_json::json!({
   "samples":samples,"p95_range":range(95),"p99_range":range(99),"log2_buckets":buckets,
   "scope":"one per 128 sorted slots within each timed launch, only unique candidates missing from visited; counts all snapshot probes plus raw runs passing metadata filters, including retries; excludes rejected raw directory entries"
  })
 }
 pub fn memory_geometry(&self)->serde_json::Value{let(mut free,mut total)=(0usize,0usize);let valid=unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)}==0;let segment_rows=self.segments.iter().map(|bucket|bucket.iter().map(|s|s.desc.capacity).collect::<Vec<_>>()).collect::<Vec<_>>();let mut out=serde_json::json!({"segment_strategy":self.strategy.name(),"resident_codec":if self.pack4.is_some(){"pack4-v1"}else{"unpacked-soa"},"rows_per_bank":self.layout.rows_per_bank,"segment_rows_per_bank_by_shard":segment_rows,"live_rows_by_bank":self.bank_totals,"peak_live_rows":self.peak_live_rows,"device_free_bytes":if valid{Some(free)}else{None},"device_total_bytes":if valid{Some(total)}else{None},"reserve_bytes":self.reserve_bytes,"resource_detail":self.last_resource,"configured_segment_rows":self.configured_rows,"banks":self.layout.banks,"compact_positions":self.compact_positions,"position_item_bytes_by_shard":self.segments.iter().map(|bucket|bucket.iter().map(|s|s.desc.position_item_bytes).collect::<Vec<_>>()).collect::<Vec<_>>(),"slots_per_segment":self.layout.slots,"bytes_per_segment":self.layout.device_bytes,"segment_count":self.segments.iter().map(Vec::len).sum::<usize>(),"storage_budget_bytes":self.storage_budget,"storage_allocated_bytes":self.storage_budget-self.remaining_bytes,"descriptor_cache_bytes":self.descriptor_cache.iter().filter_map(|c|c.as_ref()).map(|c|c.data.bytes as u64).sum::<u64>(), "capacity_limit_per_rank":self.capacity_limit,"capacity_limit_is_hard":true,"capacity_limit_scope":if self.layout.banks>1{"one next layer per rank"}else{"all visited states per rank"},"owner_lanes":self.lanes.len(), "batch_owner_tile":if self.strategy==SegmentStrategy::Batch{Some(self.batch_tile)}else{None},"batch_route_scratch_bytes":self.route_scratch.as_ref().map_or(0,|x|x.bytes),"batch_owner_scratch_bytes_by_lane":self.lanes.iter().map(|x|x.scratch.as_ref().map_or(0,|s|s.bytes)).collect::<Vec<_>>(),"owner_pinned_metadata":self.lanes.iter().filter(|v|v.host.is_some()).count(),"prefetched_chunks":self.prefetched_chunks,"queue_batch_ceiling":self.queue_batch_ceiling,"exact_exchange":self.exact_exchange,"transport_trace":self.transport_trace,"remote_records_sent":self.remote_records_sent,"remote_padded_records":self.remote_padded_records,"bitmap_bytes":self.bitmap.as_ref().map_or(0,|b|b.bits.bytes),"bitmap_domain":self.bitmap.as_ref().map(|b|b.domain.total),"bitmap_logical_owner":self.bitmap.as_ref().map(|b|b.owner),"bitmap_owner_rows":self.bitmap.as_ref().map(|b|b.owner_len),"bitmap_owner_page":self.bitmap.as_ref().map(|b|b.owner_page),"extra_send_bytes":(self.send[1].records.bytes+self.send[1].counts.bytes+self.send[1].states.bytes) as u64,"legacy_plan_capacity_is_not_the_segment_limit":true});let extra=serde_json::json!({"admitted_batch":self.admitted_batch,"batch_recovery":self.batch_recovery,"batch_recoveries":self.batch_recoveries,"storage_peak_requested_bytes":self.storage_peak_requested,
 "storage_peak_scope":"segment states, index, counters, run directory and descriptors; excludes queue, scratch, CUDA/NCCL internals",
 "segment_state_allocated_bytes":self.segments.iter().flatten().map(|v|v.state_bytes).sum::<u64>(),
 "segment_index_allocated_bytes":self.segments.iter().flatten().map(|v|v.index_bytes).sum::<u64>(),
 "segment_counter_allocated_bytes":self.segments.iter().flatten().map(|v|v.counts.bytes as u64).sum::<u64>(),
 "queues_allocated_bytes":self.send.iter().chain(std::iter::once(&self.recv)).map(|v|(v.records.bytes+v.counts.bytes+v.states.bytes) as u64).sum::<u64>(),
 "actual_queue_capacity":self.plan.queue_capacity,
 "owner_spill_allocated_bytes":self.lanes.iter().map(|v|(v.spill[0].bytes+v.spill[1].bytes) as u64).sum::<u64>(),
 "owner_control_allocated_bytes":self.lanes.iter().map(|v|v.control.bytes as u64).sum::<u64>(),
 "previous_snapshot_allocated_bytes":self.previous_snapshot.as_ref().map_or(0,|v|v.bytes as u64),
 "batch_input_layout":if self.strategy==SegmentStrategy::Batch{Some(if self.packet.is_some(){"compact_packet"}else if self.batch_offsets.is_some(){"dense"}else{"rectangular"})}else{None},
 "batch_offsets_allocated_bytes":self.batch_offsets.as_ref().map_or(0,|v|v.bytes),
 "batch_hash_merge_rows":if self.batch_merge_enabled{Some(self.batch_merge_rows)}else{None},
 "batch_hash_directory_bytes":self.hash_shards.iter().flatten().map(|v|v.dir.bytes as u64).sum::<u64>(),
 "batch_hash_view_bytes":self.hash_shards.iter().flatten().filter_map(|v|v.view.as_ref()).map(|v|v.data.bytes as u64).sum::<u64>(),
 "batch_hash_index_bytes":if self.strategy==SegmentStrategy::Batch&&self.bitmap.is_none(){self.segments.iter().flatten().map(|v|v.index_bytes).sum::<u64>()}else{0},
 "batch_hash_runs_by_shard":self.hash_shards.iter().map(|x|x.as_ref().map(|v|serde_json::json!({"live":v.host_count,"lookup_series_live":v.searched_series(),"lookup_series_max_observed":v.lookup_max_series,"snapshot_rebuilds":v.snapshot_rebuilds,"snapshot_retirements":v.snapshot_retirements,"snapshot_evicted_for_rebuild":v.snapshot_evicted_for_rebuild,"snapshot_rebuilds_deferred":v.snapshot_deferred,"snapshot_rebuild_seconds":v.snapshot_rebuild_seconds,"snapshot_deferred_seconds":v.snapshot_deferred_seconds,"max_observed":v.max_runs,"median_observed":v.median(),"timed_window_max_observed":if self.hash_timing_bounds.is_some(){Some(v.timed_max_runs)}else{None},"timed_window_median_observed":v.timed_median(),"timed_window_observations":v.timed_run_histogram.values().copied().sum::<u64>(),"merged":v.merged_runs,"deferred":v.deferred_merges}))).collect::<Vec<_>>()});out.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());if let Some(cuts)=&self.admitted_owner_cuts{out["admitted_owner_cuts"]=serde_json::json!(cuts);}
 let window_bytes=self.packet.as_ref().map_or(0,PacketRuntime::window_bytes);
 let queue_bytes=self.send.iter().chain(std::iter::once(&self.recv))
  .map(|queue|queue.records.bytes+queue.states.bytes+queue.counts.bytes).sum::<usize>();
 let scratch_bytes=self.route_scratch.as_ref().map_or(0,|scratch|scratch.bytes)
  +self.batch_offsets.as_ref().map_or(0,|offsets|offsets.bytes);
 out["transport_allocated_bytes"]=serde_json::json!(window_bytes+queue_bytes+scratch_bytes);
 let route_status_bytes=self.route_errors.iter().map(|buffer|buffer.bytes).sum::<usize>();
 out["transport_device_peak_bytes"]=serde_json::json!(window_bytes+queue_bytes+scratch_bytes+route_status_bytes);
 out["transport_device_peak_breakdown"]=serde_json::json!({
  "route_queues":queue_bytes,"packet_windows_and_counts":window_bytes,
  "route_scratch_and_offsets":scratch_bytes,"route_status":route_status_bytes,
  "scope":"fixed concurrently allocated application buffers; excludes shared BFS control, owner lanes and NCCL internal buffers"
 });
 out["bitmap_xor_route"]=serde_json::json!(self.bitmap.as_ref().is_some_and(|b|b.xor.is_some()));
 out["transport_allocated_scope"]=serde_json::json!("route queues, packet windows and route scratch; excludes NCCL and owner storage");
 out["remote_payload_transfers"]=serde_json::json!(self.packet.as_ref().map_or(self.remote_payload_transfers,PacketRuntime::submitted_packets));
 out["remote_payload_transfers_scope"]=serde_json::json!(if self.packet.is_some(){"actual peer packet sends"}
  else if self.exact_exchange{"nonempty remote peer-shard bucket sends"}else{"padded remote peer-shard bucket sends"});
 let (candidate_bytes,data_bytes,metadata_bytes,sends)=self.transport_totals();
 out["remote_candidate_payload_bytes"]=candidate_bytes.into();
 out["remote_submitted_nccl_bytes"]=data_bytes.into();
 out["remote_submitted_nccl_metadata_bytes"]=metadata_bytes.into();
 out["remote_nccl_send_calls"]=sends.into();
 out["global_sync"]=serde_json::json!(self.global_sync.get());
 out["hash_lookup_series_limit"]=serde_json::json!(self.hash_lookup_limit);
 out["hash_snapshot_installations"]=self.hash_snapshot_installations.into();
 out["hash_snapshot_storage_evictions"]=self.hash_snapshot_storage_evictions.into();
 out["hash_snapshot_rebuild_peak_bytes"]=self.hash_snapshot_rebuild_peak_bytes.into();
 out["hash_snapshot_mapped_host_installations"]=self.hash_snapshot_mapped_host_installations.into();
 out["hash_snapshot_mapped_host_bytes"]=self.hash_shards.iter().flatten().flat_map(|item|&item.snapshots)
  .map(|snapshot|snapshot.storage.mapped_bytes()).sum::<u64>().into();
 out["hash_snapshot_rows_by_shard"]=serde_json::json!(self.hash_shards.iter().map(|s|
  s.as_ref().map_or(0,|item|item.snapshots.iter().map(|snap|u64::from(snap.view.length)).sum::<u64>())).collect::<Vec<_>>());
 out["hash_snapshot_bytes"]=serde_json::json!(self.hash_shards.iter().flatten().flat_map(|item|&item.snapshots)
  .map(|snap|snap.storage.device_bytes()).sum::<u64>());
 out["hash_reclaimed_segments"]=self.hash_reclaimed_segments.into();
 out["hash_segment_reuses"]=self.hash_segment_reuses.into();
 out["hash_reclaimed_device_bytes"]=self.hash_reclaimed_device_bytes.into();
 out["batch_hash_prefix_probe"]=(self.bitmap.is_none()&&self.strategy==SegmentStrategy::Batch
   &&std::env::var("MGBFS_SEGMENT_HASH_PREFIX_PROBE").as_deref()==Ok("1")).into();
 out["packet_transport"]=self.packet.as_ref().map_or(serde_json::Value::Null,|packet|
  packet.diagnostics(self.send[0].records.bytes+self.send[0].states.bytes+self.send[0].counts.bytes));
 out["hash_miss_histogram_device_bytes"]=serde_json::json!(
  self.lanes.iter().filter_map(|lane|lane.timing.as_ref()).map(|timing|timing.misses.bytes).sum::<usize>());
 out["batch_hash_timing"]=if let Some((low,high))=self.hash_timing_bounds{
  let denom=self.hash_owner_ms+self.hash_merge_ms;
  let lookup_directory_runs=self.hash_lookup_directory_runs.summary();
  let exact=self.lanes.iter().filter_map(|lane|lane.timing.as_ref()).any(|timing|timing.exact.is_some());
  let mut timing=serde_json::json!({"target_depth_min":low,"target_depth_max":high,"owner_lane_ms":self.hash_owner_ms,"unique_probe_lane_ms":self.hash_lookup_ms,"merge_lane_ms":self.hash_merge_ms,"unique_probe_plus_merge_fraction_of_owner_and_merge":if denom>0.{Some((self.hash_lookup_ms+self.hash_merge_ms)/denom)}else{None},"tiles":lookup_directory_runs.samples,"tiles_with_snapshot":self.hash_snapshot_timed_lookups,"lookup_directory_runs":lookup_directory_runs,"lookup_directory_runs_scope":"per rank, searchable series at entry per timed lookup launch, including retries and any consolidated snapshot; directory entries scanned by a full miss; upper bound for a hit","method":"sum of CUDA-event intervals per lane; includes contention and scheduling gaps; overlaps between lanes are counted; not wall time"});
  timing[if exact{"exact_miss_runs"}else{"sampled_miss_runs"}]=self.hash_miss_summary(exact);
  timing
 }else{serde_json::Value::Null};
 out["batch_input_stats"]=if let Some((low,high))=self.input_stats_bounds{
  serde_json::json!({"target_depth_min":low,"target_depth_max":high,"valid_received":self.input_valid,
   "raw_sort_scan_elements":self.input_raw_elements,"raw_tiles":self.input_raw_tiles,
   "spill_sort_scan_elements":self.input_spill_elements,"spill_tiles":self.input_spill_tiles,
   "empty_self_source_queues":self.input_empty_self,"empty_peer_source_queues":self.input_empty_peer,
   "scope":"per-rank host counts for active shard waves; raw and spill are separate actual launches"})
 }else{serde_json::Value::Null};out}
 pub fn size_profile_events(&self)->Vec<serde_json::Value>{vec![]}
 pub fn set_size_profile(&mut self,batch:u32,gemm:bool)->Result<()>{if batch==0||batch>self.plan.batch||gemm{return Err("SEGMENTED_UNSUPPORTED_SIZE_PROFILE".into())}self.active_batch=batch;Ok(())}
 pub fn quiesce(&self)->Result<()>{
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.route_stream.ptr)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;
  for lane in &self.lanes{crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(lane.stream.ptr)})?;}Ok(())
 }
 pub fn permit_healthy_communicator_reuse(&mut self){
  if !self.comm.aborted.get() {
   self.comm.healthy.set(true);
   crate::session_cache::certify_segmented_comm();
   crate::session_cache::permit_comm(true);
  }
 }
}
// Abort before GenericNativeBfs and its stream fields are dropped. Ordinary
// Rust errors must not put a potentially poisoned communicator in the cache.
impl Drop for SegmentedDistributedBfs{fn drop(&mut self){
  if !self.comm.healthy.get()||crate::native_fault::failed(){
   self.signal_packet_failure();
  }
  // Abort failed communicators before any blocking stream drain. An
  // outstanding transfer can depend on a rank that already exited.
  if !self.comm.healthy.get()||crate::native_fault::failed(){self.comm.abort_once();}
  if self.comm.aborted.get() {
   if let Some(transport)=&self.packet {
    // Packet staging also reads the producer tile owned by this BFS.
    // Drain before field destruction, after aborting any outstanding P2P.
    unsafe{cudaStreamSynchronize(transport.stream.ptr);}
   }
  }
  // Drain prefetched routing before its frontier and queues are dropped.
  unsafe{cudaStreamSynchronize(self.route_stream.ptr);}
  if self.packet.as_ref().is_some_and(|transport|transport.client.is_some())&&!self.comm.aborted.get(){
   // Clear the borrowed cancellation pointer before caching or dropping Comm.
   unsafe{mgbfs_nccl_bind_cancel(self.comm.ptr,None,ptr::null_mut());}
  }
}}

impl SegmentedDistributedBfs {
 // Called after owner and route streams have drained and the retired bank's
 // references have been erased. CUDA allocations and their byte charges
 // return to the rank, making this capacity available to any shard.
 fn reclaim_empty_hash_segments(&mut self)->Result<u32>{
  let sorted=self.strategy==SegmentStrategy::Batch;
  type RemapFn=unsafe extern "C" fn(*mut BatchHashRun,u32,*const u32,u32,*mut u32,*mut c_void)->i32;
  let symbol=if sorted{unsafe{libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_hash_batch_remap_segments_v2\0".as_ptr().cast())}}else{ptr::null_mut()};
  let remap_runs=(!symbol.is_null()).then(||unsafe{std::mem::transmute::<*mut c_void,RemapFn>(symbol)});
  let mut released=0u64;
  let stream=self.base.stream.ptr;
  for (shard,bucket) in self.segments.iter_mut().enumerate(){
   if !bucket.iter().any(|segment|segment.banks==[0;3]){continue}
   let n=u32::try_from(bucket.len()).map_err(|_|"SEGMENTED_DESCRIPTOR_SIZE")?;
   let banks=bucket.iter().map(|segment|segment.banks).collect::<Vec<_>>();
   let remap=crate::segmented_history::segment_remap(&banks,sorted&&remap_runs.is_none());
   if !remap.contains(&u32::MAX){continue}
   for r in self.current.iter().chain(&self.previous).filter(|r|r.shard==shard){
    if remap.get(r.segment).copied().unwrap_or(u32::MAX)==u32::MAX{
     return Err("SEGMENTED_RECLAIM_LIVE_FRONTIER".into())
    }
   }
   let indices_moved=remap.iter().enumerate().any(|(i,&id)|id!=u32::MAX&&id!=i as u32);
   if let (Some(remap_runs),Some(directory))=(remap_runs,self.hash_shards[shard].as_ref()){
    // Validate removed IDs even when all retained IDs are unchanged,
    // such as when the released extent is at the end of the directory.
    if directory.host_count!=0{
     let view=directory.view.as_ref().ok_or("SEGMENTED_HASH_VIEW_MISSING")?;
     let bytes=remap.len().checked_mul(4).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
     if view.data.bytes<bytes{return Err("SEGMENTED_DESCRIPTOR_SIZE".into())}
     // Rebuild the view after remapping. Its allocation already covers the
     // temporary remap, including when the remaining device budget is zero.
     crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(view.data.ptr,remap.as_ptr().cast(),bytes,1)})?;
     let error=self.control.at::<u32>(8);
     crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(error.cast(),0,4,stream)})?;
     crate::generic_native::checked!(mgbfs_hash_batch_remap_segments_v2,unsafe{remap_runs(directory.dir.ptr.cast(),directory.host_count,view.data.ptr.cast(),n,error,stream)})?;
     if indices_moved{
      for snapshot in &directory.snapshots{
       let refs=snapshot.view.references as *mut c_void;
       crate::generic_native::checked!(mgbfs_hash_extent_remap_v1,unsafe{
        mgbfs_hash_extent_remap_v1(refs,snapshot.view.length,snapshot.view.reference_bytes,snapshot.view.row_bits,
         view.data.ptr.cast(),n,error,stream)
       })?;
      }
     }
     let mut code=0u32;
     crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync((&mut code as *mut u32).cast(),error.cast(),4,2,stream)})?;
     crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
     if code!=0{return Ok(code)}
    }
   }
   for r in self.current.iter_mut().chain(&mut self.previous).filter(|r|r.shard==shard){
    r.segment=remap[r.segment] as usize;
   }
   // Keep at most one regular-sized allocation, charged to the rank and
   // available to the next shard that grows. Smaller tail extents are freed.
   let spare_index=if self.reuse_hash_segments && self.spare_hash_segment.is_none(){
    bucket.iter().zip(&remap).position(|(seg,&id)|id==u32::MAX && seg.desc.capacity==self.layout.rows_per_bank)
   }else{None};
   let mut remap=remap;
   if let Some(i)=spare_index{
    self.spare_hash_segment=Some(bucket.remove(i));
    remap.remove(i);
    self.hash_reclaimed_segments=self.hash_reclaimed_segments.saturating_add(1);
   }
   let mut indices=remap.into_iter();
   bucket.retain(|segment|{
    if indices.next()==Some(u32::MAX){
     released+=segment.device_bytes;
     self.hash_reclaimed_segments=self.hash_reclaimed_segments.saturating_add(1);
     self.hash_reclaimed_device_bytes=self.hash_reclaimed_device_bytes.saturating_add(segment.device_bytes);
     false
    }else{true}
   });
   if let Some(cache)=self.descriptor_cache[shard].take(){released+=cache.data.bytes as u64;}
   if bucket.is_empty(){
    if let Some(directory)=self.hash_shards[shard].take(){
     if directory.host_count!=0{return Err("SEGMENTED_RECLAIM_LIVE_DIRECTORY".into())}
     released+=directory.dir.bytes as u64;
     if let Some(view)=directory.view{released+=view.data.bytes as u64;}
    }
   }else if let Some(directory)=self.hash_shards[shard].as_mut(){
    let view=directory.view.as_mut().ok_or("SEGMENTED_HASH_VIEW_MISSING")?;
    let descriptors=bucket.iter().map(Segment::sorted_view).collect::<Vec<_>>();
    let bytes=descriptors.len().checked_mul(std::mem::size_of::<BatchHashSegment>()).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
    if view.data.bytes<bytes{return Err("SEGMENTED_DESCRIPTOR_SIZE".into())}
    crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(view.data.ptr,descriptors.as_ptr().cast(),bytes,1)})?;
    view.count=descriptors.len();
   }
  }
  self.remaining_bytes=self.remaining_bytes.checked_add(released).filter(|bytes|*bytes<=self.storage_budget).ok_or("SEGMENTED_STORAGE_ACCOUNTING")?;
  Ok(0)
 }

 // The main stream and all owner lanes are drained before retiring or
 // rebuilding a segment. A shard has exactly one accepting lane at a time.
 fn maintain(&mut self,bank:u32)->Result<u32>{
  let st=self.base.stream.ptr;let error=self.control.at::<u32>(8);
  if self.bitmap.is_some(){
   for bucket in &mut self.segments{for seg in bucket{if seg.banks[bank as usize]!=0{seg.banks[bank as usize]=0;crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(seg.counts.at::<c_void>(bank as usize*8),0,8,st)})?;}}}
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;self.bank_totals[bank as usize]=0;return Ok(0)
  }
  if self.strategy==SegmentStrategy::Batch{
   if self.layout.banks==1{return Ok(0)}
   crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(error.cast(),0,4,st)})?;
   for shard in 0..self.segments.len(){
    let Some(item)=self.hash_shards[shard].as_ref() else{continue};
    if !self.segments[shard].iter().any(|seg|seg.banks[bank as usize]!=0){continue}
    let view=item.view.as_ref().ok_or("SEGMENTED_HASH_VIEW_MISSING")?;
    crate::generic_native::checked!(mgbfs_hash_batch_retire_bank_v2,unsafe{mgbfs_hash_batch_retire_bank_v2(item.dir.ptr.cast(),item.count_ptr(),item.capacity,view.data.ptr.cast(),u32::try_from(view.count).map_err(|_|"SEGMENTED_DESCRIPTOR_SIZE")?,bank,error,st)})?;
    for seg in &mut self.segments[shard]{if seg.banks[bank as usize]!=0{seg.banks[bank as usize]=0;crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(seg.counts.at::<c_void>(bank as usize*8),0,8,st)})?;}}
   }
   let mut x=0u32;crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync((&mut x as *mut u32).cast(),error.cast(),4,2,st)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
   if x!=0{return Ok(x)}
   for item in self.hash_shards.iter_mut().flatten(){
    if item.snapshots.first().map_or(false,|snapshot|snapshot.bank==bank){
     self.remaining_bytes+=item.snapshots.remove(0).storage.device_bytes();
     item.snapshot_retirements+=1;
    }else if item.snapshots.iter().any(|snapshot|snapshot.bank==bank){
     return Err("SEGMENTED_HASH_SERIES_BANK_ORDER".into())
    }
    let mut n=0u32;crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut n as *mut u32).cast(),item.count_ptr().cast(),4,2)})?;if n>item.capacity{return Err("SEGMENTED_HASH_DIRECTORY_CORRUPT".into())}item.observe(n,false);
   }
   self.bank_totals[bank as usize]=0;return self.reclaim_empty_hash_segments()
  }
  crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(error.cast(),0,4,st)})?;
  for bucket in &mut self.segments{for seg in bucket{
   let count=if self.layout.banks==3{seg.banks[bank as usize]}else{0};if count!=0{
    seg.banks[bank as usize]=0;
    // Entirely retired segments are freed below. Their private hash table
    // and device counters do not need to be rewritten first.
    if seg.banks==[0;3]{continue}
    crate::generic_native::checked!(mgbfs_segmented_retire,unsafe{mgbfs_segmented_retire(seg.desc,bank,count,error,st)})?;
    seg.retired+=u64::from(count);
    crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(seg.counts.at::<c_void>(bank as usize*8),0,8,st)})?;
   }
   if seg.retired>=seg.desc.slots_count/4{
    let c=seg.banks;let status=unsafe{if self.plan.state_bytes==1{mgbfs_segmented_rehash_u8(self.plan.elements,seg.desc,c[0],c[1],c[2],self.seed,self.bits,error,st)}else{mgbfs_segmented_rehash_i64(self.plan.elements,seg.desc,c[0],c[1],c[2],self.seed,self.bits,error,st)}};
    crate::generic_native::checked!(mgbfs_segmented_rehash,status)?;seg.retired=0;
   }
  }}
  let mut x=0u32;crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync((&mut x as *mut u32).cast(),error.cast(),4,2,st)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
  if x!=0{return Ok(x)}
  if self.layout.banks==3{
   self.bank_totals[bank as usize]=0;
   return self.reclaim_empty_hash_segments();
  }
  Ok(0)
 }

 // A published tail may now satisfy the size rule again. All first merges
 // have drained; repeat only successful tails, parallel over independent shards.
 // No host intervention occurs between prepare, state write and publication.
 fn input_stats_active(&self)->bool{self.input_stats_bounds.map_or(false,|(a,b)|{let depth=self.depth.saturating_add(1);a<=depth&&depth<=b})}
 fn hash_timing_active(&self)->bool{self.hash_timing_bounds.map_or(false,|(a,b)|{let depth=self.depth.saturating_add(1);a<=depth&&depth<=b})}
 fn compact_hash_more(&mut self,prepared:&[Option<OwnerPrepared>],progress:&[OwnerProgress],active:&mut [bool])->Result<()>{
  let timed=self.hash_timing_active();
  while active.iter().any(|&v|v){
   let mut launched=vec![false;active.len()];
   let submit=(||->Result<()>{
    for i in 0..active.len(){
     if !active[i]{continue}
     let item=self.hash_shards[progress[i].shard].as_ref().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
     let replaced=item.replaced_runs();
     if replaced!=0&&item.host_count.saturating_sub(replaced)<2{active[i]=false;continue}
     let d=prepared[i].as_ref().ok_or("SEGMENTED_HASH_VIEW_MISSING")?;
     let lane=&self.lanes[i];let st=lane.stream.ptr;
     let scratch=lane.scratch.as_ref().ok_or("SEGMENTED_BATCH_LANE_SCRATCH_MISSING")?;
     crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(lane.control.at::<c_void>(8),0,8,st)})?;
     if timed{lane.timing.as_ref().ok_or("SEGMENTED_HASH_TIMING_MISSING")?.record(4,st)?;}
     crate::generic_native::checked!(mgbfs_hash_batch_compact_tail_v2,unsafe{mgbfs_hash_batch_compact_tail_v2(self.plan.state_bytes,self.plan.elements,d.hash_segments,d.hash_count,d.hash_runs,d.hash_run_count,self.batch_merge_rows,scratch.ptr,scratch.bytes,lane.control.at(8),lane.control.at(12),st)})?;
     launched[i]=true;
     if timed{lane.timing.as_ref().ok_or("SEGMENTED_HASH_TIMING_MISSING")?.record(5,st)?;}
     if let Some(host)=&lane.host{
      crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(host.ptr.cast(),lane.control.at::<c_void>(8),8,2,st)})?;
      crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(host.ptr.add(2).cast(),d.hash_run_count.cast(),4,2,st)})?;
     }
    }Ok(())
   })();
   let mut failure=submit.err();
   for(i,lane)in self.lanes.iter().enumerate().take(active.len()){
    if launched[i]{if let Err(e)=crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(lane.stream.ptr)}){if failure.is_none(){failure=Some(e)}}}
   }
   if let Some(e)=failure{return Err(e)}
   for i in 0..active.len(){
    if !launched[i]{continue}
    let lane=&self.lanes[i];let d=prepared[i].as_ref().ok_or("SEGMENTED_HASH_VIEW_MISSING")?;let mut values=[0u32;3];
    if timed{self.hash_merge_ms+=lane.timing.as_ref().ok_or("SEGMENTED_HASH_TIMING_MISSING")?.elapsed(4,5)?;}
    if let Some(host)=&lane.host{unsafe{ptr::copy_nonoverlapping(host.ptr,values.as_mut_ptr(),3);}}
    else{
     crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),lane.control.at::<c_void>(8),8,2)})?;
     crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut values[2] as *mut u32).cast(),d.hash_run_count.cast(),4,2)})?;
    }
    if values[1]!=0{return Err(format!("SEGMENTED_HASH_COMPACTION_ERROR_{}",values[1]))}
    let item=self.hash_shards[progress[i].shard].as_mut().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
    if values[2]>item.capacity{return Err("SEGMENTED_HASH_DIRECTORY_CORRUPT".into())}
    if values[0]==1&&values[2].checked_add(1)!=Some(item.host_count){return Err("SEGMENTED_HASH_COMPACTION_COUNT".into())}
    item.observe(values[2],timed);
    match values[0]{
     0=>active[i]=false,
     1=>{item.merged_runs+=1;active[i]=true},
     2=>{item.deferred_merges+=1;active[i]=false},
     _=>return Err("SEGMENTED_HASH_COMPACTION_RESULT".into())
    }
   }
  }
  Ok(())
 }
 fn accept_parallel(&mut self,slot:usize,shards:&[usize],bank:u32,visible:u32,probe:Option<&mut OverlapProbe>)->Result<u32>{
  let local=self.send[slot].side();
  let remote=if self.plan.world==1{local}else{self.recv.side()};
  let inputs=OwnerInputs::Routed{local,remote,shards,visible};
  self.accept_input(inputs,bank,probe)
 }
 fn accept_input(&mut self,inputs:OwnerInputs<'_>,bank:u32,probe:Option<&mut OverlapProbe>)->Result<u32>{
  self.accept_input_with_progress(inputs,bank,probe,None)
 }
 fn accept_input_with_progress(
  &mut self,inputs:OwnerInputs<'_>,bank:u32,mut probe:Option<&mut OverlapProbe>,
  mut pump:Option<&mut dyn FnMut(&[(usize,*mut c_void)])->Result<()>>,
 )->Result<u32>{
  let p=self.plan.clone();
  let (shard_count,routed_rows,packet_config)=match &inputs{
   OwnerInputs::Routed{shards,visible,..}=>{
    let rows=p.world.checked_mul(*visible).ok_or("SEGMENTED_QUEUE_EXTENT")?;
    (shards.len(),rows,None)
   },
   OwnerInputs::Compact{shards,config}=>(shards.len(),0,Some(*config)),
  };
  debug_assert!(packet_config.is_none()||self.batch_offsets.is_none());
  let timed=self.hash_timing_active();
  for wave_start in (0..shard_count).step_by(self.lanes.len()){
   let wave_end=(wave_start+self.lanes.len()).min(shard_count);
   self.peak_owner_lanes=self.peak_owner_lanes.max((wave_end-wave_start) as u32);
   let mut progress=(wave_start..wave_end).map(|index|->Result<OwnerProgress>{
    let (shard,selected_max)=match &inputs{
     OwnerInputs::Routed{shards,..}=>(shards[index],routed_rows),
     OwnerInputs::Compact{shards,..}=>(shards[index].shard,shards[index].rows),
    };
    // When every bank is full, inspect existing history first. A chunk
    // can consist entirely of visited candidates and require no new rows.
    // New candidates spill and cause growth in the following round.
    let segments=&self.segments[shard];
    let pass=segments.iter().position(|v|v.banks[bank as usize]<v.desc.capacity)
     .unwrap_or_else(||segments.len().saturating_sub(1));
    let tile=if self.strategy==SegmentStrategy::Batch{self.batch_tile}else{selected_max.max(1)};
    let source_total=if self.batch_offsets.is_some(){
     self.batch_offsets_host[shard*(p.world as usize+1)+p.world as usize]
    }else{selected_max};
    let mut raw=RawTileCursor::new(source_total,tile).map_err(str::to_owned)?;
    let(raw_begin,selected)=raw.next_tile().unwrap_or((0,0));
    Ok(OwnerProgress{shard,pass,round:0,selected,cap_probe:false,grow_runs:false,raw_begin,raw})
   }).collect::<Result<Vec<_>>>()?;
   while progress.iter().any(|v|v.selected!=0){
    // Prepare all descriptors and segment allocations before starting any lane.
    // cudaMemsetAsync initializes new segments on the main stream; that stream
    // must reach the barrier below before another stream observes them.
    let mut prepared:Vec<Option<OwnerPrepared>>=(0..progress.len()).map(|_|None).collect();
    for (i,v) in progress.iter_mut().enumerate(){
     if v.selected==0{continue}
     let inserted=v.pass==self.segments[v.shard].len();
      let resolved=if inserted{
       let spilled_rows=if v.round==0{0}else{v.selected};
       self.make_segment_prepared(v.shard,v.grow_runs,spilled_rows)?
      }else{
       self.owner_prepared(v.shard,v.grow_runs)?
      };
      let Some(view)=resolved else{
       eprintln!("SEGMENTED_RESOURCE rank={} phase=segment_or_descriptor_allocation detail={}",self.rank,self.last_resource.as_ref().map_or("unknown".to_owned(),|v|v.to_string()));return Ok(10)
      };
      v.grow_runs=false;prepared[i]=Some(view);
    }
    crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;
    if let Some(max_lookup_series)=self.hash_lookup_limit{
     // Preparation above can evict an index. Recheck each launching shard
     // after all allocations. Rebuilding a snapshot only replaces snapshots
     // of that shard, so successful consolidation needs one pass.
     for v in progress.iter().filter(|v|v.selected!=0){
      let shard=v.shard;
      let series=self.hash_shards[shard].as_ref().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?.searched_series();
      if series<=max_lookup_series{continue}
      self.refresh_hash_series_snapshots(2,Some(shard))?;
      let series=self.hash_shards[shard].as_ref().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?.searched_series();
      if series>max_lookup_series{
       self.last_resource=Some(serde_json::json!({
        "reason":"hash_lookup_series_limit","rank":self.rank,
        "completed_depth":self.depth,"shard":shard,
        "series":series,"limit":max_lookup_series,
       }));
       return Ok(10)
      }
     }
    }
    if self.input_stats_active(){
     for v in &progress{if v.selected!=0{
      if v.round==0{
       self.input_raw_tiles=self.input_raw_tiles.checked_add(1).ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;
       self.input_raw_elements=self.input_raw_elements.checked_add(u64::from(v.selected)).ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;
      }else{
       self.input_spill_tiles=self.input_spill_tiles.checked_add(1).ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;
       self.input_spill_elements=self.input_spill_elements.checked_add(u64::from(v.selected)).ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;
      }
     }}
    }
    let mut launched=vec![false;progress.len()];
    let mut merge_launched=vec![false;progress.len()];
    let launch_result=(||->Result<()>{
     for (i,v) in progress.iter().enumerate(){
      if v.selected==0{continue}
      let lane=&self.lanes[i];let st=lane.stream.ptr;let active=&self.segments[v.shard][v.pass];let d=prepared[i].as_ref().ok_or("SEGMENTED_DESCRIPTOR_MISSING")?;
      // Drain this lane even when a later submission fails: the mapping kernel,
      // memset and events may already be in flight.
      launched[i]=true;
      if i==0&&v.round==0{if let Some(q)=probe.as_deref_mut(){q.start_owner(st)?;}}
      if timed{lane.timing.as_ref().ok_or("SEGMENTED_HASH_TIMING_MISSING")?.record(0,st)?;}
      let selection:*const u32=if v.round==0{
       if let Some(offsets)=&self.batch_offsets{
        let bytes=v.shard.checked_mul(p.world as usize+1).and_then(|x|x.checked_mul(4)).ok_or("SEGMENTED_OFFSET_SIZE")?;
        let out=lane.spill[1].ptr.cast::<u32>();
        let f=self.batch_dense_origins.ok_or("SEGMENT_STRATEGY_BATCH_NATIVE_ABI_MISSING")?;
        crate::generic_native::checked!(mgbfs_segmented_dense_origins,unsafe{f(offsets.at(bytes),p.world,p.queue_capacity,v.raw_begin,v.selected,out,st)})?;
        out.cast_const()
       }else{ptr::null()}
      }else{lane.spill[(v.round-1)%2].ptr.cast()};
      let raw_begin=if v.round==0&&selection.is_null(){v.raw_begin}else{0};
      let sealed=d.sealed;
      crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(lane.control.ptr,0,12,st)})?;
      let (local,remote,route_rank,route_world,route_shard,route_shards,route_capacity,visible)=match &inputs{
       OwnerInputs::Routed{local,remote,visible,..}=>
        (*local,*remote,self.rank,p.world,v.shard as u32,p.shards,p.queue_capacity,*visible),
       OwnerInputs::Compact{shards,..}=>{
        let input=shards[wave_start+i];
        (input.queue,input.queue,0,1,0,1,input.rows,input.rows)
       },
      };
      let byte_wire=match &inputs{
       OwnerInputs::Compact{shards,config} if shards[wave_start+i].byte_wire=>config.byte_wire_owner,
       _=>None,
      };
      let status=unsafe{if let Some((bits,owner))=byte_wire{
       owner(bits,p.elements,u32::from(self.strategy==SegmentStrategy::Warp),local.records,route_capacity,
        selection,v.selected,sealed,d.descriptor_count,active.desc,bank,active.counts.at(bank as usize*8),
        lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),
        if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},
        self.capacity_limit.unwrap_or(0),st)
      }else if let Some(b)=&self.bitmap{
       if let Some(packed)=self.pack4{
         let owner=if let Some(config)=packet_config{
          config.pack4_owner.ok_or("SEGMENT_PACKET_PACK4_OWNER_ABI_UNAVAILABLE")?
         }else{packed.fns.owner};
         owner(Pack4OwnerArgs{
         dictionary:packed.dictionary,elements:p.elements,state_bytes:p.state_bytes,packed:u32::from(p.packed_candidates()),
         local:local.states,remote:remote.states,
         local_meta:local.records.cast(),remote_meta:remote.records.cast(),
         local_counts:local.counts.cast(),remote_counts:remote.counts.cast(),
          rank:self.rank,world:p.world,owner:b.owner,total:b.domain.total,owner_len:b.owner_len,bits:b.bits.ptr.cast(),
          shard:v.shard as u32,shards:p.shards,q:route_capacity,visible,selection,selected:v.selected,
         capacity:active.desc.capacity,banks:active.desc.banks,bank,rows:active.desc.states.cast(),
         accepted:active.counts.at(bank as usize*8),global_count:if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},
         global_limit:self.capacity_limit.unwrap_or(0),spill:lane.spill[v.round%2].ptr.cast(),
         spill_count:lane.control.at(0),error:lane.control.at(4)},st)
        }else if let Some(config)=packet_config{
         let scratch=lane.scratch.as_ref().map_or((ptr::null_mut(),0),|scratch|(scratch.ptr,scratch.bytes));
         let args=BitmapPacketOwnerArgs{
          state_bytes:p.state_bytes,packed:u32::from(p.packed_candidates()),elements:p.elements,
          states:local.states,records:local.records.cast(),counts:local.counts,
          world:p.world,owner:b.owner,shard:v.shard as u32,shards:p.shards,total:b.domain.total,
          owner_len:b.owner_len,bits:b.bits.ptr.cast(),capacity:route_capacity,visible,selection,
          selected:v.selected,raw_begin,active:active.desc,bank,accepted:active.counts.at(bank as usize*8),
          spill:lane.spill[v.round%2].ptr.cast(),spill_count:lane.control.at(0),error:lane.control.at(4),
          global_count:if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},
          limit:self.capacity_limit.unwrap_or(0),scratch:scratch.0,scratch_bytes:scratch.1,
         };
         if self.strategy==SegmentStrategy::Batch{
          let f=config.bitmap_batch.ok_or("SEGMENT_PACKET_BITMAP_OWNER_ABI_UNAVAILABLE")?;
          f(&args,st)
         }else{
          let f=config.bitmap_owner.ok_or("SEGMENT_PACKET_BITMAP_OWNER_ABI_UNAVAILABLE")?;
          f(&args,u32::from(self.strategy==SegmentStrategy::Warp),st)
         }
        }else if self.strategy==SegmentStrategy::Batch{
        let scratch=lane.scratch.as_ref().ok_or("SEGMENTED_BATCH_LANE_SCRATCH_MISSING")?;
        mgbfs_bitmap_accept_batch(p.state_bytes,u32::from(p.packed_candidates()),p.elements,local.states,remote.states,local.records.cast(),remote.records.cast(),local.counts.cast(),remote.counts.cast(),route_rank,route_world,b.owner,b.domain.total,b.owner_len,b.bits.ptr.cast(),route_shard,route_shards,route_capacity,visible,selection,v.selected,raw_begin,active.desc,bank,active.counts.at(bank as usize*8),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),scratch.ptr,scratch.bytes,st)
       }else if self.strategy==SegmentStrategy::Warp{mgbfs_bitmap_accept_grouped(p.state_bytes,u32::from(p.packed_candidates()),p.elements,local.states,remote.states,local.records.cast(),remote.records.cast(),local.counts.cast(),remote.counts.cast(),route_rank,route_world,b.owner,b.domain.total,b.owner_len,b.bits.ptr.cast(),route_shard,route_shards,route_capacity,visible,selection,v.selected,active.desc,bank,active.counts.at(bank as usize*8),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)}else{mgbfs_bitmap_accept(p.state_bytes,u32::from(p.packed_candidates()),p.elements,local.states,remote.states,local.records.cast(),remote.records.cast(),local.counts.cast(),remote.counts.cast(),route_rank,route_world,b.owner,b.domain.total,b.owner_len,b.bits.ptr.cast(),route_shard,route_shards,route_capacity,visible,selection,v.selected,active.desc,bank,active.counts.at(bank as usize*8),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)}
      }else if self.strategy==SegmentStrategy::Batch{
       let scratch=lane.scratch.as_ref().ok_or("SEGMENTED_BATCH_LANE_SCRATCH_MISSING")?;
       let directory=self.hash_shards[v.shard].as_ref().map(|item|item.snapshot_directory());
       let view=directory.as_ref().map_or(ptr::null(),|view|view as *const HashSeriesDirectory);
       let timing=lane.timing.as_ref().filter(|_|timed);
       macro_rules! accept_batch {
        ($accept:expr $(,$extra:expr)*) => {
         ($accept)(p.state_bytes,u32::from(p.packed_candidates()),p.elements,
          local.states,remote.states,local.records.cast(),remote.records.cast(),local.counts.cast(),remote.counts.cast(),
          route_rank,route_world,route_shard,route_shards,route_capacity,visible,selection,v.selected,raw_begin,
          d.hash_segments,d.hash_count,u32::try_from(v.pass).map_err(|_|"SEGMENTED_DESCRIPTOR_SIZE")?,
          bank,d.hash_runs,d.hash_run_count,d.hash_run_capacity,active.counts.at(bank as usize*8),
          lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),
          if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},
          self.capacity_limit.unwrap_or(0),scratch.ptr,scratch.bytes,
          timing.map_or(ptr::null_mut(), |timing| timing.events[2]),
          timing.map_or(ptr::null_mut(), |timing| timing.events[3]),
          timing.map_or(ptr::null_mut(), |timing| timing.misses.ptr.cast()),st,view,
          self.hash_lookup_limit.unwrap_or(0) $(,$extra)*)
        };
       }
       if let Some(nearest)=&self.nearest{
        accept_batch!(nearest.fns.accept,active.parents,p.world,self.base.generators)
       }else{
        let accept=timing.and_then(|timing|timing.exact).unwrap_or(mgbfs_hash_batch_accept_profiled_v6);
        accept_batch!(accept)
       }
      }else if p.state_bytes==1{
       if self.strategy==SegmentStrategy::Warp{mgbfs_segmented_accept_routes_limited_grouped_u8(p.elements,local.states.cast(),remote.states.cast(),local.records.cast(),remote.records.cast(),local.counts.cast(),remote.counts.cast(),route_rank,route_world,route_shard,route_shards,route_capacity,visible,u32::from(p.packed_candidates()),selection,v.selected,sealed,d.descriptor_count,active.desc,bank,active.counts.at(bank as usize*8),ptr::null_mut(),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)}else{mgbfs_segmented_accept_routes_limited_u8(p.elements,local.states.cast(),remote.states.cast(),local.records.cast(),remote.records.cast(),local.counts.cast(),remote.counts.cast(),route_rank,route_world,route_shard,route_shards,route_capacity,visible,u32::from(p.packed_candidates()),selection,v.selected,sealed,d.descriptor_count,active.desc,bank,active.counts.at(bank as usize*8),ptr::null_mut(),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)}
      }else{
       if self.strategy==SegmentStrategy::Warp{mgbfs_segmented_accept_routes_limited_grouped_i64(p.elements,local.states.cast(),remote.states.cast(),local.records.cast(),remote.records.cast(),local.counts.cast(),remote.counts.cast(),route_rank,route_world,route_shard,route_shards,route_capacity,visible,selection,v.selected,sealed,d.descriptor_count,active.desc,bank,active.counts.at(bank as usize*8),ptr::null_mut(),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)}else{mgbfs_segmented_accept_routes_limited_i64(p.elements,local.states.cast(),remote.states.cast(),local.records.cast(),remote.records.cast(),local.counts.cast(),remote.counts.cast(),route_rank,route_world,route_shard,route_shards,route_capacity,visible,selection,v.selected,sealed,d.descriptor_count,active.desc,bank,active.counts.at(bank as usize*8),ptr::null_mut(),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)}
      }};
      crate::generic_native::checked!(mgbfs_segmented_accept_routes,status)?;
      if timed{lane.timing.as_ref().ok_or("SEGMENTED_HASH_TIMING_MISSING")?.record(1,st)?;}
      if !d.hash_run_count.is_null()&&self.batch_merge_enabled{
       let item=self.hash_shards[v.shard].as_ref().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
       // A directory-full tile spills all fresh candidates without
       // publishing. Tail compaction can free an entry before replay.
       // Never merge the first unindexed append with a run already in
       // the snapshot. Both input runs must be strictly inside the tail.
       let replaced=item.replaced_runs();
       if (replaced==0&&item.host_count>0)||item.host_count.saturating_sub(replaced)>=2{
        let scratch=lane.scratch.as_ref().ok_or("SEGMENTED_BATCH_LANE_SCRATCH_MISSING")?;
        if timed{lane.timing.as_ref().ok_or("SEGMENTED_HASH_TIMING_MISSING")?.record(4,st)?;}
        crate::generic_native::checked!(mgbfs_hash_batch_compact_tail_v2,unsafe{mgbfs_hash_batch_compact_tail_v2(p.state_bytes,p.elements,d.hash_segments,d.hash_count,d.hash_runs,d.hash_run_count,self.batch_merge_rows,scratch.ptr,scratch.bytes,lane.control.at(8),lane.control.at(4),st)})?;
        merge_launched[i]=true;
        if timed{lane.timing.as_ref().ok_or("SEGMENTED_HASH_TIMING_MISSING")?.record(5,st)?;}
       }
      }
      if i==0&&v.round==0{if let Some(q)=probe.as_deref_mut(){q.end_owner(st)?;}}
      if let Some(host)=&lane.host{
       crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(host.ptr.cast(),lane.control.ptr,8,2,st)})?;
       crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(host.ptr.add(2).cast(),active.counts.at::<c_void>(bank as usize*8),8,2,st)})?;
       if !d.hash_run_count.is_null(){
        crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(host.ptr.add(4).cast(),d.hash_run_count.cast(),4,2,st)})?;
        crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(host.ptr.add(5).cast(),lane.control.at::<c_void>(8),4,2,st)})?;
       }
      }
     }Ok(())
    })();
    // Drain every launched lane even if another launch failed, before its
    // descriptors or input queues can be freed or recycled.
    let mut failure=launch_result.err();
    let mut all_ready=false;
    if failure.is_none(){
     if let Some(pump)=pump.as_mut(){
      let started=std::time::Instant::now();
      let mut pending:Vec<_>=self.lanes.iter().zip(&launched).zip(&progress)
       .filter_map(|((lane,&submitted),entry)|submitted.then_some((entry.shard,lane.stream.ptr))).collect();
      while !pending.is_empty(){
       pending.retain(|&(_,stream)|match unsafe{cudaStreamQuery(stream)}{
        0=>false,
        600=>true,
        status=>{
         if failure.is_none(){failure=Some(format!("SEGMENTED_OWNER_STREAM_QUERY_{status}"));}
         false
        }
       });
       if failure.is_some()||pending.is_empty(){break}
       if let Err(error)=pump(&pending){failure=Some(error);break}
       if started.elapsed()<std::time::Duration::from_micros(100){std::thread::yield_now();}
       else{std::thread::sleep(std::time::Duration::from_micros(50));}
      }
      all_ready=failure.is_none();
     }
    }
    if !all_ready{
     for (i,lane) in self.lanes.iter().enumerate().take(progress.len()){
      if launched[i]{if let Err(e)=crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(lane.stream.ptr)}){if failure.is_none(){failure=Some(e)}}}
     }
    }
    if let Some(e)=failure{return Err(e)}
    let mut owner_error=0;let mut compact_repeat=vec![false;progress.len()];
    for (i,v) in progress.iter_mut().enumerate(){
     if !launched[i]{continue}
     let lane=&self.lanes[i];let active=&self.segments[v.shard][v.pass];let mut values=[0u32;6];
     if timed{
      let t=lane.timing.as_ref().ok_or("SEGMENTED_HASH_TIMING_MISSING")?;
      self.hash_owner_ms+=t.elapsed(0,1)?;self.hash_lookup_ms+=t.elapsed(2,3)?;
      if merge_launched[i]{self.hash_merge_ms+=t.elapsed(4,5)?;}
     }
     if let Some(host)=&lane.host{unsafe{ptr::copy_nonoverlapping(host.ptr,values.as_mut_ptr(),if prepared[i].as_ref().unwrap().hash_run_count.is_null(){4}else{6});}}
     else{
      crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),lane.control.ptr,8,2)})?;
      crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut values[2] as *mut u32).cast(),active.counts.at::<c_void>(bank as usize*8),8,2)})?;
       let d=prepared[i].as_ref().ok_or("SEGMENTED_DESCRIPTOR_MISSING")?;
       if !d.hash_run_count.is_null(){
       crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut values[4] as *mut u32).cast(),d.hash_run_count.cast(),4,2)})?;
       crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut values[5] as *mut u32).cast(),lane.control.at::<c_void>(8),4,2)})?;
      }
     }
     if self.layer_stats_active(){
      // Spill records include retries, so this is traffic, not unique states.
      self.profile_spill=self.profile_spill.saturating_add(u64::from(values[0]));
      self.profile_owner_passes=self.profile_owner_passes.saturating_add(1);
     }
     let accepted=u64::from(values[2])|(u64::from(values[3])<<32);
     let sorted=self.strategy==SegmentStrategy::Batch&&self.bitmap.is_none();
     let allowed=if sorted{8|16|256}else if self.pack4.is_some()||matches!(self.strategy,SegmentStrategy::Warp|SegmentStrategy::Batch){8|16}else{8};
     if values[0]>v.selected||accepted>active.desc.capacity||(values[1]&!allowed)!=0{owner_error=owner_error.max(100+(values[1]&!allowed).max(1));continue}
     if sorted{
      let hs=self.hash_shards[v.shard].as_mut().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
      if values[4]>hs.capacity{owner_error=owner_error.max(104);continue}
      // This shard's single lane has drained. The host count still describes
      // the lookup's input directory, before this tile's publish or merge.
      let series=hs.searched_series();
      hs.lookup_max_series=hs.lookup_max_series.max(series);
      if timed{
       self.hash_lookup_directory_runs.observe(series)?;
       self.hash_snapshot_timed_lookups+=u64::from(!hs.snapshots.is_empty());
      }
      hs.observe(values[4],timed);
      match values[5]{0=>{},1=>{hs.merged_runs+=1;compact_repeat[i]=true},2=>hs.deferred_merges+=1,_=>{owner_error=owner_error.max(106);continue}}
     }
     let seg=&mut self.segments[v.shard][v.pass];let old=seg.banks[bank as usize];
     if accepted<old{owner_error=owner_error.max(101);continue}
     self.bank_totals[bank as usize]=self.bank_totals[bank as usize].checked_add(accepted-old).ok_or("SEGMENTED_BANK_TOTAL_OVERFLOW")?;
     let live=self.bank_totals.iter().try_fold(0u64,|a,&v|a.checked_add(v).ok_or("SEGMENTED_LIVE_COUNT_OVERFLOW"))?;self.peak_live_rows=self.peak_live_rows.max(live);
     seg.banks[bank as usize]=accepted;
     if values[1]&256!=0{
      if !sorted||values[1]&(8|16)!=0||values[0]==0||accepted!=old{owner_error=owner_error.max(105);continue}
      v.selected=values[0];v.round+=1;v.grow_runs=true;continue
     }
     if values[1]&8!=0{
      let mut observed=0u64;crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut observed as *mut u64).cast(),self.control.at::<c_void>(24),8,2)})?;
      if let Some(limit)=self.capacity_limit{
       eprintln!("SEGMENTED_CAPACITY_OBSERVED rank={} limit={} count={}",self.rank,limit,observed);
       if observed>limit{owner_error=owner_error.max(199)}else{self.last_resource=Some(serde_json::json!({"reason":"user_capacity","rank":self.rank,"completed_depth":self.depth,"target_depth":self.depth.saturating_add(1),"shard":v.shard,"bank":bank,"limit_states":limit,"observed_states":observed}));owner_error=owner_error.max(10)}
      }else{owner_error=owner_error.max(199)}continue
     }
     // Upper bound on possible tombstones, used for conservative rehashing.
     seg.retired=seg.retired.saturating_add(u64::from(values[0]));
     if self.pack4.is_some()||matches!(self.strategy,SegmentStrategy::Warp|SegmentStrategy::Batch){
      let mut observed=0u64;
      if self.capacity_limit.is_some(){crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut observed as *mut u64).cast(),self.control.at::<c_void>(24),8,2)})?;}
      let decision=owner_decision(values[0],self.capacity_limit,observed,accepted==seg.desc.capacity,values[1]&16!=0,v.cap_probe,accepted>old).map_err(str::to_owned)?;
      v.selected=values[0];
      match decision{
       OwnerDecision::Done=>{
        // Never advance the raw cursor until the filtered spill is drained.
        // Keep the same bank even if full: the new tile may be duplicate-only.
        if let Some((begin,count))=v.raw.next_tile(){
         v.raw_begin=begin;v.selected=count;v.round=0;v.cap_probe=false;
        }
       },
       OwnerDecision::Same(probe)=>{v.cap_probe=probe;v.round+=1;},
       OwnerDecision::Next=>{v.cap_probe=false;v.pass+=1;v.round+=1;},
       OwnerDecision::Exhausted=>{
        let limit=self.capacity_limit.ok_or("SEGMENT_CAP_RETRY_WITHOUT_LIMIT")?;
        self.last_resource=Some(serde_json::json!({"reason":"user_capacity","rank":self.rank,"completed_depth":self.depth,"target_depth":self.depth.saturating_add(1),"shard":v.shard,"bank":bank,"limit_states":limit,"observed_states":observed}));
        owner_error=owner_error.max(10);
       }
      }
     }else{v.selected=values[0];v.pass+=1;v.round+=1;}
    }
    if owner_error!=0{return Ok(owner_error)}
    self.compact_hash_more(&prepared,&progress,&mut compact_repeat)?;
    for v in progress.iter_mut().filter(|v| v.grow_runs) {
     let directory=self.hash_shards[v.shard].as_ref().ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
     v.grow_runs=directory.host_count>=directory.capacity;
    }
   }
  }
  Ok(0)
 }
 // The sender slot is independent of the previous slot leased to owners.
 // Routing may run while those owners deduplicate; the slot is never reused
 // until the previous accept_parallel call has drained every owner lane.
 fn start_route(&self,slot:usize,work:&[Range],wi:usize,cursor:u64,batch:u32)->Result<()>{
  let st=self.route_stream.ptr;let p=&self.plan;let send=&self.send[slot];let errors=&self.route_errors[slot];
  crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(send.counts.ptr,0,send.counts.bytes,st)})?;
  crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(errors.ptr,0,12,st)})?;
  if wi<work.len(){
   let r=work[wi];let seg=&self.segments[r.shard][r.segment];let count=u32::try_from(u64::from(batch).min(r.end-cursor)).map_err(|_|"SEGMENTED_BATCH_RANGE")?;
   let stride=seg.desc.capacity.checked_mul(u64::from(seg.desc.banks)).ok_or("SEGMENTED_PARENT_OFFSET")?;
   let row=u64::from(r.bank).checked_mul(seg.desc.capacity).and_then(|v|v.checked_add(cursor)).ok_or("SEGMENTED_PARENT_OFFSET")?;
   let off=row.checked_mul(if self.pack4.is_some(){8}else{u64::from(p.state_bytes)}).ok_or("SEGMENTED_PARENT_OFFSET")?;
   let parent=unsafe{seg.desc.states.cast::<u8>().add(usize::try_from(off).map_err(|_|"SEGMENTED_PARENT_OFFSET")?)};
   if let Some(packed)=self.pack4{
    let b=self.bitmap.as_ref().ok_or("BITMAP_PACK4_REQUIRES_BITMAP")?;
    crate::generic_native::checked!(mgbfs_pack4_route,unsafe{(packed.fns.route)(Pack4RouteArgs{
     dictionary:packed.dictionary,domain:b.domain,parents:parent.cast(),permutations:self.base.perms.ptr.cast(),
     count,generators:self.base.generators,world:p.world,source:self.rank,shards:p.shards,capacity:p.queue_capacity,
     state_bytes:p.state_bytes,grouped:0,map:self.map.ptr.cast(),records:send.records.ptr.cast(),payload:send.states.ptr,
     counts:send.counts.ptr.cast(),error:errors.at(0),owner_page:b.owner_page,owner_boundary:b.owner_boundary},st)})?;
   }else if self.strategy==SegmentStrategy::Batch{
    let (bitmap,word_shard,domain)=if let Some(b)=&self.bitmap{(1,1,b.domain)}else{(0,0,BitmapDeviceDomain{symbols:ptr::null(),counts:ptr::null(),distinct:0,elements:0,total:0})};
    let scratch=self.route_scratch.as_ref().ok_or("SEGMENTED_BATCH_ROUTE_SCRATCH_MISSING")?;
    if let Some(b)=self.bitmap.as_ref().filter(|b|b.immutable_routes.is_some()){
     let f=b.immutable_routes.unwrap().batch;
     crate::generic_native::checked!(mgbfs_segmented_batch_route_v2,unsafe{f(bitmap,word_shard,domain,p.state_bytes,self.base.kind,p.elements,self.base.rows,self.base.cols,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),self.base.matrices.ptr.cast(),self.base.moduli.ptr.cast(),self.seed,self.bits,p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),self.cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),send.states.ptr,scratch.ptr,scratch.bytes,st,b.owner_page,b.owner_boundary)})?;
    }else{
    crate::generic_native::checked!(mgbfs_segmented_batch_route,unsafe{mgbfs_segmented_batch_route(bitmap,word_shard,domain,p.state_bytes,self.base.kind,p.elements,self.base.rows,self.base.cols,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),self.base.matrices.ptr.cast(),self.base.moduli.ptr.cast(),self.seed,self.bits,p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),self.cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),send.states.ptr,scratch.ptr,scratch.bytes,st)})?;
    }
   }else if let Some(b)=&self.bitmap{
    if let Some(f)=b.immutable_routes{
     let call=if self.strategy==SegmentStrategy::Warp{f.grouped}else{f.normal};
     crate::generic_native::checked!(mgbfs_bitmap_route_v2,unsafe{call(p.state_bytes,b.domain,p.elements,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),st,b.owner_page,b.owner_boundary)})?;
    }else{
    if self.strategy==SegmentStrategy::Warp{crate::generic_native::checked!(mgbfs_bitmap_route_grouped,unsafe{mgbfs_bitmap_route_grouped(p.state_bytes,b.domain,p.elements,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),st)})?;}else{crate::generic_native::checked!(mgbfs_bitmap_route,unsafe{mgbfs_bitmap_route(p.state_bytes,b.domain,p.elements,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),st)})?;}
    }
   }else{
    if self.strategy==SegmentStrategy::Warp{crate::generic_native::checked!(mgbfs_segmented_route_storage_grouped,unsafe{mgbfs_segmented_route_storage_grouped(p.state_bytes,self.base.kind,p.elements,self.base.rows,self.base.cols,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),self.base.matrices.ptr.cast(),self.base.moduli.ptr.cast(),self.seed,self.bits,p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),self.cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),st)})?;}else{crate::generic_native::checked!(mgbfs_segmented_route_storage,unsafe{mgbfs_segmented_route_storage(p.state_bytes,self.base.kind,p.elements,self.base.rows,self.base.cols,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),self.base.matrices.ptr.cast(),self.base.moduli.ptr.cast(),self.seed,self.bits,p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),self.cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),st)})?;}
   }
   if self.pack4.is_none()&&!p.packed_candidates()&&self.strategy!=SegmentStrategy::Batch{
    let queues=p.world.checked_mul(p.shards).ok_or("SEGMENTED_QUEUE_EXTENT")?;
    crate::generic_native::checked!(mgbfs_generic_regenerate_all_routes_count_storage_wide_stride,unsafe{mgbfs_generic_regenerate_all_routes_count_storage_wide_stride(p.state_bytes,self.base.kind,p.elements,self.base.rows,self.base.cols,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),self.base.matrices.ptr.cast(),self.base.moduli.ptr.cast(),self.rank,0,send.records.ptr.cast(),send.counts.ptr.cast(),p.queue_capacity,queues,send.states.ptr,errors.at(4),st)})?;
   }
  }
  let queues=p.world.checked_mul(p.shards).ok_or("SEGMENTED_QUEUE_EXTENT")?;
  crate::generic_native::checked!(mgbfs_generic_max_queue_count,unsafe{mgbfs_generic_max_queue_count(send.counts.ptr.cast(),queues,errors.at(8),st)})?;
  Ok(())
 }
 fn finish_route(&self,slot:usize)->Result<[u32;3]>{
  let st=self.route_stream.ptr;let mut errors=[0u32;3];
  crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(errors.as_mut_ptr().cast(),self.route_errors[slot].ptr,12,2,st)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;Ok(errors)
 }

 pub(crate) fn uses_packet_transport(&self) -> bool { self.packet.is_some() }

 pub fn advance(&mut self)->Result<DistributedAdvance>{
  self.advance_with_stop(|| 0, &mut 0)
 }

 pub(crate) fn advance_with_stop(
  &mut self, boundary_stop: impl Fn() -> u32, next_stop: &mut u32,
 ) -> Result<DistributedAdvance> {
  *next_stop = 0;
  self.connect_packet_control()?;
  let transport_start=self.layer_stats_active().then(||self.transport_totals());
  // Exact payload exchange and padded exchange have different NCCL message
  // contracts. Agree once before the first chunk so a per-rank environment
  // mismatch fails cleanly instead of deadlocking point-to-point transfers.
   if self.depth==0&&self.packet.is_none(){
   let(exact,padded)=self.vote_route_and_max(u32::from(self.exact_exchange),u32::from(!self.exact_exchange))?;
   if exact!=0&&padded!=0{return Err("SEGMENTED_EXCHANGE_MODE_MISMATCH".into())}
  }
  if let (Ok(r),Ok(d))=(std::env::var("MGBFS_SEGMENT_TEST_FATAL_RANK"),std::env::var("MGBFS_SEGMENT_TEST_FATAL_AFTER_DEPTH")){
   if r.parse::<u32>().ok()==Some(self.rank)&&d.parse::<u64>().ok().map_or(false,|v|self.depth>=v){return Err("SEGMENTED_TEST_ONE_RANK_FAILURE".into())}
  }
  let st=self.base.stream.ptr;let p=self.plan.clone();let future=if self.layout.banks>1{((self.depth+1)%u64::from(self.layout.banks)) as u32}else{0};
  let tracing=std::env::var("MGBFS_SEGMENT_TEST_TIMING").as_deref()==Ok("1");let mut tick=std::time::Instant::now();let mut phases=[0u128;7];
  macro_rules! record{($i:expr)=>{if tracing{phases[$i]+=tick.elapsed().as_micros();tick=std::time::Instant::now();}}}
  let maintenance=(||->Result<u32>{
   self.snapshot_previous_before_reuse()?;
   let code=self.maintain(future)?;
   if code==0{
    if let Some(eager_tail)=self.hash_eager_tail{
     // The launch-time check independently enforces the hard series limit.
     self.refresh_hash_series_snapshots(eager_tail,None)?;
    }
   }
   Ok(code)
  })();
  // Reuse the existing legacy boundary vote for host failures. A rank
  // must still enter it if local maintenance returned an error.
  let local_status=match &maintenance{Ok(0)=>0,Ok(code)=>100+code,Err(_)=>99};
  // Packet peers cannot commit before this rank seals. A local failure
  // closes the job through the control channel, with no pre-layer vote.
  let status=if self.packet.is_some(){local_status}else{self.vote(local_status)?};
  maintenance?;
  if status!=0{return Err(format!("SEGMENTED_MAINTENANCE_{status}"))}
  if self.capacity_limit.is_some(){
   let initial=self.bank_totals[future as usize];
   crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(self.control.at::<c_void>(24),(&initial as *const u64).cast(),8,1)})?;
  }
  // Rolling banks start at row zero. Only retain-all storage needs the
  // previous high-water marks to delimit this layer's newly appended rows.
  let before=if self.layout.banks==1{
   self.segments.iter().map(|bucket|bucket.iter().map(|s|s.banks[0]).collect::<Vec<_>>()).collect::<Vec<_>>()
  }else{Vec::new()};
  // Maintenance has finished remapping segment IDs. Keep the committed
  // ranges intact until execution either publishes the layer or returns.
  let work=std::mem::take(&mut self.current);
  let result=(||->Result<DistributedAdvance>{
  let mut wi=0usize;let mut cursor=work.first().map_or(0,|r|r.begin);let mut batch=self.active_batch.max(1).min(self.queue_batch_ceiling);
  // A profile may choose a smaller initial batch. Check agreement once per
  // layer in opt-in mode before its exact exchange lengths can diverge.
   if self.batch_recovery&&self.packet.is_none(){
   let(maximum,complement_maximum)=self.vote_route_and_max(batch,!batch)?;
   if maximum!=!complement_maximum{return Err("SEGMENTED_BATCH_RANK_MISMATCH".into())}
  }
  let layer_retries_start=self.source_retries;
  let stats=self.layer_stats_active();
  let spill_start=self.profile_spill;let passes_start=self.profile_owner_passes;
  let stat_len=if stats{p.world as usize*p.shards as usize}else{0};
  let mut sent_counts=vec![0u64;stat_len];let mut received_counts=vec![0u64;stat_len];
  let mut parents=0u64;let mut stat_waves=0u64;let mut peak_send=0u32;let mut peak_receive=0u32;
  let mut layer_peak_queue=0u32;
   let layer_had_work=if self.packet.is_some(){
    let layer=match self.advance_packets(&work,future,&mut batch,stats,&boundary_stop)?{
     Ok(layer)=>layer,
     Err(code)=>{
      *next_stop=match code{PACKET_CANCELLED=>1,PACKET_DEADLINE=>2,_=>0};
      return Ok(DistributedAdvance::Resource{fatal:code})
     }
    };
    parents=layer.parents;stat_waves=layer.waves;layer_peak_queue=layer.max_bucket;
    if stats{
     sent_counts=layer.sent;
     received_counts=layer.received;
     peak_send=layer.max_bucket;
     peak_receive=layer.max_received_bucket;
    }
    record!(4);
    !work.is_empty()
   }else{
   let mut rounds=self.global_rounds(&work,wi,cursor,batch)?;
   let layer_had_work=rounds>0;
  let mut pending_owner=0u32;let mut slot=0usize;record!(0);
  if rounds>0{self.start_route(slot,&work,wi,cursor,batch)?;}
  loop{
   if rounds==0{
    self.note_sync(|counts|counts.legacy_chunk_votes+=1);
    let global=self.vote(crate::segmented_history::route_owner_vote(0,pending_owner))?;record!(5);
    match crate::segmented_history::decode_route_owner_vote(global){
     crate::segmented_history::SegmentVote::Continue=>{},
     crate::segmented_history::SegmentVote::Resource=>return Ok(DistributedAdvance::Resource{fatal:10}),
     crate::segmented_history::SegmentVote::OwnerFatal(e)=>return Err(format!("SEGMENTED_OWNER_{e}")),
     _=>return Err("SEGMENTED_FINAL_VOTE_INVALID".into()),
    }
    break
   }
   let mut source_errors=self.finish_route(slot)?;
   #[cfg(debug_assertions)]
   if self.depth>=1 {
    if std::env::var("MGBFS_SEGMENT_TEST_VOTE_SOURCE").as_deref()==Ok("1"){source_errors[0]|=2;}
    if std::env::var("MGBFS_SEGMENT_TEST_VOTE_OWNER_RESOURCE").as_deref()==Ok("1"){pending_owner=10;}
   }
   record!(1);let classified=crate::segmented_history::classify_route_and_regeneration(source_errors[0],source_errors[1]);
   if classified>1||pending_owner!=0{eprintln!("SEGMENTED_RANK_VOTE rank={} route={} regeneration={} owner={}",self.rank,source_errors[0],source_errors[1],pending_owner);}
   self.note_sync(|counts|counts.legacy_chunk_votes+=1);
   let stop=crate::segmented_history::route_owner_stop_vote(boundary_stop());
   let local_vote=crate::segmented_history::route_owner_vote(classified,pending_owner).max(stop);
   let (source,global_max)=self.vote_route_and_max(local_vote,source_errors[2])?;pending_owner=0;record!(2);
   match crate::segmented_history::decode_route_owner_vote(source){
    crate::segmented_history::SegmentVote::Continue=>{},
    crate::segmented_history::SegmentVote::Stop(reason)=>{
     *next_stop=reason;
     return Ok(DistributedAdvance::Resource{fatal:0})
    },
    crate::segmented_history::SegmentVote::Retry if batch>1=>{
     batch=(batch/2).max(1);self.source_retries+=1;
      // The collective retry vote is identical across ranks. Retain this
      // safe upper bound for later layers of this BFS; size profiles may
      // still choose a smaller batch. A new BFS starts with a fresh bound.
      self.queue_batch_ceiling=self.queue_batch_ceiling.min(batch);
      rounds=self.global_rounds(&work,wi,cursor,batch)?;
     self.start_route(slot,&work,wi,cursor,batch)?;continue
    },
    crate::segmented_history::SegmentVote::Retry=>return Err("SEGMENTED_ROUTE_1".into()),
    crate::segmented_history::SegmentVote::Resource=>return Ok(DistributedAdvance::Resource{fatal:10}),
    crate::segmented_history::SegmentVote::OwnerFatal(e)=>return Err(format!("SEGMENTED_OWNER_{e}")),
    crate::segmented_history::SegmentVote::SourceFatal(e)=>return Err(format!("SEGMENTED_ROUTE_{e}")),
   }
   let upper=(u64::from(batch)*u64::from(self.base.generators)).min(u64::from(p.queue_capacity)) as u32;
   if global_max>upper{return Err("SEGMENTED_ROUTE_COUNT_BOUND".into())}
   layer_peak_queue=layer_peak_queue.max(global_max);
   let used=global_max;
   if self.bounds_trace{eprintln!("SEGMENT_BOUNDS rank={} depth={} local={} global={} upper={}",self.rank,self.depth,source_errors[2],global_max,upper);}
   let send=&self.send[slot];
   self.note_sync(|counts|counts.legacy_chunk_count_exchanges+=1);
   if p.world>1&&self.exact_exchange{
    // Counts establish matched per-peer lengths before payload submission.
    crate::generic_native::checked!(mgbfs_nccl_exchange_triplets,unsafe{mgbfs_nccl_exchange_triplets(self.comm.ptr,self.rank,p.world,send.counts.ptr,u64::from(p.shards)*4,ptr::null(),0,ptr::null(),0,self.recv.counts.ptr,ptr::null_mut(),ptr::null_mut(),st)})?;
   }else if p.world>1{
    if used!=0{self.note_sync(|counts|counts.legacy_chunk_payload_submissions+=1);}
    crate::generic_native::checked!(mgbfs_nccl_exchange_bounded_triplets,unsafe{mgbfs_nccl_exchange_bounded_triplets(self.comm.ptr,self.rank,p.world,p.shards,p.queue_capacity,used,p.queue_payload_bytes() as u64,p.state_bytes,send.counts.ptr,send.records.ptr,send.states.ptr,self.recv.counts.ptr,self.recv.records.ptr,self.recv.states.ptr,st)})?;
   }
   let n=p.world as usize*p.shards as usize;let mut lc=vec![0u32;n];let mut rc=vec![0u32;n];
   crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(lc.as_mut_ptr().cast(),send.counts.ptr,n*4,2,st)})?;
   if p.world>1{
    crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(rc.as_mut_ptr().cast(),self.recv.counts.ptr,n*4,2,st)})?;
   }
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
   if p.world>1&&self.exact_exchange{
    if self.test_count_payload{crate::segmented_host_ready::injected("count-payload")?;}
    self.note_sync(|counts|counts.legacy_chunk_payload_submissions+=1);
    crate::generic_native::checked!(mgbfs_nccl_exchange_exact_payload,unsafe{mgbfs_nccl_exchange_exact_payload(self.comm.ptr,self.rank,p.world,p.shards,p.queue_capacity,p.queue_payload_bytes() as u64,p.state_bytes,lc.as_ptr(),rc.as_ptr(),send.records.ptr,send.states.ptr,self.recv.records.ptr,self.recv.states.ptr,st)})?;
   }
   if stats{
    stat_waves=stat_waves.saturating_add(1);
    if wi<work.len(){parents=parents.saturating_add((work[wi].end-cursor).min(u64::from(batch)));}
    for peer in 0..p.world as usize{for shard in 0..p.shards as usize{
     let index=peer*p.shards as usize+shard;
     let sent=lc[index];let recv=if peer==self.rank as usize{lc[index]}else{rc[index]};
     sent_counts[index]=sent_counts[index].saturating_add(u64::from(sent));
     received_counts[index]=received_counts[index].saturating_add(u64::from(recv));
     peak_send=peak_send.max(sent);peak_receive=peak_receive.max(recv);
    }}
   }
   if p.world>1{
    for peer in 0..p.world as usize{if peer==self.rank as usize{continue}for shard in 0..p.shards as usize{
     let actual=u64::from(lc[peer*p.shards as usize+shard]);
     if if self.exact_exchange{actual>0}else{used>0}{self.remote_payload_transfers=self.remote_payload_transfers.saturating_add(1);}
     self.remote_records_sent=self.remote_records_sent.saturating_add(actual);
     if !self.exact_exchange{
      self.remote_padded_records=self.remote_padded_records.saturating_add(u64::from(used).saturating_sub(actual));
     }
    }}
   }
   record!(3);let active=(0..p.shards as usize).filter(|&shard|
    (0..p.world as usize).any(|src|if src==self.rank as usize{lc[src*p.shards as usize+shard]!=0}else{rc[src*p.shards as usize+shard]!=0})
   ).collect::<Vec<_>>();
   if let Some(offsets)=&self.batch_offsets{
    let stride=p.world as usize+1;
    for &shard in &active{
     let begin=shard*stride;self.batch_offsets_host[begin]=0;
     for src in 0..p.world as usize{
      let n=if src==self.rank as usize{lc[src*p.shards as usize+shard]}else{rc[src*p.shards as usize+shard]};
      if n>used{return Err("SEGMENTED_OFFSET_COUNT_BOUND".into())}
      self.batch_offsets_host[begin+src+1]=self.batch_offsets_host[begin+src].checked_add(n).ok_or("SEGMENTED_OFFSET_SIZE")?;
     }
    }
    crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(offsets.ptr,self.batch_offsets_host.as_ptr().cast(),offsets.bytes,1,st)})?;
   }
   if self.input_stats_active(){
    let mut valid=0u64;
    for &shard in &active{for src in 0..p.world as usize{
     let count=if src==self.rank as usize{lc[src*p.shards as usize+shard]}else{rc[src*p.shards as usize+shard]};
     if count>used{return Err("SEGMENTED_INPUT_STATS_COUNT_BOUND".into())}
     if count==0{
      if src==self.rank as usize{self.input_empty_self=self.input_empty_self.checked_add(1).ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;}
      else{self.input_empty_peer=self.input_empty_peer.checked_add(1).ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;}
     }
     valid=valid.checked_add(u64::from(count)).ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;
    }}
    self.input_valid=self.input_valid.checked_add(valid).ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;
   }
   if wi<work.len(){cursor=cursor.saturating_add(u64::from(batch)).min(work[wi].end);}
   rounds=rounds.checked_sub(1).ok_or("SEGMENTED_ROUND_COUNT_UNDERFLOW")?;
   while wi<work.len()&&cursor>=work[wi].end{wi+=1;if wi<work.len(){cursor=work[wi].begin;}}
   // Start next generation before waiting for this chunk's owner kernels.
   // A later resource/fatal vote discards the prefetched, unpublished slot.
   let probe_enabled=self.overlap_bounds.map_or(false,|(low,high)|{let target=self.depth.saturating_add(1);low<=target&&target<=high})&&rounds>0&&wi<work.len()&&!active.is_empty()&&self.overlap_samples<16;
   let mut probe=if probe_enabled{Some(OverlapProbe::new(self.base.device)?)}else{None};
   if let Some(q)=probe.as_ref(){q.start_route(self.route_stream.ptr)?;}
   if rounds>0{self.start_route(1-slot,&work,wi,cursor,batch)?;self.prefetched_chunks+=1;}
   if let Some(q)=probe.as_ref(){q.end_route(self.route_stream.ptr)?;}
   pending_owner=self.accept_parallel(slot,&active,future,used,probe.as_mut())?;
   if let Some(q)=probe.as_ref(){q.report(self.rank,self.depth)?;self.overlap_samples+=1;}
   record!(4);
   slot=1-slot;
  }
   layer_had_work
   };
   let ready = self.prepare_layer_commit(future, &before, &boundary_stop, next_stop)?;
   let local = ready.local;
   let global = ready.global;
   record!(6);
  if tracing{eprintln!("SEGMENT_TIMING rank={} depth={} us={phases:?}",self.rank,self.depth);}
  let sync_after_layer = self.global_sync.get();
  let layer_sync = sync_after_layer.since(self.last_layer_sync);
  self.last_layer_sync = sync_after_layer;
  if let Some((candidate_start,bytes_start,metadata_start,sends_start))=transport_start{
   let (candidate_bytes,data_bytes,metadata_bytes,sends)=self.transport_totals();
   eprintln!("SEGMENT_LAYER_STATS {}",serde_json::json!({
    "rank":self.rank,"depth":self.depth.saturating_add(1),"world":p.world,"shards":p.shards,
    "source_parents":parents,"generated_edges":parents.saturating_mul(u64::from(self.base.generators)),
    "source_frontier":self.counts[self.rank as usize],"accepted":local,"global_accepted":global,
    "waves":stat_waves,"source_retries":self.source_retries-layer_retries_start,
    "send_by_peer_shard":sent_counts,"receive_by_peer_shard":received_counts,
    "peak_send_bucket":peak_send,"peak_receive_bucket":peak_receive,"global_peak_queue":layer_peak_queue,
    "queue_capacity":p.queue_capacity,"effective_batch":batch,"admitted_batch":self.admitted_batch,
    "spill_records_including_replays":self.profile_spill-spill_start,
    "owner_passes":self.profile_owner_passes-passes_start,"live_rows_by_bank":self.bank_totals,
    "storage_allocated_bytes":self.storage_budget-self.remaining_bytes,
    "hash_segment_reuses":self.hash_segment_reuses,
    "storage_peak_requested_bytes":self.storage_peak_requested,
    "bitmap_bytes":self.bitmap.as_ref().map_or(0,|b|b.bits.bytes),
    "reserve_bytes":self.reserve_bytes,
    "remote_candidate_payload_bytes":candidate_bytes-candidate_start,
    "remote_submitted_nccl_bytes":data_bytes-bytes_start,
    "remote_submitted_nccl_metadata_bytes":metadata_bytes-metadata_start,
    "remote_nccl_send_calls":sends-sends_start,
    "global_sync":layer_sync
   }));
  }
  if global==0{
   // This detached extent holds no live states and the exhausted search
   // will never grow another shard. Leave only the committed output banks.
   self.release_spare_hash_segment();
   return Ok(DistributedAdvance::Complete)
  }
  if self.batch_recovery{
   let before=self.queue_batch_ceiling;
   let after=self.batch_recovery_policy.observe(before,self.admitted_batch,layer_peak_queue,p.queue_capacity,
    layer_had_work&&batch==before,self.source_retries!=layer_retries_start);
   if after>before{
    self.queue_batch_ceiling=after;
    self.batch_recoveries=self.batch_recoveries.saturating_add(1);
    if self.bounds_trace{eprintln!("SEGMENT_BATCH_RECOVER rank={} depth={} from={} to={}",self.rank,self.depth,before,after);}
   }
  }
  self.publish_layer(ready)
  })();
  if matches!(result,Ok(DistributedAdvance::Layer{..})){
   self.previous=work;
  }else{
   self.current=work;
  }
  result
 }
 fn snapshot_previous_before_reuse(&mut self)->Result<()> {
  if self.previous_snapshot.is_none() || self.previous_global>=1000 {self.previous_snapshot_valid=false;return Ok(())}
  let snapshot=self.previous_snapshot.as_ref().unwrap();let st=self.base.stream.ptr;
  let width=self.plan.elements as usize;let bytes=self.plan.state_bytes as usize;let mut offset=0usize;
  if self.pack4.is_some(){
   for r in &self.previous {
    let seg=&self.segments[r.shard][r.segment];let n=usize::try_from(r.end-r.begin).map_err(|_|"BITMAP_SNAPSHOT_SIZE")?;
    if offset.checked_add(n).map_or(true,|x|x>1000){return Err("BITMAP_SNAPSHOT_OVERFLOW".into())}
    let row=u64::from(r.bank).checked_mul(seg.desc.capacity).and_then(|v|v.checked_add(r.begin)).ok_or("BITMAP_SNAPSHOT_SIZE")?;
    let src_off=usize::try_from(row.checked_mul(8).ok_or("BITMAP_SNAPSHOT_SIZE")?).map_err(|_|"BITMAP_SNAPSHOT_SIZE")?;
    let src=unsafe{seg.desc.states.cast::<u8>().add(src_off)};
    let dst=unsafe{snapshot.ptr.cast::<u8>().add(offset*8)};
    crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(dst.cast(),src.cast(),n*8,3,st)})?;
    offset+=n;
   }
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
   self.previous_snapshot_rows=offset;self.previous_snapshot_valid=true;return Ok(())
  }
  for r in &self.previous {
   let seg=&self.segments[r.shard][r.segment];let n=usize::try_from(r.end-r.begin).map_err(|_|"BITMAP_SNAPSHOT_SIZE")?;
   if offset.checked_add(n).map_or(true,|x|x>1000){return Err("BITMAP_SNAPSHOT_OVERFLOW".into())}
   let stride=seg.desc.capacity as usize*seg.desc.banks as usize;
   let start=r.bank as usize*seg.desc.capacity as usize+r.begin as usize;
   for e in 0..width {
    let src=unsafe{seg.desc.states.cast::<u8>().add((e*stride+start)*bytes)};
    let dst=unsafe{snapshot.ptr.cast::<u8>().add((e*1000+offset)*bytes)};
    crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(dst.cast(),src.cast(),n*bytes,3,st)})?;
   }
   offset+=n;
  }
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
  self.previous_snapshot_rows=offset;self.previous_snapshot_valid=true;Ok(())
 }
 fn sample_previous_snapshot(&self)->Result<Vec<Vec<i64>>> {
  let snap=self.previous_snapshot.as_ref().ok_or("BITMAP_SNAPSHOT_MISSING")?;
  let n=self.previous_snapshot_rows;let width=self.plan.elements as usize;let bytes=self.plan.state_bytes as usize;
  if let Some(packed)=self.pack4 {
   let mut words=vec![0u64;n];
   crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(words.as_mut_ptr().cast(),snap.ptr,n*8,2)})?;
   return words.into_iter().map(|v|decode_pack4(packed.dictionary,v)).collect()
  }
  let mut rows=vec![vec![0i64;width];n];
  for e in 0..width {
   let from=unsafe{snap.ptr.cast::<u8>().add(e*1000*bytes)};
   if bytes==1 {let mut v=vec![0u8;n];crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(v.as_mut_ptr().cast(),from.cast(),n,2)})?;for i in 0..n{rows[i][e]=i64::from(v[i]);}}
   else {let mut v=vec![0i64;n];crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(v.as_mut_ptr().cast(),from.cast(),n*8,2)})?;for i in 0..n{rows[i][e]=v[i];}}
  }
  Ok(rows)
 }
 fn sample_ranges(&self,ranges:&[Range],limit:u32)->Result<Vec<Vec<i64>>>{
  let width=self.plan.elements as usize;let bytes=self.plan.state_bytes as usize;let mut result=Vec::new();
  if let Some(packed)=self.pack4 {
   for r in ranges{
    let seg=&self.segments[r.shard][r.segment];
    let available=usize::try_from(r.end-r.begin).map_err(|_|"BITMAP_PACK4_SAMPLE_RANGE")?;
    let n=available.min(limit as usize-result.len());if n==0{break}
    let row=u64::from(r.bank).checked_mul(seg.desc.capacity).and_then(|v|v.checked_add(r.begin)).ok_or("BITMAP_PACK4_SAMPLE_RANGE")?;
    let offset=usize::try_from(row.checked_mul(8).ok_or("BITMAP_PACK4_SAMPLE_RANGE")?).map_err(|_|"BITMAP_PACK4_SAMPLE_RANGE")?;
    let from=unsafe{seg.desc.states.cast::<u8>().add(offset)};
    let mut words=vec![0u64;n];crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(words.as_mut_ptr().cast(),from.cast(),n*8,2)})?;
    for word in words{result.push(decode_pack4(packed.dictionary,word)?)}
   }
   return Ok(result)
  }
  for r in ranges{let seg=&self.segments[r.shard][r.segment];let available=(r.end-r.begin) as usize;let n=available.min(limit as usize-result.len());if n==0{break}
   let stride=seg.desc.capacity as usize*seg.desc.banks as usize;let start=r.bank as usize*seg.desc.capacity as usize+r.begin as usize;
   let mut rows=vec![vec![0i64;width];n];for e in 0..width{let offset=(e*stride+start)*bytes;let from=unsafe{seg.desc.states.cast::<u8>().add(offset)};
    if bytes==1{let mut values=vec![0u8;n];crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),from.cast(),n,2)})?;for i in 0..n{rows[i][e]=i64::from(values[i]);}}
    else{let mut values=vec![0i64;n];crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),from.cast(),n*8,2)})?;for i in 0..n{rows[i][e]=values[i];}}
   }result.extend(rows);
  }Ok(result)
 }
 pub fn sample_global(&self,limit:u32)->Result<Vec<Vec<i64>>>{let preceding=self.counts[..self.rank as usize].iter().sum::<u64>();let quota=u64::from(limit).saturating_sub(preceding).min(self.counts[self.rank as usize]) as u32;self.sample_ranges(&self.current,quota)}
 pub fn previous_small_sample(&self,previous:u64)->Result<Vec<Vec<i64>>>{if previous>=1000{Ok(vec![])}else if self.previous_snapshot_valid{self.sample_previous_snapshot()}else{self.sample_ranges(&self.previous,previous as u32)}}

 // Call only at a committed boundary, before advance can reuse a rolling bank.
 // The caller has already bounded the global serialized size of both layers.
 pub(crate) fn terminal_layer_samples(&self) -> Result<(Vec<Vec<i64>>, Vec<Vec<i64>>)> {
  let count_error = || "PROFILE_TERMINAL_EVIDENCE_COUNT".to_owned();
  let sample = |ranges: &[Range], committed_count: Option<u64>| -> Result<_> {
   let count = ranges.iter().try_fold(0u64, |rows, range| {
    rows.checked_add(range.end.checked_sub(range.begin)?)
   }).ok_or_else(count_error)?;
   if committed_count.is_some_and(|expected| count != expected) { return Err(count_error()); }
   let states = self.sample_ranges(ranges, u32::try_from(count).map_err(|_| count_error())?)?;
   if states.len() as u64 != count { return Err(count_error()); }
   Ok(states)
  };
  Ok((sample(&self.current, Some(self.counts[self.rank as usize]))?, sample(&self.previous, None)?))
 }
}

#[cfg(test)]
mod segment_sizing_cache_gpu_test {
 use super::*;
 #[test]
 #[ignore = "requires a CUDA GPU with at least 512 MiB free"]
 fn sizing_reclaims_idle_session_gpu_storage() {
  crate::generic_pool::activate(0).expect("activate CUDA pool");
  crate::session_cache::enable();
  let bytes=256usize<<20;
  let ptr=crate::generic_pool::allocate(bytes).expect("allocate CUDA pool storage").expect("async pool unavailable");
  assert!(crate::session_cache::buffer_put(ptr,bytes));
  assert!(crate::session_cache::reusable_bytes()>=bytes as u64);
  let(mut before,mut total)=(0usize,0usize);
  crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut before,&mut total)}).unwrap();
  let(after,_)=segment_sizing_memory_info().expect("size after reclaim");
  assert_eq!(crate::session_cache::reusable_bytes(),0,"idle session buffers not released before sizing");
  assert!(after>=before.saturating_add(bytes*3/4),"free VRAM did not recover: before={before} after={after} allocated={bytes}");
 }
}
