//! General exact distributed engine: preallocated queues and independent owner streams.
//! Fixed-capacity transport is a correctness fallback, not an optimal bandwidth claim.
use std::{ffi::c_void,ptr};
use mgbfs_core::{graph_definition::GraphDefinitionV2,Result};
use mgbfs_cuda::{ffi::*,generic_graph::*};
use crate::{generic_native::{GenericNativeBfs,Buffer,Stream,check},generic_memory::GenericMemoryPlan,generic_distributed_memory::GenericDistributedMemoryPlan};
extern "C" {
 fn mgbfs_nccl_all_gather_bytes(comm:*mut c_void,send:*const c_void,recv:*mut c_void,bytes:u64,stream:*mut c_void)->i32;
 fn mgbfs_generic_pack_parent_chunk(bytes:u32,elements:u32,source:*const c_void,source_stride:u32,count:u32,out:*mut c_void,stride:u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_advance_parent_cursors(cursors:*mut u64,frontiers:*mut u32,next:*const u32,world:u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_accept_parent_origin(bytes:u32,kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,
 parents:*const c_void,parent_stride:u32,chunk:u32,begin:u64,cursors:*const u64,frontiers:*const u32,
 local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,
 slots:*mut u64,slot_capacity:u32,arena:*mut c_void,arena_stride:u32,base:u32,capacity:u32,visited_count:*mut u32,future:*mut u32,accepted:*mut u32,positions:*mut u32,error:*mut u32,rolling:u32,ordered:*const u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_sort_origins(lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,keys:*mut u64,sorted_keys:*mut u64,origins:*mut u32,sorted_origins:*mut u32,scratch:*mut c_void,scratch_bytes:u64,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_accept_sorted(bytes:u32,elements:u32,local:*const c_void,remote:*const c_void,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,slots:*mut u64,sc:u32,arena:*mut c_void,stride:u32,base:u32,capacity:u32,vc:*mut u32,future:*mut u32,accepted:*mut u32,positions:*mut u32,error:*mut u32,rolling:u32,ordered:*const u32,stream:*mut c_void)->i32;
}
struct SortCache{keys:Buffer,sorted_keys:Buffer,origins:Buffer,sorted_origins:Buffer,scratch:Buffer,scratch_stride:usize}
impl SortCache{fn new(p:&GenericDistributedMemoryPlan,device:i32)->Result<Self>{let n=p.world as usize*p.queue_capacity as usize;let size=n*p.shards as usize;let scratch_stride=n*64+65536;Ok(Self{keys:Buffer::new(size*8,device)?,sorted_keys:Buffer::new(size*8,device)?,origins:Buffer::new(size*4,device)?,sorted_origins:Buffer::new(size*4,device)?,scratch:Buffer::new(scratch_stride*p.shards as usize,device)?,scratch_stride})}}
struct ParentCache{sources:[Buffer;2],received:Buffer,cursors:Buffer,frontiers:Buffer}
impl ParentCache{fn new(p:&GenericDistributedMemoryPlan,device:i32,counts:&[u32])->Result<Self>{
 let bytes=p.batch as usize*p.elements as usize*p.state_bytes as usize;
 let sources=[Buffer::new(bytes,device)?,Buffer::new(bytes,device)?];for v in &sources{check(unsafe{cudaMemsetAsync(v.ptr,0,v.bytes,ptr::null_mut())})?;}
 check(unsafe{cudaStreamSynchronize(ptr::null_mut())})?;let received=Buffer::new(bytes*p.world as usize,device)?;let cursors=Buffer::new(p.world as usize*8,device)?;cursors.upload(&vec![0u64;p.world as usize])?;
 let frontiers=Buffer::new(p.world as usize*4,device)?;frontiers.upload(counts)?;Ok(Self{sources,received,cursors,frontiers})
}}
struct Event{ptr:*mut c_void,device:i32}
impl Event{fn new(device:i32)->Result<Self>{let mut p=ptr::null_mut();check(unsafe{cudaEventCreateWithFlags(&mut p,2)})?;Ok(Self{ptr:p,device})}}
impl Drop for Event{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.device);cudaEventDestroy(self.ptr);}}}
struct QueueBank{records:Buffer,counts:Buffer,states:Buffer}
impl QueueBank{fn new(p:&GenericDistributedMemoryPlan,device:i32)->Result<Self>{let boxes=p.world as usize*p.shards as usize;let q=p.queue_capacity as usize;Ok(Self{records:Buffer::new(boxes*q*32,device)?,counts:Buffer::new(boxes*4,device)?,states:Buffer::new(boxes*q*p.queue_payload_bytes(),device)?})}}
#[derive(Debug,PartialEq,Eq)]
pub enum DistributedAdvance{Layer{local:u32,global:u64},Complete,Resource{fatal:u32}}
pub struct GenericDistributedBfs{
 bfs:GenericNativeBfs,plan:GenericDistributedMemoryPlan,rank:u32,comm:*mut c_void,
 sort_cache:Option<SortCache>,parent_cache:Option<ParentCache>,banks:[QueueBank;2],inbox:QueueBank,control:Buffer,owner_map:Buffer,owner_cuts:Option<Buffer>,
 streams:Vec<Stream>,ready:Event,done:Vec<Event>,depth:u64,parent_cursor:u64,max_frontier:u32,counts:Vec<u32>,source_retries:u64,positions:Option<Buffer>,rolling_counts:[u32;3],retired_rows:u64,
}
fn mix(mut x:u64)->u64{x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
impl GenericDistributedBfs{
 /// All ranks must agree on graph, seed, hash bits, batch/queue geometry and
 /// rank map before entering this constructor. The launcher owns that gate.
 pub fn new(graph:&GraphDefinitionV2,device:u32,rank:u32,plan:GenericDistributedMemoryPlan,id:&[u8;128],seed:u64,hash_bits:u32)->Result<Self>{
  Self::new_with_cuts(graph,device,rank,plan,id,seed,hash_bits,None)
 }
 pub fn new_with_cuts(graph:&GraphDefinitionV2,device:u32,rank:u32,plan:GenericDistributedMemoryPlan,id:&[u8;128],seed:u64,hash_bits:u32,cuts:Option<&[u64]>)->Result<Self>{
  if let Some(c)=cuts{if c.len()!=plan.world as usize+1||c[0]!=0||c[c.len()-1]!=(1u64<<32)||c.windows(2).any(|v|v[0]>=v[1]){return Err("GENERIC_OWNER_CUTS".into());}}
  plan.validate(graph.generator_count() as u32)?;if rank>=plan.world||hash_bits>64{return Err("GENERIC_DISTRIBUTED_RANK".into());}
  let local=GenericMemoryPlan::with_storage(plan.elements,plan.capacity,plan.state_bytes)?;
  let rolling=plan.history_layers==3;if rolling&&!graph.inverse_closed()?{return Err("GENERIC_ROLLING_REQUIRES_INVERSE_CLOSED".into());}
  let mut bfs=if rolling{GenericNativeBfs::new_unseeded_rolling(graph,device,local,seed,hash_bits)?}else{GenericNativeBfs::new_unseeded(graph,device,local,seed,hash_bits)?};let d=bfs.device;
  if bfs.slots.bytes!=plan.table_slots as usize*8{return Err("GENERIC_SHARED_TABLE_ALLOCATION_SHAPE".into());}
  check(unsafe{cudaMemsetAsync(bfs.slots.ptr,255,bfs.slots.bytes,bfs.stream.ptr)})?;
  let mut hash=seed;for(e,v)in graph.start.iter().enumerate(){hash=mix(hash^(*v as u64)^(e as u64));}hash=mix(hash);if hash_bits<64{hash&=if hash_bits==0{0}else{(1u64<<hash_bits)-1};}
  let owner=if let Some(c)=cuts{c.windows(2).position(|v|v[0]<=hash>>32&&hash>>32<v[1]).ok_or("GENERIC_ROOT_OWNER")? as u32}else{(((hash>>32)*u64::from(plan.world))>>32) as u32};
  // Swap root logical owner with rank zero, preserving a bijective rank map.
  let mut map=(0..plan.world).collect::<Vec<_>>();if cuts.is_none(){map.swap(0,owner as usize);}
  let root_rank=map[owner as usize];
  let owner_cuts=if let Some(c)=cuts{let v=Buffer::new(c.len()*8,d)?;v.upload(c)?;Some(v)}else{None};
  let owner_map=Buffer::new(map.len()*4,d)?;owner_map.upload(&map)?;
  bfs.count=if rank==root_rank{1}else{0};bfs.visited_used=bfs.count;bfs.control.upload(&[bfs.count,0u32,0,0,0,0])?;
  let positions=if rolling{Some(Buffer::new(bfs.arena_stride as usize*4,d)?)}else{None};
  if rank==root_rank&&rolling{check(unsafe{mgbfs_generic_reseed_rows_storage(plan.state_bytes,plan.elements,bfs.visited.ptr.cast(),bfs.arena_stride,0,1,bfs.slots.ptr.cast(),plan.table_slots,positions.as_ref().unwrap().ptr.cast(),seed,hash_bits,bfs.control.at(8),bfs.stream.ptr)})?;}
  if rank==root_rank&&!rolling{check(unsafe{mgbfs_generic_seed_shared_storage(plan.state_bytes,plan.elements,bfs.visited.ptr.cast(),plan.capacity,1,bfs.slots.ptr.cast(),plan.table_slots,seed,hash_bits,bfs.control.at(8),bfs.stream.ptr)})?;}
  check(unsafe{cudaStreamSynchronize(bfs.stream.ptr)})?;
  let banks=[QueueBank::new(&plan,d)?,QueueBank::new(&plan,d)?];let inbox=QueueBank::new(&plan,d)?;
  let control=Buffer::new(64+plan.world as usize*8,d)?;
  let mut streams=vec![];let mut done=vec![];for _ in 0..plan.shards{streams.push(Stream::new(d)?);done.push(Event::new(d)?);}
  let ready=Event::new(d)?;let mut comm=ptr::null_mut();let mut error=[0i8;512];
  check(unsafe{mgbfs_nccl_create(rank,plan.world,device,id.as_ptr().cast(),&mut comm,error.as_mut_ptr(),error.len())})?;
  let counts=(0..plan.world).map(|r|if r==root_rank{1u32}else{0}).collect::<Vec<_>>();
  let sort_cache=if plan.sort_candidates{Some(SortCache::new(&plan,d)?)}else{None};
  let parent_cache=if plan.parent_transport{Some(ParentCache::new(&plan,d,&counts)?)}else{None};
  let rolling_counts=[bfs.count,0,0];
  Ok(Self{sort_cache,parent_cache,positions,rolling_counts,retired_rows:0,bfs,plan,rank,comm,banks,inbox,control,owner_map,owner_cuts,streams,ready,done,depth:0,parent_cursor:0,max_frontier:1,counts,source_retries:0})
 }
 pub fn source_retries(&self)->u64{self.source_retries}
 pub fn frontier_len(&self)->u32{self.bfs.count}
 pub fn sample_global(&mut self,limit:u32)->Result<Vec<Vec<i64>>>{let preceding=self.counts[..self.rank as usize].iter().map(|&v|u64::from(v)).sum::<u64>();let quota=u64::from(limit).saturating_sub(preceding).min(u64::from(self.bfs.count)) as u32;self.bfs.sample(quota)}
 /// Layer-boundary cancellation vote: every rank must call in the same order.
 pub fn collective_stop(&mut self,reason:u32)->Result<u32>{if self.bfs.terminal{return Err("GENERIC_DISTRIBUTED_TERMINAL".into());}
  let stream=self.bfs.stream.ptr;check(unsafe{cudaMemcpyAsync(self.control.at::<c_void>(16),(&reason as *const u32).cast(),4,1,stream)})?;
  check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm,self.control.at(16),self.control.at(20),stream)})?;
  let mut global=0u32;check(unsafe{cudaMemcpyAsync((&mut global as *mut u32).cast(),self.control.at::<c_void>(20),4,2,stream)})?;check(unsafe{cudaStreamSynchronize(stream)})?;if global!=0{self.bfs.stop();}Ok(global)
 }

 pub fn stop(&mut self){self.bfs.stop();}
 pub fn sample(&mut self,limit:u32)->Result<Vec<Vec<i64>>>{self.bfs.sample(limit)}
 pub fn previous_small_sample(&mut self,global_previous:u64)->Result<Vec<Vec<i64>>>{if global_previous>=1000{return Ok(vec![]);}self.bfs.previous_small_sample()}
 pub fn advance(&mut self)->Result<DistributedAdvance>{
  let b=&mut self.bfs;if b.terminal{return Err("GENERIC_DISTRIBUTED_TERMINAL".into());}b.terminal=true;
  check(unsafe{mgbfs_cuda::native_owner::cudaSetDevice(b.device)})?;let stream=b.stream.ptr;let p=&self.plan;
  let rolling=p.history_layers==3;let future_bank=((self.depth+1)%3) as usize;let future_base=future_bank as u32*p.capacity;
  if rolling{
   let positions=self.positions.as_ref().ok_or("GENERIC_ROLLING_POSITIONS")?;let expired=self.rolling_counts[future_bank];
   if expired!=0{check(unsafe{mgbfs_generic_retire_rows(b.slots.ptr.cast(),p.table_slots,positions.ptr.cast(),b.arena_stride,future_base,expired,b.control.at(8),stream)})?;self.retired_rows+=u64::from(expired);self.rolling_counts[future_bank]=0;}
   // Tombstones preserve probe chains. Maintenance is amortized over at
   // least table_slots/4 retirements, rather than rebuilding each layer.
   if self.retired_rows>=u64::from(p.table_slots)/4{
    check(unsafe{cudaMemsetAsync(b.slots.ptr,255,b.slots.bytes,stream)})?;
    for bank in 0..3{let count=self.rolling_counts[bank];if count!=0{check(unsafe{mgbfs_generic_reseed_rows_storage(p.state_bytes,p.elements,b.visited.ptr.cast(),b.arena_stride,bank as u32*p.capacity,count,b.slots.ptr.cast(),p.table_slots,positions.ptr.cast(),b.seed,b.hash_bits,b.control.at(8),stream)})?;}}
    self.retired_rows=0;
   }
  }else{check(unsafe{mgbfs_generic_gather_storage(p.state_bytes,p.elements,b.visited.ptr.cast(),b.arena_stride,b.front.ptr.cast(),b.count,b.parents.ptr.cast(),p.capacity,stream)})?;}
  // Current remains immutable; generation reads its bank directly.
  let parent_ptr=if rolling{b.visited.at::<i64>(b.current_start as usize*p.state_bytes as usize)}else{b.parents.ptr.cast()};let parent_stride=if rolling{b.arena_stride}else{p.capacity};
  check(unsafe{cudaMemsetAsync(b.control.at::<c_void>(4),0,4,stream)})?;
  let mut global_begin=0u64;let mut batch=p.batch;let mut round=0u64;
  let queues=p.world as usize*p.shards as usize;let q=p.queue_capacity as usize;let width=p.elements as usize;let shards=p.shards as usize;let state_bytes=p.state_bytes as usize;
  while global_begin<u64::from(self.max_frontier){
   let bank=&self.banks[((self.depth+round)%2) as usize];let begin=global_begin.min(u64::from(b.count)) as u32;let count=batch.min(b.count-begin);
   check(unsafe{cudaMemsetAsync(self.control.at::<c_void>(8),0,4,stream)})?;
   check(unsafe{cudaMemsetAsync(bank.counts.ptr,0,bank.counts.bytes,stream)})?;
   check(unsafe{mgbfs_generic_route_storage(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),b.seed,b.hash_bits,p.world,self.rank,p.shards,p.queue_capacity,self.parent_cursor+u64::from(begin),self.owner_map.ptr.cast(),self.owner_cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),bank.records.ptr.cast(),bank.counts.ptr.cast(),self.control.at(8),stream)})?;
   if !p.packed_candidates()&&!p.parent_transport{for queue in 0..queues{check(unsafe{mgbfs_generic_regenerate_routes_count_storage(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),self.rank,self.parent_cursor+u64::from(begin),bank.records.at(queue*q*32),p.queue_capacity,bank.counts.at(queue*4),bank.states.at(queue*q*width*state_bytes),p.queue_capacity,self.control.at(8),stream)})?;}}
   if let Some(cache)=&self.parent_cache{
    check(unsafe{mgbfs_generic_pack_parent_chunk(p.state_bytes,p.elements,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),parent_stride,count,cache.sources[((self.depth+round)%2) as usize].ptr,p.batch,stream)})?;
   }
   // Invalid source counts must not be presented as valid owner inboxes.
   // Generate the next immutable source bank while preceding owners work.
   // Retire their inbox leases only before reading owner errors/exchanging.
   if round>0{for event in &self.done{check(unsafe{cudaStreamWaitEvent(stream,event.ptr,0)})?;}}
   check(unsafe{mgbfs_generic_route_retry_vote(self.control.at(8),b.control.at(8),self.control.at(12),stream)})?;
   check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm,self.control.at(12),self.control.at(0),stream)})?;
   let mut fatal=0u32;check(unsafe{cudaMemcpyAsync((&mut fatal as *mut u32).cast(),self.control.ptr,4,2,stream)})?;check(unsafe{cudaStreamSynchronize(stream)})?;
   if fatal==1&&batch>1{batch=(batch/2).max(1);self.source_retries+=1;continue;}
   if fatal!=0{return Ok(DistributedAdvance::Resource{fatal});}
   // A single matched NCCL group submits all peer lanes without host count reads.
   check(unsafe{mgbfs_nccl_exchange_triplets(self.comm,self.rank,p.world,bank.counts.ptr,(shards*4) as u64,bank.records.ptr,(shards*q*32) as u64,bank.states.ptr,(shards*q*p.queue_payload_bytes()) as u64,self.inbox.counts.ptr,self.inbox.records.ptr,self.inbox.states.ptr,stream)})?;
   if let Some(cache)=&self.parent_cache{
    check(unsafe{mgbfs_nccl_all_gather_bytes(self.comm,cache.sources[((self.depth+round)%2) as usize].ptr,cache.received.ptr,u64::from(p.batch)*u64::from(p.elements)*u64::from(p.state_bytes),stream)})?;
   }
   check(unsafe{cudaEventRecord(self.ready.ptr,stream)})?;
   for shard in 0..shards{let owner_stream=self.streams[shard].ptr;check(unsafe{cudaStreamWaitEvent(owner_stream,self.ready.ptr,0)})?;
    let ordered=if let Some(cache)=&self.sort_cache{
     let offset=shard*p.world as usize*q;
     check(unsafe{mgbfs_generic_sort_origins(bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,cache.keys.at(offset*8),cache.sorted_keys.at(offset*8),cache.origins.at(offset*4),cache.sorted_origins.at(offset*4),cache.scratch.at::<c_void>(shard*cache.scratch_stride),cache.scratch_stride as u64,b.control.at(8),owner_stream)})?;cache.sorted_origins.at::<u32>(offset*4).cast_const()
    }else{ptr::null()};
    if let Some(cache)=&self.parent_cache{
     check(unsafe{mgbfs_generic_accept_parent_origin(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),cache.received.ptr,p.batch,batch,global_begin,cache.cursors.ptr.cast(),cache.frontiers.ptr.cast(),bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr,b.arena_stride,if rolling{future_base}else{0},p.capacity,b.control.at(0),b.future.ptr.cast(),b.control.at(4),self.positions.as_ref().map_or(ptr::null_mut(),|v|v.ptr.cast()),b.control.at(8),u32::from(rolling),ordered,owner_stream)})?;
    }else if self.sort_cache.is_some(){
     check(unsafe{mgbfs_generic_accept_sorted(p.state_bytes,p.elements,bank.states.ptr, self.inbox.states.ptr,bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr,b.arena_stride,if rolling{future_base}else{0},p.capacity,b.control.at(0),b.future.ptr.cast(),b.control.at(4),self.positions.as_ref().map_or(ptr::null_mut(),|v|v.ptr.cast()),b.control.at(8),u32::from(rolling),ordered,owner_stream)})?;
    }else if rolling{check(unsafe{mgbfs_generic_accept_rolling_storage(p.state_bytes,p.elements,bank.states.ptr.cast(),self.inbox.states.ptr.cast(),bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr.cast(),b.arena_stride,future_base,p.capacity,b.control.at(4),b.future.ptr.cast(),self.positions.as_ref().unwrap().ptr.cast(),b.control.at(8),owner_stream)})?;}
    else{    check(unsafe{mgbfs_generic_accept_all_storage(p.state_bytes,p.elements,bank.states.ptr.cast(),self.inbox.states.ptr.cast(),bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr.cast(),p.capacity,b.control.at(0),b.future.ptr.cast(),p.capacity,b.control.at(4),b.control.at(8),owner_stream)})?;}
    check(unsafe{cudaEventRecord(self.done[shard].ptr,owner_stream)})?;
   }
   // Next round may generate into the other source bank; inbox retirement
   // waits occur after that generation and before the next exchange.
   global_begin+=u64::from(batch);round+=1;
  }
  if round>0{for event in &self.done{check(unsafe{cudaStreamWaitEvent(stream,event.ptr,0)})?;}}
  check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm,b.control.at(8),self.control.at(0),stream)})?;
  check(unsafe{mgbfs_nccl_all_gather_u32(self.comm,b.control.at(4),self.control.at(32),stream)})?;
  let mut fatal=0u32;let mut counts=vec![0u32;p.world as usize];let mut visited=0u32;
  check(unsafe{cudaMemcpyAsync((&mut fatal as *mut u32).cast(),self.control.ptr,4,2,stream)})?;
  check(unsafe{cudaMemcpyAsync(counts.as_mut_ptr().cast(),self.control.at::<c_void>(32),counts.len()*4,2,stream)})?;
  check(unsafe{cudaMemcpyAsync((&mut visited as *mut u32).cast(),b.control.ptr,4,2,stream)})?;check(unsafe{cudaStreamSynchronize(stream)})?;
  if fatal!=0{return Ok(DistributedAdvance::Resource{fatal});}
  let global=counts.iter().map(|&v|u64::from(v)).sum::<u64>();if global==0{return Ok(DistributedAdvance::Complete);}
  let local=counts[self.rank as usize];if local>p.capacity{return Err("GENERIC_DISTRIBUTED_COUNT_BOUNDS".into());}
  b.previous=Some((b.current_start,b.count));if rolling{b.current_start=future_base;self.rolling_counts[future_bank]=local;}else{b.current_start=b.visited_used;b.visited_used=visited;}
  self.parent_cursor=self.parent_cursor.checked_add(u64::from(b.count)).ok_or("GENERIC_PARENT_CURSOR_OVERFLOW")?;
  if let Some(cache)=&self.parent_cache{check(unsafe{mgbfs_generic_advance_parent_cursors(cache.cursors.ptr.cast(),cache.frontiers.ptr.cast(),self.control.at(32),p.world,stream)})?;}
  self.depth+=1;self.max_frontier=*counts.iter().max().unwrap();self.counts=counts;std::mem::swap(&mut b.front,&mut b.future);b.count=local;b.terminal=false;
  Ok(DistributedAdvance::Layer{local,global})
 }
}
impl Drop for GenericDistributedBfs{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.bfs.device);cudaStreamSynchronize(self.bfs.stream.ptr);for stream in &self.streams{cudaStreamSynchronize(stream.ptr);}mgbfs_nccl_destroy(self.comm);}}}
