//! General exact distributed engine: preallocated queues and independent owner streams.
//! Fixed-capacity transport is a correctness fallback, not an optimal bandwidth claim.
use std::{ffi::c_void,ptr};
use mgbfs_core::{graph_definition::GraphDefinitionV2,Result};
use mgbfs_cuda::{ffi::*,generic_graph::*};
use crate::{generic_native::{GenericNativeBfs,Buffer,Stream,check},generic_memory::GenericMemoryPlan,generic_distributed_memory::GenericDistributedMemoryPlan};
struct Event{ptr:*mut c_void,device:i32}
impl Event{fn new(device:i32)->Result<Self>{let mut p=ptr::null_mut();check(unsafe{cudaEventCreateWithFlags(&mut p,2)})?;Ok(Self{ptr:p,device})}}
impl Drop for Event{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.device);cudaEventDestroy(self.ptr);}}}
struct QueueBank{records:Buffer,counts:Buffer,states:Buffer}
impl QueueBank{fn new(p:&GenericDistributedMemoryPlan,device:i32)->Result<Self>{let boxes=p.world as usize*p.shards as usize;let q=p.queue_capacity as usize;Ok(Self{records:Buffer::new(boxes*q*32,device)?,counts:Buffer::new(boxes*4,device)?,states:Buffer::new(boxes*q*p.elements as usize*p.state_bytes as usize,device)?})}}
#[derive(Debug,PartialEq,Eq)]
pub enum DistributedAdvance{Layer{local:u32,global:u64},Complete,Resource{fatal:u32}}
pub struct GenericDistributedBfs{
 bfs:GenericNativeBfs,plan:GenericDistributedMemoryPlan,rank:u32,comm:*mut c_void,
 banks:[QueueBank;2],inbox:QueueBank,control:Buffer,owner_map:Buffer,
 streams:Vec<Stream>,ready:Event,done:Vec<Event>,depth:u64,parent_cursor:u64,max_frontier:u32,counts:Vec<u32>,
}
fn mix(mut x:u64)->u64{x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
impl GenericDistributedBfs{
 /// All ranks must agree on graph, seed, hash bits, batch/queue geometry and
 /// rank map before entering this constructor. The launcher owns that gate.
 pub fn new(graph:&GraphDefinitionV2,device:u32,rank:u32,plan:GenericDistributedMemoryPlan,id:&[u8;128],seed:u64,hash_bits:u32)->Result<Self>{
  plan.validate(graph.generator_count() as u32)?;if rank>=plan.world||hash_bits>64{return Err("GENERIC_DISTRIBUTED_RANK".into());}
  let local=GenericMemoryPlan::with_storage(plan.elements,plan.capacity,plan.state_bytes)?;
  let mut bfs=GenericNativeBfs::new(graph,device,local,seed,hash_bits)?;let d=bfs.device;
  let slots=Buffer::new(plan.slots_per_shard as usize*plan.shards as usize*8,d)?;
  check(unsafe{cudaMemsetAsync(slots.ptr,255,slots.bytes,bfs.stream.ptr)})?;
  let mut hash=seed;for(e,v)in graph.start.iter().enumerate(){hash=mix(hash^(*v as u64)^(e as u64));}hash=mix(hash);if hash_bits<64{hash&=if hash_bits==0{0}else{(1u64<<hash_bits)-1};}
  let owner=(((hash>>32)*u64::from(plan.world))>>32) as u32;
  let shard=(((hash&0xffffffff)*u64::from(plan.shards))>>32) as usize;
  // Swap root logical owner with rank zero, preserving a bijective rank map.
  let mut map=(0..plan.world).collect::<Vec<_>>();map.swap(0,owner as usize);
  let owner_map=Buffer::new(map.len()*4,d)?;owner_map.upload(&map)?;
  bfs.slots=slots;bfs.count=if rank==0{1}else{0};bfs.visited_used=bfs.count;bfs.control.upload(&[bfs.count,0u32,0,0,0,0])?;
  if rank==0{check(unsafe{mgbfs_generic_seed_storage(plan.state_bytes,plan.elements,bfs.visited.ptr.cast(),plan.capacity,1,bfs.slots.at(shard*plan.slots_per_shard as usize*8),plan.slots_per_shard,seed,hash_bits,bfs.control.at(8),bfs.stream.ptr)})?;}
  check(unsafe{cudaStreamSynchronize(bfs.stream.ptr)})?;
  let banks=[QueueBank::new(&plan,d)?,QueueBank::new(&plan,d)?];let inbox=QueueBank::new(&plan,d)?;
  let control=Buffer::new(64+plan.world as usize*8,d)?;
  let mut streams=vec![];let mut done=vec![];for _ in 0..plan.shards{streams.push(Stream::new(d)?);done.push(Event::new(d)?);}
  let ready=Event::new(d)?;let mut comm=ptr::null_mut();let mut error=[0i8;512];
  check(unsafe{mgbfs_nccl_create(rank,plan.world,device,id.as_ptr().cast(),&mut comm,error.as_mut_ptr(),error.len())})?;
  let counts=(0..plan.world).map(|r|if r==0{1}else{0}).collect();
  Ok(Self{bfs,plan,rank,comm,banks,inbox,control,owner_map,streams,ready,done,depth:0,parent_cursor:0,max_frontier:1,counts})
 }
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
  check(unsafe{mgbfs_generic_gather_storage(p.state_bytes,p.elements,b.visited.ptr.cast(),p.capacity,b.front.ptr.cast(),b.count,b.parents.ptr.cast(),p.capacity,stream)})?;
  check(unsafe{cudaMemsetAsync(b.control.at::<c_void>(4),0,4,stream)})?;
  let rounds=(u64::from(self.max_frontier)+u64::from(p.batch)-1)/u64::from(p.batch);
  let queues=p.world as usize*p.shards as usize;let q=p.queue_capacity as usize;let width=p.elements as usize;let shards=p.shards as usize;let state_bytes=p.state_bytes as usize;
  for round in 0..rounds{
   let bank=&self.banks[((self.depth+round)%2) as usize];let begin=(round*u64::from(p.batch)).min(u64::from(b.count)) as u32;let count=p.batch.min(b.count-begin);
   check(unsafe{cudaMemsetAsync(bank.counts.ptr,0,bank.counts.bytes,stream)})?;
   check(unsafe{mgbfs_generic_route_storage(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,b.parents.at(begin as usize*state_bytes),count,p.capacity,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),b.seed,b.hash_bits,p.world,self.rank,p.shards,p.queue_capacity,self.parent_cursor+u64::from(begin),self.owner_map.ptr.cast(),ptr::null(),bank.records.ptr.cast(),bank.counts.ptr.cast(),b.control.at(8),stream)})?;
   for queue in 0..queues{check(unsafe{mgbfs_generic_regenerate_routes_count_storage(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,b.parents.at(begin as usize*state_bytes),count,p.capacity,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),self.rank,self.parent_cursor+u64::from(begin),bank.records.at(queue*q*32),p.queue_capacity,bank.counts.at(queue*4),bank.states.at(queue*q*width*state_bytes),p.queue_capacity,b.control.at(8),stream)})?;}
   // Invalid source counts must not be presented as valid owner inboxes.
   check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm,b.control.at(8),self.control.at(0),stream)})?;
   let mut fatal=0u32;check(unsafe{cudaMemcpyAsync((&mut fatal as *mut u32).cast(),self.control.ptr,4,2,stream)})?;check(unsafe{cudaStreamSynchronize(stream)})?;
   if fatal!=0{return Ok(DistributedAdvance::Resource{fatal});}
   // A single matched NCCL group submits all peer lanes without host count reads.
   check(unsafe{mgbfs_nccl_exchange_triplets(self.comm,self.rank,p.world,bank.counts.ptr,(shards*4) as u64,bank.records.ptr,(shards*q*32) as u64,bank.states.ptr,(shards*q*width*state_bytes) as u64,self.inbox.counts.ptr,self.inbox.records.ptr,self.inbox.states.ptr,stream)})?;
   check(unsafe{cudaEventRecord(self.ready.ptr,stream)})?;
   for shard in 0..shards{let owner_stream=self.streams[shard].ptr;check(unsafe{cudaStreamWaitEvent(owner_stream,self.ready.ptr,0)})?;
    check(unsafe{mgbfs_generic_accept_all_storage(p.state_bytes,p.elements,bank.states.ptr.cast(),self.inbox.states.ptr.cast(),bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.at(shard*p.slots_per_shard as usize*8),p.slots_per_shard,b.visited.ptr.cast(),p.capacity,b.control.at(0),b.future.ptr.cast(),p.capacity,b.control.at(4),b.control.at(8),owner_stream)})?;
    check(unsafe{cudaEventRecord(self.done[shard].ptr,owner_stream)})?;check(unsafe{cudaStreamWaitEvent(stream,self.done[shard].ptr,0)})?;
   }
   // This dependency retires all inbox and immutable source leases before reuse.
  }
  check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm,b.control.at(8),self.control.at(0),stream)})?;
  check(unsafe{mgbfs_nccl_all_gather_u32(self.comm,b.control.at(4),self.control.at(32),stream)})?;
  let mut fatal=0u32;let mut counts=vec![0u32;p.world as usize];let mut visited=0u32;
  check(unsafe{cudaMemcpyAsync((&mut fatal as *mut u32).cast(),self.control.ptr,4,2,stream)})?;
  check(unsafe{cudaMemcpyAsync(counts.as_mut_ptr().cast(),self.control.at::<c_void>(32),counts.len()*4,2,stream)})?;
  check(unsafe{cudaMemcpyAsync((&mut visited as *mut u32).cast(),b.control.ptr,4,2,stream)})?;check(unsafe{cudaStreamSynchronize(stream)})?;
  if fatal!=0{return Ok(DistributedAdvance::Resource{fatal});}
  let global=counts.iter().map(|&v|u64::from(v)).sum::<u64>();if global==0{return Ok(DistributedAdvance::Complete);}
  let local=counts[self.rank as usize];if local>p.capacity{return Err("GENERIC_DISTRIBUTED_COUNT_BOUNDS".into());}
  b.previous=Some((b.current_start,b.count));b.current_start=b.visited_used;b.visited_used=visited;
  self.parent_cursor=self.parent_cursor.checked_add(u64::from(b.count)).ok_or("GENERIC_PARENT_CURSOR_OVERFLOW")?;
  self.depth+=1;self.max_frontier=*counts.iter().max().unwrap();self.counts=counts;std::mem::swap(&mut b.front,&mut b.future);b.count=local;b.terminal=false;
  Ok(DistributedAdvance::Layer{local,global})
 }
}
impl Drop for GenericDistributedBfs{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.bfs.device);cudaStreamSynchronize(self.bfs.stream.ptr);for stream in &self.streams{cudaStreamSynchronize(stream.ptr);}mgbfs_nccl_destroy(self.comm);}}}
