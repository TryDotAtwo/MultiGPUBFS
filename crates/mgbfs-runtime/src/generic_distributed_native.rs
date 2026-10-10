//! General exact distributed engine: preallocated queues and independent owner streams.
//! Fixed-capacity transport is a correctness fallback, not an optimal bandwidth claim.
use std::{ffi::c_void,ptr};
use crate::generic_sorted_native::{Input as SortedInput,Destination as SortedDestination,OwnerShape,SortedOwner};
use mgbfs_core::{graph_definition::{GraphDefinitionV2,GraphAction},Result};
use mgbfs_cuda::{ffi::*,generic_graph::*};
use crate::{generic_native::{GenericNativeBfs,Buffer,Stream,check},generic_memory::GenericMemoryPlan,generic_distributed_memory::GenericDistributedMemoryPlan};
extern "C" {
 fn mgbfs_nccl_exchange_bounded_triplets(comm:*mut c_void,rank:u32,world:u32,shards:u32,capacity:u32,used:u32,payload:u64,state_bytes:u32,counts:*const c_void,records:*const c_void,states:*const c_void,recv_counts:*mut c_void,recv_records:*mut c_void,recv_states:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_nccl_all_gather_bytes(comm:*mut c_void,send:*const c_void,recv:*mut c_void,bytes:u64,stream:*mut c_void)->i32;
 fn mgbfs_generic_pack_parent_chunk(bytes:u32,elements:u32,source:*const c_void,source_stride:u32,count:u32,out:*mut c_void,stride:u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_advance_parent_cursors(cursors:*mut u64,frontiers:*mut u32,next:*const u32,world:u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_accept_parent_origin_wide(bytes:u32,kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,permutations:*const u32,matrices:*const i64,moduli:*const u32,
 parents:*const c_void,parent_stride:u32,chunk:u32,begin:u64,cursors:*const u64,frontiers:*const u32,
 local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,
 slots:*mut u64,slot_capacity:u64,arena:*mut c_void,arena_stride:u32,base:u32,capacity:u32,visited_count:*mut u32,future:*mut u32,accepted:*mut u32,positions:*mut u64,error:*mut u32,rolling:u32,ordered:*const u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_sort_origins(lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,keys:*mut u64,sorted_keys:*mut u64,origins:*mut u32,sorted_origins:*mut u32,scratch:*mut c_void,scratch_bytes:u64,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_accept_sorted_wide(bytes:u32,elements:u32,local:*const c_void,remote:*const c_void,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,slots:*mut u64,sc:u64,arena:*mut c_void,stride:u32,base:u32,capacity:u32,vc:*mut u32,future:*mut u32,accepted:*mut u32,positions:*mut u64,error:*mut u32,rolling:u32,ordered:*const u32,stream:*mut c_void)->i32;
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
 online_size_tuning:bool,online_bucket:Option<u32>,online_choices:std::collections::BTreeMap<u32,(u32,bool)>,profile_events:Vec<serde_json::Value>,active_batch:u32,use_gemm:bool,gemm:Option<crate::generic_gemm::Context>,bfs:GenericNativeBfs,plan:GenericDistributedMemoryPlan,rank:u32,comm:*mut c_void,
 sorted_owner:Option<SortedOwner>,sort_cache:Option<SortCache>,parent_cache:Option<ParentCache>,banks:[QueueBank;2],inbox:QueueBank,control:Buffer,owner_map:Buffer,owner_cuts:Option<Buffer>,
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
  if crate::generic_gemm::requested()?&&!crate::generic_gemm::eligible(graph){return Err("GEMM_GRAPH_NOT_EXACTLY_SUPPORTED".into());}
  let use_sorted=plan.history_algorithm=="SORTED_RUNS";
  if !use_sorted&&std::env::var("MGBFS_GENERIC_HISTORY").as_deref()==Ok("sorted"){return Err("SORTED_HISTORY_REQUIRES_SERIALIZED_ADMISSION".into());}
  let admitted_sorted_shape=if use_sorted{
   graph.validate()?;
   if graph.start.len()!=plan.elements as usize{return Err("SORTED_GRAPH_PLAN_WIDTH".into());}
   let(kind,rows,cols)=match &graph.action{GraphAction::Permutation{degree,..}=>(0,*degree,1),GraphAction::Matrix{rows,cols,..}=>(1,*rows,*cols)};
   let ordinal=i32::try_from(device).map_err(|_|"SORTED_DEVICE_ORDINAL")?;
   let input=SortedInput{elements:plan.elements,world:plan.world,rank,shards:plan.shards,queue_capacity:plan.queue_capacity,state_bytes:plan.state_bytes,transport:if plan.packed_candidates(){1}else if plan.parent_transport{2}else{0},kind,rows,cols,generators:graph.generator_count() as u32,parent_stride:plan.batch,hash_bits,seed,..Default::default()};
   let shape=OwnerShape::query(&input,plan.capacity,plan.history_layers,plan.owner_lanes,ordinal)?;
   if shape.allocated_bytes as u64!=plan.sorted_owner_bytes{return Err("SORTED_OWNER_SERIALIZED_SHAPE_MISMATCH".into());}
   let(mut free,mut total)=(0,0);check(unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
   if plan.device_bytes>free as u64{return Err("SORTED_PLAN_FREE_MEMORY_CHANGED".into());}
   Some(shape)
  }else{None};
  let local=GenericMemoryPlan::with_storage(plan.elements,plan.capacity,plan.state_bytes)?;
  let rolling=plan.history_layers==3;if rolling&&!graph.inverse_closed()?{return Err("GENERIC_ROLLING_REQUIRES_INVERSE_CLOSED".into());}
  let mut bfs=if use_sorted{GenericNativeBfs::new_unseeded_sorted(graph,device,local,seed,hash_bits,rolling)?}else if rolling{GenericNativeBfs::new_unseeded_rolling(graph,device,local,seed,hash_bits)?}else{GenericNativeBfs::new_unseeded(graph,device,local,seed,hash_bits)?};let d=bfs.device;
  if bfs.slots.bytes!=plan.table_slots as usize*8{return Err("GENERIC_SHARED_TABLE_ALLOCATION_SHAPE".into());}
  if !use_sorted{check(unsafe{cudaMemsetAsync(bfs.slots.ptr,255,bfs.slots.bytes,bfs.stream.ptr)})?;}
  let mut hash=seed;for(e,v)in graph.start.iter().enumerate(){hash=mix(hash^(*v as u64)^(e as u64));}hash=mix(hash);if hash_bits<64{hash&=if hash_bits==0{0}else{(1u64<<hash_bits)-1};}
  let owner=if let Some(c)=cuts{c.windows(2).position(|v|v[0]<=hash>>32&&hash>>32<v[1]).ok_or("GENERIC_ROOT_OWNER")? as u32}else{(((hash>>32)*u64::from(plan.world))>>32) as u32};
  // Swap root logical owner with rank zero, preserving a bijective rank map.
  let mut map=(0..plan.world).collect::<Vec<_>>();if cuts.is_none(){map.swap(0,owner as usize);}
  let root_rank=map[owner as usize];
  let owner_cuts=if let Some(c)=cuts{let v=Buffer::new(c.len()*8,d)?;v.upload(c)?;Some(v)}else{None};
  let owner_map=Buffer::new(map.len()*4,d)?;owner_map.upload(&map)?;
  bfs.count=if rank==root_rank{1}else{0};bfs.visited_used=bfs.count;bfs.control.upload(&[bfs.count,0u32,0,0,0,0])?;
  let positions=if rolling&&!use_sorted{Some(Buffer::new(bfs.arena_stride as usize*8,d)?)}else{None};
  if rank==root_rank&&rolling&&!use_sorted{check(unsafe{mgbfs_generic_reseed_rows_storage(plan.state_bytes,plan.elements,bfs.visited.ptr.cast(),bfs.arena_stride,0,1,bfs.slots.ptr.cast(),plan.table_slots,positions.as_ref().unwrap().ptr.cast(),seed,hash_bits,bfs.control.at(8),bfs.stream.ptr)})?;}
  if rank==root_rank&&!rolling&&!use_sorted{check(unsafe{mgbfs_generic_seed_shared_storage(plan.state_bytes,plan.elements,bfs.visited.ptr.cast(),plan.capacity,1,bfs.slots.ptr.cast(),plan.table_slots,seed,hash_bits,bfs.control.at(8),bfs.stream.ptr)})?;}
  check(unsafe{cudaStreamSynchronize(bfs.stream.ptr)})?;
  let banks=[QueueBank::new(&plan,d)?,QueueBank::new(&plan,d)?];let inbox=QueueBank::new(&plan,d)?;
  let control=Buffer::new(64+plan.world as usize*8,d)?;
  let lanes=if use_sorted{plan.owner_lanes}else{plan.shards};
  let mut streams=vec![];let mut done=vec![];for _ in 0..lanes{streams.push(Stream::new(d)?);done.push(Event::new(d)?);}
  let sorted_owner=if use_sorted{
   let input=SortedInput{elements:plan.elements,world:plan.world,rank,shards:plan.shards,queue_capacity:plan.queue_capacity,state_bytes:plan.state_bytes,transport:if plan.packed_candidates(){1}else if plan.parent_transport{2}else{0},kind:bfs.kind,rows:bfs.rows,cols:bfs.cols,generators:bfs.generators,parent_stride:plan.batch,hash_bits,seed,..Default::default()};
   let shape=admitted_sorted_shape.ok_or("SORTED_OWNER_ADMITTED_SHAPE_MISSING")?;
   let(mut free,mut total)=(0,0);check(unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
   // Revalidate the serialized native shape; driver graph reserve is separate
   // from application buffers and is not advertised as byte-exact driver use.
   if shape.allocated_bytes as u64!=plan.sorted_owner_bytes{return Err("SORTED_OWNER_SERIALIZED_SHAPE_MISMATCH".into());}
   let graph_reserve=usize::try_from(plan.driver_graph_reserve_bytes).map_err(|_|"SORTED_OWNER_ADMISSION_OVERFLOW")?;
   if shape.allocated_bytes.checked_add(graph_reserve).ok_or("SORTED_OWNER_ADMISSION_OVERFLOW")?>free{return Err("SORTED_OWNER_ADMISSION_NO_CAPACITY".into());}
   let pointers=streams.iter().map(|v|v.ptr).collect::<Vec<_>>();
   let owner=SortedOwner::new(&input,shape,bfs.visited.ptr,bfs.arena_stride,bfs.control.at(8),&pointers,d)?;
   if rank==root_rank{let shard=((u64::from(hash as u32)*u64::from(plan.shards))>>32) as u32;owner.seed(shard,hash,bfs.control.at(8),bfs.stream.ptr)?;}
   check(unsafe{cudaStreamSynchronize(bfs.stream.ptr)})?;Some(owner)
  }else{None};
  let ready=Event::new(d)?;let mut comm=ptr::null_mut();let mut error=[0i8;512];
  check(unsafe{mgbfs_nccl_create(rank,plan.world,device,id.as_ptr().cast(),&mut comm,error.as_mut_ptr(),error.len())})?;
  let counts=(0..plan.world).map(|r|if r==root_rank{1u32}else{0}).collect::<Vec<_>>();
  let sort_cache=if plan.sort_candidates&&!use_sorted{Some(SortCache::new(&plan,d)?)}else{None};
  let parent_cache=if plan.parent_transport{Some(ParentCache::new(&plan,d,&counts)?)}else{None};
  let rolling_counts=[bfs.count,0,0];
  let gemm=if crate::generic_gemm::requested()?{Some(crate::generic_gemm::Context::new(graph,plan.batch,d)?)}else{None};
  Ok(Self{online_size_tuning:false,online_bucket:None,online_choices:std::collections::BTreeMap::new(),profile_events:vec![],active_batch:plan.batch,use_gemm:gemm.is_some(),gemm,sorted_owner,sort_cache,parent_cache,positions,rolling_counts,retired_rows:0,bfs,plan,rank,comm,banks,inbox,control,owner_map,owner_cuts,streams,ready,done,depth:0,parent_cursor:0,max_frontier:1,counts,source_retries:0})
 }
 pub fn enable_online_size_tuning(&mut self){self.online_size_tuning=true;}
 pub fn size_profile_events(&self)->&[serde_json::Value]{&self.profile_events}
 pub fn set_size_profile(&mut self,batch:u32,gemm:bool)->Result<()>{
  if batch==0||batch>self.plan.batch||(gemm&&self.gemm.is_none()){return Err("SIZE_PROFILE_EXCEEDS_ADMISSION".into());}
  let bucket=31-self.max_frontier.max(1).leading_zeros();let(batch,gemm)=self.online_choices.get(&bucket).copied().unwrap_or((batch,gemm));self.active_batch=batch;self.use_gemm=gemm;Ok(())
 }
 pub fn cache_current_size_profile(&mut self,batch:u32,gemm:bool)->Result<()>{
  self.set_size_profile(batch,gemm)?;let bucket=31-self.max_frontier.max(1).leading_zeros();self.online_choices.insert(bucket,(batch,gemm));Ok(())
 }
 pub fn global_frontier(&self)->u64{self.counts.iter().map(|&v|u64::from(v)).sum()}
 pub fn history_algorithm(&self)->&str{&self.plan.history_algorithm}
 pub fn owner_lanes(&self)->usize{self.streams.len()}
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
  if rolling&&self.sorted_owner.is_some(){
   self.sorted_owner.as_ref().unwrap().retire_bank(future_bank as u32,b.control.at(8),stream)?;self.rolling_counts[future_bank]=0;
  }else if rolling{
   let positions=self.positions.as_ref().ok_or("GENERIC_ROLLING_POSITIONS")?;let expired=self.rolling_counts[future_bank];
   if expired!=0{check(unsafe{mgbfs_generic_retire_rows_wide(b.slots.ptr.cast(),p.table_slots,positions.ptr.cast(),b.arena_stride,future_base,expired,b.control.at(8),stream)})?;self.retired_rows+=u64::from(expired);self.rolling_counts[future_bank]=0;}
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
  let mut global_begin=0u64;let mut batch=self.active_batch.min(self.max_frontier);let mut round=0u64;
  // Each trial processes actual new parents once. Only scalar timing/counts
  // cross the host boundary; states and history stay resident.
  let bucket=31-self.max_frontier.max(1).leading_zeros();
  let mut profiling=self.online_size_tuning&&self.online_bucket!=Some(bucket)&&!self.online_choices.contains_key(&bucket)&&self.max_frontier>=p.batch.saturating_mul(5);
  let mut choices=if profiling{vec![(p.batch,self.use_gemm),((p.batch/4).max(1),self.use_gemm)]}else{vec![]};
  if profiling&&self.gemm.is_some(){choices.push((p.batch,!self.use_gemm));choices.push(((p.batch/4).max(1),!self.use_gemm));}
  let schedule=(0..choices.len()).chain((0..choices.len()).rev()).collect::<Vec<_>>();
  let mut trial_scores=vec![(0u64,0u64);choices.len()];let mut trial=0usize;
  // Drain shared layer maintenance before the first candidate timer. A
  // collective boundary also excludes another rank's retirement/reseed work.
  if profiling{
   check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm,b.control.at(4),self.control.at(16),stream)})?;
   check(unsafe{cudaStreamSynchronize(stream)})?;
  }
  let queues=p.world as usize*p.shards as usize;let q=p.queue_capacity as usize;let width=p.elements as usize;let shards=p.shards as usize;let state_bytes=p.state_bytes as usize;
  while global_begin<u64::from(self.max_frontier){
   if profiling&&trial<schedule.len(){let choice=choices[schedule[trial]];batch=choice.0;self.use_gemm=choice.1;}
   let trial_started=if profiling&&trial<schedule.len(){Some(std::time::Instant::now())}else{None};

   let bank=&self.banks[((self.depth+round)%2) as usize];let begin=global_begin.min(u64::from(b.count)) as u32;let count=batch.min(b.count-begin);
   check(unsafe{cudaMemsetAsync(self.control.at::<c_void>(8),0,4,stream)})?;
   check(unsafe{cudaMemsetAsync(bank.counts.ptr,0,bank.counts.bytes,stream)})?;
   if let Some(ctx)=self.gemm.as_ref().filter(|_|self.use_gemm){check(unsafe{mgbfs_generic_route_gemm_storage(ctx.ptr,p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),b.seed,b.hash_bits,p.world,self.rank,p.shards,p.queue_capacity,self.parent_cursor+u64::from(begin),self.owner_map.ptr.cast(),self.owner_cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),bank.records.ptr.cast(),bank.counts.ptr.cast(),self.control.at(8),stream)})?;}else{check(unsafe{mgbfs_generic_route_storage(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),b.seed,b.hash_bits,p.world,self.rank,p.shards,p.queue_capacity,self.parent_cursor+u64::from(begin),self.owner_map.ptr.cast(),self.owner_cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),bank.records.ptr.cast(),bank.counts.ptr.cast(),self.control.at(8),stream)})?;}
   if !p.packed_candidates()&&!p.parent_transport{for queue in 0..queues{if let Some(ctx)=self.gemm.as_ref().filter(|_|self.use_gemm){check(unsafe{mgbfs_generic_regenerate_gemm_routes_count_storage(ctx.ptr,p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),self.rank,self.parent_cursor+u64::from(begin),bank.records.at(queue*q*32),p.queue_capacity,bank.counts.at(queue*4),bank.states.at(queue*q*width*state_bytes),p.queue_capacity,self.control.at(8),stream)})?;}else{check(unsafe{mgbfs_generic_regenerate_routes_count_storage(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),self.rank,self.parent_cursor+u64::from(begin),bank.records.at(queue*q*32),p.queue_capacity,bank.counts.at(queue*4),bank.states.at(queue*q*width*state_bytes),p.queue_capacity,self.control.at(8),stream)})?;}}}
   if let Some(cache)=&self.parent_cache{
    check(unsafe{mgbfs_generic_pack_parent_chunk(p.state_bytes,p.elements,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),parent_stride,count,cache.sources[((self.depth+round)%2) as usize].ptr,batch,stream)})?;
   }
   // Invalid source counts must not be presented as valid owner inboxes.
   // Generate the next immutable source bank while preceding owners work.
   // Retire their inbox leases only before reading owner errors/exchanging.
   if round>0{for event in &self.done{check(unsafe{cudaStreamWaitEvent(stream,event.ptr,0)})?;}}
   check(unsafe{mgbfs_generic_route_retry_vote(self.control.at(8),b.control.at(8),self.control.at(12),stream)})?;
   check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm,self.control.at(12),self.control.at(0),stream)})?;
   let mut fatal=0u32;check(unsafe{cudaMemcpyAsync((&mut fatal as *mut u32).cast(),self.control.ptr,4,2,stream)})?;check(unsafe{cudaStreamSynchronize(stream)})?;
   if fatal==1&&batch>1{profiling=false;self.online_bucket=Some(bucket);batch=(batch/2).max(1);self.source_retries+=1;continue;}
   if fatal!=0{return Ok(DistributedAdvance::Resource{fatal});}
   // A single matched NCCL group submits all peer lanes without host count reads.
   let used=u32::try_from((u64::from(batch)*u64::from(b.generators)).min(u64::from(p.queue_capacity))).map_err(|_|"BOUNDED_QUEUE_EXTENT")?;
   check(unsafe{mgbfs_nccl_exchange_bounded_triplets(self.comm,self.rank,p.world,p.shards,p.queue_capacity,used,p.queue_payload_bytes() as u64,p.state_bytes,bank.counts.ptr,bank.records.ptr,bank.states.ptr,self.inbox.counts.ptr,self.inbox.records.ptr,self.inbox.states.ptr,stream)})?;
   if let Some(cache)=&self.parent_cache{
    check(unsafe{mgbfs_nccl_all_gather_bytes(self.comm,cache.sources[((self.depth+round)%2) as usize].ptr,cache.received.ptr,u64::from(batch)*u64::from(p.elements)*u64::from(p.state_bytes),stream)})?;
   }
   check(unsafe{cudaEventRecord(self.ready.ptr,stream)})?;
   for shard in 0..shards{let owner_stream=self.streams[shard%self.streams.len()].ptr;if shard<self.streams.len(){check(unsafe{cudaStreamWaitEvent(owner_stream,self.ready.ptr,0)})?;}
    if let Some(owner)=&self.sorted_owner{
     let mut input=SortedInput{elements:p.elements,world:p.world,rank:self.rank,shard:shard as u32,shards:p.shards,queue_capacity:p.queue_capacity,state_bytes:p.state_bytes,transport:if p.packed_candidates(){1}else if p.parent_transport{2}else{0},kind:b.kind,rows:b.rows,cols:b.cols,generators:b.generators,parent_stride:batch,chunk:batch,hash_bits:b.hash_bits,begin:global_begin,seed:b.seed,local:bank.states.ptr,remote:self.inbox.states.ptr,local_meta:bank.records.ptr,remote_meta:self.inbox.records.ptr,local_counts:bank.counts.ptr,remote_counts:self.inbox.counts.ptr,permutations:b.perms.ptr,matrices:b.matrices.ptr,moduli:b.moduli.ptr,..Default::default()};
     if let Some(cache)=&self.parent_cache{input.parents=cache.received.ptr;input.cursors=cache.cursors.ptr;input.frontiers=cache.frontiers.ptr;}
     let destination=SortedDestination{arena:b.visited.ptr,stride:b.arena_stride,base:if rolling{future_base}else{0},capacity:p.capacity,rolling:u32::from(rolling),row_count:if rolling{b.control.at(4)}else{b.control.at(0)},frontier_count:b.control.at(4),future:b.future.ptr,error:b.control.at(8),..Default::default()};
     owner.accept(&input,destination,if rolling{future_bank as u32}else{0})?;
     continue;
    }
    let ordered=if let Some(cache)=&self.sort_cache{
     let offset=shard*p.world as usize*q;
     check(unsafe{mgbfs_generic_sort_origins(bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,cache.keys.at(offset*8),cache.sorted_keys.at(offset*8),cache.origins.at(offset*4),cache.sorted_origins.at(offset*4),cache.scratch.at::<c_void>(shard*cache.scratch_stride),cache.scratch_stride as u64,b.control.at(8),owner_stream)})?;cache.sorted_origins.at::<u32>(offset*4).cast_const()
    }else{ptr::null()};
    if let Some(cache)=&self.parent_cache{
     check(unsafe{mgbfs_generic_accept_parent_origin_wide(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),cache.received.ptr,batch,batch,global_begin,cache.cursors.ptr.cast(),cache.frontiers.ptr.cast(),bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr,b.arena_stride,if rolling{future_base}else{0},p.capacity,b.control.at(0),b.future.ptr.cast(),b.control.at(4),self.positions.as_ref().map_or(ptr::null_mut(),|v|v.ptr.cast()),b.control.at(8),u32::from(rolling),ordered,owner_stream)})?;
    }else if self.sort_cache.is_some(){
     check(unsafe{mgbfs_generic_accept_sorted_wide(p.state_bytes,p.elements,bank.states.ptr, self.inbox.states.ptr,bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr,b.arena_stride,if rolling{future_base}else{0},p.capacity,b.control.at(0),b.future.ptr.cast(),b.control.at(4),self.positions.as_ref().map_or(ptr::null_mut(),|v|v.ptr.cast()),b.control.at(8),u32::from(rolling),ordered,owner_stream)})?;
    }else if rolling{check(unsafe{mgbfs_generic_accept_rolling_storage(p.state_bytes,p.elements,bank.states.ptr.cast(),self.inbox.states.ptr.cast(),bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr.cast(),b.arena_stride,future_base,p.capacity,b.control.at(4),b.future.ptr.cast(),self.positions.as_ref().unwrap().ptr.cast(),b.control.at(8),owner_stream)})?;}
    else{    check(unsafe{mgbfs_generic_accept_all_storage(p.state_bytes,p.elements,bank.states.ptr.cast(),self.inbox.states.ptr.cast(),bank.records.ptr.cast(),self.inbox.records.ptr.cast(),bank.counts.ptr.cast(),self.inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr.cast(),p.capacity,b.control.at(0),b.future.ptr.cast(),p.capacity,b.control.at(4),b.control.at(8),owner_stream)})?;}
   }
   for(lane,owner_stream)in self.streams.iter().enumerate(){check(unsafe{cudaEventRecord(self.done[lane].ptr,owner_stream.ptr)})?;}
   // Next round may generate into the other source bank; inbox retirement
   // waits occur after that generation and before the next exchange.
   if profiling&&trial<schedule.len(){
    for event in &self.done{check(unsafe{cudaStreamWaitEvent(stream,event.ptr,0)})?;}
    check(unsafe{cudaStreamSynchronize(stream)})?;
    let mut owner_error=0u32;check(unsafe{cudaMemcpyAsync((&mut owner_error as *mut u32).cast(),b.control.at::<c_void>(8),4,2,stream)})?;check(unsafe{cudaStreamSynchronize(stream)})?;
    let micros=if owner_error!=0{u32::MAX}else{trial_started.unwrap().elapsed().as_micros().min(u128::from(u32::MAX-1)) as u32};
    check(unsafe{cudaMemcpyAsync(self.control.at::<c_void>(16),(&micros as *const u32).cast(),4,1,stream)})?;
    check(unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm,self.control.at(16),self.control.at(20),stream)})?;
    let mut maximum=0u32;check(unsafe{cudaMemcpyAsync((&mut maximum as *mut u32).cast(),self.control.at::<c_void>(20),4,2,stream)})?;check(unsafe{cudaStreamSynchronize(stream)})?;
    let parents=self.counts.iter().map(|&v|u64::from(v).saturating_sub(global_begin).min(u64::from(batch))).sum::<u64>();
    if maximum==u32::MAX{profiling=false;self.online_bucket=Some(bucket);}
    let score=&mut trial_scores[schedule[trial]];score.0+=u64::from(maximum);score.1+=parents;trial+=1;
    if profiling&&trial==schedule.len(){
     let rate=|i:usize|trial_scores[i].0 as f64/trial_scores[i].1.max(1) as f64;
     let mut winner=0usize;for i in 1..choices.len(){if rate(i)<rate(winner){winner=i;}}
     if rate(winner)>=rate(0)*0.90{winner=0;}
     let choice=choices[winner];self.online_choices.insert(bucket,choice);self.online_bucket=Some(bucket);self.active_batch=choice.0;self.use_gemm=choice.1;
     self.profile_events.push(serde_json::json!({"frontier_global":self.counts.iter().map(|&v|u64::from(v)).sum::<u64>(),"frontier_max_rank":self.max_frontier,"bucket":bucket,"choices":choices,"microseconds_and_parents":trial_scores,"winner":winner,"scope":"live disjoint parent chunks, full generation/route/exchange/dedup, synchronized pilot boundary; sampling heuristic, not identical-work replay"}));
    }
   }
   global_begin+=u64::from(batch);round+=1;
   if profiling&&trial==schedule.len(){batch=self.active_batch;}
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
