//! General CayleyPy exact GPU engine. The visited arena is mandatory for
//! directed actions. Specialized compact SHARD_AB remains a separate fast path.
//! No CPU generation/deduplication and no per-batch state readback.
use std::{ffi::c_void,ptr};
use mgbfs_core::{graph_definition::{GraphDefinitionV2,GraphAction},Result};
use mgbfs_cuda::{ffi::*,generic_graph::*,native_owner::{cudaSetDevice,cudaMemGetInfo}};
use crate::generic_memory::GenericMemoryPlan;
pub(crate) fn check(code:i32)->Result<()> {if code==0{Ok(())}else{Err(format!("GENERIC_CUDA_{code}"))}}
pub(crate) struct Buffer {pub(crate) ptr:*mut c_void,pub(crate) bytes:usize,pub(crate) device:i32}
impl Buffer {
 pub(crate) fn new(bytes:usize,device:i32)->Result<Self>{let mut p=ptr::null_mut();check(unsafe{cudaMalloc(&mut p,bytes.max(1))})?;Ok(Self{ptr:p,bytes,device})}
 pub(crate) fn upload<T>(&self,data:&[T])->Result<()> {let bytes=std::mem::size_of_val(data);if bytes>self.bytes{return Err("GENERIC_UPLOAD_BOUNDS".into());}check(unsafe{cudaMemcpy(self.ptr,data.as_ptr().cast(),bytes,1)})}
 pub(crate) fn at<T>(&self,offset:usize)->*mut T{unsafe{self.ptr.cast::<u8>().add(offset).cast()}}
}
impl Drop for Buffer {fn drop(&mut self){unsafe{cudaSetDevice(self.device);cudaFree(self.ptr);}}}
pub(crate) struct Stream {pub(crate) ptr:*mut c_void,pub(crate) device:i32}
impl Stream {pub(crate) fn new(device:i32)->Result<Self>{let mut p=ptr::null_mut();check(unsafe{cudaStreamCreateWithFlags(&mut p,1)})?;Ok(Self{ptr:p,device})}}
impl Drop for Stream {fn drop(&mut self){unsafe{cudaSetDevice(self.device);cudaStreamSynchronize(self.ptr);cudaStreamDestroy(self.ptr);}}}
#[derive(Debug,PartialEq,Eq)]
pub enum GenericAdvance {Layer{count:u32},Complete,Resource{fatal:u32}}
pub struct GenericNativeBfs {
 pub(crate) device:i32,
 pub(crate) stream:Stream,
 pub(crate) plan:GenericMemoryPlan,
 pub(crate) kind:u32,
 pub(crate) rows:u32,
 pub(crate) cols:u32,
 pub(crate) generators:u32,
 pub(crate) count:u32,
 pub(crate) seed:u64,
 pub(crate) hash_bits:u32,
 pub(crate) terminal:bool,
 pub(crate) visited_used:u32,
 pub(crate) current_start:u32,
 pub(crate) previous:Option<(u32,u32)>,
 pub(crate) visited:Buffer,
 pub(crate) parents:Buffer,
 pub(crate) front:Buffer,
 pub(crate) future:Buffer,
 pub(crate) slots:Buffer,
 pub(crate) control:Buffer,
 pub(crate) perms:Buffer,
 pub(crate) matrices:Buffer,
 pub(crate) moduli:Buffer,
}
impl GenericNativeBfs {
 pub fn automatic_plan(graph:&GraphDefinitionV2,device:u32)->Result<GenericMemoryPlan>{
  graph.validate()?;let ordinal=i32::try_from(device).map_err(|_|"GENERIC_DEVICE_ORDINAL")?;check(unsafe{cudaSetDevice(ordinal)})?;
  let mut free=0;let mut total=0;check(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
  let elements=u32::try_from(graph.start.len()).map_err(|_|"GENERIC_STATE_WIDTH")?;
  GenericMemoryPlan::automatic_storage(elements,free as u64,crate::generic_memory::state_space_bound(graph),crate::generic_memory::preferred_state_bytes(graph))
 }
 pub fn new(graph:&GraphDefinitionV2,device:u32,plan:GenericMemoryPlan,seed:u64,hash_bits:u32)->Result<Self>{
  graph.validate()?;if plan.elements as usize!=graph.start.len()||hash_bits>64{return Err("GENERIC_GRAPH_PLAN".into());}
  // Recompute the public plan contract before admitting any allocation.
  if plan.state_bytes==1&&crate::generic_memory::preferred_state_bytes(graph)!=1{return Err("GENERIC_COMPACT_ALPHABET".into());}
  let checked=GenericMemoryPlan::with_storage(plan.elements,plan.capacity,plan.state_bytes)?;
  if checked.table_slots!=plan.table_slots||checked.device_bytes!=plan.device_bytes||plan.batch==0||plan.batch>plan.capacity{return Err("GENERIC_PLAN_MUTATED".into());}
  let device=i32::try_from(device).map_err(|_|"GENERIC_DEVICE_ORDINAL")?;check(unsafe{cudaSetDevice(device)})?;
  #[cfg(target_os="linux")] crate::cuda_loading::verify_driver_before_allocations()?;
  let (kind,rows,cols,ps,ms,mods)=match &graph.action {
   GraphAction::Permutation{degree,generators}=>(0,*degree,1,generators.iter().flatten().copied().collect::<Vec<u32>>(),vec![],vec![]),
   GraphAction::Matrix{rows,cols,generators}=>(1,*rows,*cols,vec![],generators.iter().flat_map(|g|g.matrix.iter().copied()).collect::<Vec<i64>>(),generators.iter().map(|g|g.modulo).collect::<Vec<u32>>()),
  };
  let generators=u32::try_from(graph.generator_count()).map_err(|_|"GENERIC_GENERATOR_COUNT")?;
  if generators==0||generators>=0x7fff_ffff{return Err("GENERIC_GENERATOR_COUNT".into());}
  let width=plan.elements as usize;let cap=plan.capacity as usize;let state_bytes=plan.state_bytes as usize;
  let stream=Stream::new(device)?;
  let visited=Buffer::new(width*cap*state_bytes,device)?;let parents=Buffer::new(width*cap*state_bytes,device)?;
  let front=Buffer::new(cap*4,device)?;let future=Buffer::new(cap*4,device)?;let slots=Buffer::new(plan.table_slots as usize*8,device)?;let control=Buffer::new(24,device)?;
  let perms=Buffer::new(ps.len()*4,device)?;perms.upload(&ps)?;let matrices=Buffer::new(ms.len()*8,device)?;matrices.upload(&ms)?;let moduli=Buffer::new(mods.len()*4,device)?;moduli.upload(&mods)?;
  front.upload(&[0u32])?;control.upload(&[1u32,0,0,0,0,0])?;
  for (e,value) in graph.start.iter().enumerate(){let byte=*value as u8;let source=if state_bytes==1{(&byte as *const u8).cast()}else{(value as *const i64).cast()};check(unsafe{cudaMemcpy(visited.at::<c_void>(e*cap*state_bytes),source,state_bytes,1)})?;}
  check(unsafe{cudaMemsetAsync(slots.ptr,255,slots.bytes,stream.ptr)})?;
  check(unsafe{mgbfs_generic_seed_storage(plan.state_bytes,plan.elements,visited.ptr.cast(),plan.capacity,1,slots.ptr.cast(),plan.table_slots,seed,hash_bits,control.at(8),stream.ptr)})?;
  check(unsafe{cudaStreamSynchronize(stream.ptr)})?;
  Ok(Self{device,stream,plan,kind,rows,cols,generators,visited,parents,front,future,slots,control,perms,matrices,moduli,count:1,seed,hash_bits,terminal:false,visited_used:1,current_start:0,previous:None})
 }
 fn read_soa(&self,count:u32)->Result<Vec<i64>>{
  let length=count as usize*self.plan.elements as usize;
  if self.plan.state_bytes==1{let mut raw=vec![0u8;length];check(unsafe{cudaMemcpyAsync(raw.as_mut_ptr().cast(),self.parents.ptr,length,2,self.stream.ptr)})?;check(unsafe{cudaStreamSynchronize(self.stream.ptr)})?;Ok(raw.into_iter().map(i64::from).collect())}
  else{let mut raw=vec![0i64;length];check(unsafe{cudaMemcpyAsync(raw.as_mut_ptr().cast(),self.parents.ptr,length*8,2,self.stream.ptr)})?;check(unsafe{cudaStreamSynchronize(self.stream.ptr)})?;Ok(raw)}
 }
 pub fn stop(&mut self){self.terminal=true;}
 pub fn frontier_len(&self)->u32{self.count}
 pub fn advance(&mut self)->Result<GenericAdvance>{
  if self.terminal{return Err("GENERIC_TERMINAL_ARENA".into());}check(unsafe{cudaSetDevice(self.device)})?;let s=self.stream.ptr;self.terminal=true;
  check(unsafe{mgbfs_generic_gather_storage(self.plan.state_bytes,self.plan.elements,self.visited.ptr.cast(),self.plan.capacity,self.front.ptr.cast(),self.count,self.parents.ptr.cast(),self.plan.capacity,s)})?;
  check(unsafe{cudaMemsetAsync(self.control.at::<c_void>(4),0,4,s)})?;
  let batch=self.plan.batch.min(0x7fff_fffe/self.generators);if batch==0{return Err("GENERIC_BATCH_CAPACITY".into());}
  let mut begin=0;
  while begin<self.count {
   let count=batch.min(self.count-begin);
   check(unsafe{mgbfs_generic_expand_storage(self.plan.state_bytes,self.kind,self.plan.elements,self.rows,self.cols,self.generators,
    self.parents.at::<i64>(begin as usize*self.plan.state_bytes as usize),count,self.plan.capacity,self.perms.ptr.cast(),self.matrices.ptr.cast(),self.moduli.ptr.cast(),
    self.slots.ptr.cast(),self.plan.table_slots,self.visited.ptr.cast(),self.plan.capacity,self.control.at(0),self.future.ptr.cast(),self.plan.capacity,self.control.at(4),self.seed,self.hash_bits,self.control.at(8),s)})?;
   begin+=count;
  }
  // One scalar layer-boundary readback, independent of the number of batches.
  let mut state=[0u32;3];check(unsafe{cudaMemcpyAsync(state.as_mut_ptr().cast(),self.control.ptr,12,2,s)})?;check(unsafe{cudaStreamSynchronize(s)})?;
  if state[2]!=0 {self.terminal=true;return Ok(GenericAdvance::Resource{fatal:state[2]});}
  if state[1]==0 {self.terminal=true;return Ok(GenericAdvance::Complete);}
  if state[1]>self.plan.capacity{return Err("GENERIC_COUNT_BOUNDS".into());}
  self.previous=Some((self.current_start,self.count));self.current_start=self.visited_used;self.visited_used=state[0];
  std::mem::swap(&mut self.front,&mut self.future);self.count=state[1];self.terminal=false;Ok(GenericAdvance::Layer{count:self.count})
 }
 /// Compact output keeps the previous layer only when it is strictly <1000.
 /// Retained all-visited rows allow this without any hot-path state copies.
 pub fn previous_small_sample(&mut self)->Result<Vec<Vec<i64>>>{
  if !self.terminal{return Err("GENERIC_PREVIOUS_REQUIRES_TERMINAL".into());}
  let (start,count)=match self.previous {Some(pair) if pair.1<1000=>pair,_=>return Ok(vec![])};
  check(unsafe{cudaSetDevice(self.device)})?;
  let indices=(start..start+count).collect::<Vec<u32>>();self.future.upload(&indices)?;
  check(unsafe{mgbfs_generic_gather_storage(self.plan.state_bytes,self.plan.elements,self.visited.ptr.cast(),self.plan.capacity,self.future.ptr.cast(),count,self.parents.ptr.cast(),count,self.stream.ptr)})?;
  let soa=self.read_soa(count)?;
  Ok((0..count as usize).map(|i|(0..self.plan.elements as usize).map(|e|soa[e*count as usize+i]).collect()).collect())
 }
 /// Terminal/sample readout. No state copies occur in advance().
 pub fn sample(&mut self,limit:u32)->Result<Vec<Vec<i64>>>{
  check(unsafe{cudaSetDevice(self.device)})?;let count=limit.min(self.count);if count==0{return Ok(vec![]);}
  check(unsafe{mgbfs_generic_gather_storage(self.plan.state_bytes,self.plan.elements,self.visited.ptr.cast(),self.plan.capacity,self.front.ptr.cast(),count,self.parents.ptr.cast(),count,self.stream.ptr)})?;
  let soa=self.read_soa(count)?;
  Ok((0..count as usize).map(|i|(0..self.plan.elements as usize).map(|e|soa[e*count as usize+i]).collect()).collect())
 }
}
