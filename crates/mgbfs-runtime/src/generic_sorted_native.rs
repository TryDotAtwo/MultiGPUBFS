//! Native sorted owner ABI and exact CUB workspace allocation. Counters and
//! state identities stay on GPU; shape queries run once during admission.
use std::{ffi::c_void,mem,ptr};
use mgbfs_core::Result;
use crate::generic_native::{Buffer,check};
#[repr(C)]#[derive(Clone,Copy)]
pub(crate) struct Input {
 pub elements:u32,pub world:u32,pub rank:u32,pub shard:u32,pub shards:u32,pub queue_capacity:u32,pub state_bytes:u32,pub transport:u32,
 pub kind:u32,pub rows:u32,pub cols:u32,pub generators:u32,pub parent_stride:u32,pub chunk:u32,pub hash_bits:u32,pub reserved:u32,
 pub begin:u64,pub seed:u64,
 pub local:*const c_void,pub remote:*const c_void,pub local_meta:*const c_void,pub remote_meta:*const c_void,pub local_counts:*const c_void,pub remote_counts:*const c_void,
 pub parents:*const c_void,pub cursors:*const c_void,pub frontiers:*const c_void,pub permutations:*const c_void,pub matrices:*const c_void,pub moduli:*const c_void,
}
impl Default for Input{fn default()->Self{unsafe{mem::zeroed()}}}
#[repr(C)]#[derive(Clone,Copy)]
pub(crate) struct Workspace {
 pub hashes:*mut c_void,pub origins:*mut c_void,pub sorted:*mut c_void,pub flags:*mut c_void,pub prefix:*mut c_void,pub unique_hashes:*mut c_void,pub unique_origins:*mut c_void,pub unique_count:*mut c_void,pub temporary:*mut c_void,pub temporary_bytes:u64,
}
#[repr(C)]#[derive(Clone,Copy)]
pub(crate) struct Destination {
 pub snapshots:*const c_void,pub snapshot_count:u32,pub stride:u32,pub base:u32,pub capacity:u32,pub rolling:u32,pub reserved:u32,
 pub pool:*mut c_void,pub run_hashes:*mut c_void,pub run_rows:*mut c_void,pub arena:*mut c_void,pub row_count:*mut c_void,pub frontier_count:*mut c_void,pub future:*mut c_void,pub reservation:*mut c_void,pub carry:*mut c_void,pub error:*mut c_void,
}
impl Default for Destination{fn default()->Self{unsafe{mem::zeroed()}}}
extern "C" {
 fn mgbfs_generic_sorted_native_shape(input:*const Input,total:*mut u64,temporary:*mut u64,count:*mut u64,stream:*mut c_void)->i32;
 fn mgbfs_generic_sorted_native_metadata_shape(out:*mut u64,count:u32)->i32;
 fn mgbfs_generic_sorted_native_origin(input:*const Input,workspace:*const Workspace,error:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_generic_sorted_native_accept(input:*const Input,workspace:*const Workspace,destination:*const Destination,stream:*mut c_void)->i32;
}
fn align(value:u64)->Result<u64>{Ok(value.checked_add(255).ok_or("SORTED_SHAPE_OVERFLOW")?&!255)}
#[derive(Clone,Copy,Debug)]
pub(crate) struct OriginShape{pub count:u32,pub allocation_bytes:usize,pub temporary_bytes:usize}
impl OriginShape{
 pub fn query(input:&Input,device:i32)->Result<Self>{
  check(unsafe{mgbfs_cuda::native_owner::cudaSetDevice(device)})?;
  let(mut bytes,mut temporary,mut count)=(0u64,0u64,0u64);
  check(unsafe{mgbfs_generic_sorted_native_shape(input,&mut bytes,&mut temporary,&mut count,ptr::null_mut())})?;
  let expected=u64::from(input.world)*u64::from(input.queue_capacity);
  if count!=expected||count>=0x7fff_ffff{return Err("SORTED_SHAPE_COUNT".into());}
  let admitted=align(count*8)?.checked_mul(2).and_then(|v|v.checked_add(align(count*4).ok()?.checked_mul(5)?)).and_then(|v|v.checked_add(align(temporary).ok()?)).and_then(|v|v.checked_add(256)).ok_or("SORTED_SHAPE_OVERFLOW")?;
  if bytes!=admitted{return Err("SORTED_SHAPE_ABI_MISMATCH".into());}
  Ok(Self{count:count as u32,allocation_bytes:usize::try_from(bytes).map_err(|_|"SORTED_SHAPE_OVERFLOW")?,temporary_bytes:usize::try_from(temporary).map_err(|_|"SORTED_SHAPE_OVERFLOW")?})
 }
}
pub(crate) fn metadata_sizes()->Result<[usize;5]>{let mut raw=[0u64;5];check(unsafe{mgbfs_generic_sorted_native_metadata_shape(raw.as_mut_ptr(),5)})?;let mut sizes=[0usize;5];for (out,value) in sizes.iter_mut().zip(raw){if value==0{return Err("SORTED_METADATA_SHAPE".into());}*out=usize::try_from(value).map_err(|_|"SORTED_SHAPE_OVERFLOW")?;}Ok(sizes)}
fn action_profile(input:&Input)->[u32;9]{[input.state_bytes,input.transport,input.kind,input.elements,input.world,input.shards,input.queue_capacity,input.rows,input.cols]}
pub(crate) struct OriginBuffers{buffer:Buffer,pub shape:OriginShape,workspace:Workspace,profile:[u32;9]}
impl OriginBuffers{
 pub fn new(input:&Input,device:i32)->Result<Self>{
  let shape=OriginShape::query(input,device)?;let buffer=Buffer::new(shape.allocation_bytes,device)?;let n=u64::from(shape.count);
  let hash_stride=align(n*8)? as usize;let index_stride=align(n*4)? as usize;
  let temp_offset=hash_stride*2+index_stride*5;let count_offset=temp_offset+align(shape.temporary_bytes as u64)? as usize;
  if count_offset+256!=shape.allocation_bytes{return Err("SORTED_WORKSPACE_LAYOUT".into());}
  let workspace=Workspace{hashes:buffer.at(0),unique_hashes:buffer.at(hash_stride),origins:buffer.at(hash_stride*2),sorted:buffer.at(hash_stride*2+index_stride),unique_origins:buffer.at(hash_stride*2+index_stride*2),flags:buffer.at(hash_stride*2+index_stride*3),prefix:buffer.at(hash_stride*2+index_stride*4),temporary:buffer.at(temp_offset),unique_count:buffer.at(count_offset),temporary_bytes:shape.temporary_bytes as u64};
  Ok(Self{buffer,shape,workspace,profile:action_profile(input)})
 }
 pub fn allocated_bytes(&self)->usize{self.buffer.bytes}
 pub fn generate_unique(&self,input:&Input,error:*mut c_void,stream:*mut c_void)->Result<()>{
  if action_profile(input)!=self.profile||u64::from(input.world)*u64::from(input.queue_capacity)!=u64::from(self.shape.count){return Err("SORTED_WORKSPACE_INPUT_GEOMETRY".into());}
  check(unsafe{mgbfs_generic_sorted_native_origin(input,&self.workspace,error,stream)})
 }
 pub fn accept(&self,input:&Input,destination:&Destination,stream:*mut c_void)->Result<()>{
  if action_profile(input)!=self.profile||u64::from(input.world)*u64::from(input.queue_capacity)!=u64::from(self.shape.count){return Err("SORTED_WORKSPACE_INPUT_GEOMETRY".into());}
  check(unsafe{mgbfs_generic_sorted_native_accept(input,&self.workspace,destination,stream)})
 }
}
#[cfg(test)]mod tests{use super::*;#[test]fn native_layout(){assert_eq!(mem::size_of::<Input>(),176);assert_eq!(mem::size_of::<Workspace>(),80);assert_eq!(mem::size_of::<Destination>(),112);assert_eq!(align(257).unwrap(),512);assert!(align(u64::MAX).is_err());}}

// Fixed-size owner lanes share origin workspaces. Logical shards retain their
// own epoch roots and immutable graph handles; no modulo reuse of live roots.
#[repr(C)]#[derive(Clone,Copy)]
struct Pool {occupied:*mut c_void,descriptors:*mut c_void,regions:u32,page_entries:u32}
#[repr(C)]struct CarryConfig {
 pool:*const Pool,tiers:*mut c_void,carry:*mut c_void,hashes:*mut c_void,rows:*mut c_void,arena:*mut c_void,error:*mut c_void,control:*mut c_void,handles:*mut c_void,
 stride:u32,width:u32,state_bytes:u32,classes:u32,
}
extern "C" {
 fn mgbfs_generic_sorted_native_carry_shape(page:u32,classes:u32,control:*mut u64,handles:*mut u64)->i32;
 fn mgbfs_generic_sorted_native_carry_create(config:*const CarryConfig,stream:*mut c_void,out:*mut *mut c_void)->i32;
 fn mgbfs_generic_sorted_native_carry_launch(graph:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_generic_sorted_native_carry_destroy(graph:*mut c_void)->i32;
 fn mgbfs_generic_sorted_native_snapshot_acquire(pool:*const Pool,tiers:*const c_void,snapshot:*mut c_void,hashes:*mut c_void,rows:*mut c_void,error:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_generic_sorted_native_owner_snapshot_release(pool:*const Pool,snapshot:*mut c_void,error:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_generic_sorted_native_retire(pool:*const Pool,tiers:*mut c_void,error:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_generic_sorted_native_seed(pool:*const Pool,tiers:*mut c_void,hashes:*mut c_void,rows:*mut c_void,hash:u64,row:u32,error:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_generic_sorted_native_owner_status(reservation:*const c_void,carry:*const c_void,error:*mut c_void,stream:*mut c_void)->i32;
}
struct CarryGraph(*mut c_void);
impl Drop for CarryGraph {fn drop(&mut self){unsafe{mgbfs_generic_sorted_native_carry_destroy(self.0);}}}
struct Lane {origin:OriginBuffers,metadata:Buffer,control:Buffer,stream:*mut c_void}
#[derive(Clone,Debug)]pub(crate) struct OwnerShape {
 pub allocated_bytes:usize,pub lanes:u32,pub banks:u32,regions:u32,classes:u32,
 pool_offsets:[usize;4],pool_bytes:usize,tier_stride:usize,snapshot_stride:usize,
 carry_stride:usize,reservation_stride:usize,control_bytes:usize,handle_stride:usize,
 origin:OriginShape,
}
impl OwnerShape {
 pub fn query(input:&Input,capacity:u32,banks:u32,lanes:u32,device:i32)->Result<Self>{
  if capacity==0||![1,3].contains(&banks)||lanes==0||lanes>input.shards||lanes>8{return Err("SORTED_OWNER_GEOMETRY".into());}
  let sizes=metadata_sizes()?;let origin=OriginShape::query(input,device)?;
  let page=256u64;let needed=(u64::from(capacity)+page-1)/page;
  let classes=64-needed.next_power_of_two().leading_zeros();
  let(mut control,mut handles)=(0,0);check(unsafe{mgbfs_generic_sorted_native_carry_shape(page as u32,classes,&mut control,&mut handles)})?;
  // Rounded roots, transient carry outputs and small per-shard roots. Shared
  // rank credits do not assume balanced ownership of logical shards.
  let target=u64::from(capacity).checked_mul(u64::from(banks)*4).and_then(|v|v.checked_add(u64::from(input.shards)*u64::from(banks)*page*2)).ok_or("SORTED_POOL_OVERFLOW")?;
  let regions=((target+page*64-1)/(page*64)).checked_next_power_of_two().ok_or("SORTED_POOL_OVERFLOW")?;
  let entries=regions*64*page;if entries>u64::from(u32::MAX){return Err("SORTED_POOL_INDEX_RANGE".into());}
  let mut pool_offsets=[0usize;4];let mut pool_bytes=0u64;
  for(i,size)in [regions*8,regions*64*sizes[0] as u64,entries*8,entries*4].into_iter().enumerate(){pool_offsets[i]=usize::try_from(pool_bytes).map_err(|_|"SORTED_POOL_OVERFLOW")?;pool_bytes=pool_bytes.checked_add(align(size)?).ok_or("SORTED_POOL_OVERFLOW")?;}
  let tier_stride=align(sizes[1] as u64)?;let carry_stride=align(sizes[2] as u64)?;let snapshot_stride=sizes[3] as u64;let reservation_stride=align(sizes[4] as u64)?;
  let lane_bytes=origin.allocation_bytes as u64+control+align(carry_stride+reservation_stride+snapshot_stride*u64::from(banks))?;
  let allocated=pool_bytes+(tier_stride+handles)*u64::from(banks)*u64::from(input.shards)+lane_bytes*u64::from(lanes);
  Ok(Self{allocated_bytes:usize::try_from(allocated).map_err(|_|"SORTED_OWNER_BYTES")?,lanes,banks,regions:regions as u32,classes,pool_offsets,pool_bytes:pool_bytes as usize,tier_stride:tier_stride as usize,snapshot_stride:snapshot_stride as usize,carry_stride:carry_stride as usize,reservation_stride:reservation_stride as usize,control_bytes:control as usize,handle_stride:handles as usize,origin})
 }
}
pub(crate) struct SortedOwner {
 // Graphs are destroyed (and streams joined) BEFORE their bound buffers.
 graphs:Vec<CarryGraph>,lanes:Vec<Lane>,pool_storage:Buffer,tiers:Buffer,handles:Buffer,
 pool:Pool,pub shape:OwnerShape,shards:u32,
}
impl SortedOwner {
 pub fn new(input:&Input,shape:OwnerShape,arena:*mut c_void,stride:u32,error:*mut c_void,streams:&[*mut c_void],device:i32)->Result<Self>{
  if streams.len()!=shape.lanes as usize{return Err("SORTED_OWNER_STREAMS".into());}
  check(unsafe{mgbfs_cuda::native_owner::cudaSetDevice(device)})?;
  let pool_storage=Buffer::new(shape.pool_bytes,device)?;let roots=shape.banks as usize*input.shards as usize;
  let tiers=Buffer::new(shape.tier_stride*roots,device)?;let handles=Buffer::new(shape.handle_stride*roots,device)?;
  let pool=Pool{occupied:pool_storage.at(shape.pool_offsets[0]),descriptors:pool_storage.at(shape.pool_offsets[1]),regions:shape.regions,page_entries:256};
  for buffer in [&pool_storage,&tiers]{check(unsafe{mgbfs_cuda::ffi::cudaMemsetAsync(buffer.ptr,0,buffer.bytes,ptr::null_mut())})?;}
  let mut lanes=Vec::new();for &stream in streams{
   let origin=OriginBuffers::new(input,device)?;if origin.allocated_bytes()!=shape.origin.allocation_bytes{return Err("SORTED_OWNER_ORIGIN_SHAPE_CHANGED".into());}
   let metadata=Buffer::new(align((shape.carry_stride+shape.reservation_stride+shape.banks as usize*shape.snapshot_stride) as u64)? as usize,device)?;
   check(unsafe{mgbfs_cuda::ffi::cudaMemsetAsync(metadata.ptr,0,metadata.bytes,ptr::null_mut())})?;
   let control=Buffer::new(shape.control_bytes,device)?;lanes.push(Lane{origin,metadata,control,stream});
  }
  check(unsafe{mgbfs_cuda::ffi::cudaStreamSynchronize(ptr::null_mut())})?;
  let mut graphs=Vec::new();for root in 0..roots{
   let lane=&lanes[(root%input.shards as usize)%lanes.len()];
   let config=CarryConfig{pool:&pool,tiers:tiers.at(root*shape.tier_stride),carry:lane.metadata.ptr,hashes:pool_storage.at(shape.pool_offsets[2]),rows:pool_storage.at(shape.pool_offsets[3]),arena,error,control:lane.control.ptr,handles:handles.at(root*shape.handle_stride),stride,width:input.elements,state_bytes:input.state_bytes,classes:shape.classes};
   let mut graph=ptr::null_mut();check(unsafe{mgbfs_generic_sorted_native_carry_create(&config,lane.stream,&mut graph)})?;graphs.push(CarryGraph(graph));
  }
  let out=Self{graphs,lanes,pool_storage,tiers,handles,pool,shape,shards:input.shards};
  if out.allocated_bytes()!=out.shape.allocated_bytes{return Err("SORTED_OWNER_ALLOCATION_MISMATCH".into());}Ok(out)
 }
 pub fn allocated_bytes(&self)->usize{self.pool_storage.bytes+self.tiers.bytes+self.handles.bytes+self.lanes.iter().map(|l|l.origin.allocated_bytes()+l.metadata.bytes+l.control.bytes).sum::<usize>()}
 pub fn stream(&self,shard:u32)->*mut c_void{self.lanes[shard as usize%self.lanes.len()].stream}
 pub fn seed(&self,shard:u32,hash:u64,error:*mut c_void,stream:*mut c_void)->Result<()>{if shard>=self.shards{return Err("SORTED_ROOT_SHARD".into());}check(unsafe{mgbfs_generic_sorted_native_seed(&self.pool,self.tiers.at(shard as usize*self.shape.tier_stride),self.pool_storage.at(self.shape.pool_offsets[2]),self.pool_storage.at(self.shape.pool_offsets[3]),hash,0,error,stream)})}
 // Caller joins all owner streams before retirement and arena bank reuse.
 pub fn retire_bank(&self,bank:u32,error:*mut c_void,stream:*mut c_void)->Result<()>{
  if bank>=self.shape.banks{return Err("SORTED_EPOCH_BANK".into());}
  for shard in 0..self.shards{check(unsafe{mgbfs_generic_sorted_native_retire(&self.pool,self.tiers.at((bank*self.shards+shard) as usize*self.shape.tier_stride),error,stream)})?;}Ok(())
 }
 pub fn accept(&self,input:&Input,mut destination:Destination,bank:u32)->Result<()>{
  if input.shard>=self.shards||bank>=self.shape.banks{return Err("SORTED_EPOCH_BANK".into());}
  let lane=&self.lanes[input.shard as usize%self.lanes.len()];let stream=lane.stream;
  let snapshot_offset=self.shape.carry_stride+self.shape.reservation_stride;
  destination.snapshots=lane.metadata.at(snapshot_offset);destination.snapshot_count=self.shape.banks;
  destination.pool=(&self.pool as *const Pool).cast_mut().cast();destination.run_hashes=self.pool_storage.at(self.shape.pool_offsets[2]);destination.run_rows=self.pool_storage.at(self.shape.pool_offsets[3]);
  destination.carry=lane.metadata.ptr;destination.reservation=lane.metadata.at(self.shape.carry_stride);
  for b in 0..self.shape.banks{check(unsafe{mgbfs_generic_sorted_native_snapshot_acquire(&self.pool,self.tiers.at((b*self.shards+input.shard) as usize*self.shape.tier_stride),lane.metadata.at(snapshot_offset+b as usize*self.shape.snapshot_stride),destination.run_hashes,destination.run_rows,destination.error,stream)})?;}
  lane.origin.generate_unique(input,destination.error,stream)?;lane.origin.accept(input,&destination,stream)?;
  for b in 0..self.shape.banks{check(unsafe{mgbfs_generic_sorted_native_owner_snapshot_release(&self.pool,lane.metadata.at(snapshot_offset+b as usize*self.shape.snapshot_stride),destination.error,stream)})?;}
  // Releasing snapshot leases before carry returns old epoch credits promptly.
  let graph=&self.graphs[(bank*self.shards+input.shard) as usize];
  check(unsafe{mgbfs_generic_sorted_native_carry_launch(graph.0,stream)})?;
  // The graph retries once on actual allocation pressure. Normal publication
  // needs one host launch; persistent pressure remains a collective failure.
  check(unsafe{mgbfs_generic_sorted_native_owner_status(destination.reservation,destination.carry,destination.error,stream)})
 }
}
