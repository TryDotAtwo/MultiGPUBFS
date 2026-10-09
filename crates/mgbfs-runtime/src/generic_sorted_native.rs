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
