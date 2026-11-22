// Private ABI for the experimental GPU-resident packed-bitmap path. Loading is
// dynamic so older published native bundles continue to run the host path.
use super::{BitmapDeviceDomain, BitmapPack4Dictionary};
use std::{ffi::{c_char,c_void},mem};

#[repr(C)]
#[derive(Clone,Copy,Debug)]
pub(super) struct Extent {
 pub rows:*mut u64, pub counts:*mut u64, pub capacity:u64,
 pub banks:u32, pub reserved:u32,
}
#[repr(C)]
#[derive(Clone,Copy,Debug)]
pub(super) struct Mailbox {
 pub abi_version:u32,pub struct_bytes:u32,pub world:u32,pub lanes:u32,
 pub slots_per_ring:u32,pub requested_payload_bytes:u32,pub payload_stride_bytes:u32,pub reserved:u32,
 pub ring_count:u64,pub slots_offset:u64,pub payload_offset:u64,pub close_offset:u64,pub total_bytes:u64,
}
#[repr(C)]
#[derive(Clone,Copy)]
pub(super) struct Args {
 pub abi_version:u32,pub struct_bytes:u32,pub world:u32,pub rank:u32,
 pub owner:u32,pub shards:u32,pub generators:u32,pub banks:u32,
 pub owner_page:u32,pub reserved0:u32,
 pub owner_boundary:u64,pub owner_len:u64,pub job_generation:u64,pub rank_limit:u64,
 pub domain:BitmapDeviceDomain,pub dictionary:BitmapPack4Dictionary,pub mailbox:Mailbox,
 pub permutations:*const u32,pub physical_of_logical:*const u32,
 pub extents:*mut Extent,pub extent_owner:*mut u32,pub extent_count:u64,
 pub shard_locks:*mut u32,pub visited:*mut u64,pub rank_count:*mut u64,
 pub layer_sizes:*mut u64,pub layer_capacity:u64,pub rank_counts:*mut u64,
 pub capture:*mut c_void,pub capture_capacity:u64,pub capture_count:*mut u64,
 pub terminal:*mut u32,
}
#[repr(C)]
#[derive(Clone,Copy,Debug,Default)]
pub(super) struct Report {
 pub depth:u64,pub sent:u64,pub received:u64,pub duplicates:u64,
 pub duplicates_full:u64,pub blocked:u64,pub capture_count:u64,pub global_next:u64,
 pub status:u32,pub rank:u32,pub reason:u32,pub max_active_shards_first64:u32,
}
#[repr(C)]
#[derive(Clone,Copy,Debug,Default)]
pub(super) struct Topology {
 pub abi_version:u32,pub struct_size:u32,pub status:u32,pub stage:u32,
 pub api_code:i32,pub src_rank:i32,pub dst_rank:i32,pub src_device:i32,pub dst_device:i32,
 pub cuda_runtime_version:i32,pub cuda_driver_version:i32,pub nccl_version:i32,
 pub lsa_compiled:u32,pub rank_count:u32,
}

type Query=unsafe extern "C" fn(u32)->u32;
type Smoke=unsafe extern "C" fn()->i32;
type Layout=unsafe extern "C" fn(u32,u32,u32,u32,u64,usize,*mut Mailbox)->i32;
type TopologyQuery=unsafe extern "C" fn(*const i32,u32,u32,*mut Topology)->i32;
type Prepare=unsafe extern "C" fn(*mut c_void,u64,*mut c_char,usize)->i32;
type Activate=unsafe extern "C" fn(*mut c_void,*mut c_char,usize)->i32;
type Initialize=unsafe extern "C" fn(*mut c_void,*const Args,*mut c_void)->i32;
type Launch=unsafe extern "C" fn(*mut c_void,*const Args,*mut Report,*mut c_void)->i32;
type Retire=unsafe extern "C" fn(*mut c_void)->i32;
type WindowInfo=unsafe extern "C" fn(*mut c_void,*mut *mut c_void,*mut u64)->i32;
type PeerSmoke=unsafe extern "C" fn(*mut c_void,*const Args,*mut u32,*mut c_void)->i32;

#[derive(Clone,Copy)]
pub(super) struct Functions {
 pub query:Query,pub smoke:Smoke,pub layout:Layout,pub topology:TopologyQuery,
 pub prepare:Prepare,pub activate:Activate,pub initialize:Initialize,pub launch:Launch,pub retire:Retire,pub window_info:WindowInfo,pub peer_smoke:PeerSmoke,
 pub control_bytes:u64,
}
impl Functions {
 pub(super) unsafe fn load()->Result<Self,String> {
  unsafe fn symbol(name:&[u8])->Result<*mut c_void,String>{
   let p=libc::dlsym(libc::RTLD_DEFAULT,name.as_ptr().cast());
   if p.is_null(){Err(format!("RESIDENT_ABI_MISSING {}",String::from_utf8_lossy(&name[..name.len()-1])))}else{Ok(p)}
  }
  let query:Query=mem::transmute(symbol(b"mgbfs_device_pack4_bfs_abi_query\0")?);
  let expected=[1,mem::size_of::<Args>() as u32,mem::size_of::<Report>() as u32,mem::size_of::<Extent>() as u32];
  for (i,&size) in expected.iter().enumerate(){
   let found=query(i as u32);
   if found!=size{return Err(format!("RESIDENT_ABI_MISMATCH field={i} expected={size} actual={found}"))}
  }
  let control=query(4);
  if control==0||control%64!=0||query(5)!=16||query(999)!=0{return Err("RESIDENT_ABI_CONTROL_MISMATCH".into())}
  Ok(Self {
   query,smoke:mem::transmute(symbol(b"mgbfs_device_pack4_bfs_smoke_v1\0")?),
   layout:mem::transmute(symbol(b"mgbfs_device_mailbox_layout_v1\0")?),
   topology:mem::transmute(symbol(b"mgbfs_device_topology_v1\0")?),
   prepare:mem::transmute(symbol(b"mgbfs_nccl_resident_prepare_v1\0")?),
   activate:mem::transmute(symbol(b"mgbfs_nccl_resident_activate_v1\0")?),
   initialize:mem::transmute(symbol(b"mgbfs_nccl_resident_init_v1\0")?),
   launch:mem::transmute(symbol(b"mgbfs_nccl_resident_launch_v1\0")?),
   retire:mem::transmute(symbol(b"mgbfs_nccl_resident_retire_v1\0")?),
   window_info:mem::transmute(symbol(b"mgbfs_nccl_resident_window_info_v1\0")?),
   peer_smoke:mem::transmute(symbol(b"mgbfs_nccl_resident_smoke_v1\0")?),
   control_bytes:u64::from(control),
  })
 }
 pub(super) fn preflight(&self,devices:&[u32],rank:u32)->Result<Topology,String>{
  if devices.is_empty()||devices.len()>8||rank as usize>=devices.len(){return Err("RESIDENT_DEVICE_GEOMETRY".into())}
  let ordinals=devices.iter().map(|&v|i32::try_from(v).map_err(|_|"RESIDENT_DEVICE_ORDINAL".to_owned())).collect::<Result<Vec<_>,_>>()?;
  let mut topology=Topology::default();
  let call=unsafe{(self.topology)(ordinals.as_ptr(),ordinals.len() as u32,mem::size_of::<Topology>() as u32,&mut topology)};
  if call!=0||topology.abi_version!=1||topology.struct_size as usize!=mem::size_of::<Topology>()||topology.rank_count as usize!=devices.len(){
   return Err(format!("RESIDENT_TOPOLOGY_QUERY code={call} abi={} size={} world={}",topology.abi_version,topology.struct_size,topology.rank_count))
  }
  if topology.status!=0{return Err(format!("RESIDENT_TOPOLOGY_UNSUPPORTED status={} stage={} api_code={} source_rank={} target_rank={} source_device={} target_device={}",topology.status,topology.stage,topology.api_code,topology.src_rank,topology.dst_rank,topology.src_device,topology.dst_device))}
  let selected=ordinals[rank as usize];
  let code=unsafe{mgbfs_cuda::native_owner::cudaSetDevice(selected)};
  if code!=0{return Err(format!("RESIDENT_CUDA_SET_DEVICE rank={rank} device={selected} code={code}"))}
  let code=unsafe{(self.smoke)()};
  if code!=0{return Err(format!("RESIDENT_CUDA_KERNEL_UNSUPPORTED rank={rank} device={selected} code={code}"))}
  extern "C" {fn cudaDeviceEnablePeerAccess(device:i32,flags:u32)->i32;}
  for (peer,&device) in ordinals.iter().enumerate(){
   if peer==rank as usize{continue}
   let code=unsafe{cudaDeviceEnablePeerAccess(device,0)};
   if code==704 {
    let last=unsafe{super::cudaGetLastError()};
    if last!=0&&last!=704{return Err(format!("RESIDENT_PEER_LAST_ERROR rank={rank} peer={peer} code={last}"))}
   }else if code!=0{return Err(format!("RESIDENT_PEER_ENABLE rank={rank} peer={peer} code={code}"))}
  }
  Ok(topology)
 }
 pub(super) fn mailbox(&self,world:u32,slots:u32,budget:u64)->Result<(Mailbox,u64),String>{
  let mut m=Mailbox{abi_version:0,struct_bytes:0,world:0,lanes:0,slots_per_ring:0,requested_payload_bytes:0,payload_stride_bytes:0,reserved:0,ring_count:0,slots_offset:0,payload_offset:0,close_offset:0,total_bytes:0};
  let code=unsafe{(self.layout)(world,4,slots,16,budget,mem::size_of::<Mailbox>(),&mut m)};
  if code!=0{return Err(format!("RESIDENT_MAILBOX_ADMISSION code={code}"))}
  let bytes=self.control_bytes.checked_add(m.total_bytes).ok_or("RESIDENT_MAILBOX_OVERFLOW")?;
  if bytes>budget{return Err("RESIDENT_MAILBOX_BUDGET".into())}
  Ok((m,bytes))
 }
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 #[ignore = "requires the experimental resident CUDA library and a GPU"]
 fn resident_preflight_executes_compiled_kernel() {
  let f=unsafe{Functions::load()}.expect("resident ABI in matching test bundle");
  let result=f.preflight(&[0],0).expect("one GPU must support the resident kernel");
  assert_eq!((result.status,result.rank_count),(0,1));
  assert!(f.preflight(&[],0).is_err());
  assert!(f.preflight(&[0,0],0).is_err());
 }
 #[test]
 #[ignore = "requires the experimental resident CUDA library"]
 fn resident_ffi_matches_loaded_library() {
  let f=unsafe{Functions::load()}.expect("resident ABI in matching test bundle");
  assert_eq!(unsafe{(f.query)(0)},1);
  assert_eq!(mem::size_of::<Topology>(),56);
  let (layout,bytes)=f.mailbox(2,2,1<<20).unwrap();
  assert_eq!((layout.world,layout.lanes,layout.slots_per_ring),(2,4,2));
  assert!(bytes>layout.total_bytes);
  assert!(f.mailbox(2,2,1).is_err());
 }
}
