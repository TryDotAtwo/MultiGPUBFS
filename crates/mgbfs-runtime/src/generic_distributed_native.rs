//! General exact distributed engine: preallocated queues and independent owner streams.
//! Fixed-capacity transport is a correctness fallback, not an optimal bandwidth claim.
use std::{cell::RefCell,ffi::c_void,ptr,rc::Rc};
use crate::generic_sorted_native::{Input as SortedInput,Destination as SortedDestination,OwnerShape,SortedOwner};
use crate::generic_route::RoundVote;
use crate::segmented_packet_transport::PacketEndpoint;
use mgbfs_core::{graph_definition::{GraphDefinitionV2,GraphAction},Result};
use mgbfs_cuda::{ffi::*,generic_graph::*};
extern "C" {fn cudaDeviceSynchronize()->i32;}
mod compact_path;

pub fn compact_path_available() -> bool { compact_path::abi_ready() }
pub(crate) fn compact_path_scratch_bound(world:u32,shards:u32,generators:u32)->Result<u64>{
 Ok(compact_path::route_scratch_bound(world,shards,generators)? as u64)
}
use crate::{generic_native::{GenericNativeBfs,Buffer,Stream,check},generic_memory::GenericMemoryPlan,generic_distributed_memory::GenericDistributedMemoryPlan};
extern "C" {
 fn mgbfs_generic_regenerate_all_routes_count_storage(
  state_bytes:u32,kind:u32,elements:u32,rows:u32,cols:u32,generators:u32,
  parents:*const c_void,count:u32,stride:u32,permutations:*const u32,matrices:*const i64,
  moduli:*const u32,source:u32,begin:u64,records:*const GenericRouteRecord,
  counts:*const u32,capacity:u32,queues:u32,states:*mut c_void,error:*mut u32,stream:*mut c_void,
 )->i32;
 fn mgbfs_nccl_exchange_bounded_triplets(comm:*mut c_void,rank:u32,world:u32,shards:u32,capacity:u32,used:u32,payload:u64,state_bytes:u32,counts:*const c_void,records:*const c_void,states:*const c_void,recv_counts:*mut c_void,recv_records:*mut c_void,recv_states:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_generic_resolve_incoming_edges_v3(
  state_bytes:u32,elements:u32,local:*const c_void,remote:*const c_void,
  local_meta:*const GenericRouteRecord,remote_meta:*const GenericRouteRecord,
  local_counts:*const u32,remote_counts:*const u32,rank:u32,world:u32,shard:u32,shards:u32,queue_capacity:u32,
  slots:*const u64,slot_count:u64,arena:*const c_void,stride:u32,first_child:u32,child_limit:u32,
  children:*mut WindowEdge,output_capacity:u32,begin:u32,length:u32,error:*mut u32,stream:*mut c_void,
 )->i32;
 fn mgbfs_nccl_exchange_exact_payload(
  comm:*mut c_void,rank:u32,world:u32,shards:u32,capacity:u32,payload:u64,state_bytes:u32,
  send_counts:*const u32,recv_counts:*const u32,records:*const c_void,states:*const c_void,
  recv_records:*mut c_void,recv_states:*mut c_void,stream:*mut c_void,
 )->i32;
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
type FusedPathRoute=unsafe extern "C" fn(
 u32,u32,u32,u32,u32,u32,*const c_void,u32,u32,*const u32,*const i64,*const u32,u64,
 u32,u32,u32,u32,u32,u64,*const u32,*const u64,*mut GenericRouteRecord,*mut u32,*mut u32,
 *mut c_void,*mut c_void,
)->i32;

fn fused_path_route()->Option<FusedPathRoute>{
 let symbol=unsafe{libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_generic_route_with_states_v1\0".as_ptr().cast())};
 (!symbol.is_null()).then(||unsafe{std::mem::transmute::<*mut c_void,FusedPathRoute>(symbol)})
}

type RouteVoteKernel =
    unsafe extern "C" fn(*const u32, *const u32, *mut u32, u32, *mut c_void) -> i32;

fn route_vote_kernel() -> Option<RouteVoteKernel> {
    let symbol = unsafe {
        libc::dlsym(libc::RTLD_DEFAULT, b"mgbfs_generic_route_retry_vote_v3\0".as_ptr().cast())
    };
    (!symbol.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, RouteVoteKernel>(symbol) })
}

fn round_vote(
    kernel: Option<RouteVoteKernel>,
    control: &Buffer,
    owner: *const u32,
    comm: *mut c_void,
    stop: u32,
    stream: *mut c_void,
) -> Result<RoundVote> {
    crate::generic_native::checked!(generic_route_retry_vote, unsafe {
        kernel.unwrap_or(mgbfs_generic_route_retry_vote_v2)(
            control.at(8), owner, control.at(12), stop, stream,
        )
    })?;
    crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32, unsafe {
        mgbfs_nccl_all_reduce_max_u32(comm, control.at(12), control.at(0), stream)
    })?;
    let mut vote = 0u32;
    crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
        cudaMemcpyAsync((&mut vote as *mut u32).cast(), control.ptr, 4, 2, stream)
    })?;
    crate::generic_native::checked!(cudaStreamSynchronize, unsafe { cudaStreamSynchronize(stream) })?;
    if kernel.is_some() || vote <= 1 {
        return RoundVote::current(vote);
    }

    // All ranks saw the same ambiguous legacy vote. Preserve the usual
    // one-collective round and reconstruct the separate errors here.
    let zero = control.at::<u32>(24);
    crate::generic_native::checked!(cudaMemsetAsync, unsafe {
        cudaMemsetAsync(zero.cast(), 0, 4, stream)
    })?;
    crate::generic_native::checked!(mgbfs_generic_route_retry_vote_v2, unsafe {
        mgbfs_generic_route_retry_vote_v2(control.at(8), zero, control.at(12), 0, stream)
    })?;
    crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32, unsafe {
        mgbfs_nccl_all_reduce_max_u32(comm, control.at(12), control.at(0), stream)
    })?;
    crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32, unsafe {
        mgbfs_nccl_all_reduce_max_u32(comm, owner, control.at(4), stream)
    })?;
    let mut errors = [0u32; 2];
    crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
        cudaMemcpyAsync(errors.as_mut_ptr().cast(), control.ptr, std::mem::size_of_val(&errors), 2, stream)
    })?;
    crate::generic_native::checked!(cudaStreamSynchronize, unsafe { cudaStreamSynchronize(stream) })?;
    Ok(RoundVote::legacy_error(errors[0], errors[1], vote & PATH_STOP_VOTE != 0))
}

struct SortCache{keys:Buffer,sorted_keys:Buffer,origins:Buffer,sorted_origins:Buffer,scratch:Buffer,scratch_stride:usize}
impl SortCache{fn new(p:&GenericDistributedMemoryPlan,device:i32)->Result<Self>{let n=p.world as usize*p.queue_capacity as usize;let size=n*p.shards as usize;let scratch_stride=n*64+65536;Ok(Self{keys:Buffer::new(size*8,device)?,sorted_keys:Buffer::new(size*8,device)?,origins:Buffer::new(size*4,device)?,sorted_origins:Buffer::new(size*4,device)?,scratch:Buffer::new(scratch_stride*p.shards as usize,device)?,scratch_stride})}}
struct ParentCache{sources:[Buffer;2],received:Buffer,cursors:Buffer,frontiers:Buffer}
impl ParentCache{fn new(p:&GenericDistributedMemoryPlan,device:i32,counts:&[u32])->Result<Self>{
 let bytes=p.batch as usize*p.elements as usize*p.state_bytes as usize;
 let sources=[Buffer::new(bytes,device)?,Buffer::new(bytes,device)?];for v in &sources{crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(v.ptr,0,v.bytes,ptr::null_mut())})?;}
 crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(ptr::null_mut())})?;let received=Buffer::new(bytes*p.world as usize,device)?;let cursors=Buffer::new(p.world as usize*8,device)?;cursors.upload(&vec![0u64;p.world as usize])?;
 let frontiers=Buffer::new(p.world as usize*4,device)?;frontiers.upload(counts)?;Ok(Self{sources,received,cursors,frontiers})
}}
struct Event{ptr:*mut c_void,device:i32}
impl Event{fn new(device:i32)->Result<Self>{let mut p=crate::session_cache::event_take(device).unwrap_or(ptr::null_mut());if p.is_null(){crate::generic_native::checked!(cudaEventCreateWithFlags, unsafe{cudaEventCreateWithFlags(&mut p,2)})?;}Ok(Self{ptr:p,device})}}
impl Drop for Event{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.device);if !crate::session_cache::event_put(self.ptr,self.device){cudaEventDestroy(self.ptr);}}}}
struct QueueBank{records:Buffer,counts:Buffer,states:Buffer}
impl QueueBank {
 fn new(plan: &GenericDistributedMemoryPlan, device: i32) -> Result<Self> {
  let [records, counts, states] = plan.dense_queue_bank_extents();
  Ok(Self { records: Buffer::new(records, device)?, counts: Buffer::new(counts, device)?,
   states: Buffer::new(states, device)? })
 }
}

/// Scratch belongs to the active direction. A completed layer has drained
/// the source stream and all owner events before the opposite side can run.
struct QueueWorkspace {
 banks: [QueueBank; 2],
 inbox: Option<QueueBank>,
}
impl QueueWorkspace {
 fn new(plan: &GenericDistributedMemoryPlan, device: i32) -> Result<Self> {
  Ok(Self {
   banks: [QueueBank::new(plan, device)?, QueueBank::new(plan, device)?],
   inbox: if plan.world > 1 { Some(QueueBank::new(plan, device)?) } else { None },
  })
 }
}
#[derive(Debug,PartialEq,Eq)]
pub enum DistributedAdvance{Layer{local:u64,global:u64},Complete,Resource{fatal:u32}}
struct PathCommunicator {
 ptr: *mut c_void,
 aborted: std::cell::Cell<bool>,
}
impl Drop for PathCommunicator {
 fn drop(&mut self) {
  if self.aborted.get() || !crate::session_cache::comm_put(self.ptr) {
   unsafe { mgbfs_nccl_destroy(self.ptr); }
  }
 }
}
impl PathCommunicator {
 fn abort(&self, rank: u32) {
  if self.aborted.replace(true) { return; }
  crate::session_cache::permit_comm(false);
  let status=unsafe { mgbfs_nccl_abort(self.ptr) };
  if status!=0{eprintln!("MGBFS_FAILED_COMM_ABORT rank={rank} code={status}");}
 }
}
pub struct GenericDistributedBfs{
 compact_path:Option<Rc<RefCell<compact_path::Workspace>>>,
 online_size_tuning:bool,online_bucket:Option<u32>,online_choices:std::collections::BTreeMap<u32,(u32,bool)>,profile_events:Vec<serde_json::Value>,active_batch:u32,use_gemm:bool,gemm:Option<crate::generic_gemm::Context>,bfs:GenericNativeBfs,plan:GenericDistributedMemoryPlan,rank:u32,comm:Rc<PathCommunicator>,
 sorted_owner:Option<SortedOwner>,sort_cache:Option<SortCache>,parent_cache:Option<ParentCache>,queues:Option<Rc<QueueWorkspace>>,control:Buffer,owner_map:Buffer,owner_cuts:Option<Buffer>,
 host_owner_map:Vec<u32>,host_owner_cuts:Option<Vec<u64>>,
 streams:Vec<Stream>,ready:Event,done:Vec<Event>,depth:u64,parent_cursor:u64,max_frontier:u32,counts:Vec<u32>,source_retries:u64,positions:Option<Buffer>,rolling_counts:[u32;3],retired_rows:u64,
}
fn mix(mut x:u64)->u64{x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn state_hash(state: &[i64], seed: u64, bits: u32) -> u64 {
 let hash=state.iter().enumerate().fold(seed,|h,(e,&v)|mix(h^v as u64^e as u64));
 match bits {64=>mix(hash),bits=>mix(hash)&((1u64<<bits)-1)}
}
fn owner_partition(hash:u64,world:u32,cuts:Option<&[u64]>)->usize{
 let key=hash>>32;
 match cuts{
  Some(c)=>c.partition_point(|&cut|cut<=key)-1,
  None=>((key*u64::from(world))>>32) as usize,
 }
}
impl GenericDistributedBfs{
 pub(crate) fn owns_path_target(&self,state:&[i64])->bool{
  let hash=state_hash(state,self.bfs.seed,self.bfs.hash_bits);
  self.host_owner_map[owner_partition(hash,self.plan.world,self.host_owner_cuts.as_deref())]==self.rank
 }
 /// Raw-buffer extents for exact idle-storage reuse across graph geometry.
 /// SORTED_RUNS owns additional queried native shapes and keeps its existing
 /// exact-layout admission. HASH/parent/radix paths have explicit extents.
 pub fn buffer_requirements(g:&GraphDefinitionV2,p:&GenericDistributedMemoryPlan,cuts:Option<&[u64]>)->Result<Vec<usize>> {
  Self::buffer_requirements_with_queues(g,p,cuts,true)
 }
 pub(crate) fn reverse_buffer_requirements(
  g: &GraphDefinitionV2, p: &GenericDistributedMemoryPlan, cuts: Option<&[u64]>,
 ) -> Result<Vec<usize>> {
  Self::buffer_requirements_with_queues(g,p,cuts,false)
 }
 fn buffer_requirements_with_queues(
  g:&GraphDefinitionV2,p:&GenericDistributedMemoryPlan,cuts:Option<&[u64]>,needs_queues:bool,
 )->Result<Vec<usize>> {
  if p.history_algorithm!="HASH" {return Err("GENERIC_REUSE_REQUIRES_HASH_LAYOUT".into());}
  g.validate()?;p.validate(g.generator_count() as u32)?;if p.elements as usize!=g.start.len(){return Err("GENERIC_REUSE_GRAPH_WIDTH".into());}
  // validate recomputes checked u64 byte extents and the rolling u32 stride.
  // CUDA resident workers are 64-bit; each raw extent is a term of that plan.
  let n=p.elements as usize;let cap=p.capacity as usize;let w=p.world as usize;let shards=p.shards as usize;let b=p.batch as usize;let bytes=p.state_bytes as usize;let rolling=p.history_layers==3;
  let stride=cap*if rolling{3}else{1};let readout=if rolling{cap.min(1000)}else if crate::generic_memory::nearest_direct_rows(){1}else{cap};
  let (perms,matrices,moduli)=match &g.action {
   GraphAction::Permutation{generators,..}=>(generators.iter().map(|v|v.len()).sum::<usize>()*4,0,0),
   GraphAction::Matrix{generators,..}=>(0,generators.iter().map(|v|v.matrix.len()).sum::<usize>()*8,generators.len()*4),
  };
  let mut sizes=vec![n*stride*bytes,n*readout*bytes,cap*4,cap*4,p.table_slots as usize*8,24,perms,matrices,moduli,w*4,64+w*8];
  if cuts.is_some(){sizes.push((w+1)*8);}
  if rolling{sizes.push(stride*8);}
  // The two producer banks exist on every rank. Only a distributed rank
  // allocates an inbox, so a single-GPU job must not reserve its extents.
  if needs_queues{
   if p.compact_transport{
    sizes.extend(compact_path::Geometry::new(p,g.generator_count() as u32)?.buffer_requirements(p.shards)?);
   }else{
    for _ in 0..p.dense_queue_bank_count(){sizes.extend(p.dense_queue_bank_extents());}
   }
  }
  if p.sort_candidates{let count=w*p.queue_capacity as usize;let size=count*shards;sizes.extend([size*8,size*8,size*4,size*4,(count*64+65536)*shards]);}
  if p.parent_transport{let size=b*n*bytes;sizes.extend([size,size,size*w,w*8,w*4]);}
  for value in &mut sizes{*value=(*value).max(1);}
  Ok(sizes)
 }
 /// All ranks must agree on graph, seed, hash bits, batch/queue geometry and
 /// rank map before entering this constructor. The launcher owns that gate.
 pub fn new(graph:&GraphDefinitionV2,device:u32,rank:u32,plan:GenericDistributedMemoryPlan,id:&[u8;128],seed:u64,hash_bits:u32)->Result<Self>{
  Self::new_with_cuts(graph,device,rank,plan,id,seed,hash_bits,None)
 }
 pub fn new_with_cuts(graph:&GraphDefinitionV2,device:u32,rank:u32,plan:GenericDistributedMemoryPlan,id:&[u8;128],seed:u64,hash_bits:u32,cuts:Option<&[u64]>)->Result<Self>{
  Self::new_with_roots(graph,device,rank,plan,id,seed,hash_bits,cuts,None,None,None)
 }
 pub(crate) fn new_with_packet_endpoint(
  graph:&GraphDefinitionV2,device:u32,rank:u32,plan:GenericDistributedMemoryPlan,id:&[u8;128],
  seed:u64,hash_bits:u32,cuts:Option<&[u64]>,endpoint:PacketEndpoint,
 )->Result<Self>{
  Self::new_with_roots(graph,device,rank,plan,id,seed,hash_bits,cuts,None,None,Some(endpoint))
 }
 /// Both search directions retain the routing of graph.start. Reverse roots
 /// select the initial frontier without changing its owner partition.
 fn new_with_roots(
  graph: &GraphDefinitionV2, device: u32, rank: u32, plan: GenericDistributedMemoryPlan,
  id: &[u8; 128], seed: u64, hash_bits: u32, cuts: Option<&[u64]>, roots: Option<&[Vec<i64>]>, shared: Option<&Self>,
  endpoint: Option<PacketEndpoint>,
 ) -> Result<Self> {
  if let Some(c)=cuts{if c.len()!=plan.world as usize+1||c[0]!=0||c[c.len()-1]!=(1u64<<32)||c.windows(2).any(|v|v[0]>=v[1]){return Err("GENERIC_OWNER_CUTS".into());}}
  plan.validate(graph.generator_count() as u32)?;if rank>=plan.world||hash_bits>64{return Err("GENERIC_DISTRIBUTED_RANK".into());}
  let shared_packet=shared.and_then(|engine|engine.compact_path.clone());
  if plan.compact_transport!=(endpoint.is_some()||shared_packet.is_some()){
   return Err("NATIVE_PATH_COMPACT_ENDPOINT".into());
  }
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
   let(mut free,mut total)=(0,0);crate::generic_native::checked!(cudaMemGetInfo, unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
   if plan.device_bytes>(free as u64+crate::session_cache::reusable_bytes()+crate::generic_pool::available()?){return Err("SORTED_PLAN_FREE_MEMORY_CHANGED".into());}
   Some(shape)
  }else{None};
  let local=GenericMemoryPlan::with_storage(plan.elements,plan.capacity,plan.state_bytes)?;
  let rolling=plan.history_layers==3;if rolling&&!graph.inverse_closed()?{return Err("GENERIC_ROLLING_REQUIRES_INVERSE_CLOSED".into());}
  let mut bfs=if use_sorted{GenericNativeBfs::new_unseeded_sorted(graph,device,local,seed,hash_bits,rolling)?}else if rolling{GenericNativeBfs::new_unseeded_rolling(graph,device,local,seed,hash_bits)?}else{GenericNativeBfs::new_unseeded(graph,device,local,seed,hash_bits)?};let d=bfs.device;
  if bfs.slots.bytes!=plan.table_slots as usize*8{return Err("GENERIC_SHARED_TABLE_ALLOCATION_SHAPE".into());}
  if !use_sorted{crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(bfs.slots.ptr,255,bfs.slots.bytes,bfs.stream.ptr)})?;}
  let hash_state=|state:&[i64]|state_hash(state,seed,hash_bits);
  let logical_owner=|hash|owner_partition(hash,plan.world,cuts);
  let hash = hash_state(&graph.start);
  let owner = logical_owner(hash);
  let mut map = (0..plan.world).collect::<Vec<_>>();
  if cuts.is_none() { map.swap(0, owner); }
  let root_rank = map[owner];
  let (counts, local_roots) = if let Some(roots) = roots {
   if rolling || use_sorted {
    return Err("GENERIC_MULTI_ROOT_REQUIRES_APPEND_ONLY_HASH".into());
   }
   if plan.state_bytes == 1 && roots.iter().flatten().any(|&v| u8::try_from(v).is_err()) {
    return Err("GENERIC_MULTI_ROOT_STATE_ENCODING".into());
   }
   crate::path_meeting::partition_roots(roots, plan.elements as usize, plan.world, rank, |state| {
    map[logical_owner(hash_state(state))]
   })?
  } else {
   ((0..plan.world).map(|r| u32::from(r == root_rank)).collect(), Vec::new())
  };
  if counts[rank as usize] > plan.capacity {
   return Err("GENERIC_MULTI_ROOT_CAPACITY".into());
  }
  let owner_cuts=if let Some(c)=cuts{let v=Buffer::new(c.len()*8,d)?;v.upload(c)?;Some(v)}else{None};
  let owner_map=Buffer::new(map.len()*4,d)?;owner_map.upload(&map)?;
  bfs.count=counts[rank as usize];bfs.visited_used=bfs.count;bfs.control.upload(&[bfs.count,0u32,0,0,0,0])?;
  let positions=if rolling&&!use_sorted{Some(Buffer::new(bfs.arena_stride as usize*8,d)?)}else{None};
  if rank==root_rank&&rolling&&!use_sorted{crate::generic_native::checked!(mgbfs_generic_reseed_rows_storage, unsafe{mgbfs_generic_reseed_rows_storage(plan.state_bytes,plan.elements,bfs.visited.ptr.cast(),bfs.arena_stride,0,1,bfs.slots.ptr.cast(),plan.table_slots,positions.as_ref().unwrap().ptr.cast(),seed,hash_bits,bfs.control.at(8),bfs.stream.ptr)})?;}
  if !local_roots.is_empty() {
   let indices = (0..bfs.count).collect::<Vec<_>>();
   bfs.front.upload(&indices)?;
   let mut column = Vec::with_capacity(local_roots.len());
   let mut narrow_column = Vec::new();
   for coordinate in 0..plan.elements as usize {
    column.clear();
    column.extend(local_roots.iter().map(|state| state[coordinate]));
    let offset = coordinate * bfs.arena_stride as usize * plan.state_bytes as usize;
    let (source, bytes) = if plan.state_bytes == 1 {
     narrow_column.clear();
     narrow_column.extend(column.iter().map(|&value| value as u8));
     (narrow_column.as_ptr().cast(), narrow_column.len())
    } else {
     (column.as_ptr().cast(), std::mem::size_of_val(column.as_slice()))
    };
    crate::generic_native::checked!(cudaMemcpy, unsafe{
     cudaMemcpy(bfs.visited.at::<c_void>(offset), source, bytes, 1)
    })?;
   }
  }
  if bfs.count != 0 && !rolling && !use_sorted {
   crate::generic_native::checked!(mgbfs_generic_seed_shared_storage, unsafe{
    mgbfs_generic_seed_shared_storage(plan.state_bytes,plan.elements,bfs.visited.ptr.cast(),
     plan.capacity,bfs.count,bfs.slots.ptr.cast(),plan.table_slots,seed,hash_bits,bfs.control.at(8),bfs.stream.ptr)
   })?;
  }
  crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(bfs.stream.ptr)})?;
  let queues=match shared {
   Some(forward)=>forward.queues.clone(),
   None if endpoint.is_some()=>None,
   None=>Some(Rc::new(QueueWorkspace::new(&plan,d)?)),
  };
  let compact_path=match (shared_packet,endpoint) {
   (Some(packet),None)=>Some(packet),
   (None,Some(endpoint))=>Some(Rc::new(RefCell::new(
    compact_path::Workspace::new(&plan,bfs.generators,rank,id,endpoint,d)?
   ))),
   (None,None)=>None,
   (Some(_),Some(_))=>return Err("NATIVE_PATH_COMPACT_DUPLICATE_ENDPOINT".into()),
  };
  let control=Buffer::new(64+plan.world as usize*8,d)?;
  // Raw paths share one exact table and consume runs in order on bfs.stream.
  let lanes=if compact_path.is_some(){0}else if use_sorted{plan.owner_lanes}else{plan.shards};
  let mut streams=vec![];let mut done=vec![];for _ in 0..lanes{streams.push(Stream::new(d)?);done.push(Event::new(d)?);}
  let sorted_owner=if use_sorted{
   let input=SortedInput{elements:plan.elements,world:plan.world,rank,shards:plan.shards,queue_capacity:plan.queue_capacity,state_bytes:plan.state_bytes,transport:if plan.packed_candidates(){1}else if plan.parent_transport{2}else{0},kind:bfs.kind,rows:bfs.rows,cols:bfs.cols,generators:bfs.generators,parent_stride:plan.batch,hash_bits,seed,..Default::default()};
   let shape=admitted_sorted_shape.ok_or("SORTED_OWNER_ADMITTED_SHAPE_MISSING")?;
   let(mut free,mut total)=(0,0);crate::generic_native::checked!(cudaMemGetInfo, unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
   // Revalidate the serialized native shape; driver graph reserve is separate
   // from application buffers and is not advertised as byte-exact driver use.
   if shape.allocated_bytes as u64!=plan.sorted_owner_bytes{return Err("SORTED_OWNER_SERIALIZED_SHAPE_MISMATCH".into());}
   let graph_reserve=usize::try_from(plan.driver_graph_reserve_bytes).map_err(|_|"SORTED_OWNER_ADMISSION_OVERFLOW")?;
   if shape.allocated_bytes.checked_add(graph_reserve).ok_or("SORTED_OWNER_ADMISSION_OVERFLOW")? as u64>free as u64+crate::session_cache::reusable_bytes()+crate::generic_pool::available()?{return Err("SORTED_OWNER_ADMISSION_NO_CAPACITY".into());}
   crate::generic_pool::external_headroom(plan.driver_graph_reserve_bytes)?;
   let pointers=streams.iter().map(|v|v.ptr).collect::<Vec<_>>();
   let owner=SortedOwner::new(&input,shape,bfs.visited.ptr,bfs.arena_stride,bfs.control.at(8),&pointers,d)?;
   if rank==root_rank{let shard=((u64::from(hash as u32)*u64::from(plan.shards))>>32) as u32;owner.seed(shard,hash,bfs.control.at(8),bfs.stream.ptr)?;}
   crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(bfs.stream.ptr)})?;Some(owner)
  }else{None};
  let ready=Event::new(d)?;
  let comm = match shared {
   Some(forward) => Rc::clone(&forward.comm),
   None => {
    let mut comm=ptr::null_mut();
    let mut error=[0i8;512];
    if let Some(cached)=crate::session_cache::comm_take(){comm=cached;}else{crate::generic_native::checked!(mgbfs_nccl_create, unsafe{mgbfs_nccl_create(rank,plan.world,device,id.as_ptr().cast(),&mut comm,error.as_mut_ptr(),error.len())})?;}
    Rc::new(PathCommunicator {ptr:comm,aborted:std::cell::Cell::new(false)})
   }
  };
  let max_frontier = counts.iter().copied().max().unwrap_or(0);
  let sort_cache=if plan.sort_candidates&&!use_sorted{Some(SortCache::new(&plan,d)?)}else{None};
  let parent_cache=if plan.parent_transport{Some(ParentCache::new(&plan,d,&counts)?)}else{None};
  let rolling_counts=[bfs.count,0,0];
  let gemm=if crate::generic_gemm::requested()?{Some(crate::generic_gemm::Context::new(graph,plan.batch,d)?)}else{None};
  if shared.is_none(){
   if let Some(workspace)=compact_path.as_ref(){
    if let Err(error)=unsafe{workspace.borrow_mut().transport.connect(comm.ptr)}{
     comm.abort(rank);
     return Err(error);
    }
   }
  }
  Ok(Self{compact_path,online_size_tuning:false,online_bucket:None,online_choices:std::collections::BTreeMap::new(),profile_events:vec![],active_batch:plan.batch,use_gemm:gemm.is_some(),gemm,sorted_owner,sort_cache,parent_cache,positions,rolling_counts,retired_rows:0,bfs,plan,rank,comm,queues,control,owner_map,owner_cuts,host_owner_map:map,host_owner_cuts:cuts.map(<[u64]>::to_vec),streams,ready,done,depth:0,parent_cursor:0,max_frontier,counts,source_retries:0})
 }
 pub(crate) fn reverse_path_arena(
  &self, graph: &GraphDefinitionV2, cuts: Option<&[u64]>, goals: &[Vec<i64>],
 ) -> Result<Self> {
  self.plan.shared_path_queue_bytes()?;
  Self::new_with_roots(graph,self.bfs.device as u32,self.rank,self.plan.clone(),&[0;128],
   self.bfs.seed,self.bfs.hash_bits,cuts,Some(goals),Some(self),None)
 }
 pub fn quiesce(&self)->Result<()>{crate::generic_native::checked!(cudaDeviceSynchronize, unsafe{cudaDeviceSynchronize()})?;crate::generic_native::checked!(cudaSetDevice, unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.bfs.device)})?;crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(self.bfs.stream.ptr)})?;for stream in &self.streams{crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream.ptr)})?;}Ok(())}
 pub fn enable_online_size_tuning(&mut self){self.online_size_tuning=true;}
 pub fn size_profile_events(&self)->&[serde_json::Value]{&self.profile_events}
 pub fn set_size_profile(&mut self,batch:u32,gemm:bool)->Result<()>{
  if batch==0||batch>self.plan.batch||(gemm&&self.gemm.is_none()){return Err("SIZE_PROFILE_EXCEEDS_ADMISSION".into());}
  let bucket=31-self.max_frontier.max(1).leading_zeros();let(batch,gemm)=self.online_choices.get(&bucket).copied().unwrap_or((batch,gemm));self.active_batch=batch;self.use_gemm=gemm;Ok(())
 }
 pub fn frontier_bucket(&self)->u32{31-self.max_frontier.max(1).leading_zeros()}
 pub fn cache_current_size_profile(&mut self,batch:u32,gemm:bool)->Result<()>{
  self.set_size_profile(batch,gemm)?;let bucket=31-self.max_frontier.max(1).leading_zeros();self.online_choices.insert(bucket,(batch,gemm));Ok(())
 }
 pub fn global_frontier(&self)->u64{self.counts.iter().map(|&v|u64::from(v)).sum()}
 pub(crate) fn path_layer_work(&self)->(u32,u64){
  (self.max_frontier.div_ceil(self.active_batch),self.global_frontier())
 }
 pub fn history_algorithm(&self)->&str{&self.plan.history_algorithm}
 pub fn owner_lanes(&self)->usize{self.streams.len()}
 pub fn source_retries(&self)->u64{self.source_retries}
 pub fn frontier_len(&self)->u32{self.bfs.count}
 pub(crate) fn path_arena(&self) -> Result<&GenericNativeBfs> {
  if self.plan.history_layers != 1 || self.plan.history_algorithm != "HASH" {
   return Err("NATIVE_DISTRIBUTED_PATH_REQUIRES_APPEND_ONLY_HASH".into());
  }
  Ok(&self.bfs)
 }
 pub(crate) fn path_world(&self) -> usize { self.plan.world as usize }
 pub(crate) fn path_rank(&self) -> usize { self.rank as usize }
 pub(crate) fn gather_path_words(&self, words: &[i64], tx: &Buffer, rx: &Buffer, output: &mut [i64]) -> Result<()> {
  let bytes = std::mem::size_of_val(words);
  let total = bytes.checked_mul(self.path_world()).ok_or("NATIVE_PATH_GATHER_SIZE")?;
  if total != std::mem::size_of_val(output) || total > rx.bytes || bytes > tx.bytes {
   return Err("NATIVE_PATH_GATHER_SIZE".into());
  }
  let stream = self.bfs.stream.ptr;
  let queued=(||->Result<()>{
   crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
    cudaMemcpyAsync(tx.ptr, words.as_ptr().cast(), bytes, 1, stream)
   })?;
   crate::generic_native::checked!(mgbfs_nccl_all_gather_bytes, unsafe {
    mgbfs_nccl_all_gather_bytes(self.comm.ptr, tx.ptr, rx.ptr, bytes as u64, stream)
   })?;
   crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
    cudaMemcpyAsync(output.as_mut_ptr().cast(), rx.ptr, total, 2, stream)
   })
  })();
  // A partial submission may still reference either caller-owned slice.
  let drained=crate::generic_native::checked!(cudaStreamSynchronize, unsafe { cudaStreamSynchronize(stream) });
  queued.and(drained)
 }
 pub fn sample_global(&mut self,limit:u32)->Result<Vec<Vec<i64>>>{let preceding=self.counts[..self.rank as usize].iter().map(|&v|u64::from(v)).sum::<u64>();let quota=u64::from(limit).saturating_sub(preceding).min(u64::from(self.bfs.count)) as u32;self.bfs.sample(quota)}
 /// Layer-boundary cancellation vote: every rank must call in the same order.
 pub fn collective_stop(&mut self,reason:u32)->Result<u32>{if self.bfs.terminal{return Err("GENERIC_DISTRIBUTED_TERMINAL".into());}
  let global=self.path_vote(reason)?;if global!=0{self.bfs.stop();}Ok(global)
 }
 pub(crate) fn path_vote(&self,reason:u32)->Result<u32>{
  let stream=self.bfs.stream.ptr;
  let mut global=0u32;
  let queued=(||->Result<()>{
   crate::generic_native::checked!(cudaMemcpyAsync, unsafe{
    cudaMemcpyAsync(self.control.at::<c_void>(16),(&reason as *const u32).cast(),4,1,stream)
   })?;
   crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32, unsafe{
    mgbfs_nccl_all_reduce_max_u32(self.comm.ptr,self.control.at(16),self.control.at(20),stream)
   })?;
   crate::generic_native::checked!(cudaMemcpyAsync, unsafe{
    cudaMemcpyAsync((&mut global as *mut u32).cast(),self.control.at::<c_void>(20),4,2,stream)
   })
  })();
  let drained=crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)});
  queued.and(drained)?;
  Ok(global)
 }

 pub fn stop(&mut self){self.bfs.stop();}
 pub fn sample(&mut self,limit:u32)->Result<Vec<Vec<i64>>>{self.bfs.sample(limit)}
 pub fn previous_small_sample(&mut self,global_previous:u64)->Result<Vec<Vec<i64>>>{if global_previous>=1000{return Ok(vec![]);}self.bfs.previous_small_sample()}
 pub fn advance(&mut self)->Result<DistributedAdvance>{
  self.advance_internal(None,None,None)
 }
 pub(crate) fn window_plan(&self)->&GenericDistributedMemoryPlan{&self.plan}
 pub(crate) const WINDOW_ROWS_PER_COPY:usize=8192;
 pub(crate) const WINDOW_EDGE_BYTES:usize=std::mem::size_of::<WindowEdge>();
 fn window_tile_rows(plan:&GenericDistributedMemoryPlan)->usize{
  (u64::from(plan.queue_capacity)*u64::from(plan.world)*u64::from(plan.shards))
   .min(Self::WINDOW_ROWS_PER_COPY as u64) as usize
 }
 pub(crate) fn window_plan_edge_bytes(plan:&GenericDistributedMemoryPlan)->usize{
  Self::window_tile_rows(plan)*Self::WINDOW_EDGE_BYTES
 }
 pub(crate) fn window_edge_bytes(&self)->usize{
  Self::window_plan_edge_bytes(&self.plan)
 }
 pub(crate) fn advance_window(
  &mut self, spool:&mut crate::path_csr::TransitionSpool, child_ids:&Buffer, tile:&mut WindowTile, checkpoint:&dyn Fn()->Result<()>,
 )->Result<DistributedAdvance>{
  if self.plan.history_layers!=1 || self.plan.history_algorithm!="HASH" || self.plan.packed_candidates()
     || self.plan.parent_transport || self.plan.sort_candidates || self.gemm.is_some() || self.online_size_tuning {
   return Err("NATIVE_WINDOW_OWNER_LAYOUT".into());
  }
  let required=self.window_edge_bytes();
  if child_ids.device!=self.bfs.device || child_ids.bytes<required ||
     tile.edges.len()*Self::WINDOW_EDGE_BYTES!=required ||
     tile.counts.len()!=self.plan.world as usize*self.plan.shards as usize{
   return Err("NATIVE_WINDOW_TILE_WORKSPACE".into());
  }
  if self.compact_path.is_some(){
   return self.advance_compact_path(None,Some((spool,child_ids,tile)),checkpoint);
  }
  self.advance_internal(Some((spool,child_ids,tile)),None,Some(checkpoint))
 }
 pub(crate) fn advance_nearest(&mut self, parents:&Buffer, checkpoint:&dyn Fn()->Result<()>)->Result<DistributedAdvance>{
  let p=&self.plan;
  if p.history_layers!=1||p.history_algorithm!="HASH"||p.packed_candidates()||
     p.parent_transport||p.sort_candidates||self.gemm.is_some()||
     parents.device!=self.bfs.device||parents.bytes<p.capacity as usize*crate::generic_path_native::PARENT_BYTES{
   return Err("NATIVE_NEAREST_PARENT_LAYOUT".into());
  }
  if self.compact_path.is_some(){
   return self.advance_compact_path(Some(parents),None,checkpoint);
  }
  self.advance_internal(None,Some(parents),Some(checkpoint))
 }
 fn advance_compact_path(
  &mut self, parents:Option<&Buffer>,
  lineage:Option<(&mut crate::path_csr::TransitionSpool,&Buffer,&mut WindowTile)>,
  checkpoint:&dyn Fn()->Result<()>,
 )->Result<DistributedAdvance>{
  let window=lineage.is_some();
  if self.bfs.terminal{return Err("GENERIC_DISTRIBUTED_TERMINAL".into());}
  self.bfs.terminal=true;
  let workspace=Rc::clone(self.compact_path.as_ref().ok_or("NATIVE_PATH_COMPACT_WORKSPACE")?);
  let mut workspace=workspace.borrow_mut();
  let expanded=(||{
   let b=&mut self.bfs;
   crate::generic_native::checked!(cudaSetDevice, unsafe{mgbfs_cuda::native_owner::cudaSetDevice(b.device)})?;
   if window{Self::prepare_window_bank(b)?;}
   crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(b.control.at::<c_void>(4),0,4,b.stream.ptr)})?;
   workspace.expand(self,parents,lineage,checkpoint)
  })();
  if expanded.is_err(){
   workspace.transport.notify_failure();
   self.comm.abort(self.rank);
   let _=workspace.transport.drain_after_abort();
  }
  self.finish_layer(window,None,Some(checkpoint),expanded?)
 }
 fn advance_internal(
  &mut self, mut lineage:Option<(&mut crate::path_csr::TransitionSpool,&Buffer,&mut WindowTile)>,
  path_parents:Option<&Buffer>, checkpoint:Option<&dyn Fn()->Result<()>>,
 )->Result<DistributedAdvance>{
  let mut path_error=None;
  let dense_queues=self.queues.as_deref().ok_or("GENERIC_DISTRIBUTED_QUEUE_LAYOUT")?;

  let b=&mut self.bfs;if b.terminal{return Err("GENERIC_DISTRIBUTED_TERMINAL".into());}b.terminal=true;
  crate::generic_native::checked!(cudaSetDevice, unsafe{mgbfs_cuda::native_owner::cudaSetDevice(b.device)})?;let stream=b.stream.ptr;let p=&self.plan;
  let rolling=p.history_layers==3;let future_bank=((self.depth+1)%3) as usize;let future_base=future_bank as u32*p.capacity;
  if rolling&&self.sorted_owner.is_some(){
   self.sorted_owner.as_ref().unwrap().retire_bank(future_bank as u32,b.control.at(8),stream)?;self.rolling_counts[future_bank]=0;
  }else if rolling{
   let positions=self.positions.as_ref().ok_or("GENERIC_ROLLING_POSITIONS")?;let expired=self.rolling_counts[future_bank];
   if expired!=0{crate::generic_native::checked!(mgbfs_generic_retire_rows_wide, unsafe{mgbfs_generic_retire_rows_wide(b.slots.ptr.cast(),p.table_slots,positions.ptr.cast(),b.arena_stride,future_base,expired,b.control.at(8),stream)})?;self.retired_rows+=u64::from(expired);self.rolling_counts[future_bank]=0;}
   // Tombstones preserve probe chains. Maintenance is amortized over at
   // least table_slots/4 retirements, rather than rebuilding each layer.
   if self.retired_rows>=u64::from(p.table_slots)/4{
    crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(b.slots.ptr,255,b.slots.bytes,stream)})?;
    for bank in 0..3{let count=self.rolling_counts[bank];if count!=0{crate::generic_native::checked!(mgbfs_generic_reseed_rows_storage, unsafe{mgbfs_generic_reseed_rows_storage(p.state_bytes,p.elements,b.visited.ptr.cast(),b.arena_stride,bank as u32*p.capacity,count,b.slots.ptr.cast(),p.table_slots,positions.ptr.cast(),b.seed,b.hash_bits,b.control.at(8),stream)})?;}}
    self.retired_rows=0;
   }
  }else if lineage.is_none()&&path_parents.is_none(){crate::generic_native::checked!(mgbfs_generic_gather_storage, unsafe{mgbfs_generic_gather_storage(p.state_bytes,p.elements,b.visited.ptr.cast(),b.arena_stride,b.front.ptr.cast(),b.count,b.parents.ptr.cast(),p.capacity,stream)})?;}
  if lineage.is_some(){Self::prepare_window_bank(b)?;}
  // Current remains immutable; generation reads its bank directly.
  // Nearest uses stable arena row IDs. Frontier queue order can differ
  // from row allocation order, so generate directly from contiguous rows.
  let direct_rows=rolling||path_parents.is_some();
  let parent_ptr=if direct_rows{b.visited.at::<i64>(b.current_start as usize*p.state_bytes as usize)}else{b.parents.ptr.cast()};
  let parent_stride=if direct_rows{b.arena_stride}else{p.capacity};
  crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(b.control.at::<c_void>(4),0,4,stream)})?;
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
   crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32, unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm.ptr,b.control.at(4),self.control.at(16),stream)})?;
   crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)})?;
  }
  let queues=p.world as usize*p.shards as usize;let q=p.queue_capacity as usize;let width=p.elements as usize;let shards=p.shards as usize;let state_bytes=p.state_bytes as usize;
  let fused_route=if (lineage.is_some()||path_parents.is_some())&&p.elements<=16
    &&std::env::var("MGBFS_GENERIC_PATH_FUSED_ROUTE").as_deref()!=Ok("0"){
   fused_path_route()
  }else{None};
  let mut exact_counts=if p.world>1&&(lineage.is_some()||path_parents.is_some())
    &&std::env::var("MGBFS_GENERIC_PATH_EXACT_EXCHANGE").as_deref()==Ok("1"){
   Some(vec![0u32;2*queues])
  }else{None};
  let vote_kernel = route_vote_kernel();
  while global_begin<u64::from(self.max_frontier){
   if profiling&&trial<schedule.len(){let choice=choices[schedule[trial]];batch=choice.0;self.use_gemm=choice.1;}
   let trial_started=if profiling&&trial<schedule.len(){Some(std::time::Instant::now())}else{None};

   let bank=&dense_queues.banks[((self.depth+round)%2) as usize];let begin=global_begin.min(u64::from(b.count)) as u32;let count=batch.min(b.count-begin);
   let origin_begin=if lineage.is_some(){u64::from(begin)}else if path_parents.is_some(){u64::from(b.current_start)+u64::from(begin)}else{self.parent_cursor+u64::from(begin)};
   crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(self.control.at::<c_void>(8),0,4,stream)})?;
   crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(bank.counts.ptr,0,bank.counts.bytes,stream)})?;
   if let Some(route)=fused_route{
    crate::generic_native::checked!(mgbfs_generic_route_with_states_v1,unsafe{
     route(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,
      parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,
      b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),b.seed,b.hash_bits,p.world,
      self.rank,p.shards,p.queue_capacity,origin_begin,self.owner_map.ptr.cast(),
      self.owner_cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),bank.records.ptr.cast(),
      bank.counts.ptr.cast(),self.control.at(8),bank.states.ptr,stream)
    })?;
   }else if let Some(ctx)=self.gemm.as_ref().filter(|_|self.use_gemm){crate::generic_native::checked!(mgbfs_generic_route_gemm_storage, unsafe{mgbfs_generic_route_gemm_storage(ctx.ptr,p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),b.seed,b.hash_bits,p.world,self.rank,p.shards,p.queue_capacity,origin_begin,self.owner_map.ptr.cast(),self.owner_cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),bank.records.ptr.cast(),bank.counts.ptr.cast(),self.control.at(8),stream)})?;}else{crate::generic_native::checked!(mgbfs_generic_route_storage, unsafe{mgbfs_generic_route_storage(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),b.seed,b.hash_bits,p.world,self.rank,p.shards,p.queue_capacity,origin_begin,self.owner_map.ptr.cast(),self.owner_cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),bank.records.ptr.cast(),bank.counts.ptr.cast(),self.control.at(8),stream)})?;}
   // An exhausted source still joins exchange and receives inbound work. Its
   // zeroed counts make all source payloads empty, so skip regeneration.
   if fused_route.is_none()&&count!=0&&!p.packed_candidates()&&!p.parent_transport{
    let source=unsafe{parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast()};
    if let Some(ctx)=self.gemm.as_ref().filter(|_|self.use_gemm){
     for queue in 0..queues{
      crate::generic_native::checked!(mgbfs_generic_regenerate_gemm_routes_count_storage,unsafe{
       mgbfs_generic_regenerate_gemm_routes_count_storage(ctx.ptr,p.state_bytes,b.kind,p.elements,
        b.rows,b.cols,b.generators,source,count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),
        b.moduli.ptr.cast(),self.rank,origin_begin,bank.records.at(queue*q*32),p.queue_capacity,
        bank.counts.at(queue*4),bank.states.at(queue*q*width*state_bytes),p.queue_capacity,self.control.at(8),stream)
      })?;
     }
    }else{
     crate::generic_native::checked!(mgbfs_generic_regenerate_all_routes_count_storage,unsafe{
      mgbfs_generic_regenerate_all_routes_count_storage(p.state_bytes,b.kind,p.elements,b.rows,b.cols,
       b.generators,source,count,parent_stride,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),
       self.rank,origin_begin,bank.records.ptr.cast(),bank.counts.ptr.cast(),p.queue_capacity,
       queues as u32,bank.states.ptr,self.control.at(8),stream)
     })?;
    }
   }
   if let Some(cache)=&self.parent_cache{
    crate::generic_native::checked!(mgbfs_generic_pack_parent_chunk, unsafe{mgbfs_generic_pack_parent_chunk(p.state_bytes,p.elements,parent_ptr.cast::<u8>().add(begin as usize*state_bytes).cast(),parent_stride,count,cache.sources[((self.depth+round)%2) as usize].ptr,batch,stream)})?;
   }
   // Invalid source counts must not be presented as valid owner inboxes.
   // Generate the next immutable source bank while preceding owners work.
   // Retire their inbox leases only before reading owner errors/exchanging.
   if round>0{for event in &self.done{crate::generic_native::checked!(cudaStreamWaitEvent, unsafe{cudaStreamWaitEvent(stream,event.ptr,0)})?;}}
   if let Some(checkpoint)=checkpoint{
    if path_error.is_none(){path_error=checkpoint().err();}
   }
   // Piggyback a path stop on the existing per-round vote. Outstanding
   // owners have drained, and no rank has entered the next exchange.
   let stop=if path_error.is_some(){PATH_STOP_VOTE}else{0};
   match round_vote(vote_kernel, &self.control, b.control.at(8), self.comm.ptr, stop, stream)? {
    RoundVote::Continue => {}
    RoundVote::Stop => break,
    RoundVote::Retry if batch>1 => {
     profiling=false;self.online_bucket=Some(bucket);batch=(batch/2).max(1);self.source_retries+=1;continue;
    }
    RoundVote::Retry => return Ok(DistributedAdvance::Resource{fatal:1}),
    RoundVote::Capacity(fatal) | RoundVote::Fatal(fatal) => return Ok(DistributedAdvance::Resource{fatal}),
   }
   // A single rank consumes its source queues directly. Owner kernels select
   // local inputs by sender rank; the remote arguments are read-only.
   let inbox=dense_queues.inbox.as_ref().unwrap_or(bank);
   if dense_queues.inbox.is_some(){
    if let Some(counts)=exact_counts.as_mut(){
     let (outgoing,incoming)=counts.split_at_mut(queues);
     crate::generic_native::checked!(mgbfs_nccl_exchange_triplets,unsafe{
      mgbfs_nccl_exchange_triplets(self.comm.ptr,self.rank,p.world,bank.counts.ptr,u64::from(p.shards)*4,
       ptr::null(),0,ptr::null(),0,inbox.counts.ptr,ptr::null_mut(),ptr::null_mut(),stream)
     })?;
     let copied=(||->Result<()>{
      crate::generic_native::checked!(cudaMemcpyAsync,unsafe{
       cudaMemcpyAsync(outgoing.as_mut_ptr().cast(),bank.counts.ptr,queues*4,2,stream)
      })?;
      crate::generic_native::checked!(cudaMemcpyAsync,unsafe{
       cudaMemcpyAsync(incoming.as_mut_ptr().cast(),inbox.counts.ptr,queues*4,2,stream)
      })
     })();
     let drained=crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)});
     copied.and(drained)?;
     crate::generic_native::checked!(mgbfs_nccl_exchange_exact_payload,unsafe{
      mgbfs_nccl_exchange_exact_payload(self.comm.ptr,self.rank,p.world,p.shards,p.queue_capacity,
       p.queue_payload_bytes() as u64,p.state_bytes,outgoing.as_ptr(),incoming.as_ptr(),bank.records.ptr,
       bank.states.ptr,inbox.records.ptr,inbox.states.ptr,stream)
     })?;
    }else{
     // Every rank has the same committed maximum frontier and chunk
     // cursor. Even the busiest source cannot exceed this remaining work.
     let parents=(u64::from(self.max_frontier)-global_begin).min(u64::from(batch));
     let used=(parents*u64::from(b.generators)).min(u64::from(p.queue_capacity)) as u32;
     crate::generic_native::checked!(mgbfs_nccl_exchange_bounded_triplets, unsafe{mgbfs_nccl_exchange_bounded_triplets(self.comm.ptr,self.rank,p.world,p.shards,p.queue_capacity,used,p.queue_payload_bytes() as u64,p.state_bytes,bank.counts.ptr,bank.records.ptr,bank.states.ptr,inbox.counts.ptr,inbox.records.ptr,inbox.states.ptr,stream)})?;
    }
   }
   if let Some(cache)=&self.parent_cache{
    crate::generic_native::checked!(mgbfs_nccl_all_gather_bytes, unsafe{mgbfs_nccl_all_gather_bytes(self.comm.ptr,cache.sources[((self.depth+round)%2) as usize].ptr,cache.received.ptr,u64::from(batch)*u64::from(p.elements)*u64::from(p.state_bytes),stream)})?;
   }
   crate::generic_native::checked!(cudaEventRecord, unsafe{cudaEventRecord(self.ready.ptr,stream)})?;
   for shard in 0..shards{let owner_stream=self.streams[shard%self.streams.len()].ptr;if shard<self.streams.len(){crate::generic_native::checked!(cudaStreamWaitEvent, unsafe{cudaStreamWaitEvent(owner_stream,self.ready.ptr,0)})?;}
    if let Some(counts)=exact_counts.as_ref(){
     let (outgoing,incoming)=counts.split_at(queues);
     let rank=self.rank as usize;
     // NCCL leaves the self receive slot untouched. Its live count comes
     // from the source bank, while the other slots are indexed by sender.
     if outgoing[rank*shards+shard]==0
       && incoming[shard..].iter().step_by(shards).enumerate()
        .all(|(sender,&count)|sender==rank||count==0){
      continue;
     }
    }
    if let Some(owner)=&self.sorted_owner{
     let mut input=SortedInput{elements:p.elements,world:p.world,rank:self.rank,shard:shard as u32,shards:p.shards,queue_capacity:p.queue_capacity,state_bytes:p.state_bytes,transport:if p.packed_candidates(){1}else if p.parent_transport{2}else{0},kind:b.kind,rows:b.rows,cols:b.cols,generators:b.generators,parent_stride:batch,chunk:batch,hash_bits:b.hash_bits,begin:global_begin,seed:b.seed,local:bank.states.ptr,remote:inbox.states.ptr,local_meta:bank.records.ptr,remote_meta:inbox.records.ptr,local_counts:bank.counts.ptr,remote_counts:inbox.counts.ptr,permutations:b.perms.ptr,matrices:b.matrices.ptr,moduli:b.moduli.ptr,..Default::default()};
     if let Some(cache)=&self.parent_cache{input.parents=cache.received.ptr;input.cursors=cache.cursors.ptr;input.frontiers=cache.frontiers.ptr;}
     let destination=SortedDestination{arena:b.visited.ptr,stride:b.arena_stride,base:if rolling{future_base}else{0},capacity:p.capacity,rolling:u32::from(rolling),row_count:if rolling{b.control.at(4)}else{b.control.at(0)},frontier_count:b.control.at(4),future:b.future.ptr,error:b.control.at(8),..Default::default()};
     owner.accept(&input,destination,if rolling{future_bank as u32}else{0})?;
     continue;
    }
    let ordered=if let Some(cache)=&self.sort_cache{
     let offset=shard*p.world as usize*q;
     crate::generic_native::checked!(mgbfs_generic_sort_origins, unsafe{mgbfs_generic_sort_origins(bank.records.ptr.cast(),inbox.records.ptr.cast(),bank.counts.ptr.cast(),inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,cache.keys.at(offset*8),cache.sorted_keys.at(offset*8),cache.origins.at(offset*4),cache.sorted_origins.at(offset*4),cache.scratch.at::<c_void>(shard*cache.scratch_stride),cache.scratch_stride as u64,b.control.at(8),owner_stream)})?;cache.sorted_origins.at::<u32>(offset*4).cast_const()
    }else{ptr::null()};
    if let Some(cache)=&self.parent_cache{
     crate::generic_native::checked!(mgbfs_generic_accept_parent_origin_wide, unsafe{mgbfs_generic_accept_parent_origin_wide(p.state_bytes,b.kind,p.elements,b.rows,b.cols,b.generators,b.perms.ptr.cast(),b.matrices.ptr.cast(),b.moduli.ptr.cast(),cache.received.ptr,batch,batch,global_begin,cache.cursors.ptr.cast(),cache.frontiers.ptr.cast(),bank.records.ptr.cast(),inbox.records.ptr.cast(),bank.counts.ptr.cast(),inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr,b.arena_stride,if rolling{future_base}else{0},p.capacity,b.control.at(0),b.future.ptr.cast(),b.control.at(4),self.positions.as_ref().map_or(ptr::null_mut(),|v|v.ptr.cast()),b.control.at(8),u32::from(rolling),ordered,owner_stream)})?;
    }else if self.sort_cache.is_some(){
     crate::generic_native::checked!(mgbfs_generic_accept_sorted_wide, unsafe{mgbfs_generic_accept_sorted_wide(p.state_bytes,p.elements,bank.states.ptr, inbox.states.ptr,bank.records.ptr.cast(),inbox.records.ptr.cast(),bank.counts.ptr.cast(),inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr,b.arena_stride,if rolling{future_base}else{0},p.capacity,b.control.at(0),b.future.ptr.cast(),b.control.at(4),self.positions.as_ref().map_or(ptr::null_mut(),|v|v.ptr.cast()),b.control.at(8),u32::from(rolling),ordered,owner_stream)})?;
    }else if let Some(parents)=path_parents{
     crate::generic_native::checked!(mgbfs_generic_accept_path_parents_v2, unsafe{
      mgbfs_generic_accept_path_parents_v2(p.state_bytes,p.elements,b.generators,bank.states.ptr,inbox.states.ptr,
       bank.records.ptr.cast(),inbox.records.ptr.cast(),bank.counts.ptr.cast(),inbox.counts.ptr.cast(),
       self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr,
       p.capacity,b.control.at(0),b.future.ptr.cast(),p.capacity,b.control.at(4),parents.ptr.cast(),b.control.at(8),owner_stream)
     })?;
    }else if rolling{crate::generic_native::checked!(mgbfs_generic_accept_rolling_storage, unsafe{mgbfs_generic_accept_rolling_storage(p.state_bytes,p.elements,bank.states.ptr.cast(),inbox.states.ptr.cast(),bank.records.ptr.cast(),inbox.records.ptr.cast(),bank.counts.ptr.cast(),inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr.cast(),b.arena_stride,future_base,p.capacity,b.control.at(4),b.future.ptr.cast(),self.positions.as_ref().unwrap().ptr.cast(),b.control.at(8),owner_stream)})?;}
    else{    crate::generic_native::checked!(mgbfs_generic_accept_all_storage, unsafe{mgbfs_generic_accept_all_storage(p.state_bytes,p.elements,bank.states.ptr.cast(),inbox.states.ptr.cast(),bank.records.ptr.cast(),inbox.records.ptr.cast(),bank.counts.ptr.cast(),inbox.counts.ptr.cast(),self.rank,p.world,shard as u32,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr.cast(),p.capacity,b.control.at(0),b.future.ptr.cast(),p.capacity,b.control.at(4),b.control.at(8),owner_stream)})?;}
   }
   for(lane,owner_stream)in self.streams.iter().enumerate(){crate::generic_native::checked!(cudaEventRecord, unsafe{cudaEventRecord(self.done[lane].ptr,owner_stream.ptr)})?;}
   // Next round may generate into the other source bank; inbox retirement
   // waits occur after that generation and before the next exchange.
   if let Some((spool,ids,tile))=lineage.as_mut(){
    for event in &self.done{
     crate::generic_native::checked!(cudaStreamWaitEvent, unsafe{cudaStreamWaitEvent(stream,event.ptr,0)})?;
    }
    let status=tile.write(
      b,p,self.rank,bank,inbox,exact_counts.as_deref(),ids,stream,&mut **spool,
      checkpoint.ok_or("NATIVE_WINDOW_CHECKPOINT")?);
    if path_error.is_none(){path_error=status.err();}
   }
   if profiling&&trial<schedule.len(){
    for event in &self.done{crate::generic_native::checked!(cudaStreamWaitEvent, unsafe{cudaStreamWaitEvent(stream,event.ptr,0)})?;}
    crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)})?;
    let mut owner_error=0u32;crate::generic_native::checked!(cudaMemcpyAsync, unsafe{cudaMemcpyAsync((&mut owner_error as *mut u32).cast(),b.control.at::<c_void>(8),4,2,stream)})?;crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)})?;
    let micros=if owner_error!=0{u32::MAX}else{trial_started.unwrap().elapsed().as_micros().min(u128::from(u32::MAX-1)) as u32};
    crate::generic_native::checked!(cudaMemcpyAsync, unsafe{cudaMemcpyAsync(self.control.at::<c_void>(16),(&micros as *const u32).cast(),4,1,stream)})?;
    crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32, unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm.ptr,self.control.at(16),self.control.at(20),stream)})?;
    let mut maximum=0u32;crate::generic_native::checked!(cudaMemcpyAsync, unsafe{cudaMemcpyAsync((&mut maximum as *mut u32).cast(),self.control.at::<c_void>(20),4,2,stream)})?;crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)})?;
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
  if round>0{for event in &self.done{crate::generic_native::checked!(cudaStreamWaitEvent, unsafe{cudaStreamWaitEvent(stream,event.ptr,0)})?;}}
  self.finish_layer(lineage.is_some(),rolling.then_some(future_bank),checkpoint,path_error)
 }

 fn prepare_window_bank(b: &mut GenericNativeBfs) -> Result<()> {
  // CSR IDs are arena rows. Dedup writes a fresh depth into the other bank.
  if b.current_start!=0||b.visited_used!=b.count||b.parents.bytes!=b.visited.bytes{
   return Err("NATIVE_WINDOW_BANK_LAYOUT".into());
  }
  std::mem::swap(&mut b.parents,&mut b.visited);
  let stream=b.stream.ptr;
  crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(b.slots.ptr,255,b.slots.bytes,stream)})?;
  crate::generic_native::checked!(cudaMemsetAsync, unsafe{cudaMemsetAsync(b.control.ptr,0,4,stream)})?;
  b.visited_used=0;
  Ok(())
 }

 /// All producer and owner streams must precede this boundary on the BFS
 /// stream. Publish counters and swap banks only after the rank agreement.
 fn finish_layer(
  &mut self, window: bool, rolling_bank: Option<usize>,
  checkpoint: Option<&dyn Fn()->Result<()>>, path_error: Option<String>,
 )->Result<DistributedAdvance>{
  let b=&self.bfs;
  let stream=b.stream.ptr;
  crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32, unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm.ptr,b.control.at(8),self.control.at(0),stream)})?;
  crate::generic_native::checked!(mgbfs_nccl_all_gather_u32, unsafe{mgbfs_nccl_all_gather_u32(self.comm.ptr,b.control.at(4),self.control.at(32),stream)})?;
  // Keep the current layer's counts available when the last owner wave
  // stops the next layer before it can commit.
  let mut local_counts = [0u32; 128];
  let next_counts = &mut local_counts[..self.counts.len()];
  let mut fatal=0u32;let mut visited=0u32;
  let queued=(||->Result<()>{
   crate::generic_native::checked!(cudaMemcpyAsync, unsafe{cudaMemcpyAsync((&mut fatal as *mut u32).cast(),self.control.ptr,4,2,stream)})?;
   crate::generic_native::checked!(cudaMemcpyAsync, unsafe{cudaMemcpyAsync(next_counts.as_mut_ptr().cast(),self.control.at::<c_void>(32),std::mem::size_of_val(next_counts),2,stream)})?;
   crate::generic_native::checked!(cudaMemcpyAsync, unsafe{cudaMemcpyAsync((&mut visited as *mut u32).cast(),b.control.ptr,4,2,stream)})
  })();
  let drained=crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)});
  queued.and(drained)?;
  let vote = RoundVote::owner(fatal);
  if let RoundVote::Fatal(fatal) = vote {
   return Ok(DistributedAdvance::Resource{fatal});
  }
  if let Some(checkpoint)=checkpoint{
   // Drain the last exchange and its consumers before committing or stopping.
   let result=path_error.map_or_else(checkpoint,Err);
   if let Some(reason)=crate::generic_path_native::agree_path_stop(self,&result)?{
    return Err(reason.into());
   }
  }
  let b=&mut self.bfs;let p=&self.plan;
  if let RoundVote::Capacity(fatal) = vote {return Ok(DistributedAdvance::Resource{fatal});}
  let global=next_counts.iter().map(|&v|u64::from(v)).sum::<u64>();if global==0{return Ok(DistributedAdvance::Complete);}
  let local=next_counts[self.rank as usize];if local>p.capacity{return Err("GENERIC_DISTRIBUTED_COUNT_BOUNDS".into());}
  b.previous=if window{None}else{Some((b.current_start,b.count))};if let Some(future_bank)=rolling_bank{b.current_start=future_bank as u32*p.capacity;self.rolling_counts[future_bank]=local;}else{b.current_start=b.visited_used;b.visited_used=visited;}
  self.parent_cursor=self.parent_cursor.checked_add(u64::from(b.count)).ok_or("GENERIC_PARENT_CURSOR_OVERFLOW")?;
  if let Some(cache)=&self.parent_cache{crate::generic_native::checked!(mgbfs_generic_advance_parent_cursors, unsafe{mgbfs_generic_advance_parent_cursors(cache.cursors.ptr.cast(),cache.frontiers.ptr.cast(),self.control.at(32),p.world,stream)})?;}
  self.counts.copy_from_slice(next_counts);
  self.depth+=1;self.max_frontier=*self.counts.iter().max().unwrap();std::mem::swap(&mut b.front,&mut b.future);b.count=local;b.terminal=false;
  Ok(DistributedAdvance::Layer{local:u64::from(local),global})
 }
}
const PATH_STOP_VOTE:u32=1<<31;

/// Export only matched route metadata and child IDs. Candidate state vectors
/// remain on the device while the host writes the bounded transition runs.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WindowEdge {parent:u64,child:u32,generator:u32}
pub(crate) struct WindowTile {
 edges:Vec<WindowEdge>,
 senders:Vec<u32>,
 counts:Vec<u32>,
 transitions:Vec<crate::path_csr::Transition>,
}
impl WindowTile {
 pub(crate) fn new(plan:&GenericDistributedMemoryPlan)->Self{
  let tile_rows=GenericDistributedBfs::window_tile_rows(plan);
  Self{
   edges:vec![WindowEdge::default();tile_rows],
   senders:vec![0;tile_rows],
   counts:vec![0;plan.world as usize*plan.shards as usize],
   transitions:Vec::with_capacity(GenericDistributedBfs::WINDOW_ROWS_PER_COPY),
  }
 }
 fn write(
 &mut self,b:&GenericNativeBfs,p:&GenericDistributedMemoryPlan,rank:u32,source:&QueueBank,inbox:&QueueBank,
 exchanged_counts:Option<&[u32]>,ids:&Buffer,stream:*mut c_void,spool:&mut crate::path_csr::TransitionSpool,checkpoint:&dyn Fn()->Result<()>,
)->Result<()>{
 use crate::path_csr::Transition;
 checkpoint()?;
 let mut counters=[0u32;3];
 let q=p.queue_capacity as usize;
 let Self{edges,senders,counts,transitions}=self;
 // Discard staging from a previous failed write before reusing the tile.
 transitions.clear();
 let tile_rows=edges.len();
 let queues=counts.len();
 let local_begin=rank as usize*p.shards as usize;
 let local_end=local_begin+p.shards as usize;
 let copied=(||->Result<()>{
  crate::generic_native::checked!(cudaMemcpyAsync, unsafe{
   cudaMemcpyAsync(counters.as_mut_ptr().cast(),b.control.ptr,12,2,stream)
  })?;
  if let Some(exchanged)=exchanged_counts{
   let (outgoing,incoming)=exchanged.split_at(queues);
   counts[..local_begin].copy_from_slice(&incoming[..local_begin]);
   counts[local_begin..local_end].copy_from_slice(&outgoing[local_begin..local_end]);
   counts[local_end..].copy_from_slice(&incoming[local_end..]);
  }else{
   if p.world>1{
    crate::generic_native::checked!(cudaMemcpyAsync, unsafe{
     cudaMemcpyAsync(counts.as_mut_ptr().cast(),inbox.counts.ptr,queues*4,2,stream)
    })?;
   }
   crate::generic_native::checked!(cudaMemcpyAsync, unsafe{
    cudaMemcpyAsync(counts.as_mut_ptr().add(local_begin).cast(),source.counts.at::<c_void>(local_begin*4),
     p.shards as usize*4,2,stream)
   })?;
  }
  Ok(())
 })();
 let drained=crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)});
 copied.and(drained)?;
 if counters[2]!=0{return Ok(());}
 let child_count=counters[0].checked_sub(b.visited_used).ok_or("NATIVE_WINDOW_CHILD_COUNT")?;
 let mut flush=|edges:&mut [WindowEdge],senders:&[u32]|->Result<()>{
  if edges.is_empty(){return Ok(());}
  let copied=crate::generic_native::checked!(cudaMemcpyAsync, unsafe{
   cudaMemcpyAsync(edges.as_mut_ptr().cast(),ids.ptr,std::mem::size_of_val(edges),2,stream)
  });
  let drained=crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)});
  copied.and(drained)?;
  checkpoint()?;
  for (edge,&sender) in edges.iter().zip(senders){
   // Only live queue entries are submitted. Every entry must resolve to a
   // child in this layer before any edge from this tile reaches the spool.
   if edge.child>=child_count{return Err("NATIVE_WINDOW_UNRESOLVED_EDGE".into());}
   if edge.generator>=b.generators{return Err("NATIVE_WINDOW_ROUTE_METADATA".into());}
   transitions.push(Transition{child:edge.child,parent_rank:sender,parent:edge.parent,generator:edge.generator});
  }
  spool.add(transitions)?;
  transitions.clear();
  Ok(())
 };
 let written=(||->Result<()>{
  let mut used=0;
  for shard in 0..p.shards {
   for sender in 0..p.world as usize{
    let queue=sender*p.shards as usize+shard as usize;
    let count=counts[queue] as usize;
    if count>q{return Err("NATIVE_WINDOW_ROUTE_COUNT".into());}
    let mut begin=0;
    while begin<count{
     checkpoint()?;
     let n=(count-begin).min(tile_rows-used);
     crate::generic_native::checked!(mgbfs_generic_resolve_incoming_edges_v3, unsafe{
      mgbfs_generic_resolve_incoming_edges_v3(p.state_bytes,p.elements,source.states.ptr,inbox.states.ptr,
       source.records.ptr.cast(),inbox.records.ptr.cast(),source.counts.ptr.cast(),inbox.counts.ptr.cast(),
       rank,p.world,shard,p.shards,p.queue_capacity,b.slots.ptr.cast(),p.table_slots,b.visited.ptr,b.arena_stride,
       b.visited_used,counters[0],ids.at(used*GenericDistributedBfs::WINDOW_EDGE_BYTES),(tile_rows-used) as u32,
       (sender*q+begin) as u32,n as u32,b.control.at(8),stream)
     })?;
     senders[used..used+n].fill(sender as u32);
     used+=n;
     begin+=n;
     if used==tile_rows{
      flush(edges,senders)?;
      used=0;
     }
    }
   }
  }
  flush(&mut edges[..used],&senders[..used])?;
  checkpoint()?;
  Ok(())
 })();
 if written.is_err(){
  let _=crate::generic_native::checked!(cudaStreamSynchronize, unsafe{cudaStreamSynchronize(stream)});
 }
 written
 }
}

impl Drop for GenericDistributedBfs{
 fn drop(&mut self){unsafe{
  mgbfs_cuda::native_owner::cudaSetDevice(self.bfs.device);
  let failed=crate::native_fault::failed();
  if failed {
   // A fatal rank cannot finish pending collectives normally. Abort before
   // stream drains; never park this communicator in the healthy session cache.
   self.comm.abort(self.rank);
  }
  cudaStreamSynchronize(self.bfs.stream.ptr);
  for stream in &self.streams{cudaStreamSynchronize(stream.ptr);}
 }}
}

