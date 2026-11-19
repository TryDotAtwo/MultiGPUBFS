//! Experimental independently segmented distributed exact BFS.
//! Not selected by the public runtime until multi-process and failure gates pass.
use std::{ffi::c_void,ptr};
use mgbfs_core::{graph_definition::GraphDefinitionV2,Result};
use mgbfs_cuda::{ffi::*,generic_graph::*};
use crate::{generic_distributed_memory::GenericDistributedMemoryPlan as Plan,
 generic_distributed_native::DistributedAdvance,generic_memory::GenericMemoryPlan,
 generic_native::{Buffer,GenericNativeBfs,Stream},segmented_history::{SegmentLayout,LOCAL_ROWS}};

#[repr(C)] #[derive(Clone,Copy)]
struct SegmentDescriptor{slots:*mut u64,states:*mut c_void,positions:*mut u64,slots_count:u64,capacity:u64,banks:u32}
extern "C" {
 fn cudaGetLastError()->i32;
 fn cudaEventSynchronize(event:*mut c_void)->i32;
 fn cudaEventElapsedTime(ms:*mut f32,start:*mut c_void,end:*mut c_void)->i32;
 fn mgbfs_segmented_route_storage(state_bytes:u32,kind:u32,elements:u32,n:u32,m:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,matrices:*const i64,moduli:*const u32,seed:u64,bits:u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,cuts:*const u64,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_generic_max_queue_count(counts:*const u32,n:u32,maximum:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_nccl_all_reduce_max_pair_u32(comm:*mut c_void,send:*const u32,recv:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_route(state_bytes:u32,domain:BitmapDeviceDomain,elements:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,world:u32,source:u32,shards:u32,capacity:u32,begin:u64,map:*const u32,output:*mut GenericRouteRecord,counts:*mut u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_seed(state_bytes:u32,input:*const c_void,elements:u32,active:SegmentDescriptor,bits:*mut u64,local:u64,accepted:*mut u64,stream:*mut c_void)->i32;
 fn mgbfs_bitmap_accept(state_bytes:u32,packed:u32,elements:u32,local:*const c_void,remote:*const c_void,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,owner:u32,total:u64,owner_len:u64,bits:*mut u64,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,spill:*mut u32,spill_count:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_generic_regenerate_all_routes_count_storage_wide_stride(state_bytes:u32,kind:u32,elements:u32,n:u32,m:u32,generators:u32,parents:*const c_void,count:u32,stride:u64,permutations:*const u32,matrices:*const i64,moduli:*const u32,source:u32,begin:u64,requests:*const GenericRouteRecord,counts:*const u32,capacity:u32,queues:u32,output:*mut c_void,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_nccl_exchange_bounded_triplets(comm:*mut c_void,rank:u32,world:u32,shards:u32,capacity:u32,used:u32,payload:u64,state_bytes:u32,counts:*const c_void,records:*const c_void,states:*const c_void,recv_counts:*mut c_void,recv_records:*mut c_void,recv_states:*mut c_void,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_u8(input:*const u8,hashes:*const u64,n:u32,elements:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_i64(input:*const i64,hashes:*const u64,n:u32,elements:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_routes_limited_u8(elements:u32,local:*const u8,remote:*const u8,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,visible:u32,packed:u32,selection:*const u32,selected:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,spill:*mut u32,sc:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_segmented_accept_routes_limited_i64(elements:u32,local:*const i64,remote:*const i64,lm:*const GenericRouteRecord,rm:*const GenericRouteRecord,lc:*const u32,rc:*const u32,rank:u32,world:u32,shard:u32,shards:u32,q:u32,visible:u32,selection:*const u32,selected:u32,sealed:*const SegmentDescriptor,ns:u32,active:SegmentDescriptor,bank:u32,accepted:*mut u64,future:*mut u64,spill:*mut u32,sc:*mut u32,error:*mut u32,global_count:*mut u64,limit:u64,stream:*mut c_void)->i32;
 fn mgbfs_segmented_retire(seg:SegmentDescriptor,bank:u32,count:u64,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_rehash_u8(width:u32,seg:SegmentDescriptor,c0:u64,c1:u64,c2:u64,seed:u64,bits:u32,error:*mut u32,stream:*mut c_void)->i32;
 fn mgbfs_segmented_rehash_i64(width:u32,seg:SegmentDescriptor,c0:u64,c1:u64,c2:u64,seed:u64,bits:u32,error:*mut u32,stream:*mut c_void)->i32;
}
struct RawBuffer{ptr:*mut c_void,bytes:usize,device:i32}
impl RawBuffer{
 fn try_new(bytes:usize,device:i32)->Result<Option<Self>>{let mut p=ptr::null_mut();let bytes=bytes.max(1);let code=unsafe{cudaMalloc(&mut p,bytes)};if code==2{
  // Only classify allocation failure after prior stream work has been drained.
  // Clear the same runtime/thread's last error, but never hide a different one.
  let last=unsafe{cudaGetLastError()};if last!=0&&last!=2{return Err(format!("SEGMENTED_MALLOC_LAST_ERROR_{last}"))}return Ok(None)
 }crate::generic_native::checked!(cudaMalloc,code)?;Ok(Some(Self{ptr:p,bytes,device}))}
 fn at<T>(&self,offset:usize)->*mut T{unsafe{self.ptr.cast::<u8>().add(offset).cast()}}
}
impl Drop for RawBuffer{fn drop(&mut self){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(self.device);cudaFree(self.ptr);}}}
struct Segment{desc:SegmentDescriptor,_slots:Option<RawBuffer>,_states:RawBuffer,_positions:Option<RawBuffer>,counts:RawBuffer,banks:[u64;3],retired:u64}
impl Segment{
 fn try_new(layout:SegmentLayout,device:i32,stream:*mut c_void)->Result<Option<Self>>{
  macro_rules! alloc{($n:expr)=>{match RawBuffer::try_new(usize::try_from($n).map_err(|_|"SEGMENT_ALLOCATION_RANGE")?,device)?{Some(v)=>v,None=>return Ok(None)}}}
  let slots=if layout.slots>0{Some(alloc!(layout.slots.checked_mul(8).ok_or("SEGMENT_ALLOCATION_RANGE")?))}else{None};
  let states=alloc!(layout.state_bytes);
  let positions=if layout.position_bytes>0{Some(alloc!(layout.position_bytes))}else{None};
  let counts=alloc!(24u64);
  if let Some(v)=&slots{crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(v.ptr,255,v.bytes,stream)})?;}
  crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(counts.ptr,0,counts.bytes,stream)})?;
  let desc=SegmentDescriptor{slots:slots.as_ref().map_or(ptr::null_mut(),|v|v.ptr.cast()),states:states.ptr,positions:positions.as_ref().map_or(ptr::null_mut(),|v|v.ptr.cast()),slots_count:layout.slots,capacity:layout.rows_per_bank,banks:layout.banks};
  Ok(Some(Self{desc,_slots:slots,_states:states,_positions:positions,counts,banks:[0;3],retired:0}))
 }
}
struct Queue{records:Buffer,counts:Buffer,states:Buffer}
impl Queue{fn new(p:&Plan,d:i32)->Result<Self>{let n=p.world as usize*p.shards as usize;let q=p.queue_capacity as usize;Ok(Self{records:Buffer::new(n*q*32,d)?,counts:Buffer::new(n*4,d)?,states:Buffer::new(n*q*p.queue_payload_bytes(),d)?})}}
// Pinned metadata lets each lane enqueue its D2H copies before the host
// waits, avoiding synchronous default-stream copies while routing overlaps.
struct PinnedStatus{ptr:*mut u32}
impl PinnedStatus{
 fn try_new()->Result<Option<Self>>{
  if std::env::var("MGBFS_SEGMENT_TEST_NO_PINNED").as_deref()==Ok("1"){return Ok(None)}
  let mut p=ptr::null_mut();
  let code=unsafe{mgbfs_cuda::native_owner::cudaHostAlloc(&mut p,16,0)};
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
struct Lane{stream:Stream,control:Buffer,spill:[Buffer;2],host:Option<PinnedStatus>}
impl Lane{fn new(device:i32,bound:usize)->Result<Self>{let bytes=bound.checked_mul(4).ok_or("SEGMENTED_SPILL_SIZE")?;Ok(Self{stream:Stream::new(device)?,control:Buffer::new(8,device)?,spill:[Buffer::new(bytes,device)?,Buffer::new(bytes,device)?],host:PinnedStatus::try_new()?})}}
struct OwnerProgress{shard:usize,pass:usize,round:usize,selected:u32}
#[repr(C)] #[derive(Clone,Copy)]
struct BitmapDeviceDomain{symbols:*const i64,counts:*const u32,distinct:u32,elements:u32,total:u64}
struct BitmapState{domain:BitmapDeviceDomain,_symbols:Buffer,_counts:Buffer,bits:RawBuffer,owner:u32,owner_len:u64}
struct DescriptorCache{data:RawBuffer,count:usize}
struct OwnerPrepared{sealed:*const SegmentDescriptor,descriptor_count:u32}
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
struct Comm{ptr:*mut c_void,healthy:bool,aborted:bool}
impl Comm{
 fn abort_once(&mut self){if !self.aborted&&!self.ptr.is_null(){crate::session_cache::permit_comm(false);let status=unsafe{mgbfs_nccl_abort(self.ptr)};self.aborted=true;if status!=0{eprintln!("SEGMENTED_COMM_ABORT_ERROR code={status}");}}}
}
impl Drop for Comm{fn drop(&mut self){unsafe{if !self.ptr.is_null(){if !self.healthy||crate::native_fault::failed(){self.abort_once();}if self.aborted||!crate::session_cache::comm_put(self.ptr){mgbfs_nccl_destroy(self.ptr);}}}}}
#[derive(Clone,Copy)]struct Range{shard:usize,segment:usize,bank:u32,begin:u64,end:u64}
fn mix(mut x:u64)->u64{x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^(x>>31)}

pub struct SegmentedDistributedBfs{
 base:GenericNativeBfs,plan:Plan,rank:u32,seed:u64,bits:u32,layout:SegmentLayout,bitmap:Option<BitmapState>,
 segments:Vec<Vec<Segment>>,descriptor_cache:Vec<Option<DescriptorCache>>,remaining_bytes:u64,storage_budget:u64,configured_rows:u64,bank_totals:[u64;3],capacity_limit:Option<u64>,current:Vec<Range>,previous:Vec<Range>,
 counts:Vec<u64>,depth:u64,source_retries:u64,active_batch:u32,peak_owner_lanes:u32,prefetched_chunks:u64,overlap_samples:u32,
 send:[Queue;2],recv:Queue,route_stream:Stream,route_errors:[Buffer;2],control:Buffer,map:Buffer,cuts:Option<Buffer>,lanes:Vec<Lane>,comm:Comm,test_oom_done:bool,bounds_trace:bool,overlap_enabled:bool,
}
impl SegmentedDistributedBfs{
 pub fn new(g:&GraphDefinitionV2,device:u32,rank:u32,p:Plan,id:&[u8;128],seed:u64,bits:u32,cuts:Option<&[u64]>)->Result<Self>{
  p.validate(g.generator_count() as u32)?;g.validate()?;
  let mut p=p;
  let bitmap_mode=std::env::var("MGBFS_EXPERIMENTAL_BITMAP").as_deref()==Ok("1");
  let bitmap_domain=if bitmap_mode{
   if !matches!(&g.action,mgbfs_core::graph_definition::GraphAction::Permutation{..}){return Err("BITMAP_REQUIRES_PERMUTATION_ACTION".into())}
   if cuts.is_some(){return Err("BITMAP_CUSTOM_OWNER_CUTS_UNSUPPORTED".into())}
   p.history_layers=3;
   Some(crate::permutation_bitmap::MultisetDomain::new(&g.start).map_err(str::to_owned)?)
  }else{None};
  if let Ok(v)=std::env::var("MGBFS_SEGMENT_TEST_QUEUE_CAP"){let n=v.parse::<u32>().map_err(|_|"SEGMENTED_TEST_QUEUE_CAP")?;if n==0||n>p.queue_capacity{return Err("SEGMENTED_TEST_QUEUE_CAP".into())}p.queue_capacity=n;}
  if p.history_algorithm!="HASH"||p.sort_candidates||p.parent_transport||p.generator_backend!="cuda"{return Err("SEGMENTED_UNSUPPORTED_PROFILE".into())}
  if rank>=p.world||bits>64||p.elements as usize!=g.start.len(){return Err("SEGMENTED_PLAN_SHAPE".into())}
  if bitmap_domain.is_none()&&p.history_layers==3&&!g.inverse_closed()?{return Err("SEGMENTED_ROLLING_REQUIRES_INVERSE_CLOSED".into())}
  if let Some(c)=cuts{if c.len()!=p.world as usize+1||c[0]!=0||c[c.len()-1]!=(1u64<<32)||c.windows(2).any(|x|x[0]>=x[1]){return Err("SEGMENTED_OWNER_CUTS".into())}}
  let tiny=GenericMemoryPlan::with_storage(p.elements,1,p.state_bytes)?;
  let base=GenericNativeBfs::new_unseeded(g,device,tiny,seed,bits)?;let d=base.device;let stream=base.stream.ptr;
  let mut raw=crate::session_cache::comm_take().unwrap_or(ptr::null_mut());
  if raw.is_null(){let mut error=[0i8;512];crate::generic_native::checked!(mgbfs_nccl_create,unsafe{mgbfs_nccl_create(rank,p.world,device,id.as_ptr().cast(),&mut raw,error.as_mut_ptr(),error.len())})?;}
  let comm=Comm{ptr:raw,healthy:false,aborted:false};
  let send=[Queue::new(&p,d)?,Queue::new(&p,d)?];let recv=Queue::new(&p,d)?;
  let route_stream=Stream::new(d)?;let route_errors=[Buffer::new(12,d)?,Buffer::new(12,d)?];
  let control=Buffer::new(128+p.world as usize*8,d)?;
  let bound=usize::try_from(u64::from(p.world)*u64::from(p.queue_capacity)).map_err(|_|"SEGMENTED_SPILL_SIZE")?;
  if bound>=0x80000000usize{return Err("SEGMENTED_SPILL_SIZE".into())}
  let mut h=seed;for(e,v)in g.start.iter().enumerate(){h=mix(h^(*v as u64)^(e as u64));}h=mix(h);if bits<64{h&=if bits==0{0}else{(1u64<<bits)-1};}
  if let Some(domain)=&bitmap_domain{h=domain.rank(&g.start).map_err(str::to_owned)?;}
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
  let(mut sizing_free,mut sizing_total)=(0usize,0usize);
  crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut sizing_free,&mut sizing_total)})?;
  let sizing_reserve=crate::generic_memory::admission_reserve_bytes()?;
  let bitmap_bytes=if let Some(domain)=&bitmap_domain{
   let logical=owner_map.iter().position(|&physical|physical==rank).ok_or("BITMAP_OWNER_MAP")? as u32;
   crate::permutation_bitmap::BitmapPartition::new(domain.size(),p.world).map_err(str::to_owned)?.owner_bytes(logical).map_err(str::to_owned)?.max(8)
  }else{0};
  // Leave space for at least one segment per shard and one owner lane.
  let lane_sizing=(bound as u64).checked_mul(8).and_then(|v|v.checked_add(8)).ok_or("SEGMENTED_SPILL_SIZE")?;
  let available=(sizing_free as u64).saturating_sub(sizing_reserve).saturating_sub(bitmap_bytes).saturating_sub(lane_sizing);
  let per_segment=available/u64::from(p.shards);
  let make_layout=|rows|if bitmap_mode{SegmentLayout::bitmap(p.elements,p.state_bytes,rows)}else{SegmentLayout::new(p.elements,p.state_bytes,rows,p.history_layers)};
  let rows=if let Some(exact)=explicit_rows{exact}else{
   let mut lo=0u64;let mut hi=configured_rows.min(estimated);
   while lo<hi{let mid=lo+(hi-lo+1)/2;if make_layout(mid).map_or(false,|layout|layout.device_bytes<=per_segment){lo=mid}else{hi=mid-1}}
   if lo==0{return Err("SEGMENTED_NO_CAPACITY".into())}lo
  };
  let layout=if bitmap_mode{SegmentLayout::bitmap(p.elements,p.state_bytes,rows)}else{SegmentLayout::new(p.elements,p.state_bytes,rows,p.history_layers)}.map_err(str::to_owned)?;
  let bitmap=if let Some(domain)=&bitmap_domain{
   let logical=owner_map.iter().position(|&physical|physical==rank).ok_or("BITMAP_OWNER_MAP")? as u32;
   let partition=crate::permutation_bitmap::BitmapPartition::new(domain.size(),p.world).map_err(str::to_owned)?;
   let owner_len=partition.owner_len(logical).map_err(str::to_owned)?;
   let bytes=partition.owner_bytes(logical).map_err(str::to_owned)?.max(8);
   let(mut free,mut total)=(0usize,0usize);crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
   let reserve=crate::generic_memory::admission_reserve_bytes()?;
   if bytes>(free as u64).saturating_sub(reserve).saturating_sub(layout.device_bytes){return Err("BITMAP_DOMAIN_NO_VRAM".into())}
   let bit_storage=RawBuffer::try_new(usize::try_from(bytes).map_err(|_|"BITMAP_SIZE_OVERFLOW")?,d)?.ok_or("BITMAP_DOMAIN_NO_VRAM")?;
   crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(bit_storage.ptr,0,bit_storage.bytes,stream)})?;
   let symbols=Buffer::new(domain.symbols().len()*8,d)?;symbols.upload(domain.symbols())?;
   let counts=Buffer::new(domain.counts().len()*4,d)?;counts.upload(domain.counts())?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
   Some(BitmapState{domain:BitmapDeviceDomain{symbols:symbols.ptr.cast(),counts:counts.ptr.cast(),distinct:domain.symbols().len() as u32,elements:p.elements,total:domain.size()},_symbols:symbols,_counts:counts,bits:bit_storage,owner:logical,owner_len})
  }else{None};
  // Release every cached extent not already taken by the base and route
  // buffers before computing lane affordability. These extents may belong
  // to the previous job and are reclaimable, but not usable by cudaMalloc.
  crate::session_cache::release_unused_buffers();crate::generic_pool::trim_idle()?;
  let(mut free,mut total)=(0usize,0usize);crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
  let reserve=crate::generic_memory::admission_reserve_bytes()?;
  let lane_bytes=(bound as u64).checked_mul(8).and_then(|v|v.checked_add(8)).ok_or("SEGMENTED_SPILL_SIZE")?;
  let affordable=(free as u64).saturating_sub(reserve).saturating_sub(layout.device_bytes)/lane_bytes;
  let requested=std::env::var("MGBFS_SEGMENT_OWNER_LANES").or_else(|_|std::env::var("MGBFS_SEGMENT_TEST_LANES")).ok().map(|v|v.parse::<u32>().map_err(|_|"SEGMENTED_OWNER_LANES".to_owned())).transpose()?.unwrap_or(4);
  if !(1..=64).contains(&requested){return Err("SEGMENTED_OWNER_LANES".into())}
  let lane_count=(requested as u64).min(u64::from(p.shards)).min(affordable) as usize;
  if lane_count==0{return Err("SEGMENTED_LANE_ADMISSION_FAILED".into())}
  let mut lanes=Vec::with_capacity(lane_count);for _ in 0..lane_count{lanes.push(Lane::new(d,bound)?);}
  eprintln!("SEGMENTED_OWNER_LANES rank={rank} lanes={lane_count}");
  // Live fixed buffers have been taken; parked unmatched buffers cannot
  // serve ordinary cudaMalloc segment storage and are safe to return now.
  crate::session_cache::release_unused_buffers();crate::generic_pool::trim_idle()?;
  crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
  let mut remaining_bytes=(free as u64).saturating_sub(reserve);
  if let Ok(v)=std::env::var("MGBFS_SEGMENT_TEST_LIMIT"){let count=v.parse::<u64>().map_err(|_|"SEGMENTED_TEST_LIMIT")?;remaining_bytes=remaining_bytes.min(count.saturating_mul(layout.device_bytes));}
  let capacity_limit=std::env::var("MGBFS_SEGMENT_MAX_RANK_STATES").ok().map(|v|v.parse::<u64>().map_err(|_|"SEGMENTED_CAPACITY_LIMIT".to_owned())).transpose()?;
  if capacity_limit==Some(0){return Err("SEGMENTED_CAPACITY_LIMIT".into())}
  let mut result=Self{base,plan:p.clone(),rank,seed,bits,layout,bitmap,segments:(0..p.shards).map(|_|Vec::new()).collect(),descriptor_cache:(0..p.shards).map(|_|None).collect(),remaining_bytes,storage_budget:remaining_bytes,configured_rows,bank_totals:[0;3],capacity_limit,current:vec![],previous:vec![],counts:(0..p.world).map(|r|u64::from(r==root)).collect(),depth:0,source_retries:0,active_batch:p.batch,peak_owner_lanes:0,prefetched_chunks:0,overlap_samples:0,send,recv,route_stream,route_errors,control,map,cuts,lanes,comm,test_oom_done:false,bounds_trace:std::env::var("MGBFS_SEGMENT_TEST_BOUNDS_TRACE").as_deref()==Ok("1"),overlap_enabled:std::env::var("MGBFS_SEGMENT_TEST_OVERLAP").as_deref()==Ok("1")};
  let mut failed=0;
  if rank==root{
   let shard_hash=if bitmap_mode{mix(h)}else{h};
   let shard=(((shard_hash as u32 as u64)*u64::from(p.shards))>>32) as usize;
   if !result.make_segment(shard)?{failed=1}else{
    let input=Buffer::new(p.elements as usize*p.state_bytes as usize,d)?;
    if p.state_bytes==1{input.upload(&g.start.iter().map(|&v|v as u8).collect::<Vec<_>>())?;}else{input.upload(&g.start)?;}
    let hashes=Buffer::new(8,d)?;hashes.upload(&[h])?;let seg=&result.segments[shard][0];
    crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(result.control.at::<c_void>(0),0,4,stream)})?;
    let status=unsafe{if let Some(b)=&result.bitmap{
     mgbfs_bitmap_seed(p.state_bytes,input.ptr,p.elements,seg.desc,b.bits.ptr.cast(),h/u64::from(p.world),seg.counts.ptr.cast(),stream)
    }else if p.state_bytes==1{mgbfs_segmented_accept_u8(input.ptr.cast(),hashes.ptr.cast(),1,p.elements,ptr::null(),0,seg.desc,0,seg.counts.ptr.cast(),ptr::null_mut(),result.control.at(0),stream)}else{mgbfs_segmented_accept_i64(input.ptr.cast(),hashes.ptr.cast(),1,p.elements,ptr::null(),0,seg.desc,0,seg.counts.ptr.cast(),ptr::null_mut(),result.control.at(0),stream)}};
    crate::generic_native::checked!(mgbfs_segmented_accept_seed,status)?;crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(stream)})?;
    let mut error=0u32;crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut error as *mut u32).cast(),result.control.ptr,4,2)})?;
    if error!=0{failed=error+1}else{result.segments[shard][0].banks[0]=1;result.bank_totals[0]=1;result.current.push(Range{shard,segment:0,bank:0,begin:0,end:1});}
   }
  }
  if result.vote(failed)?!=0{return Err("SEGMENTED_ROOT_ADMISSION_FAILED".into())}Ok(result)
 }
 fn make_segment(&mut self,shard:usize)->Result<bool>{
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
      None=>{eprintln!("SEGMENTED_TEST_REAL_OOM rank={} depth={} requested={} total={} code=2",self.rank,self.depth,requested,total);return Ok(false)},
      Some(v)=>{drop(v);return Err("SEGMENTED_TEST_OOM_INCONCLUSIVE_ALLOCATION_SUCCEEDED".into())}
     }
    }
   }
  }
  if self.layout.device_bytes>self.remaining_bytes{return Ok(false)}
  if let Some(seg)=Segment::try_new(self.layout,self.base.device,self.base.stream.ptr)?{self.remaining_bytes-=self.layout.device_bytes;self.segments[shard].push(seg);Ok(true)}else{Ok(false)}
 }
 // Called only between drained waves. All descriptors for a shard remain
 // immutable until a segment is appended; no per-chunk cudaMalloc/cudaFree.
 fn descriptor_view(&mut self,shard:usize)->Result<Option<(*const SegmentDescriptor,u32)>>{
  let count=self.segments[shard].len();
  if count<=1{return Ok(Some((ptr::null(),0)))}
  if self.descriptor_cache[shard].as_ref().map_or(true,|c|c.count!=count){
   let descriptors=self.segments[shard].iter().map(|s|s.desc).collect::<Vec<_>>();
   let bytes=count.checked_mul(std::mem::size_of::<SegmentDescriptor>()).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
   let old=self.descriptor_cache[shard].as_ref().map_or(0,|c|c.data.bytes);
   let delta=bytes.checked_sub(old).ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
   if delta as u64>self.remaining_bytes{return Ok(None)}
   // Both allocations coexist briefly. A real allocation failure is a
   // resource stop even when the final delta would fit.
   let data=match RawBuffer::try_new(bytes,self.base.device)?{Some(v)=>v,None=>return Ok(None)};
   crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(data.ptr,descriptors.as_ptr().cast(),bytes,1)})?;
   self.descriptor_cache[shard]=Some(DescriptorCache{data,count});
   self.remaining_bytes-=delta as u64;
  }
  let c=self.descriptor_cache[shard].as_ref().ok_or("SEGMENTED_DESCRIPTOR_MISSING")?;
  Ok(Some((c.data.ptr.cast(),u32::try_from(c.count).map_err(|_|"SEGMENTED_DESCRIPTOR_SIZE")?)))
 }
 fn vote(&self,value:u32)->Result<u32>{let s=self.base.stream.ptr;self.control.upload(&[value])?;
  crate::generic_native::checked!(mgbfs_nccl_all_reduce_max_u32,unsafe{mgbfs_nccl_all_reduce_max_u32(self.comm.ptr,self.control.at(0),self.control.at(4),s)})?;
  let mut v=0u32;crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync((&mut v as *mut u32).cast(),self.control.at::<c_void>(4),4,2,s)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(s)})?;Ok(v)
 }
 // One collective independently reduces the error vote and the actual
 // maximum queue size, so every rank uses identical bounded NCCL messages.
 fn vote_route_and_max(&self,value:u32,local_max:u32)->Result<(u32,u32)>{
  let st=self.base.stream.ptr;self.control.upload(&[value,local_max])?;
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
  let st=self.base.stream.ptr;
  crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(self.control.at::<c_void>(32),(&local as *const u64).cast(),8,1,st)})?;
  crate::generic_native::checked!(mgbfs_nccl_all_gather_u64,unsafe{mgbfs_nccl_all_gather_u64(self.comm.ptr,self.control.at(32),self.control.at(64),st)})?;
  let mut ranks=vec![0u64;self.plan.world as usize];
  crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(ranks.as_mut_ptr().cast(),self.control.at::<c_void>(64),ranks.len()*8,2,st)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
  Ok(ranks.into_iter().max().unwrap_or(0))
 }
 pub fn collective_stop(&mut self,v:u32)->Result<u32>{self.vote(v)}
 pub fn global_frontier(&self)->u64{self.counts.iter().sum()}
 pub fn frontier_bucket(&self)->u32{63-self.counts.iter().copied().max().unwrap_or(0).max(1).leading_zeros()}
 pub fn source_retries(&self)->u64{self.source_retries}
 pub fn peak_owner_lanes(&self)->u32{self.peak_owner_lanes}
 pub fn memory_geometry(&self)->serde_json::Value{serde_json::json!({"rows_per_bank":self.layout.rows_per_bank,"configured_segment_rows":self.configured_rows,"banks":self.layout.banks,"slots_per_segment":self.layout.slots,"bytes_per_segment":self.layout.device_bytes,"segment_count":self.segments.iter().map(Vec::len).sum::<usize>(),"storage_budget_bytes":self.storage_budget,"storage_allocated_bytes":self.storage_budget-self.remaining_bytes,"descriptor_cache_bytes":self.descriptor_cache.iter().filter_map(|c|c.as_ref()).map(|c|c.data.bytes as u64).sum::<u64>(), "capacity_limit_per_rank":self.capacity_limit,"capacity_limit_is_hard":true,"capacity_limit_scope":if self.layout.banks==3{"one next layer per rank"}else{"all visited states per rank"},"owner_lanes":self.lanes.len(),"owner_pinned_metadata":self.lanes.iter().filter(|v|v.host.is_some()).count(),"prefetched_chunks":self.prefetched_chunks,"bitmap_bytes":self.bitmap.as_ref().map_or(0,|b|b.bits.bytes),"bitmap_domain":self.bitmap.as_ref().map(|b|b.domain.total),"bitmap_logical_owner":self.bitmap.as_ref().map(|b|b.owner),"bitmap_owner_rows":self.bitmap.as_ref().map(|b|b.owner_len),"extra_send_bytes":(self.send[1].records.bytes+self.send[1].counts.bytes+self.send[1].states.bytes) as u64,"legacy_plan_capacity_is_not_the_segment_limit":true})}
 pub fn size_profile_events(&self)->Vec<serde_json::Value>{vec![]}
 pub fn set_size_profile(&mut self,batch:u32,gemm:bool)->Result<()>{if batch==0||batch>self.plan.batch||gemm{return Err("SEGMENTED_UNSUPPORTED_SIZE_PROFILE".into())}self.active_batch=batch;Ok(())}
 pub fn quiesce(&self)->Result<()>{
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.route_stream.ptr)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;
  for lane in &self.lanes{crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(lane.stream.ptr)})?;}Ok(())
 }
 pub fn mark_healthy(&mut self){self.comm.healthy=true;}
}
// Abort before GenericNativeBfs and its stream fields are dropped. Ordinary
// Rust errors must not put a potentially poisoned communicator in the cache.
impl Drop for SegmentedDistributedBfs{fn drop(&mut self){
 // A prefetched route may still read the old frontier after an error. Drain it
 // before fields containing that frontier or its output queues are dropped.
 unsafe{cudaStreamSynchronize(self.route_stream.ptr);}
 if !self.comm.healthy||crate::native_fault::failed(){self.comm.abort_once();}
}}

impl SegmentedDistributedBfs {
 // The main stream and all owner lanes are drained before retiring or
 // rebuilding a segment. A shard has exactly one accepting lane at a time.
 fn maintain(&mut self,bank:u32)->Result<u32>{
  let st=self.base.stream.ptr;let error=self.control.at::<u32>(8);
  if self.bitmap.is_some(){
   for bucket in &mut self.segments{for seg in bucket{if seg.banks[bank as usize]!=0{seg.banks[bank as usize]=0;crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(seg.counts.at::<c_void>(bank as usize*8),0,8,st)})?;}}}
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;self.bank_totals[bank as usize]=0;return Ok(0)
  }
  crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(error.cast(),0,4,st)})?;
  for bucket in &mut self.segments{for seg in bucket{
   let count=if self.layout.banks==3{seg.banks[bank as usize]}else{0};if count!=0{
    crate::generic_native::checked!(mgbfs_segmented_retire,unsafe{mgbfs_segmented_retire(seg.desc,bank,count,error,st)})?;
    seg.retired+=u64::from(count);seg.banks[bank as usize]=0;
    crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(seg.counts.at::<c_void>(bank as usize*8),0,8,st)})?;
   }
   if seg.retired>=seg.desc.slots_count/4{
    let c=seg.banks;let status=unsafe{if self.plan.state_bytes==1{mgbfs_segmented_rehash_u8(self.plan.elements,seg.desc,c[0],c[1],c[2],self.seed,self.bits,error,st)}else{mgbfs_segmented_rehash_i64(self.plan.elements,seg.desc,c[0],c[1],c[2],self.seed,self.bits,error,st)}};
    crate::generic_native::checked!(mgbfs_segmented_rehash,status)?;seg.retired=0;
   }
  }}
  let mut x=0u32;crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync((&mut x as *mut u32).cast(),error.cast(),4,2,st)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
  if x==0&&self.layout.banks==3{self.bank_totals[bank as usize]=0;}Ok(x)
 }
 fn accept_parallel(&mut self,slot:usize,shards:&[usize],bank:u32,visible:u32,mut probe:Option<&mut OverlapProbe>)->Result<u32>{
  let p=self.plan.clone();let selected_max=p.world.checked_mul(visible).ok_or("SEGMENTED_QUEUE_EXTENT")?;
  for wave in shards.chunks(self.lanes.len()){
   self.peak_owner_lanes=self.peak_owner_lanes.max(wave.len() as u32);
   let mut progress=wave.iter().map(|&shard|{
    // A full future bank cannot accept a new row. Its exact table
    // remains in the sealed descriptor view for duplicate lookups.
    let pass=self.segments[shard].iter().position(|v|v.banks[bank as usize]<self.layout.rows_per_bank).unwrap_or(self.segments[shard].len());
    OwnerProgress{shard,pass,round:0,selected:selected_max}
   }).collect::<Vec<_>>();
   while progress.iter().any(|v|v.selected!=0){
    // Prepare all descriptors and segment allocations before starting any lane.
    // cudaMemsetAsync initializes new segments on the main stream; that stream
    // must reach the barrier below before another stream observes them.
    let mut prepared:Vec<Option<OwnerPrepared>>=(0..progress.len()).map(|_|None).collect();
    for (i,v) in progress.iter().enumerate(){
     if v.selected==0{continue}
     if v.pass==self.segments[v.shard].len()&&!self.make_segment(v.shard)?{return Ok(10)}
     let (sealed,descriptor_count)=if self.bitmap.is_some(){(ptr::null(),0)}else{match self.descriptor_view(v.shard)?{Some(v)=>v,None=>return Ok(10)}};
     prepared[i]=Some(OwnerPrepared{sealed,descriptor_count});
    }
    crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;
    let mut launched=vec![false;progress.len()];
    let launch_result=(||->Result<()>{
     for (i,v) in progress.iter().enumerate(){
      if v.selected==0{continue}
      let lane=&self.lanes[i];let st=lane.stream.ptr;let active=&self.segments[v.shard][v.pass];let d=prepared[i].as_ref().ok_or("SEGMENTED_DESCRIPTOR_MISSING")?;
      let selection=if v.round==0{ptr::null()}else{lane.spill[(v.round-1)%2].ptr.cast()};
      let sealed=d.sealed;
      crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(lane.control.ptr,0,8,st)})?;
      if i==0&&v.round==0{if let Some(q)=probe.as_deref_mut(){q.start_owner(st)?;}}
      let status=unsafe{if let Some(b)=&self.bitmap{
       mgbfs_bitmap_accept(p.state_bytes,u32::from(p.packed_candidates()),p.elements,self.send[slot].states.ptr,self.recv.states.ptr,self.send[slot].records.ptr.cast(),self.recv.records.ptr.cast(),self.send[slot].counts.ptr.cast(),self.recv.counts.ptr.cast(),self.rank,p.world,b.owner,b.domain.total,b.owner_len,b.bits.ptr.cast(),v.shard as u32,p.shards,p.queue_capacity,visible,selection,v.selected,active.desc,bank,active.counts.at(bank as usize*8),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)
      }else if p.state_bytes==1{
       mgbfs_segmented_accept_routes_limited_u8(p.elements,self.send[slot].states.ptr.cast(),self.recv.states.ptr.cast(),self.send[slot].records.ptr.cast(),self.recv.records.ptr.cast(),self.send[slot].counts.ptr.cast(),self.recv.counts.ptr.cast(),self.rank,p.world,v.shard as u32,p.shards,p.queue_capacity,visible,u32::from(p.packed_candidates()),selection,v.selected,sealed,d.descriptor_count,active.desc,bank,active.counts.at(bank as usize*8),ptr::null_mut(),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)
      }else{
       mgbfs_segmented_accept_routes_limited_i64(p.elements,self.send[slot].states.ptr.cast(),self.recv.states.ptr.cast(),self.send[slot].records.ptr.cast(),self.recv.records.ptr.cast(),self.send[slot].counts.ptr.cast(),self.recv.counts.ptr.cast(),self.rank,p.world,v.shard as u32,p.shards,p.queue_capacity,visible,selection,v.selected,sealed,d.descriptor_count,active.desc,bank,active.counts.at(bank as usize*8),ptr::null_mut(),lane.spill[v.round%2].ptr.cast(),lane.control.at(0),lane.control.at(4),if self.capacity_limit.is_some(){self.control.at(24)}else{ptr::null_mut()},self.capacity_limit.unwrap_or(0),st)
      }};
      launched[i]=true;crate::generic_native::checked!(mgbfs_segmented_accept_routes,status)?;
      if i==0&&v.round==0{if let Some(q)=probe.as_deref_mut(){q.end_owner(st)?;}}
      if let Some(host)=&lane.host{
       crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(host.ptr.cast(),lane.control.ptr,8,2,st)})?;
       crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(host.ptr.add(2).cast(),active.counts.at::<c_void>(bank as usize*8),8,2,st)})?;
      }
     }Ok(())
    })();
    // Drain every launched lane even if another launch failed, before its
    // descriptors or input queues can be freed or recycled.
    let mut failure=launch_result.err();
    for (i,lane) in self.lanes.iter().enumerate().take(progress.len()){
     if launched[i]{if let Err(e)=crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(lane.stream.ptr)}){if failure.is_none(){failure=Some(e)}}}
    }
    if let Some(e)=failure{return Err(e)}
    let mut owner_error=0;
    for (i,v) in progress.iter_mut().enumerate(){
     if !launched[i]{continue}
     let lane=&self.lanes[i];let active=&self.segments[v.shard][v.pass];let mut values=[0u32;4];
     if let Some(host)=&lane.host{unsafe{ptr::copy_nonoverlapping(host.ptr,values.as_mut_ptr(),4);}}
     else{
      crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),lane.control.ptr,8,2)})?;
      crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut values[2] as *mut u32).cast(),active.counts.at::<c_void>(bank as usize*8),8,2)})?;
     }
     let accepted=u64::from(values[2])|(u64::from(values[3])<<32);
     if values[0]>v.selected||accepted>self.layout.rows_per_bank||(values[1]&!8)!=0{owner_error=owner_error.max(100+(values[1]&!8).max(1));continue}
     if values[1]&8!=0{
      let mut observed=0u64;crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut observed as *mut u64).cast(),self.control.at::<c_void>(24),8,2)})?;
      if let Some(limit)=self.capacity_limit{
       eprintln!("SEGMENTED_CAPACITY_OBSERVED rank={} limit={} count={}",self.rank,limit,observed);
       if observed>limit{owner_error=owner_error.max(199)}else{owner_error=owner_error.max(10)}
      }else{owner_error=owner_error.max(199)}continue
     }
     let seg=&mut self.segments[v.shard][v.pass];let old=seg.banks[bank as usize];
     if accepted<old{owner_error=owner_error.max(101);continue}
     self.bank_totals[bank as usize]=self.bank_totals[bank as usize].checked_add(accepted-old).ok_or("SEGMENTED_BANK_TOTAL_OVERFLOW")?;
     seg.banks[bank as usize]=accepted;
     // Upper bound on possible tombstones, used for conservative rehashing.
     seg.retired=seg.retired.saturating_add(u64::from(values[0]));
     v.selected=values[0];v.pass+=1;v.round+=1;
    }
    if owner_error!=0{return Ok(owner_error)}
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
   let stride=seg.desc.capacity*u64::from(seg.desc.banks);let off=(u64::from(r.bank)*u64::from(seg.desc.capacity)+u64::from(cursor))*u64::from(p.state_bytes);
   let parent=unsafe{seg.desc.states.cast::<u8>().add(usize::try_from(off).map_err(|_|"SEGMENTED_PARENT_OFFSET")?)};
   if let Some(b)=&self.bitmap{
    crate::generic_native::checked!(mgbfs_bitmap_route,unsafe{mgbfs_bitmap_route(p.state_bytes,b.domain,p.elements,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),st)})?;
   }else{
    crate::generic_native::checked!(mgbfs_segmented_route_storage,unsafe{mgbfs_segmented_route_storage(p.state_bytes,self.base.kind,p.elements,self.base.rows,self.base.cols,self.base.generators,parent.cast(),count,stride,self.base.perms.ptr.cast(),self.base.matrices.ptr.cast(),self.base.moduli.ptr.cast(),self.seed,self.bits,p.world,self.rank,p.shards,p.queue_capacity,0,self.map.ptr.cast(),self.cuts.as_ref().map_or(ptr::null(),|v|v.ptr.cast()),send.records.ptr.cast(),send.counts.ptr.cast(),errors.at(0),st)})?;
   }
   if !p.packed_candidates(){
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
 pub fn advance(&mut self)->Result<DistributedAdvance>{
  if let (Ok(r),Ok(d))=(std::env::var("MGBFS_SEGMENT_TEST_FATAL_RANK"),std::env::var("MGBFS_SEGMENT_TEST_FATAL_AFTER_DEPTH")){
   if r.parse::<u32>().ok()==Some(self.rank)&&d.parse::<u64>().ok().map_or(false,|v|self.depth>=v){return Err("SEGMENTED_TEST_ONE_RANK_FAILURE".into())}
  }
  let st=self.base.stream.ptr;let p=self.plan.clone();let future=if self.layout.banks==3{((self.depth+1)%3) as u32}else{0};
  let tracing=std::env::var("MGBFS_SEGMENT_TEST_TIMING").as_deref()==Ok("1");let mut tick=std::time::Instant::now();let mut phases=[0u128;7];
  macro_rules! record{($i:expr)=>{if tracing{phases[$i]+=tick.elapsed().as_micros();tick=std::time::Instant::now();}}}
  let maintenance=self.maintain(future)?;let vote=self.vote(if maintenance==0{0}else{100+maintenance})?;
  if self.capacity_limit.is_some(){
   let initial=self.bank_totals[future as usize];
   crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(self.control.at::<c_void>(24),(&initial as *const u64).cast(),8,1)})?;
  }
  if vote!=0{return Err(format!("SEGMENTED_MAINTENANCE_{vote}"))}
  let before=self.segments.iter().map(|bucket|bucket.iter().map(|s|s.banks[0]).collect::<Vec<_>>()).collect::<Vec<_>>();
  let work=self.current.clone();let mut wi=0usize;let mut cursor=work.first().map_or(0,|r|r.begin);let mut batch=self.active_batch.max(1);
  let mut rounds=self.global_rounds(&work,wi,cursor,batch)?;let mut pending_owner=0u32;let mut slot=0usize;record!(0);
  if rounds>0{self.start_route(slot,&work,wi,cursor,batch)?;}
  loop{
   if rounds==0{
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
   let (source,global_max)=self.vote_route_and_max(crate::segmented_history::route_owner_vote(classified,pending_owner),source_errors[2])?;pending_owner=0;record!(2);
   match crate::segmented_history::decode_route_owner_vote(source){
    crate::segmented_history::SegmentVote::Continue=>{},
    crate::segmented_history::SegmentVote::Retry if batch>1=>{
     batch=(batch/2).max(1);self.source_retries+=1;rounds=self.global_rounds(&work,wi,cursor,batch)?;
     self.start_route(slot,&work,wi,cursor,batch)?;continue
    },
    crate::segmented_history::SegmentVote::Retry=>return Err("SEGMENTED_ROUTE_1".into()),
    crate::segmented_history::SegmentVote::Resource=>return Ok(DistributedAdvance::Resource{fatal:10}),
    crate::segmented_history::SegmentVote::OwnerFatal(e)=>return Err(format!("SEGMENTED_OWNER_{e}")),
    crate::segmented_history::SegmentVote::SourceFatal(e)=>return Err(format!("SEGMENTED_ROUTE_{e}")),
   }
   let upper=(u64::from(batch)*u64::from(self.base.generators)).min(u64::from(p.queue_capacity)) as u32;
   if global_max>upper{return Err("SEGMENTED_ROUTE_COUNT_BOUND".into())}
   let used=global_max;
   if self.bounds_trace{eprintln!("SEGMENT_BOUNDS rank={} depth={} local={} global={} upper={}",self.rank,self.depth,source_errors[2],global_max,upper);}
   let send=&self.send[slot];
   crate::generic_native::checked!(mgbfs_nccl_exchange_bounded_triplets,unsafe{mgbfs_nccl_exchange_bounded_triplets(self.comm.ptr,self.rank,p.world,p.shards,p.queue_capacity,used,p.queue_payload_bytes() as u64,p.state_bytes,send.counts.ptr,send.records.ptr,send.states.ptr,self.recv.counts.ptr,self.recv.records.ptr,self.recv.states.ptr,st)})?;
   let n=p.world as usize*p.shards as usize;let mut lc=vec![0u32;n];let mut rc=vec![0u32;n];
   crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(lc.as_mut_ptr().cast(),send.counts.ptr,n*4,2,st)})?;
   crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(rc.as_mut_ptr().cast(),self.recv.counts.ptr,n*4,2,st)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
   record!(3);let active=(0..p.shards as usize).filter(|&shard|
    (0..p.world as usize).any(|src|if src==self.rank as usize{lc[src*p.shards as usize+shard]!=0}else{rc[src*p.shards as usize+shard]!=0})
   ).collect::<Vec<_>>();
   if wi<work.len(){cursor=cursor.saturating_add(u64::from(batch)).min(work[wi].end);}
   rounds=rounds.checked_sub(1).ok_or("SEGMENTED_ROUND_COUNT_UNDERFLOW")?;
   while wi<work.len()&&cursor>=work[wi].end{wi+=1;if wi<work.len(){cursor=work[wi].begin;}}
   // Start next generation before waiting for this chunk's owner kernels.
   // A later resource/fatal vote discards the prefetched, unpublished slot.
   let probe_enabled=self.overlap_enabled&&self.depth>=10&&rounds>0&&wi<work.len()&&!active.is_empty()&&self.overlap_samples<16;
   let mut probe=if probe_enabled{Some(OverlapProbe::new(self.base.device)?)}else{None};
   if let Some(q)=probe.as_ref(){q.start_route(self.route_stream.ptr)?;}
   if rounds>0{self.start_route(1-slot,&work,wi,cursor,batch)?;self.prefetched_chunks+=1;}
   if let Some(q)=probe.as_ref(){q.end_route(self.route_stream.ptr)?;}
   pending_owner=self.accept_parallel(slot,&active,future,used,probe.as_mut())?;
   if let Some(q)=probe.as_ref(){q.report(self.rank,self.depth)?;self.overlap_samples+=1;}
   record!(4);
   slot=1-slot;
  }
  let mut next=Vec::new();let mut local=0u64;
  for(shard,bucket)in self.segments.iter().enumerate(){for(index,seg)in bucket.iter().enumerate(){
   let end=seg.banks[future as usize];let begin=if self.layout.banks==3{0}else{before.get(shard).and_then(|b|b.get(index)).copied().unwrap_or(0)};
   if end<begin{return Err("SEGMENTED_COUNT_DECREASED".into())}if end>begin{next.push(Range{shard,segment:index,bank:future,begin,end});local=local.checked_add(u64::from(end-begin)).ok_or("SEGMENTED_LOCAL_COUNT_OVERFLOW")?;}
  }}
  crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(self.control.at::<c_void>(32),(&local as *const u64).cast(),8,1,st)})?;
  crate::generic_native::checked!(mgbfs_nccl_all_gather_u64,unsafe{mgbfs_nccl_all_gather_u64(self.comm.ptr,self.control.at(32),self.control.at(64),st)})?;
  let mut counts=vec![0u64;p.world as usize];crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(counts.as_mut_ptr().cast(),self.control.at::<c_void>(64),counts.len()*8,2,st)})?;
  crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(st)})?;
  let global=counts.iter().try_fold(0u64,|a,&b|a.checked_add(b).ok_or("SEGMENTED_GLOBAL_COUNT_OVERFLOW"))?;record!(6);
  if tracing{eprintln!("SEGMENT_TIMING rank={} depth={} us={phases:?}",self.rank,self.depth);}
  if global==0{return Ok(DistributedAdvance::Complete)}
  self.previous=std::mem::replace(&mut self.current,next);self.counts=counts;self.depth+=1;
  Ok(DistributedAdvance::Layer{local,global})
 }
 fn sample_ranges(&self,ranges:&[Range],limit:u32)->Result<Vec<Vec<i64>>>{
  let width=self.plan.elements as usize;let bytes=self.plan.state_bytes as usize;let mut result=Vec::new();
  for r in ranges{let seg=&self.segments[r.shard][r.segment];let available=(r.end-r.begin) as usize;let n=available.min(limit as usize-result.len());if n==0{break}
   let stride=seg.desc.capacity as usize*seg.desc.banks as usize;let start=r.bank as usize*seg.desc.capacity as usize+r.begin as usize;
   let mut rows=vec![vec![0i64;width];n];for e in 0..width{let offset=(e*stride+start)*bytes;let from=unsafe{seg.desc.states.cast::<u8>().add(offset)};
    if bytes==1{let mut values=vec![0u8;n];crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),from.cast(),n,2)})?;for i in 0..n{rows[i][e]=i64::from(values[i]);}}
    else{let mut values=vec![0i64;n];crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),from.cast(),n*8,2)})?;for i in 0..n{rows[i][e]=values[i];}}
   }result.extend(rows);
  }Ok(result)
 }
 pub fn sample_global(&self,limit:u32)->Result<Vec<Vec<i64>>>{let preceding=self.counts[..self.rank as usize].iter().sum::<u64>();let quota=u64::from(limit).saturating_sub(preceding).min(self.counts[self.rank as usize]) as u32;self.sample_ranges(&self.current,quota)}
 pub fn previous_small_sample(&self,previous:u64)->Result<Vec<Vec<i64>>>{if previous>=1000{Ok(vec![])}else{self.sample_ranges(&self.previous,previous as u32)}}
}
