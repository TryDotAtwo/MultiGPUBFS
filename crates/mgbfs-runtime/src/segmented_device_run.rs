//! Experimental resident pack4 entry used by the normal rank runner.
//! The rank supervisor owns cancellation and never retries a partial job.
use super::*;
use crate::resident_extent_validation::{AllocationRegion,ExtentAllocation,validate_extent_allocations,validate_extent_allocations_with_width};
use std::{mem,time::{Duration,Instant},sync::atomic::{AtomicU32,Ordering}};
use super::resident_ffi::{Args,Extent,Functions,HashConfig,HashExtent,HashFunctions,Report};

pub(crate) struct ResidentResult {
 pub sizes:Vec<u64>,pub status:&'static str,pub reason:&'static str,
 pub bfs_seconds:f64,pub setup_seconds:f64,pub evidence:serde_json::Value,
}
struct Event(*mut c_void);
impl Event {
 fn new()->Result<Self>{
  let mut p=ptr::null_mut();
  crate::generic_native::checked!(cudaEventCreateWithFlags,unsafe{cudaEventCreateWithFlags(&mut p,2)})?;
  Ok(Self(p))
 }
 fn wait(&self,stream:*mut c_void,comm:*mut c_void,host:*mut u32,startup:bool)->Result<()> {
  crate::generic_native::checked!(cudaEventRecord,unsafe{cudaEventRecord(self.0,stream)})?;
  let start=Instant::now();
  loop {
   let status=unsafe{cudaEventQuery(self.0)};
   if status==0{return Ok(())}
   // cudaErrorNotReady in CUDA runtime driver_types.h.
   if status!=600 {
    unsafe{(&*(host as *const AtomicU32)).fetch_or(1,Ordering::Release)};
    return crate::generic_native::checked!(cudaEventQuery,status)
   }
   let cancelled=crate::generic_run::CANCELLED.load(Ordering::Relaxed)||crate::job_phase::cancelled();
   let poll=unsafe{mgbfs_nccl_poll(comm)};
   if cancelled||!matches!(poll,0|4)||(startup&&start.elapsed()>Duration::from_secs(60)) {
    unsafe{(&*(host as *const AtomicU32)).fetch_or(1,Ordering::Release)};
    let message=format!("RESIDENT_WAIT_FAILED cancelled={cancelled} nccl_poll={poll} startup={startup}");
    crate::native_fault::fatal_host("resident-wait",&message);
    return Err(message)
   }
   std::thread::sleep(Duration::from_millis(5));
  }
 }
}
impl Drop for Event{fn drop(&mut self){unsafe{cudaEventDestroy(self.0);}}}

struct Resources {
 layers:Buffer,rank_counts:Buffer,rank_count:Buffer,locks:Buffer,descriptors:Buffer,leases:Buffer,report:Buffer,smoke:Buffer,
 mapping:Vec<(usize,usize)>,args:Args,event:Event,window_bytes:u64,layer_capacity:u64,
 _hash_config:Option<Buffer>,hash_stats:Option<Buffer>,hash_index:Option<Buffer>,hash_parent_index:Option<Buffer>,
}

fn checked_size(n:u64,item:usize)->Result<usize>{
 usize::try_from(n).ok().and_then(|x|x.checked_mul(item)).ok_or("RESIDENT_ALLOCATION_SIZE".into())
}
fn region(ptr:*mut c_void,bytes:usize,alignment:u64)->Option<AllocationRegion>{
 if bytes==0{None}else{Some(AllocationRegion{base:ptr as usize as u64,bytes:bytes as u64,alignment})}
}
fn code(name:&str,value:i32)->Result<()>{
 if value==0{Ok(())}else{Err(format!("{name} code={value}"))}
}

impl SegmentedDistributedBfs {
 pub(crate) fn run_resident(&mut self,ready:&crate::segmented_host_ready::HostAgreement,test_depth:Option<u32>)->Result<ResidentResult>{
  let start=Instant::now();
  let f=unsafe{Functions::load()}?;
  let world=self.plan.world;let rank=self.rank;
  let hash=self.resident_hash.is_some();
  if self.layout.banks!=3||self.strategy!=SegmentStrategy::Atomic||
     (hash&&(self.pack4.is_some()||self.bitmap.is_some()))||
     (!hash&&(self.pack4.is_none()||self.bitmap.is_none())) {
   return Err("RESIDENT_REQUIRES_ATOMIC_THREE_BANKS".into())
  }
  let hf=if hash{Some(unsafe{HashFunctions::load()}?)}else{None};
  let initialize=hf.map_or(f.initialize,|h|h.initialize);
  let peer_smoke=hf.map_or(f.peer_smoke,|h|h.peer_smoke);
  let launch=hf.map_or(f.launch,|h|h.launch);
  let domain_total=if let Some((_,bound,_,_,_))=self.resident_hash {bound}
                   else {self.bitmap.as_ref().ok_or("RESIDENT_BITMAP_MISSING")?.domain.total};
  let hash_parallel=self.resident_hash.map_or(false,|v|v.2);
  let hash_instrumented=self.resident_hash.map_or(false,|v|v.3);
  let hash_indexed=self.resident_hash.map_or(false,|v|v.4);
  let geometry:Result<_>=(||{
   let requested=match std::env::var("MGBFS_RESIDENT_MAILBOX_SLOTS"){
    Ok(v)=>v.parse::<u32>().map_err(|_|"RESIDENT_MAILBOX_SLOTS_INVALID".to_owned())?,
    Err(std::env::VarError::NotPresent)=>1,
    Err(_)=>return Err("RESIDENT_MAILBOX_SLOTS_INVALID".into()),
   };
   let slots=resident_ffi::resident_slot_count(unsafe{(f.query)(8)},requested)?;
   if let Some(h)=hf {resident_ffi::resident_slot_count(unsafe{(h.query)(6)},requested)?;}
   let parent_index=if let Some(h)=hf {
    let value=match std::env::var("MGBFS_RESIDENT_HASH_PARENT_INDEX"){
     Ok(v)=>v,Err(std::env::VarError::NotPresent)=>"0".to_owned(),
     Err(_)=>return Err("RESIDENT_HASH_PARENT_INDEX_INVALID".into()),
    };
    resident_ffi::resident_hash_parent_index(unsafe{(h.query)(7)},&value)?
   }else{false};
   let (m,bytes)=f.mailbox(world,slots,u64::MAX)?;
   Ok((m,bytes,parent_index))
  })();
  let geometry_identity=geometry.as_ref().ok().map(|(m,bytes,parent_index)|serde_json::json!({
   "world":m.world,"lanes":m.lanes,"slots_per_ring":m.slots_per_ring,
   "payload_bytes":m.requested_payload_bytes,"payload_stride_bytes":m.payload_stride_bytes,
   "total_bytes":m.total_bytes,"window_bytes":bytes,"hash_parent_index":parent_index,
  }));
  ready.agree("resident-mailbox-layout",&geometry,geometry_identity.as_ref())?;
  let (mailbox,window_bytes,hash_parent_index_enabled)=geometry?;
  let mut error=[0i8;512];
  let preparation=code("RESIDENT_WINDOW_PREPARE",unsafe{(f.prepare)(self.comm.ptr,window_bytes,error.as_mut_ptr(),error.len())})
   .map_err(|e|format!("{e} {}",unsafe{std::ffi::CStr::from_ptr(error.as_ptr())}.to_string_lossy()));
  ready.agree("resident-window-prepare",&preparation,None)?;preparation?;
  let activation=code("RESIDENT_WINDOW_ACTIVATE",unsafe{(f.activate)(self.comm.ptr,error.as_mut_ptr(),error.len())})
   .map_err(|e|format!("{e} {}",unsafe{std::ffi::CStr::from_ptr(error.as_ptr())}.to_string_lossy()));
  ready.agree("resident-window-activate",&activation,None)?;activation?;

  // All ranks use the same capacity for the layer log. The default is a
  // disclosed admission fraction rather than an unexplained layer-count cap.
  let proposal:Result<u64>=(||{
   let(mut free,mut total)=(0usize,0usize);
   crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
   let domain=domain_total;
   let allowed=(free as u64).saturating_sub(self.reserve_bytes);
   let default=(allowed/8192).max(1);
   let wanted=match std::env::var("MGBFS_RESIDENT_LAYER_CAPACITY"){
    Ok(v)=>v.parse::<u64>().map_err(|_|"RESIDENT_LAYER_CAPACITY")?,
    Err(std::env::VarError::NotPresent)=>default,
    Err(_)=>return Err("RESIDENT_LAYER_CAPACITY".into()),
   };
   if wanted==0{return Err("RESIDENT_LAYER_CAPACITY".into())}
   let mut n=domain.min(wanted);
   if let Some(depth)=test_depth {n=n.min(u64::from(depth)+1);}
   if n.checked_mul(8).ok_or("RESIDENT_LAYER_CAPACITY")?>allowed{return Err("RESIDENT_LAYER_LOG_NO_VRAM".into())}
   crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(self.control.at::<c_void>(32),(&n as *const u64).cast(),8,1,self.base.stream.ptr)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;
   Ok(n)
  })();
  ready.agree("resident-log-proposal",&proposal,None)?;proposal?;
  crate::generic_native::checked!(mgbfs_nccl_all_gather_u64,unsafe{mgbfs_nccl_all_gather_u64(self.comm.ptr,self.control.at(32),self.control.at(64),self.base.stream.ptr)})?;
  let mut candidates=vec![0u64;world as usize];
  let gathered:Result<_>=(||{
   crate::generic_native::checked!(cudaMemcpyAsync,unsafe{cudaMemcpyAsync(candidates.as_mut_ptr().cast(),self.control.at::<c_void>(64),candidates.len()*8,2,self.base.stream.ptr)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;
   if candidates.iter().any(|&v|v==0){return Err("RESIDENT_LAYER_LOG_CONSENSUS".into())}
   Ok(())
  })();
  ready.agree("resident-log-gather",&gathered,None)?;gathered?;
  let layer_capacity=*candidates.iter().min().ok_or("RESIDENT_LAYER_LOG_CONSENSUS")?;

  let resource:Result<Resources>=(||{
   let d=self.base.device;
   let layers=Buffer::new(checked_size(layer_capacity,8)?,d)?;
   let rank_counts=Buffer::new(checked_size(u64::from(world),8)?,d)?;
   let rank_count=Buffer::new(8,d)?;
   let locks=Buffer::new(checked_size(u64::from(self.plan.shards),4)?,d)?;
   let report=Buffer::new(mem::size_of::<Report>(),d)?;
   let smoke=Buffer::new(4,d)?;
   let event=Event::new()?;

   // Every unallocated extent is leaseable by any shard. The count below is
   // sufficient for the domain (or three cap-bounded live banks) including
   // segment-rounding by all potentially active shards, if VRAM admits it.
   // Any hash rank may own the entire reachable domain; using domain/world
   // would turn skew into an artificial per-rank limit.
   let owner_len=if hash{domain_total}else{self.bitmap.as_ref().unwrap().owner_len};
   let target=owner_len.min(self.capacity_limit.map_or(owner_len,|c|c.saturating_mul(3)));
   let active=u64::from(self.plan.shards).min(target);
   let rows=self.layout.rows_per_bank;
   let needed=if target==0{0}else{target.checked_add(rows-1).ok_or("RESIDENT_EXTENT_COUNT")?/rows+active-1};
   let mut present=self.segments.iter().map(Vec::len).sum::<usize>() as u64;
   while present<needed {
    let future=present.checked_add(1).ok_or("RESIDENT_EXTENT_COUNT")?;
    let extent_bytes=if hash{mem::size_of::<HashExtent>()}else{mem::size_of::<Extent>()};
    let per_extent=(extent_bytes+4) as u64+if hash_indexed{8}else{0}+
      if hash_parent_index_enabled{8}else{0};
    let per_shard=if hash_indexed{u64::from(self.plan.shards).checked_mul(16).ok_or("RESIDENT_EXTENT_COUNT")?}else{0};
    let metadata=future.checked_mul(per_extent)
      .and_then(|v|v.checked_add(per_shard))
      .and_then(|v|v.checked_add(4096)).ok_or("RESIDENT_EXTENT_COUNT")?;
    let prior=self.reserve_bytes;
    self.reserve_bytes=prior.checked_add(metadata).ok_or("RESIDENT_EXTENT_COUNT")?;
    let allocated=self.make_segment_bounded(0,rows);
    self.reserve_bytes=prior;
    if !allocated?{break}
    present=future;
   }

   let mut host=Vec::new();let mut hash_host=Vec::new();let mut owners=Vec::new();let mut mapping=Vec::new();let mut allocations=Vec::new();
   for (shard,bucket) in self.segments.iter().enumerate(){
    for (index,s) in bucket.iter().enumerate(){
     let owner=if s.banks.iter().any(|&n|n!=0){u32::try_from(shard).map_err(|_|"RESIDENT_SHARD")?}else{u32::MAX};
     host.push(Extent{rows:s.desc.states.cast(),counts:s.counts.ptr.cast(),capacity:s.desc.capacity,banks:s.desc.banks,reserved:0});
     hash_host.push(HashExtent{segment:s.desc,counts:s.counts.ptr.cast()});
     owners.push(owner);mapping.push((shard,index));
     allocations.push(ExtentAllocation{rows:s.desc.states as usize as u64,row_bytes:s._states.bytes as u64,
      counts:s.counts.ptr as usize as u64,count_bytes:s.counts.bytes as u64,capacity:s.desc.capacity,banks:s.desc.banks});
    }
   }
   let n=u64::try_from(host.len()).map_err(|_|"RESIDENT_EXTENT_COUNT")?;
   if hash&&n==0{return Err("RESIDENT_HASH_NO_ADMITTED_EXTENT".into())}
   let extent_size=if hash{mem::size_of::<HashExtent>()}else{mem::size_of::<Extent>()};
   let descriptors=Buffer::new(checked_size(n,extent_size)?,d)?;
   let leases=Buffer::new(checked_size(n,4)?,d)?;
   if hash{descriptors.upload(&hash_host)?}else{descriptors.upload(&host)?}
   leases.upload(&owners)?;
   let hash_stats=if hash&&hash_instrumented {
    let b=Buffer::new(20,d)?;b.upload(&[0u32;5])?;Some(b)
   }else{None};
   let shards=u64::from(self.plan.shards);
   let next_offset=checked_size(shards,8)?;
   let cursor_offset=checked_size(shards.checked_add(n).ok_or("RESIDENT_HASH_INDEX_SIZE")?,8)?;
   let hash_index=if hash&&hash_indexed {
    let count=n.checked_add(shards.checked_mul(2).ok_or("RESIDENT_HASH_INDEX_SIZE")?)
      .ok_or("RESIDENT_HASH_INDEX_SIZE")?;
    Some(Buffer::new(checked_size(count,8)?,d)?)
   }else{None};
   let hash_parent_index=if hash_parent_index_enabled {
    let count=n.checked_add(1).ok_or("RESIDENT_HASH_PARENT_INDEX_SIZE")?;
    let bytes=checked_size(count,8)?;
    let(mut free,mut total)=(0usize,0usize);
    crate::generic_native::checked!(cudaMemGetInfo,unsafe{mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free,&mut total)})?;
    if (free as u64).saturating_sub(self.reserve_bytes)<bytes as u64 {
     return Err("RESIDENT_HASH_PARENT_INDEX_BUDGET".into())
    }
    Some(Buffer::new(bytes,d)?)
   }else{None};
   let hash_config=if hash {
    let b=Buffer::new(mem::size_of::<HashConfig>(),d)?;
    let config=HashConfig{seed:self.seed,bits:self.bits,state_bytes:self.plan.state_bytes,
      owner_cuts:self.cuts.as_ref().map_or(ptr::null(),|c|c.ptr.cast()),
      parallel_owners:u32::from(hash_parallel),owner_peak:hash_stats.as_ref().map_or(ptr::null_mut(),|b|b.ptr.cast()),
      shard_head:hash_index.as_ref().map_or(ptr::null_mut(),|b|b.ptr.cast()),
      extent_next:hash_index.as_ref().map_or(ptr::null_mut(),|b|b.at(next_offset)),
      free_cursor:hash_index.as_ref().map_or(ptr::null_mut(),|b|b.at(cursor_offset))};
    b.upload(&[config])?;Some(b)
   }else{None};
   let mut host_terminal=ptr::null_mut();let mut device_terminal=ptr::null_mut();
   crate::generic_native::checked!(mgbfs_nccl_cancel_words,unsafe{mgbfs_nccl_cancel_words(self.comm.ptr,&mut host_terminal,&mut device_terminal)})?;
   if host_terminal.is_null()||device_terminal.is_null()||unsafe{(&*(host_terminal as *const AtomicU32)).load(Ordering::Acquire)}!=0{
    return Err("RESIDENT_TERMINAL_NOT_READY".into())
   }
   let mut window=ptr::null_mut();let mut actual_window=0u64;
   code("RESIDENT_WINDOW_INFO",unsafe{(f.window_info)(self.comm.ptr,&mut window,&mut actual_window)})?;
   if actual_window<window_bytes{return Err("RESIDENT_WINDOW_SIZE".into())}
   let mut others=vec![];
   for (p,n,a) in [
    (window,usize::try_from(actual_window).map_err(|_|"RESIDENT_WINDOW_SIZE")?,64),
    (self.map.ptr,self.map.bytes,4),
    (self.base.perms.ptr,self.base.perms.bytes,4),(layers.ptr,layers.bytes,8),
    (rank_counts.ptr,rank_counts.bytes,8),(rank_count.ptr,rank_count.bytes,8),
    (locks.ptr,locks.bytes,4),(report.ptr,report.bytes,8),(smoke.ptr,smoke.bytes,4),
    (descriptors.ptr,descriptors.bytes,8),(leases.ptr,leases.bytes,4),
    (device_terminal.cast(),4,4),
   ] {if let Some(r)=region(p,n,a){others.push(r)}}
   let (owner,owner_page,owner_boundary,domain,dictionary,visited)=if hash {
    for bucket in &self.segments {for seg in bucket {
     let slots=seg._slots.as_ref().ok_or("RESIDENT_HASH_SLOTS_MISSING")?;
     let positions=seg._positions.as_ref().ok_or("RESIDENT_HASH_POSITIONS_MISSING")?;
     for (p,n) in [(slots.ptr,slots.bytes),(positions.ptr,positions.bytes)] {
      if let Some(r)=region(p,n,8){others.push(r)}
     }
    }}
    if let Some(cuts)=&self.cuts {if let Some(r)=region(cuts.ptr,cuts.bytes,8){others.push(r)}}
    if let Some(stats)=&hash_stats {if let Some(r)=region(stats.ptr,stats.bytes,4){others.push(r)}}
    if let Some(index)=&hash_index {if let Some(r)=region(index.ptr,index.bytes,8){others.push(r)}}
    if let Some(index)=&hash_parent_index {if let Some(r)=region(index.ptr,index.bytes,8){others.push(r)}}
    let config=hash_config.as_ref().ok_or("RESIDENT_HASH_CONFIG_MISSING")?;
    if let Some(r)=region(config.ptr,config.bytes,8){others.push(r)}
    let (dictionary,_,_,_,_)=self.resident_hash.ok_or("RESIDENT_HASH_CONFIG_MISSING")?;
    let domain=BitmapDeviceDomain{symbols:ptr::null(),counts:ptr::null(),
      distinct:dictionary.distinct,elements:dictionary.elements,total:domain_total};
    let width=u64::from(self.plan.elements).checked_mul(u64::from(self.plan.state_bytes)).ok_or("RESIDENT_ROW_WIDTH")?;
    validate_extent_allocations_with_width(&allocations,&others,width).map_err(str::to_owned)?;
    (rank,0,hash_parent_index.as_ref().map_or(0,|b|b.ptr as usize as u64),domain,dictionary,config.ptr.cast())
   }else{
    let b=self.bitmap.as_ref().ok_or("RESIDENT_BITMAP_MISSING")?;
    for (p,n,a) in [(b.bits.ptr,b.bits.bytes,8),(b._symbols.ptr,b._symbols.bytes,8),
      (b._counts.ptr,b._counts.bytes,4)] {if let Some(r)=region(p,n,a){others.push(r)}}
    validate_extent_allocations(&allocations,&others).map_err(str::to_owned)?;
    (b.owner,b.owner_page,b.owner_boundary,b.domain,self.pack4.unwrap().dictionary,b.bits.ptr.cast())
   };
   let args=Args{
    abi_version:1,struct_bytes:mem::size_of::<Args>() as u32,world,rank,
    owner,shards:self.plan.shards,generators:self.base.generators,banks:3,
    owner_page,reserved0:0,owner_boundary,owner_len,
    job_generation:1,rank_limit:self.capacity_limit.unwrap_or(0),
    domain,dictionary,mailbox,
    permutations:self.base.perms.ptr.cast(),physical_of_logical:self.map.ptr.cast(),
    extents:descriptors.ptr.cast(),extent_owner:leases.ptr.cast(),extent_count:n,
    shard_locks:locks.ptr.cast(),visited,rank_count:rank_count.ptr.cast(),
    layer_sizes:layers.ptr.cast(),layer_capacity,rank_counts:rank_counts.ptr.cast(),
    capture:ptr::null_mut(),capture_capacity:0,capture_count:ptr::null_mut(),terminal:device_terminal,
   };
   crate::generic_native::checked!(cudaMemsetAsync,unsafe{cudaMemsetAsync(report.ptr,0,report.bytes,self.base.stream.ptr)})?;
   Ok(Resources{layers,rank_counts,rank_count,locks,descriptors,leases,report,smoke,mapping,args,event,window_bytes:actual_window,layer_capacity,_hash_config:hash_config,hash_stats,hash_index,hash_parent_index})
  })();
  ready.agree("resident-admission",&resource,None)?;
  // If a rank fails while a resident kernel still accesses these buffers,
  // intentionally retain them until the supervisor terminates the poisoned
  // worker. Never recycle device storage while peers may still read it.
  let mut res=std::mem::ManuallyDrop::new(resource?);
  let mut host_terminal=ptr::null_mut();let mut ignored=ptr::null_mut();
  crate::generic_native::checked!(mgbfs_nccl_cancel_words,unsafe{mgbfs_nccl_cancel_words(self.comm.ptr,&mut host_terminal,&mut ignored)})?;

  let init:Result<_>=(||{
   code("RESIDENT_INITIALIZE",unsafe{(initialize)(self.comm.ptr,&res.args,self.base.stream.ptr)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;
   Ok(())
  })();
  ready.agree("resident-initialize",&init,None)?;init?;
  let smoke:Result<_>=(||{
   code("RESIDENT_PEER_SMOKE_LAUNCH",unsafe{(peer_smoke)(self.comm.ptr,&res.args,res.smoke.ptr.cast(),self.base.stream.ptr)})?;
   res.event.wait(self.base.stream.ptr,self.comm.ptr,host_terminal,true)?;
   let mut value=99u32;
   crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut value as *mut u32).cast(),res.smoke.ptr,4,2)})?;
   if value!=0{return Err(format!("RESIDENT_PEER_SMOKE_FAILURE rank={rank} code={value}"))}Ok(())
  })();
  ready.agree("resident-peer-smoke",&smoke,None)?;smoke?;
  let reset:Result<_>=(||{
   code("RESIDENT_REINITIALIZE",unsafe{(initialize)(self.comm.ptr,&res.args,self.base.stream.ptr)})?;
   crate::generic_native::checked!(cudaStreamSynchronize,unsafe{cudaStreamSynchronize(self.base.stream.ptr)})?;Ok(())
  })();
  ready.agree("resident-ready",&reset,None)?;reset?;
  let setup_seconds=start.elapsed().as_secs_f64();
  crate::job_phase::set("resident-bfs")?;
  let started=Instant::now();
  let execution:Result<Report>=(||{
   code("RESIDENT_BFS_LAUNCH",unsafe{(launch)(self.comm.ptr,&res.args,res.report.ptr.cast(),self.base.stream.ptr)})?;
   res.event.wait(self.base.stream.ptr,self.comm.ptr,host_terminal,false)?;
   let mut report=Report::default();
   crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy((&mut report as *mut Report).cast(),res.report.ptr,mem::size_of::<Report>(),2)})?;
   if report.rank!=rank||!matches!(report.status,2|3)||report.depth>=res.layer_capacity {
    return Err(format!("RESIDENT_BFS_FAILED rank={rank} status={} reason={} depth={}",report.status,report.reason,report.depth))
   }
   Ok(report)
  })();
  // Failure is written to the independent supervisor before waiting on peers.
  if let Err(e)=&execution{crate::native_fault::fatal_host("resident-execute",e)}
  ready.agree("resident-drained",&execution,None)?;
  let out=execution?;let bfs_seconds=started.elapsed().as_secs_f64();

  let mut hash_owner_stats=None;
  let readback:Result<_>=(||{
   if let Some(stats)=&res.hash_stats {
    let mut values=[0u32;5];
    crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),stats.ptr,20,2)})?;
    hash_owner_stats=Some(values);
   }
   let mut sizes=vec![0u64;usize::try_from(out.depth+1).map_err(|_|"RESIDENT_DEPTH_RANGE")?];
   crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(sizes.as_mut_ptr().cast(),res.layers.ptr,sizes.len()*8,2)})?;
   let mut counts=vec![0u64;world as usize];
   crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(counts.as_mut_ptr().cast(),res.rank_counts.ptr,counts.len()*8,2)})?;
   if sizes.first()!=Some(&1)||sizes.iter().any(|&v|v==0)||sizes.iter().try_fold(0u64,|a,&b|a.checked_add(b)).ok_or("RESIDENT_SIZE_OVERFLOW")?>domain_total {
    return Err("RESIDENT_LAYER_LOG_CORRUPT".into())
   }
   if counts.iter().try_fold(0u64,|a,&b|a.checked_add(b)).ok_or("RESIDENT_COUNT_OVERFLOW")?!=*sizes.last().unwrap(){
    return Err("RESIDENT_RANK_COUNT_MISMATCH".into())
   }
   let mut current=vec![];let mut previous=vec![];let mut banks=[0u64;3];
   for &(shard,index) in &res.mapping {
    let seg=&mut self.segments[shard][index];let mut values=[0u64;3];
    crate::generic_native::checked!(cudaMemcpy,unsafe{cudaMemcpy(values.as_mut_ptr().cast(),seg.counts.ptr,24,2)})?;
    if values.iter().any(|&v|v>seg.desc.capacity){return Err("RESIDENT_EXTENT_COUNT_CORRUPT".into())}
    seg.banks=values;
    for i in 0..3 {banks[i]=banks[i].checked_add(values[i]).ok_or("RESIDENT_BANK_COUNT_OVERFLOW")?;}
    let cur=(out.depth%3) as u32;
    if values[cur as usize]!=0{current.push(Range{shard,segment:index,bank:cur,begin:0,end:values[cur as usize]})}
    if out.depth>0 {
     let prev=((out.depth+2)%3) as u32;
     if values[prev as usize]!=0{previous.push(Range{shard,segment:index,bank:prev,begin:0,end:values[prev as usize]})}
    }
   }
   let local=counts[rank as usize];
   if current.iter().try_fold(0u64,|a,r|a.checked_add(r.end-r.begin)).ok_or("RESIDENT_COUNT_OVERFLOW")?!=local{
    return Err("RESIDENT_CURRENT_ROWS_MISMATCH".into())
   }
   self.current=current;self.previous=previous;self.counts=counts;self.depth=out.depth;self.bank_totals=banks;
   Ok(sizes)
  })();
  let identity=readback.as_ref().ok().map(|sizes|serde_json::json!({"status":out.status,"reason":out.reason,"sizes":sizes}));
  ready.agree("resident-readback",&readback,identity.as_ref())?;
  let sizes=readback?;
  let retired=code("RESIDENT_RETIRE",unsafe{(f.retire)(self.comm.ptr)});
  ready.agree("resident-retired",&retired,None)?;retired?;
  let (status,reason)=match (out.status,out.reason){
   (2,0)=>("COMPLETE","FRONTIER_EXHAUSTED"),
   (3,1)=>("INCOMPLETE","RESIDENT_RESOURCE_CAPACITY"),
   (3,2) if test_depth.is_some()=>("INCOMPLETE","PROFILE_LAYER_LIMIT"),
   (3,2)=>("INCOMPLETE","RESIDENT_RESOURCE_LAYER_LOG_CAPACITY"),
   _=>return Err(format!("RESIDENT_UNEXPECTED_STATUS status={} reason={}",out.status,out.reason)),
  };
  let evidence=serde_json::json!({"transport":"NCCL_LSA_DEVICE_RING",
    "storage":if hash{"segmented_hash"}else{"packed_bitmap"},
    "hash_owner_execution":if hash{Some(if hash_parallel{"parallel_per_shard"}else{"serialized_per_rank"})}else{None::<&str>},
    "hash_indexed_extents":if hash{Some(hash_indexed)}else{None},
    "hash_extent_index_bytes":res.hash_index.as_ref().map_or(0,|b|b.bytes),
    "hash_parent_index_enabled":hash_parent_index_enabled,
    "hash_parent_index_bytes":res.hash_parent_index.as_ref().map_or(0,|b|b.bytes),
    "hash_owner_peak_active":hash_owner_stats.map(|v|v[0]),
    "hash_extent_full_seen":hash_owner_stats.map(|v|v[1]),
    "hash_freeze_retry":hash_owner_stats.map(|v|v[2]),
    "hash_extent_full_recheck":hash_owner_stats.map(|v|v[3]),
    "hash_extent_full_confirmed":hash_owner_stats.map(|v|v[4]),
    "mailbox_bytes":res.window_bytes,
    "slots_per_ring":res.args.mailbox.slots_per_ring,"lanes":res.args.mailbox.lanes,
    "packet_records":res.args.mailbox.requested_payload_bytes/16,
    "extent_count":res.args.extent_count,"layer_log_capacity":res.layer_capacity,
    "layer_log_bytes":res.layers.bytes,"layer_log_policy":"min(domain, weakest_rank_available_bytes/8192), unless explicitly configured",
    "sent_packets":out.sent,"received_packets":out.received,"duplicates":out.duplicates,
    "duplicates_at_rank_cap":out.duplicates_full,"blocked_send_attempts":out.blocked,
    "max_active_shards_first64":out.max_active_shards_first64,
    "max_active_shards_scope":"accepted rows per attempted layer on this rank; first 64 shards",
    "layer_seconds_scope":"not measured","peak_live_rows_scope":"not measured"});
  // All ranks completed readback and retired the shared window before any
  // of these buffers may be returned to the persistent pool.
  unsafe{std::mem::ManuallyDrop::drop(&mut res);}
  Ok(ResidentResult{sizes,status,reason,bfs_seconds,setup_seconds,evidence})
 }
}
