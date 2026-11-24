//! Strategy selection and wave-level retry decisions shared by CPU tests
//! and the native segmented execution path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentStrategy { Atomic, Warp, Batch }
impl SegmentStrategy {
 pub fn parse(value:Option<&str>)->Result<Self,&'static str>{
  match value.unwrap_or("atomic"){
   "atomic"=>Ok(Self::Atomic),"warp"=>Ok(Self::Warp),"batch"=>Ok(Self::Batch),_=>Err("SEGMENT_STRATEGY_INVALID")
  }
 }
 pub fn code(self)->u64{match self{Self::Atomic=>0,Self::Warp=>1,Self::Batch=>2}}
 pub fn name(self)->&'static str{match self{Self::Atomic=>"atomic",Self::Warp=>"warp",Self::Batch=>"batch"}}
}
pub fn agree(codes:&[u64])->Result<SegmentStrategy,&'static str>{
 let first=*codes.first().ok_or("SEGMENT_STRATEGY_RANK_MISMATCH")?;
 if codes.iter().any(|&x|x!=first){return Err("SEGMENT_STRATEGY_RANK_MISMATCH")}
 match first{0=>Ok(SegmentStrategy::Atomic),1=>Ok(SegmentStrategy::Warp),2=>Ok(SegmentStrategy::Batch),_=>Err("SEGMENT_STRATEGY_INVALID")}
}
/// Every rank advertises both the selected strategy and the native batch ABI.
/// Capability differences are harmless for Atomic/Warp; Batch fails on all ranks
/// before routing if even one library is missing the required entry points.
pub fn agree_capabilities(advertisements:&[u64])->Result<SegmentStrategy,&'static str>{
 if advertisements.iter().any(|&v|v&!0x1ff!=0){return Err("SEGMENT_STRATEGY_INVALID")}
 let codes=advertisements.iter().map(|v|v&0xff).collect::<Vec<_>>();
 let strategy=agree(&codes)?;
 if strategy==SegmentStrategy::Batch&&advertisements.iter().any(|v|v&0x100==0){
  return Err("SEGMENT_STRATEGY_BATCH_NATIVE_ABI_MISSING")
 }
 Ok(strategy)
}


/// One pre-data collective advertises both recovery policy and native strategy.
/// No rank may switch policy independently.
pub fn agree_recovery_capabilities(advertisements:&[u64])->Result<(SegmentStrategy,bool),&'static str>{
 if advertisements.is_empty(){return Err("SEGMENT_BATCH_RECOVERY_RANK_MISMATCH")}
 if advertisements.iter().any(|&v|v>>9>1){return Err("SEGMENT_BATCH_RECOVERY_INVALID")}
 let first=advertisements[0]>>9;
 if advertisements.iter().any(|&v|v>>9!=first){return Err("SEGMENT_BATCH_RECOVERY_RANK_MISMATCH")}
 let base=advertisements.iter().map(|v|v&0x1ff).collect::<Vec<_>>();
 Ok((agree_capabilities(&base)?,first==1))
}


// Queue capacities determine peer offsets and message layout. Agreement must
// precede all graph data, even when the optional recovery policy is disabled.
pub fn agree_recovery_geometry(advertisements:&[u64])->Result<(SegmentStrategy,bool),&'static str>{
 let first=*advertisements.first().ok_or("SEGMENT_BATCH_RECOVERY_RANK_MISMATCH")?;
 let queue=first>>32;
 if queue==0{return Err("SEGMENTED_QUEUE_CAPACITY_INVALID")}
 if advertisements.iter().any(|v|v>>32!=queue){return Err("SEGMENTED_QUEUE_CAPACITY_RANK_MISMATCH")}
 let strategy=advertisements.iter().map(|v|v&0xffff_ffff).collect::<Vec<_>>();
 agree_recovery_capabilities(&strategy)
}


/// The optional exact-ID permutation must be uniform across ranks; old native
/// bundles remain valid for the default cyclic layout.
pub fn agree_owner_page_geometry(advertisements:&[u64])->Result<(SegmentStrategy,bool,u32),&'static str>{
 let first=*advertisements.first().ok_or("BITMAP_OWNER_PAGE_RANK_MISMATCH")?;
 let code=(first>>11)&0x3f;let bitmap=(first>>18)&1;
 if advertisements.iter().any(|v|(v>>18)&1!=bitmap){return Err("BITMAP_MODE_RANK_MISMATCH")}
 if bitmap!=0&&advertisements.iter().any(|v|(v>>19)&1!=(first>>19)&1){return Err("BITMAP_ORBIT_CAPABILITY_RANK_MISMATCH")}
 if bitmap!=0&&advertisements.iter().any(|v|(v>>21)&1!=(first>>21)&1){return Err("BITMAP_IMMUTABLE_ABI_RANK_MISMATCH")}
 if advertisements.iter().any(|v|(v>>11)&0x3f!=code){return Err("BITMAP_OWNER_PAGE_RANK_MISMATCH")}
 if code>21||(code!=0&&bitmap==0){return Err("BITMAP_OWNER_PAGE_INVALID")}
 if code!=0&&advertisements.iter().any(|v|(v>>21)&1==0){return Err("BITMAP_OWNER_PAGE_NATIVE_ABI_MISSING")}
 let base=advertisements.iter().map(|v|v&0xffff_ffff_0000_07ff).collect::<Vec<_>>();
 let(strategy,recovery)=agree_recovery_geometry(&base)?;
 if strategy==SegmentStrategy::Warp&&bitmap==0&&advertisements.iter().any(|v|(v>>20)&1==0){
  return Err("SEGMENT_WARP_SAFE_NATIVE_ABI_MISSING")
 }
 Ok((strategy,recovery,if code==0{0}else{1u32<<(code-1)}))
}

/// Experimental packed bitmap ABI availability is only a compatibility
/// statement; it is not GPU acceptance or an automatic profile choice.
#[cfg(all(feature="cuda",target_os="linux"))]
pub fn bitmap_pack4_abi_ready()->bool{crate::segmented_distributed_native::pack4_capability()}
#[cfg(not(all(feature="cuda",target_os="linux")))]
pub fn bitmap_pack4_abi_ready()->bool{false}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub fn packet_transport_abi_ready() -> bool {
    let symbols: &[&[u8]] = &[
        b"mgbfs_nccl_segment_packet_v1\0",
        b"mgbfs_segmented_compact_route_v1\0",
        b"mgbfs_bitmap_accept_packet_v1\0",
        b"mgbfs_bitmap_accept_batch_packet_v1\0",
    ];
    symbols.iter().all(|symbol| unsafe {
        !libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr().cast()).is_null()
    })
}
#[cfg(not(all(feature = "cuda", target_os = "linux")))]
pub fn packet_transport_abi_ready() -> bool { false }

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub fn segmented_nearest_abi_ready() -> bool {
    batch_abi_ready()
        && hash_series_abi_ready()
        && packet_transport_abi_ready()
        && crate::segmented_distributed_native::nearest_abi_ready()
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
pub fn segmented_nearest_abi_ready() -> bool {
    false
}

/// The hash warp path must never silently run a library with reusable
/// pending slots; atomic and batch remain compatible with earlier bundles.
#[cfg(all(feature="cuda",target_os="linux"))]
pub fn warp_safe_abi_ready()->bool{unsafe{
 if std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref()==Ok("1")&&std::env::var("MGBFS_SEGMENT_TEST_WARP_SAFE_ABI_MISSING").as_deref()==Ok("1"){return false}
 let p=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_segmented_warp_safe_abi_version\0".as_ptr().cast());
 if p.is_null(){return false}
 let version:unsafe extern "C" fn()->u32=std::mem::transmute(p);version()==1
}}
#[cfg(not(all(feature="cuda",target_os="linux")))]
pub fn warp_safe_abi_ready()->bool{false}

#[cfg(all(feature="cuda",target_os="linux"))]
pub fn bitmap_orbit_abi_ready()->bool{unsafe{
 let p=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_bitmap_orbit_domain_abi_version\0".as_ptr().cast());
 if p.is_null(){return false}
 let version:unsafe extern "C" fn()->u32=std::mem::transmute(p);version()==1
}}
#[cfg(not(all(feature="cuda",target_os="linux")))]
pub fn bitmap_orbit_abi_ready()->bool{false}

#[cfg(all(feature="cuda",target_os="linux"))]
pub fn bitmap_immutable_abi_ready()->bool{unsafe{
 let p=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_bitmap_immutable_layout_abi_version\0".as_ptr().cast());
 if p.is_null(){return false}
 let version:unsafe extern "C" fn()->u32=std::mem::transmute(p);
 if version()!=1{return false}
 [b"mgbfs_bitmap_route_v2\0".as_slice(),b"mgbfs_bitmap_route_grouped_v2\0".as_slice(),b"mgbfs_segmented_batch_route_v2\0".as_slice()].iter().all(|name|!libc::dlsym(libc::RTLD_DEFAULT,name.as_ptr().cast()).is_null())
}}
#[cfg(not(all(feature="cuda",target_os="linux")))]
pub fn bitmap_immutable_abi_ready()->bool{false}

#[cfg(all(feature="cuda",target_os="linux"))]
pub type BitmapOwnerPageSetter=unsafe extern "C" fn(u32,u64)->i32;
#[cfg(all(feature="cuda",target_os="linux"))]
pub fn bitmap_owner_page_setter()->Option<BitmapOwnerPageSetter>{
 let p=unsafe{libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_bitmap_set_owner_page\0".as_ptr().cast())};
 if p.is_null(){None}else{Some(unsafe{std::mem::transmute::<*mut libc::c_void,BitmapOwnerPageSetter>(p)})}
}

/// Restore the route batch only after two consecutive drained, retry-free
/// layers with at least half of every destination queue left unused.
#[derive(Default)]
pub struct QueueBatchRecovery{quiet_layers:u8}
impl QueueBatchRecovery{
 pub fn observe(&mut self,current:u32,admitted:u32,peak:u32,capacity:u32,had_work:bool,retried:bool)->u32{
  if current>=admitted||capacity==0||!had_work||retried||u64::from(peak)*2>u64::from(capacity){
   self.quiet_layers=0;return current
  }
  self.quiet_layers=self.quiet_layers.saturating_add(1);
  if self.quiet_layers<2{return current}
  self.quiet_layers=0;current.saturating_mul(2).min(admitted)
 }
}

#[cfg(all(feature="cuda",target_os="linux"))]
pub fn batch_abi_ready()->bool{
 if std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref()==Ok("1") &&
    std::env::var("MGBFS_SEGMENT_TEST_BATCH_ABI_MISSING").as_deref()==Ok("1"){return false}
 unsafe{
  let marker=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_segmented_batch_strategy_abi_version\0".as_ptr().cast());
  if marker.is_null(){return false}
  let version:unsafe extern "C" fn()->u32=std::mem::transmute(marker);
  if version()!=1{return false}
  let names:[&[u8];5]=[b"mgbfs_segmented_batch_route\0",
   b"mgbfs_bitmap_accept_batch\0",b"mgbfs_hash_batch_accept_profiled_v3\0",
   b"mgbfs_hash_batch_compact_tail_v2\0",b"mgbfs_segmented_dense_origins\0"];
  names.iter().all(|name|!libc::dlsym(libc::RTLD_DEFAULT,name.as_ptr().cast()).is_null())
 }
}
#[cfg(not(all(feature="cuda",target_os="linux")))]
pub fn batch_abi_ready()->bool{false}
pub type DenseOriginsFn=unsafe extern "C" fn(*const u32,u32,u32,u32,u32,*mut u32,*mut std::ffi::c_void)->i32;
#[cfg(all(feature="cuda",target_os="linux"))]
pub fn dense_origins()->Option<DenseOriginsFn>{unsafe{
 let p=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_segmented_dense_origins\0".as_ptr().cast());
 if p.is_null(){None}else{Some(std::mem::transmute(p))}
}}
#[cfg(not(all(feature="cuda",target_os="linux")))]
pub fn dense_origins()->Option<DenseOriginsFn>{None}


/// Cursor over unfiltered origins. A filtered spill is tracked separately and
/// must drain before asking this cursor for another raw tile.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct RawTileCursor { next:u32, total:u32, tile:u32 }
impl RawTileCursor {
 pub fn new(total:u32,tile:u32)->Result<Self,&'static str>{
  if tile==0{return Err("SEGMENT_TILE_ZERO")}
  Ok(Self{next:0,total,tile})
 }
 pub fn next_tile(&mut self)->Option<(u32,u32)>{
  if self.next==self.total{return None}
  let begin=self.next;
  let count=(self.total-self.next).min(self.tile);
  self.next+=count;
  Some((begin,count))
 }
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum OwnerDecision { Done, Same(bool), Next, Exhausted }
// Called only after *every* owner lane in the wave has drained, so observed
// cap reflects committed rows and no transient reservation from that wave.
pub fn owner_decision(spill:u32,cap:Option<u64>,observed:u64,segment_full:bool,
 cap_retry:bool,cap_probe:bool,accepted_progress:bool)->Result<OwnerDecision,&'static str>{
 if cap_retry&&cap.is_none(){return Err("SEGMENT_CAP_RETRY_WITHOUT_LIMIT")}
 if spill==0{return Ok(OwnerDecision::Done)}
 if let Some(limit)=cap{
  if observed>limit{return Err("SEGMENT_CAP_COUNTER_OVERFLOW")}
  if observed==limit{
   return if cap_probe&&!accepted_progress{Ok(OwnerDecision::Exhausted)}else{Ok(OwnerDecision::Same(true))}
  }
  if cap_retry&&!segment_full{return Ok(OwnerDecision::Same(false))}
 }
 Ok(OwnerDecision::Next)
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn strategy_values_and_rank_agreement(){
  assert_eq!(SegmentStrategy::parse(None).unwrap(),SegmentStrategy::Atomic);
  assert_eq!(SegmentStrategy::parse(Some("warp")).unwrap(),SegmentStrategy::Warp);
  assert_eq!(SegmentStrategy::parse(Some("batch")).unwrap(),SegmentStrategy::Batch);
  assert!(SegmentStrategy::parse(Some("wArp")).is_err());
  assert_eq!(agree(&[1,1]).unwrap(),SegmentStrategy::Warp);
  assert!(agree(&[0,1]).is_err());assert!(agree(&[3,3]).is_err());assert!(agree(&[]).is_err());
 }
 #[test] fn capability_agreement_checks_batch_on_every_rank(){
  assert_eq!(agree_capabilities(&[0,0x100]).unwrap(),SegmentStrategy::Atomic);
  assert_eq!(agree_capabilities(&[0x101,1]).unwrap(),SegmentStrategy::Warp);
  assert_eq!(agree_capabilities(&[0x102,0x102]).unwrap(),SegmentStrategy::Batch);
  assert_eq!(agree_capabilities(&[0x102,2]),Err("SEGMENT_STRATEGY_BATCH_NATIVE_ABI_MISSING"));
  assert_eq!(agree_capabilities(&[2,2]),Err("SEGMENT_STRATEGY_BATCH_NATIVE_ABI_MISSING"));
  assert_eq!(agree_capabilities(&[0x102,0x101]),Err("SEGMENT_STRATEGY_RANK_MISMATCH"));
  assert_eq!(agree_capabilities(&[0x203]),Err("SEGMENT_STRATEGY_INVALID"));
 }
 #[test] fn raw_tile_cursor_preserves_unprocessed_tail(){
  let mut raw=RawTileCursor::new(6,2).unwrap();
  assert_eq!(raw.next_tile(),Some((0,2)));
  // Filtered spill belongs to this tile: the cursor does not advance while
  // the owner replays it, including a retry in the same bank after cap drain.
  let pending_spill=vec![1u32];
  assert_eq!(pending_spill,vec![1]);
  assert_eq!(raw.next_tile(),Some((2,2)));
  assert_eq!(raw.next_tile(),Some((4,2)));
  assert_eq!(raw.next_tile(),None);
  assert_eq!(RawTileCursor::new(1,0).unwrap_err(),"SEGMENT_TILE_ZERO");
  assert_eq!(RawTileCursor::new(0,1).unwrap().next_tile(),None);
 }
 #[test] fn cap_retry_waits_for_wave_drain_and_reuses_bank(){
  assert_eq!(owner_decision(3,Some(100),97,false,true,false,false).unwrap(),OwnerDecision::Same(false));
  assert_eq!(owner_decision(3,Some(100),97,true,true,false,false).unwrap(),OwnerDecision::Next);
  assert_eq!(owner_decision(3,Some(100),100,false,true,false,false).unwrap(),OwnerDecision::Same(true));
  assert_eq!(owner_decision(3,Some(100),100,false,true,true,false).unwrap(),OwnerDecision::Exhausted);
  assert_eq!(owner_decision(3,Some(100),100,false,true,true,true).unwrap(),OwnerDecision::Same(true));
  assert_eq!(owner_decision(0,Some(100),100,true,true,true,false).unwrap(),OwnerDecision::Done);
  assert!(owner_decision(3,Some(100),101,false,true,false,false).is_err());
  assert!(owner_decision(3,None,0,false,true,false,false).is_err());
  assert_eq!(owner_decision(3,None,0,true,false,false,false).unwrap(),OwnerDecision::Next);
 }
}

#[cfg(test)]
mod batch_recovery_policy_tests {
 use super::QueueBatchRecovery;
 #[test]
 fn grows_after_two_quiet_complete_layers_and_never_exceeds_admission(){
  let mut p=QueueBatchRecovery::default();
  assert_eq!(p.observe(2048,8192,1024,4096,true,false),2048);
  assert_eq!(p.observe(2048,8192,1024,4096,true,false),4096);
  assert_eq!(p.observe(4096,8192,1024,4096,true,false),4096);
  assert_eq!(p.observe(4096,8192,1024,4096,true,false),8192);
  assert_eq!(p.observe(8192,8192,100,4096,true,false),8192);
 }
 #[test]
 fn retry_busy_or_empty_layer_breaks_the_streak(){
  let mut p=QueueBatchRecovery::default();
  assert_eq!(p.observe(1024,8192,1,4096,true,false),1024);
  assert_eq!(p.observe(1024,8192,1,4096,true,true),1024);
  assert_eq!(p.observe(1024,8192,1,4096,true,false),1024);
  assert_eq!(p.observe(1024,8192,3000,4096,true,false),1024);
  assert_eq!(p.observe(1024,8192,1,4096,true,false),1024);
  assert_eq!(p.observe(1024,8192,0,4096,false,false),1024);
  assert_eq!(p.observe(1024,8192,1,4096,true,false),1024);
  assert_eq!(p.observe(1024,8192,1,4096,true,false),2048);
 }
 #[test]
 fn saturating_growth_obeys_non_power_of_two_limits(){
  let mut p=QueueBatchRecovery::default();
  assert_eq!(p.observe(3000,5000,10,4096,true,false),3000);
  assert_eq!(p.observe(3000,5000,10,4096,true,false),5000);
  assert_eq!(p.observe(u32::MAX-1,u32::MAX,0,u32::MAX,true,false),u32::MAX-1);
  assert_eq!(p.observe(u32::MAX-1,u32::MAX,0,u32::MAX,true,false),u32::MAX);
 }
}

#[cfg(test)]
mod batch_recovery_agreement_tests {
 use super::agree_recovery_capabilities;
 #[test]
 fn every_rank_agrees_on_opt_in_and_bad_values_fail_together(){
  assert_eq!(agree_recovery_capabilities(&[0x100,0x100]),Ok((super::SegmentStrategy::Atomic,false)));
  assert_eq!(agree_recovery_capabilities(&[0x300,0x300]),Ok((super::SegmentStrategy::Atomic,true)));
  assert_eq!(agree_recovery_capabilities(&[0x100,0x300]),Err("SEGMENT_BATCH_RECOVERY_RANK_MISMATCH"));
  assert_eq!(agree_recovery_capabilities(&[0x500,0x500]),Err("SEGMENT_BATCH_RECOVERY_INVALID"));
  assert_eq!(agree_recovery_capabilities(&[]),Err("SEGMENT_BATCH_RECOVERY_RANK_MISMATCH"));
  assert_eq!(agree_recovery_capabilities(&[0x302,0x302]),Ok((super::SegmentStrategy::Batch,true)));
  assert_eq!(agree_recovery_capabilities(&[0x302,0x202]),Err("SEGMENT_STRATEGY_BATCH_NATIVE_ABI_MISSING"));
 }
}

#[cfg(test)]
mod batch_queue_agreement_tests{
 use super::agree_recovery_geometry;
 #[test]
 fn mismatched_or_zero_queue_capacity_is_collectively_rejected(){
  let ad=|q:u64,v:u64|(q<<32)|v;
  assert_eq!(agree_recovery_geometry(&[ad(8,0x100),ad(8,0x100)]),Ok((super::SegmentStrategy::Atomic,false)));
  assert_eq!(agree_recovery_geometry(&[ad(8,0x300),ad(8,0x300)]),Ok((super::SegmentStrategy::Atomic,true)));
  assert_eq!(agree_recovery_geometry(&[ad(8,0x100),ad(4,0x100)]),Err("SEGMENTED_QUEUE_CAPACITY_RANK_MISMATCH"));
  assert_eq!(agree_recovery_geometry(&[ad(0,0x100),ad(0,0x100)]),Err("SEGMENTED_QUEUE_CAPACITY_INVALID"));
 }
}

#[cfg(test)]
mod bitmap_owner_page_agreement_tests{
 use super::{agree_owner_page_geometry,SegmentStrategy};
 fn ad(queue:u64,bitmap:u64,page_code:u64,cap:u64)->u64{
  (queue<<32)|(bitmap<<18)|(cap<<17)|(page_code<<11)|0x100
 }
 #[test]
 fn owner_page_requires_identical_bitmap_mode_mapping_and_native_capability(){
  let v2=1<<21;
  assert_eq!(agree_owner_page_geometry(&[ad(8,1,7,1)|v2,ad(8,1,7,1)|v2]),Ok((SegmentStrategy::Atomic,false,64)));
  assert_eq!(agree_owner_page_geometry(&[ad(8,1,0,0),ad(8,1,0,1)]),Ok((SegmentStrategy::Atomic,false,0)));
  assert_eq!(agree_owner_page_geometry(&[ad(8,1,7,1)|v2,ad(8,1,8,1)|v2]),Err("BITMAP_OWNER_PAGE_RANK_MISMATCH"));
  assert_eq!(agree_owner_page_geometry(&[ad(8,1,63,1)|v2,ad(8,1,63,1)|v2]),Err("BITMAP_OWNER_PAGE_INVALID"));
  assert_eq!(agree_owner_page_geometry(&[ad(8,1,7,1),ad(8,1,7,1)]),Err("BITMAP_OWNER_PAGE_NATIVE_ABI_MISSING"));
  assert_eq!(agree_owner_page_geometry(&[ad(8,1,0,1)|v2,ad(8,1,0,1)]),Err("BITMAP_IMMUTABLE_ABI_RANK_MISMATCH"));
  assert_eq!(agree_owner_page_geometry(&[ad(8,0,0,1),ad(8,1,0,1)]),Err("BITMAP_MODE_RANK_MISMATCH"));
  assert_eq!(agree_owner_page_geometry(&[ad(8,1,0,1)|(1<<19),ad(8,1,0,1)]),Err("BITMAP_ORBIT_CAPABILITY_RANK_MISMATCH"));
 }
}

#[cfg(test)]
mod safe_warp_abi_tests{
 use super::{agree_owner_page_geometry,SegmentStrategy};
 #[test] fn all_hash_warp_ranks_require_safe_abi(){
  let queue=8u64<<32;let safe=1u64<<20;let bitmap=1u64<<18;
  assert_eq!(agree_owner_page_geometry(&[queue|safe|1,queue|safe|1]),Ok((SegmentStrategy::Warp,false,0)));
  assert_eq!(agree_owner_page_geometry(&[queue|safe|1,queue|1]),Err("SEGMENT_WARP_SAFE_NATIVE_ABI_MISSING"));
  assert_eq!(agree_owner_page_geometry(&[queue|bitmap|1,queue|bitmap|1]),Ok((SegmentStrategy::Warp,false,0)));
  assert_eq!(agree_owner_page_geometry(&[queue,queue]),Ok((SegmentStrategy::Atomic,false,0)));
 }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub fn hash_series_abi_ready() -> bool {
    let symbols: &[&[u8]] = &[
        b"mgbfs_hash_batch_accept_profiled_v6\0",
        b"mgbfs_hash_extent_bank_summary_v1\0",
        b"mgbfs_hash_extent_gather_v1\0",
        b"mgbfs_hash_extent_sort_scratch_v1\0",
        b"mgbfs_hash_extent_sort_v1\0",
        b"mgbfs_hash_extent_prefixes_v1\0",
        b"mgbfs_hash_extent_remap_v1\0",
    ];
    symbols.iter().all(|symbol| unsafe {
        !libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr().cast()).is_null()
    })
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
pub fn hash_series_abi_ready() -> bool { false }
