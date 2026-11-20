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

#[cfg(all(feature="cuda",target_os="linux"))]
pub fn batch_abi_ready()->bool{
 if std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref()==Ok("1") &&
    std::env::var("MGBFS_SEGMENT_TEST_BATCH_ABI_MISSING").as_deref()==Ok("1"){return false}
 unsafe{
  let marker=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_segmented_batch_strategy_abi_version\0".as_ptr().cast());
  if marker.is_null(){return false}
  let version:unsafe extern "C" fn()->u32=std::mem::transmute(marker);
  if version()!=1{return false}
  let names:[&[u8];4]=[b"mgbfs_segmented_batch_route\0",
   b"mgbfs_bitmap_accept_batch\0",b"mgbfs_hash_batch_accept_timed\0",
   b"mgbfs_hash_batch_compact_tail\0"];
  names.iter().all(|name|!libc::dlsym(libc::RTLD_DEFAULT,name.as_ptr().cast()).is_null())
 }
}
#[cfg(not(all(feature="cuda",target_os="linux")))]
pub fn batch_abi_ready()->bool{false}

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
