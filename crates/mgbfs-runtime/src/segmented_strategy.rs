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
