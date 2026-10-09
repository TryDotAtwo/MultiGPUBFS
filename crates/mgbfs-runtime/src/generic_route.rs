use mgbfs_core::Result;
/// Logical geometry only. This does not assert transport/hardware acceptance.
#[derive(Debug,Clone)]
pub struct GenericRouteGeometry {pub world:u32,pub local_shards:u32,pub owner_to_rank:Vec<u32>,pub owner_cuts:Vec<u64>,pub queue_capacity:u32,pub bank_bytes:u64}
impl GenericRouteGeometry {
 pub fn new(weights:&[u64],map:Vec<u32>,local_shards:u32,queue_capacity:u32)->Result<Self>{
  let world=weights.len() as u32;
  if world==0||world>128||local_shards==0||local_shards>4096||queue_capacity==0||map.len()!=weights.len()||weights.iter().any(|&v|v==0){return Err("GENERIC_ROUTE_SHAPE".into());}
  let mut seen=vec![false;weights.len()];for &rank in &map {let at=rank as usize;if at>=seen.len()||seen[at]{return Err("GENERIC_ROUTE_MAP".into());}seen[at]=true;}
  let total=weights.iter().map(|&v|u128::from(v)).sum::<u128>();let mut sum=0u128;let mut cuts=vec![0u64];
  for &weight in weights {sum+=u128::from(weight);cuts.push(((sum*(1u128<<32))/total) as u64);}
  if cuts.windows(2).any(|pair|pair[0]>=pair[1]){return Err("GENERIC_ROUTE_WEIGHT_RESOLUTION".into());}
  let bytes=u64::from(world)*u64::from(local_shards)*u64::from(queue_capacity)*32;
  Ok(Self{world,local_shards,owner_to_rank:map,owner_cuts:cuts,queue_capacity,bank_bytes:bytes})
 }
 pub fn destination(&self,hash:u64)->(u32,u32){
  let high=hash>>32;let at=self.owner_cuts.partition_point(|&cut|cut<=high)-1;
  let shard=((hash&0xffff_ffff)*u64::from(self.local_shards)>>32) as u32;
  (self.owner_to_rank[at],shard)
 }
}
