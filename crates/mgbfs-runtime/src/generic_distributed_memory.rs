//! Byte-exact admission for retained-history routing. This is admission,
//! not a measured throughput tuner. Queue capacity tolerates complete skew.
use mgbfs_core::Result;
#[derive(Clone,Debug,PartialEq,Eq,serde::Serialize,serde::Deserialize)]
pub struct GenericDistributedMemoryPlan {
 pub elements:u32,pub world:u32,pub shards:u32,pub capacity:u32,pub batch:u32,
 pub queue_capacity:u32,pub slots_per_shard:u32,pub generator_bytes:u64,pub device_bytes:u64,
}
impl GenericDistributedMemoryPlan {
 pub fn new(elements:u32,world:u32,shards:u32,capacity:u32,batch:u32,generators:u32,generator_bytes:u64)->Result<Self>{
  if elements==0||!(1..=128).contains(&world)||!(1..=4096).contains(&shards)||capacity==0||capacity>(1<<28)||batch==0||batch>capacity||generators==0 {return Err("GENERIC_DISTRIBUTED_SHAPE".into());}
  let queue_capacity=batch.checked_mul(generators).filter(|v|*v<0x80000000).ok_or("GENERIC_QUEUE_INDEX_RANGE")?;
  // Each shard can retain the whole local arena: correctness does not depend
  // on hash balance, even for forced collisions. Tune this only with spill.
  let slots_per_shard=capacity.checked_mul(2).and_then(u32::checked_next_power_of_two).ok_or("GENERIC_SHARD_SLOTS")?;
  let checked=||->Option<u64>{
   let arena=u64::from(elements).checked_mul(u64::from(capacity))?.checked_mul(16)?.checked_add(u64::from(capacity)*8)?;
   let tables=u64::from(slots_per_shard).checked_mul(u64::from(shards))?.checked_mul(8)?;
   let boxes=u64::from(world).checked_mul(u64::from(shards))?;
   let queue=u64::from(queue_capacity).checked_mul(32+u64::from(elements)*8)?.checked_add(4)?;
   // Two source banks and one received inbox for every source. Control,
   // graph tables and rank count gather are included, not hidden headroom.
   arena.checked_add(tables)?.checked_add(boxes.checked_mul(queue)?.checked_mul(3)?)?.checked_add(generator_bytes)?.checked_add(64+u64::from(world)*8)
  };
  let device_bytes=checked().ok_or("GENERIC_DISTRIBUTED_BYTES_OVERFLOW")?;
  Ok(Self{elements,world,shards,capacity,batch,queue_capacity,slots_per_shard,generator_bytes,device_bytes})
 }
 pub fn automatic(elements:u32,world:u32,shards:u32,generators:u32,generator_bytes:u64,free_bytes:u64,upper_bound:Option<u64>)->Result<Self>{
  let reserve=(1u64<<30).max(free_bytes/10);let budget=free_bytes.checked_sub(reserve).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
  let ceiling=upper_bound.unwrap_or(1<<28).min(1<<28) as u32;
  if ceiling==0{return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());}
  let mut low=1;let mut high=ceiling;
  // Preserve room for batched routing instead of consuming all VRAM with history.
  let history_budget=budget*3/4;
  while low<high {let mid=low+(high-low+1)/2;let p=Self::new(elements,world,shards,mid,1,generators,generator_bytes)?;if p.device_bytes<=history_budget{low=mid;}else{high=mid-1;}}
  let capacity=low;let mut lo=1;let mut hi=capacity.min((1<<20)/generators.max(1)).max(1);
  if Self::new(elements,world,shards,capacity,1,generators,generator_bytes)?.device_bytes>budget{return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());}
  while lo<hi {let mid=lo+(hi-lo+1)/2;if Self::new(elements,world,shards,capacity,mid,generators,generator_bytes)?.device_bytes<=budget{lo=mid;}else{hi=mid-1;}}
  Self::new(elements,world,shards,capacity,lo,generators,generator_bytes)
 }
 pub fn validate(&self,generators:u32)->Result<()> {
  if *self!=Self::new(self.elements,self.world,self.shards,self.capacity,self.batch,generators,self.generator_bytes)?{return Err("GENERIC_DISTRIBUTED_PLAN_MUTATED".into());}Ok(())
 }
}
