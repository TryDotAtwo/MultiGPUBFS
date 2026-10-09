//! Byte-exact admission for retained-history routing. This is admission,
//! not a measured throughput tuner. Skew is handled by immutable-source batch retry before owner mutation.
use mgbfs_core::Result;
#[derive(Clone,Debug,PartialEq,Eq,serde::Serialize,serde::Deserialize)]
pub struct GenericDistributedMemoryPlan {
 pub state_bytes:u32,pub elements:u32,pub world:u32,pub shards:u32,pub capacity:u32,pub batch:u32,
 pub queue_capacity:u32,pub slots_per_shard:u32,pub generator_bytes:u64,pub device_bytes:u64,
}
impl GenericDistributedMemoryPlan {
 pub fn new(elements:u32,world:u32,shards:u32,capacity:u32,batch:u32,generators:u32,generator_bytes:u64)->Result<Self>{Self::with_storage(elements,world,shards,capacity,batch,generators,generator_bytes,8)}
 pub fn with_storage(elements:u32,world:u32,shards:u32,capacity:u32,batch:u32,generators:u32,generator_bytes:u64,state_bytes:u32)->Result<Self>{
  if ![1,8].contains(&state_bytes)||elements==0||!(1..=128).contains(&world)||!(1..=4096).contains(&shards)||capacity==0||capacity>(1<<28)||batch==0||batch>capacity||generators==0 {return Err("GENERIC_DISTRIBUTED_SHAPE".into());}
  let full=batch.checked_mul(generators).filter(|v|*v<0x80000000).ok_or("GENERIC_QUEUE_INDEX_RANGE")?;
  let boxes=u64::from(world)*u64::from(shards);
  let balanced=((u64::from(full)+boxes-1)/boxes).saturating_mul(4).max(u64::from(generators));
  let queue_capacity=u64::from(full).min(balanced) as u32;
  if u64::from(queue_capacity)*u64::from(world)>=0x7fffffff{return Err("GENERIC_QUEUE_INDEX_RANGE".into());}
  // Each shard can retain the whole local arena: correctness does not depend
  // on hash balance, even for forced collisions. Tune this only with spill.
  let slots_per_shard=capacity.checked_mul(2).and_then(u32::checked_next_power_of_two).ok_or("GENERIC_SHARD_SLOTS")?;
  let checked=||->Option<u64>{
   let arena=u64::from(elements).checked_mul(u64::from(capacity))?.checked_mul(u64::from(state_bytes)*2)?.checked_add(u64::from(capacity)*8)?;
   let tables=u64::from(slots_per_shard).checked_mul(u64::from(shards))?.checked_mul(8)?;
   let boxes=u64::from(world).checked_mul(u64::from(shards))?;
   let payload=if state_bytes==1&&elements<=16{0}else{u64::from(elements)*u64::from(state_bytes)};
   let queue=u64::from(queue_capacity).checked_mul(32+payload)?.checked_add(4)?;
   // Two source banks and one received inbox for every source. Control,
   // graph tables and rank count gather are included, not hidden headroom.
   arena.checked_add(tables)?.checked_add(boxes.checked_mul(queue)?.checked_mul(3)?)?.checked_add(generator_bytes)?.checked_add(64+u64::from(world)*8+if state_bytes==1&&elements<=16{3}else{0})
  };
  let device_bytes=checked().ok_or("GENERIC_DISTRIBUTED_BYTES_OVERFLOW")?;
  Ok(Self{state_bytes,elements,world,shards,capacity,batch,queue_capacity,slots_per_shard,generator_bytes,device_bytes})
 }
 pub fn automatic(elements:u32,world:u32,shards:u32,generators:u32,generator_bytes:u64,free_bytes:u64,upper_bound:Option<u64>)->Result<Self>{Self::automatic_storage(elements,world,shards,generators,generator_bytes,free_bytes,upper_bound,8)}
 pub fn automatic_storage(elements:u32,world:u32,shards:u32,generators:u32,generator_bytes:u64,free_bytes:u64,upper_bound:Option<u64>,state_bytes:u32)->Result<Self>{
  let reserve=(1u64<<30).max(free_bytes/10);let budget=free_bytes.checked_sub(reserve).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
  let ceiling=upper_bound.unwrap_or(1<<28).min(1<<28) as u32;
  if ceiling==0{return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());}
  let mut low=1;let mut high=ceiling;
  // Preserve room for batched routing instead of consuming all VRAM with history.
  let history_budget=budget*3/4;
  while low<high {let mid=low+(high-low+1)/2;let p=Self::with_storage(elements,world,shards,mid,1,generators,generator_bytes,state_bytes)?;if p.device_bytes<=history_budget{low=mid;}else{high=mid-1;}}
  let capacity=low;let mut lo=1;let mut hi=capacity.min((1<<20)/generators.max(1)).max(1);
  if Self::with_storage(elements,world,shards,capacity,1,generators,generator_bytes,state_bytes)?.device_bytes>budget{return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());}
  while lo<hi {let mid=lo+(hi-lo+1)/2;if Self::with_storage(elements,world,shards,capacity,mid,generators,generator_bytes,state_bytes)?.device_bytes<=budget{lo=mid;}else{hi=mid-1;}}
  Self::with_storage(elements,world,shards,capacity,lo,generators,generator_bytes,state_bytes)
 }
 pub fn packed_candidates(&self)->bool{self.state_bytes==1&&self.elements<=16}
 pub fn queue_payload_bytes(&self)->usize{if self.packed_candidates(){0}else{self.elements as usize*self.state_bytes as usize}}
 pub fn validate(&self,generators:u32)->Result<()> {
  if *self!=Self::with_storage(self.elements,self.world,self.shards,self.capacity,self.batch,generators,self.generator_bytes,self.state_bytes)?{return Err("GENERIC_DISTRIBUTED_PLAN_MUTATED".into());}Ok(())
 }
}

/// Independent per-rank admission with matched transport geometry and exact
/// weighted hash intervals. No runtime state crosses the host planner.
pub fn heterogeneous_plans(elements:u32,shards:u32,generators:u32,generator_bytes:u64,free:&[u64],upper:Option<u64>,state_bytes:u32)->Result<(Vec<GenericDistributedMemoryPlan>,Vec<u64>)>{
 if free.is_empty()||free.len()>128{return Err("GENERIC_HETEROGENEOUS_WORLD".into());}
 let world=free.len() as u32;let mut admitted=Vec::new();
 for &bytes in free{admitted.push(GenericDistributedMemoryPlan::automatic_storage(elements,world,shards,generators,generator_bytes,bytes,upper,state_bytes)?);}
 let batch=admitted.iter().map(|p|p.batch).min().unwrap();let mut plans=Vec::new();
 for p in admitted{plans.push(GenericDistributedMemoryPlan::with_storage(elements,world,shards,p.capacity,batch,generators,generator_bytes,state_bytes)?);}
 let total: u64=plans.iter().map(|p|u64::from(p.capacity)).sum();let mut cuts=vec![0];let mut cumulative=0u64;
 for p in &plans{cumulative+=u64::from(p.capacity);cuts.push(((u128::from(cumulative)*(1u128<<32)+u128::from(total)-1)/u128::from(total)) as u64);}
 Ok((plans,cuts))
}
