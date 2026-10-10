//! Byte-exact admission for retained-history routing. This is admission,
//! not a measured throughput tuner. Skew is handled by immutable-source batch retry before owner mutation.
use mgbfs_core::Result;
use crate::generic_memory::MAX_ARENA_CAPACITY;
const MAX_ROLLING_CAPACITY:u32=(u32::MAX-3)/3;
#[derive(Clone,Debug,PartialEq,Eq,serde::Serialize,serde::Deserialize)]
pub struct GenericDistributedMemoryPlan {
 #[serde(default="cuda_generator")]pub generator_backend:String,
 #[serde(default="hash_history_algorithm")]pub history_algorithm:String,#[serde(default)]pub owner_lanes:u32,#[serde(default)]pub sorted_owner_bytes:u64,#[serde(default)]pub driver_graph_reserve_bytes:u64,
 #[serde(default)]pub parent_transport:bool,#[serde(default)]pub sort_candidates:bool,
 #[serde(default="retained_history_layers")]pub history_layers:u32, pub state_bytes:u32,pub elements:u32,pub world:u32,pub shards:u32,pub capacity:u32,pub batch:u32,
 pub table_layout:String,pub queue_capacity:u32,pub table_slots:u64,pub generator_bytes:u64,pub device_bytes:u64,
}
fn retained_history_layers()->u32{1}
fn cuda_generator()->String{"cuda".into()}
fn hash_history_algorithm()->String{"HASH".into()}
impl GenericDistributedMemoryPlan {
 pub fn new(elements:u32,world:u32,shards:u32,capacity:u32,batch:u32,generators:u32,generator_bytes:u64)->Result<Self>{Self::with_storage(elements,world,shards,capacity,batch,generators,generator_bytes,8)}
 pub fn with_storage(elements:u32,world:u32,shards:u32,capacity:u32,batch:u32,generators:u32,generator_bytes:u64,state_bytes:u32)->Result<Self>{
  if ![1,8].contains(&state_bytes)||elements==0||!(1..=128).contains(&world)||!(1..=4096).contains(&shards)||capacity==0||capacity>MAX_ARENA_CAPACITY||batch==0||batch>capacity||generators==0 {return Err("GENERIC_DISTRIBUTED_SHAPE".into());}
  if crate::generic_gemm::requested()?&&u64::from(batch)*u64::from(if state_bytes==1{1}else{elements})>1048576{return Err("GEMM_BATCH_EXCEEDS_ADMISSION".into());}
  let full=batch.checked_mul(generators).filter(|v|*v<0x80000000).ok_or("GENERIC_QUEUE_INDEX_RANGE")?;
  let boxes=u64::from(world)*u64::from(shards);
  let balanced=((u64::from(full)+boxes-1)/boxes).saturating_mul(4).max(u64::from(generators));
  let queue_capacity=u64::from(full).min(balanced) as u32;
  if u64::from(queue_capacity)*u64::from(world)*u64::from(shards)>=0x7fffffff{return Err("GENERIC_QUEUE_INDEX_RANGE".into());}
  // Logical shard hash regions share one exact table and overflow reserve.
  // Immutable pending origins include the source queue AND shard.
  let table_slots=u64::from(capacity).checked_mul(2).and_then(u64::checked_next_power_of_two).ok_or("GENERIC_SHARD_SLOTS")?;
  let checked=||->Option<u64>{
   let arena=u64::from(elements).checked_mul(u64::from(capacity))?.checked_mul(u64::from(state_bytes)*2)?.checked_add(u64::from(capacity)*8)?;
   let tables=u64::from(table_slots).checked_mul(8)?;
   let boxes=u64::from(world).checked_mul(u64::from(shards))?;
   let payload=if state_bytes==1&&elements<=24{0}else{u64::from(elements)*u64::from(state_bytes)};
   let queue=u64::from(queue_capacity).checked_mul(32+payload)?.checked_add(4)?;
   // Two source banks and one received inbox for every source. Control,
   // graph tables and rank count gather are included, not hidden headroom.
   arena.checked_add(tables)?.checked_add(boxes.checked_mul(queue)?.checked_mul(3)?)?.checked_add(generator_bytes)?.checked_add(98+u64::from(world)*20+if state_bytes==1&&elements<=24{3}else{0})
  };
  let device_bytes=checked().ok_or("GENERIC_DISTRIBUTED_BYTES_OVERFLOW")?.checked_add(crate::generic_gemm::workspace(elements,generators,batch)?).ok_or("GEMM_WORKSPACE_OVERFLOW")?;
  Self{generator_backend:if crate::generic_gemm::requested()?{"gemm".into()}else{cuda_generator()},history_algorithm:hash_history_algorithm(),owner_lanes:0,sorted_owner_bytes:0,driver_graph_reserve_bytes:0,parent_transport:false,sort_candidates:false,history_layers:1,table_layout:"SHARD_HASH_REGIONS_SHARED_OVERFLOW".into(),state_bytes,elements,world,shards,capacity,batch,queue_capacity,table_slots,generator_bytes,device_bytes}.with_parent_transport(match std::env::var("MGBFS_GENERIC_TRANSPORT").as_deref(){Ok("parent")=>true,Ok("full")|Err(_)=>false,_=>return Err("GENERIC_TRANSPORT_MODE".into())})?.with_sort_candidates(match std::env::var("MGBFS_GENERIC_SORT").as_deref(){Ok("radix")=>true,Ok("none")|Err(_)=>false,_=>return Err("GENERIC_SORT_MODE".into())})
 }
 pub fn automatic(elements:u32,world:u32,shards:u32,generators:u32,generator_bytes:u64,free_bytes:u64,upper_bound:Option<u64>)->Result<Self>{Self::automatic_storage(elements,world,shards,generators,generator_bytes,free_bytes,upper_bound,8)}
 pub fn automatic_storage(elements:u32,world:u32,shards:u32,generators:u32,generator_bytes:u64,free_bytes:u64,upper_bound:Option<u64>,state_bytes:u32)->Result<Self>{
  let reserve=(1u64<<30).max(free_bytes/10);let budget=free_bytes.checked_sub(reserve).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
  let ceiling=upper_bound.unwrap_or(u64::from(MAX_ARENA_CAPACITY)).min(u64::from(MAX_ARENA_CAPACITY)) as u32;
  if ceiling==0{return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());}
  // Reserve at most a quarter for predictable transport, then admit the
  // largest history arena with the actual bounded queues included.
  let ceiling_batch=ceiling.min((1<<20)/generators.max(1)).min(if crate::generic_gemm::requested()?{(1<<20)/if state_bytes==1{1}else{elements.max(1)}}else{u32::MAX}).max(1);
  let minimum=Self::with_storage(elements,world,shards,1,1,generators,generator_bytes,state_bytes)?;
  if minimum.device_bytes>budget{return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());}
  let queue_budget=(budget/4).max(minimum.transport_bytes());
  let mut lo=1;let mut hi=ceiling_batch;
  while lo<hi{let mid=lo+(hi-lo+1)/2;let p=Self::with_storage(elements,world,shards,mid,mid,generators,generator_bytes,state_bytes)?;if p.transport_bytes()<=queue_budget{lo=mid;}else{hi=mid-1;}}
  Self::automatic_storage_fixed_batch(elements,world,shards,generators,generator_bytes,free_bytes,upper_bound,state_bytes,lo)
 }
 pub fn automatic_storage_fixed_batch(elements:u32,world:u32,shards:u32,generators:u32,generator_bytes:u64,free_bytes:u64,upper_bound:Option<u64>,state_bytes:u32,batch_target:u32)->Result<Self>{
  if batch_target==0{return Err("GENERIC_BATCH_TARGET".into());}
  let reserve=(1u64<<30).max(free_bytes/10);let budget=free_bytes.checked_sub(reserve).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
  let ceiling=upper_bound.unwrap_or(u64::from(MAX_ARENA_CAPACITY)).min(u64::from(MAX_ARENA_CAPACITY)) as u32;if ceiling==0{return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());}
  if Self::with_storage(elements,world,shards,1,1,generators,generator_bytes,state_bytes)?.device_bytes>budget{return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());}
  let mut low=1;let mut high=ceiling;
  while low<high{let mid=low+(high-low+1)/2;let p=Self::with_storage(elements,world,shards,mid,mid.min(batch_target).min(if crate::generic_gemm::requested()?{(1<<20)/if state_bytes==1{1}else{elements.max(1)}}else{u32::MAX}),generators,generator_bytes,state_bytes)?;if p.device_bytes<=budget{low=mid;}else{high=mid-1;}}
  Self::with_storage(elements,world,shards,low,low.min(batch_target).min(if crate::generic_gemm::requested()?{(1<<20)/if state_bytes==1{1}else{elements.max(1)}}else{u32::MAX}),generators,generator_bytes,state_bytes)
 }
 pub fn transport_bytes(&self)->u64{u64::from(self.world)*u64::from(self.shards)*(u64::from(self.queue_capacity)*(32+self.queue_payload_bytes() as u64)+4)*3+if self.packed_candidates()||self.parent_transport{3}else{0}+self.parent_cache_bytes()+self.sort_cache_bytes()}
 pub fn packed_candidates(&self)->bool{self.state_bytes==1&&self.elements<=24}
 pub fn queue_payload_bytes(&self)->usize{if self.packed_candidates()||self.parent_transport{0}else{self.elements as usize*self.state_bytes as usize}}
 pub fn parent_cache_bytes(&self)->u64{if self.parent_transport{u64::from(self.world+2)*u64::from(self.batch)*u64::from(self.elements)*u64::from(self.state_bytes)+u64::from(self.world)*12}else{0}}
 pub fn sort_cache_bytes(&self)->u64{if self.sort_candidates{u64::from(self.shards)*(u64::from(self.world)*u64::from(self.queue_capacity)*88+65536)}else{0}}
 pub fn with_sort_candidates(mut self,enabled:bool)->Result<Self>{if enabled==self.sort_candidates{return Ok(self);}let old=self.transport_bytes();self.sort_candidates=enabled;let new=self.transport_bytes();self.device_bytes=self.device_bytes.checked_sub(old).and_then(|b|b.checked_add(new)).ok_or("GENERIC_SORT_TRANSPORT_BYTES")?;Ok(self)}
 pub fn with_parent_transport(mut self,enabled:bool)->Result<Self>{
  let enabled=enabled&&!self.packed_candidates();if enabled==self.parent_transport{return Ok(self);}
  let old=self.transport_bytes();self.parent_transport=enabled;let new=self.transport_bytes();
  self.device_bytes=self.device_bytes.checked_sub(old).and_then(|b|b.checked_add(new)).ok_or("GENERIC_PARENT_TRANSPORT_BYTES")?;Ok(self)
 }

 pub fn with_storage_history(elements:u32,world:u32,shards:u32,capacity:u32,batch:u32,generators:u32,generator_bytes:u64,state_bytes:u32,history_layers:u32)->Result<Self>{
  if history_layers==3&&capacity>MAX_ROLLING_CAPACITY{return Err("GENERIC_ROLLING_ROW_RANGE".into());}
  if ![1,3].contains(&history_layers){return Err("GENERIC_HISTORY_GEOMETRY".into());}
  let mut p=Self::with_storage(elements,world,shards,capacity,batch,generators,generator_bytes,state_bytes)?;
  if history_layers==3{
   let old=p.table_slots;let slots=u64::from(capacity).checked_mul(6).and_then(u64::checked_next_power_of_two).ok_or("GENERIC_ROLLING_TABLE")?;
   p.device_bytes=p.device_bytes.checked_add(u64::from(elements)*u64::from(capacity+capacity.min(1000))*u64::from(state_bytes)+u64::from(capacity)*24+u64::from(slots-old)*8).ok_or("GENERIC_ROLLING_BYTES")?;
   p.table_slots=slots;p.table_layout="THREE_BANK_SHARED_TOMBSTONE_HISTORY".into();
  }
  p.history_layers=history_layers;Ok(p)
 }
 pub fn automatic_storage_history(elements:u32,world:u32,shards:u32,generators:u32,generator_bytes:u64,free_bytes:u64,upper_bound:Option<u64>,state_bytes:u32,history_layers:u32)->Result<Self>{
  let reserve=(1u64<<30).max(free_bytes/10);let budget=free_bytes.checked_sub(reserve).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
  let ceiling=upper_bound.unwrap_or(u64::from(MAX_ARENA_CAPACITY)).min(u64::from(if history_layers==3{MAX_ROLLING_CAPACITY}else{MAX_ARENA_CAPACITY})) as u32;
  if ceiling==0{return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());}
  // Reserve at most a quarter for predictable transport, then admit the
  // largest history arena with the actual bounded queues included.
  let ceiling_batch=ceiling.min((1<<20)/generators.max(1)).min(if crate::generic_gemm::requested()?{(1<<20)/if state_bytes==1{1}else{elements.max(1)}}else{u32::MAX}).max(1);
  let minimum=Self::with_storage_history(elements,world,shards,1,1,generators,generator_bytes,state_bytes,history_layers)?;
  if minimum.device_bytes>budget{return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());}
  let queue_budget=(budget/4).max(minimum.transport_bytes());
  let mut lo=1;let mut hi=ceiling_batch;
  while lo<hi{let mid=lo+(hi-lo+1)/2;let p=Self::with_storage_history(elements,world,shards,mid,mid,generators,generator_bytes,state_bytes,history_layers)?;if p.transport_bytes()<=queue_budget{lo=mid;}else{hi=mid-1;}}
  Self::automatic_storage_fixed_batch_history(elements,world,shards,generators,generator_bytes,free_bytes,upper_bound,state_bytes,lo,history_layers)
 }
 pub fn automatic_storage_fixed_batch_history(elements:u32,world:u32,shards:u32,generators:u32,generator_bytes:u64,free_bytes:u64,upper_bound:Option<u64>,state_bytes:u32,batch_target:u32,history_layers:u32)->Result<Self>{
  if batch_target==0{return Err("GENERIC_BATCH_TARGET".into());}
  let reserve=(1u64<<30).max(free_bytes/10);let budget=free_bytes.checked_sub(reserve).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;
  let ceiling=upper_bound.unwrap_or(u64::from(MAX_ARENA_CAPACITY)).min(u64::from(if history_layers==3{MAX_ROLLING_CAPACITY}else{MAX_ARENA_CAPACITY})) as u32;if ceiling==0{return Err("GENERIC_DISTRIBUTED_EMPTY_BOUND".into());}
  if Self::with_storage_history(elements,world,shards,1,1,generators,generator_bytes,state_bytes,history_layers)?.device_bytes>budget{return Err("GENERIC_DISTRIBUTED_NO_CAPACITY".into());}
  let mut low=1;let mut high=ceiling;
  while low<high{let mid=low+(high-low+1)/2;let p=Self::with_storage_history(elements,world,shards,mid,mid.min(batch_target).min(if crate::generic_gemm::requested()?{(1<<20)/if state_bytes==1{1}else{elements.max(1)}}else{u32::MAX}),generators,generator_bytes,state_bytes,history_layers)?;if p.device_bytes<=budget{low=mid;}else{high=mid-1;}}
  Self::with_storage_history(elements,world,shards,low,low.min(batch_target).min(if crate::generic_gemm::requested()?{(1<<20)/if state_bytes==1{1}else{elements.max(1)}}else{u32::MAX}),generators,generator_bytes,state_bytes,history_layers)
 }
 pub fn with_sorted_admission(mut self,lanes:u32,owner_bytes:u64,driver_bytes:u64)->Result<Self>{
  if self.history_algorithm!="HASH"||lanes==0||lanes>self.shards||lanes>8||owner_bytes==0||driver_bytes<(128u64<<20).max(u64::from(self.shards)*u64::from(self.history_layers)*65536){return Err("SORTED_PLAN_ADMISSION".into());}
  self=self.with_sort_candidates(false)?;
  let obsolete=u64::from(self.table_slots)*8+if self.history_layers==3{u64::from(self.capacity)*24}else{0};
  // Buffer::new(0) has a one-byte allocation for the unused table handle.
  self.device_bytes=self.device_bytes.checked_sub(obsolete).and_then(|v|v.checked_add(1)).and_then(|v|v.checked_add(owner_bytes)).and_then(|v|v.checked_add(driver_bytes)).ok_or("SORTED_PLAN_BYTES")?;
  self.table_slots=0;self.table_layout="SORTED_EPOCH_RUNS_SHARED_CREDITS".into();self.history_algorithm="SORTED_RUNS".into();self.owner_lanes=lanes;self.sorted_owner_bytes=owner_bytes;self.driver_graph_reserve_bytes=driver_bytes;Ok(self)
 }
 #[cfg(feature="cuda")]
 pub fn with_sorted_for_graph(self,graph:&mgbfs_core::graph_definition::GraphDefinitionV2,device:i32,lanes:u32)->Result<Self>{
  use mgbfs_core::graph_definition::GraphAction;use crate::generic_sorted_native::{Input,OwnerShape};
  if graph.start.len()!=self.elements as usize{return Err("SORTED_GRAPH_SHAPE".into());}
  let(kind,rows,cols)=match &graph.action{GraphAction::Permutation{degree,..}=>(0,*degree,1),GraphAction::Matrix{rows,cols,..}=>(1,*rows,*cols)};
  let input=Input{elements:self.elements,world:self.world,shards:self.shards,queue_capacity:self.queue_capacity,state_bytes:self.state_bytes,transport:if self.packed_candidates(){1}else if self.parent_transport{2}else{0},kind,rows,cols,generators:graph.generator_count() as u32,parent_stride:self.batch,hash_bits:64,..Default::default()};
  let shape=OwnerShape::query(&input,self.capacity,self.history_layers,lanes,device)?;
  let reserve=(128u64<<20).max(u64::from(self.shards)*u64::from(self.history_layers)*65536);
  self.with_sorted_admission(lanes,shape.allocated_bytes as u64,reserve)
 }
 #[cfg(feature="cuda")]
 pub fn automatic_sorted_for_graph(graph:&mgbfs_core::graph_definition::GraphDefinitionV2,device:i32,world:u32,shards:u32,generator_bytes:u64,free:u64,upper:Option<u64>,bytes:u32,batch_target:u32,banks:u32,lanes:u32)->Result<Self>{
  if batch_target==0{return Err("SORTED_BATCH_TARGET".into());}
  let budget=free.checked_sub((1u64<<30).max(free/10)).ok_or("SORTED_ADMISSION_HEADROOM")?;
  let ceiling=upper.unwrap_or(u64::from(MAX_ARENA_CAPACITY)).min(u64::from(if banks==3{MAX_ROLLING_CAPACITY}else{MAX_ARENA_CAPACITY})) as u32;if ceiling==0{return Err("SORTED_EMPTY_BOUND".into());}
  let candidate=|cap:u32|->Result<Self>{Self::with_storage_history(graph.start.len() as u32,world,shards,cap,cap.min(batch_target),graph.generator_count() as u32,generator_bytes,bytes,banks)?.with_sorted_for_graph(graph,device,lanes)};
  if candidate(1)?.device_bytes>budget{return Err("SORTED_ADMISSION_NO_CAPACITY".into());}
  let(mut low,mut high)=(1,ceiling);
  while low<high{let mid=low+(high-low+1)/2;match candidate(mid){Ok(p) if p.device_bytes<=budget=>low=mid,_=>high=mid-1}}
  candidate(low)
 }
 pub fn validate(&self,generators:u32)->Result<()> {
  let mut expected=Self::with_storage_history(self.elements,self.world,self.shards,self.capacity,self.batch,generators,self.generator_bytes,self.state_bytes,self.history_layers)?.with_parent_transport(self.parent_transport)?.with_sort_candidates(self.sort_candidates)?;
  if self.history_algorithm=="SORTED_RUNS"{expected=expected.with_sorted_admission(self.owner_lanes,self.sorted_owner_bytes,self.driver_graph_reserve_bytes)?;}
  if *self!=expected{return Err("GENERIC_DISTRIBUTED_PLAN_MUTATED".into());}Ok(())
 }

}

/// Independent per-rank admission with matched transport geometry and exact
/// weighted hash intervals. No runtime state crosses the host planner.
pub fn heterogeneous_plans(elements:u32,shards:u32,generators:u32,generator_bytes:u64,free:&[u64],upper:Option<u64>,state_bytes:u32)->Result<(Vec<GenericDistributedMemoryPlan>,Vec<u64>)>{
 if free.is_empty()||free.len()>128{return Err("GENERIC_HETEROGENEOUS_WORLD".into());}
 let world=free.len() as u32;let mut admitted=Vec::new();
 for &bytes in free{admitted.push(GenericDistributedMemoryPlan::automatic_storage(elements,world,shards,generators,generator_bytes,bytes,upper,state_bytes)?);}
 let batch=admitted.iter().map(|p|p.batch).min().unwrap();let mut plans=Vec::new();
 for &bytes in free{plans.push(GenericDistributedMemoryPlan::automatic_storage_fixed_batch(elements,world,shards,generators,generator_bytes,bytes,upper,state_bytes,batch)?);}
 let total: u64=plans.iter().map(|p|u64::from(p.capacity)).sum();let mut cuts=vec![0];let mut cumulative=0u64;
 for p in &plans{cumulative+=u64::from(p.capacity);cuts.push(((u128::from(cumulative)*(1u128<<32)+u128::from(total)-1)/u128::from(total)) as u64);}
 Ok((plans,cuts))
}

pub fn heterogeneous_plans_history(elements:u32,shards:u32,generators:u32,generator_bytes:u64,free:&[u64],upper:Option<u64>,state_bytes:u32,history_layers:u32)->Result<(Vec<GenericDistributedMemoryPlan>,Vec<u64>)>{
 if free.is_empty()||free.len()>128{return Err("GENERIC_HETEROGENEOUS_WORLD".into());}
 let world=free.len() as u32;let mut admitted=Vec::new();
 for &bytes in free{admitted.push(GenericDistributedMemoryPlan::automatic_storage_history(elements,world,shards,generators,generator_bytes,bytes,upper,state_bytes,history_layers)?);}
 let batch=admitted.iter().map(|p|p.batch).min().unwrap();let mut plans=Vec::new();
 for &bytes in free{plans.push(GenericDistributedMemoryPlan::automatic_storage_fixed_batch_history(elements,world,shards,generators,generator_bytes,bytes,upper,state_bytes,batch,history_layers)?);}
 let total: u64=plans.iter().map(|p|u64::from(p.capacity)).sum();let mut cuts=vec![0];let mut cumulative=0u64;
 for p in &plans{cumulative+=u64::from(p.capacity);cuts.push(((u128::from(cumulative)*(1u128<<32)+u128::from(total)-1)/u128::from(total)) as u64);}
 Ok((plans,cuts))
}
