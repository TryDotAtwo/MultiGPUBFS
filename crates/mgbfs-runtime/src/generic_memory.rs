use mgbfs_core::Result;
/// Exact retained-history arena for general directed CayleyPy graphs.
/// Separate from the compact inverse-closed specialized SHARD_AB layout.
#[derive(Clone,Debug)]
pub struct GenericMemoryPlan {pub state_bytes:u32,pub elements:u32,pub capacity:u32,pub table_slots:u32,pub batch:u32,pub device_bytes:u64}
impl GenericMemoryPlan {
 pub fn new(elements:u32,capacity:u32)->Result<Self>{Self::with_storage(elements,capacity,8)}
 pub fn with_storage(elements:u32,capacity:u32,state_bytes:u32)->Result<Self>{
  if ![1,8].contains(&state_bytes)||elements==0||capacity==0||capacity>(1<<28){return Err("GENERIC_ARENA_SHAPE".into());}
  let slots=capacity.checked_mul(2).and_then(u32::checked_next_power_of_two).ok_or("GENERIC_TABLE_SHAPE")?;
  let bytes=u64::from(elements).checked_mul(u64::from(capacity)).and_then(|v|v.checked_mul(u64::from(state_bytes)*2)).and_then(|v|v.checked_add(u64::from(capacity)*8)).and_then(|v|v.checked_add(u64::from(slots)*8+24)).ok_or("GENERIC_ARENA_OVERFLOW")?;
  Ok(Self{state_bytes,elements,capacity,table_slots:slots,batch:capacity.min(1<<20),device_bytes:bytes})
 }
 pub fn automatic(elements:u32,free_bytes:u64,upper_bound:Option<u64>)->Result<Self>{Self::automatic_storage(elements,free_bytes,upper_bound,8)}
 pub fn automatic_storage(elements:u32,free_bytes:u64,upper_bound:Option<u64>,state_bytes:u32)->Result<Self>{
  let reserve=(1u64<<30).max(free_bytes/10);let available=free_bytes.checked_sub(reserve).ok_or("GENERIC_VRAM_HEADROOM")?;
  let mut low=1u32;let mut high=upper_bound.unwrap_or(1<<28).min(1<<28) as u32;
  if high==0||Self::with_storage(elements,1,state_bytes)?.device_bytes>available{return Err("GENERIC_VRAM_CAPACITY".into());}
  while low<high {let mid=low+(high-low+1)/2;if Self::with_storage(elements,mid,state_bytes)?.device_bytes<=available{low=mid;}else{high=mid-1;}}
  Self::with_storage(elements,low,state_bytes)
 }
}

/// Mathematical state-space bound, capped only at the arena index limit.
/// Used for admission, not as evidence that the graph has been enumerated.
pub fn state_space_bound(graph:&mgbfs_core::graph_definition::GraphDefinitionV2)->Option<u64>{
 use mgbfs_core::graph_definition::GraphAction;let limit=1u64<<28;
 let proven=match &graph.action{
  GraphAction::Permutation{..}=>{
   let mut counts=std::collections::BTreeMap::<i64,u64>::new();for &v in &graph.start{*counts.entry(v).or_default()+=1;}
   let mut remaining=graph.start.len() as u64;let mut bound=1u64;
   for &count in counts.values(){let k=count.min(remaining-count);let mut choose=1u64;for i in 1..=k{choose=choose.saturating_mul(remaining-k+i)/i;if choose>=limit{choose=limit;break;}}bound=bound.saturating_mul(choose).min(limit);remaining-=count;}
   Some(bound)
  },
  GraphAction::Matrix{generators,..}=>{
   let modulus=generators.first().map(|v|v.modulo).unwrap_or(0);
   if modulus==0||generators.iter().any(|v|v.modulo!=modulus){None}else{
    let mut bound=1u64;for _ in &graph.start{bound=bound.saturating_mul(u64::from(modulus)).min(limit);if bound==limit{break;}}
    if graph.start.iter().any(|&v|v<0||v>=i64::from(modulus)){bound=bound.saturating_add(1).min(limit);}Some(bound)
   }
  },
 };
 match (proven,graph.expected_max_unique_states){(Some(a),Some(b))=>Some(a.min(b)),(a,b)=>a.or(b)}
}

/// Permutations preserve the value alphabet exactly; matrix arithmetic does not.
pub fn preferred_state_bytes(graph:&mgbfs_core::graph_definition::GraphDefinitionV2)->u32{
 if matches!(graph.action,mgbfs_core::graph_definition::GraphAction::Permutation{..})&&graph.start.iter().all(|v|(0..=255).contains(v)){1}else{8}
}
