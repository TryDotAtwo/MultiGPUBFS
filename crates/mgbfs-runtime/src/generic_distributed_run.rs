//! Single-host distributed launch worker. No CPU state generation or dedup.
use std::{path::Path,time::{Duration,Instant},sync::atomic::Ordering};
use mgbfs_core::{graph_definition::{GraphDefinitionV2,GraphAction},Result};
use mgbfs_cuda::{ffi::*,native_owner::{cudaSetDevice,cudaMemGetInfo}};
use crate::{generic_native::check,generic_distributed_memory::{GenericDistributedMemoryPlan as Plan,HistoryMemory},generic_distributed_native::{GenericDistributedBfs,DistributedAdvance},generic_run::{Signals,CANCELLED}};
use crate::segmented_packet_transport::{parse_job_token,PacketJobToken};
extern "C" {fn mgbfs_generic_hardware_info(device:i32,values:*mut u64)->i32;}
fn graph(path:&str)->Result<GraphDefinitionV2>{let g:GraphDefinitionV2=serde_json::from_reader(std::fs::File::open(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;g.validate()?;Ok(g)}
fn digest(g:&GraphDefinitionV2)->Result<String>{Ok(g.semantic_digest()?.iter().map(|v|format!("{v:02x}")).collect())}
fn graph_bytes(g:&GraphDefinitionV2)->u64{match &g.action{GraphAction::Permutation{generators,..}=>generators.iter().map(|v|v.len() as u64*4).sum(),GraphAction::Matrix{generators,..}=>generators.iter().map(|v|v.matrix.len() as u64*8+4).sum()}}

fn segmented_byte_dictionary(graph: &GraphDefinitionV2) -> Option<Vec<i64>> {
 if std::env::var("MGBFS_SEGMENT_RESIDENT_ALPHABET8").as_deref() != Ok("1")
  || std::env::var("MGBFS_EXPERIMENTAL_SEGMENTED24").as_deref() != Ok("1")
  || std::env::var("MGBFS_EXPERIMENTAL_BITMAP").as_deref() == Ok("1")
  || !matches!(&graph.action, GraphAction::Permutation { .. })
  || graph.start.len() <= 24
  || crate::generic_memory::preferred_state_bytes(graph) == 1
 {
  return None;
 }
 let symbols = graph.start.iter().copied().collect::<std::collections::BTreeSet<_>>();
 (symbols.len() <= 256).then(|| symbols.into_iter().collect())
}
extern "C"{fn cudaGetDeviceCount(count:*mut i32)->i32;}
/// Startup-only diagnostic checkpoint; no GPU dispatch or allocation.
pub fn startup_trace(phase:&str)->Result<()>{crate::job_phase::set(phase)}
pub fn visible_count()->Result<()>{
 let mut count=0;
 startup_trace("graph-count-cudaGetDeviceCount-enter")?;
 let status=unsafe{cudaGetDeviceCount(&mut count)};
 startup_trace("graph-count-cudaGetDeviceCount-returned")?;
 crate::generic_native::checked!(cudaGetDeviceCount, status)?;
 if count<=0{return Err("NO_VISIBLE_CUDA_DEVICES".into());}
 println!("{count}");Ok(())
}

pub fn info(args:&[String])->Result<()>{
 crate::job_phase::set("graph-info-read-definition")?;
 if args.len()<3{return Err("CLI_GRAPH_INFO_ARGUMENTS".into());}let g=graph(&args[0])?;let shards:u32=args[2].parse().map_err(|_|"CLI_GRAPH_SHARDS")?;
 let mut count=0i32;crate::generic_native::checked!(cudaGetDeviceCount, unsafe{cudaGetDeviceCount(&mut count)})?;if count<=0{return Err("NO_VISIBLE_CUDA_DEVICES".into());}
 let devices=if args[1]=="auto"{(0..count as u32).collect::<Vec<_>>()}else{args[1].split(',').map(|v|v.parse::<u32>().map_err(|_|"CLI_GRAPH_DEVICES".to_owned())).collect::<Result<Vec<_>>>()?};
 if devices.is_empty()||devices.len()>128||devices.iter().any(|&d|d>=count as u32)||devices.iter().enumerate().any(|(i,d)|devices[..i].contains(d)){return Err("CLI_GRAPH_DEVICE_SELECTION".into());}
 let mut inventory=vec![];let mut minimum=u64::MAX;for &device in &devices{crate::job_phase::set("graph-info-cudaSetDevice")?;crate::generic_native::checked!(cudaSetDevice, unsafe{cudaSetDevice(device as i32)})?;crate::job_phase::set("graph-info-cudaMemGetInfo")?;let(mut free,mut total)=(0usize,0usize);crate::generic_native::checked!(cudaMemGetInfo, unsafe{cudaMemGetInfo(&mut free,&mut total)})?;minimum=minimum.min(free as u64);crate::job_phase::set("graph-info-hardware-info")?;let mut hw=[0u64;8];crate::generic_native::checked!(mgbfs_generic_hardware_info, unsafe{mgbfs_generic_hardware_info(device as i32,hw.as_mut_ptr())})?;inventory.push(serde_json::json!({"device":device,"free_bytes":free,"total_bytes":total,"sm_count":hw[0],"compute_major":hw[1],"compute_minor":hw[2],"l2_bytes":hw[3],"warp_size":hw[4],"threads_per_sm":hw[5],"threads_per_block":hw[6],"shared_bytes_per_sm":hw[7]}));}
 let requested=if args.len()>3&&args[3]!="auto"{Some(args[3].parse::<u32>().map_err(|_|"CLI_GRAPH_CAPACITY")?)}else{None};
 let batch=if args.len()>4{Some(args[4].parse::<u32>().map_err(|_|"CLI_GRAPH_BATCH")?)}else{None};
 crate::job_phase::set("graph-info-memory-admission")?;println!("{}",admit(&g,devices,inventory,shards,requested,batch)?);Ok(())
}
fn admit(g:&GraphDefinitionV2,devices:Vec<u32>,inventory:Vec<serde_json::Value>,shards:u32,requested:Option<u32>,batch:Option<u32>)->Result<serde_json::Value>{
 if devices.is_empty()||devices.len()>128||inventory.len()!=devices.len(){return Err("GRAPH_GLOBAL_INVENTORY_SHAPE".into());}
 let mut minimum=u64::MAX;
 let reserve=crate::generic_memory::admission_reserve_bytes()?;
 if inventory.iter().any(|v|v.get("additional_reserve_bytes").and_then(|v|v.as_u64()).map_or(false,|bytes|bytes!=reserve)){return Err("GRAPH_GLOBAL_RESERVE_CONSENSUS".into());}
 for v in &inventory{let free=v["free_bytes"].as_u64().ok_or("GRAPH_GLOBAL_FREE_BYTES")?;let total=v["total_bytes"].as_u64().ok_or("GRAPH_GLOBAL_TOTAL_BYTES")?;if free==0||free>total{return Err("GRAPH_GLOBAL_MEMORY_RANGE".into());}minimum=minimum.min(free);}
 let window=std::env::var("MGBFS_GENERIC_PATH_WINDOW").as_deref()==Ok("1");
 let path_mode=std::env::var("MGBFS_GENERIC_PATH_APPEND_ONLY").as_deref()==Ok("1");
 let extra_row_bytes=if path_mode&&!window{crate::generic_path_native::PARENT_BYTES as u64}else{0};
 let full_segment_history=std::env::var("MGBFS_EXPERIMENTAL_SEGMENTED24").as_deref()==Ok("1")&&std::env::var("MGBFS_SEGMENT_FULL_HISTORY").as_deref()==Ok("1");
 let history_layers=if !path_mode&&!full_segment_history&&g.inverse_closed()?{3}else{1};
 let bidirectional=std::env::var("MGBFS_GENERIC_PATH_BIDIRECTIONAL").as_deref()==Ok("1");
 // Byte states with more than 24 elements already use separate full
 // route metadata. Forward targets are compared as signed int64 and never
 // inserted in the byte arena. Recheck reverse roots at worker launch.
 let byte_targets=std::env::var("MGBFS_GENERIC_PATH_BYTE_TARGETS").as_deref()==Ok("1");
 let wide_path=path_mode&&((bidirectional&&!byte_targets)||g.start.len()<=24);
 let resident_alphabet = !path_mode && segmented_byte_dictionary(g).is_some();
 let state_bytes=if wide_path{8}else if resident_alphabet{1}else{crate::generic_memory::preferred_state_bytes(g)};
 if (bidirectional||window)&&!path_mode{return Err("NATIVE_PATH_REQUIRES_APPEND_ONLY".into());}
 let arenas=if bidirectional{2}else{1};
 let path_workspace=if window{g.start.len() as u64*(8+u64::from(state_bytes))+12+GenericDistributedBfs::WINDOW_ROWS_PER_COPY as u64*GenericDistributedBfs::WINDOW_EDGE_BYTES as u64}else if path_mode{
  crate::generic_path_native::distributed_workspace(g.start.len() as u64,state_bytes,devices.len(),bidirectional)?
   .into_iter().try_fold(0u64,|total,size|total.checked_add(size as u64)).ok_or("NATIVE_PATH_GATHER_SIZE")?
 }else{0};
 let history_memory=HistoryMemory{extra_row_bytes,shared_path_queues:bidirectional};
 let available_for_history=|free:u64|->Result<u64>{
  free.checked_sub(path_workspace).ok_or_else(||"NATIVE_PATH_WORKSPACE_HEADROOM".into())
 };
 
 let sorted=match std::env::var("MGBFS_GENERIC_HISTORY").as_deref(){Ok("sorted")=>true,Ok("hash")|Err(_)=>false,_=>return Err("GENERIC_HISTORY_MODE".into())};
 if path_mode&&sorted{return Err("NATIVE_PATH_REQUIRES_APPEND_ONLY_HASH".into());}
 let (mut rank_plans,owner_cuts)=if sorted{
  if inventory.iter().all(|v|v.get("local_sorted_plan").is_some()){
   let mut plans=Vec::new();for item in &inventory{let p:Plan=serde_json::from_value(item["local_sorted_plan"].clone()).map_err(|e|e.to_string())?;p.validate(g.generator_count() as u32)?;let free=item["free_bytes"].as_u64().unwrap();let budget=free.checked_sub(crate::generic_memory::admission_reserve_bytes()?).ok_or("SORTED_ADMISSION_HEADROOM")?;
    if p.elements!=g.start.len() as u32||p.world!=devices.len() as u32||p.shards!=shards||p.generator_bytes!=graph_bytes(g)||p.state_bytes!=state_bytes||p.history_layers!=history_layers||p.history_algorithm!="SORTED_RUNS"||p.device_bytes>budget||requested.map_or(false,|v|p.capacity!=v)||batch.map_or(false,|v|p.batch!=v){return Err("LOCAL_SORTED_ADMISSION_IDENTITY".into());}plans.push(p);}
   let common=plans.iter().map(|p|p.batch).min().unwrap();if plans.iter().any(|p|p.batch!=common){return Err("LOCAL_SORTED_BATCH_CONSENSUS".into());}
   let cuts=if requested.is_some(){None}else{let total=plans.iter().map(|p|u64::from(p.capacity)).sum::<u64>();let mut cumulative=0u64;let mut cuts=vec![0];for p in &plans{cumulative+=u64::from(p.capacity);cuts.push(((u128::from(cumulative)*(1u128<<32)+u128::from(total)-1)/u128::from(total)) as u64);}Some(cuts)};(plans,cuts)
  }else{
  let lanes=std::env::var("MGBFS_GENERIC_OWNER_LANES").ok().map(|v|v.parse::<u32>().map_err(|_|"SORTED_OWNER_LANES")).transpose()?.unwrap_or(shards.min(4));
  let upper=requested.map(u64::from).or_else(||crate::generic_memory::state_space_bound(g));
  let target=batch.unwrap_or(upper.unwrap_or(u64::from(crate::generic_memory::MAX_ARENA_CAPACITY)).min(u64::from(crate::generic_memory::MAX_ARENA_CAPACITY)).min(u64::from((1<<20)/(g.generator_count() as u32).max(1))) as u32).max(1);
  let mut plans=Vec::new();for(&device,item)in devices.iter().zip(&inventory){plans.push(Plan::automatic_sorted_for_graph(g,device as i32,devices.len() as u32,shards,graph_bytes(g),item["free_bytes"].as_u64().unwrap(),upper,state_bytes,target,history_layers,lanes)?);}
  let common=plans.iter().map(|p|p.batch).min().unwrap();
  if let Some(wanted)=batch{if wanted==0||common!=wanted{return Err("REQUESTED_BATCH_EXCEEDS_SORTED_ADMISSION".into());}}
  for((p,&device),item)in plans.iter_mut().zip(&devices).zip(&inventory){*p=Plan::automatic_sorted_for_graph(g,device as i32,devices.len() as u32,shards,graph_bytes(g),item["free_bytes"].as_u64().unwrap(),upper,state_bytes,common,history_layers,lanes)?;if requested.map_or(false,|v|p.capacity!=v){return Err("REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION".into());}}
  let cuts=if requested.is_some(){None}else{let total=plans.iter().map(|p|u64::from(p.capacity)).sum::<u64>();let mut cuts=vec![0];let mut cumulative=0u64;for p in &plans{cumulative+=u64::from(p.capacity);cuts.push(((u128::from(cumulative)*(1u128<<32)+u128::from(total)-1)/u128::from(total)) as u64);}Some(cuts)};
  (plans,cuts)
  }
 }else if let Some(capacity)=requested{
  let auto=Plan::automatic_storage_with_history_memory(g.start.len() as u32,devices.len() as u32,shards,g.generator_count() as u32,graph_bytes(&g),available_for_history(minimum)?,Some(u64::from(capacity)),state_bytes,history_layers,history_memory)?;
  if auto.capacity!=capacity{return Err("REQUESTED_CAPACITY_EXCEEDS_ADMISSION".into());}
  (vec![auto;devices.len()],None)
 }else{
  let free=inventory.iter().map(|v|available_for_history(v["free_bytes"].as_u64().unwrap())).collect::<Result<Vec<_>>>()?;
  let upper=if bidirectional{None}else{crate::generic_memory::state_space_bound(g)};
  let(plans,cuts)=crate::generic_distributed_memory::heterogeneous_plans_with_history_memory(g.start.len() as u32,shards,g.generator_count() as u32,graph_bytes(&g),&free,upper,state_bytes,history_layers,history_memory)?;(plans,Some(cuts))
 };
 // Lower the transient route workspace without increasing a smaller admitted batch.
 let batch=batch.or_else(||{
  if path_mode&&std::env::var("MGBFS_GENERIC_PATH_MAXMEMORY").as_deref()==Ok("1"){
   rank_plans.iter().map(|p|p.batch).min().filter(|&batch|batch>8192).map(|_|8192)
  }else{None}
 });
 if let Some(batch)=batch.filter(|_|!sorted){
  for(p,item)in rank_plans.iter_mut().zip(&inventory){let candidate=Plan::with_storage_history(p.elements,p.world,p.shards,p.capacity,batch,g.generator_count() as u32,p.generator_bytes,state_bytes,history_layers)?;let free=item["free_bytes"].as_u64().unwrap();let budget=available_for_history(free)?.checked_sub(reserve).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;if history_memory.required_bytes(&candidate)?>budget{return Err("REQUESTED_BATCH_EXCEEDS_ADMISSION".into());}*p=candidate;}
 }
 let plan=rank_plans.iter().min_by_key(|p|p.capacity).unwrap().clone();
 let shared_path_queue_bytes=bidirectional.then(||plan.shared_path_queue_bytes()).transpose()?;
 Ok(serde_json::json!({"size_profile_capability":1,"graph_digest":digest(g)?,"devices":devices,"inventory":inventory,"plan":plan,"rank_plans":rank_plans,"path_arena_copies":arenas,"path_workspace_bytes":path_workspace,"shared_path_queue_bytes":shared_path_queue_bytes,"owner_cuts":owner_cuts,"additional_reserve_bytes":crate::generic_memory::admission_reserve_bytes()?,"reserve_policy":"ABSOLUTE_BYTES_COLD_2GIB_WARM_256MIB","profile_status":"MEMORY_ADMITTED_NOT_THROUGHPUT_TUNED"}))
}
/// Query only this rank's physical GPU; remote IDs must never be queried here.
pub fn local_info(args:&[String])->Result<()>{println!("{}",local_plan(args)?);Ok(())}
pub fn local_plan(args:&[String])->Result<serde_json::Value>{
 if args.len()!=6{return Err("CLI_GRAPH_LOCAL_PLAN_ARGUMENTS".into());}
 let g=graph(&args[0])?;let device:i32=args[1].parse().map_err(|_|"CLI_GRAPH_DEVICE")?;
 let world:u32=args[2].parse().map_err(|_|"CLI_GRAPH_WORLD")?;let shards:u32=args[3].parse().map_err(|_|"CLI_GRAPH_SHARDS")?;
 if device<0||world==0||world>128{return Err("CLI_GRAPH_LOCAL_PLAN_GEOMETRY".into());}
 if std::env::var("MGBFS_GENERIC_HISTORY").as_deref()!=Ok("sorted"){return Err("LOCAL_SORTED_PLAN_REQUIRES_SORTED_HISTORY".into());}
 let requested=if args[4]=="auto"{None}else{Some(args[4].parse::<u32>().map_err(|_|"CLI_GRAPH_CAPACITY")?)};
 let target=args[5].parse::<u32>().map_err(|_|"CLI_GRAPH_BATCH")?;if target==0{return Err("CLI_GRAPH_BATCH".into());}
 crate::generic_native::checked!(cudaSetDevice, unsafe{cudaSetDevice(device)})?;crate::job_phase::set("graph-info-cudaMemGetInfo")?;let(mut free,mut total)=(0usize,0usize);crate::generic_native::checked!(cudaMemGetInfo, unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
 let free=free as u64+crate::session_cache::reusable_bytes()+crate::generic_pool::available()?;let lanes=std::env::var("MGBFS_GENERIC_OWNER_LANES").ok().map(|v|v.parse::<u32>().map_err(|_|"SORTED_OWNER_LANES")).transpose()?.unwrap_or(shards.min(4));
 let upper=requested.map(u64::from).or_else(||crate::generic_memory::state_space_bound(&g));let bytes=crate::generic_memory::preferred_state_bytes(&g);let banks=if g.inverse_closed()?{3}else{1};
 let plan=Plan::automatic_sorted_for_graph(&g,device,world,shards,graph_bytes(&g),free as u64,upper,bytes,target,banks,lanes)?;
 if requested.map_or(false,|v|plan.capacity!=v){return Err("REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION".into());}
 crate::job_phase::set("graph-info-hardware-info")?;let mut hw=[0u64;8];crate::generic_native::checked!(mgbfs_generic_hardware_info, unsafe{mgbfs_generic_hardware_info(device,hw.as_mut_ptr())})?;Ok(serde_json::json!({"size_profile_capability":1,"graph_digest":digest(&g)?,"device":device,"free_bytes":free,"total_bytes":total,"sm_count":hw[0],"compute_major":hw[1],"compute_minor":hw[2],"l2_bytes":hw[3],"plan":plan}))
}
pub fn global_info(args:&[String])->Result<()>{println!("{}",global_plan(args)?);Ok(())}
pub fn global_plan(args:&[String])->Result<serde_json::Value>{
 if args.len()<3{return Err("CLI_GRAPH_GLOBAL_ARGUMENTS".into());}let g=graph(&args[0])?;let inventory:Vec<serde_json::Value>=serde_json::from_reader(std::fs::File::open(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let devices=inventory.iter().map(|v|v["device"].as_u64().filter(|&d|d<=u64::from(u32::MAX)).map(|d|d as u32).ok_or("GRAPH_GLOBAL_DEVICE".to_owned())).collect::<Result<Vec<_>>>()?;
 let shards=args[2].parse().map_err(|_|"CLI_GRAPH_SHARDS")?;let requested=if args.len()>3&&args[3]!="auto"{Some(args[3].parse().map_err(|_|"CLI_GRAPH_CAPACITY")?)}else{None};let batch=if args.len()>4{Some(args[4].parse().map_err(|_|"CLI_GRAPH_BATCH")?)}else{None};admit(&g,devices,inventory,shards,requested,batch)
}

#[derive(serde::Deserialize)]struct SizeProfile{minimum_frontier:u64,batch:u32,#[serde(default)]status:String,#[serde(default)]maximum_measured_frontier:u64,#[serde(default)]frontier_max_rank_bucket:Option<u32>,generator_backend:String}
#[derive(serde::Deserialize)]
struct WindowRequest { #[serde(default)]min_length:u32, max_length:u32, disk_budget:u64 }
#[derive(serde::Deserialize)]struct Launch{#[serde(default)]online_size_tuning:bool,#[serde(default)]size_profiles:Vec<SizeProfile>,graph_digest:String,devices:Vec<u32>,plan:Plan,#[serde(default)]rank_plans:Vec<Plan>,#[serde(default)]owner_cuts:Option<Vec<u64>>,#[serde(default)]profile_max_layers:Option<u32>,#[serde(default)]packet_job_token:Option<String>,#[serde(default)]path_targets:Option<Vec<Vec<i64>>>,#[serde(default)]path_bidirectional:bool,#[serde(default)]path_window:Option<WindowRequest>}
fn valid_frontier_bucket(bucket:u32, segmented:bool)->bool{bucket<=if segmented{63}else{31}}

#[cfg(test)] mod bucket_width_tests{
 #[test] fn legacy_and_segmented_widths(){
  assert!(super::valid_frontier_bucket(31,false));assert!(!super::valid_frontier_bucket(32,false));
  assert!(super::valid_frontier_bucket(32,true));assert!(super::valid_frontier_bucket(63,true));assert!(!super::valid_frontier_bucket(64,true));
 }
}

pub fn run(args:&[String])->Result<()>{
 let segmented=std::env::var("MGBFS_EXPERIMENTAL_SEGMENTED24").as_deref()==Ok("1");
 if !segmented{
  let value=match std::env::var("MGBFS_SEGMENT_STRATEGY"){
   Ok(v)=>Some(v),Err(std::env::VarError::NotPresent)=>None,
   Err(std::env::VarError::NotUnicode(_))=>return Err("SEGMENT_STRATEGY_INVALID".into()),
  };
  let strategy=crate::segmented_strategy::SegmentStrategy::parse(value.as_deref()).map_err(str::to_owned)?;
  if strategy!=crate::segmented_strategy::SegmentStrategy::Atomic{return Err("SEGMENT_STRATEGY_REQUIRES_SEGMENTED_BACKEND".into())}
 }
 if args.len()!=6{return Err("CLI_GRAPH_RANK_ARGUMENTS".into());}
 let launch:Launch=serde_json::from_reader(std::fs::File::open(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let rank:u32=args[2].parse().map_err(|_|"CLI_GRAPH_RANK")?;
 let bootstrap=Path::new(&args[3]);let output=Path::new(&args[4]);
 crate::native_fault::reset(Some(rank),launch.devices.get(rank as usize).copied());
 // File/graph validation is a local result until all segmented ranks report.
 // A missing or malformed launch file cannot establish a world size and is
 // still handled by the parent supervisor rather than this agreement.
 let ready=if segmented{Some(crate::segmented_host_ready::HostAgreement::new(bootstrap,rank,launch.plan.world)?)}else{None};
 let validated:Result<_>=(||{
 let g=graph(&args[0])?;
 if launch.graph_digest!=digest(&g)?||launch.devices.len()!=launch.plan.world as usize||rank>=launch.plan.world||launch.plan.generator_bytes<graph_bytes(&g){return Err("GRAPH_LAUNCH_IDENTITY_OR_GEOMETRY".into());}
 if launch.path_window.as_ref().is_some_and(|window| window.min_length>window.max_length
  ||launch.path_targets.is_none()){
  return Err("NATIVE_WINDOW_OWNER_LAYOUT".into());
 }
 if crate::generic_memory::nearest_direct_rows()&&(launch.path_targets.is_none()||launch.path_window.is_some()){
  return Err("NATIVE_PATH_ADMISSION_MODE_MISMATCH".into());
 }
 let invalid_byte_path=launch.plan.state_bytes==1&&(launch.plan.packed_candidates()
  ||crate::generic_memory::preferred_state_bytes(&g)!=1
  ||(launch.path_bidirectional&&launch.path_targets.as_ref()
    .is_some_and(|goals|goals.iter().flatten().any(|&value|u8::try_from(value).is_err()))));
 if launch.path_targets.is_some()&&(invalid_byte_path||launch.plan.parent_transport
  ||launch.plan.sort_candidates||launch.plan.generator_backend!="cuda"||launch.online_size_tuning){
  return Err("NATIVE_DISTRIBUTED_PATH_OWNER_LAYOUT".into());
 }
 if launch.plan.compact_transport&&!segmented{return Err("SEGMENT_PACKET_PLAN_REQUIRES_SEGMENTED_BACKEND".into());}
 if segmented && launch.plan.state_bytes == 1
  && crate::generic_memory::preferred_state_bytes(&g) != 1
  && segmented_byte_dictionary(&g).is_none()
 {
  return Err("SEGMENTED_RESIDENT_ALPHABET_MISMATCH".into());
 }
 if launch.path_targets.is_some()&&(segmented||launch.plan.history_layers!=1||launch.plan.history_algorithm!="HASH"||launch.profile_max_layers.is_some()){
  return Err("NATIVE_DISTRIBUTED_PATH_REQUIRES_APPEND_ONLY_HASH".into());
 }
 let packet_job_token=if segmented&&launch.plan.world>1&&std::env::var_os("MGBFS_SEGMENT_PACKET_CONTROL_ADDR").is_some(){
  Some(parse_job_token(launch.packet_job_token.as_deref())?)
 }else{None};
 let plans=if launch.rank_plans.is_empty(){vec![launch.plan.clone();launch.devices.len()]}else{launch.rank_plans.clone()};
 if plans.len()!=launch.devices.len(){return Err("GRAPH_RANK_PLANS".into());}
 for p in &plans{p.validate(g.generator_count() as u32)?;if p.history_algorithm!=launch.plan.history_algorithm||p.history_layers!=launch.plan.history_layers||p.world!=launch.plan.world||p.shards!=launch.plan.shards||p.elements!=launch.plan.elements||p.state_bytes!=launch.plan.state_bytes||p.batch!=launch.plan.batch||p.queue_capacity!=launch.plan.queue_capacity||p.compact_transport!=launch.plan.compact_transport||p.compact_packet_slots!=launch.plan.compact_packet_slots||p.compact_receive_slots!=launch.plan.compact_receive_slots||p.generator_bytes<graph_bytes(&g){return Err("GRAPH_TRANSPORT_GEOMETRY".into());}}
 let mut previous_base=None;let mut buckets=std::collections::BTreeSet::new();
 for p in &launch.size_profiles{
  if p.batch==0||p.batch>launch.plan.batch||!matches!(p.generator_backend.as_str(),"cuda"|"gemm")||(p.generator_backend=="gemm"&&launch.plan.generator_backend!="gemm"){return Err("SIZE_PROFILE_LAUNCH_INVALID".into());}
  if let Some(bucket)=p.frontier_max_rank_bucket{if !valid_frontier_bucket(bucket,segmented)||!buckets.insert(bucket)||!matches!(p.status.as_str(),"LIVE_CHUNK_EMPIRICAL_PROFILE"|"REUSED_SOURCE_LIVE_BUCKET"){return Err("SIZE_PROFILE_BUCKET_INVALID".into());}}
  else{if previous_base.map_or(p.minimum_frontier!=0,|previous|p.minimum_frontier<=previous){return Err("SIZE_PROFILE_BASE_INVALID".into());}previous_base=Some(p.minimum_frontier);}
 }
 if !launch.size_profiles.is_empty()&&previous_base.is_none(){return Err("SIZE_PROFILE_BASE_MISSING".into());}
 let seconds:u64=args[5].parse().map_err(|_|"CLI_GRAPH_SECONDS")?;
 if output.exists(){return Err("CLI_GRAPH_OUTPUT_EXISTS".into());}
 std::fs::create_dir_all(output).map_err(|e|e.to_string())?;
 Ok((g,plans,seconds,packet_job_token))
 })();
 if let Some(agreement)=&ready{
  let descriptor=validated.as_ref().ok().map(|(_,_,_,packet_job_token)|serde_json::json!({"schema":1,"graph_digest":launch.graph_digest,"devices":launch.devices,"plan":launch.plan,"owner_cuts":launch.owner_cuts,"packet_job_token":packet_job_token}));
  agreement.agree("arguments",&validated,descriptor.as_ref())?;
 }
 let(g,plans,seconds,packet_job_token)=validated?;
 crate::native_fault::reset(Some(rank),Some(launch.devices[rank as usize]));
 let rank_plan=plans[rank as usize].clone();
 if let Some(agreement)=&ready {
  // This host-only vote precedes both bootstrap-ID reuse and comm_take/create.
  // The ordered device list and transport environment are part of the epoch.
  let mode:Result<_>=(||{
   crate::segmented_host_ready::injected("comm-mode")?;
   crate::session_cache::test_drop_parked_segmented_comm()?;
   let external=std::env::var("MGBFS_SEGMENT_EXTERNAL_LINK").as_deref()==Ok("1");
   // Rank-local ordinals, loader paths, and NIC names may differ across hosts.
   // The runtime byte hashes in the descriptor still cover the loaded libraries.
   let transport=std::env::vars().filter(|(key,_)|{
      let local=matches!(key.as_str(),"CUDA_VISIBLE_DEVICES"|"CUDA_DEVICE_ORDER"|"LD_LIBRARY_PATH"|"LD_PRELOAD"|"LD_AUDIT"|"NCCL_SOCKET_IFNAME"|"NCCL_IB_HCA"|"MGBFS_SEGMENT_PACKET_BIND_ADDR");
      let relevant=local||key.starts_with("NCCL_")||matches!(key.as_str(),"CUDA_MODULE_LOADING"|"CUDA_DISABLE_PTX_JIT"|"MGBFS_SEGMENT_PACKET_CONTROL_ADDR");
      relevant&&!(external&&local)
     })
     .collect::<std::collections::BTreeMap<_,_>>();
   Ok(serde_json::json!({"schema":1,"communicator":crate::session_cache::segmented_comm_identity()?,
      "runtime":crate::segmented_host_ready::runtime_identity()?,"devices":launch.devices,"transport":transport}))
  })();
  agreement.agree("comm-mode",&mode,mode.as_ref().ok())?;
  eprintln!("SEGMENTED_COMM_AGREEMENT rank={rank} {}",mode.as_ref().map_err(Clone::clone)?["communicator"]);
 }
 let mut id=[0u8;128];if rank==0{if bootstrap.exists(){return Err("GRAPH_BOOTSTRAP_ALREADY_EXISTS".into());}id=crate::session_cache::bootstrap_id(||{let mut fresh=[0u8;128];crate::generic_native::checked!(mgbfs_nccl_unique_id, unsafe{mgbfs_nccl_unique_id(fresh.as_mut_ptr().cast())})?;Ok(fresh)})?;let temporary=bootstrap.with_extension("tmp");std::fs::write(&temporary,id).map_err(|e|e.to_string())?;std::fs::rename(temporary,bootstrap).map_err(|e|e.to_string())?;}else{let waiter=crate::session_wait::FileWaiter::new(bootstrap.parent().ok_or("GRAPH_BOOTSTRAP_PARENT")?);if !waiter.wait(bootstrap,None,Some(Duration::from_secs(60)))?{return Err("GRAPH_BOOTSTRAP_TIMEOUT".into());}let raw=std::fs::read(bootstrap).map_err(|e|e.to_string())?;if raw.len()!=128{return Err("GRAPH_BOOTSTRAP_SIZE".into());}id.copy_from_slice(&raw);let cached=crate::session_cache::bootstrap_id(||Ok(id))?;if cached!=id{return Err("GENERIC_SESSION_COMM_ID_MISMATCH".into());}}
 if segmented{return run_segmented(&g,&launch,rank,rank_plan,&id,packet_job_token,seconds,output,ready.as_ref().ok_or("SEGMENTED_HOST_AGREEMENT_MISSING")?)}
 let reverse_graph=if launch.path_bidirectional{
  let goals=launch.path_targets.as_deref().ok_or("NATIVE_PATH_TARGETS")?;
  Some(g.reversed_for_paths(goals)?)
 }else{None};
 let cache_shape=format!("{}:{}",serde_json::to_string(&rank_plan).map_err(|e|e.to_string())?,std::env::var("MGBFS_GENERIC_GENERATOR").unwrap_or_default());if rank_plan.history_algorithm=="HASH"{let mut required=GenericDistributedBfs::buffer_requirements(&g,&rank_plan,launch.owner_cuts.as_deref())?;
 if let Some(reverse)=&reverse_graph{required.extend(GenericDistributedBfs::reverse_buffer_requirements(reverse,&rank_plan,launch.owner_cuts.as_deref())?);}
 if launch.path_window.is_some(){
  let row=g.start.len()*8;
  let edge_bytes=GenericDistributedBfs::window_plan_edge_bytes(&rank_plan);
  let scratch=if launch.path_bidirectional{edge_bytes.max(crate::generic_path_native::WINDOW_JOIN_ROWS*4)}else{edge_bytes};
  required.extend([row,8,4,g.start.len()*rank_plan.state_bytes as usize,scratch]);
 }else if launch.path_targets.is_some(){
  required.extend(crate::generic_path_native::distributed_workspace(g.start.len() as u64,rank_plan.state_bytes,rank_plan.world as usize,launch.path_bidirectional)?);
  required.push(rank_plan.capacity as usize*crate::generic_path_native::PARENT_BYTES);
  if launch.path_bidirectional{required.push(rank_plan.capacity as usize*crate::generic_path_native::PARENT_BYTES);}
 }
 crate::session_cache::begin_generic(cache_shape,&required)?;}else{crate::session_cache::begin(cache_shape);}let _signals=Signals::new();let setup=Instant::now();let mut bfs=GenericDistributedBfs::new_with_cuts(&g,launch.devices[rank as usize],rank,rank_plan.clone(),&id,0x19f856a2,64,launch.owner_cuts.as_deref())?;if launch.online_size_tuning{bfs.enable_online_size_tuning();}let setup_seconds=setup.elapsed().as_secs_f64();eprintln!("MGBFS_GRAPH_READY rank={rank} device={}",launch.devices[rank as usize]);
 crate::job_phase::set("bfs")?;
 if let Some(goals)=&launch.path_targets {
  let mut reverse=if let Some(graph)=&reverse_graph{
   let setup=bfs.reverse_path_arena(graph,launch.owner_cuts.as_deref(),goals);
   if bfs.collective_stop(u32::from(setup.is_err()))?!=0{
    return Err(setup.err().unwrap_or_else(||"NATIVE_PATH_REMOTE_REVERSE_ARENA".into()));
   }
   Some(setup?)
  }else{None};
  let result=if let Some(window)=&launch.path_window{
   if let Some(reverse)=reverse.as_mut(){
    crate::generic_path_native::distributed_window_bidirectional(&mut bfs,reverse,goals,
     window.min_length,window.max_length,window.disk_budget,seconds,output)?
   }else{
    crate::generic_path_native::distributed_window_forward(&mut bfs,goals,window.max_length,window.disk_budget,seconds,output)?
   }
  }else if let Some(reverse)=reverse.as_mut(){
   crate::generic_path_native::distributed_bidirectional(&mut bfs,reverse,goals,seconds)?
  }else{
   crate::generic_path_native::distributed_forward(&mut bfs,goals,seconds)?
  };
  crate::job_phase::set("retiring")?;
  bfs.quiesce()?;
  if let Some(reverse)=reverse.as_ref(){reverse.quiesce()?;}
  let receipt=serde_json::to_vec_pretty(&result.receipt).map_err(|e|e.to_string())?;
  std::fs::write(output.join("path.json.tmp"),receipt).map_err(|e|e.to_string())?;
  std::fs::rename(output.join("path.json.tmp"),output.join("path.json")).map_err(|e|e.to_string())?;
  crate::session_cache::permit_comm(true);
  drop(result);
  return Ok(());
 }
 crate::session_cache::generic_requirements_consumed()?;
 let start=Instant::now();let mut sizes=vec![1u64];let mut times=vec![];let(status,reason)=loop{
   crate::native_fault::layer(sizes.len() as u64,sizes.len() as u64-1);
  let local=if CANCELLED.load(Ordering::Relaxed)||crate::job_phase::cancelled(){1}else if launch.profile_max_layers.map_or(false,|limit|sizes.len() as u32>limit){3}else if seconds!=0 && start.elapsed()>=Duration::from_secs(seconds){2}else{0};let vote=bfs.collective_stop(local)?;if vote!=0{break("INCOMPLETE",if vote==1{"CANCELLED".to_owned()}else if vote==3{"PROFILE_LAYER_LIMIT".to_owned()}else{"DEADLINE".to_owned()});}
  if let Some(p)=launch.size_profiles.iter().rev().find(|p|p.frontier_max_rank_bucket==Some(bfs.frontier_bucket())).or_else(||launch.size_profiles.iter().rev().find(|p|p.frontier_max_rank_bucket.is_none()&&p.minimum_frontier<=bfs.global_frontier())){bfs.set_size_profile(p.batch,p.generator_backend=="gemm")?;if matches!(p.status.as_str(),"LIVE_CHUNK_EMPIRICAL_PROFILE"|"REUSED_SOURCE_LIVE_BUCKET")&&(p.frontier_max_rank_bucket==Some(bfs.frontier_bucket())||p.frontier_max_rank_bucket.is_none()&&p.maximum_measured_frontier==bfs.global_frontier()){bfs.cache_current_size_profile(p.batch,p.generator_backend=="gemm")?;}}
  let layer=Instant::now();match bfs.advance()?{DistributedAdvance::Layer{global,..}=>{sizes.push(global);crate::native_fault::completed(sizes.len() as u64-1);times.push(layer.elapsed().as_secs_f64());},DistributedAdvance::Complete=>break("COMPLETE","FRONTIER_EXHAUSTED".to_owned()),DistributedAdvance::Resource{fatal}=>break("INCOMPLETE",format!("RESOURCE_{fatal}")),}
 };let bfs_seconds=start.elapsed().as_secs_f64();crate::job_phase::set("retiring")?;let current=bfs.sample_global(1000)?;let previous=bfs.previous_small_sample(*sizes.iter().rev().nth(1).unwrap_or(&0))?;
 let states=serde_json::json!({"schema":2,"state_encoding":"signed_int64_vectors","current":current,"previous_small":previous,"current_sample_global_limit":1000});let state_bytes=serde_json::to_vec(&states).map_err(|e|e.to_string())?;
 use sha2::{Digest,Sha256};let sha=Sha256::digest(&state_bytes).iter().map(|v|format!("{v:02x}")).collect::<String>();std::fs::write(output.join("states.json"),state_bytes).map_err(|e|e.to_string())?;
 let report=serde_json::json!({"schema":2,"rank":rank,"world":launch.plan.world,"device":launch.devices[rank as usize],"status":status,"reason":reason,"graph_digest":launch.graph_digest,"states_sha256":sha,"layer_sizes":sizes,"layer_seconds":times,"bfs_seconds":bfs_seconds,"source_retries":bfs.source_retries(),"size_profile_count":launch.size_profiles.len(),"online_size_profile_events":bfs.size_profile_events(),"setup_seconds":setup_seconds,"plan":rank_plan,"backend":if rank_plan.history_layers==3{"GENERIC_DISTRIBUTED_EXACT_THREE_BANK_HISTORY"}else{"GENERIC_DISTRIBUTED_EXACT_ALL_VISITED"},"profile_status":"MEMORY_ADMITTED_NOT_THROUGHPUT_TUNED"});
 std::fs::write(output.join("report.json.tmp"),serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;std::fs::rename(output.join("report.json.tmp"),output.join("report.json")).map_err(|e|e.to_string())?;bfs.quiesce()?;crate::session_cache::permit_comm(true);Ok(())
}

// Private opt-in used solely for integration tests; no public API promises yet.
fn run_segmented(g:&GraphDefinitionV2,launch:&Launch,rank:u32,rank_plan:Plan,id:&[u8;128],packet_job_token:Option<PacketJobToken>,seconds:u64,output:&Path,ready:&crate::segmented_host_ready::HostAgreement)->Result<()>{
 let alphabet = (rank_plan.state_bytes == 1).then(|| segmented_byte_dictionary(g)).flatten();
 let encoded_graph = alphabet.as_ref().map(|symbols| {
  let mut graph = g.clone();
  for value in &mut graph.start {
   *value = symbols.binary_search(value).expect("dictionary contains all root symbols") as i64;
  }
  graph
 });
 let entry:Result<_>=(||{
  crate::segmented_host_ready::injected("entry")?;
  if launch.online_size_tuning||launch.size_profiles.iter().any(|p|p.generator_backend!="cuda"){return Err("SEGMENTED_EXPERIMENT_PROFILE_NOT_SUPPORTED".into())}
  let backend=if std::env::var("MGBFS_EXPERIMENTAL_BITMAP").as_deref()==Ok("1"){"EXPERIMENTAL_PERMUTATION_BITMAP"}else{"EXPERIMENTAL_SEGMENTED24"};
  let identity=crate::segmented_host_ready::runtime_identity()?;
  let descriptor=serde_json::json!({"schema":1,"runtime":identity,"graph_digest":digest(g)?,"devices":launch.devices,"world":launch.plan.world,"shards":launch.plan.shards,"elements":launch.plan.elements,"state_bytes":launch.plan.state_bytes,"batch":launch.plan.batch,"queue_capacity":launch.plan.queue_capacity,"history_algorithm":launch.plan.history_algorithm,"history_layers":launch.plan.history_layers,"owner_cuts":launch.owner_cuts,"backend":backend,"external_reserve_bytes":if std::env::var("MGBFS_SEGMENT_EXTERNAL_LINK").as_deref()==Ok("1"){Some(crate::generic_memory::admission_reserve_bytes()?)}else{None}});
  crate::session_cache::begin(format!("{}:{}",backend,serde_json::to_string(&rank_plan).map_err(|e|e.to_string())?));
  crate::generic_native::checked!(cudaSetDevice,unsafe{cudaSetDevice(launch.devices[rank as usize] as i32)})?;
  let pool_idle_before=crate::generic_pool::available()?;crate::generic_pool::trim_idle()?;let pool_idle_after=crate::generic_pool::available()?;
  let hash_bits=match std::env::var("MGBFS_SEGMENT_TEST_HASH_BITS"){
   Ok(v) if std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref()==Ok("1")=>v.parse::<u32>().map_err(|_|"SEGMENTED_TEST_HASH_BITS")?,
   Ok(_)=>return Err("SEGMENTED_TEST_HASH_BITS_REQUIRES_TEST_GATE".into()),
   Err(std::env::VarError::NotPresent)=>64,
   Err(_)=>return Err("SEGMENTED_TEST_HASH_BITS".into()),
  };
  if hash_bits>64{return Err("SEGMENTED_TEST_HASH_BITS".into())}
  Ok((backend,pool_idle_before,pool_idle_after,hash_bits,descriptor))
 })();
 let descriptor=entry.as_ref().ok().map(|x|&x.4);
 ready.agree("entry",&entry,descriptor)?;
 let(backend,pool_idle_before,pool_idle_after,hash_bits,_)=entry?;
 let _signals=Signals::new();let setup=Instant::now();
 let mut bfs=crate::segmented_distributed_native::SegmentedDistributedBfs::new(encoded_graph.as_ref().unwrap_or(g),launch.devices[rank as usize],rank,rank_plan.clone(),id,packet_job_token,0x19f856a2,hash_bits,launch.owner_cuts.as_deref(),ready)?;
 let setup_seconds=setup.elapsed().as_secs_f64();eprintln!("MGBFS_GRAPH_READY rank={rank} device={} backend={backend}",launch.devices[rank as usize]);crate::job_phase::set("bfs")?;
 let start=Instant::now();let mut sizes=vec![1u64];let mut times=vec![];let mut committed=vec![0.0];
 let packet = bfs.uses_packet_transport();
 let mut next_stop = 0;
 let stop_reason = |layers: usize| {
  if CANCELLED.load(Ordering::Relaxed) || crate::job_phase::cancelled() { 1 }
  else if launch.profile_max_layers.is_some_and(|limit| layers > limit as usize) { 3 }
  else if seconds != 0 && start.elapsed() >= Duration::from_secs(seconds) { 2 }
  else { 0 }
 };
 let(status,reason)=loop{
  crate::native_fault::layer(sizes.len() as u64,sizes.len() as u64-1);
  // The root needs an initial vote. Each subsequent packet layer carries
  // its stop vote alongside the committed rank counts.
  let stop = if packet && sizes.len() > 1 { next_stop }
   else { bfs.collective_stop(stop_reason(sizes.len()))? };
  if stop!=0{break("INCOMPLETE",if stop==1{"CANCELLED".to_owned()}else if stop==3{"PROFILE_LAYER_LIMIT".to_owned()}else{"DEADLINE".to_owned()});}
  if let Some(p)=launch.size_profiles.iter().rev().find(|p|p.frontier_max_rank_bucket==Some(bfs.frontier_bucket())).or_else(||launch.size_profiles.iter().rev().find(|p|p.frontier_max_rank_bucket.is_none()&&p.minimum_frontier<=bfs.global_frontier())){bfs.set_size_profile(p.batch,false)?;}
  let layer=Instant::now();match bfs.advance_with_stop(|| stop_reason(sizes.len()+1), &mut next_stop)?{DistributedAdvance::Layer{global,..}=>{
   let commit=Instant::now();
   sizes.push(global);
   committed.push(commit.duration_since(start).as_secs_f64());
   crate::native_fault::completed(sizes.len() as u64-1);
   times.push(layer.elapsed().as_secs_f64());
  },DistributedAdvance::Complete=>break("COMPLETE","FRONTIER_EXHAUSTED".to_owned()),DistributedAdvance::Resource{fatal}=>break("INCOMPLETE",match next_stop{
   1=>"CANCELLED".to_owned(),
   2=>"DEADLINE".to_owned(),
   _=>format!("RESOURCE_{fatal}"),
  })}
 };
 let bfs_seconds=start.elapsed().as_secs_f64();
 // No rank publishes a receipt or parks a communicator until every rank has
 // sampled, written private attempt files, and drained its own streams.
 let prepared:Result<_>=(||{
  crate::job_phase::set("retiring")?;
  crate::segmented_host_ready::injected("finish-prepare")?;
  if crate::segmented_host_ready::injected("finish-write").is_err(){
   std::fs::write(output.join(".missing-test-parent").join("states"),b"x").map_err(|e|format!("SEGMENTED_TEST_FINISH_WRITE rank={rank} {e}"))?;
  }
  let mut current=bfs.sample_global(1000)?;
  let mut previous=bfs.previous_small_sample(*sizes.iter().rev().nth(1).unwrap_or(&0))?;
  if let Some(symbols) = &alphabet {
   for value in current.iter_mut().chain(&mut previous).flatten() {
    let index = usize::try_from(*value).map_err(|_| "SEGMENTED_ALPHABET_INDEX")?;
    *value = *symbols.get(index).ok_or("SEGMENTED_ALPHABET_INDEX")?;
   }
  }
  let states=serde_json::json!({"schema":2,"state_encoding":"signed_int64_vectors","current":current,"previous_small":previous,"current_sample_global_limit":1000});let state_bytes=serde_json::to_vec(&states).map_err(|e|e.to_string())?;
  use sha2::{Digest,Sha256};let sha=Sha256::digest(&state_bytes).iter().map(|v|format!("{v:02x}")).collect::<String>();
  std::fs::write(output.join("states.json.tmp"),state_bytes).map_err(|e|format!("SEGMENTED_FINISH_STATES rank={rank} {e}"))?;
  let mut memory=bfs.memory_geometry();memory["pool_idle_before_trim_bytes"]=pool_idle_before.into();memory["pool_idle_after_trim_bytes"]=pool_idle_after.into();
  if let Some(symbols) = &alphabet {
   memory["resident_codec"] = "alphabet8-v1".into();
   memory["resident_alphabet_size"] = symbols.len().into();
  }
  let report=serde_json::json!({"schema":2,"rank":rank,"world":launch.plan.world,"device":launch.devices[rank as usize],"status":status,"reason":reason,"graph_digest":launch.graph_digest,"states_sha256":sha,"layer_sizes":sizes,"layer_seconds":times,"layer_committed_seconds":committed,"bfs_seconds":bfs_seconds,"source_retries":bfs.source_retries(),"peak_owner_lanes":bfs.peak_owner_lanes(),"segment_strategy":bfs.strategy_name(),"segment_memory":memory,"size_profile_count":launch.size_profiles.len(),"online_size_profile_events":bfs.size_profile_events(),"setup_seconds":setup_seconds,"plan":rank_plan,"backend":backend,"profile_status":"NOT_THROUGHPUT_TUNED"});
  std::fs::write(output.join("report.json.tmp"),serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|format!("SEGMENTED_FINISH_REPORT rank={rank} {e}"))?;
  bfs.quiesce()?;Ok(())
 })();
 let result_identity=serde_json::json!({"status":status,"reason":reason,"layer_sizes":sizes,"graph_digest":launch.graph_digest});
 ready.agree("finish-prepared",&prepared,Some(&result_identity))?;prepared?;
 let published:Result<_>=(||{
  if crate::segmented_host_ready::injected("finish-publish").is_err(){
   std::fs::remove_file(output.join("states.json.tmp")).map_err(|e|e.to_string())?;
  }
  std::fs::rename(output.join("states.json.tmp"),output.join("states.json")).map_err(|e|format!("SEGMENTED_PUBLISH_STATES rank={rank} {e}"))?;
  std::fs::rename(output.join("report.json.tmp"),output.join("report.json")).map_err(|e|format!("SEGMENTED_PUBLISH_REPORT rank={rank} {e}"))?;
  Ok(())
 })();
 let vote=ready.agree("finish-published",&published,Some(&result_identity));
 if vote.is_err()||published.is_err(){let _=std::fs::remove_file(output.join("report.json"));let _=std::fs::remove_file(output.join("states.json"));}
 vote?;published?;
 bfs.permit_healthy_communicator_reuse();Ok(())
}
