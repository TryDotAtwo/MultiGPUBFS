//! Single-host distributed launch worker. No CPU state generation or dedup.
use std::{path::Path,time::{Duration,Instant},sync::atomic::Ordering};
use mgbfs_core::{graph_definition::{GraphDefinitionV2,GraphAction},Result};
use mgbfs_cuda::{ffi::*,native_owner::{cudaSetDevice,cudaMemGetInfo}};
use crate::{generic_native::check,generic_distributed_memory::GenericDistributedMemoryPlan as Plan,generic_distributed_native::{GenericDistributedBfs,DistributedAdvance},generic_run::{Signals,CANCELLED}};
extern "C" {fn mgbfs_generic_hardware_info(device:i32,values:*mut u64)->i32;}
fn graph(path:&str)->Result<GraphDefinitionV2>{let g:GraphDefinitionV2=serde_json::from_reader(std::fs::File::open(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;g.validate()?;Ok(g)}
fn digest(g:&GraphDefinitionV2)->Result<String>{Ok(g.semantic_digest()?.iter().map(|v|format!("{v:02x}")).collect())}
fn graph_bytes(g:&GraphDefinitionV2)->u64{match &g.action{GraphAction::Permutation{generators,..}=>generators.iter().map(|v|v.len() as u64*4).sum(),GraphAction::Matrix{generators,..}=>generators.iter().map(|v|v.matrix.len() as u64*8+4).sum()}}
extern "C"{fn cudaGetDeviceCount(count:*mut i32)->i32;}
pub fn info(args:&[String])->Result<()>{
 if args.len()<3{return Err("CLI_GRAPH_INFO_ARGUMENTS".into());}let g=graph(&args[0])?;let shards:u32=args[2].parse().map_err(|_|"CLI_GRAPH_SHARDS")?;
 let mut count=0i32;check(unsafe{cudaGetDeviceCount(&mut count)})?;if count<=0{return Err("NO_VISIBLE_CUDA_DEVICES".into());}
 let devices=if args[1]=="auto"{(0..count as u32).collect::<Vec<_>>()}else{args[1].split(',').map(|v|v.parse::<u32>().map_err(|_|"CLI_GRAPH_DEVICES".to_owned())).collect::<Result<Vec<_>>>()?};
 if devices.is_empty()||devices.len()>128||devices.iter().any(|&d|d>=count as u32)||devices.iter().enumerate().any(|(i,d)|devices[..i].contains(d)){return Err("CLI_GRAPH_DEVICE_SELECTION".into());}
 let mut inventory=vec![];let mut minimum=u64::MAX;for &device in &devices{check(unsafe{cudaSetDevice(device as i32)})?;let(mut free,mut total)=(0usize,0usize);check(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;minimum=minimum.min(free as u64);let mut hw=[0u64;8];check(unsafe{mgbfs_generic_hardware_info(device as i32,hw.as_mut_ptr())})?;inventory.push(serde_json::json!({"device":device,"free_bytes":free,"total_bytes":total,"sm_count":hw[0],"compute_major":hw[1],"compute_minor":hw[2],"l2_bytes":hw[3],"warp_size":hw[4],"threads_per_sm":hw[5],"threads_per_block":hw[6],"shared_bytes_per_sm":hw[7]}));}
 let requested=if args.len()>3&&args[3]!="auto"{Some(args[3].parse::<u32>().map_err(|_|"CLI_GRAPH_CAPACITY")?)}else{None};
 let batch=if args.len()>4{Some(args[4].parse::<u32>().map_err(|_|"CLI_GRAPH_BATCH")?)}else{None};
 println!("{}",admit(&g,devices,inventory,shards,requested,batch)?);Ok(())
}
fn admit(g:&GraphDefinitionV2,devices:Vec<u32>,inventory:Vec<serde_json::Value>,shards:u32,requested:Option<u32>,batch:Option<u32>)->Result<serde_json::Value>{
 if devices.is_empty()||devices.len()>128||inventory.len()!=devices.len(){return Err("GRAPH_GLOBAL_INVENTORY_SHAPE".into());}
 let mut minimum=u64::MAX;
 for v in &inventory{let free=v["free_bytes"].as_u64().ok_or("GRAPH_GLOBAL_FREE_BYTES")?;let total=v["total_bytes"].as_u64().ok_or("GRAPH_GLOBAL_TOTAL_BYTES")?;if free==0||free>total{return Err("GRAPH_GLOBAL_MEMORY_RANGE".into());}minimum=minimum.min(free);}
 let state_bytes=crate::generic_memory::preferred_state_bytes(&g);let history_layers=if g.inverse_closed()?{3}else{1};
 
 let sorted=match std::env::var("MGBFS_GENERIC_HISTORY").as_deref(){Ok("sorted")=>true,Ok("hash")|Err(_)=>false,_=>return Err("GENERIC_HISTORY_MODE".into())};
 let (mut rank_plans,owner_cuts)=if sorted{
  let lanes=std::env::var("MGBFS_GENERIC_OWNER_LANES").ok().map(|v|v.parse::<u32>().map_err(|_|"SORTED_OWNER_LANES")).transpose()?.unwrap_or(shards.min(4));
  let upper=requested.map(u64::from).or_else(||crate::generic_memory::state_space_bound(g));
  let target=batch.unwrap_or(upper.unwrap_or(u64::from(crate::generic_memory::MAX_ARENA_CAPACITY)).min(u64::from(crate::generic_memory::MAX_ARENA_CAPACITY)).min(u64::from((1<<20)/(g.generator_count() as u32).max(1))) as u32).max(1);
  let mut plans=Vec::new();for(&device,item)in devices.iter().zip(&inventory){plans.push(Plan::automatic_sorted_for_graph(g,device as i32,devices.len() as u32,shards,graph_bytes(g),item["free_bytes"].as_u64().unwrap(),upper,state_bytes,target,history_layers,lanes)?);}
  let common=plans.iter().map(|p|p.batch).min().unwrap();
  if let Some(wanted)=batch{if wanted==0||common!=wanted{return Err("REQUESTED_BATCH_EXCEEDS_SORTED_ADMISSION".into());}}
  for((p,&device),item)in plans.iter_mut().zip(&devices).zip(&inventory){*p=Plan::automatic_sorted_for_graph(g,device as i32,devices.len() as u32,shards,graph_bytes(g),item["free_bytes"].as_u64().unwrap(),upper,state_bytes,common,history_layers,lanes)?;if requested.map_or(false,|v|p.capacity!=v){return Err("REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION".into());}}
  let cuts=if requested.is_some(){None}else{let total=plans.iter().map(|p|u64::from(p.capacity)).sum::<u64>();let mut cuts=vec![0];let mut cumulative=0u64;for p in &plans{cumulative+=u64::from(p.capacity);cuts.push(((u128::from(cumulative)*(1u128<<32)+u128::from(total)-1)/u128::from(total)) as u64);}Some(cuts)};
  (plans,cuts)
 }else if let Some(capacity)=requested{
  let auto=Plan::automatic_storage_history(g.start.len() as u32,devices.len() as u32,shards,g.generator_count() as u32,graph_bytes(&g),minimum,Some(u64::from(capacity)),state_bytes,history_layers)?;
  if auto.capacity!=capacity{return Err("REQUESTED_CAPACITY_EXCEEDS_ADMISSION".into());}
  (vec![auto;devices.len()],None)
 }else{
  let free=inventory.iter().map(|v|v["free_bytes"].as_u64().unwrap()).collect::<Vec<_>>();
  let(plans,cuts)=crate::generic_distributed_memory::heterogeneous_plans_history(g.start.len() as u32,shards,g.generator_count() as u32,graph_bytes(&g),&free,crate::generic_memory::state_space_bound(&g),state_bytes,history_layers)?;(plans,Some(cuts))
 };
 if let Some(batch)=batch.filter(|_|!sorted){
  for(p,item)in rank_plans.iter_mut().zip(&inventory){let candidate=Plan::with_storage_history(p.elements,p.world,p.shards,p.capacity,batch,g.generator_count() as u32,p.generator_bytes,state_bytes,history_layers)?;let free=item["free_bytes"].as_u64().unwrap();let budget=free.checked_sub((1u64<<30).max(free/10)).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;if candidate.device_bytes>budget{return Err("REQUESTED_BATCH_EXCEEDS_ADMISSION".into());}*p=candidate;}
 }
 let plan=rank_plans.iter().min_by_key(|p|p.capacity).unwrap().clone();
 Ok(serde_json::json!({"size_profile_capability":1,"graph_digest":digest(g)?,"devices":devices,"inventory":inventory,"plan":plan,"rank_plans":rank_plans,"owner_cuts":owner_cuts,"profile_status":"MEMORY_ADMITTED_NOT_THROUGHPUT_TUNED"}))
}
/// Query only this rank's physical GPU; remote IDs must never be queried here.
pub fn local_info(args:&[String])->Result<()>{
 if args.len()!=6{return Err("CLI_GRAPH_LOCAL_PLAN_ARGUMENTS".into());}
 let g=graph(&args[0])?;let device:i32=args[1].parse().map_err(|_|"CLI_GRAPH_DEVICE")?;
 let world:u32=args[2].parse().map_err(|_|"CLI_GRAPH_WORLD")?;let shards:u32=args[3].parse().map_err(|_|"CLI_GRAPH_SHARDS")?;
 if device<0||world==0||world>128{return Err("CLI_GRAPH_LOCAL_PLAN_GEOMETRY".into());}
 if std::env::var("MGBFS_GENERIC_HISTORY").as_deref()!=Ok("sorted"){return Err("LOCAL_SORTED_PLAN_REQUIRES_SORTED_HISTORY".into());}
 let requested=if args[4]=="auto"{None}else{Some(args[4].parse::<u32>().map_err(|_|"CLI_GRAPH_CAPACITY")?)};
 let target=args[5].parse::<u32>().map_err(|_|"CLI_GRAPH_BATCH")?;if target==0{return Err("CLI_GRAPH_BATCH".into());}
 check(unsafe{cudaSetDevice(device)})?;let(mut free,mut total)=(0usize,0usize);check(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
 let lanes=std::env::var("MGBFS_GENERIC_OWNER_LANES").ok().map(|v|v.parse::<u32>().map_err(|_|"SORTED_OWNER_LANES")).transpose()?.unwrap_or(shards.min(4));
 let upper=requested.map(u64::from).or_else(||crate::generic_memory::state_space_bound(&g));let bytes=crate::generic_memory::preferred_state_bytes(&g);let banks=if g.inverse_closed()?{3}else{1};
 let plan=Plan::automatic_sorted_for_graph(&g,device,world,shards,graph_bytes(&g),free as u64,upper,bytes,target,banks,lanes)?;
 if requested.map_or(false,|v|plan.capacity!=v){return Err("REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION".into());}
 let mut hw=[0u64;8];check(unsafe{mgbfs_generic_hardware_info(device,hw.as_mut_ptr())})?;println!("{}",serde_json::json!({"size_profile_capability":1,"graph_digest":digest(&g)?,"device":device,"free_bytes":free,"total_bytes":total,"sm_count":hw[0],"compute_major":hw[1],"compute_minor":hw[2],"l2_bytes":hw[3],"plan":plan}));Ok(())
}
pub fn global_info(args:&[String])->Result<()>{
 if args.len()<3{return Err("CLI_GRAPH_GLOBAL_ARGUMENTS".into());}let g=graph(&args[0])?;let inventory:Vec<serde_json::Value>=serde_json::from_reader(std::fs::File::open(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let devices=inventory.iter().map(|v|v["device"].as_u64().filter(|&d|d<=u64::from(u32::MAX)).map(|d|d as u32).ok_or("GRAPH_GLOBAL_DEVICE".to_owned())).collect::<Result<Vec<_>>>()?;
 let shards=args[2].parse().map_err(|_|"CLI_GRAPH_SHARDS")?;let requested=if args.len()>3&&args[3]!="auto"{Some(args[3].parse().map_err(|_|"CLI_GRAPH_CAPACITY")?)}else{None};let batch=if args.len()>4{Some(args[4].parse().map_err(|_|"CLI_GRAPH_BATCH")?)}else{None};println!("{}",admit(&g,devices,inventory,shards,requested,batch)?);Ok(())
}

#[derive(serde::Deserialize)]struct SizeProfile{minimum_frontier:u64,batch:u32,#[serde(default)]status:String,#[serde(default)]maximum_measured_frontier:u64,generator_backend:String}
#[derive(serde::Deserialize)]struct Launch{#[serde(default)]online_size_tuning:bool,#[serde(default)]size_profiles:Vec<SizeProfile>,graph_digest:String,devices:Vec<u32>,plan:Plan,#[serde(default)]rank_plans:Vec<Plan>,#[serde(default)]owner_cuts:Option<Vec<u64>>,#[serde(default)]profile_max_layers:Option<u32>}
pub fn run(args:&[String])->Result<()>{
 if args.len()!=6{return Err("CLI_GRAPH_RANK_ARGUMENTS".into());}let g=graph(&args[0])?;let launch:Launch=serde_json::from_reader(std::fs::File::open(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;let rank:u32=args[2].parse().map_err(|_|"CLI_GRAPH_RANK")?;
 if launch.graph_digest!=digest(&g)?||launch.devices.len()!=launch.plan.world as usize||rank>=launch.plan.world||launch.plan.generator_bytes<graph_bytes(&g){return Err("GRAPH_LAUNCH_IDENTITY_OR_GEOMETRY".into());}
 let plans=if launch.rank_plans.is_empty(){vec![launch.plan.clone();launch.devices.len()]}else{launch.rank_plans.clone()};
 if plans.len()!=launch.devices.len(){return Err("GRAPH_RANK_PLANS".into());}
 for p in &plans{p.validate(g.generator_count() as u32)?;if p.history_algorithm!=launch.plan.history_algorithm||p.history_layers!=launch.plan.history_layers||p.world!=launch.plan.world||p.shards!=launch.plan.shards||p.elements!=launch.plan.elements||p.state_bytes!=launch.plan.state_bytes||p.batch!=launch.plan.batch||p.queue_capacity!=launch.plan.queue_capacity||p.generator_bytes<graph_bytes(&g){return Err("GRAPH_TRANSPORT_GEOMETRY".into());}}
 for(i,p)in launch.size_profiles.iter().enumerate(){
  if (i==0&&p.minimum_frontier!=0)||(i>0&&p.minimum_frontier<=launch.size_profiles[i-1].minimum_frontier)||p.batch==0||p.batch>launch.plan.batch||!matches!(p.generator_backend.as_str(),"cuda"|"gemm")||(p.generator_backend=="gemm"&&launch.plan.generator_backend!="gemm"){return Err("SIZE_PROFILE_LAUNCH_INVALID".into());}
 }
 let rank_plan=plans[rank as usize].clone();
 let seconds:u64=args[5].parse().map_err(|_|"CLI_GRAPH_SECONDS")?;if seconds==0{return Err("CLI_GRAPH_SECONDS".into());}let output=Path::new(&args[4]);if output.exists(){return Err("CLI_GRAPH_OUTPUT_EXISTS".into());}std::fs::create_dir_all(output).map_err(|e|e.to_string())?;
 let bootstrap=Path::new(&args[3]);let mut id=[0u8;128];if rank==0{if bootstrap.exists(){return Err("GRAPH_BOOTSTRAP_ALREADY_EXISTS".into());}check(unsafe{mgbfs_nccl_unique_id(id.as_mut_ptr().cast())})?;let temporary=bootstrap.with_extension("tmp");std::fs::write(&temporary,id).map_err(|e|e.to_string())?;std::fs::rename(temporary,bootstrap).map_err(|e|e.to_string())?;}else{let started=Instant::now();while !bootstrap.exists(){if started.elapsed()>Duration::from_secs(60){return Err("GRAPH_BOOTSTRAP_TIMEOUT".into());}std::thread::sleep(Duration::from_millis(10));}let raw=std::fs::read(bootstrap).map_err(|e|e.to_string())?;if raw.len()!=128{return Err("GRAPH_BOOTSTRAP_SIZE".into());}id.copy_from_slice(&raw);}
 let _signals=Signals::new();let setup=Instant::now();let mut bfs=GenericDistributedBfs::new_with_cuts(&g,launch.devices[rank as usize],rank,rank_plan.clone(),&id,0x19f856a2,64,launch.owner_cuts.as_deref())?;if launch.online_size_tuning{bfs.enable_online_size_tuning();}let setup_seconds=setup.elapsed().as_secs_f64();eprintln!("MGBFS_GRAPH_READY rank={rank} device={}",launch.devices[rank as usize]);
 let start=Instant::now();let mut sizes=vec![1u64];let mut times=vec![];let(status,reason)=loop{
  let local=if CANCELLED.load(Ordering::Relaxed){1}else if launch.profile_max_layers.map_or(false,|limit|sizes.len() as u32>limit){3}else if start.elapsed()>=Duration::from_secs(seconds){2}else{0};let vote=bfs.collective_stop(local)?;if vote!=0{break("INCOMPLETE",if vote==1{"CANCELLED".to_owned()}else if vote==3{"PROFILE_LAYER_LIMIT".to_owned()}else{"DEADLINE".to_owned()});}
  if let Some(p)=launch.size_profiles.iter().rev().find(|p|p.minimum_frontier<=bfs.global_frontier()){bfs.set_size_profile(p.batch,p.generator_backend=="gemm")?;if p.status=="LIVE_CHUNK_EMPIRICAL_PROFILE"&&p.maximum_measured_frontier==bfs.global_frontier(){bfs.cache_current_size_profile(p.batch,p.generator_backend=="gemm")?;}}
  let layer=Instant::now();match bfs.advance()?{DistributedAdvance::Layer{global,..}=>{sizes.push(global);times.push(layer.elapsed().as_secs_f64());},DistributedAdvance::Complete=>break("COMPLETE","FRONTIER_EXHAUSTED".to_owned()),DistributedAdvance::Resource{fatal}=>break("INCOMPLETE",format!("RESOURCE_{fatal}")),}
 };let bfs_seconds=start.elapsed().as_secs_f64();let current=bfs.sample_global(1000)?;let previous=bfs.previous_small_sample(*sizes.iter().rev().nth(1).unwrap_or(&0))?;
 let states=serde_json::json!({"schema":2,"state_encoding":"signed_int64_vectors","current":current,"previous_small":previous,"current_sample_global_limit":1000});let state_bytes=serde_json::to_vec(&states).map_err(|e|e.to_string())?;
 use sha2::{Digest,Sha256};let sha=Sha256::digest(&state_bytes).iter().map(|v|format!("{v:02x}")).collect::<String>();std::fs::write(output.join("states.json"),state_bytes).map_err(|e|e.to_string())?;
 let report=serde_json::json!({"schema":2,"rank":rank,"world":launch.plan.world,"device":launch.devices[rank as usize],"status":status,"reason":reason,"graph_digest":launch.graph_digest,"states_sha256":sha,"layer_sizes":sizes,"layer_seconds":times,"bfs_seconds":bfs_seconds,"source_retries":bfs.source_retries(),"size_profile_count":launch.size_profiles.len(),"online_size_profile_events":bfs.size_profile_events(),"setup_seconds":setup_seconds,"plan":rank_plan,"backend":if rank_plan.history_layers==3{"GENERIC_DISTRIBUTED_EXACT_THREE_BANK_HISTORY"}else{"GENERIC_DISTRIBUTED_EXACT_ALL_VISITED"},"profile_status":"MEMORY_ADMITTED_NOT_THROUGHPUT_TUNED"});
 std::fs::write(output.join("report.json.tmp"),serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;std::fs::rename(output.join("report.json.tmp"),output.join("report.json")).map_err(|e|e.to_string())?;Ok(())
}
