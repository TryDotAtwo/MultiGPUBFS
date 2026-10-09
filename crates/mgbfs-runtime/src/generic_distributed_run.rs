//! Single-host distributed launch worker. No CPU state generation or dedup.
use std::{path::Path,time::{Duration,Instant},sync::atomic::Ordering};
use mgbfs_core::{graph_definition::{GraphDefinitionV2,GraphAction},Result};
use mgbfs_cuda::{ffi::*,native_owner::{cudaSetDevice,cudaMemGetInfo}};
use crate::{generic_native::check,generic_distributed_memory::GenericDistributedMemoryPlan as Plan,generic_distributed_native::{GenericDistributedBfs,DistributedAdvance},generic_run::{Signals,CANCELLED}};
fn graph(path:&str)->Result<GraphDefinitionV2>{let g:GraphDefinitionV2=serde_json::from_reader(std::fs::File::open(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;g.validate()?;Ok(g)}
fn digest(g:&GraphDefinitionV2)->Result<String>{Ok(g.semantic_digest()?.iter().map(|v|format!("{v:02x}")).collect())}
fn graph_bytes(g:&GraphDefinitionV2)->u64{match &g.action{GraphAction::Permutation{generators,..}=>generators.iter().map(|v|v.len() as u64*4).sum(),GraphAction::Matrix{generators,..}=>generators.iter().map(|v|v.matrix.len() as u64*8+4).sum()}}
extern "C"{fn cudaGetDeviceCount(count:*mut i32)->i32;}
pub fn info(args:&[String])->Result<()>{
 if args.len()<3{return Err("CLI_GRAPH_INFO_ARGUMENTS".into());}let g=graph(&args[0])?;let shards:u32=args[2].parse().map_err(|_|"CLI_GRAPH_SHARDS")?;
 let mut count=0i32;check(unsafe{cudaGetDeviceCount(&mut count)})?;if count<=0{return Err("NO_VISIBLE_CUDA_DEVICES".into());}
 let devices=if args[1]=="auto"{(0..count as u32).collect::<Vec<_>>()}else{args[1].split(',').map(|v|v.parse::<u32>().map_err(|_|"CLI_GRAPH_DEVICES".to_owned())).collect::<Result<Vec<_>>>()?};
 if devices.is_empty()||devices.len()>128||devices.iter().any(|&d|d>=count as u32)||devices.iter().enumerate().any(|(i,d)|devices[..i].contains(d)){return Err("CLI_GRAPH_DEVICE_SELECTION".into());}
 let mut inventory=vec![];let mut minimum=u64::MAX;for &device in &devices{check(unsafe{cudaSetDevice(device as i32)})?;let(mut free,mut total)=(0usize,0usize);check(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;minimum=minimum.min(free as u64);inventory.push(serde_json::json!({"device":device,"free_bytes":free,"total_bytes":total}));}
 let requested=if args.len()>3&&args[3]!="auto"{Some(args[3].parse::<u32>().map_err(|_|"CLI_GRAPH_CAPACITY")?)}else{None};
 let batch=if args.len()>4{Some(args[4].parse::<u32>().map_err(|_|"CLI_GRAPH_BATCH")?)}else{None};
 println!("{}",admit(&g,devices,inventory,shards,requested,batch)?);Ok(())
}
fn admit(g:&GraphDefinitionV2,devices:Vec<u32>,inventory:Vec<serde_json::Value>,shards:u32,requested:Option<u32>,batch:Option<u32>)->Result<serde_json::Value>{
 if devices.is_empty()||devices.len()>128||inventory.len()!=devices.len(){return Err("GRAPH_GLOBAL_INVENTORY_SHAPE".into());}
 let mut minimum=u64::MAX;
 for v in &inventory{let free=v["free_bytes"].as_u64().ok_or("GRAPH_GLOBAL_FREE_BYTES")?;let total=v["total_bytes"].as_u64().ok_or("GRAPH_GLOBAL_TOTAL_BYTES")?;if free==0||free>total{return Err("GRAPH_GLOBAL_MEMORY_RANGE".into());}minimum=minimum.min(free);}
 let state_bytes=crate::generic_memory::preferred_state_bytes(&g);
 
 let (mut rank_plans,owner_cuts)=if let Some(capacity)=requested{
  let auto=Plan::automatic_storage(g.start.len() as u32,devices.len() as u32,shards,g.generator_count() as u32,graph_bytes(&g),minimum,Some(u64::from(capacity)),state_bytes)?;
  if auto.capacity!=capacity{return Err("REQUESTED_CAPACITY_EXCEEDS_ADMISSION".into());}
  (vec![auto;devices.len()],None)
 }else{
  let free=inventory.iter().map(|v|v["free_bytes"].as_u64().unwrap()).collect::<Vec<_>>();
  let(plans,cuts)=crate::generic_distributed_memory::heterogeneous_plans(g.start.len() as u32,shards,g.generator_count() as u32,graph_bytes(&g),&free,crate::generic_memory::state_space_bound(&g),state_bytes)?;(plans,Some(cuts))
 };
 if let Some(batch)=batch{
  for(p,item)in rank_plans.iter_mut().zip(&inventory){let candidate=Plan::with_storage(p.elements,p.world,p.shards,p.capacity,batch,g.generator_count() as u32,p.generator_bytes,state_bytes)?;let free=item["free_bytes"].as_u64().unwrap();let budget=free.checked_sub((1u64<<30).max(free/10)).ok_or("GENERIC_DISTRIBUTED_HEADROOM")?;if candidate.device_bytes>budget{return Err("REQUESTED_BATCH_EXCEEDS_ADMISSION".into());}*p=candidate;}
 }
 let plan=rank_plans.iter().min_by_key(|p|p.capacity).unwrap().clone();
 Ok(serde_json::json!({"graph_digest":digest(g)?,"devices":devices,"inventory":inventory,"plan":plan,"rank_plans":rank_plans,"owner_cuts":owner_cuts,"profile_status":"MEMORY_ADMITTED_NOT_THROUGHPUT_TUNED"}))
}
pub fn global_info(args:&[String])->Result<()>{
 if args.len()<3{return Err("CLI_GRAPH_GLOBAL_ARGUMENTS".into());}let g=graph(&args[0])?;let inventory:Vec<serde_json::Value>=serde_json::from_reader(std::fs::File::open(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let devices=inventory.iter().map(|v|v["device"].as_u64().filter(|&d|d<=u64::from(u32::MAX)).map(|d|d as u32).ok_or("GRAPH_GLOBAL_DEVICE".to_owned())).collect::<Result<Vec<_>>>()?;
 let shards=args[2].parse().map_err(|_|"CLI_GRAPH_SHARDS")?;let requested=if args.len()>3&&args[3]!="auto"{Some(args[3].parse().map_err(|_|"CLI_GRAPH_CAPACITY")?)}else{None};let batch=if args.len()>4{Some(args[4].parse().map_err(|_|"CLI_GRAPH_BATCH")?)}else{None};println!("{}",admit(&g,devices,inventory,shards,requested,batch)?);Ok(())
}

#[derive(serde::Deserialize)]struct Launch{graph_digest:String,devices:Vec<u32>,plan:Plan,#[serde(default)]rank_plans:Vec<Plan>,#[serde(default)]owner_cuts:Option<Vec<u64>>,#[serde(default)]profile_max_layers:Option<u32>}
pub fn run(args:&[String])->Result<()>{
 if args.len()!=6{return Err("CLI_GRAPH_RANK_ARGUMENTS".into());}let g=graph(&args[0])?;let launch:Launch=serde_json::from_reader(std::fs::File::open(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;let rank:u32=args[2].parse().map_err(|_|"CLI_GRAPH_RANK")?;
 if launch.graph_digest!=digest(&g)?||launch.devices.len()!=launch.plan.world as usize||rank>=launch.plan.world||launch.plan.generator_bytes<graph_bytes(&g){return Err("GRAPH_LAUNCH_IDENTITY_OR_GEOMETRY".into());}
 let plans=if launch.rank_plans.is_empty(){vec![launch.plan.clone();launch.devices.len()]}else{launch.rank_plans.clone()};
 if plans.len()!=launch.devices.len(){return Err("GRAPH_RANK_PLANS".into());}
 for p in &plans{p.validate(g.generator_count() as u32)?;if p.world!=launch.plan.world||p.shards!=launch.plan.shards||p.elements!=launch.plan.elements||p.state_bytes!=launch.plan.state_bytes||p.batch!=launch.plan.batch||p.queue_capacity!=launch.plan.queue_capacity||p.generator_bytes<graph_bytes(&g){return Err("GRAPH_TRANSPORT_GEOMETRY".into());}}
 let rank_plan=plans[rank as usize].clone();
 let seconds:u64=args[5].parse().map_err(|_|"CLI_GRAPH_SECONDS")?;if seconds==0{return Err("CLI_GRAPH_SECONDS".into());}let output=Path::new(&args[4]);if output.exists(){return Err("CLI_GRAPH_OUTPUT_EXISTS".into());}std::fs::create_dir_all(output).map_err(|e|e.to_string())?;
 let bootstrap=Path::new(&args[3]);let mut id=[0u8;128];if rank==0{if bootstrap.exists(){return Err("GRAPH_BOOTSTRAP_ALREADY_EXISTS".into());}check(unsafe{mgbfs_nccl_unique_id(id.as_mut_ptr().cast())})?;let temporary=bootstrap.with_extension("tmp");std::fs::write(&temporary,id).map_err(|e|e.to_string())?;std::fs::rename(temporary,bootstrap).map_err(|e|e.to_string())?;}else{let started=Instant::now();while !bootstrap.exists(){if started.elapsed()>Duration::from_secs(60){return Err("GRAPH_BOOTSTRAP_TIMEOUT".into());}std::thread::sleep(Duration::from_millis(10));}let raw=std::fs::read(bootstrap).map_err(|e|e.to_string())?;if raw.len()!=128{return Err("GRAPH_BOOTSTRAP_SIZE".into());}id.copy_from_slice(&raw);}
 let _signals=Signals::new();let setup=Instant::now();let mut bfs=GenericDistributedBfs::new_with_cuts(&g,launch.devices[rank as usize],rank,rank_plan.clone(),&id,0x19f856a2,64,launch.owner_cuts.as_deref())?;let setup_seconds=setup.elapsed().as_secs_f64();eprintln!("MGBFS_GRAPH_READY rank={rank} device={}",launch.devices[rank as usize]);
 let start=Instant::now();let mut sizes=vec![1u64];let mut times=vec![];let(status,reason)=loop{
  let local=if CANCELLED.load(Ordering::Relaxed){1}else if launch.profile_max_layers.map_or(false,|limit|sizes.len() as u32>limit){3}else if start.elapsed()>=Duration::from_secs(seconds){2}else{0};let vote=bfs.collective_stop(local)?;if vote!=0{break("INCOMPLETE",if vote==1{"CANCELLED".to_owned()}else if vote==3{"PROFILE_LAYER_LIMIT".to_owned()}else{"DEADLINE".to_owned()});}
  let layer=Instant::now();match bfs.advance()?{DistributedAdvance::Layer{global,..}=>{sizes.push(global);times.push(layer.elapsed().as_secs_f64());},DistributedAdvance::Complete=>break("COMPLETE","FRONTIER_EXHAUSTED".to_owned()),DistributedAdvance::Resource{fatal}=>break("INCOMPLETE",format!("RESOURCE_{fatal}")),}
 };let bfs_seconds=start.elapsed().as_secs_f64();let current=bfs.sample_global(1000)?;let previous=bfs.previous_small_sample(*sizes.iter().rev().nth(1).unwrap_or(&0))?;
 let states=serde_json::json!({"schema":2,"state_encoding":"signed_int64_vectors","current":current,"previous_small":previous,"current_sample_global_limit":1000});let state_bytes=serde_json::to_vec(&states).map_err(|e|e.to_string())?;
 use sha2::{Digest,Sha256};let sha=Sha256::digest(&state_bytes).iter().map(|v|format!("{v:02x}")).collect::<String>();std::fs::write(output.join("states.json"),state_bytes).map_err(|e|e.to_string())?;
 let report=serde_json::json!({"schema":2,"rank":rank,"world":launch.plan.world,"device":launch.devices[rank as usize],"status":status,"reason":reason,"graph_digest":launch.graph_digest,"states_sha256":sha,"layer_sizes":sizes,"layer_seconds":times,"bfs_seconds":bfs_seconds,"source_retries":bfs.source_retries(),"setup_seconds":setup_seconds,"plan":rank_plan,"backend":"GENERIC_DISTRIBUTED_EXACT_ALL_VISITED","profile_status":"MEMORY_ADMITTED_NOT_THROUGHPUT_TUNED"});
 std::fs::write(output.join("report.json.tmp"),serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;std::fs::rename(output.join("report.json.tmp"),output.join("report.json")).map_err(|e|e.to_string())?;Ok(())
}
