//! Resident generic rank: fresh per-job buffers and live device inventory.
use mgbfs_core::Result;
use std::{path::Path,collections::BTreeMap};
use mgbfs_cuda::native_owner::{cudaSetDevice,cudaMemGetInfo};
extern "C" {fn cudaDeviceSynchronize()->i32;fn mgbfs_generic_hardware_info(device:i32,values:*mut u64)->i32;}
pub fn run(root:&Path,rank:u32,device:u32)->Result<()> {
 crate::session_cache::enable();crate::generic_pool::activate(device as i32)?;
 let waiter=crate::session_wait::FileWaiter::new(root);let shutdown=root.join("shutdown");
 let mut previous=BTreeMap::<String,Option<String>>::new();
 for sequence in 0u64.. {
  let path=root.join(format!("job-{sequence:08}.json"));
  if !waiter.wait(&path,Some(&shutdown),None)?{return Ok(());}
  let outcome=(||->Result<serde_json::Value>{
   let v:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
   if v["schema"]!=1||v["sequence"].as_u64()!=Some(sequence){return Err("GENERIC_SESSION_SEQUENCE".into());}
   for(k,value)in std::mem::take(&mut previous){match value{Some(value)=>std::env::set_var(k,value),None=>std::env::remove_var(k)}}
   let env:BTreeMap<String,String>=serde_json::from_value(v["env"].clone()).map_err(|e|e.to_string())?;
   if env.keys().any(|k|!k.starts_with("MGBFS_")&&!k.starts_with("NCCL_")){return Err("GENERIC_SESSION_ENV".into());}
   for(k,value)in env{previous.insert(k.clone(),std::env::var(&k).ok());std::env::set_var(k,value);}
   crate::native_fault::reset(Some(rank),Some(device));
   crate::job_phase::set("preparing")?;
   crate::job_phase::set("control-thread-start")?;
   let _job_control=crate::job_phase::Control::start();
   crate::job_phase::set("control-thread-ready")?;
   match v["command"].as_str(){
    Some("release-memory")=>{crate::generic_native::checked!(cudaSetDevice, unsafe{cudaSetDevice(device as i32)})?;crate::generic_native::checked!(cudaDeviceSynchronize, unsafe{cudaDeviceSynchronize()})?;crate::session_cache::release_idle_storage();crate::generic_pool::trim_idle()?;Ok(serde_json::json!({"released":true}))},
    Some("bench")=>{
     // Generic and host-NCCL reference jobs share the same healthy rank.
     std::env::set_var("WORLD_SIZE",std::env::var("MGBFS_BENCH_WORLD_SIZE").map_err(|_|"ENV_WORLD_SIZE")?);
     let args:Vec<String>=serde_json::from_value(v["args"].clone()).map_err(|e|e.to_string())?;
     if args.len()!=6||!args[1].starts_with("lrx"){return Err("GENERIC_SESSION_BENCH_ARGS".into());}
     std::env::set_var("TORCHELASTIC_RUN_ID",&args[3]);
     match crate::reference_bench::run(args){Ok(())=>Ok(serde_json::json!({"pid":std::process::id(),"exit_code":0})),Err(error) if error=="MEMORY_QUERY_DONE"=>{eprintln!("{}",serde_json::json!({"status":"ERROR","rank":rank,"error":error}));Ok(serde_json::json!({"pid":std::process::id(),"exit_code":1,"query_only":true}))},Err(error)=>Err(error)}
    },
    Some("inventory")=>{
     crate::job_phase::set("inventory-cudaSetDevice")?;
     crate::generic_native::checked!(cudaSetDevice, unsafe{cudaSetDevice(device as i32)})?;
     crate::job_phase::set("inventory-cudaMemGetInfo")?;
     let(mut free,mut total)=(0usize,0usize);crate::generic_native::checked!(cudaMemGetInfo, unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
     crate::job_phase::set("inventory-hardware-info")?;
     let mut hw=[0u64;8];crate::generic_native::checked!(mgbfs_generic_hardware_info, unsafe{mgbfs_generic_hardware_info(device as i32,hw.as_mut_ptr())})?;
     crate::job_phase::set("inventory-pool-accounting")?;
     Ok(serde_json::json!({"device":device,"free_bytes":(free as u64+crate::session_cache::reusable_bytes()+crate::generic_pool::available()?).min(total as u64),"additional_reserve_bytes":crate::generic_memory::admission_reserve_bytes()?,"nccl_warmed":crate::session_cache::warmed_communicator(),"cached_reusable_bytes":crate::session_cache::reusable_bytes(),"pool_reusable_bytes":crate::generic_pool::available()?,"total_bytes":total,"sm_count":hw[0],"compute_major":hw[1],"compute_minor":hw[2],"l2_bytes":hw[3],"warp_size":hw[4],"threads_per_sm":hw[5],"threads_per_block":hw[6],"shared_bytes_per_sm":hw[7]}))
    },
    Some("graph-local-plan")=>{let args:Vec<String>=serde_json::from_value(v["args"].clone()).map_err(|e|e.to_string())?;if args.get(1)!=Some(&device.to_string()){return Err("GENERIC_SESSION_LOCAL_PLAN_DEVICE".into());}crate::generic_distributed_run::local_plan(&args)},
    Some("graph-plan")=>{let args:Vec<String>=serde_json::from_value(v["args"].clone()).map_err(|e|e.to_string())?;crate::generic_distributed_run::global_plan(&args)},
    Some("graph-rank")=>{
     // Backend-specific buffers are retired at shape changes, not contexts.
     let args:Vec<String>=serde_json::from_value(v["args"].clone()).map_err(|e|e.to_string())?;
     if args.len()!=6||args[2]!=rank.to_string(){return Err("GENERIC_SESSION_RANK".into());}
     let launch:serde_json::Value=serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
     if launch["devices"][rank as usize].as_u64()!=Some(device as u64){return Err("GENERIC_SESSION_DEVICE".into());}
     crate::generic_distributed_run::run(&args)?;Ok(serde_json::json!({"pid":std::process::id()}))
    },_=>Err("GENERIC_SESSION_COMMAND".into())
   }
  })();
  crate::job_phase::set(if outcome.is_ok(){"done"}else{"failed"})?;
  let response=match &outcome{Ok(v)=>serde_json::json!({"schema":1,"sequence":sequence,"ok":true,"result":v,"cache":crate::session_cache::stats()}),Err(e)=>serde_json::json!({"schema":1,"sequence":sequence,"ok":false,"error":e})};
  let destination=root.join(format!("response-{sequence:08}.json"));let temporary=destination.with_extension("tmp");
  std::fs::write(&temporary,serde_json::to_vec(&response).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;std::fs::rename(temporary,destination).map_err(|e|e.to_string())?;
  outcome?; // Errors terminate the worker; never reuse failed CUDA/NCCL state.
 }
 Ok(())
}
