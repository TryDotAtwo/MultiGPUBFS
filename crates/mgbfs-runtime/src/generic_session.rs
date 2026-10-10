//! Resident generic rank: fresh per-job buffers and live device inventory.
use mgbfs_core::Result;
use std::{path::Path,time::Duration,collections::BTreeMap};
use mgbfs_cuda::native_owner::{cudaSetDevice,cudaMemGetInfo};
extern "C" {fn cudaDeviceSynchronize()->i32;fn mgbfs_generic_hardware_info(device:i32,values:*mut u64)->i32;}
pub fn run(root:&Path,rank:u32,device:u32)->Result<()> {
 crate::session_cache::enable();let bench_mode=std::env::var("MGBFS_SESSION_KIND").as_deref()==Ok("bench");if !bench_mode{crate::generic_pool::activate(device as i32)?;}
 let mut previous=BTreeMap::<String,Option<String>>::new();
 for sequence in 0u64.. {
  let path=root.join(format!("job-{sequence:08}.json"));
  while !path.exists(){if root.join("shutdown").exists(){return Ok(());}std::thread::sleep(Duration::from_millis(2));}
  let outcome=(||->Result<serde_json::Value>{
   let v:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
   if v["schema"]!=1||v["sequence"].as_u64()!=Some(sequence){return Err("GENERIC_SESSION_SEQUENCE".into());}
   for(k,value)in std::mem::take(&mut previous){match value{Some(value)=>std::env::set_var(k,value),None=>std::env::remove_var(k)}}
   let env:BTreeMap<String,String>=serde_json::from_value(v["env"].clone()).map_err(|e|e.to_string())?;
   if env.keys().any(|k|!k.starts_with("MGBFS_")&&!k.starts_with("NCCL_")){return Err("GENERIC_SESSION_ENV".into());}
   for(k,value)in env{previous.insert(k.clone(),std::env::var(&k).ok());std::env::set_var(k,value);}
   match v["command"].as_str(){
    Some("release-memory")=>{crate::generic_native::check(unsafe{cudaSetDevice(device as i32)})?;crate::generic_native::check(unsafe{cudaDeviceSynchronize()})?;crate::session_cache::release_idle_storage();crate::generic_pool::trim_idle()?;Ok(serde_json::json!({"released":true}))},
    Some("bench")=>{
     if !bench_mode{return Err("GENERIC_SESSION_BACKEND".into());}
     let args:Vec<String>=serde_json::from_value(v["args"].clone()).map_err(|e|e.to_string())?;
     if args.len()!=6||!args[1].starts_with("lrx"){return Err("GENERIC_SESSION_BENCH_ARGS".into());}
     match crate::reference_bench::run(args){Ok(())=>Ok(serde_json::json!({"pid":std::process::id(),"exit_code":0})),Err(error) if error=="MEMORY_QUERY_DONE"=>{eprintln!("{}",serde_json::json!({"status":"ERROR","rank":rank,"error":error}));Ok(serde_json::json!({"pid":std::process::id(),"exit_code":1,"query_only":true}))},Err(error)=>Err(error)}
    },
    Some("inventory")=>{
     crate::generic_native::check(unsafe{cudaSetDevice(device as i32)})?;
     let(mut free,mut total)=(0usize,0usize);crate::generic_native::check(unsafe{cudaMemGetInfo(&mut free,&mut total)})?;
     let mut hw=[0u64;8];crate::generic_native::check(unsafe{mgbfs_generic_hardware_info(device as i32,hw.as_mut_ptr())})?;
     Ok(serde_json::json!({"device":device,"free_bytes":(free as u64+crate::session_cache::reusable_bytes()+crate::generic_pool::available()?).min(total as u64),"cached_reusable_bytes":crate::session_cache::reusable_bytes(),"pool_reusable_bytes":crate::generic_pool::available()?,"total_bytes":total,"sm_count":hw[0],"compute_major":hw[1],"compute_minor":hw[2],"l2_bytes":hw[3],"warp_size":hw[4],"threads_per_sm":hw[5],"threads_per_block":hw[6],"shared_bytes_per_sm":hw[7]}))
    },
    Some("graph-local-plan")=>{let args:Vec<String>=serde_json::from_value(v["args"].clone()).map_err(|e|e.to_string())?;if args.get(1)!=Some(&device.to_string()){return Err("GENERIC_SESSION_LOCAL_PLAN_DEVICE".into());}crate::generic_distributed_run::local_plan(&args)},
    Some("graph-plan")=>{let args:Vec<String>=serde_json::from_value(v["args"].clone()).map_err(|e|e.to_string())?;crate::generic_distributed_run::global_plan(&args)},
    Some("graph-rank")=>{
     if bench_mode{return Err("GENERIC_SESSION_BACKEND".into());}
     let args:Vec<String>=serde_json::from_value(v["args"].clone()).map_err(|e|e.to_string())?;
     if args.len()!=6||args[2]!=rank.to_string(){return Err("GENERIC_SESSION_RANK".into());}
     let launch:serde_json::Value=serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
     if launch["devices"][rank as usize].as_u64()!=Some(device as u64){return Err("GENERIC_SESSION_DEVICE".into());}
     crate::generic_distributed_run::run(&args)?;Ok(serde_json::json!({"pid":std::process::id()}))
    },_=>Err("GENERIC_SESSION_COMMAND".into())
   }
  })();
  let response=match &outcome{Ok(v)=>serde_json::json!({"schema":1,"sequence":sequence,"ok":true,"result":v}),Err(e)=>serde_json::json!({"schema":1,"sequence":sequence,"ok":false,"error":e})};
  let destination=root.join(format!("response-{sequence:08}.json"));let temporary=destination.with_extension("tmp");
  std::fs::write(&temporary,serde_json::to_vec(&response).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;std::fs::rename(temporary,destination).map_err(|e|e.to_string())?;
  outcome?; // Errors terminate the worker; never reuse failed CUDA/NCCL state.
 }
 Ok(())
}
