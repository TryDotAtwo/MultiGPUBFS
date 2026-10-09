use std::{path::Path,time::{Instant,Duration},sync::atomic::{AtomicBool,Ordering}};
use mgbfs_core::{graph_definition::GraphDefinitionV2,Result};
use crate::{generic_native::{GenericNativeBfs,GenericAdvance},generic_memory::GenericMemoryPlan};
pub(crate) static CANCELLED:AtomicBool=AtomicBool::new(false);
extern "C" fn cancel(_:i32){CANCELLED.store(true,Ordering::Relaxed);}
pub(crate) struct Signals{term:libc::sighandler_t,int:libc::sighandler_t}
impl Signals{pub(crate) fn new()->Self{CANCELLED.store(false,Ordering::Relaxed);unsafe{Self{term:libc::signal(libc::SIGTERM,cancel as libc::sighandler_t),int:libc::signal(libc::SIGINT,cancel as libc::sighandler_t)}}}}
impl Drop for Signals{fn drop(&mut self){unsafe{libc::signal(libc::SIGTERM,self.term);libc::signal(libc::SIGINT,self.int);}}}
pub fn run(args:&[String])->Result<()> {
 if args.len()<2{return Err("CLI_USAGE: mgbfs graph <graph.json> <output-dir> [--device N] [--capacity N] [--seconds N]".into());}
 let mut device=0u32;let mut capacity=None;let mut seconds=3600u64;let mut i=2;
 while i<args.len(){if i+1>=args.len(){return Err("CLI_GRAPH_OPTION_VALUE".into());}match args[i].as_str(){
  "--device"=>device=args[i+1].parse().map_err(|_|"CLI_GRAPH_DEVICE")?,
  "--capacity"=>capacity=Some(args[i+1].parse::<u32>().map_err(|_|"CLI_GRAPH_CAPACITY")?),
  "--seconds"=>seconds=args[i+1].parse().map_err(|_|"CLI_GRAPH_SECONDS")?,
  _=>return Err("CLI_GRAPH_UNKNOWN_OPTION".into()),}i+=2;}
 if seconds==0{return Err("CLI_GRAPH_SECONDS".into());}
 if std::env::var("WORLD_SIZE").ok().is_some_and(|v|v!="1"){return Err("GENERIC_DISTRIBUTED_INTEGRATION_NOT_READY".into());}
 let graph:GraphDefinitionV2=serde_json::from_reader(std::fs::File::open(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;graph.validate()?;
 let output=Path::new(&args[1]);if output.exists(){return Err("CLI_GRAPH_OUTPUT_EXISTS".into());}
 std::fs::create_dir_all(output).map_err(|e|e.to_string())?;
 let plan=match capacity{Some(c)=>GenericMemoryPlan::new(graph.start.len() as u32,c)?,None=>GenericNativeBfs::automatic_plan(&graph,device)?};
 let allocation=plan.device_bytes;let capacity=plan.capacity;let setup=Instant::now();let mut bfs=GenericNativeBfs::new(&graph,device,plan,0x19f856a2,64)?;let setup_seconds=setup.elapsed().as_secs_f64();
 let _signals=Signals::new();eprintln!("MGBFS_GRAPH_READY device={device}");let started=Instant::now();let mut counts=vec![1u32];let mut times=vec![];let (status,reason)=loop{
  if CANCELLED.load(Ordering::Relaxed){bfs.stop();break ("INCOMPLETE","CANCELLED".to_owned());}
  if started.elapsed()>=Duration::from_secs(seconds){bfs.stop();break ("INCOMPLETE","DEADLINE".to_owned());}
  let layer=Instant::now();match bfs.advance()?{
   GenericAdvance::Layer{count}=>{counts.push(count);times.push(layer.elapsed().as_secs_f64());},
   GenericAdvance::Complete=>break ("COMPLETE","FRONTIER_EXHAUSTED".to_owned()),
   GenericAdvance::Resource{fatal}=>break ("INCOMPLETE",format!("RESOURCE_{fatal}")),
  }
 };
 let bfs_seconds=started.elapsed().as_secs_f64();let current=bfs.sample(1000)?;let previous=bfs.previous_small_sample()?;
 let states=serde_json::json!({"schema":2,"state_encoding":"signed_int64_vectors","current":current,"previous_small":previous,"current_sample_limit":1000});
 let state_bytes=serde_json::to_vec(&states).map_err(|e|e.to_string())?;
 use sha2::{Digest,Sha256};let state_sha=Sha256::digest(&state_bytes).iter().map(|b|format!("{b:02x}")).collect::<String>();
 let report=serde_json::json!({"schema":2,"status":status,"reason":reason,"backend":"GENERIC_EXACT_ALL_VISITED","device":device,"world":1,"graph_digest":graph.semantic_digest()?.iter().map(|b|format!("{b:02x}")).collect::<String>(),"layer_sizes":counts,"layer_seconds":times,"setup_seconds":setup_seconds,"bfs_seconds":bfs_seconds,"capacity":capacity,"arena_bytes":allocation,"states_sha256":state_sha,"retention":"current sample <=1000 and previous layer only if <1000","scope":"single-device general path; distributed SHARD_AB integration pending"});
 std::fs::write(output.join("states.json"),state_bytes).map_err(|e|e.to_string())?;
 let bytes=serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?;std::fs::write(output.join("report.json.tmp"),bytes).map_err(|e|e.to_string())?;std::fs::rename(output.join("report.json.tmp"),output.join("report.json")).map_err(|e|e.to_string())?;
 println!("{report}");Ok(())
}
