#[cfg(not(feature="cuda"))]fn main(){panic!("CUDA required");}
#[cfg(feature="cuda")]fn main(){if let Err(e)=run(){eprintln!("{e}");std::process::exit(1);}}
#[cfg(feature="cuda")]fn run()->mgbfs_core::Result<()>{
 use std::{path::Path,time::{Instant,Duration}};
 use mgbfs_core::graph_definition::GraphDefinitionV2;
 use mgbfs_runtime::{generic_distributed_memory::GenericDistributedMemoryPlan,generic_distributed_native::{GenericDistributedBfs,DistributedAdvance}};
 use mgbfs_cuda::ffi::mgbfs_nccl_unique_id;
 let a=std::env::args().collect::<Vec<_>>();let rank:u32=a[1].parse().unwrap();let root=Path::new(&a[2]);let bits:u32=a[4].parse().unwrap();let shards:u32=a[5].parse().unwrap();let capacity:u32=a[6].parse().unwrap();
 let graph:GraphDefinitionV2=serde_json::from_reader(std::fs::File::open(&a[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let mut id=[0u8;128];let bootstrap=root.join("id");if rank==0{let code=unsafe{mgbfs_nccl_unique_id(id.as_mut_ptr().cast())};if code!=0{return Err(format!("UID_{code}"));}std::fs::write(root.join("id.tmp"),id).map_err(|e|e.to_string())?;std::fs::rename(root.join("id.tmp"),&bootstrap).map_err(|e|e.to_string())?;}else{let started=Instant::now();while !bootstrap.exists(){if started.elapsed()>Duration::from_secs(20){return Err("BOOTSTRAP_TIMEOUT".into());}std::thread::sleep(Duration::from_millis(10));}id.copy_from_slice(&std::fs::read(bootstrap).map_err(|e|e.to_string())?);}
 let plan=GenericDistributedMemoryPlan::new(graph.start.len() as u32,2,shards,capacity,capacity.min(2),graph.generator_count() as u32,4096)?;
 let mut bfs=GenericDistributedBfs::new(&graph,rank,rank,plan,&id,123,bits)?;
 let mut layers=vec![];let mut global_sizes=vec![1u64];let mut fatal=0;
 for _ in 0..2048{layers.push(bfs.sample(1000)?);match bfs.advance()?{DistributedAdvance::Layer{global,..}=>global_sizes.push(global),DistributedAdvance::Complete=>break,DistributedAdvance::Resource{fatal:f}=>{fatal=f;break;}}}
 let previous=if fatal!=0{bfs.previous_small_sample(*global_sizes.iter().rev().nth(1).unwrap_or(&0))?}else{vec![]};
 let report=serde_json::json!({"rank":rank,"layers":layers,"fatal":fatal,"global_sizes":global_sizes,"previous_small":previous,"shards":shards,"hash_bits":bits});std::fs::write(root.join(format!("rank-{rank}.json")),serde_json::to_vec(&report).unwrap()).map_err(|e|e.to_string())?;Ok(())
}
