use mgbfs_core::{matrix::MatrixGroup, Result};
use mgbfs_cuda::ffi::mgbfs_nccl_unique_id;
use mgbfs_runtime::distributed_native::{DistributedConfig, DistributedNativeBfs};
use std::{path::Path, time::{Duration, Instant}};

fn env_u32(key:&str)->Result<u32>{std::env::var(key).map_err(|_|format!("ENV_{key}"))?.parse().map_err(|_|format!("ENV_{key}"))}
fn bootstrap(path:&Path,rank:u32,world:u32)->Result<[u8;128]>{
    const MAGIC:&[u8;8]=b"MGBNCCL1";
    if rank==0{
        let mut id=[0u8;128];if unsafe{mgbfs_nccl_unique_id(id.as_mut_ptr().cast())}!=0{return Err("NCCL_UNIQUE_ID".into())}
        let mut bytes=Vec::with_capacity(140);bytes.extend_from_slice(MAGIC);bytes.extend_from_slice(&world.to_le_bytes());bytes.extend_from_slice(&id);
        let temporary=path.with_extension("rank0.tmp");if temporary.exists()||path.exists(){return Err("BOOTSTRAP_EXISTS".into())}std::fs::write(&temporary,&bytes).map_err(|e|e.to_string())?;std::fs::rename(temporary,path).map_err(|e|e.to_string())?;return Ok(id)
    }
    let started=Instant::now();loop{if let Ok(bytes)=std::fs::read(path){if bytes.len()!=140||&bytes[..8]!=MAGIC||u32::from_le_bytes(bytes[8..12].try_into().unwrap())!=world{return Err("BOOTSTRAP_FORMAT".into())}return Ok(bytes[12..].try_into().unwrap())}if started.elapsed()>Duration::from_secs(60){return Err("BOOTSTRAP_TIMEOUT".into())}std::thread::sleep(Duration::from_millis(10));}
}
fn hex(bytes:&[u8])->String{const H:&[u8;16]=b"0123456789abcdef";let mut out=String::with_capacity(bytes.len()*2);for &x in bytes{out.push(H[(x>>4)as usize]as char);out.push(H[(x&15)as usize]as char)}out}
fn run()->Result<()> {
    let rank=env_u32("RANK")?;let local=env_u32("LOCAL_RANK")?;let world=env_u32("WORLD_SIZE")?;if rank!=local{return Err("SINGLE_NODE_RANK_MAP".into())}
    let path=std::env::args().nth(1).ok_or("BOOTSTRAP_ARG")?;let id=bootstrap(Path::new(&path),rank,world)?;let graph=MatrixGroup::unitriangular(4,2)?;
    let rank_map=match std::env::var("MGBFS_RANK_MAP").as_deref(){Ok("1,0")=>[1,0],Ok("0,1")|Err(_)=>[0,1],_=>return Err("RANK_MAP".into())};
    let mut bfs=DistributedNativeBfs::new(&graph,20260828u128.to_le_bytes(),id,DistributedConfig{rank,world,logical_owner_to_rank:rank_map,batch:7,layer_capacity:64,future_capacity:64,prededup:true,generation_variant:1})?;
    let mut counts=Vec::new();let mut states=Vec::new();loop{counts.push(bfs.frontier_len());let mut layer=bfs.snapshot()?;layer.sort();states.push(layer.iter().map(|x|hex(x)).collect::<Vec<_>>());if !bfs.advance()?{break}}
    println!("{{\"status\":\"COMPLETE\",\"rank\":{rank},\"layer_sizes\":{counts:?},\"states\":{states:?}}}");Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("DISTRIBUTED_INCOMPLETE: {e}");std::process::exit(1)}}
