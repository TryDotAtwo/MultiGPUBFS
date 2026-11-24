//! Bounded host-side agreement before segmented CUDA/NCCL phases.
//! The path is scoped to one unique bootstrap; every rank publishes once.
use std::{io::Read,path::{Path,PathBuf},sync::OnceLock,time::{Duration,Instant}};
use mgbfs_core::Result;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};

pub(crate) struct HostAgreement{root:PathBuf,rank:u32,world:u32,timeout:Duration,prefix:&'static str}
impl HostAgreement{
 pub fn new(bootstrap:&Path,rank:u32,world:u32)->Result<Self>{
  if world==0||world>128||rank>=world{return Err("SEGMENTED_HOST_GEOMETRY".into())}
  let filename=bootstrap.file_name().ok_or("SEGMENTED_HOST_BOOTSTRAP")?.to_string_lossy();
  let root=bootstrap.parent().ok_or("SEGMENTED_HOST_BOOTSTRAP")?.join(format!("{filename}.segmented-host"));
  std::fs::create_dir_all(&root).map_err(|e|format!("SEGMENTED_HOST_DIRECTORY {e}"))?;
  // This limits startup handshakes only; the BFS itself has no deadline.
  let seconds=std::env::var("MGBFS_SEGMENT_STARTUP_AGREEMENT_SECONDS").ok()
   .map(|v|v.parse::<u64>().map_err(|_|"SEGMENTED_HOST_TIMEOUT_RANGE".to_owned()))
   .transpose()?.unwrap_or(300);
  if !(1..=1800).contains(&seconds){return Err("SEGMENTED_HOST_TIMEOUT_RANGE".into())}
  let ms=if std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref()==Ok("1"){
   std::env::var("MGBFS_SEGMENT_TEST_HOST_TIMEOUT_MS").ok().and_then(|v|v.parse::<u64>().ok()).unwrap_or(seconds*1000)
  }else{seconds*1000};
  if !(100..=1800000).contains(&ms){return Err("SEGMENTED_HOST_TIMEOUT_RANGE".into())}
  Ok(Self{root,rank,world,timeout:Duration::from_millis(ms),prefix:""})
 }
 pub fn reverse(&self)->Self{
  Self{root:self.root.clone(),rank:self.rank,world:self.world,timeout:self.timeout,prefix:"reverse-"}
 }
 pub fn agree<T>(&self,phase:&str,local:&Result<T>,descriptor:Option<&Value>)->Result<()>{
  let phase=format!("{}{phase}",self.prefix);
  let result=self.agree_inner(&phase,local,descriptor);
  if let Err(error)=&result{crate::native_fault::fatal_host("host-agreement",error);}
  result
 }
 fn agree_inner<T>(&self,phase:&str,local:&Result<T>,descriptor:Option<&Value>)->Result<()>{
  if phase.is_empty()||!phase.bytes().all(|v|v.is_ascii_alphanumeric()||v==b'-'){return Err("SEGMENTED_HOST_PHASE".into())}
  crate::job_phase::set(&format!("segmented-host-{phase}"))?;
  let record=json!({"schema":1,"phase":phase,"rank":self.rank,"world":self.world,
    "error":local.as_ref().err(),"descriptor":descriptor});
  let target=self.root.join(format!("{phase}-rank-{}.json",self.rank));
  if target.exists(){return Err(format!("SEGMENTED_HOST_DUPLICATE phase={phase} rank={}",self.rank))}
  let temporary=self.root.join(format!(".{phase}-rank-{}-{}.tmp",self.rank,std::process::id()));
  std::fs::write(&temporary,serde_json::to_vec(&record).map_err(|e|e.to_string())?).map_err(|e|format!("SEGMENTED_HOST_WRITE {e}"))?;
  std::fs::rename(&temporary,&target).map_err(|e|format!("SEGMENTED_HOST_RENAME {e}"))?;
  let waiter=crate::session_wait::FileWaiter::new(&self.root);let deadline=Instant::now()+self.timeout;
  let mut records=Vec::with_capacity(self.world as usize);
  for rank in 0..self.world{
   let file=self.root.join(format!("{phase}-rank-{rank}.json"));
   let left=deadline.saturating_duration_since(Instant::now());
   if !waiter.wait(&file,None,Some(left))?{return Err(format!("SEGMENTED_HOST_TIMEOUT phase={phase} rank_waiting={} missing_rank={rank}",self.rank))}
   let record:Value=serde_json::from_slice(&std::fs::read(&file).map_err(|e|format!("SEGMENTED_HOST_READ {e}"))?).map_err(|e|format!("SEGMENTED_HOST_JSON {e}"))?;
   if record["schema"]!=1||record["phase"]!=phase||record["rank"]!=rank||record["world"]!=self.world{return Err(format!("SEGMENTED_HOST_RECORD phase={phase} rank={rank}"))}
   records.push(record);
  }
  for (rank,record) in records.iter().enumerate(){if let Some(error)=record["error"].as_str(){return Err(format!("SEGMENTED_HOST_FAILURE phase={phase} rank={rank} error={error}"))}}
  let first=&records[0]["descriptor"];
  for (rank,record) in records.iter().enumerate().skip(1){if &record["descriptor"]!=first{return Err(format!("SEGMENTED_HOST_MISMATCH phase={phase} rank={rank} expected={first} actual={}",record["descriptor"]))}}
  Ok(())
 }
}
fn sha(path:&Path)->Result<String>{
 let mut f=std::fs::File::open(path).map_err(|e|format!("SEGMENTED_HOST_IDENTITY_OPEN {e}"))?;
 let mut digest=Sha256::new();let mut buf=[0u8;65536];loop{let n=f.read(&mut buf).map_err(|e|format!("SEGMENTED_HOST_IDENTITY_READ {e}"))?;if n==0{break}digest.update(&buf[..n]);}
 Ok(digest.finalize().iter().map(|v|format!("{v:02x}")).collect())
}
/// Hash the running executable and the CUDA library path once per worker.
/// Native artifacts must remain immutable for the worker lifetime.
pub(crate) fn runtime_identity()->Result<Value>{
 static IDENTITY:OnceLock<std::result::Result<Value,String>>=OnceLock::new();
 IDENTITY.get_or_init(||unsafe{
  let executable=PathBuf::from("/proc/self/exe");
  let mut info:libc::Dl_info=std::mem::zeroed();
  let symbol=libc::dlsym(libc::RTLD_DEFAULT,b"mgbfs_segmented_abi_version\0".as_ptr().cast());
  if symbol.is_null()||libc::dladdr(symbol,&mut info)==0||info.dli_fname.is_null(){return Err("SEGMENTED_HOST_LIBRARY_IDENTITY".into())}
  let library=PathBuf::from(std::ffi::CStr::from_ptr(info.dli_fname).to_string_lossy().as_ref());
  let library_for=|name:&[u8]|->Result<PathBuf>{
   let symbol=libc::dlsym(libc::RTLD_DEFAULT,name.as_ptr().cast());let mut info:libc::Dl_info=std::mem::zeroed();
   if symbol.is_null()||libc::dladdr(symbol,&mut info)==0||info.dli_fname.is_null(){return Err(format!("SEGMENTED_HOST_DEPENDENCY_IDENTITY {:?}",name));}
   Ok(PathBuf::from(std::ffi::CStr::from_ptr(info.dli_fname).to_string_lossy().as_ref()))
  };
  let cudart=library_for(b"cudaRuntimeGetVersion\0")?;let nccl=library_for(b"ncclGetVersion\0")?;
  Ok(json!({"exe_sha256":sha(&executable)?,"library_sha256":sha(&library)?,
   "cudart_sha256":sha(&cudart)?,"nccl_sha256":sha(&nccl)?}))
 }).clone()
}
pub(crate) fn injected(phase:&str)->Result<()>{
 if std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref()!=Ok("1"){return Ok(())}
 let fail=std::env::var("MGBFS_SEGMENT_TEST_HOST_FAIL").unwrap_or_default();
 let selected=std::env::var("MGBFS_SEGMENT_TEST_FAIL_RANK").ok().map_or(true,|r|std::env::var("RANK").ok().as_deref()==Some(r.as_str()));
 if !selected{return Ok(())}
 if phase=="entry"&&fail=="exit"{std::process::exit(42)}
 if fail==phase{
  let error=format!("SEGMENTED_TEST_HOST_FAIL_{}",phase.to_ascii_uppercase());
  if matches!(phase,"after-seed"|"count-payload"){crate::native_fault::fatal_host(phase,&error);}
  return Err(error)
 }
 Ok(())
}

#[cfg(test)] mod tests{
 use super::*;
 use std::sync::atomic::{AtomicU64,Ordering};
 static NEXT:AtomicU64=AtomicU64::new(0);
 fn cases(desc0:Value,desc1:Value,fail:bool)->(String,String){
  let root=std::env::temp_dir().join(format!("mgbfs-host-agree-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
  std::fs::create_dir_all(&root).unwrap();
  let workers:Vec<_>=[desc0,desc1].into_iter().enumerate().map(|(rank,desc)|{
   let root=root.clone();std::thread::spawn(move||{
    let a=HostAgreement{root,rank:rank as u32,world:2,timeout:Duration::from_secs(2),prefix:""};
    let local:Result<()>=if rank==1&&fail{Err("UNIT_EXPECTED_FAILURE".into())}else{Ok(())};
    a.agree("cpu-test",&local,Some(&desc)).err().unwrap_or_else(||"OK".to_owned())
   })
  }).collect();
  let mut out=workers.into_iter().map(|x|x.join().unwrap()).collect::<Vec<_>>();
  std::fs::remove_dir_all(root).unwrap();(out.remove(0),out.remove(0))
 }
 #[test]fn host_agreement_and_original_error(){
  assert_eq!(cases(json!({"id":1}),json!({"id":1}),false),("OK".into(),"OK".into()));
  let(a,b)=cases(json!({"id":1}),json!({"id":2}),false);assert!(a.contains("SEGMENTED_HOST_MISMATCH")&&b.contains("SEGMENTED_HOST_MISMATCH"));
  let(a,b)=cases(json!({"id":1}),json!({"id":1}),true);assert!(a.contains("rank=1 error=UNIT_EXPECTED_FAILURE")&&b.contains("rank=1 error=UNIT_EXPECTED_FAILURE"));
 }
 #[test]fn missing_rank_times_out_and_stale_phase_rejected(){
  let root=std::env::temp_dir().join(format!("mgbfs-host-timeout-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
  std::fs::create_dir_all(&root).unwrap();
  let a=HostAgreement{root:root.clone(),rank:0,world:2,timeout:Duration::from_millis(120),prefix:""};
  assert!(a.agree("lost",&Ok(()),None).unwrap_err().contains("missing_rank=1"));
  assert!(a.agree("lost",&Ok(()),None).unwrap_err().contains("SEGMENTED_HOST_DUPLICATE"));
  std::fs::remove_dir_all(root).unwrap();
 }
}
