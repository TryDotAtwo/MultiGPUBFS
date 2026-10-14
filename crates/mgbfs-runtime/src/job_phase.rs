//! Job-boundary telemetry; never called inside GPU batch dispatch.
use mgbfs_core::Result;
pub fn set(phase:&str)->Result<()> {
 if let Some(path)=std::env::var_os("MGBFS_DIAGNOSTIC_TRACE_FILE") {
  use std::io::Write;
  let mut file=std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e|e.to_string())?;
  writeln!(file,"PHASE pid={} {}",std::process::id(),phase).map_err(|e|e.to_string())?;
  file.flush().map_err(|e|e.to_string())?;
 }
 let Some(path)=std::env::var_os("MGBFS_JOB_PHASE_PATH") else{return Ok(());};
 let path=std::path::PathBuf::from(path);
 let unix=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e|e.to_string())?.as_secs_f64();
 let mut value=if path.exists(){serde_json::from_slice::<serde_json::Value>(&std::fs::read(&path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?}else{serde_json::json!({"schema":1,"pid":std::process::id(),"sequence":std::env::var("MGBFS_JOB_SEQUENCE").unwrap_or_default(),"entered":{}})};
 value["phase"]=serde_json::json!(phase);value["entered"][phase]=serde_json::json!(unix);
 let tmp=path.with_extension("tmp");std::fs::write(&tmp,serde_json::to_vec(&value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;std::fs::rename(tmp,path).map_err(|e|e.to_string())?;Ok(())
}

static REQUESTED:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);
pub fn cancelled()->bool{REQUESTED.load(std::sync::atomic::Ordering::Relaxed)}
pub struct Control{stop:std::sync::Arc<std::sync::atomic::AtomicBool>,thread:Option<std::thread::JoinHandle<()>>}
impl Control{pub fn start()->Self{
 REQUESTED.store(false,std::sync::atomic::Ordering::Relaxed);
 let stop=std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));let done=stop.clone();
 let thread=std::env::var_os("MGBFS_JOB_CANCEL_PATH").map(|path|std::thread::spawn(move||{
  while !done.load(std::sync::atomic::Ordering::Relaxed){
   if std::path::Path::new(&path).exists(){REQUESTED.store(true,std::sync::atomic::Ordering::Relaxed);break;}
   std::thread::sleep(std::time::Duration::from_millis(5));
  }
 }));Self{stop,thread}
}}
impl Drop for Control{fn drop(&mut self){self.stop.store(true,std::sync::atomic::Ordering::Relaxed);if let Some(thread)=self.thread.take(){let _=thread.join();}}}
