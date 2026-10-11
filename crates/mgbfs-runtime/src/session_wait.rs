//! Kernel-notified local file handoff. No spinning between resident jobs.
use std::{path::Path,time::{Duration,Instant},os::unix::ffi::OsStrExt};
use mgbfs_core::Result;
pub(crate) struct FileWaiter{fd:i32}
impl FileWaiter {
 pub fn new(directory:&Path)->Self {
  let fd=unsafe{libc::inotify_init1(libc::IN_CLOEXEC|libc::IN_NONBLOCK)};
  if fd<0{return Self{fd:-1};}
  let name=match std::ffi::CString::new(directory.as_os_str().as_bytes()){Ok(v)=>v,Err(_)=>{unsafe{libc::close(fd);}return Self{fd:-1};}};
  if unsafe{libc::inotify_add_watch(fd,name.as_ptr(),libc::IN_MOVED_TO|libc::IN_CREATE|libc::IN_CLOSE_WRITE)}<0{unsafe{libc::close(fd);}return Self{fd:-1};}
  Self{fd}
 }
 pub fn wait(&self,path:&Path,shutdown:Option<&Path>,timeout:Option<Duration>)->Result<bool>{
  let start=Instant::now();
  loop {
   if path.exists(){return Ok(true);}
   if shutdown.is_some_and(Path::exists)||timeout.is_some_and(|v|start.elapsed()>=v){return Ok(false);}
   if self.fd<0{std::thread::sleep(Duration::from_millis(1));continue;}
   let millis=timeout.map_or(100,|v|v.saturating_sub(start.elapsed()).as_millis().min(100).max(1) as i32);
   let mut item=libc::pollfd{fd:self.fd,events:libc::POLLIN,revents:0};
   let result=unsafe{libc::poll(&mut item,1,millis)};
   if result<0{let e=std::io::Error::last_os_error();if e.raw_os_error()==Some(libc::EINTR){continue;}return Err(format!("SESSION_FILE_WAIT_{e}"));}
   if result>0 {
    if item.revents&(libc::POLLERR|libc::POLLHUP|libc::POLLNVAL)!=0{return Err("SESSION_FILE_WAIT_FD".into());}
    let mut events=[0u8;8192];while unsafe{libc::read(self.fd,events.as_mut_ptr().cast(),events.len())}>0{}
   }
  }
 }
}
impl Drop for FileWaiter{fn drop(&mut self){if self.fd>=0{unsafe{libc::close(self.fd);}}}}
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn rename_wakes_without_missing_early_event(){
  let root=std::env::temp_dir().join(format!("mgbfs-wait-{}",std::process::id()));std::fs::create_dir_all(&root).unwrap();let waiter=FileWaiter::new(&root);let target=root.join("job.json");let temporary=root.join("tmp");
  std::fs::write(&temporary,b"job").unwrap();std::fs::rename(&temporary,&target).unwrap();assert!(waiter.wait(&target,None,Some(Duration::from_millis(100))).unwrap());
  std::fs::remove_file(&target).unwrap();assert!(!waiter.wait(&target,None,Some(Duration::from_millis(5))).unwrap());
  let shutdown=root.join("shutdown");std::fs::write(&shutdown,b"").unwrap();assert!(!waiter.wait(&target,Some(&shutdown),None).unwrap());std::fs::remove_dir_all(root).unwrap();
 }
}
