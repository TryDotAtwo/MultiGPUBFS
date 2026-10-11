//! Only resident-session jobs use this rank-local, shape-bounded cache.
use std::{cell::RefCell, ffi::c_void};
use mgbfs_cuda::ffi::*;
struct Cache {
    enabled: bool,
    shape: String,
    generic_required:Option<std::collections::BTreeMap<usize,usize>>,
    generic_manifest:Vec<usize>,
    buffers: Vec<(*mut c_void, usize)>,
    pinned: Vec<(*mut c_void, usize, *mut c_void, std::sync::Arc<crate::pinned_archive::HostBlock>)>,
    comm: *mut c_void,
    streams:Vec<(*mut c_void,i32)>,events:Vec<(*mut c_void,i32)>,stream_hits:u64,event_hits:u64,
    buffer_hits: u64,
    comm_hits: u64,
    pool_hits: u64,
    permit_comm: bool,
    nccl_id: Option<[u8;128]>,
    #[cfg(feature = "library-owner")]
    pool: Option<(*mut c_void, u64)>,
}
impl Default for Cache {
    fn default() -> Self {
        Self { enabled:false, shape:String::new(), generic_required:None,generic_manifest:Vec::new(),buffers:Vec::new(), pinned:Vec::new(),
            comm:std::ptr::null_mut(),streams:Vec::new(),events:Vec::new(),stream_hits:0,event_hits:0, buffer_hits:0,comm_hits:0,pool_hits:0,permit_comm:false,nccl_id:None,
            #[cfg(feature = "library-owner")]
            pool:None,
        }
    }
}
impl Cache {
    fn release_storage(&mut self) {
        for (ptr, _) in self.buffers.drain(..) { if !crate::generic_pool::release(ptr){unsafe { cudaFree(ptr); }} }
        #[cfg(feature = "library-owner")]
        if let Some((ptr, _)) = self.pool.take() {
            unsafe { mgbfs_cuda::library_owner::mgbfs_library_pool_destroy_v1(ptr); }
        }
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        self.release_storage();
        for (ptr,device) in self.streams.drain(..){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(device);cudaStreamDestroy(ptr);}}
        for (ptr,device) in self.events.drain(..){unsafe{mgbfs_cuda::native_owner::cudaSetDevice(device);cudaEventDestroy(ptr);}}
        for (_, _, event, _) in self.pinned.drain(..) { unsafe { cudaEventDestroy(event); } }
        if !self.comm.is_null() { unsafe { mgbfs_nccl_destroy(self.comm); } }
    }
}
pub fn prepare_pinned(bytes: usize, slots: usize) {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        // A subset of slots still retains its entire shared allocation. Only
        // reuse an exact geometry; otherwise release the whole pool so the
        // advertised bounded host budget is also the physical allocation.
        if c.pinned.len()!=slots || c.pinned.iter().any(|(_,n,_,_)|*n!=bytes) {
            for (_,_,event,_) in c.pinned.drain(..) {unsafe {cudaEventDestroy(event);}}
            return;
        }
        let mut keep = Vec::new();
        for (ptr, n, event, allocation) in c.pinned.drain(..) {
            if n == bytes && keep.len() < slots { keep.push((ptr,n,event,allocation)); }
            else { unsafe { cudaEventDestroy(event); } }
        }
        c.pinned = keep;
    });
}
pub fn pinned_count(bytes: usize) -> usize {
    CACHE.with(|c|c.borrow().pinned.iter().filter(|(_,n,_,_)|*n==bytes).count())
}
pub fn pinned_take(bytes: usize) -> Option<(*mut c_void,*mut c_void,std::sync::Arc<crate::pinned_archive::HostBlock>)> {
    CACHE.with(|c| {
        let mut c=c.borrow_mut();
        let i=c.pinned.iter().position(|(_,n,_,_)| *n==bytes)?;
        let (ptr,_,event,allocation)=c.pinned.swap_remove(i);Some((ptr,event,allocation))
    })
}
pub fn pinned_put(ptr: *mut c_void, bytes: usize, event: *mut c_void, allocation: std::sync::Arc<crate::pinned_archive::HostBlock>) -> bool {
    CACHE.with(|c| {
        let mut c=c.borrow_mut();if !c.enabled { return false; }
        c.pinned.push((ptr,bytes,event,allocation));true
    })
}
thread_local! { static CACHE: RefCell<Cache> = RefCell::new(Cache::default()); }
pub fn enable() { CACHE.with(|c| c.borrow_mut().enabled = true); }
pub fn enabled() -> bool { CACHE.with(|c| c.borrow().enabled) }
pub fn warmed_communicator() -> bool { CACHE.with(|c| {let c=c.borrow();c.enabled && !c.comm.is_null() && c.permit_comm}) }
pub fn permit_comm(value: bool) { CACHE.with(|c| {
    let mut c=c.borrow_mut();c.permit_comm=value;if !value { c.nccl_id=None; }
}); }
pub fn bootstrap_id(create: impl FnOnce() -> mgbfs_core::Result<[u8;128]>) -> mgbfs_core::Result<[u8;128]> {
    CACHE.with(|c| {
        let mut c=c.borrow_mut();
        // A unique ID belongs to the parked communicator, not the rank process.
        // Query-only LSA construction destroys its communicator on unwind.
        if c.enabled && !c.comm.is_null() { if let Some(id)=c.nccl_id { return Ok(id); } }
        let id=create()?;if c.enabled { c.nccl_id=Some(id); }Ok(id)
    })
}
pub fn begin(shape: String) {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.generic_required=None;c.generic_manifest.clear();c.buffer_hits = 0;c.comm_hits = 0;c.pool_hits = 0;c.stream_hits=0;c.event_hits=0;
        c.permit_comm = false;
        if c.enabled && c.shape != shape { c.release_storage(); c.shape = shape; }
    });
}
/// Retain only exact raw allocation extents required by the incoming job.
/// All prior buffers are idle after the all-rank quiescence gate; no content
/// or graph-specific interpretation is reused. Multiplicities bound storage.
pub fn begin_generic(shape:String,required:&[usize])->mgbfs_core::Result<()> {
 CACHE.with(|cell| {
  let mut c=cell.borrow_mut();
  if std::env::var("MGBFS_REQUIRE_OWNED_RAW").as_deref()==Ok("1") {
   let mut parked=c.buffers.iter().map(|(_,n)|*n).collect::<Vec<_>>();parked.sort_unstable();let mut incoming=required.iter().map(|&n|n.max(1)).collect::<Vec<_>>();incoming.sort_unstable();
   if !c.enabled||!c.permit_comm||c.comm.is_null()||parked!=incoming{return Err("GENERIC_OWNED_RESOURCE_PROOF_CHANGED".into());}
  }
  if !c.shape.starts_with("generic:") {c.release_storage();}
  else {
   let mut need=std::collections::BTreeMap::<usize,usize>::new();
   for &n in required {*need.entry(n.max(1)).or_default()+=1;}
   let mut keep=Vec::new();
   for (ptr,n) in c.buffers.drain(..) {
    if let Some(count)=need.get_mut(&n).filter(|count|**count>0) {*count-=1;keep.push((ptr,n));}
    else if !crate::generic_pool::release(ptr){unsafe{cudaFree(ptr);}}
   }
   c.buffers=keep;
  }
  let mut required_counts=std::collections::BTreeMap::new();for &n in required{*required_counts.entry(n.max(1)).or_insert(0usize)+=1;}c.generic_required=Some(required_counts);c.generic_manifest=required.iter().map(|&n|n.max(1)).collect();c.generic_manifest.sort_unstable();
  c.shape=format!("generic:{shape}");c.buffer_hits=0;c.comm_hits=0;c.pool_hits=0;c.stream_hits=0;c.event_hits=0;c.permit_comm=false;
  Ok(())
 })
}
pub fn generic_buffer_request(bytes:usize)->mgbfs_core::Result<()> {
 CACHE.with(|cell|{let mut c=cell.borrow_mut();if let Some(required)=c.generic_required.as_mut(){let n=required.get_mut(&bytes.max(1)).ok_or("GENERIC_REUSE_EXTENT_NOT_PLANNED")?;if *n==0{return Err("GENERIC_REUSE_EXTENT_MULTIPLICITY".into());}*n-=1;}Ok(())})
}
pub fn generic_requirements_consumed()->mgbfs_core::Result<()> {
 CACHE.with(|cell|{let c=cell.borrow();if c.generic_required.as_ref().is_some_and(|v|v.values().any(|&n|n!=0)){return Err("GENERIC_REUSE_EXTENTS_UNCONSUMED".into());}Ok(())})
}
pub fn stream_take(device:i32)->Option<*mut c_void>{CACHE.with(|cell|{let mut c=cell.borrow_mut();if !c.enabled{return None;}let i=c.streams.iter().position(|(_,d)|*d==device)?;let p=c.streams.swap_remove(i).0;c.stream_hits+=1;Some(p)})}
pub fn event_take(device:i32)->Option<*mut c_void>{CACHE.with(|cell|{let mut c=cell.borrow_mut();if !c.enabled{return None;}let i=c.events.iter().position(|(_,d)|*d==device)?;let p=c.events.swap_remove(i).0;c.event_hits+=1;Some(p)})}
pub fn stream_put(ptr:*mut c_void,device:i32)->bool{CACHE.with(|cell|{let mut c=cell.borrow_mut();if !c.enabled||!c.permit_comm{return false;}c.streams.push((ptr,device));true})}
pub fn event_put(ptr:*mut c_void,device:i32)->bool{CACHE.with(|cell|{let mut c=cell.borrow_mut();if !c.enabled||!c.permit_comm{return false;}c.events.push((ptr,device));true})}
pub fn buffer_take(bytes: usize) -> Option<*mut c_void> {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if !c.enabled { return None; }
        let i = c.buffers.iter().position(|(_, n)| *n == bytes)?;
        c.buffer_hits += 1;
        Some(c.buffers.swap_remove(i).0)
    })
}
pub fn buffer_put(ptr: *mut c_void, bytes: usize) -> bool {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if !c.enabled { return false; }
        c.buffers.push((ptr, bytes)); true
    })
}
pub fn reusable_bytes() -> u64 {
    CACHE.with(|c| {
        let c = c.borrow();
        let bytes = c.buffers.iter().map(|(_, n)| *n as u64).sum::<u64>();
        #[cfg(feature = "library-owner")]
        let bytes = bytes + c.pool.map_or(0, |(_, n)| n);
        bytes
    })
}
pub fn comm_take() -> Option<*mut c_void> {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if !c.enabled || c.comm.is_null() { return None; }
        c.comm_hits += 1;
        Some(std::mem::replace(&mut c.comm, std::ptr::null_mut()))
    })
}
pub fn comm_put(ptr: *mut c_void) -> bool {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if !c.enabled || !c.permit_comm || !c.comm.is_null() || ptr.is_null() { return false; }
        // Park and all-rank agreement happen before ownership is cached.
        // Never let just one rank reuse an old communicator.
        c.comm = ptr; true
    })
}
#[cfg(feature = "library-owner")]
pub fn pool_take(bytes: u64) -> Option<*mut c_void> {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.pool.as_ref().is_some_and(|(_, n)| *n == bytes) { c.pool_hits += 1;c.pool.take().map(|(p, _)| p) }
        else { None }
    })
}
#[cfg(target_os = "linux")]
pub fn stats() -> serde_json::Value {
    CACHE.with(|c| { let c = c.borrow();let mut parked=c.buffers.iter().map(|(_,n)|*n).collect::<Vec<_>>();parked.sort_unstable();let owned=c.enabled&&c.permit_comm&&!c.comm.is_null()&&!c.generic_manifest.is_empty()&&parked==c.generic_manifest;let geometry=c.shape.strip_prefix("generic:").and_then(|shape|shape.rsplit_once(':'));let plan=geometry.and_then(|(plan,_)|serde_json::from_str::<serde_json::Value>(plan).ok());serde_json::json!({
        "owned_raw_complete":owned,"owned_raw_plan":plan,"owned_raw_generator":geometry.map(|(_,generator)|generator),
        "buffer_hits":c.buffer_hits,"comm_hits":c.comm_hits,"pool_hits":c.pool_hits,
        "cached_buffers":c.buffers.len(),"stream_hits":c.stream_hits,"event_hits":c.event_hits,"cached_streams":c.streams.len(),"cached_events":c.events.len()
    }) })
}
#[cfg(feature = "library-owner")]
pub fn pool_put(ptr: *mut c_void) -> bool {
    use mgbfs_cuda::library_owner::*;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if !c.enabled || c.pool.is_some() { return false; }
        let mut usage = PoolUsageV1::default();
        if unsafe { mgbfs_library_pool_usage_v1(ptr, &mut usage) } != 0 || usage.live_bytes != 0 { return false; }
        c.pool = Some((ptr, usage.reserved_bytes)); true
    })
}

#[cfg(test)]
mod bootstrap_tests {
    use super::*;
    #[test]
    fn successive_queries_without_parked_comm_get_fresh_ids() {
        CACHE.with(|cache| { let mut c=cache.borrow_mut();c.enabled=true;c.nccl_id=None;assert!(c.comm.is_null()); });
        assert_eq!(bootstrap_id(|| Ok([1;128])).unwrap(),[1;128]);
        assert_eq!(bootstrap_id(|| Ok([2;128])).unwrap(),[2;128]);
        CACHE.with(|cache| { let mut c=cache.borrow_mut();c.enabled=false;c.nccl_id=None; });
    }
    #[test]
    fn factory_failure_does_not_reuse_a_destroyed_comm_id() {
        CACHE.with(|cache| { let mut c=cache.borrow_mut();c.enabled=true;c.nccl_id=Some([1;128]);assert!(c.comm.is_null()); });
        assert_eq!(bootstrap_id(|| Err("fresh id failed".into())).unwrap_err(),"fresh id failed");
        CACHE.with(|cache| { let mut c=cache.borrow_mut();c.enabled=false;c.nccl_id=None; });
    }
}

pub fn generic_buffer_put(ptr:*mut c_void,bytes:usize)->bool{let allowed=CACHE.with(|c|c.borrow().permit_comm);allowed&&buffer_put(ptr,bytes)}

pub fn release_idle_storage(){CACHE.with(|c|{let mut c=c.borrow_mut();c.release_storage();for(_,_,event,_)in c.pinned.drain(..){unsafe{cudaEventDestroy(event);}}c.shape.clear();c.generic_required=None;c.generic_manifest.clear();});}
