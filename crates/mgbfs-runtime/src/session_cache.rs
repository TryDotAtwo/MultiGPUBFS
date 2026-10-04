//! Only resident-session jobs use this rank-local, shape-bounded cache.
use std::{cell::RefCell, ffi::c_void};
use mgbfs_cuda::ffi::*;
#[derive(Default)]
struct Cache {
    enabled: bool,
    shape: String,
    buffers: Vec<(*mut c_void, usize)>,
    pinned: Vec<(*mut c_void, usize, *mut c_void)>,
    comm: *mut c_void,
    buffer_hits: u64,
    comm_hits: u64,
    pool_hits: u64,
    #[cfg(feature = "library-owner")]
    pool: Option<(*mut c_void, u64)>,
}
impl Cache {
    fn release_storage(&mut self) {
        for (ptr, _) in self.buffers.drain(..) { unsafe { cudaFree(ptr); } }
        #[cfg(feature = "library-owner")]
        if let Some((ptr, _)) = self.pool.take() {
            unsafe { mgbfs_cuda::library_owner::mgbfs_library_pool_destroy_v1(ptr); }
        }
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        self.release_storage();
        for (ptr, _, event) in self.pinned.drain(..) { unsafe { cudaEventDestroy(event);cudaFreeHost(ptr); } }
        if !self.comm.is_null() { unsafe { mgbfs_nccl_destroy(self.comm); } }
    }
}
pub fn prepare_pinned(bytes: usize, slots: usize) {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let mut keep = Vec::new();
        for (ptr, n, event) in c.pinned.drain(..) {
            if n == bytes && keep.len() < slots { keep.push((ptr,n,event)); }
            else { unsafe { cudaEventDestroy(event);cudaFreeHost(ptr); } }
        }
        c.pinned = keep;
    });
}
pub fn pinned_take(bytes: usize) -> Option<(*mut c_void,*mut c_void)> {
    CACHE.with(|c| {
        let mut c=c.borrow_mut();
        let i=c.pinned.iter().position(|(_,n,_)| *n==bytes)?;
        let (ptr,_,event)=c.pinned.swap_remove(i);Some((ptr,event))
    })
}
pub fn pinned_put(ptr: *mut c_void, bytes: usize, event: *mut c_void) -> bool {
    CACHE.with(|c| {
        let mut c=c.borrow_mut();if !c.enabled { return false; }
        c.pinned.push((ptr,bytes,event));true
    })
}
thread_local! { static CACHE: RefCell<Cache> = RefCell::new(Cache::default()); }
pub fn enable() { CACHE.with(|c| c.borrow_mut().enabled = true); }
pub fn begin(shape: String) {
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.buffer_hits = 0;c.comm_hits = 0;c.pool_hits = 0;
        if c.enabled && c.shape != shape { c.release_storage(); c.shape = shape; }
    });
}
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
        if !c.enabled || !c.comm.is_null() || ptr.is_null() { return false; }
        // Native park refuses aborted/LSA communicators and clears callbacks
        // before their Arc contexts die. One boundary sync, never per batch.
        if unsafe { mgbfs_nccl_session_park(ptr) } != 0 { return false; }
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
pub fn stats() -> serde_json::Value {
    CACHE.with(|c| { let c = c.borrow();serde_json::json!({
        "buffer_hits":c.buffer_hits,"comm_hits":c.comm_hits,"pool_hits":c.pool_hits,
        "cached_buffers":c.buffers.len()
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
