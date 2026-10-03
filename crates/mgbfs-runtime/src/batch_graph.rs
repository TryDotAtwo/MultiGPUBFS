//! One dispatcher-owned CUDA graph executor spanning three preallocated streams.
use mgbfs_core::Result;
use std::ffi::c_void;

extern "C" {
    fn mgbfs_batch_graph_create_v1(out: *mut *mut c_void) -> i32;
    fn mgbfs_batch_graph_begin_v1(handle: *mut c_void, owner: *mut c_void,
        generation: *mut c_void, exchange: *mut c_void) -> i32;
    fn mgbfs_batch_graph_submit_v1(handle: *mut c_void, batches: u32) -> i32;
    fn mgbfs_batch_graph_cancel_v1(handle: *mut c_void);
    fn mgbfs_batch_graph_stats_v1(handle: *mut c_void, launches: *mut u64,
        full_windows: *mut u64, batches: *mut u64, updates: *mut u64, rebuilds: *mut u64) -> i32;
    fn mgbfs_batch_graph_destroy_v1(handle: *mut c_void);
}

fn check(status: i32) -> Result<()> {
    if status == 0 { Ok(()) } else { Err(format!("BATCH_GRAPH_NATIVE_{status}")) }
}

pub(crate) struct BatchGraph(*mut c_void);
impl BatchGraph {
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        check(unsafe { mgbfs_batch_graph_create_v1(&mut handle) })?;
        if handle.is_null() { return Err("BATCH_GRAPH_NULL".into()); }
        Ok(Self(handle))
    }
    pub unsafe fn begin(&mut self, owner: *mut c_void, generation: *mut c_void,
                        exchange: *mut c_void) -> Result<()> {
        check(mgbfs_batch_graph_begin_v1(self.0, owner, generation, exchange))
    }
    pub fn submit(&mut self, batches: u32) -> Result<()> {
        check(unsafe { mgbfs_batch_graph_submit_v1(self.0, batches) })
    }
    pub fn cancel(&mut self) {
        unsafe { mgbfs_batch_graph_cancel_v1(self.0) };
    }
    pub fn stats(&self) -> Result<serde_json::Value> {
        let (mut launches, mut full_windows, mut batches, mut updates, mut rebuilds) = (0, 0, 0, 0, 0);
        check(unsafe { mgbfs_batch_graph_stats_v1(self.0, &mut launches,
            &mut full_windows, &mut batches, &mut updates, &mut rebuilds) })?;
        Ok(serde_json::json!({"window_batches": 32, "launches": launches,
                             "full_windows": full_windows, "batches": batches,
                             "executable_updates": updates, "executable_rebuilds": rebuilds}))
    }
}
impl Drop for BatchGraph {
    fn drop(&mut self) { unsafe { mgbfs_batch_graph_destroy_v1(self.0) }; }
}
