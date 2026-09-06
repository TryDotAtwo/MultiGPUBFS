//! Generation-bound event lifecycle; caller supplies native record/query calls.
use mgbfs_core::Result;
pub fn cuda_query_status(status: i32) -> Result<bool> {
    match status {
        0 => Ok(true),
        600 => Ok(false),
        _ => Err(format!("CUDA_EVENT_QUERY_{status}")),
    }
}

/// A timing-disabled CUDA event owned on its creating thread/context. Create
/// during setup. No raw handle escapes; every record is generation-checked.
#[cfg(feature = "cuda")]
pub struct NativeEvent {
    handle: *mut std::ffi::c_void,
    generation: EventGeneration,
}
#[cfg(feature = "cuda")]
impl NativeEvent {
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { mgbfs_cuda::ffi::cudaEventCreateWithFlags(&mut handle, 2) };
        if status != 0 {
            return Err(format!("CUDA_EVENT_CREATE_{status}"));
        }
        Ok(Self {
            handle,
            generation: EventGeneration::default(),
        })
    }
    /// # Safety
    /// Stream must be live and in this event's CUDA context. Submit all protected
    /// writes before recording. Keep their buffers live through consumer drain.
    pub unsafe fn record(&mut self, generation: u64, stream: *mut std::ffi::c_void) -> Result<()> {
        let handle = self.handle;
        self.generation.record(generation, || {
            let status = mgbfs_cuda::ffi::cudaEventRecord(handle, stream);
            if status == 0 {
                Ok(())
            } else {
                Err(format!("CUDA_EVENT_RECORD_{status}"))
            }
        })
    }
    pub fn poll(&mut self, generation: u64) -> Result<bool> {
        let handle = self.handle;
        self.generation.poll(generation, || {
            cuda_query_status(unsafe { mgbfs_cuda::native_owner::cudaEventQuery(handle) })
        })
    }
    pub fn retire(&mut self, generation: u64) -> Result<()> {
        self.generation.retire(generation)
    }
}
#[cfg(feature = "cuda")]
impl Drop for NativeEvent {
    fn drop(&mut self) {
        unsafe {
            mgbfs_cuda::ffi::cudaEventDestroy(self.handle);
        }
    }
}
#[derive(Default)]
pub struct EventGeneration {
    active: Option<u64>,
    retired: Option<u64>,
    ready: bool,
    failed: bool,
}
impl EventGeneration {
    fn apply<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.failed {
            return Err("EVENT_FAILED".into());
        }
        let result = f(self);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    /// Submit native EventRecord after this generation's protected writes.
    /// No later recording of the underlying handle may bypass this guard.
    pub fn record(&mut self, generation: u64, submit: impl FnOnce() -> Result<()>) -> Result<()> {
        self.apply(|s| {
            if s.active.is_some() || s.retired.is_some_and(|old| generation <= old) {
                return Err("EVENT_GENERATION".into());
            }
            submit()?;
            s.active = Some(generation);
            s.ready = false;
            Ok(())
        })
    }
    /// Native query maps success to true, not-ready to false, all other statuses
    /// to errors. One poll never waits for GPU work or allocates on success.
    pub fn poll(&mut self, generation: u64, query: impl FnOnce() -> Result<bool>) -> Result<bool> {
        self.apply(|s| {
            if s.active != Some(generation) {
                return Err("EVENT_GENERATION".into());
            }
            if !s.ready {
                s.ready = query()?;
            }
            Ok(s.ready)
        })
    }
    /// Completion alone does not free data: the caller must first retire all
    /// consumers/stream waits referring to this recorded generation.
    pub fn retire(&mut self, generation: u64) -> Result<()> {
        self.apply(|s| {
            if s.active != Some(generation) || !s.ready {
                return Err("EVENT_NOT_RETIRED".into());
            }
            s.retired = s.active.take();
            s.ready = false;
            Ok(())
        })
    }
}
