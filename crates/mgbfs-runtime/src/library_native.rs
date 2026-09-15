//! Experimental Rust adapter. Not yet selected by the distributed scheduler.
//! CUDA arrays stay on device; library count synchronization remains explicit.
use crate::library_owner::OwnerCommitGate;
use mgbfs_core::Result;
use mgbfs_cuda::{ffi::cudaStreamSynchronize, library_owner::*};
use std::{ffi::c_void, ptr};

pub struct LibraryShard {
    handle: OwnerHandle,
    stream: *mut c_void,
    gate: OwnerCommitGate,
}

impl LibraryShard {
    /// # Safety
    /// Install a fixed RMM pool on the current device first. Pool, immutable
    /// history and stream must outlive this object, including teardown. Calls
    /// are serialized on that device. A failure requires rank-group termination.
    pub unsafe fn new(history: KeysV1, capacity: u32, stream: *mut c_void) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let status = mgbfs_library_owner_create_v1(history, capacity, stream, &mut handle);
        if status != 0 || handle.is_null() {
            return Err(format!("LIBRARY_OWNER_CREATE_{status}"));
        }
        Ok(Self {
            handle,
            stream,
            gate: OwnerCommitGate::new(u64::from(capacity)),
        })
    }

    fn status(&mut self, status: i32, operation: &str) -> Result<()> {
        if status != 0 {
            self.gate.abort();
            return Err(format!("LIBRARY_OWNER_{operation}_{status}"));
        }
        Ok(())
    }

    /// # Safety
    /// Input device buffers must be valid and ready on this stream, and retained
    /// through comparison. Borrowed result indices must be consumed before the
    /// next comparison. No next comparison is allowed before complete().
    pub unsafe fn compare(&mut self, epoch: u64, input: CandidatesV1) -> Result<SurvivorsV1> {
        self.gate.check_compare(epoch)?;
        let mut result = SurvivorsV1 {
            source_indices: ptr::null(),
            epoch: 0,
            rows: 0,
            reserved: 0,
        };
        let status = mgbfs_library_owner_compare_v1(self.handle, epoch, input, &mut result);
        self.status(status, "COMPARE")?;
        if result.epoch != epoch
            || result.reserved != 0
            || (result.rows != 0 && result.source_indices.is_null())
        {
            self.gate.abort();
            return Err("LIBRARY_OWNER_RESULT".into());
        }
        self.gate
            .compared(epoch, u64::from(input.keys.rows), u64::from(result.rows))?;
        Ok(result)
    }

    /// # Safety
    /// granted is the successful native StateRing reservation result, including
    /// required archive/materialization credits. Never pass requested capacity.
    /// Enqueues key writes only; caller next enqueues materialization on stream.
    pub unsafe fn commit(&mut self, epoch: u64, granted: u32) -> Result<()> {
        self.gate.reserve(epoch, u64::from(granted))?;
        let status = mgbfs_library_owner_commit_v1(self.handle, epoch, granted);
        self.status(status, "COMMIT")
    }

    /// Explicit synchronization for the first integration backend, included in
    /// search time. Not an overlap claim. Caller must enqueue all result readers
    /// and state publication before calling this method.
    /// # Safety
    /// All native error controls must have been checked; enqueue work only on
    /// this owner's stream or join its dependencies before this completion.
    pub unsafe fn complete(&mut self, epoch: u64) -> Result<()> {
        let status = cudaStreamSynchronize(self.stream);
        self.status(status, "COMPLETE")?;
        self.gate.completed(epoch)
    }

    pub fn accepted(&self) -> u64 {
        self.gate.accepted()
    }

    /// # Safety
    /// Borrowed keys require owner lifetime and stream/event ordering. Export
    /// is append-order, not bucket-sorted; finalization must convert/partition it.
    pub unsafe fn export(&mut self) -> Result<KeysV1> {
        let mut keys = KeysV1 {
            words: [ptr::null(); 4],
            rows: 0,
            reserved: 0,
        };
        let status = mgbfs_library_owner_export_v1(self.handle, &mut keys);
        self.status(status, "EXPORT")?;
        Ok(keys)
    }

    /// Explicit teardown, outside the batch loop. On CUDA failure retain the
    /// handle; process termination owns cleanup, never free live GPU readers.
    /// # Safety
    /// The creating device, stream, pool and borrowed history must still exist.
    pub unsafe fn close(&mut self) -> Result<()> {
        if self.handle.is_null() {
            return Ok(());
        }
        let status = cudaStreamSynchronize(self.stream);
        self.status(status, "CLOSE")?;
        mgbfs_library_owner_destroy_v1(self.handle);
        self.handle = ptr::null_mut();
        self.gate.abort();
        Ok(())
    }
}

impl Drop for LibraryShard {
    fn drop(&mut self) {
        // Explicit close surfaces errors. On failed device synchronization this
        // deliberately leaks until rank exit rather than freeing in-flight data.
        unsafe {
            let _ = self.close();
        }
    }
}
