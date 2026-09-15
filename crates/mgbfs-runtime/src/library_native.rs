//! Experimental Rust adapter. Not yet selected by the distributed scheduler.
//! CUDA arrays stay on device; library count synchronization remains explicit.
use crate::event_generation::NativeEvent;
use crate::library_owner::OwnerCommitGate;
use mgbfs_core::Result;
use mgbfs_cuda::{ffi::cudaStreamSynchronize, library_owner::*};
use std::{ffi::c_void, ptr};

/// Finalize one rank's disjoint shards into a preallocated SoA history buffer.
/// Output ranges are shard-major, not sorted within a shard.
/// # Safety
/// Drain every producer and borrowed-result consumer before calling. All owners
/// use stream and its current device/pool. Destination planes each hold rows
/// writable u32s and do not overlap each other or accepted storage. They may
/// alias borrowed history: all history leases must end before the first copy.
/// A failure is fatal to the rank group, not a retryable partial finalization.
pub unsafe fn finalize_shards(
    _owners: &mut [LibraryShard],
    _destination: KeysV1,
    _ranges: &mut [std::ops::Range<u32>],
    _stream: *mut c_void,
) -> Result<u32> {
    Err("LIBRARY_FINALIZE_NOT_IMPLEMENTED".into())
}

pub struct LibraryShard {
    handle: OwnerHandle,
    stream: *mut c_void,
    gate: OwnerCommitGate,
    completion: NativeEvent,
}

impl LibraryShard {
    /// # Safety
    /// Same device/pool/stream lifetime contract as new(), for two independent
    /// immutable history views. Appropriate to the inverse-closed depth-one
    /// window; callers using macro edges must provide the wider required history.
    pub unsafe fn new_window(
        previous: KeysV1,
        current: KeysV1,
        capacity: u32,
        stream: *mut c_void,
    ) -> Result<Self> {
        let completion = NativeEvent::new()?;
        let mut handle = ptr::null_mut();
        let status =
            mgbfs_library_owner_create_window_v1(previous, current, capacity, stream, &mut handle);
        if status != 0 || handle.is_null() {
            return Err(format!("LIBRARY_OWNER_CREATE_WINDOW_{status}"));
        }
        Ok(Self {
            handle,
            stream,
            gate: OwnerCommitGate::new(u64::from(capacity)),
            completion,
        })
    }

    /// # Safety
    /// Install a fixed RMM pool on the current device first. Pool, immutable
    /// history and stream must outlive this object, including teardown (history
    /// may instead be released after successful seal()). Calls
    /// are serialized on that device. A failure requires rank-group termination.
    pub unsafe fn new(history: KeysV1, capacity: u32, stream: *mut c_void) -> Result<Self> {
        let completion = NativeEvent::new()?;
        let mut handle = ptr::null_mut();
        let status = mgbfs_library_owner_create_v1(history, capacity, stream, &mut handle);
        if status != 0 || handle.is_null() {
            return Err(format!("LIBRARY_OWNER_CREATE_{status}"));
        }
        Ok(Self {
            handle,
            stream,
            gate: OwnerCommitGate::new(u64::from(capacity)),
            completion,
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

    /// End comparisons and release borrowed history indices, preserving accepted
    /// keys for finalization/export. This does not synchronize the device.
    /// # Safety
    /// All GPU work and external consumers of borrowed results/history must be
    /// drained first. Pool, creating device and stream must remain valid until
    /// close(); successful return ends only the borrowed history lifetime.
    pub unsafe fn seal(&mut self) -> Result<()> {
        self.gate.check_idle()?;
        let status = mgbfs_library_owner_seal_v1(self.handle);
        self.status(status, "SEAL")?;
        // Terminal for compare/commit, but export deliberately stays available.
        self.gate.abort();
        Ok(())
    }

    /// Record only after commit, materialization, and every consumer of borrowed
    /// result indices have been enqueued. The event is allocated during setup.
    /// # Safety
    /// All protected operations must be ordered on this owner's stream.
    pub unsafe fn record_completion(&mut self, epoch: u64) -> Result<()> {
        self.gate.check_completion(epoch)?;
        if let Err(error) = self.completion.record(epoch, self.stream) {
            self.gate.abort();
            return Err(error);
        }
        Ok(())
    }

    /// Nonblocking readiness query. Does not publish counts, even when ready:
    /// caller must inspect completed native fatal controls before publication.
    pub fn poll_completion(&mut self, epoch: u64) -> Result<bool> {
        self.gate.check_completion(epoch)?;
        match self.completion.poll(epoch) {
            Ok(ready) => Ok(ready),
            Err(error) => {
                self.gate.abort();
                Err(error)
            }
        }
    }

    /// # Safety
    /// After poll_completion returned true, validate all completed native
    /// reservation/materialization/fatal controls before calling this method.
    pub unsafe fn publish_completion(&mut self, epoch: u64) -> Result<()> {
        self.gate.check_completion(epoch)?;
        if let Err(error) = self.completion.retire(epoch) {
            self.gate.abort();
            return Err(error);
        }
        self.gate.completed(epoch)
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
    /// The creating device, stream and pool must still exist. Borrowed history
    /// must exist unless seal() succeeded; external readers must be drained.
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
