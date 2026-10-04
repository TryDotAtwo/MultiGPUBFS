//! Fixed pinned-slot disk queue. Exhaustion is fatal, never producer backpressure.
use crate::archive::{Archive, ArchiveRingPlan, Extent};
use mgbfs_core::Result;
use mgbfs_cuda::{
    ffi::{cudaEventCreateWithFlags, cudaEventDestroy},
    native_owner::*,
};
use std::{
    ffi::c_void,
    sync::mpsc::{self, Receiver, SyncSender},
    thread::JoinHandle,
};

pub(crate) struct Slot {
    pub ptr: *mut c_void,
    pub bytes: usize,
    pub ready: *mut c_void,
}
// Exclusive ownership moves between the GPU producer and disk worker. The
// Worker waits for the recorded D2H event before reading, returns after write.
unsafe impl Send for Slot {}
impl Slot {
    fn new(bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { cudaHostAlloc(&mut ptr, bytes, 0) };
        if status != 0 {
            return Err(format!("ARCHIVE_PIN_ALLOC_{status}"));
        }
        let mut ready = std::ptr::null_mut();
        let status = unsafe { cudaEventCreateWithFlags(&mut ready, 2) };
        if status != 0 {
            unsafe {
                cudaFreeHost(ptr);
            }
            return Err(format!("ARCHIVE_EVENT_ALLOC_{status}"));
        }
        Ok(Self { ptr, bytes, ready })
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        unsafe {
            // Also protects queued slots dropped after a disk/queue failure.
            cudaEventSynchronize(self.ready);
            cudaEventDestroy(self.ready);
            cudaFreeHost(self.ptr);
        }
    }
}
enum Message {
    Records(Slot, u64, u32),
    Layer(u64, u64),
    Complete,
}
pub struct PinnedArchive {
    tx: Option<SyncSender<Message>>,
    free: Receiver<Slot>,
    worker: Option<JoinHandle<Result<()>>>,
    pub(crate) width: usize,
    pub(crate) rows: u32,
    pinned_bytes: usize,
    device: i32,
}
impl PinnedArchive {
    /// Disk extent is physically reserved by Archive::new before worker startup.
    pub fn new<E: Extent + Send + 'static>(
        extent: E,
        disk_bytes: u64,
        width: usize,
        config_digest: [u8; 32],
        rows: u32,
        slots: usize,
    ) -> Result<Self> {
        Self::new_with_failure_report(extent, disk_bytes, width, config_digest, rows, slots, None)
    }
    pub(crate) fn new_with_failure_report<E: Extent + Send + 'static>(
        extent: E,
        disk_bytes: u64,
        width: usize,
        config_digest: [u8; 32],
        rows: u32,
        slots: usize,
        failure_report: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        let plan = ArchiveRingPlan::new(width, rows, slots)?;
        let bytes = plan.slot_bytes;
        let pinned_bytes = plan.pinned_bytes;
        // Fail disk reservation/header validation before pinning host RAM.
        let mut archive = Archive::new_run_durable(extent, disk_bytes, width, config_digest)?;
        let (free_tx, free) = mpsc::sync_channel(slots);
        for _ in 0..slots {
            free_tx
                .try_send(Slot::new(bytes)?)
                .map_err(|_| "ARCHIVE_INIT_QUEUE")?;
        }
        let mut device = 0;
        let status = unsafe { cudaGetDevice(&mut device) };
        if status != 0 {
            return Err(format!("ARCHIVE_GET_DEVICE_{status}"));
        }
        let (tx, rx) = mpsc::sync_channel(plan.descriptor_capacity);
        let worker = std::thread::Builder::new()
            .name("mgbfs-archive".into())
            .spawn(move || {
                // This thread only publishes a sideband signal. The rank's
                // dispatcher remains the sole owner of NCCL calls and abort.
                let mut failure = crate::failure::FailureReportGuard::new(failure_report);
                let on_error = |error: String| {
                    eprintln!("MGBFS_ARCHIVE_WORKER_FATAL device={device} error={error}");
                    failure.publish();
                    error
                };
                let status = unsafe { cudaSetDevice(device) };
                if status != 0 {
                    return Err(on_error(format!("ARCHIVE_SET_DEVICE_{status}")));
                }
                while let Ok(message) = rx.recv() {
                    match message {
                        Message::Records(slot, depth, count) => {
                            let status = unsafe { cudaEventSynchronize(slot.ready) };
                            if status != 0 {
                                return Err(on_error(format!("ARCHIVE_D2H_{status}")));
                            }
                            let n = count as usize * (width + 16);
                            let bytes =
                                unsafe { std::slice::from_raw_parts(slot.ptr.cast::<u8>(), n) };
                            archive
                                .records_wire(depth, u64::from(count), bytes)
                                .map_err(&on_error)?;
                            // Receiver may have been dropped following another fatal error.
                            free_tx.try_send(slot).map_err(|rejected| {
                                let error = on_error("ARCHIVE_RETURN_QUEUE".into());
                                drop(rejected);
                                error
                            })?;
                        }
                        Message::Layer(depth, count) => {
                            archive.layer_commit(depth, count).map_err(&on_error)?
                        }
                        Message::Complete => {
                            archive.run_commit().map_err(&on_error)?;
                            eprintln!("MGBFS_ARCHIVE_TIMINGS {:?}", archive.timings);
                            failure.disarm();
                            return Ok(());
                        }
                    }
                }
                Err(on_error("ARCHIVE_INCOMPLETE".into()))
            })
            .map_err(|e| format!("ARCHIVE_THREAD: {e}"))?;
        Ok(Self {
            tx: Some(tx),
            free,
            worker: Some(worker),
            width,
            rows,
            pinned_bytes,
            device,
        })
    }
    pub fn pinned_bytes(&self) -> usize {
        self.pinned_bytes
    }
    pub(crate) fn acquire(&self) -> Result<Slot> {
        let mut current = -1;
        let status = unsafe { cudaGetDevice(&mut current) };
        if status != 0 || current != self.device {
            return Err(format!("ARCHIVE_DEVICE_MISMATCH_{current}_{}", self.device));
        }
        self.free
            .try_recv()
            .map_err(|e| format!("ARCHIVE_PIN_RING_FATAL: {e}"))
    }
    pub(crate) fn submit(&self, slot: Slot, depth: u64, rows: u32) -> Result<()> {
        self.submit_notifying(slot, depth, rows, |_| {})
    }
    /// Rejected slots still own live D2H readers. Publish/abort the rank group
    /// before dropping them; the notification does not permit buffer reuse.
    pub(crate) fn submit_notifying(
        &self,
        slot: Slot,
        depth: u64,
        rows: u32,
        on_failure: impl FnOnce(&str),
    ) -> Result<()> {
        if rows == 0 || rows > self.rows || rows as usize * (self.width + 16) > slot.bytes {
            on_failure("ARCHIVE_SLOT_SHAPE");
            return Err("ARCHIVE_SLOT_SHAPE".into());
        }
        crate::archive::send_archive_message(
            self.tx.as_ref(),
            Message::Records(slot, depth, rows),
            on_failure,
        )
    }
    fn send(&self, message: Message) -> Result<()> {
        crate::archive::send_archive_message(self.tx.as_ref(), message, |_| {})
    }
    pub(crate) fn layer(&self, depth: u64, count: u64) -> Result<()> {
        self.send(Message::Layer(depth, count))
    }
    /// Call only after search exhaustion. Waiting here is durability, not BFS backpressure.
    pub fn finish(mut self) -> Result<()> {
        let sent = self.send(Message::Complete);
        self.tx.take();
        let result = self
            .worker
            .take()
            .unwrap()
            .join()
            .map_err(|_| "ARCHIVE_WORKER_PANIC")?;
        result.and(sent)
    }
}
impl Drop for PinnedArchive {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod worker_failure_tests {
    use super::*;
    use std::{
        sync::{
            atomic::{AtomicU8, Ordering},
            Arc,
        },
        time::{Duration, Instant},
    };
    struct FaultExtent {
        write_fault: bool,
        sync_fault: bool,
    }
    impl Extent for FaultExtent {
        fn reserve(&mut self, _: u64) -> std::io::Result<()> {
            Ok(())
        }
        fn write_at(&mut self, offset: u64, data: &[u8]) -> std::io::Result<usize> {
            if self.write_fault && offset >= 48 {
                Err(std::io::Error::other("INJECTED_ARCHIVE_WRITE"))
            } else {
                Ok(data.len())
            }
        }
        fn sync(&mut self) -> std::io::Result<()> {
            if self.sync_fault {
                Err(std::io::Error::other("INJECTED_ARCHIVE_SYNC"))
            } else {
                Ok(())
            }
        }
    }
    fn archive(write_fault: bool, sync_fault: bool) -> (PinnedArchive, Arc<AtomicU8>) {
        assert_eq!(unsafe { cudaSetDevice(0) }, 0);
        let report = Arc::new(AtomicU8::new(0));
        let archive = PinnedArchive::new_with_failure_report(
            FaultExtent {
                write_fault,
                sync_fault,
            },
            4096,
            4,
            [0; 32],
            1,
            2,
            Some(report.clone()),
        )
        .unwrap();
        let slot = archive.acquire().unwrap();
        unsafe {
            std::ptr::write_bytes(slot.ptr.cast::<u8>(), 0, slot.bytes);
        }
        archive.submit(slot, 0, 1).unwrap();
        (archive, report)
    }
    #[test]
    fn write_failure_is_published_without_a_producer_poll() {
        let (archive, report) = archive(true, false);
        let deadline = Instant::now() + Duration::from_secs(1);
        while report.load(Ordering::Acquire) != 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        let published = report.load(Ordering::Acquire);
        assert!(archive
            .finish()
            .unwrap_err()
            .contains("INJECTED_ARCHIVE_WRITE"));
        assert_eq!(
            published, 2,
            "worker failure was not published before finish"
        );
    }
    #[test]
    fn final_sync_failure_is_published_and_never_returns_complete() {
        let (archive, report) = archive(false, true);
        archive.layer(0, 1).unwrap();
        assert!(archive
            .finish()
            .unwrap_err()
            .contains("INJECTED_ARCHIVE_SYNC"));
        assert_eq!(report.load(Ordering::Acquire), 2);
    }
    #[test]
    fn successful_worker_does_not_publish_failure() {
        let (archive, report) = archive(false, false);
        archive.layer(0, 1).unwrap();
        archive.finish().unwrap();
        assert_eq!(report.load(Ordering::Acquire), 0);
    }
}
