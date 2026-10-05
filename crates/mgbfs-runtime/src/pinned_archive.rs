//! Bounded pinned-slot disk queue. Waiting for disk credits is explicitly opt-in.
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
        if let Some((ptr, ready)) = crate::session_cache::pinned_take(bytes) {
            return Ok(Self {ptr,bytes,ready});
        }
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
            if cudaEventSynchronize(self.ready) == 0 &&
                crate::session_cache::pinned_put(self.ptr,self.bytes,self.ready) { return; }
            cudaEventDestroy(self.ready);
            cudaFreeHost(self.ptr);
        }
    }
}
enum Message {
    Records(Slot, u64, u32),
    Layer(u64, u64),
    Complete,
    ReplaceLast,
}
pub struct PinnedArchive {
    tx: Option<SyncSender<Message>>,
    free: Receiver<Slot>,
    worker: Option<JoinHandle<Result<()>>>,
    pub(crate) width: usize,
    pub(crate) rows: u32,
    pinned_bytes: usize,
    slots: usize,
    device: i32,
    wait_for_credit: bool,
    pub(crate) selected: bool,
    pub(crate) state_only: bool,
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
        extent: E, disk_bytes: u64, width: usize, config_digest: [u8; 32],
        rows: u32, slots: usize,
        failure_report: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        let selected = std::env::var("MGBFS_ARCHIVE_SELECTION").as_deref() == Ok("last_complete_small_1000");
        let state_only = selected || std::env::var("MGBFS_ARCHIVE_SELECTION").as_deref() == Ok("all_states");
        let mut plan = ArchiveRingPlan::new(width, rows, slots)?;
        if state_only {
            let raw = width.checked_mul(rows as usize).ok_or("ARCHIVE_PIN_OVERFLOW")?;
            // Stable geometry across graph degrees lets resident sessions
            // reuse the same host allocation instead of pinning every pair.
            plan.slot_bytes = raw.checked_add((65536 - raw % 65536) % 65536)
                .ok_or("ARCHIVE_PIN_OVERFLOW")?;
            plan.pinned_bytes = plan.slot_bytes.checked_mul(slots).ok_or("ARCHIVE_PIN_OVERFLOW")?;
        }
        let bytes = plan.slot_bytes;
        crate::session_cache::prepare_pinned(bytes, slots);
        let pinned_bytes = plan.pinned_bytes;
        // Fail disk reservation/header validation before pinning host RAM.
        let mut archive = if selected {
            Archive::new_selected(extent, disk_bytes, width, config_digest)?
        } else if state_only { Archive::new_state_only(extent, disk_bytes, width, config_digest)? } else { Archive::new_run_durable(extent, disk_bytes, width, config_digest)? };
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
                            let n = count as usize * (width + if state_only { 0 } else { 16 });
                            let bytes =
                                unsafe { std::slice::from_raw_parts(slot.ptr.cast::<u8>(), n) };
                            archive.records_wire(depth, u64::from(count), bytes)
                                .map_err(&on_error)?;
                            // Receiver may have been dropped following another fatal error.
                            free_tx.try_send(slot).map_err(|rejected| {
                                let error = on_error("ARCHIVE_RETURN_QUEUE".into());
                                drop(rejected);
                                error
                            })?;
                        }
                        Message::Layer(depth, count) => archive.layer_commit(depth, count)
                            .map_err(&on_error)?,
                        Message::ReplaceLast => archive.replace_last_layer().map_err(&on_error)?,
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
            slots,
            selected,
            state_only,
            device,
            wait_for_credit: std::env::var("MGBFS_ARCHIVE_CREDIT_MODE").as_deref() == Ok("wait"),
        })
    }
    pub fn pinned_bytes(&self) -> usize {
        self.pinned_bytes
    }
    pub(crate) fn credit_capacity(&self) -> usize {
        self.slots
    }
    pub(crate) fn acquire(&self) -> Result<Slot> {
        self.acquire_cancellable(|| false)
    }
    pub(crate) fn acquire_cancellable(&self,cancelled: impl Fn() -> bool) -> Result<Slot> {
        let mut current = -1;
        let status = unsafe { cudaGetDevice(&mut current) };
        if status != 0 || current != self.device {
            return Err(format!("ARCHIVE_DEVICE_MISMATCH_{current}_{}", self.device));
        }
        if self.wait_for_credit {
            crate::archive::recv_archive_credit(&self.free,cancelled)
        } else {
            self.free.try_recv().map_err(|e|format!("ARCHIVE_PIN_RING_FATAL: {e}"))
        }
    }
    pub(crate) fn submit(&self, slot: Slot, depth: u64, rows: u32) -> Result<()> {
        self.submit_notifying(slot, depth, rows, |_| {})
    }
    /// Rejected slots still own live D2H readers. Publish/abort the rank group
    /// before dropping them; the notification does not permit buffer reuse.
    pub(crate) fn submit_notifying(
        &self, slot: Slot, depth: u64, rows: u32, on_failure: impl FnOnce(&str),
    ) -> Result<()> {
        if rows == 0 || rows > self.rows || rows as usize * (self.width + if self.state_only { 0 } else { 16 }) > slot.bytes {
            on_failure("ARCHIVE_SLOT_SHAPE");
            return Err("ARCHIVE_SLOT_SHAPE".into());
        }
        crate::archive::send_archive_message(
            self.tx.as_ref(), Message::Records(slot, depth, rows), on_failure,
        )
    }
    fn send(&self, message: Message) -> Result<()> {
        crate::archive::send_archive_message(self.tx.as_ref(), message, |_| {})
    }
    pub(crate) fn layer(&self, depth: u64, count: u64) -> Result<()> {
        self.send(Message::Layer(depth, count))
    }
    pub(crate) fn replace_last_layer(&self) -> Result<()> {
        self.send(Message::ReplaceLast)
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
    use std::{sync::{Arc, atomic::{AtomicU8, Ordering}}, time::{Duration, Instant}};
    struct FaultExtent { write_fault: bool, sync_fault: bool }
    impl Extent for FaultExtent {
        fn reserve(&mut self, _: u64) -> std::io::Result<()> { Ok(()) }
        fn write_at(&mut self, offset: u64, data: &[u8]) -> std::io::Result<usize> {
            if self.write_fault && offset >= 48 {
                Err(std::io::Error::other("INJECTED_ARCHIVE_WRITE"))
            } else { Ok(data.len()) }
        }
        fn sync(&mut self) -> std::io::Result<()> {
            if self.sync_fault { Err(std::io::Error::other("INJECTED_ARCHIVE_SYNC")) }
            else { Ok(()) }
        }
    }
    fn archive(write_fault: bool, sync_fault: bool) -> (PinnedArchive, Arc<AtomicU8>) {
        assert_eq!(unsafe { cudaSetDevice(0) }, 0);
        let report = Arc::new(AtomicU8::new(0));
        let archive = PinnedArchive::new_with_failure_report(
            FaultExtent { write_fault, sync_fault }, 4096, 4, [0; 32], 1, 2,
            Some(report.clone()),
        ).unwrap();
        let slot = archive.acquire().unwrap();
        unsafe { std::ptr::write_bytes(slot.ptr.cast::<u8>(), 0, slot.bytes); }
        archive.submit(slot, 0, 1).unwrap();
        (archive, report)
    }
    #[test]
    fn fatal_credit_policy_does_not_wait_on_exhaustion() {
        assert_eq!(unsafe {cudaSetDevice(0)},0);
        let mut archive=PinnedArchive::new(FaultExtent {write_fault:false,sync_fault:false},4096,4,[0;32],1,2).unwrap();
        archive.wait_for_credit=false;
        let first=archive.acquire().unwrap();
        let second=archive.acquire().unwrap();
        let error=archive.acquire().err().unwrap();
        assert!(error.starts_with("ARCHIVE_PIN_RING_FATAL:"));
        archive.layer(0,0).unwrap();
        archive.finish().unwrap();
        drop(first);drop(second);
    }
    #[test]
    fn occupied_pinned_slots_wait_for_writer_without_losing_readers() {
        struct PausedExtent {
            entered: Option<std::sync::mpsc::Sender<()>>,
            release: std::sync::mpsc::Receiver<()>,
        }
        impl Extent for PausedExtent {
            fn reserve(&mut self,_:u64)->std::io::Result<()> { Ok(()) }
            fn write_at(&mut self,offset:u64,data:&[u8])->std::io::Result<usize> {
                if offset>=48 {
                    if let Some(entered)=self.entered.take() {
                        entered.send(()).unwrap();self.release.recv().unwrap();
                    }
                }
                Ok(data.len())
            }
            fn sync(&mut self)->std::io::Result<()> { Ok(()) }
        }
        assert_eq!(unsafe {cudaSetDevice(0)},0);
        let (notify,entered)=std::sync::mpsc::channel();
        let (release,resume)=std::sync::mpsc::channel();
        let mut archive=PinnedArchive::new(PausedExtent {
            entered:Some(notify),release:resume},4096,4,[0;32],1,2).unwrap();
        archive.wait_for_credit=true;
        let first=archive.acquire().unwrap();
        unsafe {std::ptr::write_bytes(first.ptr.cast::<u8>(),0,first.bytes);}
        archive.submit(first,0,1).unwrap();entered.recv().unwrap();
        let held=archive.acquire().unwrap();
        let writer=std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));release.send(()).unwrap();
        });
        let recycled=archive.acquire().unwrap();writer.join().unwrap();
        archive.layer(0,1).unwrap();archive.finish().unwrap();
        drop(held);drop(recycled);
    }
    #[test]
    fn write_failure_is_published_without_a_producer_poll() {
        let (archive, report) = archive(true, false);
        let deadline = Instant::now() + Duration::from_secs(1);
        while report.load(Ordering::Acquire) != 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        let published = report.load(Ordering::Acquire);
        assert!(archive.finish().unwrap_err().contains("INJECTED_ARCHIVE_WRITE"));
        assert_eq!(published, 2, "worker failure was not published before finish");
    }
    #[test]
    fn final_sync_failure_is_published_and_never_returns_complete() {
        let (archive, report) = archive(false, true);
        archive.layer(0, 1).unwrap();
        assert!(archive.finish().unwrap_err().contains("INJECTED_ARCHIVE_SYNC"));
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
