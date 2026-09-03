//! Fixed pinned-slot disk queue. Exhaustion is fatal, never producer backpressure.
use crate::archive::{Archive, Extent};
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
        if rows == 0 || slots < 2 {
            return Err("ARCHIVE_RING_SHAPE".into());
        }
        let bytes = width
            .checked_add(16)
            .and_then(|v| v.checked_mul(rows as usize))
            .ok_or("ARCHIVE_PIN_OVERFLOW")?;
        let pinned_bytes = bytes.checked_mul(slots).ok_or("ARCHIVE_PIN_OVERFLOW")?;
        let (free_tx, free) = mpsc::sync_channel(slots);
        for _ in 0..slots {
            free_tx
                .try_send(Slot::new(bytes)?)
                .map_err(|_| "ARCHIVE_INIT_QUEUE")?;
        }
        let mut archive = Archive::new(extent, disk_bytes, width, config_digest)?;
        let mut device = 0;
        let status = unsafe { cudaGetDevice(&mut device) };
        if status != 0 {
            return Err(format!("ARCHIVE_GET_DEVICE_{status}"));
        }
        let (tx, rx) = mpsc::sync_channel(
            slots
                .checked_mul(2)
                .and_then(|n| n.checked_add(2))
                .ok_or("ARCHIVE_QUEUE_OVERFLOW")?,
        );
        let worker = std::thread::Builder::new()
            .name("mgbfs-archive".into())
            .spawn(move || {
                let status = unsafe { cudaSetDevice(device) };
                if status != 0 {
                    return Err(format!("ARCHIVE_SET_DEVICE_{status}"));
                }
                while let Ok(message) = rx.recv() {
                    match message {
                        Message::Records(slot, depth, count) => {
                            let status = unsafe { cudaEventSynchronize(slot.ready) };
                            if status != 0 {
                                return Err(format!("ARCHIVE_D2H_{status}"));
                            }
                            let n = count as usize * (width + 16);
                            let bytes =
                                unsafe { std::slice::from_raw_parts(slot.ptr.cast::<u8>(), n) };
                            archive.records_wire(depth, u64::from(count), bytes)?;
                            // Receiver may have been dropped following another fatal error.
                            free_tx.try_send(slot).map_err(|_| "ARCHIVE_RETURN_QUEUE")?;
                        }
                        Message::Layer(depth, count) => archive.layer_commit(depth, count)?,
                        Message::Complete => {
                            archive.run_commit()?;
                            return Ok(());
                        }
                    }
                }
                Err("ARCHIVE_INCOMPLETE".into())
            })
            .map_err(|e| format!("ARCHIVE_THREAD: {e}"))?;
        Ok(Self {
            tx: Some(tx),
            free,
            worker: Some(worker),
            width,
            rows,
            pinned_bytes,
        })
    }
    pub fn pinned_bytes(&self) -> usize {
        self.pinned_bytes
    }
    pub(crate) fn acquire(&self) -> Result<Slot> {
        self.free
            .try_recv()
            .map_err(|e| format!("ARCHIVE_PIN_RING_FATAL: {e}"))
    }
    pub(crate) fn submit(&self, slot: Slot, depth: u64, rows: u32) -> Result<()> {
        if rows == 0 || rows > self.rows || rows as usize * (self.width + 16) > slot.bytes {
            return Err("ARCHIVE_SLOT_SHAPE".into());
        }
        self.send(Message::Records(slot, depth, rows))
    }
    fn send(&self, message: Message) -> Result<()> {
        self.tx
            .as_ref()
            .ok_or("ARCHIVE_CLOSED")?
            .try_send(message)
            .map_err(|e| format!("ARCHIVE_DESCRIPTOR_RING_FATAL: {e}"))
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
