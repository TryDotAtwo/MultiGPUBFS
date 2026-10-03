//! Native 1/2/4/8-rank NCCL BFS reference. Torchrun supplies only rank env.
use crate::event_generation::NativeEvent;
use crate::failure::{check_native_status as check, process_owner_pair, vote_group_error, OwnerFailurePolicy};
use crate::jobs::{split, JobSpan};
#[cfg(feature = "library-owner")]
use crate::library_native::{finalize_shards, ControlTransfer, LibraryShard};
use crate::parent_batches::{ParentBatch, ParentCursor};
use mgbfs_core::{
    config::{OwnerBackend, ReferenceOwner},
    hash::GemmHash,
    matrix::{encode_permutation_matrix, MatrixGroup},
    Result,
};
#[cfg(feature = "library-owner")]
use mgbfs_cuda::library_owner::*;
use mgbfs_cuda::{ffi::*, native_owner::*};
use std::ffi::{c_void, CStr};

#[cfg(debug_assertions)]
extern "C" {
    fn cudaStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
    fn cudaGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64) -> i32;
    fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
}

// Acceptance probe of the actual rank-owner method, not a drained leaf stub.
// Opt-in debug builds only; production never allocates a graph per batch.
#[cfg(debug_assertions)]
struct OwnerCaptureProbe {
    stream: *mut c_void,
    capturing: bool,
    graph: *mut c_void,
    executable: *mut c_void,
}
#[cfg(debug_assertions)]
impl OwnerCaptureProbe {
    fn begin(stream: *mut c_void) -> Result<Option<Self>> {
        if std::env::var_os("MGBFS_TEST_OWNER_DAG_CAPTURE").is_none() { return Ok(None); }
        check(unsafe { cudaStreamBeginCapture(stream, 1) })?;
        Ok(Some(Self { stream, capturing: true,
            graph: std::ptr::null_mut(), executable: std::ptr::null_mut() }))
    }
    fn launch(mut self) -> Result<()> {
        let status = unsafe { cudaStreamEndCapture(self.stream, &mut self.graph) };
        self.capturing = false;
        check(status)?;
        check(unsafe { cudaGraphInstantiateWithFlags(&mut self.executable, self.graph, 0) })?;
        check(unsafe { cudaGraphLaunch(self.executable, self.stream) })?;
        eprintln!("MGBFS_OWNER_DAG_CAPTURE launched");
        Ok(())
    }
}
#[cfg(debug_assertions)]
impl Drop for OwnerCaptureProbe {
    fn drop(&mut self) {
        unsafe {
            if self.capturing { cudaStreamEndCapture(self.stream, &mut self.graph); }
            if !self.executable.is_null() { cudaGraphExecDestroy(self.executable); }
            if !self.graph.is_null() { cudaGraphDestroy(self.graph); }
        }
    }
}

// Thread-local NVTX push/pop ownership survives early returns and panics.
// Diagnostic ranges never add CUDA waits or device reads.
struct TraceRange(bool);
impl TraceRange {
    fn new(enabled: bool, label: &'static [u8]) -> Self {
        if enabled { unsafe { mgbfs_trace_range_push(label.as_ptr().cast()) }; }
        Self(enabled)
    }
}
impl Drop for TraceRange {
    fn drop(&mut self) {
        if self.0 { unsafe { mgbfs_trace_range_pop() }; }
    }
}

extern "C" fn nccl_cancel_probe(context: *mut c_void) -> i32 {
    if context.is_null() { return 1; }
    let flag = unsafe { &*context.cast::<std::sync::atomic::AtomicBool>() };
    i32::from(flag.load(std::sync::atomic::Ordering::Acquire))
}
extern "C" fn nccl_retirement_probe(context: *mut c_void, publish: i32) -> i32 {
    if context.is_null() { return -1; }
    let state = unsafe { &*context.cast::<crate::bootstrap::SearchRetirement>() };
    if publish < 0 { state.failed.store(true, std::sync::atomic::Ordering::Release); }
    if publish > 0 { state.local.store(true, std::sync::atomic::Ordering::Release); }
    if state.failed.load(std::sync::atomic::Ordering::Acquire) { -1 }
    else { i32::from(state.group.load(std::sync::atomic::Ordering::Acquire)) }
}

#[cfg(debug_assertions)]
thread_local! {
    static TEST_OWNER_HOST_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(debug_assertions)]
pub fn inject_owner_host_error_once_for_test() {
    TEST_OWNER_HOST_FAULT.with(|flag| flag.set(true));
}

#[cfg(feature = "library-owner")]
struct LibraryOwnerStorage {
    control_transfer: ControlTransfer,
    shards: Vec<LibraryShard>,
    scratch: Buffer,
    previous: Vec<std::ops::Range<u32>>,
    current: Vec<std::ops::Range<u32>>,
    capacity: u32,
    plane_words: usize,
    epoch: u64,
    closed: bool,
    pool: PoolHandle,
    cuco_workspace: WorkspaceHandle,
    rank: RankHandle,
    rank_mode: bool,
    rank_accepted: *const u32,
    logical_owner: u32,
}
#[cfg(feature = "library-owner")]
impl Drop for LibraryOwnerStorage {
    fn drop(&mut self) {
        // This field is dropped BEFORE streams/history. On failed drain leave
        // pool destruction to rank-process exit rather than free GPU readers.
        let mut drained = true;
        if !self.rank.is_null() {
            drained &= unsafe { mgbfs_library_rank_destroy_v1(self.rank) == 0 };
            if drained {
                self.rank = std::ptr::null_mut();
            }
        }
        for shard in &mut self.shards {
            drained &= unsafe { shard.close().is_ok() };
        }
        if drained {
            unsafe {
                if !self.cuco_workspace.is_null() {
                    drained = mgbfs_library_cuco_workspace_destroy_v1(self.cuco_workspace) == 0;
                }
                if drained {
                    mgbfs_library_pool_destroy_v1(self.pool);
                }
            }
        }
    }
}
#[cfg(feature = "library-owner")]
unsafe fn create_rank_owner(
    library: &LibraryOwnerStorage,
    previous: &Buffer,
    current: &Buffer,
    shards: u32,
    incoming: u32,
    world: u32,
    stream: *mut c_void,
) -> Result<RankHandle> {
    let old: Vec<_> = library.previous.iter()
        .map(|range| history_view(previous, library.plane_words, range)).collect();
    let now: Vec<_> = library.current.iter()
        .map(|range| history_view(current, library.plane_words, range)).collect();
    let capacities = vec![library.capacity; shards as usize];
    let mut rank = std::ptr::null_mut();
    check(mgbfs_library_rank_create_cuco_v1(
        old.as_ptr(), now.as_ptr(), capacities.as_ptr(), shards, incoming,
        library.logical_owner, world, stream, &mut rank,
    ))?;
    if rank.is_null() {
        return Err("LIBRARY_RANK_CREATE_NULL".into());
    }
    Ok(rank)
}
#[cfg(feature = "library-owner")]
unsafe fn finalize_rank_owner(
    library: &mut LibraryOwnerStorage,
    destination: KeysV1,
    stream: *mut c_void,
) -> Result<u32> {
    if library.rank.is_null() || destination.rows == 0 {
        return Err("LIBRARY_RANK_FINALIZE_SHAPE".into());
    }
    // FinalizeDepth drains every candidate/materialization reader before
    // releasing the history sets and reusing either SoA history buffer.
    check(cudaStreamSynchronize(stream))?;
    let mut counts = vec![0u32; library.previous.len()];
    if !library.rank_accepted.is_null() {
        check(cudaMemcpy(counts.as_mut_ptr().cast(), library.rank_accepted.cast(),
            counts.len() * 4, 2))?;
    }
    let total = counts.iter().try_fold(0u32, |sum, &n| sum.checked_add(n))
        .ok_or("LIBRARY_RANK_FINALIZE_OVERFLOW")?;
    if total > destination.rows {
        return Err("LIBRARY_RANK_FINALIZE_CAPACITY".into());
    }
    check(mgbfs_library_rank_seal_v1(library.rank))?;
    let mut offset = 0u32;
    for (shard, &rows) in counts.iter().enumerate() {
        let mut keys = KeysV1 { words: [std::ptr::null(); 4], rows: 0, reserved: 0 };
        check(mgbfs_library_rank_export_shard_v1(
            library.rank, shard as u32, rows, &mut keys,
        ))?;
        if keys.rows != rows {
            return Err("LIBRARY_RANK_EXPORT_COUNT".into());
        }
        for plane in 0..4 {
            if rows != 0 {
                check(cudaMemcpyAsync(
                    destination.words[plane].cast_mut().add(offset as usize).cast(),
                    keys.words[plane].cast(), rows as usize * 4, 3, stream,
                ))?;
            }
        }
        library.previous[shard] = offset..offset + rows;
        offset += rows;
    }
    check(cudaStreamSynchronize(stream))?;
    check(mgbfs_library_rank_destroy_v1(library.rank))?;
    library.rank = std::ptr::null_mut();
    library.rank_accepted = std::ptr::null();
    Ok(total)
}
#[cfg(feature = "library-owner")]
unsafe fn history_view(
    buffer: &Buffer,
    plane_words: usize,
    range: &std::ops::Range<u32>,
) -> KeysV1 {
    KeysV1 {
        words: std::array::from_fn(|p| {
            buffer
                .ptr
                .cast::<u32>()
                .add(p * plane_words + range.start as usize)
                .cast_const()
        }),
        rows: range.end - range.start,
        reserved: 0,
    }
}

#[derive(Clone)]
pub struct DistributedConfig {
    pub epoch_window: usize,
    pub rank: u32,
    pub world: u32,
    pub logical_owner_to_rank: Vec<u32>,
    pub transport: mgbfs_core::config::ReferenceTransport,
    pub batch: u32,
    pub layer_capacity: u32,
    pub state_ring_capacity: u32,
    pub state_descriptor_capacity: u32,
    pub buckets: u32,
    pub shards: u32,
    pub job_buckets: u32,
    pub bucket_capacity: u32,
    pub prededup: bool,
    pub generation_variant: u32,
    pub untouched_vram_reserve: u64,
}
fn wait_nccl_stream(
    comm: *mut c_void,
    stream: *mut c_void,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err("REMOTE_SEARCH_CANCELLED".into());
        }
        match unsafe { cudaStreamQuery(stream) } {
            0 => return Ok(()),
            600 => {
                match unsafe { mgbfs_nccl_poll(comm) } {
                    0 | 4 => {},
                    _ => return Err("NCCL_ASYNC_FAILURE".into()),
                }
                if std::time::Instant::now() >= deadline {
                    return Err("NCCL_STREAM_TIMEOUT".into());
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            code => return Err(format!("CUDA_STREAM_QUERY_{code}")),
        }
    }
}
unsafe fn rank_directory(
    world: u32,
    keys: *const c_void,
    n: *const u32,
    cap: u32,
    b: u32,
    owner: u32,
    out: *mut Range,
    f: *mut u32,
    stream: *mut c_void,
) -> i32 {
    if world == 1 {
        mgbfs_bucket_directory(keys, n, cap, b, out, f, stream)
    } else {
        mgbfs_owner_bucket_directory_n(keys, n, cap, b, owner, world, out, f, stream)
    }
}
struct Buffer {
    ptr: *mut c_void,
    bytes: usize,
    stream: *mut c_void,
}
#[derive(Clone, Copy)]
struct LsaView {
    count: *const u32,
    fatal: *const u32,
    hashes: *const c_void,
    states: *const u8,
    // Borrowed from Comm's preallocated mapped terminal word. Comm outlives
    // this view; both GPU and host use system/Acquire atomics, never memcpy.
    terminal: *const std::sync::atomic::AtomicU32,
}
// Failure-only sideband subscription. Never reads GPU data or calls CUDA/NCCL.
// Keep it before Comm in field drop order: joining closes the host writer lease
// before CUDA frees the borrowed mapped word. The worker owns its token Arc.
struct CancelMirror {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl CancelMirror {
    unsafe fn new(token: std::sync::Arc<std::sync::atomic::AtomicBool>, word: *mut u32) -> Result<Self> {
        if word.is_null() || word as usize % std::mem::align_of::<std::sync::atomic::AtomicU32>() != 0 {
            return Err("LSA_CANCEL_WORD_ALIGNMENT".into());
        }
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_stop = stop.clone();
        let address = word as usize;
        let worker = std::thread::Builder::new().name("mgbfs-cancel-mirror".into())
            .spawn(move || {
                while !worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                    if token.load(std::sync::atomic::Ordering::Acquire) {
                        let signal = &*(address as *const std::sync::atomic::AtomicU32);
                        signal.store(1, std::sync::atomic::Ordering::Release);
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }).map_err(|e| format!("LSA_CANCEL_MIRROR_THREAD: {e}"))?;
        Ok(Self { stop, worker: Some(worker) })
    }
}
impl Drop for CancelMirror {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
    }
}
fn admit_device_group(
    comm: *mut c_void,
    stream: *mut c_void,
    required: u64,
    reserve: u64,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    let send = Buffer::new(4, stream)?;
    let recv = Buffer::new(4, stream)?;
    let vote = |value: u32| -> Result<u32> {
        send.put_u32(value)?;
        check(unsafe {
            mgbfs_nccl_all_reduce_max_u32(comm, send.ptr.cast(), recv.ptr.cast(), stream)
        })?;
        wait_nccl_stream(comm, stream, cancelled)?;
        recv.one()
    };
    vote(0)?; // Initialize the actual collective before querying free VRAM.
    let (mut free, mut total) = (0usize, 0usize);
    let local = check(unsafe { cudaMemGetInfo(&mut free, &mut total) })
        .and_then(|_| crate::distributed_memory::device_admission(required, reserve, free as u64));
    if vote(u32::from(local.is_err()))? != 0 {
        return Err(format!(
            "VRAM_PREFLIGHT_GROUP: {}",
            local
                .err()
                .unwrap_or_else(|| "peer rejected admission".into())
        ));
    }
    Ok(())
}
fn setup_failure_vote(comm: *mut c_void, stream: *mut c_void,
                      send: &Buffer, recv: &Buffer, failed: bool,
                      cancelled: Option<&std::sync::atomic::AtomicBool>) -> Result<bool> {
    send.put_u32(u32::from(failed))?;
    check(unsafe { mgbfs_nccl_all_reduce_max_u32(comm, send.ptr.cast(), recv.ptr.cast(), stream) })?;
    wait_nccl_stream(comm, stream, cancelled)?;
    Ok(recv.one::<u32>()? != 0)
}
impl Buffer {
    fn new(bytes: usize, stream: *mut c_void) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut ptr, bytes.max(1)) })?;
        let x = Self { ptr, bytes, stream };
        check(unsafe { cudaMemsetAsync(ptr, 0, bytes.max(1), stream) })?;
        Ok(x)
    }
    fn put<T: Copy>(&self, x: &[T]) -> Result<()> {
        if std::mem::size_of_val(x) > self.bytes {
            return Err("UPLOAD_CAPACITY".into());
        }
        check(unsafe {
            cudaMemcpyAsync(
                self.ptr,
                x.as_ptr().cast(),
                std::mem::size_of_val(x),
                1,
                self.stream,
            )
        })?;
        check(unsafe { cudaStreamSynchronize(self.stream) })
    }
    fn put_u32(&self, value: u32) -> Result<()> {
        if self.bytes < std::mem::size_of::<u32>() {
            return Err("UPLOAD_CAPACITY".into());
        }
        // The scalar is a launch argument, so no host slice needs to remain
        // alive while the stream consumes it. Its GPU consumers stay ordered.
        check(unsafe { mgbfs_device_store_u32(self.ptr.cast(), value, self.stream) })
    }
    fn read<T: Copy>(&self, x: &mut [T]) -> Result<()> {
        if std::mem::size_of_val(x) > self.bytes {
            return Err("READ_CAPACITY".into());
        }
        check(unsafe { cudaMemcpy(x.as_mut_ptr().cast(), self.ptr, std::mem::size_of_val(x), 2) })
    }
    fn one<T: Copy + Default>(&self) -> Result<T> {
        let mut x = [T::default()];
        self.read(&mut x)?;
        Ok(x[0])
    }
    unsafe fn at(&self, n: usize) -> *mut c_void {
        self.ptr.cast::<u8>().add(n).cast()
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}
struct Plan(*mut c_void, unsafe extern "C" fn(*mut c_void));
struct BoundedOwnerStorage {
    plan: Plan,
    accepted: Buffer,
    lengths: Buffer,
    counts: Buffer,
    selected: Buffer,
}
struct NativeRankStorage {
    previous: Buffer,
    current: Buffer,
    survivors: Buffer,
    accepted: Buffer,
    capacities: Buffer,
    offsets: Buffer,
}
impl Plan {
    fn new(
        drop: unsafe extern "C" fn(*mut c_void),
        create: impl FnOnce(*mut *mut c_void, *mut i8) -> i32,
    ) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        let mut e = [0i8; 512];
        let status = create(&mut p, e.as_mut_ptr());
        if status != 0 {
            let message = unsafe { CStr::from_ptr(e.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(if message.is_empty() {
                format!("NATIVE_PLAN_CREATE_FAILED status={status}")
            } else { message });
        }
        Ok(Self(p, drop))
    }
}
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe { self.1(self.0) }
    }
}
#[cfg(test)]
mod plan_error_tests {
    use super::*;
    #[test]
    fn invalid_epoch_window_is_rejected_before_device_or_communicator_admission() {
        let graph = MatrixGroup::unitriangular(3, 2).unwrap();
        for epoch_window in [0, 1] {
            let cfg = DistributedConfig {
                epoch_window,
                rank: 0, world: 1, logical_owner_to_rank: vec![0],
                transport: mgbfs_core::config::ReferenceTransport::Lsa,
                batch: 2, layer_capacity: 8, state_ring_capacity: 16, state_descriptor_capacity: 16,
                buckets: 8, shards: 2, job_buckets: 2, bucket_capacity: 8,
                prededup: true, generation_variant: 1, untouched_vram_reserve: 1 << 30,
            };
            match DistributedNativeBfs::new(&graph, [0;16], [0;128], cfg) {
                Ok(_) => panic!("invalid credit window admitted"),
                Err(error) => assert_eq!(error, "EPOCH_WINDOW_CONFIG"),
            }
        }
    }
    extern "C" {
        fn mgbfs_nccl_lsa_fatal_vote(comm: *mut c_void, send: *const u32,
            receive: *mut u32, stream: *mut c_void) -> i32;
    }

    #[test]
    fn tensor_hash_first_rejects_unsupported_device_at_construction() {
        extern "C" { fn cudaDeviceGetAttribute(out: *mut i32, attribute: i32, device: i32) -> i32; }
        let (mut major, mut minor) = (0, 0);
        assert_eq!(unsafe { cudaDeviceGetAttribute(&mut major, 75, 0) }, 0);
        assert_eq!(unsafe { cudaDeviceGetAttribute(&mut minor, 76, 0) }, 0);
        if (major, minor) == (7, 5) { return; } // Unsupported-device fixture only.
        let graph = MatrixGroup::unitriangular(3, 2).unwrap();
        let mut id = [0u8; 128];
        assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
        let cfg = DistributedConfig {
            epoch_window: 2,
            rank: 0, world: 1, logical_owner_to_rank: vec![0, 0],
            transport: mgbfs_core::config::ReferenceTransport::Lsa,
            batch: 2, layer_capacity: 8, state_ring_capacity: 16, state_descriptor_capacity: 16,
            buckets: 8, shards: 2, job_buckets: 2, bucket_capacity: 8,
            prededup: true, generation_variant: 1, untouched_vram_reserve: 1 << 30,
        };
        match DistributedNativeBfs::new_reference_with_owner_and_cancel(
            &graph, [0; 16], id, cfg, Some(8), OwnerBackend::CubSortMerge,
            256, true, None, None,
        ) {
            Ok(_) => panic!("unsupported Tensor backend was admitted until first batch"),
            Err(error) => assert_eq!(error, "HASH_FIRST_TC_DEVICE_UNSUPPORTED"),
        }
    }

    #[test]
    fn lsa_logical_fatal_closes_admission_after_the_completed_epoch() {
        // Removing the device-to-mapped failure publication, or ignoring it
        // in admission, must let this real runtime wrongly admit another batch.
        for (ring_fatal, owner_error) in [(0, 0), (7, 0), (0, 9)] {
            let graph = MatrixGroup::unitriangular(3, 2).unwrap();
            let mut id = [0u8; 128];
            assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
            let cfg = DistributedConfig {
                epoch_window: 2,
                rank: 0, world: 1, logical_owner_to_rank: vec![0, 0],
                transport: mgbfs_core::config::ReferenceTransport::Lsa,
                batch: 2, layer_capacity: 8, state_ring_capacity: 16, state_descriptor_capacity: 16,
                buckets: 8, shards: 2, job_buckets: 2, bucket_capacity: 8,
                prededup: true, generation_variant: 1, untouched_vram_reserve: 1 << 30,
            };
            let bfs = DistributedNativeBfs::new_reference_with_owner(
                &graph, [0; 16], id, cfg, None, OwnerBackend::CubSortMerge, 256,
            ).unwrap();
            // Generic reductions can be nonzero without an owner failure.
            bfs.collective_send.put_u32(3).unwrap();
            check(unsafe { mgbfs_nccl_lsa_fatal_vote(
                bfs.comm.0, bfs.collective_send.ptr.cast(),
                bfs.collective_recv.ptr.cast(), bfs.stream.0,
            ) }).unwrap();
            check(unsafe { cudaStreamSynchronize(bfs.stream.0) }).unwrap();
            assert!(bfs.ensure_not_cancelled().is_ok(), "non-owner vote cancelled admission");
            bfs.ring.put(&[Ring { fatal: ring_fatal, capacity: 16, ..Ring::default() }]).unwrap();
            bfs.control.put(&[Control { error: owner_error, ..Control::default() }]).unwrap();
            check(unsafe { mgbfs_owner_lsa_fatal_gate(
                bfs.comm.0, bfs.ring.ptr.cast(), bfs.control.ptr.cast(),
                bfs.collective_send.ptr.cast(), bfs.collective_recv.ptr.cast(), bfs.stream.0,
            ) }).unwrap();
            // Test-only drain stands for observing a completed epoch credit.
            check(unsafe { cudaStreamSynchronize(bfs.stream.0) }).unwrap();
            let admitted = bfs.ensure_not_cancelled();
            if ring_fatal == 0 && owner_error == 0 {
                assert!(admitted.is_ok(), "healthy epoch rejected: {admitted:?}");
            } else {
                assert!(admitted.is_err(), "completed fatal epoch reopened batch admission");
            }
        }
    }

    #[test]
    fn constructor_preserves_numeric_failure_without_vendor_message() {
        unsafe extern "C" fn unused_destroy(_: *mut c_void) {
            panic!("failed constructor must not create a live plan");
        }
        let error = match Plan::new(unused_destroy, |_, _| 3) {
            Ok(_) => panic!("failed native constructor was accepted"),
            Err(error) => error,
        };
        assert!(error.contains('3'), "native status was lost: {error:?}");
        assert!(!error.is_empty());
    }
}
struct Stream(*mut c_void);
impl Drop for Stream {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        unsafe {
            let trace = std::env::var_os("MGBFS_TRACE_FAILURE_TEARDOWN").is_some();
            if trace { eprintln!("MGBFS_FAILURE_TEARDOWN stream={:?} stage=stream_drain_begin",self.0); }
            cudaStreamSynchronize(self.0);
            if trace { eprintln!("MGBFS_FAILURE_TEARDOWN stream={:?} stage=stream_drain_end",self.0); }
            cudaStreamDestroy(self.0);
        }
    }
}
struct Event(*mut c_void);
impl Event {
    fn new() -> Result<Self> {
        let mut p = std::ptr::null_mut();
        check(unsafe { cudaEventCreateWithFlags(&mut p, 2) })?;
        Ok(Self(p))
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            cudaEventDestroy(self.0);
        }
    }
}
// A communicator created during setup must abort peers if a later local
// allocation fails before the final constructor agreement.
struct Comm(*mut c_void, bool);
impl Drop for Comm {
    fn drop(&mut self) {
        unsafe {
            if self.1 {
                mgbfs_nccl_abort(self.0);
            }
            mgbfs_nccl_destroy(self.0);
        }
    }
}

// Fixed request/response storage, allocated before depth zero. Source groups
// are kept separate because requests must return to the rank holding parents.
struct HashFirstStorage {
    ledger: mgbfs_core::memory::AllocationLedger,
    n: u32,
    modulus: u32,
    capacity: u32,
    generators: Buffer,
    coefficients: Buffer,
    offsets: Buffer,
    parent_count: Buffer,
    requests: [Buffer; 2],
    targets: [Buffer; 2],
    sorted_requests: Buffer,
    sorted_targets: Buffer,
    received_requests: Buffer,
    outgoing_responses: Buffer,
    incoming_responses: Buffer,
    count: Buffer,
    local_fatal: Buffer,
    group_fatal: Buffer,
    materialize: Plan,
    pending_counts: [u32; 2],
    pending_extents: [Vec<Extent>; 2],
    device: Option<HashFirstDevice>,
}
struct HashFirstDevice {
    counts: Buffer,
    extents: Buffer,
    controls: Buffer,
    exchange_counts: Buffer,
    generation_control: Buffer,
}
fn append_extent(extents: &mut Vec<Extent>, mut extent: Extent) -> Result<()> {
    if extent.count == 0 {
        return Ok(());
    }
    extent.padding[1] = extent.padding[1].max(extent.descriptor);
    if let Some(last) = extents.last_mut().filter(|last| {
        last.begin + last.count == extent.begin && last.sequence + last.count == extent.sequence
    }) {
        last.count += extent.count;
        last.granted_rows = u32::try_from(last.count).map_err(|_| "EXTENT_COUNT_OVERFLOW")?;
        last.padding[1] = extent.padding[1];
    } else {
        if extents.len() == extents.capacity() {
            return Err("HOST_EXTENT_CAPACITY".into());
        }
        extents.push(extent);
    }
    Ok(())
}
impl HashFirstStorage {
    fn new(
        graph: &MatrixGroup,
        hash: &GemmHash,
        capacity: u32,
        stride: usize,
        frontier: u32,
        stream: *mut c_void,
        ledger: mgbfs_core::memory::AllocationLedger,
    ) -> Result<Self> {
        let b = |name: &str| {
            let a = ledger
                .allocations
                .iter()
                .find(|a| a.name == name)
                .ok_or("MISSING_HASH_FIRST_ALLOCATION")?;
            Buffer::new(
                usize::try_from(a.payload_bytes).map_err(|_| "BYTE_OVERFLOW")?,
                stream,
            )
        };
        let matrices: Vec<u8> = graph.generators.iter().flatten().copied().collect();
        let generators = b("generators")?;
        generators.put(&matrices)?;
        let coefficients = b("coefficients")?;
        coefficients.put(&hash.coefficients)?;
        let offsets = b("offsets")?;
        offsets.put(&hash.offsets)?;
        Ok(Self {
            n: graph.rows as u32,
            modulus: graph.modulus as u32,
            capacity,
            generators,
            coefficients,
            offsets,
            parent_count: b("parent_count")?,
            requests: [b("local_requests")?, b("remote_requests")?],
            targets: [b("local_targets")?, b("remote_targets")?],
            sorted_requests: b("sorted_requests")?,
            sorted_targets: b("sorted_targets")?,
            received_requests: b("received_requests")?,
            outgoing_responses: b("outgoing_responses")?,
            incoming_responses: b("incoming_responses")?,
            count: b("request_count")?,
            local_fatal: b("local_fatal")?,
            group_fatal: b("group_fatal")?,
            materialize: Plan::new(mgbfs_materialize_destroy, |out, e| unsafe {
                mgbfs_materialize_create(stride as u32, capacity, frontier, out, e, 512)
            })?,
            pending_counts: [0; 2],
            pending_extents: [Vec::with_capacity(2), Vec::with_capacity(2)],
            device: if ledger.allocations.iter().any(|a| a.name == "device_counts") {
                Some(HashFirstDevice { counts: b("device_counts")?,
                    extents: b("device_extents")?, controls: b("device_controls")?,
                    exchange_counts: b("device_exchange_counts")?,
                    generation_control: b("device_generation_control")? })
            } else { None },
            ledger,
        })
    }
}

pub struct DistributedNativeBfs {
    #[cfg(feature = "library-owner")]
    library_owner: Option<LibraryOwnerStorage>,
    cfg: DistributedConfig,
    width: usize,
    stride: usize,
    permutation_n: Option<u32>,
    moves: u32,
    candidates: u32,
    depth: u32,
    current_count: u32,
    prev_count: u32,
    failed: bool,
    stream: Stream,
    generation_stream: Stream,
    generation_done: NativeEvent,
    pack_done: Event,
    generation_sequence: u64,
    dense_lookahead: u64,
    exchange_stream: Stream,
    exchange_done: Event,
    owner_consumed: Option<Event>,
    epoch_completed: Vec<Event>,
    epoch_outstanding: std::collections::VecDeque<usize>,
    archive_stream: Stream,
    archive_done: [Event; 2],
    archived_depth: Option<u32>,
    cancel_mirror: Option<CancelMirror>,
    comm: Comm,
    // Declared after Comm so the callback context outlives Comm::drop.
    cancel_requested: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    failure_report: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    retirement: Option<std::sync::Arc<crate::bootstrap::SearchRetirement>>,
    lsa_view: Option<LsaView>,
    generate: Option<Plan>,
    hash: Option<Plan>,
    hash_first: Option<HashFirstStorage>,
    hash_first_tensor_generation: bool,
    archive_hash: Plan,
    route: Plan,
    owner: Option<BoundedOwnerStorage>,
    shared_memory: mgbfs_core::memory::AllocationLedger,
    owned_memory: mgbfs_core::memory::AllocationLedger,
    states: Buffer,
    prev: Buffer,
    curr: Buffer,
    children: Buffer,
    child_hashes: Buffer,
    archive_hashes: Buffer,
    archive_states: Buffer,
    sorted_hashes: Buffer,
    sorted_refs: Buffer,
    route_count: Buffer,
    owner_window: Option<Buffer>,
    native_rank: Option<NativeRankStorage>,
    packed_states: Buffer,
    owner_counts: Buffer,
    recv_states: Option<Buffer>,
    recv_hashes: Option<Buffer>,
    recv_count: Option<Buffer>,
    identity_refs: Buffer,
    directory: Buffer,
    fatal: Buffer,
    jobs_gpu: Buffer,
    control: Buffer,
    ring: Buffer,
    extent: Buffer,
    next_extents: Option<Buffer>,
    next_extent_count: Option<Buffer>,
    layer_count: Buffer,
    incoming_dir: Vec<Range>,
    prev_dir: Vec<Range>,
    curr_dir: Vec<Range>,
    descriptors: Vec<BucketJob>,
    spans: Vec<JobSpan>,
    dense_results: Vec<Extent>,
    front: Vec<Extent>,
    next: Vec<Extent>,
    collective_send: Buffer,
    collective_recv: Buffer,
}
impl DistributedNativeBfs {
    fn rank_owner_mode(&self) -> bool {
        if self.native_rank.is_some() { return true; }
        #[cfg(feature = "library-owner")]
        if self.library_owner.as_ref().is_some_and(|owner| owner.rank_mode) { return true; }
        false
    }
    fn commit_rank_native_batch(
        &mut self, states: *const u8, hashes: *const c_void,
        begin: *const u32, rows: *const u32, source_rows: *const u32, group: usize,
    ) -> Result<()> {
        let owner = self.owner.as_ref().ok_or("NATIVE_OWNER_MISSING")?;
        let rank = self.native_rank.as_ref().ok_or("NATIVE_RANK_MISSING")?;
        #[cfg(debug_assertions)]
        let capture = OwnerCaptureProbe::begin(self.stream.0)?;
        let logical_owner = self.cfg.logical_owner_to_rank.iter()
            .position(|&r| r == self.cfg.rank).ok_or("OWNER_MAP")? as u32;
        let s = self.stream.0;
        let extent = self.extent.ptr.cast::<Extent>();
        let selected_count = unsafe { self.control.at(8).cast::<u32>() };
        unsafe {
            check(mgbfs_bounded_owner_rank_compare(owner.plan.0, self.jobs_gpu.ptr.cast(),
                self.cfg.buckets, hashes, begin, rows, source_rows,
                self.prev.ptr, rank.previous.ptr.cast(), self.prev_count.into(),
                self.curr.ptr, rank.current.ptr.cast(), self.current_count.into(),
                owner.accepted.ptr, owner.lengths.ptr.cast(), logical_owner,
                self.cfg.world, self.cfg.buckets/self.cfg.shards, self.depth,
                owner.counts.ptr.cast(), self.control.ptr.cast(), self.ring.ptr.cast(), s))?;
            check(mgbfs_bounded_owner_rank_metadata(owner.counts.ptr.cast(),
                owner.lengths.ptr.cast(), self.cfg.buckets, self.cfg.shards,
                self.cfg.bucket_capacity, rank.survivors.ptr.cast(), rank.accepted.ptr.cast(),
                rank.capacities.ptr.cast(), rank.offsets.ptr.cast(), self.control.ptr.cast(), s))?;
            check(mgbfs_state_reserve_rank_batch(self.ring.ptr.cast(), self.control.ptr.cast(),
                extent, rank.survivors.ptr.cast(), rank.accepted.ptr.cast(),
                rank.capacities.ptr.cast(), self.cfg.shards, rank.offsets.ptr.cast(),
                self.layer_count.ptr.cast(), self.cfg.layer_capacity,
                self.hash_first.as_ref().map_or(0, |h| h.capacity),
                u32::from(self.hash_first.is_some()), s))?;
            check(mgbfs_bounded_owner_rank_commit(owner.plan.0, self.jobs_gpu.ptr.cast(),
                self.cfg.buckets, hashes, owner.accepted.ptr, owner.lengths.ptr.cast(),
                owner.counts.ptr.cast(), self.control.ptr.cast(),
                std::ptr::addr_of!((*extent).granted_rows), owner.selected.ptr.cast(), s))?;
            if let Some(h) = self.hash_first.as_ref() {
                let d = h.device.as_ref().ok_or("HASH_FIRST_DEVICE_STORAGE")?;
                check(mgbfs_state_build_rank_requests(states.cast(), source_rows,
                    self.candidates, owner.selected.ptr.cast(), selected_count, h.capacity,
                    h.requests[group].ptr.cast(), h.targets[group].ptr.cast(),
                    d.counts.at(group*4).cast(), self.ring.ptr.cast(), self.control.ptr.cast(), extent, s))?;
                check(cudaMemcpyAsync(d.extents.at(group*std::mem::size_of::<Extent>()),
                    extent.cast(), std::mem::size_of::<Extent>(), 3, s))?;
                check(cudaMemcpyAsync(d.controls.at(group*std::mem::size_of::<Control>()),
                    self.control.ptr, std::mem::size_of::<Control>(), 3, s))?;
            } else {
                check(mgbfs_state_materialize_rank_batch(states, source_rows,
                    self.candidates, owner.selected.ptr.cast(), selected_count,
                    self.candidates, self.stride as u32, self.states.ptr.cast(),
                    self.ring.ptr.cast(), self.control.ptr.cast(), extent, s))?;
                check(mgbfs_state_publish_next_extent(self.ring.ptr.cast(), self.control.ptr.cast(),
                    extent, self.next_extent_count.as_ref().ok_or("NEXT_EXTENT_COUNT_MISSING")?.ptr.cast(),
                    self.next_extents.as_ref().ok_or("NEXT_EXTENTS_MISSING")?.ptr.cast(), 2, s))?;
            }
        }
        #[cfg(debug_assertions)]
        if let Some(capture) = capture { capture.launch()?; }
        Ok(())
    }
    /// Requested host-pinned payload; CUDA's page/registration overhead is not
    /// part of this byte-exact payload and is measured separately.
    pub fn transport_control_pinned_payload_bytes(&self) -> u64 {
        if self.lsa_view.is_some() { 4 } else { 0 }
    }
    pub fn abort_group(&mut self) {
        if !self.failed {
            self.failed = true;
            unsafe { mgbfs_nccl_abort(self.comm.0); }
        }
    }
    pub fn set_cancel_token(&mut self, token: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Result<()> {
        self.cancel_mirror = None;
        check(unsafe { mgbfs_nccl_bind_cancel(self.comm.0, Some(nccl_cancel_probe),
            std::sync::Arc::as_ptr(&token).cast_mut().cast()) })?;
        self.cancel_requested = Some(token.clone());
        if self.lsa_view.is_some() {
            let mut word = std::ptr::null_mut();
            check(unsafe { mgbfs_nccl_lsa_cancel_word(self.comm.0, &mut word) })?;
            self.cancel_mirror = Some(unsafe { CancelMirror::new(token.clone(), word)? });
        }
        Ok(())
    }
    pub fn set_failure_token(&mut self, token: std::sync::Arc<std::sync::atomic::AtomicU8>) {
        self.failure_report = Some(token);
    }
    pub fn set_retirement_token(&mut self, token: std::sync::Arc<crate::bootstrap::SearchRetirement>) -> Result<()> {
        check(unsafe { mgbfs_nccl_bind_retirement(self.comm.0, Some(nccl_retirement_probe),
            std::sync::Arc::as_ptr(&token).cast_mut().cast()) })?;
        self.retirement = Some(token);
        Ok(())
    }
    fn ensure_not_cancelled(&self) -> Result<()> {
        if let Some(view) = self.lsa_view.as_ref() {
            let stopped = unsafe { &*view.terminal }.load(std::sync::atomic::Ordering::Acquire);
            if stopped != 0 {
                return Err(if stopped & 2 != 0 { "GROUP_OWNER_OR_PRE_OWNER_FATAL:LSA_DEVICE_LOGICAL_FATAL" }
                    else { "LSA_TRANSPORT_CANCELLED" }.into());
            }
        }
        if self.cancel_requested.as_ref().is_some_and(|flag|
            flag.load(std::sync::atomic::Ordering::Acquire)) {
            Err("REMOTE_SEARCH_CANCELLED".into())
        } else { Ok(()) }
    }
    // A stream containing NCCL or LSA work must not trap the dispatcher in
    // cudaStreamSynchronize: only this dispatcher is allowed to abort its
    // communicator after the TCP sideband requests cancellation.
    fn wait_comm_stream(&self, stream: *mut c_void) -> Result<()> {
        wait_nccl_stream(self.comm.0, stream, self.cancel_requested.as_deref())
    }
    fn wait_epoch_credit(&self, slot: usize) -> Result<()> {
        let event = self.epoch_completed.get(slot).ok_or("EPOCH_CREDIT_SLOT")?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            self.ensure_not_cancelled()?;
            match unsafe { mgbfs_cuda::ffi::cudaEventQuery(event.0) } {
                // The vote may publish failure between the first host probe
                // and observing completion; never release admission unchecked.
                0 => return self.ensure_not_cancelled(),
                600 => {
                    match unsafe { mgbfs_nccl_poll(self.comm.0) } {
                        0 | 4 => {},
                        _ => return Err("EPOCH_NCCL_ASYNC_FAILURE".into()),
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err("EPOCH_CREDIT_TIMEOUT".into());
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                code => return Err(format!("EPOCH_EVENT_QUERY_{code}")),
            }
        }
    }
    /// Explicit library owner policy. Pool reservation is fixed before depth 0;
    /// no fallback to the existing bounded owner is permitted.
    #[cfg(feature = "library-owner")]
    pub fn new_library_reference(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        materialization_capacity: Option<u32>,
        pool_bytes: u64,
    ) -> Result<Self> {
        Self::new_library_reference_with_generation(
            graph,
            seed,
            id,
            cfg,
            materialization_capacity,
            pool_bytes,
            false,
        )
    }
    /// Same fixed library owner with an explicit experimental HASH_FIRST
    /// Tensor Core generator. No scalar substitution is performed.
    #[cfg(feature = "library-owner")]
    pub fn new_library_reference_with_generation(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        materialization_capacity: Option<u32>,
        pool_bytes: u64,
        tensor_generation: bool,
    ) -> Result<Self> {
        Self::new_library_reference_with_owner(
            graph,
            seed,
            id,
            cfg,
            materialization_capacity,
            pool_bytes,
            tensor_generation,
            ReferenceOwner::CudfRelational,
        )
    }

    /// Explicit reference library selection; no substitution on pool/capacity
    /// failure. All variants share the native route/materialization/archive path.
    #[cfg(feature = "library-owner")]
    pub fn new_library_reference_with_owner(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        materialization_capacity: Option<u32>,
        pool_bytes: u64,
        tensor_generation: bool,
        library_owner: ReferenceOwner,
    ) -> Result<Self> {
        Self::new_library_reference_with_owner_and_cancel(graph, seed, id, cfg,
            materialization_capacity, pool_bytes, tensor_generation, library_owner, None, None)
    }
    #[cfg(feature = "library-owner")]
    pub fn new_library_reference_with_owner_and_cancel(
        graph: &MatrixGroup, seed: [u8; 16], id: [u8; 128],
        cfg: DistributedConfig, materialization_capacity: Option<u32>,
        pool_bytes: u64, tensor_generation: bool, library_owner: ReferenceOwner,
        startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        startup_failure: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        if matches!(library_owner, ReferenceOwner::Native(_)) {
            return Err("REFERENCE_LIBRARY_OWNER".into());
        }
        if matches!(library_owner, ReferenceOwner::CucoRank) && materialization_capacity.is_some()
            && cfg.transport != mgbfs_core::config::ReferenceTransport::Lsa {
            return Err("REFERENCE_CUCO_RANK_DENSE_ONLY".into());
        }
        if tensor_generation && materialization_capacity.is_none() {
            return Err("REFERENCE_HASH_FIRST_GENERATION".into());
        }
        Self::new_profile(
            graph,
            seed,
            id,
            cfg,
            materialization_capacity,
            OwnerBackend::CubSortMerge,
            256,
            tensor_generation,
            Some((pool_bytes, library_owner)),
            None,
            startup_cancel,
            startup_failure,
        )
    }
    /// The 29 shared Buffer allocations, excluding library/profile/transport
    /// allocations. This is not the complete rank memory budget.
    pub fn shared_memory(&self) -> &mgbfs_core::memory::AllocationLedger {
        &self.shared_memory
    }
    /// All explicit runtime/library device allocations. Excludes CUDA/NCCL
    /// internal residency, pinned archive and disk; not full hardware preflight.
    pub fn owned_memory(&self) -> &mgbfs_core::memory::AllocationLedger {
        &self.owned_memory
    }
    /// Additional profile storage, not total rank VRAM. Includes CUB query
    /// results captured before allocation; reserved bytes use 256B alignment.
    pub fn hash_first_memory(&self) -> Option<&mgbfs_core::memory::AllocationLedger> {
        self.hash_first.as_ref().map(|h| &h.ledger)
    }
    pub fn new(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
    ) -> Result<Self> {
        Self::new_profile(
            graph,
            seed,
            id,
            cfg,
            None,
            OwnerBackend::CubSortMerge,
            256,
            false,
            None,
            None,
            None,
            None,
        )
    }
    /// Explicit scalar CUDA HASH_FIRST reference; never silently uses DENSE.
    pub fn new_hash_first_reference(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        materialization_capacity: u32,
    ) -> Result<Self> {
        Self::new_profile(
            graph,
            seed,
            id,
            cfg,
            Some(materialization_capacity),
            OwnerBackend::CubSortMerge,
            256,
            false,
            None,
            None,
            None,
            None,
        )
    }
    /// Explicit fixed owner policy; HASH_FIRST is selected by a nonzero
    /// materialization capacity. No backend/profile change after allocation.
    pub fn new_reference_with_owner(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        materialization_capacity: Option<u32>,
        owner: OwnerBackend,
        tile_limit: u32,
    ) -> Result<Self> {
        Self::new_reference_with_owner_and_cancel(graph, seed, id, cfg,
            materialization_capacity, owner, tile_limit, false, None, None)
    }
    pub fn new_reference_with_owner_and_cancel(
        graph: &MatrixGroup, seed: [u8; 16], id: [u8; 128],
        cfg: DistributedConfig, materialization_capacity: Option<u32>,
        owner: OwnerBackend, tile_limit: u32, tensor_generation: bool,
        startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        startup_failure: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        if tensor_generation && materialization_capacity.is_none() {
            return Err("REFERENCE_HASH_FIRST_GENERATION".into());
        }
        Self::new_profile(
            graph,
            seed,
            id,
            cfg,
            materialization_capacity,
            owner,
            tile_limit,
            tensor_generation,
            None,
            None,
            startup_cancel,
            startup_failure,
        )
    }
    /// Explicit experimental Tensor Core generation; hash projection still
    /// uses register reductions. No profile change or extra state allocation.
    pub fn new_hash_first_tc_with_owner(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        materialization_capacity: u32,
        owner: OwnerBackend,
        tile_limit: u32,
    ) -> Result<Self> {
        Self::new_profile(
            graph,
            seed,
            id,
            cfg,
            Some(materialization_capacity),
            owner,
            tile_limit,
            true,
            None,
            None,
            None,
            None,
        )
    }
    /// Explicit DENSE position-action graph with a repeated-symbol start.
    /// Ordinary MatrixGroup constructors retain their invertibility checks.
    pub fn new_lrx_multiset_reference(
        word_graph: &mgbfs_core::lrx_multiset::LrxMultiset,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        owner: ReferenceOwner,
        pool_bytes: Option<u64>,
    ) -> Result<Self> {
        Self::new_lrx_multiset_reference_and_cancel(word_graph, seed, id, cfg,
            owner, pool_bytes, None, None)
    }
    pub fn new_lrx_multiset_reference_and_cancel(
        word_graph: &mgbfs_core::lrx_multiset::LrxMultiset,
        seed: [u8; 16], id: [u8; 128], cfg: DistributedConfig,
        owner: ReferenceOwner, pool_bytes: Option<u64>,
        startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        startup_failure: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        if cfg.generation_variant != 5 {
            return Err("LRX_MULTISET_REQUIRES_COMPACT_DENSE".into());
        }
        let (native, library) = match (owner, pool_bytes) {
            (ReferenceOwner::Native(OwnerBackend::CubSortMerge), None) =>
                (OwnerBackend::CubSortMerge, None),
            (ReferenceOwner::CucoIndexed, Some(bytes)) =>
                (OwnerBackend::CubSortMerge, Some((bytes, owner))),
            _ => return Err("LRX_MULTISET_OWNER_CONFIG".into()),
        };
        let graph = word_graph.position_group()?;
        Self::new_profile(&graph, seed, id, cfg, None, native, 256, false,
            library, Some(word_graph.start()), startup_cancel, startup_failure)
    }
    fn new_profile(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        mut cfg: DistributedConfig,
        materialization_capacity: Option<u32>,
        owner_backend: OwnerBackend,
        tile_limit: u32,
        hash_first_tensor_generation: bool,
        library_options: Option<(u64, ReferenceOwner)>,
        compact_start: Option<&[u8]>,
        startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        startup_failure: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        let library_pool_bytes = library_options.map(|(bytes, _)| bytes);
        if cfg.epoch_window < 2 { return Err("EPOCH_WINDOW_CONFIG".into()); }
        if cfg.state_descriptor_capacity == 0 { return Err("STATE_DESCRIPTOR_CAPACITY".into()); }
        let epoch_window = cfg.epoch_window;
        if cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa
            && (cfg.world == 0
                || matches!(library_options, Some((_, owner)) if owner != ReferenceOwner::CucoRank))
        {
            return Err("LSA_REQUIRES_DEVICE_COUNT_RANK_OWNER".into());
        }
        if let Some(bytes) = library_pool_bytes {
            if !cfg!(feature = "library-owner")
                || bytes == 0
                || bytes % 256 != 0
                || cfg.untouched_vram_reserve < 1 << 30
            {
                return Err("LIBRARY_POOL_CONFIG".into());
            }
        }
        graph.validate()?;
        if let Some(capacity) = materialization_capacity {
            if capacity == 0
                || capacity > i32::MAX as u32
                || cfg.generation_variant != 1
                || graph.generators.len() > 65536
                || graph.modulus > 256
            {
                return Err("HASH_FIRST_REFERENCE_CONFIG".into());
            }
        }
        if owner_backend == OwnerBackend::BmmaBucket && !(1..=256).contains(&tile_limit) {
            return Err("BMMA_TILE_LIMIT".into());
        }
        let (local_buckets, local_shards) = crate::topology::reference_owner_geometry(
            cfg.world,
            cfg.rank,
            &cfg.logical_owner_to_rank,
            cfg.buckets,
            cfg.shards,
        )?;
        if cfg.batch == 0
            || cfg.layer_capacity == 0
            || cfg.state_ring_capacity == 0
            || !cfg.buckets.is_power_of_two()
            || !cfg.shards.is_power_of_two()
            || cfg.shards < cfg.world
            || cfg.shards > cfg.buckets
            || cfg.job_buckets == 0
            || cfg.job_buckets > cfg.buckets / cfg.shards
            || cfg.bucket_capacity == 0
        {
            return Err("DISTRIBUTED_CONFIG".into());
        }
        // Public config retains global prefix geometry. Persistent storage and
        // owner jobs use the local contiguous hash-prefix partition.
        cfg.buckets = local_buckets;
        cfg.shards = local_shards;
        check(unsafe { cudaSetDevice(cfg.rank as i32) })?;
        if hash_first_tensor_generation {
            match unsafe { mgbfs_hash_first_tc_validate_device() } {
                0 => (),
                3 => return Err("HASH_FIRST_TC_DEVICE_UNSUPPORTED".into()),
                status => return Err(format!("HASH_FIRST_TC_DEVICE_QUERY_{status}")),
            }
        }
        let permutation_n = encode_permutation_matrix(&graph.start, graph.rows)
            .ok()
            .filter(|_| {
                graph
                    .generators
                    .iter()
                    .all(|g| encode_permutation_matrix(g, graph.rows).is_ok())
            })
            .map(|_| graph.rows as u32);
        let start_state = if cfg.generation_variant == 5 {
            if permutation_n.is_none() {
                return Err("COMPACT_REQUIRES_PERMUTATION_GROUP".into());
            }
            if let Some(start) = compact_start {
                if start.len() != graph.rows || start.iter().any(|&x| usize::from(x) >= graph.rows) {
                    return Err("LRX_MULTISET_STATE".into());
                }
                start.to_vec()
            } else {
                encode_permutation_matrix(&graph.start, graph.rows)?
            }
        } else {
            graph.start.clone()
        };
        let width = start_state.len();
        let stride = (width + 15) & !15;
        let moves = graph.generators.len() as u32;
        let candidates = cfg.batch.checked_mul(moves).ok_or("CANDIDATE_OVERFLOW")?;
        if candidates > i32::MAX as u32 {
            return Err("CANDIDATE_CAPACITY".into());
        }
        let packet_stride = if materialization_capacity.is_some() {
            16
        } else {
            stride
        };
        let shared_shape = crate::distributed_memory::SharedBufferShape {
            state_stride: stride as u64,
            packet_stride: packet_stride as u64,
            batch: cfg.batch.into(),
            candidates: candidates.into(),
            layer_capacity: cfg.layer_capacity.into(),
            state_ring_capacity: cfg.state_ring_capacity.into(),
            buckets: cfg.buckets.into(),
            bucket_capacity: cfg.bucket_capacity.into(),
            job_buckets: cfg.job_buckets.into(),
            archive_width: permutation_n.unwrap_or(1).into(),
        };
        let native_rank_mode = library_pool_bytes.is_none()
            && cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa;
        let shared_memory = if library_pool_bytes.is_some() {
            crate::distributed_memory::library_shared_buffers_for_transport(
                shared_shape,
                cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa,
            )?
        } else if native_rank_mode {
            crate::distributed_memory::native_rank_shared_buffers(shared_shape, cfg.shards)?
        } else {
            crate::distributed_memory::shared_buffers(shared_shape)?
        };
        let mut owned_memory = mgbfs_core::memory::AllocationLedger::new(u64::MAX, 0)?;
        for a in &shared_memory.allocations {
            owned_memory.add(&format!("shared.{}", a.name), a.payload_bytes, 1, 256)?;
        }
        use crate::distributed_memory::append_query;
        use mgbfs_cuda::allocation::{query_generation, query_hash, query_route};
        let hash_first_ledger = if let Some(capacity) = materialization_capacity {
            let mut q = MaterializeBytes::default();
            check(unsafe {
                mgbfs_materialize_query(stride as u32, capacity, cfg.layer_capacity, &mut q)
            })?;
            let mut l = mgbfs_core::memory::hash_first_reference_ledger(
                width as u64,
                moves.into(),
                capacity.into(),
                stride as u64,
                [q.keys, q.sorted, q.indices, q.order, q.scratch],
            )?;
            if cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa {
                l.add("device_counts", 2, 4, 256)?;
                l.add("device_extents", 2, std::mem::size_of::<Extent>() as u64, 256)?;
                l.add("device_controls", 2, std::mem::size_of::<Control>() as u64, 256)?;
                l.add("device_exchange_counts", cfg.world.into(), 4, 256)?;
                l.add("device_generation_control", 2, 4, 256)?;
            }
            for a in &l.allocations {
                owned_memory.add(&format!("hash_first.{}", a.name), a.payload_bytes, 1, 256)?;
            }
            Some(l)
        } else {
            append_query(
                &mut owned_memory,
                "generation",
                &query_generation(
                    graph.rows as u32,
                    moves,
                    graph.modulus as u32,
                    cfg.batch,
                    cfg.generation_variant,
                )?,
            )?;
            append_query(
                &mut owned_memory,
                "hash",
                &query_hash(width as u32, candidates)?,
            )?;
            None
        };
        append_query(
            &mut owned_memory,
            "archive_hash",
            &query_hash(width as u32, cfg.batch)?,
        )?;
        append_query(&mut owned_memory, "route", &query_route(candidates)?)?;
        if let Some(bytes) = library_pool_bytes {
            owned_memory.add("library.fixed_pool", bytes, 1, 256)?;
        } else {
            let backend = u32::from(owner_backend == OwnerBackend::BmmaBucket);
            let mut oq = BoundedOwnerBytes::default();
            check(unsafe {
                mgbfs_bounded_owner_query(
                    candidates,
                    cfg.job_buckets,
                    cfg.bucket_capacity,
                    backend,
                    candidates,
                    tile_limit,
                    &mut oq,
                )
            })?;
            append_query(
                &mut owned_memory,
                "owner",
                &oq.report(candidates, cfg.job_buckets, cfg.bucket_capacity, backend)?,
            )?;
        }
        if cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa {
            let slot = crate::distributed_memory::lsa_symmetric_slot_bytes(
                candidates, stride as u32)?;
            owned_memory.add("transport.lsa_symmetric_slot", slot, 1, 256)?;
        }
        let mut raw = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw, 1) })?;
        let stream = Stream(raw);
        let mut raw_generation = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw_generation, 1) })?;
        let generation_stream = Stream(raw_generation);
        let generation_done = NativeEvent::new()?;
        let pack_done = Event::new()?;
        let mut raw_exchange = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw_exchange, 1) })?;
        let exchange_stream = Stream(raw_exchange);
        let exchange_done = Event::new()?;
        let mut epoch_completed = Vec::new();
        epoch_completed.try_reserve_exact(epoch_window)
            .map_err(|_| "EPOCH_EVENT_CAPACITY")?;
        for _ in 0..epoch_window { epoch_completed.push(Event::new()?); }
        let mut epoch_outstanding = std::collections::VecDeque::new();
        epoch_outstanding.try_reserve_exact(epoch_window)
            .map_err(|_| "EPOCH_CREDIT_CAPACITY")?;
        let mut raw_archive = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw_archive, 1) })?;
        let archive_stream = Stream(raw_archive);
        let mut comm = std::ptr::null_mut();
        let mut error = [0i8; 512];
        if unsafe {
            mgbfs_nccl_create_with_cancel(
                cfg.rank,
                cfg.world,
                cfg.rank,
                id.as_ptr().cast(),
                &mut comm,
                error.as_mut_ptr(),
                512,
                startup_cancel.as_ref().map(|_| nccl_cancel_probe as extern "C" fn(*mut c_void) -> i32),
                startup_cancel.as_ref().map_or(std::ptr::null_mut(), |token|
                    std::sync::Arc::as_ptr(token).cast_mut().cast()),
            )
        } != 0
        {
            return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        let comm = Comm(comm, true);
        // Declared after Comm: on a constructor error the sideband learns
        // failure before communicator cleanup can wait for a peer.
        let mut startup_report = crate::failure::FailureReportGuard::new(startup_failure);
        #[cfg(debug_assertions)]
        if std::env::var("MGBFS_TEST_CONSTRUCTOR_FAULT_RANK")
            .ok().and_then(|rank| rank.parse::<u32>().ok()) == Some(cfg.rank) {
            return Err("TEST_INJECTED_CONSTRUCTOR_ERROR".into());
        }
        admit_device_group(
            comm.0,
            raw,
            owned_memory.total(),
            cfg.untouched_vram_reserve,
            startup_cancel.as_deref(),
        )?;
        let setup_send = Buffer::new(4, raw)?;
        let setup_recv = Buffer::new(4, raw)?;
        let lsa_view = if cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa {
            // Reserve the control words before the potentially large symmetric
            // allocation, so an OOM in prepare can still be voted by all ranks.
            let prepared = check(unsafe { mgbfs_nccl_lsa_prepare(
                comm.0, candidates, stride as u32,
                error.as_mut_ptr(), error.len(),
            ) });
            vote_group_error(
                prepared.map_err(|error| format!("LSA_PREPARE_GROUP: {error}")),
                |failed| setup_failure_vote(comm.0, raw, &setup_send, &setup_recv,
                                           failed, startup_cancel.as_deref()),
                "LSA_PREPARE_GROUP: peer rejected LSA prepare".into(),
            ).map_err(|error| { startup_report.publish(); error })?;
            let activated = check(unsafe { mgbfs_nccl_lsa_activate(
                comm.0, error.as_mut_ptr(), error.len(),
            ) });
            vote_group_error(
                activated.map_err(|error| format!("LSA_ACTIVATE_GROUP: {error}")),
                |failed| setup_failure_vote(comm.0, raw, &setup_send, &setup_recv,
                                           failed, startup_cancel.as_deref()),
                "LSA_ACTIVATE_GROUP: peer rejected LSA activation".into(),
            ).map_err(|error| { startup_report.publish(); error })?;
            let (mut count, mut fatal, mut hashes, mut states) =
                (std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null());
            check(unsafe { mgbfs_nccl_lsa_view(
                comm.0, &mut count, &mut fatal, &mut hashes, &mut states,
            ) })?;
            let mut terminal = std::ptr::null_mut();
            check(unsafe { mgbfs_nccl_lsa_cancel_word(comm.0, &mut terminal) })?;
            if terminal.is_null() || terminal as usize %
                std::mem::align_of::<std::sync::atomic::AtomicU32>() != 0 {
                return Err("LSA_CANCEL_WORD_ALIGNMENT".into());
            }
            Some(LsaView { count, fatal, hashes, states: states.cast(), terminal: terminal.cast() })
        } else {
            None
        };
        let local_result = (|| -> Result<Self> {
        let contract = GemmHash::from_seed(width, seed)?;
        let limbs = contract.limbs();
        let matrices: Vec<u8> = graph.generators.iter().flatten().copied().collect();
        let weights = vec![1u32; moves as usize];
        let generate = if materialization_capacity.is_some() {
            None
        } else {
            Some(Plan::new(mgbfs_generate_destroy, |out, e| unsafe {
                mgbfs_generate_create_macro_variant(
                    graph.rows as u32,
                    moves,
                    graph.modulus as u32,
                    cfg.batch,
                    matrices.as_ptr(),
                    weights.as_ptr(),
                    cfg.generation_variant,
                    out,
                    e,
                    512,
                )
            })?)
        };
        let hash = if materialization_capacity.is_some() {
            None
        } else {
            Some(Plan::new(mgbfs_hash_destroy, |out, e| unsafe {
                mgbfs_hash_create(
                    width as u32,
                    candidates,
                    limbs.as_ptr(),
                    contract.offsets.as_ptr(),
                    out,
                    e,
                    512,
                )
            })?)
        };
        let hash_first = materialization_capacity
            .zip(hash_first_ledger)
            .map(|(capacity, ledger)| {
                HashFirstStorage::new(
                    graph,
                    &contract,
                    capacity,
                    stride,
                    cfg.layer_capacity,
                    raw,
                    ledger,
                )
            })
            .transpose()?;
        let route = Plan::new(mgbfs_route_destroy, |out, e| unsafe {
            mgbfs_route_create(candidates, out, e, 512)
        })?;
        let archive_hash = Plan::new(mgbfs_hash_destroy, |out, e| unsafe {
            mgbfs_hash_create(
                width as u32,
                cfg.batch,
                limbs.as_ptr(),
                contract.offsets.as_ptr(),
                out,
                e,
                512,
            )
        })?;
        let owner = if library_pool_bytes.is_some() {
            None
        } else {
            Some(Plan::new(mgbfs_bounded_owner_destroy, |out, _| unsafe {
                match owner_backend {
                    OwnerBackend::CubSortMerge => mgbfs_bounded_owner_create(
                        candidates,
                        cfg.job_buckets,
                        cfg.bucket_capacity,
                        out,
                    ),
                    OwnerBackend::BmmaBucket => mgbfs_bounded_owner_create_backend(
                        candidates,
                        cfg.job_buckets,
                        cfg.bucket_capacity,
                        1,
                        candidates,
                        tile_limit,
                        out,
                    ),
                }
            })?)
        };
        let b = |name: &str| {
            let entry = shared_memory
                .allocations
                .iter()
                .find(|a| a.name == name)
                .ok_or("MISSING_SHARED_ALLOCATION")?;
            Buffer::new(
                usize::try_from(entry.payload_bytes).map_err(|_| "BYTE_OVERFLOW")?,
                raw,
            )
        };
        let (next_extents, next_extent_count) = if library_pool_bytes.is_some() || native_rank_mode {
            (Some(b("next_extents")?), Some(b("next_extent_count")?))
        } else {
            (None, None)
        };
        let states = b("states")?;
        let prev = b("prev")?;
        let curr = b("curr")?;
        let start_hash = contract.hash(&start_state)?;
        let start_owner = crate::topology::hash_owner(cfg.world, start_hash.0[3])?;
        let start_rank = cfg.logical_owner_to_rank[start_owner];
        let current_count = (start_rank == cfg.rank) as u32;
        if current_count == 1 {
            let mut start = vec![0u8; stride];
            start[..width].copy_from_slice(&start_state);
            states.put(&start)?;
            curr.put(&[start_hash.to_le_bytes()])?;
        }
        let identity_refs = b("identity_refs")?;
        identity_refs.put(&(0..u64::from(candidates)).collect::<Vec<_>>())?;
        let archive_done = [Event::new()?, Event::new()?];
        check(unsafe { cudaEventRecord(archive_done[0].0, raw) })?;
        check(unsafe { cudaEventRecord(archive_done[1].0, raw) })?;
        check(unsafe { cudaStreamSynchronize(raw) })?;
        let buckets = cfg.buckets as usize;
        let slots = buckets + 1;
        let directory = b("directory")?;
        let fatal = b("fatal")?;
        let route_count = b("route_count")?;
        route_count.put_u32(current_count)?;
        check(unsafe {
            rank_directory(
                cfg.world,
                curr.ptr,
                route_count.ptr.cast(),
                cfg.layer_capacity,
                cfg.buckets,
                cfg.logical_owner_to_rank
                    .iter()
                    .position(|&r| r == cfg.rank)
                    .ok_or("OWNER_MAP")? as u32,
                directory.ptr.cast(),
                fatal.ptr.cast(),
                raw,
            )
        })?;
        check(unsafe { cudaStreamSynchronize(raw) })?;
        if fatal.one::<u32>()? != 0 {
            return Err("INITIAL_DIRECTORY_FATAL".into());
        }
        let mut curr_dir = vec![Range::default(); buckets];
        directory.read(&mut curr_dir)?;
        let mut front = Vec::with_capacity(2);
        if current_count != 0 {
            front.push(Extent {
                count: 1,
                granted_rows: 1,
                ready: 1,
                padding: [0, 0, 0],
                ..Extent::default()
            });
        }
        let ring = b("ring")?;
        ring.put(&[Ring {
            tail: u64::from(current_count),
            descriptor_tail: u64::from(current_count),
            capacity: u64::from(cfg.state_ring_capacity),
            descriptor_capacity: u64::from(cfg.state_descriptor_capacity),
            ..Ring::default()
        }])?;
        let mut result = Self {
            #[cfg(feature = "library-owner")]
            library_owner: None,
            cfg: cfg.clone(),
            width,
            stride,
            permutation_n,
            moves,
            candidates,
            depth: 0,
            current_count,
            prev_count: 0,
            failed: false,
            cancel_requested: startup_cancel.clone(),
            cancel_mirror: None,
            failure_report: None,
            retirement: None,
            // Keep the setup-vote stream alive outside the fallible local
            // result. An error here must not destroy it before peers vote.
            stream: Stream(std::ptr::null_mut()),
            generation_stream,
            generation_done,
            pack_done,
            generation_sequence: 0,
            dense_lookahead: 0,
            exchange_stream,
            exchange_done,
            owner_consumed: (cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa)
                .then(Event::new).transpose()?,
            epoch_completed,
            epoch_outstanding,
            archive_stream,
            archive_done,
            archived_depth: None,
            // NCCL ownership transfers only after every rank accepts the
            // complete local allocation/owner setup below.
            comm: Comm(std::ptr::null_mut(), false),
            lsa_view,
            generate,
            hash,
            hash_first,
            hash_first_tensor_generation,
            archive_hash,
            route,
            owner: owner
                .map(|plan| -> Result<BoundedOwnerStorage> {
                    Ok(BoundedOwnerStorage {
                        plan,
                        accepted: b("accepted")?,
                        lengths: b("lengths")?,
                        counts: b("counts")?,
                        selected: b("selected")?,
                    })
                })
                .transpose()?,
            states,
            prev,
            curr,
            children: b("children")?,
            child_hashes: b("child_hashes")?,
            archive_hashes: b("archive_hashes")?,
            archive_states: b("archive_states")?,
            sorted_hashes: b("sorted_hashes")?,
            sorted_refs: b("sorted_refs")?,
            route_count,
            owner_window: if library_pool_bytes.is_some() || native_rank_mode { Some(b("owner_window")?) } else { None },
            native_rank: if native_rank_mode { Some(NativeRankStorage {
                previous: b("rank_prev_directory")?, current: b("rank_curr_directory")?,
                survivors: b("rank_shard_counts")?, accepted: b("rank_shard_accepted")?,
                capacities: b("rank_shard_capacities")?, offsets: b("rank_shard_offsets")?,
            }) } else { None },
            packed_states: b("packed_states")?,
            owner_counts: b("owner_counts")?,
            recv_states: (cfg.transport != mgbfs_core::config::ReferenceTransport::Lsa)
                .then(|| b("recv_states")).transpose()?,
            recv_hashes: (cfg.transport != mgbfs_core::config::ReferenceTransport::Lsa)
                .then(|| b("recv_hashes")).transpose()?,
            recv_count: (cfg.transport != mgbfs_core::config::ReferenceTransport::Lsa)
                .then(|| b("recv_count")).transpose()?,
            identity_refs,
            directory,
            fatal,
            jobs_gpu: b("jobs_gpu")?,
            control: b("control")?,
            ring,
            extent: b("extent")?,
            next_extents,
            next_extent_count,
            layer_count: b("layer_count")?,
            incoming_dir: vec![Range::default(); buckets],
            prev_dir: vec![Range::default(); buckets],
            curr_dir,
            descriptors: vec![BucketJob::default(); slots],
            spans: vec![JobSpan::default(); slots],
            dense_results: vec![Extent::default(); slots],
            front,
            next: Vec::with_capacity(2),
            collective_send: b("collective_send")?,
            collective_recv: b("collective_recv")?,
            shared_memory,
            owned_memory,
        };
        #[cfg(feature = "library-owner")]
        if let Some(pool_bytes) = library_pool_bytes {
            let plane_words =
                (mgbfs_core::library_memory::CandidateSoaLayout::plan(cfg.layer_capacity.into())?
                    .plane_stride_bytes
                    / 4) as usize;
            // Initial directory was built from the one AoS start key. Rewrite
            // it in-place as SoA before any library view starts borrowing it.
            if current_count != 0 {
                for plane in 0..4 {
                    unsafe {
                        check(cudaMemcpyAsync(
                            result.curr.at(plane * plane_words * 4),
                            (&start_hash.0[plane] as *const u32).cast(),
                            4,
                            1,
                            raw,
                        ))?;
                    }
                }
                check(unsafe { cudaStreamSynchronize(raw) })?;
            }
            let scratch = Buffer::new(
                mgbfs_core::library_memory::CandidateSoaLayout::plan(candidates.into())?
                    .allocation_bytes as usize,
                raw,
            )?;
            let control_transfer = unsafe { ControlTransfer::new(raw)? };
            let mut pool = std::ptr::null_mut();
            check(unsafe {
                mgbfs_library_pool_create_v1(pool_bytes, cfg.untouched_vram_reserve, &mut pool)
            })?;
            let per_shard = cfg.buckets / cfg.shards;
            let capacity = u32::try_from(
                (u64::from(per_shard) * u64::from(cfg.bucket_capacity))
                    .min(cfg.layer_capacity.into()),
            )
            .map_err(|_| "LIBRARY_SHARD_CAPACITY")?;
            let mut library = LibraryOwnerStorage {
                control_transfer,
                shards: Vec::with_capacity(cfg.shards as usize),
                scratch,
                previous: vec![0..0; cfg.shards as usize],
                current: Vec::with_capacity(cfg.shards as usize),
                capacity,
                plane_words,
                epoch: 0,
                closed: false,
                pool,
                cuco_workspace: std::ptr::null_mut(),
                rank: std::ptr::null_mut(),
                rank_mode: matches!(library_options, Some((_, ReferenceOwner::CucoRank))),
                rank_accepted: std::ptr::null(),
                logical_owner: cfg.logical_owner_to_rank.iter()
                    .position(|&r| r == cfg.rank).ok_or("OWNER_MAP")? as u32,
            };
            if matches!(library_options, Some((_, ReferenceOwner::CucoIndexed))) {
                check(unsafe {
                    mgbfs_library_cuco_workspace_create_v1(
                        candidates,
                        raw,
                        &mut library.cuco_workspace,
                    )
                })?;
            }
            for directories in result.curr_dir.chunks(per_shard as usize) {
                let first = directories[0].begin as u32;
                let last = directories.last().unwrap();
                library
                    .current
                    .push(first..(last.begin + last.count) as u32);
            }
            if library.rank_mode {
                library.rank = unsafe { create_rank_owner(
                    &library, &result.prev, &result.curr, cfg.shards, candidates,
                    cfg.world, raw,
                )? };
            } else {
                for shard in 0..cfg.shards as usize {
                    unsafe {
                        library.shards.push(LibraryShard::new_window_with_workspace(
                            history_view(&result.prev, plane_words, &library.previous[shard]),
                            history_view(&result.curr, plane_words, &library.current[shard]),
                            capacity,
                            if matches!(library_options, Some((_, ReferenceOwner::CucoIndexed))) {
                                Some(candidates)
                            } else {
                                None
                            },
                            library.cuco_workspace,
                            raw,
                        )?);
                    }
                }
            }
            result.library_owner = Some(library);
        }
        Ok(result)
        })();
        let group_failed = match setup_failure_vote(comm.0, raw, &setup_send, &setup_recv,
                              local_result.is_err(), startup_cancel.as_deref()) {
            Ok(failed) => failed,
            Err(error) => {
                // Notify peers before dropping any resources held by local_result.
                startup_report.publish();
                return Err(local_result.err().unwrap_or(error));
            }
        };
        if group_failed {
            startup_report.publish();
            return Err(local_result.err().unwrap_or_else(|| "REMOTE_CONSTRUCTOR_FATAL".into()));
        }
        let mut result = local_result?;
        result.stream = stream;
        result.comm = comm;
        result.comm.1 = false;
        #[cfg(debug_assertions)]
        if std::env::var("MGBFS_TEST_CONSTRUCTOR_LATE_FAULT_RANK")
            .ok().and_then(|rank| rank.parse::<u32>().ok()) == Some(result.cfg.rank) {
            startup_report.publish();
            return Err("TEST_INJECTED_CONSTRUCTOR_LATE_ERROR".into());
        }
        if let Some(token) = startup_cancel {
            result.set_cancel_token(token)
                .map_err(|error| { startup_report.publish(); error })?;
        }
        startup_report.disarm();
        #[cfg(debug_assertions)]
        if std::env::var("MGBFS_TEST_OWNER_CAPACITY_RANK")
            .ok().and_then(|rank| rank.parse::<u32>().ok()) == Some(result.cfg.rank) {
            // Test the real device reservation limit after common bootstrap
            // and preallocation. No environment lookup in the batch hot path.
            result.cfg.layer_capacity = result.cfg.layer_capacity.min(2);
        }
        Ok(result)
    }
    pub fn depth(&self) -> u32 {
        self.depth
    }
    /// Requested pool suballocation high water, not whole-device VRAM.
    /// Call on the rank's creating device, outside timed GPU stages.
    #[cfg(feature = "library-owner")]
    pub fn library_pool_usage(&self) -> Result<Option<PoolUsageV1>> {
        match self.library_owner.as_ref() {
            None => Ok(None),
            Some(library) => {
                let mut usage = PoolUsageV1::default();
                check(unsafe { mgbfs_library_pool_usage_v1(library.pool, &mut usage) })?;
                Ok(Some(usage))
            }
        }
    }
    pub fn frontier_len(&self) -> u32 {
        self.current_count
    }
    /// Submitted lookahead batches, not a claim of measured GPU overlap.
    pub fn dense_lookahead_batches(&self) -> u64 {
        self.dense_lookahead
    }
    /// Allocated completion-credit window; payload storage is unchanged.
    pub fn epoch_window(&self) -> usize {
        self.epoch_completed.len()
    }
    pub fn state_descriptor_capacity(&self) -> u32 {
        self.cfg.state_descriptor_capacity
    }
    fn enqueue_frontier_generation(&mut self, batch: ParentBatch) -> Result<u64> {
        let sequence = self
            .generation_sequence
            .checked_add(1)
            .ok_or("GENERATION_SEQUENCE")?;
        let s = self.generation_stream.0;
        // The frontier range is still live in StateRing. Only the *previous*
        // parent batch may retire while these read-only operations are running.
        unsafe {
            if let Some(h) = self.hash_first.as_ref() {
                let d = h.device.as_ref().ok_or("HASH_FIRST_DEVICE_STORAGE")?;
                check(mgbfs_device_store_u32(h.parent_count.ptr.cast(), batch.count, s))?;
                let generate = if self.hash_first_tensor_generation {
                    mgbfs_generate_hash_only_tc_admitted
                } else { mgbfs_generate_hash_only };
                check(generate(h.n, self.moves, h.modulus, self.stride as u32,
                    self.cfg.batch, self.candidates, self.cfg.rank, batch.sequence,
                    self.states.at(batch.begin as usize * self.stride).cast(),
                    h.generators.ptr.cast(), h.coefficients.ptr.cast(), h.offsets.ptr.cast(),
                    h.parent_count.ptr.cast(), self.child_hashes.ptr.cast(), self.children.ptr.cast(),
                    // Producer count must not overwrite the current owner's
                    // routed-count before its selected origins are committed.
                    d.generation_control.at(4).cast(), d.generation_control.ptr.cast(), s))?;
            } else {
            check(mgbfs_generate_run(
                self.generate.as_ref().ok_or("DENSE_GENERATOR_MISSING")?.0,
                self.states.at(batch.begin as usize * self.stride).cast(),
                self.children.ptr.cast(),
                batch.count,
                s,
            ))?;
            check(mgbfs_hash_run(
                self.hash.as_ref().ok_or("DENSE_HASH_MISSING")?.0,
                self.children.ptr.cast(),
                self.child_hashes.ptr.cast(),
                batch.count * self.moves,
                s,
            ))?;
            }
            self.generation_done.record(sequence, s)?;
        }
        self.generation_sequence = sequence;
        Ok(sequence)
    }
    fn all_max(&self, value: u32) -> Result<u32> {
        self.collective_send.put_u32(value)?;
        check(unsafe {
            mgbfs_nccl_all_reduce_max_u32(
                self.comm.0,
                self.collective_send.ptr.cast(),
                self.collective_recv.ptr.cast(),
                self.stream.0,
            )
        })?;
        self.wait_comm_stream(self.stream.0)?;
        self.collective_recv.one()
    }
    fn all_max_ring_fatal(&self) -> Result<u32> {
        check(unsafe {
            mgbfs_state_ring_fatal_vote_word(
                self.ring.ptr.cast(),
                self.collective_send.ptr.cast(),
                self.stream.0,
            )
        })?;
        check(unsafe {
            mgbfs_nccl_all_reduce_max_u32(
                self.comm.0,
                self.collective_send.ptr.cast(),
                self.collective_recv.ptr.cast(),
                self.stream.0,
            )
        })?;
        self.wait_comm_stream(self.stream.0)?;
        self.collective_recv.one()
    }
    #[cfg(feature = "library-owner")]
    fn commit_rank_library_batch(
        &mut self,
        source_states: *const u8,
        source_hashes: *const c_void,
        begin: *const u32,
        rows: *const u32,
        source_rows: *const u32,
        source_group: usize,
    ) -> Result<()> {
        let stream = self.stream.0;
        let library = self.library_owner.as_mut().ok_or("LIBRARY_OWNER_MISSING")?;
        if library.rank.is_null() || !library.rank_mode {
            return Err("LIBRARY_RANK_NOT_OPEN".into());
        }
        library.epoch = library.epoch.checked_add(1).ok_or("LIBRARY_EPOCH_OVERFLOW")?;
        let epoch = library.epoch;
        #[cfg(debug_assertions)]
        let capture = OwnerCaptureProbe::begin(stream)?;
        let mut candidates = CandidatesV1 {
            keys: KeysV1 { words: [std::ptr::null(); 4], rows: 0, reserved: 0 },
            source_indices: std::ptr::null(),
        };
        unsafe {
            check(mgbfs_library_candidates_from_aos_window_v1(
                source_hashes, begin, rows, self.candidates, self.candidates,
                library.scratch.ptr, library.scratch.bytes as u64,
                self.ring.ptr.cast(), self.control.ptr.cast(), stream, &mut candidates,
            ))?;
        }
        let mut batch = RankDeviceBatchV1 {
            high_words: std::ptr::null(), valid_rows: std::ptr::null(),
            selected: std::ptr::null(), selected_count: std::ptr::null(),
            source_indices: std::ptr::null(), accepted_counts: std::ptr::null(),
            accepted_capacities: std::ptr::null(), shard_counts: std::ptr::null_mut(),
            shard_offsets: std::ptr::null_mut(),
        };
        unsafe {
            check(mgbfs_library_rank_compare_v1(
                library.rank, epoch, candidates, rows,
                self.control.ptr.cast(), self.ring.ptr.cast(), &mut batch,
            ))?;
            check(mgbfs_owner_shard_counts(
                batch.high_words, batch.valid_rows, batch.selected,
                batch.selected_count, self.candidates, library.logical_owner,
                self.cfg.world, self.cfg.shards, batch.shard_counts,
                batch.shard_offsets, self.ring.ptr.cast(), self.control.ptr.cast(), stream,
            ))?;
            check(mgbfs_state_reserve_rank_batch(
                self.ring.ptr.cast(), self.control.ptr.cast(), self.extent.ptr.cast(),
                batch.shard_counts, batch.accepted_counts, batch.accepted_capacities,
                self.cfg.shards, batch.shard_offsets, self.layer_count.ptr.cast(),
                self.cfg.layer_capacity,
                self.hash_first.as_ref().map_or(0, |h| h.capacity),
                u32::from(self.hash_first.is_some()), stream,
            ))?;
            check(mgbfs_library_rank_commit_v1(
                library.rank, epoch, self.control.ptr.cast(), self.ring.ptr.cast(),
                self.extent.ptr.cast(),
            ))?;
            if let Some(h) = self.hash_first.as_ref() {
                let d = h.device.as_ref().ok_or("HASH_FIRST_DEVICE_STORAGE")?;
                check(mgbfs_state_build_rank_requests(
                    source_states.cast(), source_rows, self.candidates,
                    batch.source_indices, batch.selected_count, h.capacity,
                    h.requests[source_group].ptr.cast(), h.targets[source_group].ptr.cast(),
                    d.counts.at(source_group * 4).cast(),
                    self.ring.ptr.cast(), self.control.ptr.cast(), self.extent.ptr.cast(), stream,
                ))?;
                check(cudaMemcpyAsync(d.extents.at(source_group * std::mem::size_of::<Extent>()),
                    self.extent.ptr, std::mem::size_of::<Extent>(), 3, stream))?;
                check(cudaMemcpyAsync(d.controls.at(source_group * std::mem::size_of::<Control>()),
                    self.control.ptr, std::mem::size_of::<Control>(), 3, stream))?;
            } else {
              check(mgbfs_state_materialize_rank_batch(
                source_states, source_rows, self.candidates, batch.source_indices,
                batch.selected_count, self.candidates, self.stride as u32,
                self.states.ptr.cast(), self.ring.ptr.cast(), self.control.ptr.cast(),
                self.extent.ptr.cast(), stream,
              ))?;
              check(mgbfs_state_publish_next_extent(
                self.ring.ptr.cast(), self.control.ptr.cast(), self.extent.ptr.cast(),
                self.next_extent_count.as_ref().ok_or("NEXT_EXTENT_COUNT_MISSING")?.ptr.cast(),
                self.next_extents.as_ref().ok_or("NEXT_EXTENTS_MISSING")?.ptr.cast(),
                2, stream,
              ))?;
            }
        }
        library.rank_accepted = batch.accepted_counts;
        check(unsafe { mgbfs_library_rank_complete_v1(library.rank, epoch) })?;
        #[cfg(debug_assertions)]
        if let Some(capture) = capture { capture.launch()?; }
        Ok(())
    }
    #[cfg(feature = "library-owner")]
    fn commit_library_batch(
        &mut self,
        source_states: *const u8,
        source_hashes: *const c_void,
        rows: u32,
        source_group: usize,
    ) -> Result<()> {
        if self.library_owner.as_ref().is_some_and(|library| library.rank_mode) {
            return Err("RANK_OWNER_REQUIRES_DEVICE_WINDOW".into());
        }
        let s = self.stream.0;
        self.route_count.put_u32(rows)?;
        unsafe {
            check(rank_directory(
                self.cfg.world,
                source_hashes,
                self.route_count.ptr.cast(),
                self.candidates,
                self.cfg.buckets,
                self.cfg
                    .logical_owner_to_rank
                    .iter()
                    .position(|&r| r == self.cfg.rank)
                    .ok_or("OWNER_MAP")? as u32,
                self.directory.ptr.cast(),
                self.fatal.ptr.cast(),
                s,
            ))?;
            check(cudaStreamSynchronize(s))?;
        }
        if self.fatal.one::<u32>()? != 0 {
            return Err("LIBRARY_DIRECTORY_FATAL".into());
        }
        self.directory.read(&mut self.incoming_dir)?;
        let per_shard = (self.cfg.buckets / self.cfg.shards) as usize;
        let library = self.library_owner.as_mut().ok_or("LIBRARY_OWNER_MISSING")?;
        for shard in 0..library.shards.len() {
            let directories = &self.incoming_dir[shard * per_shard..(shard + 1) * per_shard];
            let begin = directories[0].begin;
            let last = directories.last().unwrap();
            let end = last
                .begin
                .checked_add(last.count)
                .ok_or("LIBRARY_DIRECTORY_OVERFLOW")?;
            if end < begin || end > u64::from(rows) {
                return Err("LIBRARY_DIRECTORY_RANGE".into());
            }
            let count = (end - begin) as u32;
            if count == 0 {
                continue;
            }
            library.epoch = library
                .epoch
                .checked_add(1)
                .ok_or("LIBRARY_EPOCH_OVERFLOW")?;
            let epoch = library.epoch;
            let mut candidates = CandidatesV1 {
                keys: KeysV1 {
                    words: [std::ptr::null(); 4],
                    rows: 0,
                    reserved: 0,
                },
                source_indices: std::ptr::null(),
            };
            unsafe {
                check(mgbfs_library_candidates_from_aos_v1(
                    source_hashes.cast::<u8>().add(begin as usize * 16).cast(),
                    count,
                    self.candidates,
                    library.scratch.ptr,
                    library.scratch.bytes as u64,
                    s,
                    &mut candidates,
                ))?;
            }
            let owner = &mut library.shards[shard];
            let survivors = unsafe { owner.compare(epoch, candidates)? };
            if let Some(h) = &self.hash_first {
                if u64::from(h.pending_counts[source_group]) + u64::from(survivors.rows)
                    > u64::from(h.capacity)
                {
                    return Err("HASH_FIRST_REQUEST_CAPACITY".into());
                }
            }
            let mut control = Control {
                stage: 1,
                survivors: survivors.rows,
                ..Control::default()
            };
            unsafe {
                library
                    .control_transfer
                    .upload(&control, self.control.ptr.cast())?;
                check(mgbfs_state_reserve_layer(
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    self.extent.ptr.cast(),
                    self.layer_count.ptr.cast(),
                    self.cfg.layer_capacity,
                    s,
                ))?;
            }
            let reserved_snapshot = unsafe {
                library.control_transfer.read(
                    self.control.ptr.cast(),
                    self.extent.ptr.cast(),
                    self.ring.ptr.cast(),
                    std::ptr::null(),
                )?
            };
            control = reserved_snapshot.control;
            let reserved = reserved_snapshot.extent;
            if control.error != 0 || reserved_snapshot.ring.fatal != 0 {
                return Err(format!("LIBRARY_RESERVE_FATAL_{}", control.error));
            }
            unsafe {
                owner.commit(epoch, reserved.granted_rows)?;
            }
            if survivors.rows == 0 {
                // Empty cuDF outputs may have a null index pointer. Complete
                // the zero-credit transaction without calling a materializer
                // whose ABI requires a non-null selection array.
                unsafe {
                    owner.complete(epoch)?;
                }
                continue;
            }
            control.stage = 2;
            unsafe {
                library
                    .control_transfer
                    .upload(&control, self.control.ptr.cast())?;
                if let Some(h) = self.hash_first.as_ref() {
                    let offset = h.pending_counts[source_group] as usize;
                    check(mgbfs_state_build_requests(
                        source_states.cast(),
                        rows,
                        self.identity_refs.at(begin as usize * 8).cast(),
                        count,
                        survivors.source_indices,
                        self.candidates,
                        h.requests[source_group].at(offset * 16).cast(),
                        h.targets[source_group].at(offset * 8).cast(),
                        h.count.ptr.cast(),
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                        self.extent.ptr.cast(),
                        s,
                    ))?;
                } else {
                    check(mgbfs_state_materialize_packed(
                        source_states.add(begin as usize * self.stride),
                        count,
                        survivors.source_indices,
                        self.candidates,
                        self.stride as u32,
                        self.states.ptr.cast(),
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                        self.extent.ptr.cast(),
                        s,
                    ))?;
                }
            }
            let completed_snapshot = unsafe {
                library.control_transfer.read(
                    self.control.ptr.cast(),
                    self.extent.ptr.cast(),
                    self.ring.ptr.cast(),
                    self.hash_first
                        .as_ref()
                        .map_or(std::ptr::null(), |h| h.count.ptr.cast()),
                )?
            };
            let control = completed_snapshot.control;
            if control.error != 0 || completed_snapshot.ring.fatal != 0 {
                return Err(format!("LIBRARY_MATERIALIZE_FATAL_{}", control.error));
            }
            let extent = completed_snapshot.extent;
            if let Some(h) = self.hash_first.as_mut() {
                if extent.ready != 0 || u64::from(completed_snapshot.count) != extent.count {
                    return Err("HASH_FIRST_REQUEST_PUBLICATION".into());
                }
                h.pending_counts[source_group] += extent.count as u32;
                append_extent(&mut h.pending_extents[source_group], extent)?;
            } else {
                if extent.ready != 1 {
                    return Err("LIBRARY_STATE_NOT_READY".into());
                }
                append_extent(&mut self.next, extent)?;
            }
            unsafe {
                // The successful snapshot read above drained this stream AFTER
                // commit and the final borrowed-result consumer. Fatal/ready
                // checks passed; no device work was enqueued since that drain.
                // The empty branch still uses complete(): its snapshot precedes
                // commit and cannot establish post-commit completion.
                owner.complete_after_stream_drain(epoch, s)?;
            }
        }
        Ok(())
    }
    fn commit_owner_batch(
        &mut self,
        source_states: *const u8,
        source_hashes: *const c_void,
        rows: u32,
        source_group: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        #[cfg(feature = "library-owner")]
        if self.library_owner.is_some() {
            return self.commit_library_batch(source_states, source_hashes, rows, source_group);
        }
        let owner = self.owner.as_ref().ok_or("NATIVE_OWNER_MISSING")?;
        let (owner_plan, accepted, lengths, counts, selected) = (
            owner.plan.0,
            owner.accepted.ptr,
            owner.lengths.ptr,
            owner.counts.ptr,
            owner.selected.ptr,
        );
        let s = self.stream.0;
        self.route_count.put_u32(rows)?;
        unsafe {
            check(rank_directory(
                self.cfg.world,
                source_hashes,
                self.route_count.ptr.cast(),
                self.candidates,
                self.cfg.buckets,
                self.cfg
                    .logical_owner_to_rank
                    .iter()
                    .position(|&r| r == self.cfg.rank)
                    .ok_or("OWNER_MAP")? as u32,
                self.directory.ptr.cast(),
                self.fatal.ptr.cast(),
                s,
            ))?;
            check(cudaStreamSynchronize(s))?;
        }
        if self.fatal.one::<u32>()? != 0 {
            return Err("DIRECTORY_FATAL".into());
        }
        self.directory.read(&mut self.incoming_dir)?;
        let (descriptor_count, span_count) = split(
            &self.incoming_dir,
            &self.prev_dir,
            &self.curr_dir,
            self.candidates,
            self.cfg.job_buckets,
            self.cfg.buckets / self.cfg.shards,
            self.depth,
            &mut self.descriptors,
            &mut self.spans,
        )?;
        self.jobs_gpu.put(&self.descriptors[..descriptor_count])?;
        for span_index in 0..span_count {
            let span = self.spans[span_index];
            let jobs = unsafe { self.jobs_gpu.at(span.first * 64).cast::<BucketJob>() };
            let hashes = unsafe {
                source_hashes
                    .cast::<u8>()
                    .add(span.source_begin as usize * 16)
            };
            let refs = unsafe {
                self.identity_refs
                    .at(span.source_begin as usize * 8)
                    .cast::<u64>()
            };
            let lane = self.descriptors[span.first].lane;
            unsafe {
                check(mgbfs_bind_owner_jobs(
                    jobs,
                    span.buckets,
                    lengths.cast(),
                    self.cfg.buckets,
                    s,
                ))?;
                check(mgbfs_bounded_owner_compare(
                    owner_plan,
                    jobs,
                    span.buckets,
                    span.rows,
                    hashes.cast(),
                    self.prev.ptr,
                    u64::from(self.prev_count),
                    self.curr.ptr,
                    u64::from(self.current_count),
                    accepted,
                    lengths.cast(),
                    self.cfg.buckets,
                    self.cfg.buckets / self.cfg.shards,
                    lane,
                    self.depth,
                    counts.cast(),
                    self.control.ptr.cast(),
                    s,
                ))?;
                if let Some(h) = self.hash_first.as_ref() {
                    check(cudaStreamSynchronize(s))?;
                    let control = self.control.one::<Control>()?;
                    if control.error != 0 {
                        return Err(format!("HASH_FIRST_OWNER_COMPARE_{}", control.error));
                    }
                    if u64::from(h.pending_counts[source_group]) + u64::from(control.survivors)
                        > u64::from(h.capacity)
                    {
                        return Err("HASH_FIRST_REQUEST_CAPACITY".into());
                    }
                }
                check(mgbfs_state_reserve_layer(
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    self.extent.ptr.cast(),
                    self.layer_count.ptr.cast(),
                    self.cfg.layer_capacity,
                    s,
                ))?;
                let extent = self.extent.ptr.cast::<Extent>();
                check(mgbfs_bounded_owner_commit(
                    owner_plan,
                    jobs,
                    span.buckets,
                    hashes.cast(),
                    accepted,
                    lengths.cast(),
                    counts.cast(),
                    self.control.ptr.cast(),
                    std::ptr::addr_of!((*extent).granted_rows),
                    selected.cast(),
                    s,
                ))?;
                if let Some(h) = self.hash_first.as_ref() {
                    let offset = h.pending_counts[source_group] as usize;
                    check(mgbfs_state_build_requests(
                        source_states.cast(),
                        rows,
                        refs,
                        span.rows,
                        selected.cast(),
                        self.candidates,
                        h.requests[source_group].at(offset * 16).cast(),
                        h.targets[source_group].at(offset * 8).cast(),
                        h.count.ptr.cast(),
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                        extent,
                        s,
                    ))?;
                } else {
                    check(mgbfs_state_materialize_packed(
                        source_states.add(span.source_begin as usize * self.stride),
                        span.rows,
                        selected.cast(),
                        self.candidates,
                        self.stride as u32,
                        self.states.ptr.cast(),
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                        extent,
                        s,
                    ))?;
                    // All readers of these job descriptors have finished in
                    // this stream. Reuse only this span's first 64-byte slot
                    // for its extent; later spans occupy disjoint slots.
                    check(cudaMemcpyAsync(
                        jobs.cast(),
                        self.extent.ptr,
                        std::mem::size_of::<Extent>(),
                        3,
                        s,
                    ))?;
                    continue;
                }
                check(cudaStreamSynchronize(s))?;
            }
            let control = self.control.one::<Control>()?;
            if control.error != 0 {
                let ring = self.ring.one::<Ring>()?;
                return Err(format!(
                    "NATIVE_OWNER_FATAL_{} rank={} depth={} ring_head={} ring_tail={} ring_capacity={} requested_survivors={} descriptor_head={} descriptor_tail={}",
                    control.error, self.cfg.rank, self.depth, ring.head, ring.tail,
                    ring.capacity, control.survivors, ring.descriptor_head, ring.descriptor_tail
                ));
            }
            let extent = self.extent.one::<Extent>()?;
            if let Some(h) = self.hash_first.as_mut() {
                if extent.ready != 0 || u64::from(h.count.one::<u32>()?) != extent.count {
                    return Err("HASH_FIRST_REQUEST_PUBLICATION".into());
                }
                h.pending_counts[source_group] += extent.count as u32;
                append_extent(&mut h.pending_extents[source_group], extent)?;
                continue;
            }
        }
        if self.hash_first.is_none() {
            // Every owner error is propagated into the sticky ring fatal by
            // reserve/materialize before the next job can commit. Never
            // publish captured results when any job failed.
            check(unsafe { cudaStreamSynchronize(s) })?;
            let ring = self.ring.one::<Ring>()?;
            if ring.fatal != 0 {
                return Err(format!(
                    "NATIVE_OWNER_FATAL_{} rank={} depth={} ring_head={} ring_tail={} ring_capacity={} descriptor_head={} descriptor_tail={}",
                    ring.fatal, self.cfg.rank, self.depth, ring.head, ring.tail,
                    ring.capacity, ring.descriptor_head, ring.descriptor_tail
                ));
            }
            self.jobs_gpu
                .read(&mut self.dense_results[..descriptor_count])?;
            crate::owner_results::append_dense_results(
                &self.dense_results[..descriptor_count],
                &self.spans[..span_count],
                &mut self.next,
            )?;
        }
        Ok(())
    }
    fn materialize_hash_first(
        &mut self,
        parent: Option<Extent>,
        offset: u64,
        parents: u32,
        round: u32,
    ) -> Result<()> {
        if self.hash_first.as_ref().is_some_and(|h| h.device.is_some()) {
            return self.materialize_hash_first_device(parent, offset, parents, round);
        }
        use crate::hash_first_exchange::{enqueue_round_trip, ExchangeBuffers, MatrixSource};
        let s = self.stream.0;
        let h = self.hash_first.as_ref().ok_or("HASH_FIRST_STORAGE")?;
        let (begin, physical) = parent
            .map(|e| (e.sequence + offset, e.begin + offset))
            .unwrap_or((0, 0));
        let source = MatrixSource {
            n: h.n,
            moves: self.moves,
            modulus: h.modulus,
            stride: self.stride as u32,
            rank: self.cfg.rank,
            world: self.cfg.world,
            parent_begin: begin,
            parent_count: parents,
            parents: unsafe { self.states.at(physical as usize * self.stride).cast() },
            generators: h.generators.ptr.cast(),
        };
        // Preserve reservation order (local source, then remote source), so
        // final extents remain a FIFO StateRing frontier. Every rank enters
        // both groups on two ranks even with no requests; one rank has only
        // the local group and must not issue a call to nonexistent peer 1.
        // Local plus first remote source, then reuse the remote request slot
        // for one peer at a time. Keep parents alive until all rounds finish.
        for group in usize::from(round > 1)..self.cfg.world.min(2) as usize {
            let count = h.pending_counts[group];
            h.count.put_u32(count)?;
            check(unsafe {
                mgbfs_materialize_sort_origins(
                    h.materialize.0,
                    if group == 0 {
                        self.cfg.rank
                    } else {
                        self.cfg.rank ^ round
                    },
                    h.requests[group].ptr.cast(),
                    h.targets[group].ptr.cast(),
                    h.count.ptr.cast(),
                    h.sorted_requests.ptr.cast(),
                    h.sorted_targets.ptr.cast(),
                    h.local_fatal.ptr.cast(),
                    s,
                )
            })?;
            check(unsafe { cudaStreamSynchronize(s) })?;
            if self.all_max(h.local_fatal.one::<u32>()?)? != 0 {
                return Err("HASH_FIRST_REQUEST_SORT_FATAL".into());
            }
            let responses = if group == 0 {
                check(unsafe {
                    mgbfs_regenerate_selected(
                        h.n,
                        self.moves,
                        h.modulus,
                        self.stride as u32,
                        h.capacity,
                        self.cfg.rank,
                        begin,
                        parents,
                        source.parents,
                        source.generators,
                        h.sorted_requests.ptr.cast(),
                        h.count.ptr.cast(),
                        h.outgoing_responses.ptr.cast(),
                        h.local_fatal.ptr.cast(),
                        s,
                    )
                })?;
                check(unsafe { cudaStreamSynchronize(s) })?;
                let fatal = self.all_max(h.local_fatal.one::<u32>()?)?;
                h.group_fatal.put_u32(fatal)?;
                h.outgoing_responses.ptr
            } else {
                self.collective_send.put_u32(count)?;
                check(unsafe {
                    mgbfs_nccl_send_recv(
                        self.comm.0,
                        self.collective_send.ptr,
                        4,
                        self.cfg.rank ^ round,
                        self.recv_count.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr,
                        4,
                        s,
                    )
                })?;
                check(unsafe { cudaStreamSynchronize(s) })?;
                let received = self.recv_count.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                    .one::<u32>()?;
                if self.all_max(u32::from(received > h.capacity))? != 0 {
                    return Err("HASH_FIRST_REMOTE_REQUEST_CAPACITY".into());
                }
                unsafe {
                    enqueue_round_trip(
                        self.comm.0,
                        self.cfg.rank ^ round,
                        &source,
                        &ExchangeBuffers {
                            capacity: h.capacity,
                            outgoing_count: count,
                            incoming_count: received,
                            incoming_count_device: self.recv_count.as_ref()
                                .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr.cast(),
                            outgoing_requests: h.sorted_requests.ptr.cast(),
                            incoming_requests: h.received_requests.ptr.cast(),
                            outgoing_responses: h.outgoing_responses.ptr.cast(),
                            incoming_responses: h.incoming_responses.ptr.cast(),
                            local_fatal: h.local_fatal.ptr.cast(),
                            group_fatal: h.group_fatal.ptr.cast(),
                        },
                        s,
                    )?;
                }
                check(unsafe { cudaStreamSynchronize(s) })?;
                h.incoming_responses.ptr
            };
            if h.group_fatal.one::<u32>()? != 0 {
                return Err("HASH_FIRST_RESPONSE_GROUP_FATAL".into());
            }
            // Requests were ordered by parent. Restore final target order via
            // the shared radix scratch, then write each contiguous ring extent.
            // The only host metadata are at most two physical runs per source.
            let publication = (|| -> Result<()> {
                let mut sorted_offset = 0u32;
                for &extent in &h.pending_extents[group] {
                    let rows =
                        u32::try_from(extent.count).map_err(|_| "HASH_FIRST_EXTENT_COUNT")?;
                    self.control.put(&[Control {
                        stage: 2,
                        survivors: rows,
                        ..Control::default()
                    }])?;
                    self.extent.put(&[extent])?;
                    check(unsafe {
                        mgbfs_state_apply_response_span(
                            h.materialize.0,
                            responses.cast(),
                            h.sorted_targets.ptr.cast(),
                            h.count.ptr.cast(),
                            sorted_offset,
                            h.group_fatal.ptr.cast(),
                            self.states.ptr.cast(),
                            self.ring.ptr.cast(),
                            self.control.ptr.cast(),
                            self.extent.ptr.cast(),
                            s,
                        )
                    })?;
                    check(unsafe { cudaStreamSynchronize(s) })?;
                    let control = self.control.one::<Control>()?;
                    let ready = self.extent.one::<Extent>()?;
                    if control.error != 0 || ready.ready != 1 {
                        return Err(format!("HASH_FIRST_PUBLICATION_FATAL_{}", control.error));
                    }
                    append_extent(&mut self.next, ready)?;
                    sorted_offset = sorted_offset
                        .checked_add(rows)
                        .ok_or("HASH_FIRST_RESPONSE_COUNT")?;
                }
                if sorted_offset != count {
                    return Err("HASH_FIRST_RESPONSE_COVERAGE".into());
                }
                Ok(())
            })()
            .err();
            if self.all_max(u32::from(publication.is_some()))? != 0 {
                return Err(
                    publication.unwrap_or_else(|| "REMOTE_HASH_FIRST_PUBLICATION_FATAL".into())
                );
            }
        }
        let h = self.hash_first.as_mut().unwrap();
        h.pending_counts = [0; 2];
        for extents in &mut h.pending_extents {
            extents.clear();
        }
        Ok(())
    }
    /// Same bounded epoch protocol as DENSE; request counts, committed extents
    /// and response publication stay device resident until FinalizeDepth.
    fn materialize_hash_first_device(&mut self, parent: Option<Extent>, offset: u64,
        parents: u32, round: u32) -> Result<()> {
        let h = self.hash_first.as_ref().ok_or("HASH_FIRST_STORAGE")?;
        let d = h.device.as_ref().ok_or("HASH_FIRST_DEVICE_STORAGE")?;
        let view = self.lsa_view.ok_or("HASH_FIRST_LSA_STORAGE")?;
        let s = self.stream.0;
        let (begin, physical) = parent.map(|e| (e.sequence + offset, e.begin + offset))
            .unwrap_or((0, 0));
        // The scheduler retains one local owner round with world=1. There is
        // no remote source or receive lease in that case.
        let peer = if self.cfg.world == 1 { self.cfg.rank } else { self.cfg.rank ^ round };
        let logical_peer = self.cfg.logical_owner_to_rank.iter()
            .position(|&rank| rank == peer).ok_or("OWNER_MAP")?;
        let gate = || -> Result<()> { check(unsafe { mgbfs_owner_lsa_fatal_gate(
            self.comm.0, self.ring.ptr.cast(), self.control.ptr.cast(),
            self.collective_send.ptr.cast(), self.collective_recv.ptr.cast(), s,
        ) }) };
        let import = |word| -> Result<()> { check(unsafe { mgbfs_owner_import_transport_fatal(
            word, self.ring.ptr.cast(), self.control.ptr.cast(), s,
        ) }) };
        let peer_counts = |count: *const u32| -> Result<()> {
            unsafe {
                check(cudaMemsetAsync(d.exchange_counts.ptr, 0, self.cfg.world as usize * 4, s))?;
                check(cudaMemcpyAsync(d.exchange_counts.at(logical_peer * 4), count.cast(), 4, 3, s))
            }
        };
        for group in usize::from(round > 1)..self.cfg.world.min(2) as usize {
            let count = unsafe { d.counts.at(group * 4).cast::<u32>() };
            let extent = unsafe { d.extents.at(group * std::mem::size_of::<Extent>()).cast::<Extent>() };
            let control = unsafe { d.controls.at(group * std::mem::size_of::<Control>()).cast::<Control>() };
            unsafe {
                check(mgbfs_materialize_sort_origins(h.materialize.0,
                    if group == 0 { self.cfg.rank } else { peer },
                    h.requests[group].ptr.cast(), h.targets[group].ptr.cast(), count,
                    h.sorted_requests.ptr.cast(), h.sorted_targets.ptr.cast(),
                    h.local_fatal.ptr.cast(), s))?;
            }
            import(h.local_fatal.ptr.cast())?;
            gate()?;
            let origins = if group == 0 { h.sorted_requests.ptr.cast() } else {
                peer_counts(count)?;
                check(unsafe { mgbfs_nccl_lsa_exchange_rows(self.comm.0, std::ptr::null(),
                    h.sorted_requests.ptr, d.exchange_counts.ptr.cast(),
                    self.collective_recv.ptr.cast(), logical_peer as u32, peer, 16, s) })?;
                import(view.fatal)?;
                gate()?;
                view.states.cast()
            };
            let regenerate_count = if group == 0 { count as *const u32 } else { view.count };
            unsafe {
                check(mgbfs_regenerate_selected(h.n, self.moves, h.modulus,
                    self.stride as u32, h.capacity, self.cfg.rank, begin, parents,
                    self.states.at(physical as usize * self.stride).cast(),
                    h.generators.ptr.cast(), origins, regenerate_count,
                    h.outgoing_responses.ptr.cast(), h.local_fatal.ptr.cast(), s))?;
            }
            import(h.local_fatal.ptr.cast())?;
            gate()?;
            let responses = if group == 0 { h.outgoing_responses.ptr as *const u8 } else {
                // Snapshot the request count into the disjoint preallocated
                // count array before the response exchange overwrites the view.
                peer_counts(view.count)?;
                check(unsafe { mgbfs_nccl_lsa_exchange_rows(self.comm.0, std::ptr::null(),
                    h.outgoing_responses.ptr, d.exchange_counts.ptr.cast(),
                    self.collective_recv.ptr.cast(), logical_peer as u32, peer,
                    self.stride as u32, s) })?;
                import(view.fatal)?;
                check(unsafe { mgbfs_state_validate_response_count(count, view.count,
                    self.ring.ptr.cast(), self.control.ptr.cast(), s) })?;
                gate()?;
                view.states
            };
            unsafe {
                check(mgbfs_state_apply_responses(h.materialize.0, responses,
                    h.sorted_targets.ptr.cast(), count, self.collective_recv.ptr.cast(),
                    self.states.ptr.cast(), self.ring.ptr.cast(), control, extent, s))?;
                check(mgbfs_state_publish_next_extent(self.ring.ptr.cast(), control, extent,
                    self.next_extent_count.as_ref().ok_or("NEXT_EXTENT_COUNT_MISSING")?.ptr.cast(),
                    self.next_extents.as_ref().ok_or("NEXT_EXTENTS_MISSING")?.ptr.cast(), 2, s))?;
            }
            gate()?;
        }
        Ok(())
    }
    pub fn advance(&mut self) -> Result<bool> {
        if self.failed {
            return Err("DISTRIBUTED_FAILED".into());
        }
        let result = self.advance_inner(None);
        if let Err(error) = &result {
            // Publish the originating error before abort/Drop can block or a
            // fail-fast launcher terminates this rank after its peer exits.
            eprintln!("MGBFS_RUNTIME_FATAL rank={} error={error}", self.cfg.rank);
        }
        let comm = self.comm.0;
        let failure_report = self.failure_report.as_deref();
        crate::failure::abort_on_error(result, &mut self.failed, || unsafe {
            if std::env::var_os("MGBFS_TRACE_ROUTE").is_some() {
                eprintln!("MGBFS_ROUTE_TRACE rank={} stage=abort_begin", self.cfg.rank);
            }
            if let Some(token) = failure_report {
                token.store(2, std::sync::atomic::Ordering::Release);
            }
            mgbfs_nccl_abort(comm);
            if std::env::var_os("MGBFS_TRACE_ROUTE").is_some() {
                eprintln!("MGBFS_ROUTE_TRACE rank={} stage=abort_end", self.cfg.rank);
            }
        })
    }
    /// Archive each bounded parent slice before retiring its StateRing range.
    pub fn advance_archived(
        &mut self,
        archive: &mut crate::pinned_archive::PinnedArchive,
    ) -> Result<bool> {
        if self.failed {
            return Err("DISTRIBUTED_FAILED".into());
        }
        let result = self.advance_inner(Some(archive));
        if let Err(error) = &result {
            eprintln!("MGBFS_RUNTIME_FATAL rank={} error={error}", self.cfg.rank);
        }
        let comm = self.comm.0;
        let failure_report = self.failure_report.as_deref();
        crate::failure::abort_on_error(result, &mut self.failed, || unsafe {
            if let Some(token) = failure_report {
                token.store(2, std::sync::atomic::Ordering::Release);
            }
            mgbfs_nccl_abort(comm);
        })
    }
    fn advance_inner(
        &mut self,
        mut archive: Option<&mut crate::pinned_archive::PinnedArchive>,
    ) -> Result<bool> {
        self.ensure_not_cancelled()?;
        // Diagnostic only: bracket the CUB route, pack, exchange and owner
        // phases without adding synchronizations to an ordinary run.
        let trace_route = std::env::var_os("MGBFS_TRACE_ROUTE").is_some();
        let trace_sync = trace_route && std::env::var_os("MGBFS_TRACE_ROUTE_NO_SYNC").is_none();
        let trace_ranges = std::env::var_os("MGBFS_TRACE_RANGES").is_some();
        if trace_ranges && unsafe { mgbfs_trace_ranges_available() } == 0 {
            return Err("PROFILE_NVTX_NOT_BUILT".into());
        }
        let mut batch_index = 0u64;
        if let Some(a) = archive.as_ref() {
            let error = if self.archived_depth == Some(self.depth) {
                Some("ARCHIVE_DEPTH_ALREADY_SUBMITTED".to_string())
            } else if a.width != self.width && self.permutation_n != u32::try_from(a.width).ok() {
                Some("ARCHIVE_STATE_WIDTH".to_string())
            } else {
                None
            };
            if self.all_max(u32::from(error.is_some()))? != 0 {
                return Err(error.unwrap_or_else(|| "REMOTE_ARCHIVE_FATAL".into()));
            }
        }
        let s = self.stream.0;
        if let Some(rank) = self.native_rank.as_ref() {
            // Depth-boundary upload only; no batch depends on a directory readback.
            rank.previous.put(&self.prev_dir)?;
            rank.current.put(&self.curr_dir)?;
        }
        unsafe {
            if let Some(owner) = self.owner.as_ref() {
                check(cudaMemsetAsync(
                    owner.lengths.ptr,
                    0,
                    self.cfg.buckets as usize * 4,
                    s,
                ))?;
            }
            check(cudaMemsetAsync(self.layer_count.ptr, 0, 4, s))?;
            if self.rank_owner_mode() {
                check(cudaMemsetAsync(
                    self.next_extent_count.as_ref().ok_or("NEXT_EXTENT_COUNT_MISSING")?.ptr,
                    0, 4, s,
                ))?;
            }
        }
        #[cfg(feature = "library-owner")]
        if let Some(library) = self.library_owner.as_mut() {
            if library.closed {
                if library.rank_mode {
                    library.rank = unsafe { create_rank_owner(
                        library, &self.prev, &self.curr, self.cfg.shards,
                        self.candidates, self.cfg.world, s,
                    )? };
                } else {
                    for shard in 0..library.shards.len() {
                        unsafe {
                            library.shards[shard].reopen_window(
                                history_view(&self.prev, library.plane_words, &library.previous[shard]),
                                history_view(&self.curr, library.plane_words, &library.current[shard]),
                                library.capacity,
                            )?;
                        }
                    }
                }
                library.closed = false;
            }
        }
        self.next.clear();
        // Frontier extents are fixed at this depth boundary. Agree once on
        // the maximum physical-batch count, then every rank issues the same
        // number of peer epochs; exhausted ranks send zero payloads. This
        // removes the host-observed `more` collective after every batch.
        let local_rounds = ParentCursor::round_count(&self.front, self.cfg.batch);
        if self.all_max(u32::from(local_rounds.is_err()))? != 0 {
            return Err(local_rounds
                .err()
                .unwrap_or_else(|| "REMOTE_PARENT_SCHEDULE_FATAL".into()));
        }
        let scheduled_rounds = self.all_max(local_rounds?)?;
        let mut cursor = ParentCursor::default();
        let mut prefetched: Option<(ParentBatch, u64)> = None;
        let mut archive_released = [false; 2];
        let mut lsa_owner_recorded = false;
        let device_epoch = self.lsa_view.is_some() && self.rank_owner_mode();
        // The host-sized depth schedule above reuses collective_recv for a
        // nonzero round count. LSA reads this word as its device group-fatal
        // predicate before the first owner vote, so start each depth clean.
        if device_epoch {
            check(unsafe { cudaMemsetAsync(self.collective_recv.ptr, 0, 4, s) })?;
        }
        let mut epoch_serial = 0usize;
        for _ in 0..scheduled_rounds {
            let _batch_range = TraceRange::new(trace_ranges, b"mgbfs.batch\0");
            self.ensure_not_cancelled()?;
            if device_epoch && self.epoch_outstanding.len() == self.epoch_completed.len() {
                let slot = *self.epoch_outstanding.front().ok_or("EPOCH_CREDIT_EMPTY")?;
                self.wait_epoch_credit(slot)?;
                self.epoch_outstanding.pop_front();
            }
            let work = cursor.take(&self.front, self.cfg.batch)?;
            let extent_index = work.map(|b| b.extent).unwrap_or(0);
            let extent_offset = work.map(|b| b.offset).unwrap_or(0);
            let parent = work.map(|b| self.front[b.extent]);
            let parents = work.map(|b| b.count).unwrap_or(0);
            let next_work = cursor.peek(&self.front, self.cfg.batch)?;
            let mut generation = None;
            let candidate_count = parents * self.moves;
            if trace_route {
                eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=batch_begin parents={parents} candidates={candidate_count}", self.cfg.rank, self.depth);
            }
            if let Some(a) = archive.as_deref_mut() {
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=archive_begin", self.cfg.rank, self.depth);
                }
                let error = if let Some(extent) = parent {
                    self.archive_range(
                        a,
                        extent.begin + extent_offset,
                        u64::from(parents),
                        extent_index,
                    )
                    .err()
                } else {
                    None
                };
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=archive_enqueued error={}", self.cfg.rank, self.depth, error.is_some());
                }
                // Both ranks issue this epoch even when one has no parents.
                // A local archive failure must not strand its peer in exchange.
                if device_epoch {
                    // Host/API failure may prevent any further GPU submission.
                    // Return to the sole dispatcher's notification + abort path
                    // before issuing a collective or waiting for GPU progress.
                    if let Some(error) = error { return Err(error); }
                    check(unsafe { mgbfs_owner_lsa_fatal_gate(
                        self.comm.0, self.ring.ptr.cast(), self.control.ptr.cast(),
                        self.collective_send.ptr.cast(), self.collective_recv.ptr.cast(), s,
                    ) })?;
                    if trace_route {
                        eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=archive_vote_queued", self.cfg.rank, self.depth);
                    }
                } else if self.all_max(u32::from(error.is_some()))? != 0 {
                    return Err(error.unwrap_or_else(|| "REMOTE_ARCHIVE_FATAL".into()));
                }
            }
            if trace_route {
                eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=generate_begin", self.cfg.rank, self.depth);
            }
            if self.hash_first.as_ref().is_some_and(|h| h.device.is_some()) && work.is_some() {
                let batch = work.ok_or("GENERATION_BATCH_MISSING")?;
                let sequence = match prefetched.take() {
                    Some((expected, sequence)) if expected == batch => sequence,
                    Some(_) => return Err("GENERATION_BATCH_IDENTITY".into()),
                    None => self.enqueue_frontier_generation(batch)?,
                };
                unsafe { self.generation_done.wait(sequence, s)?; }
                generation = Some(sequence);
                let fatal = self.hash_first.as_ref().and_then(|h| h.device.as_ref())
                    .ok_or("HASH_FIRST_DEVICE_STORAGE")?.generation_control.ptr.cast();
                check(unsafe { mgbfs_owner_import_transport_fatal(
                    fatal, self.ring.ptr.cast(), self.control.ptr.cast(), s,
                ) })?;
                check(unsafe { mgbfs_owner_lsa_fatal_gate(
                    self.comm.0, self.ring.ptr.cast(), self.control.ptr.cast(),
                    self.collective_send.ptr.cast(), self.collective_recv.ptr.cast(), s,
                ) })?;
            } else if let Some(h) = self.hash_first.as_ref() {
                h.parent_count.put_u32(parents)?;
                let (begin, physical) = parent
                    .map(|e| (e.sequence + extent_offset, e.begin + extent_offset))
                    .unwrap_or((0, 0));
                unsafe {
                    let generate = if self.hash_first_tensor_generation {
                        mgbfs_generate_hash_only_tc_admitted
                    } else {
                        mgbfs_generate_hash_only
                    };
                    check(generate(
                        h.n,
                        self.moves,
                        h.modulus,
                        self.stride as u32,
                        self.cfg.batch,
                        self.candidates,
                        self.cfg.rank,
                        begin,
                        self.states.at(physical as usize * self.stride).cast(),
                        h.generators.ptr.cast(),
                        h.coefficients.ptr.cast(),
                        h.offsets.ptr.cast(),
                        h.parent_count.ptr.cast(),
                        self.child_hashes.ptr.cast(),
                        self.children.ptr.cast(),
                        self.route_count.ptr.cast(),
                        h.local_fatal.ptr.cast(),
                        s,
                    ))?;
                    if !device_epoch {
                        check(cudaStreamSynchronize(s))?;
                    }
                }
                if device_epoch {
                    check(unsafe { mgbfs_owner_import_transport_fatal(
                        h.local_fatal.ptr.cast(), self.ring.ptr.cast(), self.control.ptr.cast(), s,
                    ) })?;
                    check(unsafe { mgbfs_owner_lsa_fatal_gate(
                        self.comm.0, self.ring.ptr.cast(), self.control.ptr.cast(),
                        self.collective_send.ptr.cast(), self.collective_recv.ptr.cast(), s,
                    ) })?;
                } else if self.all_max(h.local_fatal.one::<u32>()?)? != 0 {
                    return Err("HASH_FIRST_GENERATION_FATAL".into());
                }
            } else if let Some(batch) = work {
                let sequence = match prefetched.take() {
                    Some((expected, sequence)) if expected == batch => sequence,
                    Some(_) => return Err("GENERATION_BATCH_IDENTITY".into()),
                    None => self.enqueue_frontier_generation(batch)?,
                };
                unsafe {
                    self.generation_done.wait(sequence, s)?;
                }
                generation = Some(sequence);
            }
            if trace_sync {
                check(unsafe { cudaStreamSynchronize(s) })?;
            }
            if trace_route {
                eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=generation_end", self.cfg.rank, self.depth);
            }
            unsafe {
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=route_begin", self.cfg.rank, self.depth);
                }
                check(mgbfs_route_run(
                    self.route.0,
                    self.child_hashes.ptr,
                    self.identity_refs.ptr.cast(),
                    self.sorted_hashes.ptr,
                    self.sorted_refs.ptr.cast(),
                    self.route_count.ptr.cast(),
                    candidate_count,
                    self.cfg.prededup as i32,
                    s,
                ))?;
            }
            let packet_stride = if self.hash_first.is_some() {
                16
            } else {
                self.stride
            };
            check(unsafe {
                mgbfs_exchange_pack_device_n(
                    self.cfg.world,
                    packet_stride as u32,
                    self.candidates,
                    self.children.ptr.cast(),
                    candidate_count,
                    self.sorted_hashes.ptr,
                    self.sorted_refs.ptr.cast(),
                    self.route_count.ptr.cast(),
                    self.packed_states.ptr.cast(),
                    self.owner_counts.ptr.cast(),
                    s,
                )
            })?;
            if self.rank_owner_mode() {
                let window = self.owner_window.as_ref().ok_or("OWNER_WINDOW_MISSING")?;
                let logical_owner = self.cfg.logical_owner_to_rank.iter()
                    .position(|&rank| rank == self.cfg.rank).ok_or("OWNER_MAP")? as u32;
                check(unsafe {
                    mgbfs_owner_window_from_counts(
                        self.cfg.world, self.candidates, logical_owner,
                        self.owner_counts.ptr.cast(), self.route_count.ptr.cast(),
                        window.ptr.cast(), window.at(4).cast(), s,
                    )
                })?;
            }
            if self.lsa_view.is_some() {
                check(unsafe { cudaEventRecord(self.pack_done.0, s) })?;
            } else {
                self.wait_comm_stream(s)?;
            }
            let host_ranges = if self.lsa_view.is_none() {
                let routed = self.route_count.one::<u32>()?;
                let mut owner_counts = [0u32; 8];
                self.owner_counts.read(&mut owner_counts[..self.cfg.world as usize])?;
                if crate::route_count::packed_count(
                    candidate_count,
                    &owner_counts[..self.cfg.world as usize],
                )? != routed {
                    return Err("EXCHANGE_COUNT_MISMATCH".into());
                }
                Some((routed, crate::route_count::packed_rank_ranges(
                    self.candidates,
                    &owner_counts[..self.cfg.world as usize],
                    &self.cfg.logical_owner_to_rank[..self.cfg.world as usize],
                )?))
            } else {
                // LSA consumes the device counts and checks their sum/capacity
                // before copying. No D2H count is needed for this route.
                None
            };
            if trace_route {
                eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=route_end routed={:?}", self.cfg.rank, self.depth, host_ranges.as_ref().map(|(rows, _)| rows));
                eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=pack_end", self.cfg.rank, self.depth);
            }
            if let Some(sequence) = generation {
                // Reuse children/child_hashes only after pack has finished.
                // LSA keeps this dependency entirely on GPU; legacy NCCL
                // retains the host-observed completion path.
                if self.lsa_view.is_some() {
                    unsafe {
                        self.generation_done.retire_after_device_barrier(
                            sequence, self.generation_stream.0, self.pack_done.0,
                        )?;
                    }
                } else {
                    if !self.generation_done.poll(sequence)? {
                        return Err("GENERATION_PACK_ORDER".into());
                    }
                    self.generation_done.retire(sequence)?;
                }
                if let Some(next) = next_work {
                    let sequence = self.enqueue_frontier_generation(next)?;
                    prefetched = Some((next, sequence));
                    self.dense_lookahead = self
                        .dense_lookahead
                        .checked_add(1)
                        .ok_or("GENERATION_COUNTER_OVERFLOW")?;
                }
            }
            let world = self.cfg.world;
            let (local_offset, local_rows) = host_ranges.as_ref()
                .map_or((0, 0), |(_, ranges)| ranges[self.cfg.rank as usize]);
            // The same bounded receive slot serves every XOR peer round.
            // All ranks enter even when their parent batch or payload is empty.
            for round in 1..world.max(2) {
                self.ensure_not_cancelled()?;
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=exchange_begin", self.cfg.rank, self.depth);
                }
                let exchange_peer = if world == 1 {
                    0
                } else {
                    crate::route_count::exchange_peer(world, self.cfg.rank, round)?
                };
                let (remote_offset, remote_rows) = host_ranges.as_ref()
                    .map_or((0, 0), |(_, ranges)| ranges[exchange_peer as usize]);
                let lsa = self.lsa_view;
                let rank_mode = self.rank_owner_mode();
                let received = if self.cfg.world == 1 {
                    0
                } else if lsa.is_some() {
                    let logical_owner = self.cfg.logical_owner_to_rank
                        .iter().position(|&rank| rank == exchange_peer)
                        .ok_or("OWNER_MAP")? as u32;
                    check(unsafe { cudaStreamWaitEvent(
                        self.exchange_stream.0, self.pack_done.0, 0,
                    ) })?;
                    if lsa_owner_recorded {
                        check(unsafe { cudaStreamWaitEvent(
                            self.exchange_stream.0,
                            self.owner_consumed.as_ref().ok_or("LSA_OWNER_EVENT_MISSING")?.0,
                            0,
                        ) })?;
                    }
                    check(unsafe {
                        mgbfs_nccl_lsa_exchange_rows(
                            self.comm.0, self.sorted_hashes.ptr,
                            self.packed_states.ptr, self.owner_counts.ptr.cast(),
                            self.collective_recv.ptr.cast(),
                            logical_owner, exchange_peer, packet_stride as u32,
                            self.exchange_stream.0,
                        )
                    })?;
                    check(unsafe { cudaEventRecord(self.exchange_done.0, self.exchange_stream.0) })?;
                    if trace_route {
                        eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=lsa_exchange_queued", self.cfg.rank, self.depth);
                    }
                    // The row count remains on device; no host-sized payload
                    // handshake or readback is needed for this peer round.
                    0
                } else {
                    let communication = self.exchange_stream.0;
                    // The count is a launch argument, not a borrowed host
                    // slice. Publish it on the consuming stream so the NCCL
                    // size exchange follows the store without a host drain.
                    check(unsafe { mgbfs_device_store_u32(
                        self.collective_send.ptr.cast(), remote_rows, communication,
                    ) })?;
                    check(unsafe {
                        mgbfs_nccl_send_recv(
                            self.comm.0,
                            self.collective_send.ptr,
                            4,
                            exchange_peer,
                            self.recv_count.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr,
                            4,
                            communication,
                        )
                    })?;
                    self.wait_comm_stream(communication)?;
                    let received = self.recv_count.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                        .one::<u32>()?;
                    if received > self.candidates {
                        return Err("EXCHANGE_CAPACITY".into());
                    }
                    check(unsafe {
                        mgbfs_nccl_send_recv(
                            self.comm.0,
                            self.sorted_hashes.at(remote_offset as usize * 16),
                            u64::from(remote_rows) * 16,
                            exchange_peer,
                            self.recv_hashes.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr,
                            u64::from(received) * 16,
                            communication,
                        )
                    })?;
                    check(unsafe {
                        mgbfs_nccl_send_recv(
                            self.comm.0,
                            self.packed_states
                                .at(remote_offset as usize * packet_stride),
                            u64::from(remote_rows) * packet_stride as u64,
                            exchange_peer,
                            self.recv_states.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr,
                            u64::from(received) * packet_stride as u64,
                            communication,
                        )
                    })?;
                    check(unsafe { cudaEventRecord(self.exchange_done.0, communication) })?;
                    received
                };
                if let Some(parent_extent) =
                    parent.filter(|_| round == 1 && self.hash_first.is_none())
                {
                    if archive.is_some()
                        || (self.archived_depth == Some(self.depth)
                            && !archive_released[extent_index])
                    {
                        check(unsafe {
                            cudaStreamWaitEvent(s, self.archive_done[extent_index].0, 0)
                        })?;
                        archive_released[extent_index] = true;
                    }
                    let mut live = parent_extent;
                    live.sequence += extent_offset;
                    live.begin = live.sequence % u64::from(self.cfg.state_ring_capacity);
                    live.count -= extent_offset;
                    live.granted_rows = live.count as u32;
                    check(unsafe {
                        mgbfs_state_retire_dense_prefix_value(
                            self.ring.ptr.cast(),
                            live,
                            u64::from(parents),
                            s,
                        )
                    })?;
                }
                if round == 1 && self.hash_first.is_none() {
                    if world > 1 {
                        // The vote must follow this round's P2P on the same
                        // communicator, including zero-payload exchange.
                        check(unsafe { cudaStreamWaitEvent(s, self.exchange_done.0, 0) })?;
                    }
                    if let Some(view) = lsa {
                        check(unsafe { mgbfs_owner_import_transport_fatal(
                            view.fatal, self.ring.ptr.cast(), self.control.ptr.cast(), s,
                        ) })?;
                    }
                    if trace_route {
                        eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=retire_import_queued", self.cfg.rank, self.depth);
                    }
                    if device_epoch {
                        // The common result poisons ring/control on-device before
                        // any rank can commit this owner epoch. Host/API errors
                        // use the cancellation sideband instead of a batch vote.
                        check(unsafe { mgbfs_owner_lsa_fatal_gate(
                            self.comm.0, self.ring.ptr.cast(), self.control.ptr.cast(),
                            self.collective_send.ptr.cast(), self.collective_recv.ptr.cast(), s,
                        ) })?;
                        if trace_route {
                            eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=preowner_vote_queued", self.cfg.rank, self.depth);
                        }
                    } else if self.all_max_ring_fatal()? != 0 {
                        return Err("GROUP_STATE_RING_RETIRE_FATAL".into());
                    }
                }
                if round != 1 || self.hash_first.is_some() {
                    if let Some(view) = lsa {
                        check(unsafe { cudaStreamWaitEvent(s, self.exchange_done.0, 0) })?;
                        check(unsafe { mgbfs_owner_import_transport_fatal(
                            view.fatal, self.ring.ptr.cast(), self.control.ptr.cast(), s,
                        ) })?;
                        if device_epoch {
                            check(unsafe { mgbfs_owner_lsa_fatal_gate(
                                self.comm.0, self.ring.ptr.cast(), self.control.ptr.cast(),
                                self.collective_send.ptr.cast(), self.collective_recv.ptr.cast(), s,
                            ) })?;
                        } else if self.all_max_ring_fatal()? != 0 {
                            return Err("GROUP_LSA_TRANSPORT_FATAL".into());
                        }
                    } else {
                        self.wait_comm_stream(s)?;
                    }
                }
                let local_states: *const u8 = unsafe {
                    self.packed_states
                        .at(local_offset as usize * packet_stride)
                        .cast()
                };
                let local_hashes: *const c_void = unsafe {
                    self.sorted_hashes.at(local_offset as usize * 16)
                };
                let (remote_states, remote_hashes, remote_count):
                    (*const u8, *const c_void, *const u32) = if let Some(view) = lsa {
                        (view.states, view.hashes, view.count)
                    } else {
                        (self.recv_states.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr.cast(),
                         self.recv_hashes.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr,
                         self.recv_count.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr.cast())
                    };
                let remote_ready = self.exchange_done.0;
                let world = self.cfg.world;
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=owner_begin received={received}", self.cfg.rank, self.depth);
                }
                let mut batch_error = process_owner_pair(
                    if device_epoch { OwnerFailurePolicy::CancelGroup }
                    else { OwnerFailurePolicy::CollectiveVote },
                    (
                        0,
                        (
                            local_states,
                            local_hashes,
                            if round == 1 { local_rows } else { 0 },
                        ),
                    ),
                    (
                        1,
                        (remote_states, remote_hashes, received),
                    ),
                    |(group, (states, hashes, rows))| {
                        if rank_mode {
                            if !crate::route_count::rank_owner_group_active(world, round, group)? {
                                return Ok(());
                            }
                            #[cfg(debug_assertions)]
                            if group == 1 && round == 1 && scheduled_rounds > 1
                                && TEST_OWNER_HOST_FAULT.with(|flag| flag.replace(false))
                            {
                                return Err("TEST_INJECTED_OWNER_HOST_ERROR".into());
                            }
                            let window = self.owner_window.as_ref().ok_or("OWNER_WINDOW_MISSING")?;
                            let (states, hashes, begin, rows, source_rows) = if group == 0 {
                                (self.packed_states.ptr as *const u8,
                                 self.sorted_hashes.ptr as *const c_void,
                                 window.ptr as *const u32,
                                 unsafe { window.at(4) } as *const u32,
                                 self.route_count.ptr as *const u32)
                            } else {
                                (remote_states, remote_hashes,
                                 unsafe { window.at(8) } as *const u32,
                                 remote_count, remote_count)
                            };
                            if self.native_rank.is_some() {
                                return self.commit_rank_native_batch(states, hashes, begin, rows, source_rows, group);
                            }
                            #[cfg(feature = "library-owner")]
                            return self.commit_rank_library_batch(states, hashes, begin, rows, source_rows, group);
                            #[cfg(not(feature = "library-owner"))]
                            return Err("RANK_OWNER_BACKEND_MISSING".into());
                        }
                        self.commit_owner_batch(states, hashes, rows, group)
                    },
                    || {
                        if world > 1 {
                            // Executed even after a local owner failure and for
                            // empty receive payloads. The next failure collective
                            // on s therefore cannot overtake the P2P operations.
                            check(unsafe { cudaStreamWaitEvent(s, remote_ready, 0) })?;
                        }
                        Ok(())
                    },
                )
                .err();
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=owner_end", self.cfg.rank, self.depth);
                }
                if device_epoch {
                    // A host/API error is not a device logical fatal: this rank
                    // may be unable to issue the next collective. Outer advance
                    // publishes cancellation before serialized communicator abort.
                    if let Some(error) = batch_error { return Err(error); }
                    // All ranks issue the post-owner vote, including an empty
                    // peer round. Its device result predicates the next LSA
                    // rendezvous without returning a count to the host.
                    check(unsafe { mgbfs_owner_lsa_fatal_gate(
                        self.comm.0, self.ring.ptr.cast(), self.control.ptr.cast(),
                        self.collective_send.ptr.cast(), self.collective_recv.ptr.cast(), s,
                    ) })?;
                }
                if lsa.is_some() && self.hash_first.is_none() && (batch_error.is_none() || device_epoch) {
                    // Every rank waits for its own receive-slot consumer before
                    // entering the next all-rank LSA barrier. A record failure
                    // must participate in the existing group error vote.
                    let recorded = self.owner_consumed.as_ref()
                        .ok_or_else(|| "LSA_OWNER_EVENT_MISSING".to_string())
                        .and_then(|event| check(unsafe { cudaEventRecord(event.0, s) }));
                    match recorded {
                        Ok(()) => lsa_owner_recorded = true,
                        Err(error) => batch_error = Some(error),
                    }
                }
                if device_epoch {
                    if let Some(error) = batch_error {
                        // Recording the last-reader event itself failed. Do not
                        // wait for GPU progress before notifying and aborting.
                        return Err(error);
                    }
                } else {
                    vote_group_error(
                        batch_error.map_or(Ok(()), Err),
                        |failed| Ok(self.all_max(u32::from(failed))? != 0),
                        "REMOTE_OWNER_BATCH_FATAL".into(),
                    )?;
                }
                if self.hash_first.is_some() {
                    self.materialize_hash_first(parent, extent_offset, parents, round)?;
                    if lsa.is_some() {
                        check(unsafe { cudaEventRecord(
                            self.owner_consumed.as_ref().ok_or("LSA_OWNER_EVENT_MISSING")?.0, s,
                        ) })?;
                        lsa_owner_recorded = true;
                    }
                }
            }
            if self.hash_first.is_some() {
                if let Some(parent_extent) = parent {
                    if archive.is_some()
                        || (self.archived_depth == Some(self.depth)
                            && !archive_released[extent_index])
                    {
                        check(unsafe {
                            cudaStreamWaitEvent(s, self.archive_done[extent_index].0, 0)
                        })?;
                        archive_released[extent_index] = true;
                    }
                    let mut live = parent_extent;
                    live.sequence += extent_offset;
                    live.begin = live.sequence % u64::from(self.cfg.state_ring_capacity);
                    live.count -= extent_offset;
                    live.granted_rows = live.count as u32;
                    check(unsafe {
                        mgbfs_state_retire_dense_prefix_value(
                            self.ring.ptr.cast(),
                            live,
                            u64::from(parents),
                            s,
                        )
                    })?;
                }
                if device_epoch {
                    check(unsafe { mgbfs_owner_lsa_fatal_gate(
                        self.comm.0, self.ring.ptr.cast(), self.control.ptr.cast(),
                        self.collective_send.ptr.cast(), self.collective_recv.ptr.cast(), s,
                    ) })?;
                } else if self.all_max_ring_fatal()? != 0 {
                    return Err("HASH_FIRST_RETIRE_FATAL".into());
                }
            }
            if device_epoch {
                let slot = epoch_serial % self.epoch_completed.len();
                check(unsafe { cudaEventRecord(self.epoch_completed[slot].0, s) })?;
                self.epoch_outstanding.push_back(slot);
                epoch_serial = epoch_serial.checked_add(1).ok_or("EPOCH_SEQUENCE")?;
            }
            if trace_route {
                batch_index = batch_index.checked_add(1).ok_or("TRACE_BATCH_OVERFLOW")?;
            }
        }
        let _finalize_range = TraceRange::new(trace_ranges, b"mgbfs.FinalizeDepth\0");
        while device_epoch && !self.epoch_outstanding.is_empty() {
            let slot = *self.epoch_outstanding.front().ok_or("EPOCH_CREDIT_EMPTY")?;
            self.wait_epoch_credit(slot)?;
            self.epoch_outstanding.pop_front();
        }
        if let Some(a) = archive.as_deref_mut() {
            let error = a
                .layer(u64::from(self.depth), u64::from(self.current_count))
                .err();
            if self.all_max(u32::from(error.is_some()))? != 0 {
                return Err(error.unwrap_or_else(|| "REMOTE_ARCHIVE_LAYER_FATAL".into()));
            }
            self.archived_depth = Some(self.depth);
        }
        if let Some(owner) = self.owner.as_ref() {
            unsafe {
                check(mgbfs_compact_hash_layer(
                    owner.accepted.ptr,
                    owner.lengths.ptr.cast(),
                    self.cfg.buckets,
                    self.cfg.bucket_capacity,
                    self.prev.ptr,
                    self.cfg.layer_capacity,
                    self.directory.ptr.cast(),
                    self.route_count.ptr.cast(),
                    self.fatal.ptr.cast(),
                    s,
                ))?;
                check(cudaStreamSynchronize(s))?;
            }
        }
        if self.rank_owner_mode() {
            let ready = (|| -> Result<Vec<Extent>> {
                check(unsafe { cudaStreamSynchronize(s) })?;
                let control = self.control.one::<Control>()?;
                let ring = self.ring.one::<Ring>()?;
                if control.error != 0 || ring.fatal != 0 {
                    return Err(format!("LIBRARY_RANK_DEPTH_FATAL_{}_{}", control.error, ring.fatal));
                }
                let count = self.next_extent_count.as_ref()
                    .ok_or("NEXT_EXTENT_COUNT_MISSING")?.one::<u32>()? as usize;
                if count > 2 || !self.next.is_empty() {
                    return Err("NEXT_EXTENT_CAPACITY".into());
                }
                let mut extents = vec![Extent::default(); count];
                self.next_extents.as_ref().ok_or("NEXT_EXTENTS_MISSING")?
                    .read(&mut extents)?;
                if extents.iter().any(|e| e.ready != 1 || e.count == 0 ||
                    e.granted_rows as u64 != e.count) ||
                    extents.iter().map(|e| e.count).sum::<u64>() !=
                        u64::from(self.layer_count.one::<u32>()?) {
                    return Err("NEXT_EXTENT_MISMATCH".into());
                }
                Ok(extents)
            })();
            if self.all_max(u32::from(ready.is_err()))? != 0 {
                return Err(ready.err().unwrap_or_else(|| "REMOTE_NEXT_EXTENT_FATAL".into()));
            }
            self.next.extend(ready?);
        }
        #[cfg(feature = "library-owner")]
        if let Some(library) = self.library_owner.as_mut() {
            let target = unsafe {
                history_view(
                    &self.prev,
                    library.plane_words,
                    &(0..self.cfg.layer_capacity),
                )
            };
            let count = if library.rank_mode {
                unsafe { finalize_rank_owner(library, target, s)? }
            } else {
                unsafe { finalize_shards(&mut library.shards, target, &mut library.previous, s)? }
            };
            library.closed = true;
            std::mem::swap(&mut library.previous, &mut library.current);
            // Finalization reads this word through the synchronous host
            // snapshot immediately below, outside the producer stream.
            self.route_count.put(&[count])?;
        }
        if self.fatal.one::<u32>()? != 0 {
            return Err("FINALIZE_FATAL".into());
        }
        let count = self.route_count.one::<u32>()?;
        if self.layer_count.one::<u32>()? != count {
            return Err("LAYER_COUNT_MISMATCH".into());
        }
        if self.owner.is_some() {
            self.directory.read(&mut self.prev_dir)?;
            std::mem::swap(&mut self.prev_dir, &mut self.curr_dir);
        }
        std::mem::swap(&mut self.prev, &mut self.curr);
        std::mem::swap(&mut self.front, &mut self.next);
        self.prev_count = self.current_count;
        self.current_count = count;
        self.depth = self.depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;
        Ok(self.all_max((count > 0) as u32)? != 0)
    }
    pub fn archive_current(
        &mut self,
        archive: &mut crate::pinned_archive::PinnedArchive,
    ) -> Result<()> {
        if self.failed {
            return Err("DISTRIBUTED_FAILED".into());
        }
        let compact_permutation = self.permutation_n == u32::try_from(archive.width).ok();
        if archive.width != self.width && !compact_permutation {
            return Err("ARCHIVE_STATE_WIDTH".into());
        }
        if self.archived_depth == Some(self.depth) {
            return Err("ARCHIVE_DEPTH_ALREADY_SUBMITTED".into());
        }
        for extent_index in 0..self.front.len() {
            let extent = self.front[extent_index];
            self.archive_range(archive, extent.begin, extent.count, extent_index)?;
        }
        archive.layer(u64::from(self.depth), u64::from(self.current_count))?;
        self.archived_depth = Some(self.depth);
        Ok(())
    }
    fn archive_range(
        &mut self,
        archive: &mut crate::pinned_archive::PinnedArchive,
        begin: u64,
        count: u64,
        extent_index: usize,
    ) -> Result<()> {
        let compact_permutation = self.permutation_n == u32::try_from(archive.width).ok();
        let _archive_range = TraceRange::new(
            std::env::var_os("MGBFS_TRACE_RANGES").is_some(), b"mgbfs.archive_d2h\0");
        let s = self.archive_stream.0;
        let mut offset = 0u64;
        while offset < count {
            let n = u64::from(archive.rows.min(self.cfg.batch)).min(count - offset) as u32;
            let slot = archive.acquire()?;
            let copied = (|| unsafe {
                let states = self.states.at((begin + offset) as usize * self.stride);
                check(mgbfs_hash_run(
                    self.archive_hash.0,
                    states.cast(),
                    self.archive_hashes.ptr.cast(),
                    n,
                    s,
                ))?;
                if compact_permutation && self.width != archive.width {
                    check(mgbfs_archive_pack_permutation_u8(
                        archive.width as u32,
                        self.stride as u32,
                        states.cast(),
                        n,
                        self.archive_states.ptr.cast(),
                        self.ring.ptr.cast(),
                        s,
                    ))?;
                    check(cudaMemcpyAsync(
                        slot.ptr,
                        self.archive_states.ptr,
                        n as usize * archive.width,
                        2,
                        s,
                    ))?;
                } else {
                    check(cudaMemcpy2DAsync(
                        slot.ptr,
                        self.width,
                        states,
                        self.stride,
                        self.width,
                        n as usize,
                        2,
                        s,
                    ))?;
                }
                check(cudaMemcpyAsync(
                    slot.ptr.cast::<u8>().add(n as usize * archive.width).cast(),
                    self.archive_hashes.ptr,
                    n as usize * 16,
                    2,
                    s,
                ))?;
                check(cudaEventRecord(slot.ready, s))
            })();
            // Publish cancellation before draining D2H: peers must not wait for
            // this rank to return from a potentially blocked CUDA cleanup.
            // Keep the slot alive until the stream releases its pinned bytes.
            let comm = self.comm.0;
            if let Err(error) = &copied {
                eprintln!("MGBFS_RUNTIME_FATAL rank={} error={error}", self.cfg.rank);
            }
            crate::failure::report_and_abort_on_error(
                copied, &mut self.failed, self.failure_report.as_deref(),
                || unsafe { mgbfs_nccl_abort(comm); }, || unsafe {
                    cudaStreamSynchronize(s);
                },
            )?;
            let rank = self.cfg.rank;
            let failed = &mut self.failed;
            let failure_report = self.failure_report.as_deref();
            archive.submit_notifying(slot, u64::from(self.depth), n, |error| {
                eprintln!("MGBFS_RUNTIME_FATAL rank={rank} error={error}");
                *failed = true;
                if let Some(report) = failure_report {
                    report.store(2, std::sync::atomic::Ordering::Release);
                }
                unsafe { mgbfs_nccl_abort(comm); }
            })?;
            offset += u64::from(n);
        }
        check(unsafe { cudaEventRecord(self.archive_done[extent_index].0, s) })?;
        Ok(())
    }
    pub fn snapshot(&self) -> Result<Vec<Vec<u8>>> {
        check(unsafe { cudaStreamSynchronize(self.stream.0) })?;
        let mut result = Vec::with_capacity(self.current_count as usize);
        for extent in &self.front {
            let mut bytes = vec![0u8; extent.count as usize * self.stride];
            check(unsafe {
                cudaMemcpy(
                    bytes.as_mut_ptr().cast(),
                    self.states.at(extent.begin as usize * self.stride),
                    bytes.len(),
                    2,
                )
            })?;
            result.extend(
                bytes
                    .chunks_exact(self.stride)
                    .map(|x| x[..self.width].to_vec()),
            );
        }
        Ok(result)
    }
}
