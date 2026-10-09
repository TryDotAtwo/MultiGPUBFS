//! Native 1/2/4/8-rank NCCL BFS reference. Torchrun supplies only rank env.
// Quiet dispatcher-only callsite evidence. No new CUDA/NCCL calls or threads.
static NATIVE_CALL_SITE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
struct NativeCallMarker {
    previous: Option<u32>,
}
impl NativeCallMarker {
    fn enter(site: u32) -> Self {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        Self::enter_enabled(
            site,
            *ENABLED.get_or_init(|| std::env::var_os("MGBFS_FAILURE_STAGE_QUIET").is_some()),
        )
    }
    fn enter_enabled(site: u32, enabled: bool) -> Self {
        Self {
            previous: enabled
                .then(|| NATIVE_CALL_SITE.swap(site, std::sync::atomic::Ordering::AcqRel)),
        }
    }
}
impl Drop for NativeCallMarker {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            NATIVE_CALL_SITE.store(previous, std::sync::atomic::Ordering::Release);
        }
    }
}
macro_rules! observed_native {
    ($call:expr) => {{
        let _marker = NativeCallMarker::enter(line!());
        $call
    }};
}

use crate::event_generation::NativeEvent;
use crate::failure::{
    check_native_status as check, process_owner_pair, vote_group_error, OwnerFailurePolicy,
};
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
use mgbfs_cuda::shard_ab::*;
use std::ffi::{c_void, CStr};

pub fn enable_session_cache() { crate::session_cache::enable(); }

extern "C" {
    fn cudaStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
    fn cudaGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64)
        -> i32;
    fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphUpload(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
}

// Capture ownership is shared by debug acceptance and startup-only production
// preparation. Dropping a failed capture releases its partially built graph.
struct OwnerCaptureProbe {
    stream: *mut c_void,
    capturing: bool,
    graph: *mut c_void,
    executable: *mut c_void,
}
impl OwnerCaptureProbe {
    #[cfg(debug_assertions)]
    fn begin(stream: *mut c_void) -> Result<Option<Self>> {
        let _native_scope = NativeCallMarker::enter(line!());
        Self::begin_enabled(
            stream,
            std::env::var_os("MGBFS_TEST_OWNER_DAG_CAPTURE").is_some(),
        )
    }
    fn begin_enabled(stream: *mut c_void, enabled: bool) -> Result<Option<Self>> {
        if !enabled {
            return Ok(None);
        }
        check(observed_native!(unsafe {
            cudaStreamBeginCapture(stream, 1)
        }))?;
        Ok(Some(Self {
            stream,
            capturing: true,
            graph: std::ptr::null_mut(),
            executable: std::ptr::null_mut(),
        }))
    }
    fn finish(mut self) -> Result<OwnedOwnerGraph> {
        let status = unsafe { cudaStreamEndCapture(self.stream, &mut self.graph) };
        self.capturing = false;
        check(observed_native!(status))?;
        check(observed_native!(unsafe {
            cudaGraphInstantiateWithFlags(&mut self.executable, self.graph, 0)
        }))?;
        Ok(OwnedOwnerGraph {
            graph: std::mem::replace(&mut self.graph, std::ptr::null_mut()),
            executable: std::mem::replace(&mut self.executable, std::ptr::null_mut()),
        })
    }
    #[cfg(debug_assertions)]
    fn launch(self) -> Result<()> {
        self.launch_named("MGBFS_OWNER_DAG_CAPTURE")
    }
    #[cfg(debug_assertions)]
    fn launch_named(self, marker: &str) -> Result<()> {
        let stream = self.stream;
        self.finish()?.launch(stream)?;
        eprintln!("{marker} launched");
        Ok(())
    }
}
impl Drop for OwnerCaptureProbe {
    fn drop(&mut self) {
        let _native_scope = NativeCallMarker::enter(line!());
        unsafe {
            if self.capturing {
                cudaStreamEndCapture(self.stream, &mut self.graph);
            }
            if !self.executable.is_null() {
                cudaGraphExecDestroy(self.executable);
            }
            if !self.graph.is_null() {
                cudaGraphDestroy(self.graph);
            }
        }
    }
}

struct OwnedOwnerGraph {
    graph: *mut c_void,
    executable: *mut c_void,
}
impl OwnedOwnerGraph {
    fn launch(&self, stream: *mut c_void) -> Result<()> {
        check(observed_native!(unsafe {
            cudaGraphLaunch(self.executable, stream)
        }))
    }
}
impl Drop for OwnedOwnerGraph {
    fn drop(&mut self) {
        unsafe {
            cudaGraphExecDestroy(self.executable);
            cudaGraphDestroy(self.graph);
        }
    }
}
struct OwnerGraphEntry {
    states: *const u8,
    hashes: *const c_void,
    source_rows: *const u32,
    graph: OwnedOwnerGraph,
}
struct OwnerGraphReplay {
    rank: u32,
    input_count: usize,
    entries: Vec<OwnerGraphEntry>,
    launches: std::cell::Cell<u64>,
    driver_free_delta_bytes: u64,
}
impl OwnerGraphReplay {
    fn launch(
        &self,
        target_slot: usize,
        input: usize,
        states: *const u8,
        hashes: *const c_void,
        source_rows: *const u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let index = target_slot
            .checked_mul(self.input_count)
            .and_then(|base| base.checked_add(input))
            .ok_or("OWNER_GRAPH_INDEX_OVERFLOW")?;
        let entry = self.entries.get(index).ok_or("OWNER_GRAPH_INDEX")?;
        if input >= self.input_count
            || entry.states != states
            || entry.hashes != hashes
            || entry.source_rows != source_rows
        {
            return Err("OWNER_GRAPH_POINTER_LEASE".into());
        }
        entry.graph.launch(stream)?;
        self.launches.set(self.launches.get().saturating_add(1));
        Ok(())
    }
}
impl Drop for OwnerGraphReplay {
    fn drop(&mut self) {
        // One receipt per runtime; no per-batch formatting or allocation.
        eprintln!("MGBFS_OWNER_GRAPH_SUMMARY {{\"rank\":{},\"instances\":{},\"launches\":{},\"late_instantiations\":0,\"driver_free_delta_bytes\":{}}}",
            self.rank, self.entries.len(), self.launches.get(), self.driver_free_delta_bytes);
    }
}

// Thread-local NVTX push/pop ownership survives early returns and panics.
// Diagnostic ranges never add CUDA waits or device reads.
struct TraceRange(bool);
impl TraceRange {
    fn new(enabled: bool, label: &'static [u8]) -> Self {
        let _native_scope = NativeCallMarker::enter(line!());
        if enabled {
            unsafe { mgbfs_trace_range_push(label.as_ptr().cast()) };
        }
        Self(enabled)
    }
}
impl Drop for TraceRange {
    fn drop(&mut self) {
        let _native_scope = NativeCallMarker::enter(line!());
        if self.0 {
            unsafe { mgbfs_trace_range_pop() };
        }
    }
}

// Failure-path diagnostics only. These wrappers preserve the original CUDA call,
// arguments and result; they do not turn a synchronous call into a safe wait.
// The caller location is required because a cancelled peer can be trapped before
// it reaches the NCCL-owner cancellation probe. Disabled in normal runs.
#[track_caller]
unsafe fn traced_stream_synchronize(stream: *mut c_void) -> i32 {
    let _native_scope = NativeCallMarker::enter(line!());
    let _marker = NativeCallMarker::enter(line!());
    let trace = std::env::var_os("MGBFS_TRACE_FAILURE_TEARDOWN").is_some()
        && std::env::var_os("MGBFS_FAILURE_STAGE_QUIET").is_none();
    let site = std::panic::Location::caller();
    if trace {
        eprintln!("MGBFS_BLOCKING_API rank={} api=cudaStreamSynchronize phase=enter site={}:{} stream={:?}",
            std::env::var("RANK").unwrap_or_default(), site.file(), site.line(), stream);
    }
    let code = mgbfs_cuda::ffi::cudaStreamSynchronize(stream);
    if trace {
        eprintln!(
            "MGBFS_BLOCKING_API rank={} api=cudaStreamSynchronize phase=exit site={}:{} code={}",
            std::env::var("RANK").unwrap_or_default(),
            site.file(),
            site.line(),
            code
        );
    }
    code
}
#[track_caller]
unsafe fn traced_device_copy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32 {
    let _native_scope = NativeCallMarker::enter(line!());
    let _marker = NativeCallMarker::enter(line!());
    let trace = std::env::var_os("MGBFS_TRACE_FAILURE_TEARDOWN").is_some()
        && std::env::var_os("MGBFS_FAILURE_STAGE_QUIET").is_none();
    let site = std::panic::Location::caller();
    if trace {
        eprintln!(
            "MGBFS_BLOCKING_API rank={} api=cudaMemcpy phase=enter site={}:{} bytes={} kind={}",
            std::env::var("RANK").unwrap_or_default(),
            site.file(),
            site.line(),
            bytes,
            kind
        );
    }
    let code = mgbfs_cuda::ffi::cudaMemcpy(dst, src, bytes, kind);
    if trace {
        eprintln!(
            "MGBFS_BLOCKING_API rank={} api=cudaMemcpy phase=exit site={}:{} code={}",
            std::env::var("RANK").unwrap_or_default(),
            site.file(),
            site.line(),
            code
        );
    }
    code
}

extern "C" fn nccl_cancel_probe(context: *mut c_void) -> i32 {
    let _native_scope = NativeCallMarker::enter(line!());
    if context.is_null() {
        return 1;
    }
    let flag = unsafe { &*context.cast::<std::sync::atomic::AtomicBool>() };
    i32::from(flag.load(std::sync::atomic::Ordering::Acquire))
}
extern "C" fn nccl_retirement_probe(context: *mut c_void, publish: i32) -> i32 {
    let _native_scope = NativeCallMarker::enter(line!());
    if context.is_null() {
        return -1;
    }
    let state = unsafe { &*context.cast::<crate::bootstrap::SearchRetirement>() };
    // Failure-only diagnostic: distinguish CUDA drain from the existing sideband ACK.
    // Never publish a reader ACK here; the original publish decision remains below.
    if publish != 0 && std::env::var_os("MGBFS_TRACE_FAILURE_TEARDOWN").is_some() {
        eprintln!("MGBFS_FAILURE_TEARDOWN rank={} stage=retirement_probe_enter publish={} local={} group={} failed={}",
            std::env::var("RANK").unwrap_or_else(|_| "unset".into()), publish,
            state.local.load(std::sync::atomic::Ordering::Acquire),
            state.group.load(std::sync::atomic::Ordering::Acquire),
            state.failed.load(std::sync::atomic::Ordering::Acquire));
    }

    if publish < 0 {
        state
            .failed
            .store(true, std::sync::atomic::Ordering::Release);
    }
    if publish > 0 {
        state
            .local
            .store(true, std::sync::atomic::Ordering::Release);
    }
    if state.failed.load(std::sync::atomic::Ordering::Acquire) {
        -1
    } else {
        i32::from(state.group.load(std::sync::atomic::Ordering::Acquire))
    }
}

#[cfg(debug_assertions)]
thread_local! {
    static TEST_OWNER_HOST_FAULT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TEST_MATERIALIZER_LOCAL_FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

#[cfg(debug_assertions)]
pub fn inject_owner_host_error_once_for_test() {
    let _native_scope = NativeCallMarker::enter(line!());
    TEST_OWNER_HOST_FAULT.with(|flag| flag.set(true));
}

#[cfg(debug_assertions)]
pub fn inject_materializer_local_error_once_for_test(after_regenerate: bool) {
    let _native_scope = NativeCallMarker::enter(line!());
    TEST_MATERIALIZER_LOCAL_FAULT.with(|flag| flag.set(if after_regenerate { 2 } else { 1 }));
}
#[cfg(debug_assertions)]
fn inject_materializer_local_stage(
    stage: u8,
    rank: u32,
    fatal: *mut u32,
    stream: *mut c_void,
) -> Result<()> {
    let _native_scope = NativeCallMarker::enter(line!());
    let inject = TEST_MATERIALIZER_LOCAL_FAULT.with(|flag| {
        if flag.get() == stage {
            flag.set(0);
            true
        } else {
            false
        }
    });
    if inject {
        check(observed_native!(unsafe {
            mgbfs_device_store_u32(fatal, 1, stream)
        }))?;
        eprintln!(
            "MGBFS_TEST_MATERIALIZER_LOCAL_FATAL rank={rank} stage={}",
            if stage == 1 { "sort" } else { "regenerate" }
        );
    }
    Ok(())
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
    weighted_ranks: Vec<RankHandle>,
    rank_mode: bool,
    rank_accepted: *const u32,
    logical_owner: u32,
}
#[cfg(feature = "library-owner")]
impl Drop for LibraryOwnerStorage {
    fn drop(&mut self) {
        let _native_scope = NativeCallMarker::enter(line!());
        // This field is dropped BEFORE streams/history. On failed drain leave
        // pool destruction to rank-process exit rather than free GPU readers.
        let mut drained = true;
        for rank in &mut self.weighted_ranks {
            if !rank.is_null() {
                let released = unsafe { mgbfs_library_rank_destroy_v1(*rank) == 0 };
                drained &= released;
                if released {
                    *rank = std::ptr::null_mut();
                }
            }
        }
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
                    if !crate::session_cache::pool_put(self.pool) { mgbfs_library_pool_destroy_v1(self.pool); }
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
    let _native_scope = NativeCallMarker::enter(line!());
    let old: Vec<_> = library
        .previous
        .iter()
        .map(|range| history_view(previous, library.plane_words, range))
        .collect();
    let now: Vec<_> = library
        .current
        .iter()
        .map(|range| history_view(current, library.plane_words, range))
        .collect();
    let capacities = vec![library.capacity; shards as usize];
    let mut rank = std::ptr::null_mut();
    check(observed_native!(mgbfs_library_rank_create_cuco_v1(
        old.as_ptr(),
        now.as_ptr(),
        capacities.as_ptr(),
        shards,
        incoming,
        library.logical_owner,
        world,
        stream,
        &mut rank,
    )))?;
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
    let _native_scope = NativeCallMarker::enter(line!());
    if library.rank.is_null() || destination.rows == 0 {
        return Err("LIBRARY_RANK_FINALIZE_SHAPE".into());
    }
    // FinalizeDepth drains every candidate/materialization reader before
    // releasing the history sets and reusing either SoA history buffer.
    check(observed_native!(traced_stream_synchronize(stream)))?;
    let mut counts = vec![0u32; library.previous.len()];
    if !library.rank_accepted.is_null() {
        check(observed_native!(traced_device_copy(
            counts.as_mut_ptr().cast(),
            library.rank_accepted.cast(),
            counts.len() * 4,
            2,
        )))?;
    }
    let total = counts
        .iter()
        .try_fold(0u32, |sum, &n| sum.checked_add(n))
        .ok_or("LIBRARY_RANK_FINALIZE_OVERFLOW")?;
    if total > destination.rows {
        return Err("LIBRARY_RANK_FINALIZE_CAPACITY".into());
    }
    check(observed_native!(mgbfs_library_rank_seal_v1(library.rank)))?;
    let mut offset = 0u32;
    for (shard, &rows) in counts.iter().enumerate() {
        let mut keys = KeysV1 {
            words: [std::ptr::null(); 4],
            rows: 0,
            reserved: 0,
        };
        check(observed_native!(mgbfs_library_rank_export_shard_v1(
            library.rank,
            shard as u32,
            rows,
            &mut keys,
        )))?;
        if keys.rows != rows {
            return Err("LIBRARY_RANK_EXPORT_COUNT".into());
        }
        for plane in 0..4 {
            if rows != 0 {
                check(observed_native!(cudaMemcpyAsync(
                    destination.words[plane]
                        .cast_mut()
                        .add(offset as usize)
                        .cast(),
                    keys.words[plane].cast(),
                    rows as usize * 4,
                    3,
                    stream,
                )))?;
            }
        }
        library.previous[shard] = offset..offset + rows;
        offset += rows;
    }
    check(observed_native!(traced_stream_synchronize(stream)))?;
    check(observed_native!(mgbfs_library_rank_destroy_v1(
        library.rank
    )))?;
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
    let _native_scope = NativeCallMarker::enter(line!());
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

#[derive(Clone, Debug)]
pub struct DistributedConfig {
    /// Physical raw/sorted/packed source payload banks, independent of credits
    /// and the single LSA receive slot. Selected before allocation.
    pub route_banks: usize,
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
// Failure-only diagnostic classification. The mapped bit is already the
// existing sticky GPU owner/ring signal; no GPU count/control readback and no
// healthy-batch wait is introduced. Preserve the original API error as cause.
fn annotate_device_logical_failure<T>(
    result: Result<T>,
    terminal: *const std::sync::atomic::AtomicU32,
) -> Result<T> {
    match result {
        Err(error)
            if !terminal.is_null()
                && unsafe { &*terminal }.load(std::sync::atomic::Ordering::Acquire) & 2 != 0 =>
        {
            Err(format!(
                "GROUP_OWNER_OR_PRE_OWNER_FATAL:DEVICE_LOGICAL_FATAL cause={error}"
            ))
        }
        other => other,
    }
}
fn wait_nccl_stream(
    comm: *mut c_void,
    stream: *mut c_void,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    let _native_scope = NativeCallMarker::enter(line!());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err("REMOTE_SEARCH_CANCELLED".into());
        }
        match unsafe { cudaStreamQuery(stream) } {
            0 => return Ok(()),
            600 => {
                match unsafe { mgbfs_nccl_poll(comm) } {
                    0 | 4 => {}
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
    let _native_scope = NativeCallMarker::enter(line!());
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

/// Stable-address raw -> sorted -> packed source bank. Every buffer is charged
/// by distributed_memory before admission and allocated before depth zero.
/// The producer may overwrite this bank only behind its last-reader event.
/// Stack metadata only: neither a payload allocation nor an extra epoch slot.
struct OwnerPacketContext {
    generation: Option<u64>,
    candidate_count: u32,
    target_depth: Option<u32>,
    packet_stride: usize,
    batch_index: u64,
    scheduled_rounds: u32,
    trace_route: bool,
    parent: Option<Extent>,
    extent_offset: u64,
    parents: u32,
    extent_index: usize,
    archive_live: bool,
    archive_dependency_imported: bool,
}
#[derive(Clone, Copy)]
struct WeightedRouteCursor {
    sequence: u64,
    parents: u32,
    next_run: usize,
    packet_live: bool,
}
struct RouteBank {
    children: Buffer,
    child_hashes: Buffer,
    sorted_hashes: Buffer,
    sorted_refs: Buffer,
    route_count: Buffer,
    packed_states: Buffer,
    owner_counts: Buffer,
    generation_control: Buffer,
    generation_done: NativeEvent,
    last_reader: Event,
    weighted_cursor: Option<WeightedRouteCursor>,
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
    unsafe fn new(
        token: std::sync::Arc<std::sync::atomic::AtomicBool>,
        word: *mut u32,
    ) -> Result<Self> {
        let _native_scope = NativeCallMarker::enter(line!());
        if word.is_null()
            || word as usize % std::mem::align_of::<std::sync::atomic::AtomicU32>() != 0
        {
            return Err("LSA_CANCEL_WORD_ALIGNMENT".into());
        }
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_stop = stop.clone();
        let address = word as usize;
        let worker = std::thread::Builder::new()
            .name("mgbfs-cancel-mirror".into())
            .spawn(move || {
                while !worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                    if token.load(std::sync::atomic::Ordering::Acquire) {
                        let signal = &*(address as *const std::sync::atomic::AtomicU32);
                        signal.fetch_or(1, std::sync::atomic::Ordering::Release);
                        // Publish cancellation first. Only the existing mirror
                        // observes the quiet probe; it never calls or aborts NCCL.
                        #[cfg(target_os = "linux")]
                        if std::env::var_os("MGBFS_FAILURE_STAGE_QUIET").is_some() {
                            report_cancel_native_site(0);
                            std::thread::sleep(std::time::Duration::from_millis(250));
                            if !worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                                report_cancel_native_site(1);
                            }
                        }
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            })
            .map_err(|e| format!("LSA_CANCEL_MIRROR_THREAD: {e}"))?;
        Ok(Self {
            stop,
            worker: Some(worker),
        })
    }
}
impl Drop for CancelMirror {
    fn drop(&mut self) {
        let _native_scope = NativeCallMarker::enter(line!());
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn admit_device_group(
    rank: u32,
    pool_bytes: Option<u64>,
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
    if std::env::var("MGBFS_MEMORY_QUERY").ok().as_deref() == Some("1") {
        let status = check(unsafe { cudaMemGetInfo(&mut free, &mut total) });
        if vote(u32::from(status.is_err()))? != 0 {
            return Err("MEMORY_QUERY_GROUP_FAILED".into());
        }
        println!("MGBFS_MEMORY_QUERY {}", serde_json::json!({
            "schema": 1, "rank": rank, "required_bytes": required, "reserve_bytes": reserve,
            "free_after_nccl_warmup_bytes": (free as u64 + crate::session_cache::reusable_bytes()).min(total as u64), "total_bytes": total,
            "session_reusable_bytes": crate::session_cache::reusable_bytes(),
            "scope": "explicit_device_allocations_including_fixed_library_pool",
            "library_pool_bytes": pool_bytes,
        }));
        std::io::Write::flush(&mut std::io::stdout()).map_err(|e| e.to_string())?;
        // Every rank must publish its record before any constructor unwinds
        // and aborts NCCL or notifies the search-cancellation sideband. Without
        // this rendezvous a fast query rank can cancel a peer still returning
        // from the preceding collective, losing that peer's memory record.
        vote(0)?;
        if crate::session_cache::enabled() {
            let ready = unsafe { mgbfs_nccl_session_park(comm) } == 0;
            crate::session_cache::permit_comm(vote(u32::from(!ready))? == 0);
        }
        // Query subprocesses intentionally stop before large allocations and
        // before BFS. No capacity failure or completed layer is claimed.
        return Err("MEMORY_QUERY_DONE".into());
    }
    let local = check(unsafe { cudaMemGetInfo(&mut free, &mut total) })
        .and_then(|_| crate::distributed_memory::device_admission(required, reserve,
            (free as u64 + crate::session_cache::reusable_bytes()).min(total as u64)));
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
fn setup_failure_vote(
    comm: *mut c_void,
    stream: *mut c_void,
    send: &Buffer,
    recv: &Buffer,
    failed: bool,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<bool> {
    let _native_scope = NativeCallMarker::enter(line!());
    send.put_u32(u32::from(failed))?;
    check(observed_native!(unsafe {
        mgbfs_nccl_all_reduce_max_u32(comm, send.ptr.cast(), recv.ptr.cast(), stream)
    }))?;
    wait_nccl_stream(comm, stream, cancelled)?;
    Ok(recv.one::<u32>()? != 0)
}
impl Buffer {
    fn new(bytes: usize, stream: *mut c_void) -> Result<Self> {
        let mut ptr = crate::session_cache::buffer_take(bytes).unwrap_or(std::ptr::null_mut());
        if ptr.is_null() { check(unsafe { cudaMalloc(&mut ptr, bytes.max(1)) })?; }
        let x = Self { ptr, bytes, stream };
        check(unsafe { cudaMemsetAsync(ptr, 0, bytes.max(1), stream) })?;
        Ok(x)
    }
    fn put<T: Copy>(&self, x: &[T]) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        if std::mem::size_of_val(x) > self.bytes {
            return Err("UPLOAD_CAPACITY".into());
        }
        check(observed_native!(unsafe {
            cudaMemcpyAsync(
                self.ptr,
                x.as_ptr().cast(),
                std::mem::size_of_val(x),
                1,
                self.stream,
            )
        }))?;
        check(observed_native!(unsafe {
            traced_stream_synchronize(self.stream)
        }))
    }
    fn put_u32(&self, value: u32) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        if self.bytes < std::mem::size_of::<u32>() {
            return Err("UPLOAD_CAPACITY".into());
        }
        // The scalar is a launch argument, so no host slice needs to remain
        // alive while the stream consumes it. Its GPU consumers stay ordered.
        check(observed_native!(unsafe {
            mgbfs_device_store_u32(self.ptr.cast(), value, self.stream)
        }))
    }
    fn read<T: Copy>(&self, x: &mut [T]) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        if std::mem::size_of_val(x) > self.bytes {
            return Err("READ_CAPACITY".into());
        }
        check(observed_native!(unsafe {
            traced_device_copy(x.as_mut_ptr().cast(), self.ptr, std::mem::size_of_val(x), 2)
        }))
    }
    fn one<T: Copy + Default>(&self) -> Result<T> {
        let _native_scope = NativeCallMarker::enter(line!());
        let mut x = [T::default()];
        self.read(&mut x)?;
        Ok(x[0])
    }
    unsafe fn at(&self, n: usize) -> *mut c_void {
        let _native_scope = NativeCallMarker::enter(line!());
        self.ptr.cast::<u8>().add(n).cast()
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if !crate::session_cache::buffer_put(self.ptr, self.bytes) { cudaFree(self.ptr); }
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
// Monotonic ring sequences, not modulo addresses. Shared owner scratch is
// leased by the sole owner stream; no new state arena or device allocation.
struct WeightedOwnerRefs {
    accepted: Buffer,
    merged: Buffer,
    targets: Vec<Option<u32>>,
    counts_host: Vec<u32>,
    settle: Plan,
    materialize: Plan,
    promotion_count: Buffer,
    compact_hashes: Buffer,
    compact_refs: Buffer,
    compact_count: Buffer,
    survivor_hashes: Buffer,
    survivor_refs: Buffer,
    survivor_count: Buffer,
    settle_state: Buffer,
    history: Buffer,
    history_counts: Buffer,
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
        let _native_scope = NativeCallMarker::enter(line!());
        let mut p = std::ptr::null_mut();
        let mut e = [0i8; 512];
        let status = create(&mut p, e.as_mut_ptr());
        if status != 0 {
            let message = unsafe { CStr::from_ptr(e.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(if message.is_empty() {
                format!("NATIVE_PLAN_CREATE_FAILED status={status}")
            } else {
                message
            });
        }
        Ok(Self(p, drop))
    }
}
impl Drop for Plan {
    fn drop(&mut self) {
        let _native_scope = NativeCallMarker::enter(line!());
        unsafe { self.1(self.0) }
    }
}
#[cfg(test)]
mod plan_error_tests {
    use super::*;
    #[test]
    fn one_rank_lsa_abort_retires_local_readers_without_peer_callback() {
        let _native_scope = NativeCallMarker::enter(line!());
        let graph = MatrixGroup::unitriangular(3, 2).unwrap();
        let mut id = [0u8; 128];
        assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
        let cfg = DistributedConfig {
            route_banks: 4,
            epoch_window: 3,
            rank: 0,
            world: 1,
            logical_owner_to_rank: vec![0, 0],
            transport: mgbfs_core::config::ReferenceTransport::Lsa,
            batch: 1,
            layer_capacity: 64,
            state_ring_capacity: 128,
            state_descriptor_capacity: 128,
            buckets: 8,
            shards: 2,
            job_buckets: 2,
            bucket_capacity: 32,
            prededup: true,
            generation_variant: 1,
            untouched_vram_reserve: 1 << 30,
        };
        let mut bfs = DistributedNativeBfs::new_reference_with_owner(
            &graph,
            [42; 16],
            id,
            cfg,
            None,
            OwnerBackend::CubSortMerge,
            256,
        )
        .unwrap();
        assert!(bfs.advance().unwrap()); // Exercise actual registered LSA readers.
        let status = unsafe { mgbfs_nccl_abort(bfs.comm.0) };
        bfs.failed = true;
        assert_eq!(
            status, 0,
            "one-rank reader drain must not require a nonexistent peer"
        );
        assert_eq!(
            unsafe { mgbfs_nccl_abort(bfs.comm.0) },
            0,
            "already completed terminal abort must be idempotent"
        );
    }
    #[test]
    fn invalid_epoch_window_is_rejected_before_device_or_communicator_admission() {
        let _native_scope = NativeCallMarker::enter(line!());
        let graph = MatrixGroup::unitriangular(3, 2).unwrap();
        for (epoch_window, state_descriptor_capacity, expected_error) in [
            (0, 16, "EPOCH_WINDOW_CONFIG"),
            (1, 16, "EPOCH_WINDOW_CONFIG"),
            (2, 0, "STATE_DESCRIPTOR_CAPACITY"),
        ] {
            let cfg = DistributedConfig {
                route_banks: 2,
                epoch_window,
                rank: 0,
                world: 1,
                logical_owner_to_rank: vec![0],
                transport: mgbfs_core::config::ReferenceTransport::Lsa,
                batch: 2,
                layer_capacity: 8,
                state_ring_capacity: 16,
                state_descriptor_capacity,
                buckets: 8,
                shards: 2,
                job_buckets: 2,
                bucket_capacity: 8,
                prededup: true,
                generation_variant: 1,
                untouched_vram_reserve: 1 << 30,
            };
            match DistributedNativeBfs::new(&graph, [0; 16], [0; 128], cfg) {
                Ok(_) => panic!("invalid credit window admitted"),
                Err(error) => assert_eq!(error, expected_error),
            }
        }
    }
    extern "C" {
        fn mgbfs_nccl_lsa_fatal_vote(
            comm: *mut c_void,
            send: *const u32,
            receive: *mut u32,
            stream: *mut c_void,
        ) -> i32;
    }

    #[test]
    fn tensor_hash_first_rejects_unsupported_device_at_construction() {
        let _native_scope = NativeCallMarker::enter(line!());
        extern "C" {
            fn cudaDeviceGetAttribute(out: *mut i32, attribute: i32, device: i32) -> i32;
        }
        let (mut major, mut minor) = (0, 0);
        assert_eq!(unsafe { cudaDeviceGetAttribute(&mut major, 75, 0) }, 0);
        assert_eq!(unsafe { cudaDeviceGetAttribute(&mut minor, 76, 0) }, 0);
        if (major, minor) == (7, 5) {
            return;
        } // Unsupported-device fixture only.
        let graph = MatrixGroup::unitriangular(3, 2).unwrap();
        let mut id = [0u8; 128];
        assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
        let cfg = DistributedConfig {
            route_banks: 2,
            epoch_window: 2,
            rank: 0,
            world: 1,
            logical_owner_to_rank: vec![0, 0],
            transport: mgbfs_core::config::ReferenceTransport::Lsa,
            batch: 2,
            layer_capacity: 8,
            state_ring_capacity: 16,
            state_descriptor_capacity: 16,
            buckets: 8,
            shards: 2,
            job_buckets: 2,
            bucket_capacity: 8,
            prededup: true,
            generation_variant: 1,
            untouched_vram_reserve: 1 << 30,
        };
        match DistributedNativeBfs::new_reference_with_owner_and_cancel(
            &graph,
            [0; 16],
            id,
            cfg,
            Some(8),
            OwnerBackend::CubSortMerge,
            256,
            true,
            None,
            None,
        ) {
            Ok(_) => panic!("unsupported Tensor backend was admitted until first batch"),
            Err(error) => assert_eq!(error, "HASH_FIRST_TC_DEVICE_UNSUPPORTED"),
        }
    }

    #[test]
    fn lsa_logical_fatal_closes_admission_after_the_completed_epoch() {
        let _native_scope = NativeCallMarker::enter(line!());
        // Removing the device-to-mapped failure publication, or ignoring it
        // in admission, must let this real runtime wrongly admit another batch.
        for (ring_fatal, owner_error) in [(0, 0), (7, 0), (0, 9)] {
            let graph = MatrixGroup::unitriangular(3, 2).unwrap();
            let mut id = [0u8; 128];
            assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
            let cfg = DistributedConfig {
                route_banks: 2,
                epoch_window: 2,
                rank: 0,
                world: 1,
                logical_owner_to_rank: vec![0, 0],
                transport: mgbfs_core::config::ReferenceTransport::Lsa,
                batch: 2,
                layer_capacity: 8,
                state_ring_capacity: 16,
                state_descriptor_capacity: 16,
                buckets: 8,
                shards: 2,
                job_buckets: 2,
                bucket_capacity: 8,
                prededup: true,
                generation_variant: 1,
                untouched_vram_reserve: 1 << 30,
            };
            let bfs = DistributedNativeBfs::new_reference_with_owner(
                &graph,
                [0; 16],
                id,
                cfg,
                None,
                OwnerBackend::CubSortMerge,
                256,
            )
            .unwrap();
            // Generic reductions can be nonzero without an owner failure.
            bfs.collective_send.put_u32(3).unwrap();
            check(observed_native!(unsafe {
                mgbfs_nccl_lsa_fatal_vote(
                    bfs.comm.0,
                    bfs.collective_send.ptr.cast(),
                    bfs.collective_recv.ptr.cast(),
                    bfs.stream.0,
                )
            }))
            .unwrap();
            check(observed_native!(unsafe {
                traced_stream_synchronize(bfs.stream.0)
            }))
            .unwrap();
            assert!(
                bfs.ensure_not_cancelled().is_ok(),
                "non-owner vote cancelled admission"
            );
            bfs.ring
                .put(&[Ring {
                    fatal: ring_fatal,
                    capacity: 16,
                    ..Ring::default()
                }])
                .unwrap();
            bfs.control
                .put(&[Control {
                    error: owner_error,
                    ..Control::default()
                }])
                .unwrap();
            check(observed_native!(unsafe {
                mgbfs_owner_lsa_fatal_gate(
                    bfs.comm.0,
                    bfs.ring.ptr.cast(),
                    bfs.control.ptr.cast(),
                    bfs.collective_send.ptr.cast(),
                    bfs.collective_recv.ptr.cast(),
                    bfs.stream.0,
                )
            }))
            .unwrap();
            // Test-only drain stands for observing a completed epoch credit.
            check(observed_native!(unsafe {
                traced_stream_synchronize(bfs.stream.0)
            }))
            .unwrap();
            let admitted = bfs.ensure_not_cancelled();
            if ring_fatal == 0 && owner_error == 0 {
                assert!(admitted.is_ok(), "healthy epoch rejected: {admitted:?}");
            } else {
                assert!(
                    admitted.is_err(),
                    "completed fatal epoch reopened batch admission"
                );
            }
        }
    }

    #[test]
    fn host_owner_only_fatal_closes_admission_after_the_completed_epoch() {
        let _native_scope = NativeCallMarker::enter(line!());
        // Removing the device-to-mapped failure publication, or ignoring it
        // in admission, must let this real runtime wrongly admit another batch.
        extern "C" {
            fn mgbfs_owner_global_fatal_gate(
                comm: *mut c_void,
                ring: *mut c_void,
                owner: *mut c_void,
                send: *mut u32,
                receive: *mut u32,
                stream: *mut c_void,
            ) -> i32;
        }
        for (ring_fatal, owner_error) in [(0, 0), (0, 9)] {
            let graph = MatrixGroup::unitriangular(3, 2).unwrap();
            let mut id = [0u8; 128];
            assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
            let cfg = DistributedConfig {
                route_banks: 2,
                epoch_window: 2,
                rank: 0,
                world: 1,
                logical_owner_to_rank: vec![0, 0],
                transport: mgbfs_core::config::ReferenceTransport::HostSizedNccl,
                batch: 2,
                layer_capacity: 8,
                state_ring_capacity: 16,
                state_descriptor_capacity: 16,
                buckets: 8,
                shards: 2,
                job_buckets: 2,
                bucket_capacity: 8,
                prededup: true,
                generation_variant: 1,
                untouched_vram_reserve: 1 << 30,
            };
            let bfs = DistributedNativeBfs::new_reference_with_owner(
                &graph,
                [0; 16],
                id,
                cfg,
                None,
                OwnerBackend::CubSortMerge,
                256,
            )
            .unwrap();
            bfs.ring
                .put(&[Ring {
                    fatal: ring_fatal,
                    capacity: 16,
                    ..Ring::default()
                }])
                .unwrap();
            bfs.control
                .put(&[Control {
                    error: owner_error,
                    ..Control::default()
                }])
                .unwrap();
            check(observed_native!(unsafe {
                mgbfs_owner_global_fatal_gate(
                    bfs.comm.0,
                    bfs.ring.ptr.cast(),
                    bfs.control.ptr.cast(),
                    bfs.collective_send.ptr.cast(),
                    bfs.collective_recv.ptr.cast(),
                    bfs.stream.0,
                )
            }))
            .unwrap();
            // Test-only drain stands for observing a completed epoch credit.
            check(observed_native!(unsafe {
                traced_stream_synchronize(bfs.stream.0)
            }))
            .unwrap();
            assert_eq!(
                bfs.collective_recv.one::<u32>().unwrap() != 0,
                owner_error != 0,
                "owner-only fatal was lost by the device group vote"
            );
            let admitted = bfs.ensure_not_cancelled();
            if ring_fatal == 0 && owner_error == 0 {
                assert!(admitted.is_ok(), "healthy epoch rejected: {admitted:?}");
            } else {
                assert!(
                    admitted.is_err(),
                    "completed fatal epoch reopened batch admission"
                );
            }
        }
    }

    #[test]
    fn host_local_fatal_closes_admission_without_a_collective() {
        let _native_scope = NativeCallMarker::enter(line!());
        // Removing the device-to-mapped failure publication, or ignoring it
        // in admission, must let this real runtime wrongly admit another batch.
        for (ring_fatal, owner_error) in [(0, 0), (7, 0), (0, 9)] {
            let graph = MatrixGroup::unitriangular(3, 2).unwrap();
            let mut id = [0u8; 128];
            assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
            let cfg = DistributedConfig {
                route_banks: 2,
                epoch_window: 2,
                rank: 0,
                world: 1,
                logical_owner_to_rank: vec![0, 0],
                transport: mgbfs_core::config::ReferenceTransport::HostSizedNccl,
                batch: 2,
                layer_capacity: 8,
                state_ring_capacity: 16,
                state_descriptor_capacity: 16,
                buckets: 8,
                shards: 2,
                job_buckets: 2,
                bucket_capacity: 8,
                prededup: true,
                generation_variant: 1,
                untouched_vram_reserve: 1 << 30,
            };
            let bfs = DistributedNativeBfs::new_reference_with_owner(
                &graph,
                [0; 16],
                id,
                cfg,
                None,
                OwnerBackend::CubSortMerge,
                256,
            )
            .unwrap();
            bfs.ring
                .put(&[Ring {
                    fatal: ring_fatal,
                    capacity: 16,
                    ..Ring::default()
                }])
                .unwrap();
            bfs.control
                .put(&[Control {
                    error: owner_error,
                    ..Control::default()
                }])
                .unwrap();
            // A stale nonzero group flag must be replaced even for healthy work.
            bfs.collective_recv.put_u32(17).unwrap();
            bfs.queue_owner_fatal_gate(bfs.stream.0).unwrap();
            // Test-only drain stands for observing a completed epoch credit.
            check(observed_native!(unsafe {
                traced_stream_synchronize(bfs.stream.0)
            }))
            .unwrap();
            assert_eq!(
                bfs.ring.one::<Ring>().unwrap().fatal != 0,
                ring_fatal != 0 || owner_error != 0,
                "local sticky fatal lost"
            );
            assert_eq!(
                bfs.collective_recv.one::<u32>().unwrap() != 0,
                ring_fatal != 0 || owner_error != 0,
                "materialization fatal flag stale"
            );
            let admitted = bfs.ensure_not_cancelled();
            if ring_fatal == 0 && owner_error == 0 {
                assert!(admitted.is_ok(), "healthy epoch rejected: {admitted:?}");
            } else {
                assert!(
                    admitted.is_err(),
                    "completed fatal epoch reopened batch admission"
                );
            }
        }
    }

    #[test]
    fn failure_classification_preserves_gpu_logical_fatal_and_original_cause() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let terminal = AtomicU32::new(0);
        assert_eq!(
            annotate_device_logical_failure::<()>(Err("CUDA_STATUS_14".into()), &terminal)
                .unwrap_err(),
            "CUDA_STATUS_14"
        );
        terminal.store(1, Ordering::Release);
        assert_eq!(
            annotate_device_logical_failure::<()>(Err("REMOTE_SEARCH_CANCELLED".into()), &terminal)
                .unwrap_err(),
            "REMOTE_SEARCH_CANCELLED"
        );
        terminal.store(2, Ordering::Release);
        for cause in [
            "CUDA_STATUS_14",
            "NCCL_ASYNC_FAILURE",
            "EPOCH_NCCL_ASYNC_FAILURE",
        ] {
            let error =
                annotate_device_logical_failure::<()>(Err(cause.into()), &terminal).unwrap_err();
            assert!(error.starts_with("GROUP_OWNER_OR_PRE_OWNER_FATAL:DEVICE_LOGICAL_FATAL"));
            assert!(
                error.ends_with(cause),
                "original failure discarded: {error}"
            );
        }
        assert!(annotate_device_logical_failure(Ok(()), &terminal).is_ok());
        assert_eq!(
            annotate_device_logical_failure::<()>(Err("unknown".into()), std::ptr::null())
                .unwrap_err(),
            "unknown"
        );
    }

    #[test]
    fn constructor_preserves_numeric_failure_without_vendor_message() {
        let _native_scope = NativeCallMarker::enter(line!());
        unsafe extern "C" fn unused_destroy(_: *mut c_void) {
            let _native_scope = NativeCallMarker::enter(line!());
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
        let _native_scope = NativeCallMarker::enter(line!());
        if self.0.is_null() {
            return;
        }
        unsafe {
            let trace = std::env::var_os("MGBFS_TRACE_FAILURE_TEARDOWN").is_some();
            if trace {
                eprintln!(
                    "MGBFS_FAILURE_TEARDOWN stream={:?} stage=stream_drain_begin",
                    self.0
                );
            }
            traced_stream_synchronize(self.0);
            if trace {
                eprintln!(
                    "MGBFS_FAILURE_TEARDOWN stream={:?} stage=stream_drain_end",
                    self.0
                );
            }
            cudaStreamDestroy(self.0);
        }
    }
}
struct Event(*mut c_void);
impl Event {
    fn new() -> Result<Self> {
        let _native_scope = NativeCallMarker::enter(line!());
        let mut p = std::ptr::null_mut();
        check(observed_native!(unsafe {
            cudaEventCreateWithFlags(&mut p, 2)
        }))?;
        Ok(Self(p))
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        let _native_scope = NativeCallMarker::enter(line!());
        unsafe {
            cudaEventDestroy(self.0);
        }
    }
}
// Diagnostic-only host snapshot. The preloaded probe returns a process-lifetime
// string literal; this path performs no CUDA/NCCL call or cancellation action.
#[cfg(target_os = "linux")]
#[link(name = "dl")]
extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
}
#[cfg(target_os = "linux")]
fn report_cancel_native_site(sample: u32) {
    unsafe {
        extern "C" {
            fn mgbfs_owner_fatal_gate_stage_snapshot() -> u32;
        }
        let gate_stage = mgbfs_owner_fatal_gate_stage_snapshot();
        let symbol = dlsym(
            std::ptr::null_mut(),
            b"mgbfs_failure_stage_snapshot\0".as_ptr().cast(),
        );
        let api = if symbol.is_null() {
            "probe_unavailable".into()
        } else {
            let snapshot: unsafe extern "C" fn() -> *const std::ffi::c_char =
                std::mem::transmute(symbol);
            let text = snapshot();
            if text.is_null() {
                "none".into()
            } else {
                CStr::from_ptr(text).to_string_lossy()
            }
        };
        eprintln!("MGBFS_CANCEL_NATIVE_SITE rank={} sample={} api={} owner_gate_stage={} rust_native_site={}",
            std::env::var("RANK").unwrap_or_default(), sample, api, gate_stage, NATIVE_CALL_SITE.load(std::sync::atomic::Ordering::Acquire));
    }
}

// A communicator created during setup must abort peers if a later local
// allocation fails before the final constructor agreement.
struct Comm(*mut c_void, bool);
impl Drop for Comm {
    fn drop(&mut self) {
        if !self.1 && crate::session_cache::comm_put(self.0) { return; }
        let _native_scope = NativeCallMarker::enter(line!());
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
}
fn append_extent(extents: &mut Vec<Extent>, mut extent: Extent) -> Result<()> {
    let _native_scope = NativeCallMarker::enter(line!());
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
        let _native_scope = NativeCallMarker::enter(line!());
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
                Some(HashFirstDevice {
                    counts: b("device_counts")?,
                    extents: b("device_extents")?,
                    controls: b("device_controls")?,
                    exchange_counts: b("device_exchange_counts")?,
                })
            } else {
                None
            },
            ledger,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ShardAbShape { maximum: u32, capacity: u32, slots: u32, forced: Option<u32> }
impl ShardAbShape {
    fn new(layer_capacity: u32) -> Result<Self> {
        fn setting(name: &str, default: u32) -> Result<u32> {
            std::env::var(name).ok().map(|v| v.parse::<u32>().map_err(|_| name.to_string()))
                .transpose().map(|v| v.unwrap_or(default))
        }
        let target = setting("MGBFS_SHARD_AB_CAPACITY", 1 << 20)?;
        if target == 0 || !target.is_power_of_two() { return Err("SHARD_AB_CAPACITY".into()); }
        let minimum = (u64::from(layer_capacity)+255)/256;
        let capacity = target.min(layer_capacity.max(1).checked_next_power_of_two().ok_or("SHARD_AB_SHAPE")?)
            .max(u32::try_from(minimum).map_err(|_| "SHARD_AB_SHAPE")?.checked_next_power_of_two().ok_or("SHARD_AB_SHAPE")?);
        let maximum = ((u64::from(layer_capacity)+u64::from(capacity)-1)/u64::from(capacity)).max(1).next_power_of_two() as u32;
        let slots = setting("MGBFS_SHARD_AB_SORT_SLOTS", 2)?;
        if !slots.is_power_of_two() || slots > 32 { return Err("SHARD_AB_SORT_SLOTS".into()); }
        let forced = std::env::var("MGBFS_SHARD_AB_SHARDS").ok().map(|v| v.parse::<u32>().map_err(|_| "SHARD_AB_SHARDS")).transpose()?;
        if forced.is_some_and(|n| !n.is_power_of_two() || n > maximum) { return Err("SHARD_AB_SHARDS".into()); }
        Ok(Self { maximum, capacity, slots, forced })
    }
}
struct ShardAbStorage { handle: *mut c_void, shape: ShardAbShape, input_bound: u32, sorted_merge: bool }
impl Drop for ShardAbStorage {
    fn drop(&mut self) { unsafe { mgbfs_shard_ab_pipeline_destroy(self.handle); } }
}

struct ShardKeyFirst { requests: Buffer, received: Buffer, count: Buffer, device_counts: Buffer, source_begin: Buffer, peer_metadata: Buffer, peer_metadata_enabled: bool, combined_status: bool, reuse_preowner_status: bool }
pub struct DistributedNativeBfs {
    shard_key_first: Option<ShardKeyFirst>,
    shard_ab: Option<ShardAbStorage>,
    batch_graph: Option<crate::batch_graph::BatchGraph>,
    compact_direct_hash: Option<Plan>,
    // Release graph handles before captured plans/buffers during teardown.
    owner_graph_replay: Option<OwnerGraphReplay>,
    #[cfg(feature = "library-owner")]
    library_owner: Option<LibraryOwnerStorage>,
    cfg: DistributedConfig,
    width: usize,
    stride: usize,
    permutation_n: Option<u32>,
    moves: u32,
    candidates: u32,
    weighted_runs: Option<Vec<(u32, u32, u32)>>,
    weighted_extents: Option<Buffer>,
    weighted_owner_refs: Option<WeightedOwnerRefs>,
    depth: u32,
    current_count: u32,
    prev_count: u32,
    failed: bool,
    stream: Stream,
    generation_stream: Stream,
    pack_done: Event,
    generation_sequence: u64,
    dense_lookahead: u64,
    route_bank_reuses: u64,
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
    terminal: *const std::sync::atomic::AtomicU32,
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
    route_banks: Vec<RouteBank>,
    active_route_bank: usize,
    prefetched: std::collections::VecDeque<(ParentBatch, u64, usize)>,
    archive_hashes: Buffer,
    archive_states: Buffer,
    owner_window: Option<Buffer>,
    native_rank: Option<NativeRankStorage>,
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
    pub fn session_parkable(&self) -> bool {
        (unsafe { mgbfs_nccl_session_park(self.comm.0) }) == 0
    }
    pub fn shard_ab_geometry(&self) -> Option<[u32; 3]> {
        self.shard_ab.as_ref().map(|ab| [ab.shape.maximum,ab.shape.capacity,ab.shape.slots])
    }
    pub fn batch_graph_stats(&self) -> Result<Option<serde_json::Value>> {
        self.batch_graph.as_ref().map(|graph| graph.stats()).transpose()
    }

    /// StateReady publication on the sole owner stream. No host count/control read.
    fn enqueue_weighted_extent_register(
        &self,
        extent: *const Extent,
        control: *mut Control,
        target_depth: u32,
        provisional: bool,
    ) -> Result<()> {
        let records = self
            .weighted_extents
            .as_ref()
            .ok_or("WEIGHTED_RING_NOT_ADMITTED")?;
        check(observed_native!(unsafe {
            mgbfs_weighted_extent_register(
                self.ring.ptr.cast(),
                control,
                extent,
                records.ptr.cast(),
                target_depth,
                u32::from(provisional),
                self.stream.0,
            )
        }))
    }
    /// Caller joins all actual generation/origin/transport/archive readers first.
    fn enqueue_weighted_extent_retire_after_readers(
        &self,
        extent: Extent,
        rows: u64,
    ) -> Result<()> {
        let records = self
            .weighted_extents
            .as_ref()
            .ok_or("WEIGHTED_RING_NOT_ADMITTED")?;
        check(observed_native!(unsafe {
            mgbfs_weighted_extent_retire(
                self.ring.ptr.cast(),
                records.ptr.cast(),
                extent,
                rows,
                self.stream.0,
            )
        }))
    }
    /// Settlement's final reader must join before releasing provisional storage.
    /// Current compacted survivors at the same target depth remain live.
    fn enqueue_weighted_discard_depth_after_readers(&self, target_depth: u32) -> Result<()> {
        let records = self
            .weighted_extents
            .as_ref()
            .ok_or("WEIGHTED_RING_NOT_ADMITTED")?;
        check(observed_native!(unsafe {
            mgbfs_weighted_discard_depth(
                self.ring.ptr.cast(),
                records.ptr.cast(),
                target_depth,
                self.stream.0,
            )
        }))
    }
    /// FinalizeDepth-only compact -> settle DAG. Every input StateRef remains
    /// leased until the later shared-ring promotion reader finishes. This node
    /// never releases a target tag, descriptor or state range on its own.
    fn queue_weighted_settlement(&self, target: u32) -> Result<()> {
        let refs = self
            .weighted_owner_refs
            .as_ref()
            .ok_or("WEIGHTED_OWNER_REFS_MISSING")?;
        let slot = target as usize % refs.targets.len();
        if refs.targets[slot].is_some() && refs.targets[slot] != Some(target) {
            return Err("WEIGHTED_SETTLE_TARGET_ALIAS".into());
        }
        let records = self.cfg.buckets as usize * self.cfg.bucket_capacity as usize;
        unsafe {
            if let Some(owner) = self.owner.as_ref() {
                check(observed_native!(mgbfs_compact_hash_refs_layer(
                    owner.accepted.at(slot * records * 16),
                    refs.accepted.at(slot * records * 8).cast(),
                    owner
                        .lengths
                        .at(slot * self.cfg.buckets as usize * 4)
                        .cast(),
                    self.cfg.buckets,
                    self.cfg.bucket_capacity,
                    refs.compact_hashes.ptr,
                    refs.compact_refs.ptr.cast(),
                    self.cfg.layer_capacity,
                    self.directory.ptr.cast(),
                    refs.compact_count.ptr.cast(),
                    self.fatal.ptr.cast(),
                    self.stream.0,
                )))?;
            } else {
                #[cfg(feature = "library-owner")]
                {
                    let library = self.library_owner.as_ref().ok_or("LIBRARY_OWNER_MISSING")?;
                    let handle = *library
                        .weighted_ranks
                        .get(slot)
                        .ok_or("WEIGHTED_LIBRARY_TARGET_SLOT")?;
                    check(observed_native!(mgbfs_library_rank_export_sorted_refs_v1(
                        handle,
                        refs.compact_hashes.ptr,
                        refs.compact_refs.ptr.cast(),
                        refs.compact_count.ptr.cast(),
                        self.cfg.layer_capacity,
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                    )))?;
                }
                #[cfg(not(feature = "library-owner"))]
                return Err("WEIGHTED_LIBRARY_NOT_BUILT".into());
            }
            check(observed_native!(mgbfs_owner_import_transport_fatal(
                self.fatal.ptr.cast(),
                self.ring.ptr.cast(),
                self.control.ptr.cast(),
                self.stream.0,
            )))?;
            check(observed_native!(mgbfs_macro_settle_run(
                refs.settle.0,
                refs.compact_hashes.ptr,
                refs.compact_refs.ptr.cast(),
                refs.compact_count.ptr.cast(),
                refs.history.ptr,
                refs.history_counts.ptr.cast(),
                refs.survivor_hashes.ptr,
                refs.survivor_refs.ptr.cast(),
                refs.survivor_count.ptr.cast(),
                refs.settle_state.ptr.cast(),
                u64::from(target) + 1,
                self.stream.0,
            )))?;
            let state = refs.settle_state.ptr.cast::<MacroSettleState>();
            check(observed_native!(mgbfs_owner_import_transport_fatal(
                std::ptr::addr_of!((*state).fatal),
                self.ring.ptr.cast(),
                self.control.ptr.cast(),
                self.stream.0,
            )))?;
        }
        Ok(())
    }
    /// Reserve and densely materialize the settled layer in the SAME ring.
    /// Sorting StateRefs preserves monotonic reads; only after the copy's last
    /// reader may provisional descriptors be discarded. Host target tags stay
    /// bound until the complete FinalizeDepth validation succeeds.
    fn queue_weighted_promotion(&self, target: u32) -> Result<()> {
        self.queue_weighted_settlement(target)?;
        let refs = self
            .weighted_owner_refs
            .as_ref()
            .ok_or("WEIGHTED_OWNER_REFS_MISSING")?;
        let records = self
            .weighted_extents
            .as_ref()
            .ok_or("WEIGHTED_RING_NOT_ADMITTED")?;
        let extent = self.extent.ptr.cast::<Extent>();
        unsafe {
            check(observed_native!(cudaMemsetAsync(
                refs.promotion_count.ptr,
                0,
                4,
                self.stream.0,
            )))?;
            check(observed_native!(mgbfs_device_store_u32(
                self.control.at(4).cast(),
                1,
                self.stream.0,
            )))?;
            check(observed_native!(cudaMemcpyAsync(
                self.control.at(8),
                refs.survivor_count.ptr,
                4,
                3,
                self.stream.0,
            )))?;
            check(observed_native!(mgbfs_state_reserve_layer(
                self.ring.ptr.cast(),
                self.control.ptr.cast(),
                extent,
                refs.promotion_count.ptr.cast(),
                self.cfg.layer_capacity,
                self.stream.0,
            )))?;
            check(observed_native!(mgbfs_device_store_u32(
                self.control.at(4).cast(),
                2,
                self.stream.0,
            )))?;
            check(observed_native!(mgbfs_state_materialize_weighted_refs(
                refs.materialize.0,
                refs.survivor_refs.ptr.cast(),
                refs.survivor_count.ptr.cast(),
                target,
                self.states.ptr.cast(),
                self.ring.ptr.cast(),
                self.control.ptr.cast(),
                extent,
                records.ptr.cast(),
                self.stream.0,
            )))?;
            // The new current extent has a fresh descriptor generation. Its
            // later retirement must validate that generation, just like the
            // ordinary materialization ready node; never reuse the zero tag.
            check(observed_native!(cudaMemcpyAsync(
                std::ptr::addr_of_mut!((*extent).padding[1]).cast(),
                std::ptr::addr_of!((*extent).descriptor).cast(),
                8,
                3,
                self.stream.0,
            )))?;
        }
        self.enqueue_weighted_extent_register(extent, self.control.ptr.cast(), target, false)?;
        self.enqueue_weighted_discard_depth_after_readers(target)
    }
    fn rank_owner_mode(&self) -> bool {
        let _native_scope = NativeCallMarker::enter(line!());
        if self.native_rank.is_some() || self.shard_ab.is_some() {
            return true;
        }
        #[cfg(feature = "library-owner")]
        if self
            .library_owner
            .as_ref()
            .is_some_and(|owner| owner.rank_mode)
        {
            return true;
        }
        false
    }
    fn prepare_owner_graph_replay(&mut self, stream: *mut c_void) -> Result<()> {
        match std::env::var("MGBFS_OWNER_GRAPH_REPLAY") {
            Err(std::env::VarError::NotPresent) => return Ok(()),
            Ok(value) if value == "1" => {}
            _ => return Err("OWNER_GRAPH_REPLAY_CONFIG".into()),
        }
        if self.native_rank.is_none() || self.hash_first.is_some() {
            return Err("OWNER_GRAPH_REPLAY_REQUIRES_NATIVE_DENSE".into());
        }
        #[cfg(debug_assertions)]
        if std::env::var_os("MGBFS_TEST_OWNER_DAG_CAPTURE").is_some() {
            return Err("OWNER_GRAPH_REPLAY_NESTED_PROBE".into());
        }
        let target_count = self
            .weighted_owner_refs
            .as_ref()
            .map_or(1, |refs| refs.targets.len());
        if target_count == 0 || self.route_banks.is_empty() {
            return Err("OWNER_GRAPH_REPLAY_EMPTY_PLAN".into());
        }
        let input_count = self
            .route_banks
            .len()
            .checked_add(usize::from(self.cfg.world > 1))
            .ok_or("OWNER_GRAPH_INPUT_OVERFLOW")?;
        let count = target_count
            .checked_mul(input_count)
            .ok_or("OWNER_GRAPH_CAPACITY_OVERFLOW")?;
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(input_count)
            .map_err(|_| "OWNER_GRAPH_INPUT_CAPACITY")?;
        for bank in &self.route_banks {
            inputs.push((
                bank.packed_states.ptr as *const u8,
                bank.sorted_hashes.ptr as *const c_void,
                bank.route_count.ptr as *const u32,
            ));
        }
        if self.cfg.world > 1 {
            inputs.push(if let Some(view) = self.lsa_view {
                (view.states, view.hashes, view.count)
            } else {
                (
                    self.recv_states
                        .as_ref()
                        .ok_or("OWNER_GRAPH_RECEIVE_STATES")?
                        .ptr as *const u8,
                    self.recv_hashes
                        .as_ref()
                        .ok_or("OWNER_GRAPH_RECEIVE_HASHES")?
                        .ptr as *const c_void,
                    self.recv_count
                        .as_ref()
                        .ok_or("OWNER_GRAPH_RECEIVE_COUNT")?
                        .ptr as *const u32,
                )
            });
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| "OWNER_GRAPH_HOST_CAPACITY")?;
        let (mut free_before, mut total) = (0usize, 0usize);
        check(observed_native!(unsafe {
            cudaMemGetInfo(&mut free_before, &mut total)
        }))?;
        for target in 0..target_count {
            for (input, &(states, hashes, source_rows)) in inputs.iter().enumerate() {
                let capture = OwnerCaptureProbe::begin_enabled(stream, true)?
                    .ok_or("OWNER_GRAPH_CAPTURE_MISSING")?;
                self.enqueue_rank_native_tail(
                    states,
                    hashes,
                    source_rows,
                    usize::from(input == self.route_banks.len()),
                    target,
                    self.weighted_runs.is_some(),
                    stream,
                )?;
                let graph = capture.finish()?;
                check(observed_native!(unsafe {
                    cudaGraphUpload(graph.executable, stream)
                }))?;
                entries.push(OwnerGraphEntry {
                    states,
                    hashes,
                    source_rows,
                    graph,
                });
            }
        }
        // Startup only. Upload all executable resources before depth zero.
        // No instantiation/upload/allocation is permitted on replay.
        check(observed_native!(unsafe {
            traced_stream_synchronize(stream)
        }))?;
        let mut free_after = 0usize;
        check(observed_native!(unsafe {
            cudaMemGetInfo(&mut free_after, &mut total)
        }))?;
        if (free_after as u64) < self.cfg.untouched_vram_reserve {
            return Err("OWNER_GRAPH_VRAM_RESERVE".into());
        }
        self.owner_graph_replay = Some(OwnerGraphReplay {
            rank: self.cfg.rank,
            input_count,
            entries,
            launches: std::cell::Cell::new(0),
            // Opaque CUDA graph resources: observed free-VRAM delta, not a
            // byte-exact state/scratch payload claim.
            driver_free_delta_bytes: free_before.saturating_sub(free_after) as u64,
        });
        Ok(())
    }
    fn commit_rank_native_batch(
        &mut self,
        states: *const u8,
        hashes: *const c_void,
        begin: *const u32,
        rows: *const u32,
        source_rows: *const u32,
        group: usize,
        target_depth: Option<u32>,
    ) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        // The target belongs to this packet, not to mutable roulette state.
        // Reject mismatched contracts before capture or any owner mutation.
        match (&self.weighted_runs, target_depth) {
            (None, None) => {}
            (Some(runs), Some(target))
                if runs
                    .iter()
                    .any(|run| self.depth.checked_add(run.0) == Some(target)) => {}
            _ => return Err("WEIGHTED_OWNER_TARGET".into()),
        }
        // Fixed target windows share one StateRing and one ordered owner scratch.
        // A live slot cannot be rebound until FinalizeDepth settles its readers.
        let target_slot = if let Some(target) = target_depth {
            let refs = self
                .weighted_owner_refs
                .as_mut()
                .ok_or("WEIGHTED_OWNER_REFS_MISSING")?;
            if refs.targets.is_empty() {
                return Err("WEIGHTED_OWNER_TARGET_SLOTS".into());
            }
            let slot = target as usize % refs.targets.len();
            match refs.targets[slot] {
                Some(live) if live != target => {
                    return Err("WEIGHTED_OWNER_TARGET_ALIAS".into());
                }
                _ => refs.targets[slot] = Some(target),
            }
            slot
        } else {
            0
        };
        let owner = self.owner.as_ref().ok_or("NATIVE_OWNER_MISSING")?;
        let rank = self.native_rank.as_ref().ok_or("NATIVE_RANK_MISSING")?;
        #[cfg(debug_assertions)]
        let capture = OwnerCaptureProbe::begin(self.stream.0)?;
        let logical_owner = self
            .cfg
            .logical_owner_to_rank
            .iter()
            .position(|&r| r == self.cfg.rank)
            .ok_or("OWNER_MAP")? as u32;
        let s = self.stream.0;
        let extent = self.extent.ptr.cast::<Extent>();
        let target_records = self.cfg.buckets as usize * self.cfg.bucket_capacity as usize;
        let accepted = unsafe { owner.accepted.at(target_slot * target_records * 16) };
        let lengths = unsafe {
            owner
                .lengths
                .at(target_slot * self.cfg.buckets as usize * 4)
        };
        unsafe {
            check(observed_native!(mgbfs_bounded_owner_rank_compare(
                owner.plan.0,
                self.jobs_gpu.ptr.cast(),
                self.cfg.buckets,
                hashes,
                begin,
                rows,
                source_rows,
                self.prev.ptr,
                rank.previous.ptr.cast(),
                self.prev_count.into(),
                self.curr.ptr,
                rank.current.ptr.cast(),
                self.current_count.into(),
                accepted,
                lengths.cast(),
                logical_owner,
                self.cfg.world,
                self.cfg.buckets / self.cfg.shards,
                self.depth,
                owner.counts.ptr.cast(),
                self.control.ptr.cast(),
                self.ring.ptr.cast(),
                s,
            )))?;
        }
        if let Some(replay) = self.owner_graph_replay.as_ref() {
            let input = if group == 0 {
                self.active_route_bank
            } else {
                self.route_banks.len()
            };
            replay.launch(target_slot, input, states, hashes, source_rows, s)?;
        } else {
            self.enqueue_rank_native_tail(
                states,
                hashes,
                source_rows,
                group,
                target_slot,
                target_depth.is_some(),
                s,
            )?;
        }
        // Absolute depth changes across modulo target reuse; never freeze it
        // into graph arguments. Compare's changing history/depth stays outside too.
        if let Some(target) = target_depth {
            self.enqueue_weighted_extent_register(extent, self.control.ptr.cast(), target, true)?;
        }
        #[cfg(debug_assertions)]
        if let Some(capture) = capture {
            capture.launch()?;
        }
        Ok(())
    }
    // Stable device-count owner tail, shared by direct launch and graph replay.
    fn enqueue_rank_native_tail(
        &self,
        states: *const u8,
        hashes: *const c_void,
        source_rows: *const u32,
        group: usize,
        target_slot: usize,
        weighted: bool,
        s: *mut c_void,
    ) -> Result<()> {
        let owner = self.owner.as_ref().ok_or("NATIVE_OWNER_MISSING")?;
        let rank = self.native_rank.as_ref().ok_or("NATIVE_RANK_MISSING")?;
        let extent = self.extent.ptr.cast::<Extent>();
        let selected_count = unsafe { self.control.at(8).cast::<u32>() };
        let target_records = self.cfg.buckets as usize * self.cfg.bucket_capacity as usize;
        let accepted = unsafe { owner.accepted.at(target_slot * target_records * 16) };
        let lengths = unsafe {
            owner
                .lengths
                .at(target_slot * self.cfg.buckets as usize * 4)
        };
        let layer_count = unsafe { self.layer_count.at(target_slot * 4) };
        unsafe {
            check(observed_native!(mgbfs_bounded_owner_rank_metadata(
                owner.counts.ptr.cast(),
                lengths.cast(),
                self.cfg.buckets,
                self.cfg.shards,
                self.cfg.bucket_capacity,
                rank.survivors.ptr.cast(),
                rank.accepted.ptr.cast(),
                rank.capacities.ptr.cast(),
                rank.offsets.ptr.cast(),
                self.control.ptr.cast(),
                s,
            )))?;
            check(observed_native!(mgbfs_state_reserve_rank_batch(
                self.ring.ptr.cast(),
                self.control.ptr.cast(),
                extent,
                rank.survivors.ptr.cast(),
                rank.accepted.ptr.cast(),
                rank.capacities.ptr.cast(),
                self.cfg.shards,
                rank.offsets.ptr.cast(),
                layer_count.cast(),
                self.cfg.layer_capacity,
                self.hash_first.as_ref().map_or(0, |h| h.capacity),
                u32::from(self.hash_first.is_some()),
                s,
            )))?;
            if weighted {
                let refs = self
                    .weighted_owner_refs
                    .as_ref()
                    .ok_or("WEIGHTED_OWNER_REFS_MISSING")?;
                check(observed_native!(mgbfs_bounded_owner_rank_commit_refs(
                    owner.plan.0,
                    self.jobs_gpu.ptr.cast(),
                    self.cfg.buckets,
                    hashes,
                    accepted,
                    lengths.cast(),
                    owner.counts.ptr.cast(),
                    self.control.ptr.cast(),
                    std::ptr::addr_of!((*extent).granted_rows),
                    owner.selected.ptr.cast(),
                    refs.accepted.at(target_slot * target_records * 8).cast(),
                    u64::from(self.cfg.buckets) * u64::from(self.cfg.bucket_capacity),
                    refs.merged.ptr.cast(),
                    u64::from(self.cfg.job_buckets) * u64::from(self.cfg.bucket_capacity),
                    std::ptr::addr_of!((*extent).sequence),
                    s,
                )))?;
            } else {
                check(observed_native!(mgbfs_bounded_owner_rank_commit(
                    owner.plan.0,
                    self.jobs_gpu.ptr.cast(),
                    self.cfg.buckets,
                    hashes,
                    accepted,
                    lengths.cast(),
                    owner.counts.ptr.cast(),
                    self.control.ptr.cast(),
                    std::ptr::addr_of!((*extent).granted_rows),
                    owner.selected.ptr.cast(),
                    s,
                )))?;
            }
            if let Some(h) = self.hash_first.as_ref() {
                let d = h.device.as_ref().ok_or("HASH_FIRST_DEVICE_STORAGE")?;
                check(observed_native!(mgbfs_state_build_rank_requests(
                    states.cast(),
                    source_rows,
                    self.candidates,
                    owner.selected.ptr.cast(),
                    selected_count,
                    h.capacity,
                    h.requests[group].ptr.cast(),
                    h.targets[group].ptr.cast(),
                    d.counts.at(group * 4).cast(),
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    extent,
                    s,
                )))?;
                check(observed_native!(cudaMemcpyAsync(
                    d.extents.at(group * std::mem::size_of::<Extent>()),
                    extent.cast(),
                    std::mem::size_of::<Extent>(),
                    3,
                    s,
                )))?;
                check(observed_native!(cudaMemcpyAsync(
                    d.controls.at(group * std::mem::size_of::<Control>()),
                    self.control.ptr,
                    std::mem::size_of::<Control>(),
                    3,
                    s,
                )))?;
            } else {
                check(observed_native!(mgbfs_state_materialize_rank_batch(
                    states,
                    source_rows,
                    self.candidates,
                    owner.selected.ptr.cast(),
                    selected_count,
                    self.candidates,
                    self.stride as u32,
                    self.states.ptr.cast(),
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    extent,
                    s,
                )))?;
                if !weighted {
                    check(observed_native!(mgbfs_state_publish_next_extent(
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                        extent,
                        self.next_extent_count
                            .as_ref()
                            .ok_or("NEXT_EXTENT_COUNT_MISSING")?
                            .ptr
                            .cast(),
                        self.next_extents
                            .as_ref()
                            .ok_or("NEXT_EXTENTS_MISSING")?
                            .ptr
                            .cast(),
                        2,
                        s,
                    )))?;
                }
            }
        }
        Ok(())
    }
    /// Requested host-pinned payload; CUDA's page/registration overhead is not
    /// part of this byte-exact payload and is measured separately.
    pub fn transport_control_pinned_payload_bytes(&self) -> u64 {
        let _native_scope = NativeCallMarker::enter(line!());
        if self.lsa_view.is_some() {
            4
        } else {
            0
        }
    }
    pub fn abort_group(&mut self) {
        let _native_scope = NativeCallMarker::enter(line!());
        if !self.failed {
            self.failed = true;
            unsafe {
                mgbfs_nccl_abort(self.comm.0);
            }
        }
    }
    pub fn set_cancel_token(
        &mut self,
        token: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        self.cancel_mirror = None;
        check(observed_native!(unsafe {
            mgbfs_nccl_bind_cancel(
                self.comm.0,
                Some(nccl_cancel_probe),
                std::sync::Arc::as_ptr(&token).cast_mut().cast(),
            )
        }))?;
        self.cancel_requested = Some(token.clone());
        {
            let mut word = std::ptr::null_mut();
            let mut device = std::ptr::null_mut();
            check(observed_native!(unsafe {
                mgbfs_nccl_cancel_words(self.comm.0, &mut word, &mut device)
            }))?;
            self.cancel_mirror = Some(unsafe { CancelMirror::new(token.clone(), word)? });
        }
        Ok(())
    }
    pub fn set_failure_token(&mut self, token: std::sync::Arc<std::sync::atomic::AtomicU8>) {
        let _native_scope = NativeCallMarker::enter(line!());
        self.failure_report = Some(token);
    }
    pub fn set_retirement_token(
        &mut self,
        token: std::sync::Arc<crate::bootstrap::SearchRetirement>,
    ) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        check(observed_native!(unsafe {
            mgbfs_nccl_bind_retirement(
                self.comm.0,
                Some(nccl_retirement_probe),
                std::sync::Arc::as_ptr(&token).cast_mut().cast(),
            )
        }))?;
        self.retirement = Some(token);
        Ok(())
    }
    fn queue_owner_fatal_gate(&self, stream: *mut c_void) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        if self.lsa_view.is_some() {
            check(observed_native!(unsafe {
                mgbfs_owner_lsa_fatal_gate(
                    self.comm.0,
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    self.collective_send.ptr.cast(),
                    self.collective_recv.ptr.cast(),
                    stream,
                )
            }))
        } else {
            // The same sticky mapped terminal closes local admission and is
            // observed by epoch-credit/NCCL progress polling. advance() reports
            // failure on the existing TCP sideband before serialized abort.
            // No healthy HOST epoch needs an extra NCCL failure collective.
            check(observed_native!(unsafe {
                mgbfs_cuda::ffi::mgbfs_owner_local_fatal_gate(
                    self.comm.0,
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    self.collective_recv.ptr.cast(),
                    stream,
                )
            }))
        }
    }
    fn ensure_not_cancelled(&self) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        if !self.terminal.is_null() {
            let stopped = unsafe { &*self.terminal }.load(std::sync::atomic::Ordering::Acquire);
            if stopped != 0 {
                return Err(if stopped & 2 != 0 {
                    "GROUP_OWNER_OR_PRE_OWNER_FATAL:DEVICE_LOGICAL_FATAL"
                } else {
                    "TRANSPORT_CANCELLED"
                }
                .into());
            }
        }
        if self
            .cancel_requested
            .as_ref()
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        {
            Err("REMOTE_SEARCH_CANCELLED".into())
        } else {
            Ok(())
        }
    }
    // A stream containing NCCL or LSA work must not trap the dispatcher in
    // cudaStreamSynchronize: only this dispatcher is allowed to abort its
    // communicator after the TCP sideband requests cancellation.
    fn wait_comm_stream(&self, stream: *mut c_void) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        wait_nccl_stream(self.comm.0, stream, self.cancel_requested.as_deref())
    }
    fn wait_epoch_credit(&self, slot: usize) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
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
                        0 | 4 => {}
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
        let _native_scope = NativeCallMarker::enter(line!());
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
        let _native_scope = NativeCallMarker::enter(line!());
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
        let _native_scope = NativeCallMarker::enter(line!());
        Self::new_library_reference_with_owner_and_cancel(
            graph,
            seed,
            id,
            cfg,
            materialization_capacity,
            pool_bytes,
            tensor_generation,
            library_owner,
            None,
            None,
        )
    }
    #[cfg(feature = "library-owner")]
    pub fn new_library_reference_with_owner_and_cancel(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        materialization_capacity: Option<u32>,
        pool_bytes: u64,
        tensor_generation: bool,
        library_owner: ReferenceOwner,
        startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        startup_failure: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        let _native_scope = NativeCallMarker::enter(line!());
        if matches!(library_owner, ReferenceOwner::Native(_)) {
            return Err("REFERENCE_LIBRARY_OWNER".into());
        }
        // CUCO_RANK HASH_FIRST uses the same preallocated device request and
        // extent protocol for both transports. HOST retains exact NCCL sizing;
        // transport-specific admission must not reject the integrated owner.
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
            None,
        )
    }
    /// The 29 shared Buffer allocations, excluding library/profile/transport
    /// allocations. This is not the complete rank memory budget.
    pub fn shared_memory(&self) -> &mgbfs_core::memory::AllocationLedger {
        let _native_scope = NativeCallMarker::enter(line!());
        &self.shared_memory
    }
    /// All explicit runtime/library device allocations. Excludes CUDA/NCCL
    /// internal residency, pinned archive and disk; not full hardware preflight.
    pub fn owned_memory(&self) -> &mgbfs_core::memory::AllocationLedger {
        let _native_scope = NativeCallMarker::enter(line!());
        &self.owned_memory
    }
    /// Additional profile storage, not total rank VRAM. Includes CUB query
    /// results captured before allocation; reserved bytes use 256B alignment.
    pub fn hash_first_memory(&self) -> Option<&mgbfs_core::memory::AllocationLedger> {
        let _native_scope = NativeCallMarker::enter(line!());
        self.hash_first.as_ref().map(|h| &h.ledger)
    }
    pub fn new(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
    ) -> Result<Self> {
        let _native_scope = NativeCallMarker::enter(line!());
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
        let _native_scope = NativeCallMarker::enter(line!());
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
        let _native_scope = NativeCallMarker::enter(line!());
        Self::new_reference_with_owner_and_cancel(
            graph,
            seed,
            id,
            cfg,
            materialization_capacity,
            owner,
            tile_limit,
            false,
            None,
            None,
        )
    }
    pub fn new_reference_with_owner_and_cancel(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        materialization_capacity: Option<u32>,
        owner: OwnerBackend,
        tile_limit: u32,
        tensor_generation: bool,
        startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        startup_failure: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        let _native_scope = NativeCallMarker::enter(line!());
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
            None,
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
        let _native_scope = NativeCallMarker::enter(line!());
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
        let _native_scope = NativeCallMarker::enter(line!());
        Self::new_lrx_multiset_reference_and_cancel(
            word_graph, seed, id, cfg, owner, pool_bytes, None, None,
        )
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
            (ReferenceOwner::Native(native @ (OwnerBackend::CubSortMerge | OwnerBackend::ShardAb)), None) =>
                (native, None),
            (ReferenceOwner::CucoIndexed, Some(bytes)) =>
                (OwnerBackend::CubSortMerge, Some((bytes, owner))),
            (ReferenceOwner::CucoRank, Some(bytes)) =>
                (OwnerBackend::CubSortMerge, Some((bytes, owner))),
            _ => return Err("LRX_MULTISET_OWNER_CONFIG".into()),
        };
        let graph = word_graph.position_action()?;
        Self::new_profile(&graph, seed, id, cfg, None, native, 256, false,
            library, Some(word_graph.start()), startup_cancel, startup_failure, None)
    }
    pub(crate) fn new_weighted_with_owner_and_cancel(
        graph: &MatrixGroup,
        schedule: &mgbfs_core::macro_generators::MacroGeneratorSet,
        seed: [u8; 16],
        id: [u8; 128],
        cfg: DistributedConfig,
        owner: ReferenceOwner,
        pool_bytes: Option<u64>,
        tile_limit: u32,
        startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        startup_failure: Option<std::sync::Arc<std::sync::atomic::AtomicU8>>,
    ) -> Result<Self> {
        let (native, library) = match (owner, pool_bytes) {
            (ReferenceOwner::Native(native), None) => (native, None),
            (ReferenceOwner::CucoRank, Some(bytes)) => {
                (OwnerBackend::CubSortMerge, Some((bytes, owner)))
            }
            _ => return Err("WEIGHTED_OWNER_CONFIG".into()),
        };
        Self::new_profile(
            graph,
            seed,
            id,
            cfg,
            None,
            native,
            tile_limit,
            false,
            library,
            None,
            startup_cancel,
            startup_failure,
            Some(schedule),
        )
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
        macro_operators: Option<&mgbfs_core::macro_generators::MacroGeneratorSet>,
    ) -> Result<Self> {
        let _native_scope = NativeCallMarker::enter(line!());
        let shard_key_first_enabled = std::env::var("MGBFS_SHARD_AB_KEY_FIRST").ok().as_deref()==Some("1");
        if shard_key_first_enabled && (owner_backend!=OwnerBackend::ShardAb || cfg.generation_variant!=5 || cfg.world>8) {
            return Err("SHARD_KEY_FIRST_REQUIRES_COMPACT_NATIVE".into());
        }
        let compact_direct_enabled=std::env::var("MGBFS_COMPACT_DIRECT_HASH").ok().as_deref()==Some("1");
        if compact_direct_enabled && !shard_key_first_enabled { return Err("COMPACT_DIRECT_REQUIRES_KEY_FIRST".into()); }
        let ab_shape = if owner_backend == OwnerBackend::ShardAb {
            if library_options.is_some() || materialization_capacity.is_some() || macro_operators.is_some() {
                return Err("SHARD_AB_REQUIRES_DENSE_NATIVE".into());
            }
            Some(ShardAbShape::new(cfg.layer_capacity)?)
        } else { None };

        if macro_operators.is_some() && materialization_capacity.is_some() {
            return Err("WEIGHTED_HASH_FIRST_NOT_READY".into());
        }
        let mut library_pool_bytes = library_options.map(|(bytes, _)| bytes);
        if ab_shape.is_none() && !(2..=4).contains(&cfg.route_banks) {
            return Err("ROUTE_BANK_CONFIG".into());
        }
        if cfg.epoch_window < 2 {
            return Err("EPOCH_WINDOW_CONFIG".into());
        }
        if cfg.state_descriptor_capacity == 0 {
            return Err("STATE_DESCRIPTOR_CAPACITY".into());
        }
        let epoch_window = if ab_shape.is_some() { crate::reference_launch::inflight_batches(std::env::var("MGBFS_INFLIGHT_BATCHES").ok().as_deref())? } else { cfg.epoch_window };
        if ab_shape.is_some() { cfg.route_banks = 1; }
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
        let requested_local = match std::env::var("LOCAL_RANK") {
            Ok(value) => Some(value.parse::<u32>().map_err(|_| "DEVICE_PLACEMENT_ORDINAL")?),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err("DEVICE_PLACEMENT_ORDINAL".into()),
        };
        let local_device = crate::topology::reference_local_device(cfg.rank,cfg.world,requested_local)?;
        check(observed_native!(unsafe { cudaSetDevice(local_device as i32) }))?;
        #[cfg(target_os = "linux")]
        crate::cuda_loading::verify_driver_before_allocations()?;
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
                if start.len() != graph.rows || start.iter().any(|&x| usize::from(x) >= graph.rows)
                {
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
        let moves = u32::try_from(macro_operators.map_or(graph.generators.len(), |schedule| {
            schedule.transitions.len()
        }))
        .map_err(|_| "CANDIDATE_OVERFLOW")?;
        if moves == 0 {
            return Err("WEIGHTED_GENERATORS_EMPTY".into());
        }
        let weighted_runs = macro_operators.map(|schedule| {
            let mut runs: Vec<(u32, u32, u32)> = Vec::new();
            for (index, transition) in schedule.transitions.iter().enumerate() {
                if let Some(last) = runs.last_mut() {
                    if last.0 == transition.weight {
                        last.2 += 1;
                        continue;
                    }
                }
                runs.push((transition.weight, index as u32, 1));
            }
            runs
        });
        let candidates = cfg.batch.checked_mul(moves).ok_or("CANDIDATE_OVERFLOW")?;
        if candidates > i32::MAX as u32 {
            return Err("CANDIDATE_CAPACITY".into());
        }
        let packet_stride = if materialization_capacity.is_some() {
            16
        } else {
            stride
        };
        #[cfg(feature = "library-owner")]
        if std::env::var("MGBFS_LIBRARY_POOL_AUTOSIZE").ok().as_deref() == Some("1") {
            if macro_operators.is_some() || !matches!(library_options, Some((_, ReferenceOwner::CucoRank))) {
                return Err("POOL_AUTOSIZE_REQUIRES_CUCO_RANK".into());
            }
            let capacity = (u64::from(cfg.buckets / cfg.shards)
                * u64::from(cfg.bucket_capacity)).min(cfg.layer_capacity.into());
            let mut rounded = 0u64;
            check(unsafe { mgbfs_library_rank_pool_query_v1(
                cfg.layer_capacity, capacity as u32, cfg.shards, candidates, &mut rounded,
            ) })?;
            // Fragmentation remains explicitly reserved until hardware peak
            // measurements justify tightening it. No allocation growth in BFS.
            let slack = (64u64 << 20).max(rounded / 20);
            let bytes = rounded.checked_add(slack).and_then(|x| x.checked_add(255))
                .ok_or("POOL_AUTOSIZE_OVERFLOW")? & !255;
            library_pool_bytes = Some(bytes);
        }
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
        // Owner publication is transport-independent for both profiles. HOST
        // retains only the exact-payload sizing boundary, not host owner spans.
        let native_rank_mode = library_pool_bytes.is_none() && ab_shape.is_none();
        let weighted_target_slots = if macro_operators.is_some() {
            weighted_runs
                .as_ref()
                .and_then(|runs| runs.iter().map(|run| run.0).max())
                .filter(|&slots| slots != 0)
                .ok_or("WEIGHTED_TARGET_DEPTH_ZERO")?
        } else {
            0
        };
        let shared_memory = if ab_shape.is_some() {
            crate::distributed_memory::shard_ab_shared_buffers(shared_shape, cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa)?
        } else if library_pool_bytes.is_some() {
            crate::distributed_memory::library_shared_buffers_for_transport(
                shared_shape,
                cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa,
            )?
        } else if native_rank_mode {
            crate::distributed_memory::native_rank_shared_buffers_for_transport(
                shared_shape,
                cfg.shards,
                cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa,
            )?
        } else {
            crate::distributed_memory::shared_buffers(shared_shape)?
        };
        let mut shared_memory =
            if ab_shape.is_some() { crate::distributed_memory::with_compact_route_bank(&shared_memory)? } else { crate::distributed_memory::with_route_banks(&shared_memory, cfg.route_banks)? };
        if weighted_target_slots != 0 {
            // Reuse the admitted owner buffers as flat K-target arrays. There
            // is still one StateRing and one owner scratch, not K state arenas.
            let mut expanded = mgbfs_core::memory::AllocationLedger::new(u64::MAX, 0)?;
            for allocation in &shared_memory.allocations {
                let factor = if matches!(
                    allocation.name.as_str(),
                    "accepted" | "lengths" | "layer_count"
                ) {
                    u64::from(weighted_target_slots)
                } else {
                    1
                };
                let bytes = allocation
                    .payload_bytes
                    .checked_mul(factor)
                    .ok_or("WEIGHTED_TARGET_BYTES")?;
                expanded.add(&allocation.name, bytes, 1, 256)?;
            }
            shared_memory = expanded;
        }
        if macro_operators.is_some() {
            shared_memory.add(
                "weighted_extents",
                u64::from(cfg.state_descriptor_capacity),
                std::mem::size_of::<WeightedExtent>() as u64,
                256,
            )?;
            if native_rank_mode {
                shared_memory.add(
                    "weighted_owner_refs",
                    u64::from(cfg.buckets)
                        .checked_mul(u64::from(cfg.bucket_capacity))
                        .and_then(|records| records.checked_mul(u64::from(weighted_target_slots)))
                        .ok_or("WEIGHTED_TARGET_RECORDS")?,
                    8,
                    256,
                )?;
                shared_memory.add(
                    "weighted_merge_refs",
                    u64::from(cfg.job_buckets) * u64::from(cfg.bucket_capacity),
                    8,
                    256,
                )?;
            } else {
                // Native merge planes are not used by CUCO. Charge the stable
                // placeholder handles explicitly; no unaccounted allocations.
                shared_memory.add("weighted_owner_refs", 1, 1, 256)?;
                shared_memory.add("weighted_merge_refs", 1, 1, 256)?;
            }
        }
        if weighted_target_slots != 0 {
            let history_layers = weighted_target_slots
                .checked_mul(2)
                .ok_or("WEIGHTED_HISTORY_LAYERS")?;
            let mut query = MacroSettleBytes::default();
            check(observed_native!(unsafe {
                mgbfs_macro_settle_query(
                    cfg.layer_capacity,
                    history_layers,
                    cfg.layer_capacity,
                    &mut query,
                )
            }))?;
            for (name, bytes) in [
                ("indices", query.indices),
                ("selected", query.selected),
                ("flags", query.flags),
                ("count", query.count),
                ("scratch", query.scratch),
            ] {
                // These allocations belong to the existing CUDA settle Plan,
                // not to additional Buffer objects. Charge each exactly once.
                shared_memory.add(&format!("weighted_settle_plan.{name}"), bytes, 1, 256)?;
            }
            let mut materialize_query = MaterializeBytes::default();
            check(observed_native!(unsafe {
                mgbfs_materialize_query(
                    stride as u32,
                    cfg.layer_capacity,
                    cfg.layer_capacity,
                    &mut materialize_query,
                )
            }))?;
            for (name, bytes) in [
                ("keys", materialize_query.keys),
                ("sorted", materialize_query.sorted),
                ("indices", materialize_query.indices),
                ("order", materialize_query.order),
                ("scratch", materialize_query.scratch),
            ] {
                shared_memory.add(&format!("weighted_materialize_plan.{name}"), bytes, 1, 256)?;
            }
            shared_memory.add("weighted_promotion_count", 1, 4, 256)?;
            for (name, records, width) in [
                ("weighted_compact_hashes", u64::from(cfg.layer_capacity), 16),
                ("weighted_compact_refs", u64::from(cfg.layer_capacity), 8),
                ("weighted_compact_count", 1, 4),
                (
                    "weighted_survivor_hashes",
                    u64::from(cfg.layer_capacity),
                    16,
                ),
                ("weighted_survivor_refs", u64::from(cfg.layer_capacity), 8),
                ("weighted_survivor_count", 1, 4),
                (
                    "weighted_settle_state",
                    1,
                    std::mem::size_of::<MacroSettleState>() as u64,
                ),
                (
                    "weighted_history",
                    u64::from(history_layers) * u64::from(cfg.layer_capacity),
                    16,
                ),
                ("weighted_history_counts", u64::from(history_layers), 4),
            ] {
                shared_memory.add(name, records, width, 256)?;
            }
        }
        let mut owned_memory = mgbfs_core::memory::AllocationLedger::new(u64::MAX, 0)?;
        for a in &shared_memory.allocations {
            owned_memory.add(&format!("shared.{}", a.name), a.payload_bytes, 1, 256)?;
        }
        if shard_key_first_enabled {
            owned_memory.add("shard_key_first.requests", u64::from(candidates), 4, 256)?;
            owned_memory.add("shard_key_first.received", u64::from(candidates), 4, 256)?;
            owned_memory.add("shard_key_first.count", 1, 4, 256)?;
            owned_memory.add("shard_key_first.device_counts", u64::from(cfg.world), 4, 256)?;
            owned_memory.add("shard_key_first.source_begin", 1, 4, 256)?;
            owned_memory.add("shard_key_first.peer_metadata", 4, 4, 256)?;
        }
        use crate::distributed_memory::append_query;
        use mgbfs_cuda::allocation::{query_generation, query_hash, query_route};
        let hash_first_ledger = if let Some(capacity) = materialization_capacity {
            let mut q = MaterializeBytes::default();
            check(observed_native!(unsafe {
                mgbfs_materialize_query(stride as u32, capacity, cfg.layer_capacity, &mut q)
            }))?;
            let mut l = mgbfs_core::memory::hash_first_reference_ledger(
                width as u64,
                moves.into(),
                capacity.into(),
                stride as u64,
                [q.keys, q.sorted, q.indices, q.order, q.scratch],
            )?;
            if native_rank_mode || matches!(library_options, Some((_, ReferenceOwner::CucoRank))) {
                l.add("device_counts", 2, 4, 256)?;
                l.add(
                    "device_extents",
                    2,
                    std::mem::size_of::<Extent>() as u64,
                    256,
                )?;
                l.add(
                    "device_controls",
                    2,
                    std::mem::size_of::<Control>() as u64,
                    256,
                )?;
                l.add("device_exchange_counts", cfg.world.into(), 4, 256)?;
            }
            for a in &l.allocations {
                owned_memory.add(&format!("hash_first.{}", a.name), a.payload_bytes, 1, 256)?;
            }
            Some(l)
        } else {
            if compact_direct_enabled {
                let mut bytes=0;check(unsafe{mgbfs_compact_map_query(width as u32,moves,cfg.batch,&mut bytes)})?;
                owned_memory.add("generation.compact_map",bytes,1,256)?;
            }else{
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
            }
            if compact_direct_enabled {
                let mut bytes=0;check(unsafe{mgbfs_compact_hash_query(width as u32,moves,cfg.batch,&mut bytes)})?;
                let weights=u64::from(moves)*16*stride as u64;
                let partials=u64::from(cfg.batch)*u64::from(moves)*64;
                if bytes!=weights+16+partials{return Err("COMPACT_DIRECT_LEDGER".into());}
                owned_memory.add("compact_direct.weights",weights,1,256)?;
                owned_memory.add("compact_direct.offsets",16,1,256)?;
                owned_memory.add("compact_direct.partials",partials,1,256)?;
            } else {
            append_query(
                &mut owned_memory,
                "hash",
                &query_hash(width as u32, candidates)?,
            )?;
            }
            None
        };
        append_query(
            &mut owned_memory,
            "archive_hash",
            &query_hash(width as u32, cfg.batch)?,
        )?;
        append_query(&mut owned_memory, "route", &query_route(candidates)?)?;
        if let Some(shape) = ab_shape {
            let mut bytes = 0;
            check(unsafe { mgbfs_shard_ab_indexed_query(shape.maximum, shape.capacity,
                stride as u32, shape.slots, &mut bytes) })?;
            owned_memory.add("owner.shard_ab.fixed_payload", bytes, 1, 256)?;
        } else if let Some(bytes) = library_pool_bytes {
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
            let slot =
                crate::distributed_memory::lsa_symmetric_slot_bytes(candidates, stride as u32)?;
            owned_memory.add("transport.lsa_symmetric_slot", slot, 1, 256)?;
        }
        crate::session_cache::begin(format!("n={};stride={stride};cfg={cfg:?};pool={library_pool_bytes:?};owner={library_options:?};materialize={materialization_capacity:?}", graph.rows));
        let mut raw = std::ptr::null_mut();
        check(observed_native!(unsafe {
            cudaStreamCreateWithFlags(&mut raw, 1)
        }))?;
        let stream = Stream(raw);
        let mut raw_generation = std::ptr::null_mut();
        check(observed_native!(unsafe {
            cudaStreamCreateWithFlags(&mut raw_generation, 1)
        }))?;
        let generation_stream = Stream(raw_generation);
        let pack_done = Event::new()?;
        let mut raw_exchange = std::ptr::null_mut();
        check(observed_native!(unsafe {
            cudaStreamCreateWithFlags(&mut raw_exchange, 1)
        }))?;
        let exchange_stream = Stream(raw_exchange);
        let exchange_done = Event::new()?;
        let mut epoch_completed = Vec::new();
        epoch_completed
            .try_reserve_exact(epoch_window)
            .map_err(|_| "EPOCH_EVENT_CAPACITY")?;
        for _ in 0..epoch_window {
            epoch_completed.push(Event::new()?);
        }
        let mut epoch_outstanding = std::collections::VecDeque::new();
        epoch_outstanding
            .try_reserve_exact(epoch_window)
            .map_err(|_| "EPOCH_CREDIT_CAPACITY")?;
        let mut raw_archive = std::ptr::null_mut();
        check(observed_native!(unsafe {
            cudaStreamCreateWithFlags(&mut raw_archive, 1)
        }))?;
        let archive_stream = Stream(raw_archive);
        let mut comm = if cfg.transport != mgbfs_core::config::ReferenceTransport::Lsa {
            crate::session_cache::comm_take().unwrap_or(std::ptr::null_mut())
        } else { std::ptr::null_mut() };
        let mut error = [0i8; 512];
        if !comm.is_null() {
            if let Some(token) = &startup_cancel {
                check(unsafe { mgbfs_nccl_bind_cancel(comm, Some(nccl_cancel_probe),
                    std::sync::Arc::as_ptr(token).cast_mut().cast()) })?;
            }
        } else if unsafe {
            mgbfs_nccl_create_with_cancel(
                cfg.rank,
                cfg.world,
                local_device,
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
        let mut comm = Comm(comm, true);
        // Declared after Comm: on a constructor error the sideband learns
        // failure before communicator cleanup can wait for a peer.
        let mut startup_report = crate::failure::FailureReportGuard::new(startup_failure);
        let mut terminal_host = std::ptr::null_mut();
        let mut terminal_device = std::ptr::null_mut();
        check(observed_native!(unsafe {
            mgbfs_nccl_cancel_words(comm.0, &mut terminal_host, &mut terminal_device)
        }))?;
        #[cfg(debug_assertions)]
        if std::env::var("MGBFS_TEST_CONSTRUCTOR_FAULT_RANK")
            .ok()
            .and_then(|rank| rank.parse::<u32>().ok())
            == Some(cfg.rank)
        {
            return Err("TEST_INJECTED_CONSTRUCTOR_ERROR".into());
        }
        admit_device_group(
            cfg.rank,
            library_pool_bytes,
            comm.0,
            raw,
            owned_memory.total(),
            cfg.untouched_vram_reserve,
            startup_cancel.as_deref(),
        ).map_err(|error| {
            if error == "MEMORY_QUERY_DONE" {
                // Admission completed its all-rank query rendezvous. Suppress
                // the constructor failure guard for this intentional exit.
                startup_report.disarm();
                comm.1 = false;
            }
            error
        })?;
        let setup_send = Buffer::new(4, raw)?;
        let setup_recv = Buffer::new(4, raw)?;
        let lsa_view = if cfg.transport == mgbfs_core::config::ReferenceTransport::Lsa {
            // Reserve the control words before the potentially large symmetric
            // allocation, so an OOM in prepare can still be voted by all ranks.
            error.fill(0);
            let prepared = check(observed_native!(unsafe {
                mgbfs_nccl_lsa_prepare(
                    comm.0,
                    candidates,
                    stride as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }));
            vote_group_error(
                prepared.map_err(|status| {
                    format!(
                        "LSA_PREPARE_GROUP: {}",
                        crate::failure::attach_native_detail(status, &error)
                    )
                }),
                |failed| {
                    setup_failure_vote(
                        comm.0,
                        raw,
                        &setup_send,
                        &setup_recv,
                        failed,
                        startup_cancel.as_deref(),
                    )
                },
                "LSA_PREPARE_GROUP: peer rejected LSA prepare".into(),
            )
            .map_err(|error| {
                startup_report.publish();
                error
            })?;
            error.fill(0);
            let activated = check(observed_native!(unsafe {
                mgbfs_nccl_lsa_activate(comm.0, error.as_mut_ptr(), error.len())
            }));
            vote_group_error(
                activated.map_err(|status| {
                    format!(
                        "LSA_ACTIVATE_GROUP: {}",
                        crate::failure::attach_native_detail(status, &error)
                    )
                }),
                |failed| {
                    setup_failure_vote(
                        comm.0,
                        raw,
                        &setup_send,
                        &setup_recv,
                        failed,
                        startup_cancel.as_deref(),
                    )
                },
                "LSA_ACTIVATE_GROUP: peer rejected LSA activation".into(),
            )
            .map_err(|error| {
                startup_report.publish();
                error
            })?;
            let (mut count, mut fatal, mut hashes, mut states) = (
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            );
            check(observed_native!(unsafe {
                mgbfs_nccl_lsa_view(comm.0, &mut count, &mut fatal, &mut hashes, &mut states)
            }))?;
            let mut terminal = std::ptr::null_mut();
            check(observed_native!(unsafe {
                mgbfs_nccl_lsa_cancel_word(comm.0, &mut terminal)
            }))?;
            if terminal.is_null()
                || terminal as usize % std::mem::align_of::<std::sync::atomic::AtomicU32>() != 0
            {
                return Err("LSA_CANCEL_WORD_ALIGNMENT".into());
            }
            Some(LsaView {
                count,
                fatal,
                hashes,
                states: states.cast(),
                terminal: terminal.cast(),
            })
        } else {
            None
        };
        let local_result = (|| -> Result<Self> {
            let contract = GemmHash::from_seed(width, seed)?;
            let limbs = contract.limbs();
            let matrices: Vec<u8> = macro_operators.map_or_else(
                || graph.generators.iter().flatten().copied().collect(),
                |schedule| {
                    schedule
                        .transitions
                        .iter()
                        .flat_map(|transition| transition.matrix.iter().copied())
                        .collect()
                },
            );
            let weights = macro_operators.map_or_else(
                || vec![1u32; moves as usize],
                |schedule| {
                    schedule
                        .transitions
                        .iter()
                        .map(|transition| transition.weight)
                        .collect()
                },
            );
        let generate = if materialization_capacity.is_some() {
            None
        } else if compact_direct_enabled {
            let mut permutation=Vec::new();
            for generator in &graph.generators {permutation.extend(encode_permutation_matrix(generator,graph.rows)?);}
            Some(Plan::new(mgbfs_generate_destroy,|out,e|unsafe{mgbfs_compact_map_create(width as u32,moves,cfg.batch,permutation.as_ptr(),out,e,512)})?)
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
        let compact_direct_hash=if compact_direct_enabled {
            let mut permutation=Vec::new();
            for generator in &graph.generators {permutation.extend(encode_permutation_matrix(generator,graph.rows)?);}
            Some(Plan::new(mgbfs_compact_hash_destroy,|out,e|unsafe{
                mgbfs_compact_hash_create(width as u32,moves,cfg.batch,permutation.as_ptr(),limbs.as_ptr(),contract.offsets.as_ptr(),1,out,e,512)
            })?)
        }else{None};
        let hash = if materialization_capacity.is_some() || compact_direct_enabled {
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
            let owner = if library_pool_bytes.is_some() || ab_shape.is_some() {
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
                        OwnerBackend::ShardAb => return 1,
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
            let (next_extents, next_extent_count) =
                if library_pool_bytes.is_some() || native_rank_mode || ab_shape.is_some() {
                    (Some(b("next_extents")?), Some(b("next_extent_count")?))
                } else {
                    (None, None)
                };
        let shard_ab = if let Some(shape) = ab_shape {
            let mut handle = std::ptr::null_mut();
            check(unsafe { mgbfs_shard_ab_indexed_create(shape.maximum, shape.capacity,
                stride as u32, shape.slots, raw, &mut handle) })?;
            if handle.is_null() { return Err("SHARD_AB_CREATE_NULL".into()); }
            Some(ShardAbStorage { handle, shape, input_bound: candidates, sorted_merge: std::env::var("MGBFS_SHARD_AB_DEDUP").ok().as_deref()==Some("SORT_MERGE") })
        } else { None };
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
            check(observed_native!(unsafe {
                cudaEventRecord(archive_done[0].0, raw)
            }))?;
            check(observed_native!(unsafe {
                cudaEventRecord(archive_done[1].0, raw)
            }))?;
            check(observed_native!(unsafe { traced_stream_synchronize(raw) }))?;
            let buckets = cfg.buckets as usize;
            let slots = buckets + 1;
            let directory = b("directory")?;
            let fatal = b("fatal")?;
            let route_count = b("route_count")?;
            route_count.put_u32(current_count)?;
            check(observed_native!(unsafe {
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
            }))?;
            check(observed_native!(unsafe { traced_stream_synchronize(raw) }))?;
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
            let weighted_extents = if macro_operators.is_some() {
                let records = b("weighted_extents")?;
                if current_count != 0 {
                    records.put(&[WeightedExtent {
                        count: 1,
                        phase: 1,
                        ..WeightedExtent::default()
                    }])?;
                }
                Some(records)
            } else {
                None
            };
            let weighted_owner_refs = if macro_operators.is_some() {
                let mut targets = Vec::new();
                targets
                    .try_reserve_exact(weighted_target_slots as usize)
                    .map_err(|_| "WEIGHTED_TARGET_HOST_CAPACITY")?;
                targets.resize(weighted_target_slots as usize, None);
                let mut counts_host = Vec::new();
                counts_host
                    .try_reserve_exact(weighted_target_slots as usize)
                    .map_err(|_| "WEIGHTED_COUNTS_HOST_CAPACITY")?;
                counts_host.resize(weighted_target_slots as usize, 0);
                Some(WeightedOwnerRefs {
                    accepted: b("weighted_owner_refs")?,
                    merged: b("weighted_merge_refs")?,
                    targets,
                    counts_host,
                    settle: Plan::new(mgbfs_macro_settle_destroy, |out, error| unsafe {
                        mgbfs_macro_settle_create(
                            cfg.layer_capacity,
                            weighted_target_slots * 2,
                            cfg.layer_capacity,
                            out,
                            error,
                            512,
                        )
                    })?,
                    materialize: Plan::new(mgbfs_materialize_destroy, |out, error| unsafe {
                        mgbfs_materialize_create(
                            stride as u32,
                            cfg.layer_capacity,
                            cfg.layer_capacity,
                            out,
                            error,
                            512,
                        )
                    })?,
                    promotion_count: b("weighted_promotion_count")?,
                    compact_hashes: b("weighted_compact_hashes")?,
                    compact_refs: b("weighted_compact_refs")?,
                    compact_count: b("weighted_compact_count")?,
                    survivor_hashes: b("weighted_survivor_hashes")?,
                    survivor_refs: b("weighted_survivor_refs")?,
                    survivor_count: b("weighted_survivor_count")?,
                    settle_state: b("weighted_settle_state")?,
                    history: b("weighted_history")?,
                    history_counts: b("weighted_history_counts")?,
                })
            } else {
                None
            };
            if let Some(refs) = weighted_owner_refs.as_ref() {
                // Depth zero is already settled, including on empty ranks.
                check(observed_native!(unsafe {
                    cudaMemcpyAsync(
                        refs.history.ptr,
                        curr.ptr,
                        current_count as usize * 16,
                        3,
                        raw,
                    )
                }))?;
                check(observed_native!(unsafe {
                    mgbfs_device_store_u32(refs.history_counts.ptr.cast(), current_count, raw)
                }))?;
            }
            let mut route_banks = Vec::new();
            route_banks
                .try_reserve_exact(cfg.route_banks)
                .map_err(|_| "ROUTE_BANK_HOST_CAPACITY")?;
            let mut initial_count = Some(route_count);
            for index in 0..cfg.route_banks {
                let name = |field: &str| {
                    if index == 0 {
                        field.to_owned()
                    } else {
                        format!("route_bank_{index}.{field}")
                    }
                };
                route_banks.push(RouteBank {
                    children: b(&name("children"))?,
                    child_hashes: b(&name("child_hashes"))?,
                    sorted_hashes: b(&name("sorted_hashes"))?,
                    sorted_refs: b(&name("sorted_refs"))?,
                    route_count: if index == 0 {
                        initial_count.take().ok_or("ROUTE_INITIAL_COUNT")?
                    } else {
                        b(&name("route_count"))?
                    },
                    packed_states: b(&name("packed_states"))?,
                    owner_counts: b(&name("owner_counts"))?,
                    generation_control: b(&name("generation_control"))?,
                    generation_done: NativeEvent::new()?,
                    last_reader: Event::new()?,
                    weighted_cursor: None,
                });
            }
            let mut prefetched = std::collections::VecDeque::new();
            prefetched
                .try_reserve_exact(cfg.route_banks)
                .map_err(|_| "ROUTE_PREFETCH_HOST_CAPACITY")?;
            let mut result = Self {
                owner_graph_replay: None,
            shard_key_first: if shard_key_first_enabled {Some(ShardKeyFirst {
                requests: Buffer::new(candidates as usize*4,raw)?,
                received: Buffer::new(candidates as usize*4,raw)?, count: Buffer::new(4,raw)?,
                device_counts: Buffer::new(cfg.world as usize*4,raw)?, source_begin: Buffer::new(4,raw)?,
                peer_metadata: Buffer::new(16,raw)?,
                peer_metadata_enabled: std::env::var("MGBFS_SHARD_AB_PEER_METADATA").ok().as_deref()==Some("1"),
                combined_status: cfg.world==2 && std::env::var("MGBFS_SHARD_AB_COMBINED_STATUS").ok().as_deref()==Some("1"),
                reuse_preowner_status: cfg.world==2 && std::env::var("MGBFS_SHARD_AB_PEER_METADATA").ok().as_deref()==Some("1") && std::env::var("MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS").ok().as_deref()==Some("1"),
            })} else {None},

                shard_ab,
                batch_graph: (crate::reference_launch::graph_batches(std::env::var("MGBFS_CUDA_GRAPH_BATCHES").ok().as_deref())? != 0).then(crate::batch_graph::BatchGraph::new).transpose()?,
                #[cfg(feature = "library-owner")]
                library_owner: None,
                cfg: cfg.clone(),
                width,
                stride,
                permutation_n,
                moves,
                candidates,
                weighted_runs,
                weighted_extents,
                weighted_owner_refs,
                depth: 0,
                current_count,
                prev_count: 0,
                failed: false,
                cancel_requested: startup_cancel.clone(),
                cancel_mirror: None,
                terminal: terminal_host.cast(),
                failure_report: None,
                retirement: None,
                // Keep the setup-vote stream alive outside the fallible local
                // result. An error here must not destroy it before peers vote.
                stream: Stream(std::ptr::null_mut()),
                generation_stream,
                pack_done,
                generation_sequence: 0,
                dense_lookahead: 0,
                route_bank_reuses: 0,
                exchange_stream,
                exchange_done,
                owner_consumed: Some(Event::new()?),
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
                compact_direct_hash,
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
                route_banks,
                active_route_bank: 0,
                prefetched,
                archive_hashes: b("archive_hashes")?,
                archive_states: b("archive_states")?,
                owner_window: if library_pool_bytes.is_some() || native_rank_mode || ab_shape.is_some() {
                    Some(b("owner_window")?)
                } else {
                    None
                },
                native_rank: if native_rank_mode {
                    Some(NativeRankStorage {
                        previous: b("rank_prev_directory")?,
                        current: b("rank_curr_directory")?,
                        survivors: b("rank_shard_counts")?,
                        accepted: b("rank_shard_accepted")?,
                        capacities: b("rank_shard_capacities")?,
                        offsets: b("rank_shard_offsets")?,
                    })
                } else {
                    None
                },
                recv_states: (cfg.transport != mgbfs_core::config::ReferenceTransport::Lsa)
                    .then(|| b("recv_states"))
                    .transpose()?,
                recv_hashes: (cfg.transport != mgbfs_core::config::ReferenceTransport::Lsa)
                    .then(|| b("recv_hashes"))
                    .transpose()?,
                recv_count: (cfg.transport != mgbfs_core::config::ReferenceTransport::Lsa)
                    .then(|| b("recv_count"))
                    .transpose()?,
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
                let plane_words = (mgbfs_core::library_memory::CandidateSoaLayout::plan(
                    cfg.layer_capacity.into(),
                )?
                .plane_stride_bytes
                    / 4) as usize;
                // Initial directory was built from the one AoS start key. Rewrite
                // it in-place as SoA before any library view starts borrowing it.
                if current_count != 0 && weighted_target_slots == 0 {
                    for plane in 0..4 {
                        unsafe {
                            check(observed_native!(cudaMemcpyAsync(
                                result.curr.at(plane * plane_words * 4),
                                (&start_hash.0[plane] as *const u32).cast(),
                                4,
                                1,
                                raw,
                            )))?;
                        }
                    }
                    check(observed_native!(unsafe { traced_stream_synchronize(raw) }))?;
                }
                let scratch = Buffer::new(
                    mgbfs_core::library_memory::CandidateSoaLayout::plan(candidates.into())?
                        .allocation_bytes as usize,
                    raw,
                )?;
                let control_transfer = unsafe { ControlTransfer::new(raw)? };
                let mut pool = crate::session_cache::pool_take(pool_bytes).unwrap_or(std::ptr::null_mut());
                if pool.is_null() { check(observed_native!(unsafe {
                    mgbfs_library_pool_create_v1(pool_bytes, cfg.untouched_vram_reserve, &mut pool)
                }))?; }
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
                    weighted_ranks: Vec::new(),
                    rank_mode: matches!(library_options, Some((_, ReferenceOwner::CucoRank))),
                    rank_accepted: std::ptr::null(),
                    logical_owner: cfg
                        .logical_owner_to_rank
                        .iter()
                        .position(|&r| r == cfg.rank)
                        .ok_or("OWNER_MAP")? as u32,
                };
                if matches!(library_options, Some((_, ReferenceOwner::CucoIndexed))) {
                    check(observed_native!(unsafe {
                        mgbfs_library_cuco_workspace_create_v1(
                            candidates,
                            raw,
                            &mut library.cuco_workspace,
                        )
                    }))?;
                }
                for directories in result.curr_dir.chunks(per_shard as usize) {
                    let first = directories[0].begin as u32;
                    let last = directories.last().unwrap();
                    library
                        .current
                        .push(first..(last.begin + last.count) as u32);
                }
                if weighted_target_slots != 0 {
                    if !library.rank_mode {
                        return Err("WEIGHTED_REQUIRES_CUCO_RANK".into());
                    }
                    library
                        .weighted_ranks
                        .try_reserve_exact(weighted_target_slots as usize)
                        .map_err(|_| "WEIGHTED_LIBRARY_HOST_CAPACITY")?;
                    let capacities = vec![capacity; cfg.shards as usize];
                    // Install the owner before fallible target creation so startup
                    // group failure unwinds all completed handles and the fixed pool.
                    result.library_owner = Some(library);
                    let library = result
                        .library_owner
                        .as_mut()
                        .ok_or("LIBRARY_OWNER_MISSING")?;
                    for _ in 0..weighted_target_slots {
                        let mut handle = std::ptr::null_mut();
                        check(observed_native!(unsafe {
                            mgbfs_library_rank_create_weighted_cuco_v1(
                                capacities.as_ptr(),
                                cfg.shards,
                                candidates,
                                library.logical_owner,
                                cfg.world,
                                cfg.layer_capacity,
                                raw,
                                &mut handle,
                            )
                        }))?;
                        if handle.is_null() {
                            return Err("WEIGHTED_LIBRARY_CREATE_NULL".into());
                        }
                        library.weighted_ranks.push(handle);
                    }
                } else if library.rank_mode {
                    library.rank = unsafe {
                        create_rank_owner(
                            &library,
                            &result.prev,
                            &result.curr,
                            cfg.shards,
                            candidates,
                            cfg.world,
                            raw,
                        )?
                    };
                    result.library_owner = Some(library);
                } else {
                    for shard in 0..cfg.shards as usize {
                        unsafe {
                            library.shards.push(LibraryShard::new_window_with_workspace(
                                history_view(&result.prev, plane_words, &library.previous[shard]),
                                history_view(&result.curr, plane_words, &library.current[shard]),
                                capacity,
                                if matches!(library_options, Some((_, ReferenceOwner::CucoIndexed)))
                                {
                                    Some(candidates)
                                } else {
                                    None
                                },
                                library.cuco_workspace,
                                raw,
                            )?);
                        }
                    }
                    result.library_owner = Some(library);
                }
            }
            #[cfg(debug_assertions)]
            if std::env::var("MGBFS_TEST_OWNER_CAPACITY_RANK")
                .ok()
                .and_then(|rank| rank.parse::<u32>().ok())
                == Some(result.cfg.rank)
            {
                // This reservation limit is a captured launch argument.
                result.cfg.layer_capacity = result.cfg.layer_capacity.min(2);
            }
            result.prepare_owner_graph_replay(raw)?;
            Ok(result)
        })();
        let group_failed = match setup_failure_vote(
            comm.0,
            raw,
            &setup_send,
            &setup_recv,
            local_result.is_err(),
            startup_cancel.as_deref(),
        ) {
            Ok(failed) => failed,
            Err(error) => {
                // Notify peers before dropping any resources held by local_result.
                startup_report.publish();
                return Err(local_result.err().unwrap_or(error));
            }
        };
        if group_failed {
            startup_report.publish();
            return Err(local_result
                .err()
                .unwrap_or_else(|| "REMOTE_CONSTRUCTOR_FATAL".into()));
        }
        let mut result = local_result?;
        result.stream = stream;
        result.comm = comm;
        result.comm.1 = false;
        #[cfg(debug_assertions)]
        if std::env::var("MGBFS_TEST_CONSTRUCTOR_LATE_FAULT_RANK")
            .ok()
            .and_then(|rank| rank.parse::<u32>().ok())
            == Some(result.cfg.rank)
        {
            startup_report.publish();
            return Err("TEST_INJECTED_CONSTRUCTOR_LATE_ERROR".into());
        }
        if let Some(token) = startup_cancel {
            result.set_cancel_token(token).map_err(|error| {
                startup_report.publish();
                error
            })?;
        }
        startup_report.disarm();
        #[cfg(debug_assertions)]
        if std::env::var("MGBFS_TEST_OWNER_CAPACITY_RANK")
            .ok()
            .and_then(|rank| rank.parse::<u32>().ok())
            == Some(result.cfg.rank)
        {
            // Test the real device reservation limit after common bootstrap
            // and preallocation. No environment lookup in the batch hot path.
            result.cfg.layer_capacity = result.cfg.layer_capacity.min(2);
            eprintln!(
                "MGBFS_TEST_CAPACITY_ARMED rank={} layer_capacity={}",
                result.cfg.rank, result.cfg.layer_capacity
            );
        }
        Ok(result)
    }
    pub fn depth(&self) -> u32 {
        let _native_scope = NativeCallMarker::enter(line!());
        self.depth
    }
    /// Requested pool suballocation high water, not whole-device VRAM.
    /// Call on the rank's creating device, outside timed GPU stages.
    #[cfg(feature = "library-owner")]
    pub fn library_pool_usage(&self) -> Result<Option<PoolUsageV1>> {
        let _native_scope = NativeCallMarker::enter(line!());
        match self.library_owner.as_ref() {
            None => Ok(None),
            Some(library) => {
                let mut usage = PoolUsageV1::default();
                check(observed_native!(unsafe {
                    mgbfs_library_pool_usage_v1(library.pool, &mut usage)
                }))?;
                Ok(Some(usage))
            }
        }
    }
    pub fn frontier_len(&self) -> u32 {
        let _native_scope = NativeCallMarker::enter(line!());
        self.current_count
    }
    /// Submitted lookahead batches, not a claim of measured GPU overlap.
    pub fn dense_lookahead_batches(&self) -> u64 {
        let _native_scope = NativeCallMarker::enter(line!());
        self.dense_lookahead
    }
    /// Allocated completion-credit window; payload storage is unchanged.
    pub fn epoch_window(&self) -> usize {
        let _native_scope = NativeCallMarker::enter(line!());
        self.epoch_completed.len()
    }
    pub fn route_bank_count(&self) -> usize {
        let _native_scope = NativeCallMarker::enter(line!());
        self.route_banks.len()
    }
    /// Repeated bank admissions within a depth, not a GPU-overlap measurement.
    pub fn route_bank_reuses(&self) -> u64 {
        let _native_scope = NativeCallMarker::enter(line!());
        self.route_bank_reuses
    }
    pub fn state_descriptor_capacity(&self) -> u32 {
        let _native_scope = NativeCallMarker::enter(line!());
        self.cfg.state_descriptor_capacity
    }
    /// Prepare a complete immutable route packet on the producer stream.
    /// The shared radix scratch has exactly one producer stream owner; reuse
    /// of all bank outputs is ordered after the previous payload readers.
    fn enqueue_route_packet(
        &self,
        bank: usize,
        candidates: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        self.enqueue_route_packet_range(bank, 0, candidates, stream)
    }
    fn enqueue_route_packet_range(
        &self,
        bank: usize,
        begin: u32,
        candidates: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        if begin
            .checked_add(candidates)
            .map_or(true, |end| end > self.candidates)
        {
            return Err("ROUTE_PACKET_RANGE".into());
        }
        let bank = self.route_banks.get(bank).ok_or("ROUTE_PACKET_BANK")?;
        let stride = if self.hash_first.is_some() {
            16
        } else {
            self.stride
        };
        unsafe {
            check(observed_native!(mgbfs_route_run(
                self.route.0,
                bank.child_hashes.at(begin as usize * 16),
                self.identity_refs.ptr.cast(),
                bank.sorted_hashes.ptr,
                bank.sorted_refs.ptr.cast(),
                bank.route_count.ptr.cast(),
                candidates,
                self.cfg.prededup as i32,
                stream,
            )))?;
            check(observed_native!(mgbfs_exchange_pack_device_n(
                self.cfg.world,
                stride as u32,
                self.candidates,
                bank.children.at(begin as usize * stride).cast(),
                candidates,
                bank.sorted_hashes.ptr,
                bank.sorted_refs.ptr.cast(),
                bank.route_count.ptr.cast(),
                bank.packed_states.ptr.cast(),
                bank.owner_counts.ptr.cast(),
                stream,
            )))?;
        }
        Ok(())
    }
    /// HOST_SIZED_NCCL must observe exact sizes, but it need not drain the
    /// owner stream to observe a packet produced independently of that owner.
    /// This does NOT make the host-sized backend device-count driven.
    fn wait_route_packet(&mut self, bank: usize, sequence: u64) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            self.ensure_not_cancelled()?;
            if self
                .route_banks
                .get_mut(bank)
                .ok_or("ROUTE_PACKET_BANK")?
                .generation_done
                .poll(sequence)?
            {
                return self.ensure_not_cancelled();
            }
            match unsafe { mgbfs_nccl_poll(self.comm.0) } {
                0 | 4 => {}
                _ => return Err("ROUTE_PACKET_NCCL_ASYNC_FAILURE".into()),
            }
            if std::time::Instant::now() >= deadline {
                return Err("ROUTE_PACKET_TIMEOUT".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    fn enqueue_frontier_generation(&mut self, batch: ParentBatch, bank: usize) -> Result<u64> {
        let _native_scope = NativeCallMarker::enter(line!());
        if bank >= self.route_banks.len() {
            return Err("GENERATION_ROUTE_BANK".into());
        }
        if self.route_banks[bank].weighted_cursor.is_some() {
            return Err("WEIGHTED_GENERATION_READERS_LIVE".into());
        }
        let sequence = self
            .generation_sequence
            .checked_add(1)
            .ok_or("GENERATION_SEQUENCE")?;
        let s = self.generation_stream.0;
        // The frontier range is still live in StateRing. Only the *previous*
        // parent batch may retire while these read-only operations are running.
        unsafe {
            if let Some(h) = self.hash_first.as_ref() {
                check(observed_native!(mgbfs_device_store_u32(
                    h.parent_count.ptr.cast(),
                    batch.count,
                    s,
                )))?;
                let generate = if self.hash_first_tensor_generation {
                    mgbfs_generate_hash_only_tc_admitted
                } else {
                    mgbfs_generate_hash_only
                };
                check(observed_native!(generate(
                    h.n,
                    self.moves,
                    h.modulus,
                    self.stride as u32,
                    self.cfg.batch,
                    self.candidates,
                    self.cfg.rank,
                    batch.sequence,
                    self.states.at(batch.begin as usize * self.stride).cast(),
                    h.generators.ptr.cast(),
                    h.coefficients.ptr.cast(),
                    h.offsets.ptr.cast(),
                    h.parent_count.ptr.cast(),
                    self.route_banks[bank].child_hashes.ptr.cast(),
                    self.route_banks[bank].children.ptr.cast(),
                    // Producer count must not overwrite the current owner's
                    // routed-count before its selected origins are committed.
                    self.route_banks[bank].generation_control.at(4).cast(),
                    self.route_banks[bank].generation_control.ptr.cast(),
                    s,
                )))?;
            } else {
                check(observed_native!(mgbfs_generate_run(
                    self.generate.as_ref().ok_or("DENSE_GENERATOR_MISSING")?.0,
                    self.states.at(batch.begin as usize * self.stride).cast(),
                    self.route_banks[bank].children.ptr.cast(),
                    batch.count,
                    s,
                )))?;
                check(observed_native!(mgbfs_hash_run(
                    self.hash.as_ref().ok_or("DENSE_HASH_MISSING")?.0,
                    self.route_banks[bank].children.ptr.cast(),
                    self.route_banks[bank].child_hashes.ptr.cast(),
                    batch.count * self.moves,
                    s,
                )))?;
            }
            if self.weighted_runs.is_none() {
                self.enqueue_route_packet(bank, batch.count * self.moves, s)?;
            }
            self.route_banks[bank].generation_done.record(sequence, s)?;
        }
        if self.weighted_runs.is_some() {
            self.route_banks[bank].weighted_cursor = Some(WeightedRouteCursor {
                sequence,
                parents: batch.count,
                next_run: 0,
                packet_live: false,
            });
        }
        self.generation_sequence = sequence;
        Ok(sequence)
    }
    fn enqueue_weighted_route_packet(
        &mut self,
        bank: usize,
        sequence: u64,
        parents: u32,
        weight: u32,
    ) -> Result<(u32, u32)> {
        if parents > self.cfg.batch {
            return Err("WEIGHTED_PARENT_CAPACITY".into());
        }
        let runs = self
            .weighted_runs
            .as_ref()
            .ok_or("WEIGHTED_SCHEDULE_MISSING")?;
        let run_index = runs
            .iter()
            .position(|run| run.0 == weight)
            .ok_or("WEIGHTED_WEIGHT_MISSING")?;
        let cursor = self
            .route_banks
            .get(bank)
            .ok_or("ROUTE_PACKET_BANK")?
            .weighted_cursor
            .ok_or("WEIGHTED_GENERATION_MISSING")?;
        if cursor.sequence != sequence {
            return Err("WEIGHTED_GENERATION_SEQUENCE".into());
        }
        if cursor.packet_live {
            return Err("WEIGHTED_ROUTE_READER_NOT_RELEASED".into());
        }
        if cursor.parents != parents {
            return Err("WEIGHTED_GENERATION_PARENT_COUNT".into());
        }
        if cursor.next_run != run_index {
            return Err("WEIGHTED_ROUTE_ORDER".into());
        }
        let run = runs[run_index];
        let begin = run.1.checked_mul(parents).ok_or("CANDIDATE_OVERFLOW")?;
        let count = run.2.checked_mul(parents).ok_or("CANDIDATE_OVERFLOW")?;
        unsafe {
            self.route_banks
                .get_mut(bank)
                .ok_or("ROUTE_PACKET_BANK")?
                .generation_done
                .wait(sequence, self.stream.0)?;
        }
        self.enqueue_route_packet_range(bank, begin, count, self.stream.0)?;
        self.route_banks[bank]
            .weighted_cursor
            .as_mut()
            .ok_or("WEIGHTED_GENERATION_MISSING")?
            .packet_live = true;
        Ok((begin, count))
    }
    /// Caller must join every transport/owner reader onto the owner stream
    /// before release. No host completion/readback is used. Raw generated
    /// children remain leased until ALL weight packets have been consumed.
    fn release_weighted_route_packet(&mut self, bank: usize, sequence: u64) -> Result<()> {
        let runs = self
            .weighted_runs
            .as_ref()
            .ok_or("WEIGHTED_SCHEDULE_MISSING")?;
        let cursor = self
            .route_banks
            .get(bank)
            .ok_or("ROUTE_PACKET_BANK")?
            .weighted_cursor
            .ok_or("WEIGHTED_GENERATION_MISSING")?;
        if cursor.sequence != sequence || !cursor.packet_live {
            return Err("WEIGHTED_RELEASE_SEQUENCE".into());
        }
        if cursor.next_run + 1 == runs.len() {
            self.release_route_generation(bank, sequence)?;
            self.route_banks[bank].weighted_cursor = None;
        } else {
            // Subsequent weight sorting runs on this same owner stream, after
            // the joined readers. The raw producer event is NOT retired here.
            check(unsafe { cudaEventRecord(self.route_banks[bank].last_reader.0, self.stream.0) })?;
            let cursor = self.route_banks[bank]
                .weighted_cursor
                .as_mut()
                .ok_or("WEIGHTED_GENERATION_MISSING")?;
            cursor.next_run += 1;
            cursor.packet_live = false;
        }
        Ok(())
    }
    /// Shared unit-cost/weighted raw-bank retirement. The owner stream has
    /// already joined transport, owner, and materialization readers.
    fn release_route_generation(&mut self, bank: usize, sequence: u64) -> Result<()> {
        let bank = self.route_banks.get_mut(bank).ok_or("ROUTE_PACKET_BANK")?;
        check(observed_native!(unsafe {
            cudaEventRecord(bank.last_reader.0, self.stream.0)
        }))?;
        unsafe {
            bank.generation_done.retire_after_device_barrier(
                sequence,
                self.generation_stream.0,
                bank.last_reader.0,
            )?;
        }
        Ok(())
    }
    /// One producer admission, using only the immutable frontier directory
    /// obtained at FinalizeDepth. No device count or event completion readback.
    fn enqueue_route_prefetch(
        &mut self,
        cursor: &mut ParentCursor,
        produced: &mut usize,
    ) -> Result<bool> {
        let _native_scope = NativeCallMarker::enter(line!());
        if self.prefetched.len() == self.route_banks.len() {
            return Err("ROUTE_PREFETCH_CAPACITY".into());
        }
        let Some(batch) = cursor.take(&self.front, self.cfg.batch)? else {
            return Ok(false);
        };
        let bank = *produced % self.route_banks.len();
        let sequence = self.enqueue_frontier_generation(batch, bank)?;
        if *produced >= self.route_banks.len() {
            self.route_bank_reuses = self
                .route_bank_reuses
                .checked_add(1)
                .ok_or("ROUTE_BANK_REUSE_COUNTER_OVERFLOW")?;
        }
        self.prefetched.push_back((batch, sequence, bank));
        if *produced != 0 {
            self.dense_lookahead = self
                .dense_lookahead
                .checked_add(1)
                .ok_or("GENERATION_COUNTER_OVERFLOW")?;
        }
        *produced = produced.checked_add(1).ok_or("GENERATION_ROUTE_SEQUENCE")?;
        Ok(true)
    }
    fn all_max(&self, value: u32) -> Result<u32> {
        let _native_scope = NativeCallMarker::enter(line!());
        self.collective_send.put_u32(value)?;
        check(observed_native!(unsafe {
            mgbfs_nccl_all_reduce_max_u32(
                self.comm.0,
                self.collective_send.ptr.cast(),
                self.collective_recv.ptr.cast(),
                self.stream.0,
            )
        }))?;
        self.wait_comm_stream(self.stream.0)?;
        self.collective_recv.one()
    }
    fn all_max_ring_fatal(&self) -> Result<u32> {
        let _native_scope = NativeCallMarker::enter(line!());
        check(observed_native!(unsafe {
            mgbfs_state_ring_fatal_vote_word(
                self.ring.ptr.cast(),
                self.collective_send.ptr.cast(),
                self.stream.0,
            )
        }))?;
        check(observed_native!(unsafe {
            mgbfs_nccl_all_reduce_max_u32(
                self.comm.0,
                self.collective_send.ptr.cast(),
                self.collective_recv.ptr.cast(),
                self.stream.0,
            )
        }))?;
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
        target_depth: Option<u32>,
    ) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        // The target belongs to this packet, not to mutable roulette state.
        // Reject mismatched contracts before capture or any owner mutation.
        match (&self.weighted_runs, target_depth) {
            (None, None) => {}
            (Some(runs), Some(target))
                if runs
                    .iter()
                    .any(|run| self.depth.checked_add(run.0) == Some(target)) => {}
            _ => return Err("WEIGHTED_OWNER_TARGET".into()),
        }
        // Fixed target windows share one StateRing and one ordered owner scratch.
        // A live slot cannot be rebound until FinalizeDepth settles its readers.
        let target_slot = if let Some(target) = target_depth {
            let refs = self
                .weighted_owner_refs
                .as_mut()
                .ok_or("WEIGHTED_OWNER_REFS_MISSING")?;
            if refs.targets.is_empty() {
                return Err("WEIGHTED_OWNER_TARGET_SLOTS".into());
            }
            let slot = target as usize % refs.targets.len();
            match refs.targets[slot] {
                Some(live) if live != target => {
                    return Err("WEIGHTED_OWNER_TARGET_ALIAS".into());
                }
                _ => refs.targets[slot] = Some(target),
            }
            slot
        } else {
            0
        };
        let stream = self.stream.0;
        let library = self.library_owner.as_mut().ok_or("LIBRARY_OWNER_MISSING")?;
        let handle = if target_depth.is_some() {
            *library
                .weighted_ranks
                .get(target_slot)
                .ok_or("WEIGHTED_LIBRARY_TARGET_SLOT")?
        } else {
            library.rank
        };
        if handle.is_null() || !library.rank_mode {
            return Err("LIBRARY_RANK_NOT_OPEN".into());
        }
        library.epoch = library
            .epoch
            .checked_add(1)
            .ok_or("LIBRARY_EPOCH_OVERFLOW")?;
        let epoch = library.epoch;
        #[cfg(debug_assertions)]
        let capture = OwnerCaptureProbe::begin(stream)?;
        let mut candidates = CandidatesV1 {
            keys: KeysV1 {
                words: [std::ptr::null(); 4],
                rows: 0,
                reserved: 0,
            },
            source_indices: std::ptr::null(),
        };
        unsafe {
            check(observed_native!(
                mgbfs_library_candidates_from_aos_window_v1(
                    source_hashes,
                    begin,
                    rows,
                    self.candidates,
                    self.candidates,
                    library.scratch.ptr,
                    library.scratch.bytes as u64,
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    stream,
                    &mut candidates,
                )
            ))?;
        }
        let mut batch = RankDeviceBatchV1 {
            high_words: std::ptr::null(),
            valid_rows: std::ptr::null(),
            selected: std::ptr::null(),
            selected_count: std::ptr::null(),
            source_indices: std::ptr::null(),
            accepted_counts: std::ptr::null(),
            accepted_capacities: std::ptr::null(),
            shard_counts: std::ptr::null_mut(),
            shard_offsets: std::ptr::null_mut(),
        };
        unsafe {
            check(observed_native!(mgbfs_library_rank_compare_v1(
                handle,
                epoch,
                candidates,
                rows,
                self.control.ptr.cast(),
                self.ring.ptr.cast(),
                &mut batch,
            )))?;
            check(observed_native!(mgbfs_owner_shard_counts(
                batch.high_words,
                batch.valid_rows,
                batch.selected,
                batch.selected_count,
                self.candidates,
                library.logical_owner,
                self.cfg.world,
                self.cfg.shards,
                batch.shard_counts,
                batch.shard_offsets,
                self.ring.ptr.cast(),
                self.control.ptr.cast(),
                stream,
            )))?;
            check(observed_native!(mgbfs_state_reserve_rank_batch(
                self.ring.ptr.cast(),
                self.control.ptr.cast(),
                self.extent.ptr.cast(),
                batch.shard_counts,
                batch.accepted_counts,
                batch.accepted_capacities,
                self.cfg.shards,
                batch.shard_offsets,
                self.layer_count.at(target_slot * 4).cast(),
                self.cfg.layer_capacity,
                self.hash_first.as_ref().map_or(0, |h| h.capacity),
                u32::from(self.hash_first.is_some()),
                stream,
            )))?;
            check(observed_native!(mgbfs_library_rank_commit_v1(
                handle,
                epoch,
                self.control.ptr.cast(),
                self.ring.ptr.cast(),
                self.extent.ptr.cast(),
            )))?;
            if let Some(h) = self.hash_first.as_ref() {
                let d = h.device.as_ref().ok_or("HASH_FIRST_DEVICE_STORAGE")?;
                check(observed_native!(mgbfs_state_build_rank_requests(
                    source_states.cast(),
                    source_rows,
                    self.candidates,
                    batch.source_indices,
                    batch.selected_count,
                    h.capacity,
                    h.requests[source_group].ptr.cast(),
                    h.targets[source_group].ptr.cast(),
                    d.counts.at(source_group * 4).cast(),
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    self.extent.ptr.cast(),
                    stream,
                )))?;
                check(observed_native!(cudaMemcpyAsync(
                    d.extents.at(source_group * std::mem::size_of::<Extent>()),
                    self.extent.ptr,
                    std::mem::size_of::<Extent>(),
                    3,
                    stream,
                )))?;
                check(observed_native!(cudaMemcpyAsync(
                    d.controls.at(source_group * std::mem::size_of::<Control>()),
                    self.control.ptr,
                    std::mem::size_of::<Control>(),
                    3,
                    stream,
                )))?;
            } else {
                check(observed_native!(mgbfs_state_materialize_rank_batch(
                    source_states,
                    source_rows,
                    self.candidates,
                    batch.source_indices,
                    batch.selected_count,
                    self.candidates,
                    self.stride as u32,
                    self.states.ptr.cast(),
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    self.extent.ptr.cast(),
                    stream,
                )))?;
                if target_depth.is_none() {
                    check(observed_native!(mgbfs_state_publish_next_extent(
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                        self.extent.ptr.cast(),
                        self.next_extent_count
                            .as_ref()
                            .ok_or("NEXT_EXTENT_COUNT_MISSING")?
                            .ptr
                            .cast(),
                        self.next_extents
                            .as_ref()
                            .ok_or("NEXT_EXTENTS_MISSING")?
                            .ptr
                            .cast(),
                        2,
                        stream,
                    )))?;
                }
            }
        }
        if target_depth.is_none() {
            library.rank_accepted = batch.accepted_counts;
        }
        check(observed_native!(unsafe {
            mgbfs_library_rank_complete_v1(handle, epoch)
        }))?;
        if let Some(target) = target_depth {
            self.enqueue_weighted_extent_register(
                self.extent.ptr.cast(),
                self.control.ptr.cast(),
                target,
                true,
            )?;
        }
        #[cfg(debug_assertions)]
        if let Some(capture) = capture {
            capture.launch()?;
        }
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
        let _native_scope = NativeCallMarker::enter(line!());
        if self
            .library_owner
            .as_ref()
            .is_some_and(|library| library.rank_mode)
        {
            return Err("RANK_OWNER_REQUIRES_DEVICE_WINDOW".into());
        }
        let s = self.stream.0;
        self.route_banks[self.active_route_bank]
            .route_count
            .put_u32(rows)?;
        unsafe {
            check(observed_native!(rank_directory(
                self.cfg.world,
                source_hashes,
                self.route_banks[self.active_route_bank]
                    .route_count
                    .ptr
                    .cast(),
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
            )))?;
            check(observed_native!(traced_stream_synchronize(s)))?;
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
                check(observed_native!(mgbfs_library_candidates_from_aos_v1(
                    source_hashes.cast::<u8>().add(begin as usize * 16).cast(),
                    count,
                    self.candidates,
                    library.scratch.ptr,
                    library.scratch.bytes as u64,
                    s,
                    &mut candidates,
                )))?;
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
                check(observed_native!(mgbfs_state_reserve_layer(
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    self.extent.ptr.cast(),
                    self.layer_count.ptr.cast(),
                    self.cfg.layer_capacity,
                    s,
                )))?;
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
                    check(observed_native!(mgbfs_state_build_requests(
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
                    )))?;
                } else {
                    check(observed_native!(mgbfs_state_materialize_packed(
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
                    )))?;
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
        let _native_scope = NativeCallMarker::enter(line!());
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
        self.route_banks[self.active_route_bank]
            .route_count
            .put_u32(rows)?;
        unsafe {
            check(observed_native!(rank_directory(
                self.cfg.world,
                source_hashes,
                self.route_banks[self.active_route_bank]
                    .route_count
                    .ptr
                    .cast(),
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
            )))?;
            check(observed_native!(traced_stream_synchronize(s)))?;
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
                check(observed_native!(mgbfs_bind_owner_jobs(
                    jobs,
                    span.buckets,
                    lengths.cast(),
                    self.cfg.buckets,
                    s,
                )))?;
                check(observed_native!(mgbfs_bounded_owner_compare(
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
                )))?;
                if let Some(h) = self.hash_first.as_ref() {
                    check(observed_native!(traced_stream_synchronize(s)))?;
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
                check(observed_native!(mgbfs_state_reserve_layer(
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    self.extent.ptr.cast(),
                    self.layer_count.ptr.cast(),
                    self.cfg.layer_capacity,
                    s,
                )))?;
                let extent = self.extent.ptr.cast::<Extent>();
                check(observed_native!(mgbfs_bounded_owner_commit(
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
                )))?;
                if let Some(h) = self.hash_first.as_ref() {
                    let offset = h.pending_counts[source_group] as usize;
                    check(observed_native!(mgbfs_state_build_requests(
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
                    )))?;
                } else {
                    check(observed_native!(mgbfs_state_materialize_packed(
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
                    )))?;
                    // All readers of these job descriptors have finished in
                    // this stream. Reuse only this span's first 64-byte slot
                    // for its extent; later spans occupy disjoint slots.
                    check(observed_native!(cudaMemcpyAsync(
                        jobs.cast(),
                        self.extent.ptr,
                        std::mem::size_of::<Extent>(),
                        3,
                        s,
                    )))?;
                    continue;
                }
                check(observed_native!(traced_stream_synchronize(s)))?;
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
            check(observed_native!(unsafe { traced_stream_synchronize(s) }))?;
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
        let _native_scope = NativeCallMarker::enter(line!());
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
            check(observed_native!(unsafe {
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
            }))?;
            check(observed_native!(unsafe { traced_stream_synchronize(s) }))?;
            if self.all_max(h.local_fatal.one::<u32>()?)? != 0 {
                return Err("HASH_FIRST_REQUEST_SORT_FATAL".into());
            }
            let responses = if group == 0 {
                check(observed_native!(unsafe {
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
                }))?;
                check(observed_native!(unsafe { traced_stream_synchronize(s) }))?;
                let fatal = self.all_max(h.local_fatal.one::<u32>()?)?;
                h.group_fatal.put_u32(fatal)?;
                h.outgoing_responses.ptr
            } else {
                self.collective_send.put_u32(count)?;
                check(observed_native!(unsafe {
                    mgbfs_nccl_send_recv(
                        self.comm.0,
                        self.collective_send.ptr,
                        4,
                        self.cfg.rank ^ round,
                        self.recv_count
                            .as_ref()
                            .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                            .ptr,
                        4,
                        s,
                    )
                }))?;
                check(observed_native!(unsafe { traced_stream_synchronize(s) }))?;
                let received = self
                    .recv_count
                    .as_ref()
                    .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
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
                            incoming_count_device: self
                                .recv_count
                                .as_ref()
                                .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                                .ptr
                                .cast(),
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
                check(observed_native!(unsafe { traced_stream_synchronize(s) }))?;
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
                    check(observed_native!(unsafe {
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
                    }))?;
                    check(observed_native!(unsafe { traced_stream_synchronize(s) }))?;
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
    fn materialize_hash_first_device(
        &mut self,
        parent: Option<Extent>,
        offset: u64,
        parents: u32,
        round: u32,
    ) -> Result<()> {
        let _native_scope = NativeCallMarker::enter(line!());
        let h = self.hash_first.as_ref().ok_or("HASH_FIRST_STORAGE")?;
        let d = h.device.as_ref().ok_or("HASH_FIRST_DEVICE_STORAGE")?;
        let view = self.lsa_view;
        let s = self.stream.0;
        let (begin, physical) = parent
            .map(|e| (e.sequence + offset, e.begin + offset))
            .unwrap_or((0, 0));
        // The scheduler retains one local owner round with world=1. There is
        // no remote source or receive lease in that case.
        let peer = if self.cfg.world == 1 {
            self.cfg.rank
        } else {
            self.cfg.rank ^ round
        };
        let logical_peer = self
            .cfg
            .logical_owner_to_rank
            .iter()
            .position(|&rank| rank == peer)
            .ok_or("OWNER_MAP")?;
        let gate = || -> Result<()> { self.queue_owner_fatal_gate(s) };
        let import = |word| -> Result<()> {
            check(observed_native!(unsafe {
                mgbfs_owner_import_transport_fatal(
                    word,
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    s,
                )
            }))
        };
        let peer_counts = |count: *const u32| -> Result<()> {
            unsafe {
                check(observed_native!(cudaMemsetAsync(
                    d.exchange_counts.ptr,
                    0,
                    self.cfg.world as usize * 4,
                    s,
                )))?;
                check(observed_native!(cudaMemcpyAsync(
                    d.exchange_counts.at(logical_peer * 4),
                    count.cast(),
                    4,
                    3,
                    s,
                )))
            }
        };
        for group in usize::from(round > 1)..self.cfg.world.min(2) as usize {
            let count = unsafe { d.counts.at(group * 4).cast::<u32>() };
            let extent = unsafe {
                d.extents
                    .at(group * std::mem::size_of::<Extent>())
                    .cast::<Extent>()
            };
            let control = unsafe {
                d.controls
                    .at(group * std::mem::size_of::<Control>())
                    .cast::<Control>()
            };
            unsafe {
                check(observed_native!(mgbfs_materialize_sort_origins(
                    h.materialize.0,
                    if group == 0 { self.cfg.rank } else { peer },
                    h.requests[group].ptr.cast(),
                    h.targets[group].ptr.cast(),
                    count,
                    h.sorted_requests.ptr.cast(),
                    h.sorted_targets.ptr.cast(),
                    h.local_fatal.ptr.cast(),
                    s,
                )))?;
            }
            #[cfg(debug_assertions)]
            if group == 0 {
                inject_materializer_local_stage(1, self.cfg.rank, h.local_fatal.ptr.cast(), s)?;
            }
            import(h.local_fatal.ptr.cast())?;
            // No peer consumes local-group buffers before its final vote.
            // Regeneration predicates on local_fatal; materialization and
            // publication predicate on the imported sticky ring fatal.
            // Preserve remote gates, receive leases and cancellation credits.
            if group != 0 {
                gate()?;
            }
            // HOST exchanges only the exact selected rows. Counts become host
            // visible here because ncclSend/Recv requires size_t, never to build
            // owner jobs or publish extents. The two counts stay in disjoint
            // preallocated buffers; no maximum-payload padding is introduced.
            let mut host_sizes = None;
            let origins = if group == 0 {
                h.sorted_requests.ptr.cast()
            } else if let Some(view) = view {
                peer_counts(count)?;
                check(observed_native!(unsafe {
                    mgbfs_nccl_lsa_exchange_rows(
                        self.comm.0,
                        std::ptr::null(),
                        h.sorted_requests.ptr,
                        d.exchange_counts.ptr.cast(),
                        self.collective_recv.ptr.cast(),
                        logical_peer as u32,
                        peer,
                        16,
                        s,
                    )
                }))?;
                import(view.fatal)?;
                gate()?;
                view.states.cast()
            } else {
                let receive_count = self
                    .recv_count
                    .as_ref()
                    .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?;
                check(observed_native!(unsafe {
                    mgbfs_nccl_send_recv(
                        self.comm.0,
                        count.cast(),
                        4,
                        peer,
                        receive_count.ptr,
                        4,
                        s,
                    )
                }))?;
                // Cancellable wait: only the dispatcher can abort NCCL after
                // a host/API/archive error arrives through the sideband.
                self.wait_comm_stream(s)?;
                let mut sent = 0u32;
                check(observed_native!(unsafe {
                    traced_device_copy((&mut sent as *mut u32).cast(), count.cast(), 4, 2)
                }))?;
                let received = receive_count.one::<u32>()?;
                if sent > h.capacity || received > h.capacity {
                    return Err("HASH_FIRST_REMOTE_REQUEST_CAPACITY".into());
                }
                host_sizes = Some((sent, received));
                check(observed_native!(unsafe {
                    mgbfs_nccl_send_recv(
                        self.comm.0,
                        h.sorted_requests.ptr,
                        u64::from(sent) * 16,
                        peer,
                        h.received_requests.ptr,
                        u64::from(received) * 16,
                        s,
                    )
                }))?;
                h.received_requests.ptr.cast()
            };
            let regenerate_count = if group == 0 {
                count as *const u32
            } else if let Some(view) = view {
                view.count
            } else {
                self.recv_count
                    .as_ref()
                    .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                    .ptr
                    .cast()
            };
            unsafe {
                check(observed_native!(mgbfs_regenerate_selected(
                    h.n,
                    self.moves,
                    h.modulus,
                    self.stride as u32,
                    h.capacity,
                    self.cfg.rank,
                    begin,
                    parents,
                    self.states.at(physical as usize * self.stride).cast(),
                    h.generators.ptr.cast(),
                    origins,
                    regenerate_count,
                    h.outgoing_responses.ptr.cast(),
                    h.local_fatal.ptr.cast(),
                    s,
                )))?;
            }
            #[cfg(debug_assertions)]
            if group == 0 {
                inject_materializer_local_stage(2, self.cfg.rank, h.local_fatal.ptr.cast(), s)?;
            }
            import(h.local_fatal.ptr.cast())?;
            // No peer consumes local-group buffers before its final vote.
            // Regeneration predicates on local_fatal; materialization and
            // publication predicate on the imported sticky ring fatal.
            // Preserve remote gates, receive leases and cancellation credits.
            if group != 0 {
                gate()?;
            }
            let responses = if group == 0 {
                h.outgoing_responses.ptr as *const u8
            } else if let Some(view) = view {
                // Snapshot the request count into the disjoint preallocated
                // count array before the response exchange overwrites the view.
                peer_counts(view.count)?;
                check(observed_native!(unsafe {
                    mgbfs_nccl_lsa_exchange_rows(
                        self.comm.0,
                        std::ptr::null(),
                        h.outgoing_responses.ptr,
                        d.exchange_counts.ptr.cast(),
                        self.collective_recv.ptr.cast(),
                        logical_peer as u32,
                        peer,
                        self.stride as u32,
                        s,
                    )
                }))?;
                import(view.fatal)?;
                check(observed_native!(unsafe {
                    mgbfs_state_validate_response_count(
                        count,
                        view.count,
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                        s,
                    )
                }))?;
                gate()?;
                view.states
            } else {
                let (sent, received) = host_sizes.ok_or("HASH_FIRST_HOST_SIZES")?;
                // Reply sizes are the inverse of the request exchange, so no
                // second D2H count or host extent/control snapshot is required.
                check(observed_native!(unsafe {
                    mgbfs_nccl_send_recv(
                        self.comm.0,
                        h.outgoing_responses.ptr,
                        u64::from(received) * self.stride as u64,
                        peer,
                        h.incoming_responses.ptr,
                        u64::from(sent) * self.stride as u64,
                        s,
                    )
                }))?;
                h.incoming_responses.ptr.cast()
            };
            unsafe {
                check(observed_native!(mgbfs_state_apply_responses(
                    h.materialize.0,
                    responses,
                    h.sorted_targets.ptr.cast(),
                    count,
                    self.collective_recv.ptr.cast(),
                    self.states.ptr.cast(),
                    self.ring.ptr.cast(),
                    control,
                    extent,
                    s,
                )))?;
                check(observed_native!(mgbfs_state_publish_next_extent(
                    self.ring.ptr.cast(),
                    control,
                    extent,
                    self.next_extent_count
                        .as_ref()
                        .ok_or("NEXT_EXTENT_COUNT_MISSING")?
                        .ptr
                        .cast(),
                    self.next_extents
                        .as_ref()
                        .ok_or("NEXT_EXTENTS_MISSING")?
                        .ptr
                        .cast(),
                    2,
                    s,
                )))?;
            }
            gate()?;
        }
        Ok(())
    }
    fn materialize_shard_selected_device(&self,parent:Option<Extent>,offset:u64,parents:u32,round:u32,group:usize)->Result<()> {
        let h=self.shard_key_first.as_ref().ok_or("SHARD_KEY_FIRST_STORAGE")?;let ab=self.shard_ab.as_ref().ok_or("SHARD_AB_MISSING")?;
        let view=self.lsa_view.ok_or("LSA_VIEW")?;let s=self.stream.0;let physical=parent.map_or(0,|e|e.begin+offset);
        let fatal=unsafe{self.ring.at(48).cast::<u32>()};let peer=self.cfg.rank^round;
        let gate=||->Result<()>{check(unsafe{mgbfs_owner_lsa_fatal_gate(self.comm.0,self.ring.ptr.cast(),self.control.ptr.cast(),self.collective_send.ptr.cast(),self.collective_recv.ptr.cast(),s)})};
        let import=|word|->Result<()>{check(unsafe{mgbfs_owner_import_transport_fatal(word,self.ring.ptr.cast(),self.control.ptr.cast(),s)})};
        check(unsafe{mgbfs_shard_ab_pipeline_row_requests(ab.handle,h.requests.ptr.cast(),h.count.ptr.cast(),self.candidates)})?;gate()?;
        let responses=if group==0 {
            check(unsafe{mgbfs_generate_selected_compact(self.generate.as_ref().ok_or("GENERATION_PLAN")?.0,self.states.at(physical as usize*self.stride).cast(),parents,self.route_banks[0].sorted_refs.ptr.cast(),0,self.candidates,h.requests.ptr.cast(),h.count.ptr.cast(),self.candidates,self.route_banks[0].packed_states.ptr.cast(),fatal,s)})?;
            self.route_banks[0].packed_states.ptr
        }else{
            let logical_peer=self.cfg.logical_owner_to_rank.iter().position(|&rank|rank==peer).ok_or("OWNER_MAP")?;
            unsafe{check(cudaMemsetAsync(h.device_counts.ptr,0,self.cfg.world as usize*4,s))?;check(cudaMemcpyAsync(h.device_counts.at(logical_peer*4),h.count.ptr,4,3,s))?;}
            check(unsafe{mgbfs_nccl_lsa_exchange_rows(self.comm.0,std::ptr::null(),h.requests.ptr,h.device_counts.ptr.cast(),self.collective_recv.ptr.cast(),logical_peer as u32,peer,4,s)})?;import(view.fatal)?;gate()?;
            check(unsafe{mgbfs_owner_window_from_counts(self.cfg.world,self.candidates,logical_peer as u32,self.route_banks[0].owner_counts.ptr.cast(),self.route_banks[0].route_count.ptr.cast(),h.source_begin.ptr.cast(),self.collective_send.ptr.cast(),s)})?;
            check(unsafe{mgbfs_generate_selected_compact_device_base(self.generate.as_ref().ok_or("GENERATION_PLAN")?.0,self.states.at(physical as usize*self.stride).cast(),parents,self.route_banks[0].sorted_refs.ptr.cast(),h.source_begin.ptr.cast(),self.candidates,view.states.cast(),view.count,self.candidates,self.route_banks[0].packed_states.ptr.cast(),fatal,s)})?;
            // Snapshot incoming count before the response exchange overwrites view.
            unsafe{check(cudaMemsetAsync(h.device_counts.ptr,0,self.cfg.world as usize*4,s))?;check(cudaMemcpyAsync(h.device_counts.at(logical_peer*4),view.count.cast(),4,3,s))?;}
            gate()?;
            check(unsafe{mgbfs_nccl_lsa_exchange_rows(self.comm.0,std::ptr::null(),self.route_banks[0].packed_states.ptr,h.device_counts.ptr.cast(),self.collective_recv.ptr.cast(),logical_peer as u32,peer,self.stride as u32,s)})?;import(view.fatal)?;gate()?;
            view.states.cast()
        };
        gate()?;
        let response_count=if group==0{h.count.ptr.cast()}else{view.count};
        check(unsafe{mgbfs_shard_ab_pipeline_apply_responses(ab.handle,responses.cast(),response_count)})?;gate()?;Ok(())
    }
    fn materialize_shard_selected(&self,parent:Option<Extent>,offset:u64,parents:u32,round:u32,group:usize,source_base:u32)->Result<()> {
        if self.lsa_view.is_some(){return self.materialize_shard_selected_device(parent,offset,parents,round,group);}
        let h=self.shard_key_first.as_ref().ok_or("SHARD_KEY_FIRST_STORAGE")?;
        let ab=self.shard_ab.as_ref().ok_or("SHARD_AB_MISSING")?;
        let s=self.stream.0;let physical=parent.map_or(0,|e|e.begin+offset);
        let fatal=unsafe{self.ring.at(48).cast::<u32>()};
        check(unsafe{mgbfs_shard_ab_pipeline_row_requests(ab.handle,h.requests.ptr.cast(),h.count.ptr.cast(),self.candidates)})?;
        // The local producer consumes device counts directly. Keep this stream
        // ordered; the following peer-selection vote (or depth finalization on
        // one rank) observes the sticky ring fatal before publishing anything.
        if group==0 {
            check(unsafe{mgbfs_generate_selected_compact(self.generate.as_ref().ok_or("GENERATION_PLAN")?.0,self.states.at(physical as usize*self.stride).cast(),parents,self.route_banks[0].sorted_refs.ptr.cast(),0,self.candidates,h.requests.ptr.cast(),h.count.ptr.cast(),self.candidates,self.route_banks[0].packed_states.ptr.cast(),fatal,s)})?;
            return check(unsafe{mgbfs_shard_ab_pipeline_apply_responses(ab.handle,self.route_banks[0].packed_states.ptr.cast(),h.count.ptr.cast())});
        }
        let peer=self.cfg.rank^round;
        let received=self.recv_count.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?;
        let (count,incoming)=if self.cfg.world==2 && h.peer_metadata_enabled {
            // Two peers receive the same pair of status words. Fatal and both
            // capacities are checked before either issues variable-size traffic.
            // Metadata stays leased on the producer stream until this round drains.
            unsafe {
                check(cudaMemcpyAsync(h.peer_metadata.ptr,h.count.ptr,4,3,s))?;
                check(cudaMemcpyAsync(h.peer_metadata.at(4),fatal.cast(),4,3,s))?;
                check(mgbfs_nccl_send_recv(self.comm.0,h.peer_metadata.ptr,8,peer,h.peer_metadata.at(8),8,s))?;
            }
            self.wait_comm_stream(s)?;
            let m=h.peer_metadata.one::<[u32;4]>()?;
            let error=m[1].max(m[3]);
            if error!=0 {return Err(format!("SHARD_AB_PREPARE_FATAL_{error}"));}
            if m[0]>self.candidates || m[2]>self.candidates {return Err("SHARD_KEY_FIRST_REQUEST_CAPACITY".into());}
            check(unsafe{cudaMemcpyAsync(received.ptr,h.peer_metadata.at(8),4,3,s)})?;
            (m[0],m[2])
        }else{
            let selection_error=self.all_max_ring_fatal()?;
            if selection_error!=0{return Err(format!("SHARD_AB_PREPARE_FATAL_{selection_error}"));}
            let count=h.count.one::<u32>()?;
            check(unsafe{mgbfs_nccl_send_recv(self.comm.0,h.count.ptr,4,peer,received.ptr,4,s)})?;
            self.wait_comm_stream(s)?;
            let incoming=received.one::<u32>()?;
            if self.all_max(u32::from(incoming>self.candidates))?!=0{return Err("SHARD_KEY_FIRST_REQUEST_CAPACITY".into());}
            (count,incoming)
        };
        let response={
            check(unsafe{mgbfs_nccl_send_recv(self.comm.0,h.requests.ptr,u64::from(count)*4,peer,h.received.ptr,u64::from(incoming)*4,s)})?;
            check(unsafe{mgbfs_generate_selected_compact(self.generate.as_ref().ok_or("GENERATION_PLAN")?.0,self.states.at(physical as usize*self.stride).cast(),parents,self.route_banks[0].sorted_refs.ptr.cast(),source_base,self.candidates,h.received.ptr.cast(),received.ptr.cast(),self.candidates,self.route_banks[0].packed_states.ptr.cast(),fatal,s)})?;
            let output=self.recv_states.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?;
            check(unsafe{mgbfs_nccl_send_recv(self.comm.0,self.route_banks[0].packed_states.ptr,u64::from(incoming)*self.stride as u64,peer,output.ptr,u64::from(count)*self.stride as u64,s)})?;
            output.ptr
        };
        if !h.combined_status && self.all_max_ring_fatal()?!=0{return Err("SHARD_KEY_FIRST_REGENERATION_FATAL".into());}
        check(unsafe{mgbfs_shard_ab_pipeline_apply_responses(ab.handle,response.cast(),h.count.ptr.cast())})?;
        Ok(())
    }
    fn globally_small(&self, value: u32) -> Result<bool> {
        self.collective_send.put_u32(value.min(1001))?;
        check(unsafe {mgbfs_nccl_all_reduce_sum_u32(self.comm.0,
            self.collective_send.ptr.cast(),self.collective_recv.ptr.cast(),self.stream.0)})?;
        self.wait_comm_stream(self.stream.0)?;
        Ok(self.collective_recv.one::<u32>()? <= 1000)
    }
    fn archive_unprocessed_snapshot(&mut self,archive:&mut crate::pinned_archive::PinnedArchive,
        view:&[Extent],depth:u32)->Result<()> {
        check(unsafe {cudaStreamSynchronize(self.stream.0)})?;
        check(unsafe {cudaStreamSynchronize(self.generation_stream.0)})?;
        let head=self.ring.one::<Ring>()?.head;
        let quota=1000/self.cfg.world+u32::from(self.cfg.rank<1000%self.cfg.world);
        let mut left=u64::from(quota);let mut saved=0u64;let mut skipped=0u64;let mut unprocessed=0u64;
        let advanced=self.depth;self.depth=depth;
        let copied=(|| {
            for(index,extent) in view.iter().enumerate(){
                let skip=head.saturating_sub(extent.sequence).min(extent.count);
                skipped+=skip;unprocessed+=extent.count-skip;
                let take=left.min(extent.count-skip);
                if take>0 {self.archive_range_inner(archive,extent.begin+skip,take,index,true)?;}
                left-=take;saved+=take;
            }
            archive.layer(u64::from(depth),saved)
        })();self.depth=advanced;copied?;
        // A single pipe write keeps this control record intact across ranks.
        std::io::Write::write_all(&mut std::io::stderr(),format!(
            "MGBFS_INCOMPLETE_SAMPLE rank={} depth={depth} saved={saved} unprocessed={unprocessed} processed={skipped} scope=unprocessed_current_suffix\n",self.cfg.rank).as_bytes())
            .map_err(|e|e.to_string())?;
        Ok(())
    }
    pub fn archive_selected_terminal_snapshot(&mut self,archive:&mut crate::pinned_archive::PinnedArchive)->Result<()> {
        if !archive.selected || self.archived_depth==Some(self.depth) {return Err("SELECTED_SNAPSHOT_PHASE".into());}
        let depth=self.depth;let count=self.current_count;
        let mut view=[Extent::default();2];let len=self.front.len();
        if len>2{return Err("SELECTED_SNAPSHOT_EXTENTS".into());}view[..len].copy_from_slice(&self.front);
        let exported=count.min(1000);let mut left=u64::from(exported);
        for(index,extent) in view[..len].iter().enumerate(){let take=left.min(extent.count);
            if take>0{self.archive_range_inner(archive,extent.begin,take,index,true)?;}left-=take;}
        archive.layer(u64::from(depth),u64::from(exported))?;
        archive.replace_last_layer()?;
        self.archive_unprocessed_snapshot(archive,&view[..len],depth)?;
        self.archived_depth=Some(depth);Ok(())
    }
    pub fn advance_selected(&mut self, archive: &mut crate::pinned_archive::PinnedArchive) -> Result<bool> {
        if !archive.selected || self.front.len() > 2 || self.failed ||
                self.front.iter().map(|e| e.count).sum::<u64>() != u64::from(self.current_count) {
            return Err("SELECTED_ARCHIVE_FRONTIER_CONTRACT".into());
        }
        let depth = self.depth;
        let count = self.current_count;
        let mut view = [Extent::default(); 2];
        let len = self.front.len();
        view[..len].copy_from_slice(&self.front);
        let small = if archive.selected_whole {self.globally_small(count)?} else {count<=1000};
        let exported = if archive.selected_whole && !small {0} else {count.min(1000)};
        let mut remaining = u64::from(exported);
        for (index, extent) in view[..len].iter().enumerate() {
            let take = remaining.min(extent.count);
            if take != 0 { self.archive_range(archive, extent.begin, take, index)?; }
            remaining -= take;
        }
        archive.layer(u64::from(depth), u64::from(exported))?;
        self.archived_depth = Some(depth);
        let result = self.advance();
        // A resource stop happens before publication writes. Its current layer
        // is still physically resident although its parent descriptors retired.
        let resource_stop = result.is_err() && (self.cancel_requested.as_ref().is_some_and(|flag|
            flag.load(std::sync::atomic::Ordering::Acquire)) || result.as_ref().err().is_some_and(|e|
            e.contains("SHARD_AB_PREPARE_FATAL_16") || e.contains("SHARD_AB_PREPARE_FATAL_112") ||
            e.contains("GROUP_STATE_RING_RETIRE_FATAL_16") ||
            e.contains("GROUP_STATE_RING_RETIRE_FATAL_112") ||
            e.contains("LIBRARY_RANK_DEPTH_FATAL_16_16") ||
            e.contains("LIBRARY_RANK_DEPTH_FATAL_0_112") ||
            e.contains("SHARD_AB_PREPARE_FATAL_11") || e.contains("SHARD_AB_PREPARE_FATAL_12") ||
            e.contains("GROUP_STATE_RING_RETIRE_FATAL_11") || e.contains("GROUP_STATE_RING_RETIRE_FATAL_12") ||
            e.contains("LIBRARY_RANK_DEPTH_FATAL_11_11") || e.contains("LIBRARY_RANK_DEPTH_FATAL_12_12")));
        if resource_stop {
            archive.replace_last_layer()?;
            self.archive_unprocessed_snapshot(archive,&view[..len],depth)?;
        }
        let terminal = matches!(result,Ok(false));
        if terminal && !archive.selected_prefix && count>0 && (if archive.selected_whole {!small} else {count>1000}) {
            archive.replace_last_layer()?;
            let advanced_depth = self.depth;
            self.depth = depth;
            let copied = (|| {
                for (index, extent) in view[..len].iter().enumerate() {
                    self.archive_range_inner(archive, extent.begin, extent.count, index, resource_stop)?;
                }
                archive.layer(u64::from(depth), u64::from(count))
            })();
            self.depth = advanced_depth;
            copied?;
        }
        result
    }
    pub fn advance(&mut self) -> Result<bool> {
        let _native_scope = NativeCallMarker::enter(line!());
        if self.failed {
            return Err("DISTRIBUTED_FAILED".into());
        }
        let result = self.advance_inner(None);
        let result = annotate_device_logical_failure(result, self.terminal);
        if result.is_err() { if let Some(graph) = self.batch_graph.as_mut() { graph.cancel(); } }
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
        let _native_scope = NativeCallMarker::enter(line!());
        if self.failed {
            return Err("DISTRIBUTED_FAILED".into());
        }
        let result = self.advance_inner(Some(archive));
        let result = annotate_device_logical_failure(result, self.terminal);
        if result.is_err() { if let Some(graph) = self.batch_graph.as_mut() { graph.cancel(); } }
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
    /// Shared prepared-packet exchange and owner DAG. Raw-bank release and
    /// weighted retirement belong to the caller; the receive slot stays singular.
    fn consume_owner_packet(
        &mut self,
        context: OwnerPacketContext,
        archive_released: &mut [bool; 2],
        mut lsa_owner_recorded: bool,
    ) -> Result<bool> {
        let OwnerPacketContext {
            generation,
            candidate_count,
            target_depth,
            packet_stride,
            batch_index,
            scheduled_rounds,
            trace_route,
            parent,
            extent_offset,
            parents,
            extent_index,
            archive_live,
            archive_dependency_imported,
        } = context;
        let s = self.stream.0;
        let device_epoch = self.rank_owner_mode();
        if self.rank_owner_mode() {
            let window = self.owner_window.as_ref().ok_or("OWNER_WINDOW_MISSING")?;
            let logical_owner = self
                .cfg
                .logical_owner_to_rank
                .iter()
                .position(|&rank| rank == self.cfg.rank)
                .ok_or("OWNER_MAP")? as u32;
            check(observed_native!(unsafe {
                mgbfs_owner_window_from_counts(
                    self.cfg.world,
                    self.candidates,
                    logical_owner,
                    self.route_banks[self.active_route_bank]
                        .owner_counts
                        .ptr
                        .cast(),
                    self.route_banks[self.active_route_bank]
                        .route_count
                        .ptr
                        .cast(),
                    window.ptr.cast(),
                    window.at(4).cast(),
                    s,
                )
            }))?;
        }
        if self.lsa_view.is_some() {
            check(observed_native!(unsafe {
                cudaEventRecord(self.pack_done.0, s)
            }))?;
        } else if let Some(sequence) = generation {
            self.wait_route_packet(self.active_route_bank, sequence)?;
        } else {
            self.wait_comm_stream(s)?;
        }
        let host_ranges = if self.lsa_view.is_none() {
            let routed = self.route_banks[self.active_route_bank]
                .route_count
                .one::<u32>()?;
            let mut owner_counts = [0u32; 8];
            self.route_banks[self.active_route_bank]
                .owner_counts
                .read(&mut owner_counts[..self.cfg.world as usize])?;
            if crate::route_count::packed_count(
                candidate_count,
                &owner_counts[..self.cfg.world as usize],
            )? != routed
            {
                return Err("EXCHANGE_COUNT_MISMATCH".into());
            }
            Some((
                routed,
                crate::route_count::packed_rank_ranges(
                    self.candidates,
                    &owner_counts[..self.cfg.world as usize],
                    &self.cfg.logical_owner_to_rank[..self.cfg.world as usize],
                )?,
            ))
        } else {
            // LSA consumes the device counts and checks their sum/capacity
            // before copying. No D2H count is needed for this route.
            None
        };
        if trace_route {
            eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=route_end routed={:?}", self.cfg.rank, self.depth, host_ranges.as_ref().map(|(rows, _)| rows));
            eprintln!(
                "MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=pack_end",
                self.cfg.rank, self.depth
            );
        }
        let world = self.cfg.world;
        let (local_offset, local_rows) = host_ranges
            .as_ref()
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
            let (remote_offset, remote_rows) = host_ranges
                .as_ref()
                .map_or((0, 0), |(_, ranges)| ranges[exchange_peer as usize]);
            let lsa = self.lsa_view;
            let rank_mode = self.rank_owner_mode();
            let received = if self.cfg.world == 1 {
                0
            } else if lsa.is_some() {
                let logical_owner = self
                    .cfg
                    .logical_owner_to_rank
                    .iter()
                    .position(|&rank| rank == exchange_peer)
                    .ok_or("OWNER_MAP")? as u32;
                check(observed_native!(unsafe {
                    cudaStreamWaitEvent(self.exchange_stream.0, self.pack_done.0, 0)
                }))?;
                // Later peer rounds reuse the same physical receive slot.
                // Their owner event belongs to this batch, unlike the
                // previous-epoch dependency imported before capture.
                if round > 1 && lsa_owner_recorded {
                    check(observed_native!(unsafe {
                        cudaStreamWaitEvent(
                            self.exchange_stream.0,
                            self.owner_consumed
                                .as_ref()
                                .ok_or("LSA_OWNER_EVENT_MISSING")?
                                .0,
                            0,
                        )
                    }))?;
                }
                check(observed_native!(unsafe {
                    mgbfs_nccl_lsa_exchange_rows(
                        self.comm.0,
                        self.route_banks[self.active_route_bank].sorted_hashes.ptr,
                        self.route_banks[self.active_route_bank].packed_states.ptr,
                        self.route_banks[self.active_route_bank]
                            .owner_counts
                            .ptr
                            .cast(),
                        self.collective_recv.ptr.cast(),
                        logical_owner,
                        exchange_peer,
                        packet_stride as u32,
                        self.exchange_stream.0,
                    )
                }))?;
                check(observed_native!(unsafe {
                    cudaEventRecord(self.exchange_done.0, self.exchange_stream.0)
                }))?;
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=lsa_exchange_queued", self.cfg.rank, self.depth);
                }
                // The row count remains on device; no host-sized payload
                // handshake or readback is needed for this peer round.
                0
            } else {
                let communication = self.exchange_stream.0;
                // DENSE packet preparation no longer drains the owner stream.
                // A current archive fatal vote can therefore still be using
                // collective_send: never alias its boolean with a row count.
                // DENSE does not write generation_control; its preallocated
                // second u32 is the bank-local host-size send word. Its lease
                // closes with all transport readers at bank.last_reader.
                // HASH_FIRST sends the already-produced immutable per-owner
                // count, not collective_send (which a fatal vote may still read).
                let count_send = if self.hash_first.is_none() {
                    let control = &self.route_banks[self.active_route_bank].generation_control;
                    if control.bytes < 8 {
                        return Err("ROUTE_COUNT_WORD_CAPACITY".into());
                    }
                    // The second u32 is wholly inside the preallocated bank.
                    unsafe { control.at(4) }
                } else {
                    let logical_peer = self
                        .cfg
                        .logical_owner_to_rank
                        .iter()
                        .position(|&rank| rank == exchange_peer)
                        .ok_or("OWNER_MAP")?;
                    unsafe {
                        self.route_banks[self.active_route_bank]
                            .owner_counts
                            .at(logical_peer * 4)
                    }
                };
                // Size exchange owns only this bank's third control word.
                // The previous owner still reads recv_count and payload;
                // neither is touched until its last-reader event below.
                let control = &self.route_banks[self.active_route_bank].generation_control;
                if control.bytes < 12 {
                    return Err("ROUTE_SIZE_WORD_CAPACITY".into());
                }
                let count_receive = unsafe { control.at(8) };
                // The count is a launch argument, not a borrowed host
                // slice. Publish it on the consuming stream so the NCCL
                // size exchange follows the store without a host drain.
                if self.hash_first.is_none() {
                    check(observed_native!(unsafe {
                        mgbfs_device_store_u32(count_send.cast(), remote_rows, communication)
                    }))?;
                }
                check(observed_native!(unsafe {
                    mgbfs_nccl_send_recv(
                        self.comm.0,
                        count_send,
                        4,
                        exchange_peer,
                        count_receive,
                        4,
                        communication,
                    )
                }))?;
                self.wait_comm_stream(communication)?;
                let mut received = 0u32;
                check(observed_native!(unsafe {
                    traced_device_copy((&mut received as *mut u32).cast(), count_receive, 4, 2)
                }))?;
                if received > self.candidates {
                    return Err("EXCHANGE_CAPACITY".into());
                }
                // The physical receive slot is still singular. Delay its
                // count publication and payload overwrite, not the independent
                // control handshake, until every previous owner reader ends.
                if lsa_owner_recorded {
                    check(observed_native!(unsafe {
                        cudaStreamWaitEvent(
                            communication,
                            self.owner_consumed.as_ref().ok_or("OWNER_EVENT_MISSING")?.0,
                            0,
                        )
                    }))?;
                }
                check(observed_native!(unsafe {
                    cudaMemcpyAsync(
                        self.recv_count
                            .as_ref()
                            .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                            .ptr,
                        count_receive,
                        4,
                        3,
                        communication,
                    )
                }))?;
                // One exact-sized payload group: hash then state. The
                // receive-slot last-reader wait above remains unchanged.
                check(observed_native!(unsafe {
                    mgbfs_nccl_send_recv_pair(
                        self.comm.0,
                        self.route_banks[self.active_route_bank]
                            .sorted_hashes
                            .at(remote_offset as usize * 16),
                        u64::from(remote_rows) * 16,
                        self.route_banks[self.active_route_bank]
                            .packed_states
                            .at(remote_offset as usize * packet_stride),
                        u64::from(remote_rows) * packet_stride as u64,
                        exchange_peer,
                        self.recv_hashes
                            .as_ref()
                            .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                            .ptr,
                        u64::from(received) * 16,
                        self.recv_states
                            .as_ref()
                            .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                            .ptr,
                        u64::from(received) * packet_stride as u64,
                        communication,
                    )
                }))?;
                check(observed_native!(unsafe {
                    cudaEventRecord(self.exchange_done.0, communication)
                }))?;
                received
            };
            if let Some(parent_extent) =
                parent.filter(|_| round == 1 && self.hash_first.is_none() && target_depth.is_none())
            {
                if !archive_dependency_imported
                    && (archive_live
                        || (self.archived_depth == Some(self.depth)
                            && !archive_released[extent_index]))
                {
                    check(observed_native!(unsafe {
                        cudaStreamWaitEvent(s, self.archive_done[extent_index].0, 0)
                    }))?;
                    archive_released[extent_index] = true;
                }
                let mut live = parent_extent;
                live.sequence += extent_offset;
                live.begin = live.sequence % u64::from(self.cfg.state_ring_capacity);
                live.count -= extent_offset;
                live.granted_rows = live.count as u32;
                check(observed_native!(unsafe {
                    mgbfs_state_retire_dense_prefix_value(
                        self.ring.ptr.cast(),
                        live,
                        u64::from(parents),
                        s,
                    )
                }))?;
            }
            if round == 1 && self.hash_first.is_none() {
                if world > 1 {
                    // The vote must follow this round's P2P on the same
                    // communicator, including zero-payload exchange.
                    check(observed_native!(unsafe {
                        cudaStreamWaitEvent(s, self.exchange_done.0, 0)
                    }))?;
                }
                if let Some(view) = lsa {
                    check(observed_native!(unsafe {
                        mgbfs_owner_import_transport_fatal(
                            view.fatal,
                            self.ring.ptr.cast(),
                            self.control.ptr.cast(),
                            s,
                        )
                    }))?;
                }
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=retire_import_queued", self.cfg.rank, self.depth);
                }
                if device_epoch {
                    // The common result poisons ring/control on-device before
                    // any rank can commit this owner epoch. Host/API errors
                    // use the cancellation sideband instead of a batch vote.
                    self.queue_owner_fatal_gate(s)?;
                    if trace_route {
                        eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=preowner_vote_queued", self.cfg.rank, self.depth);
                    }
                } else if self.all_max_ring_fatal()? != 0 {
                    return Err("GROUP_STATE_RING_RETIRE_FATAL".into());
                }
            }
            if round != 1 || self.hash_first.is_some() {
                if let Some(view) = lsa {
                    check(observed_native!(unsafe {
                        cudaStreamWaitEvent(s, self.exchange_done.0, 0)
                    }))?;
                    check(observed_native!(unsafe {
                        mgbfs_owner_import_transport_fatal(
                            view.fatal,
                            self.ring.ptr.cast(),
                            self.control.ptr.cast(),
                            s,
                        )
                    }))?;
                    if device_epoch {
                        self.queue_owner_fatal_gate(s)?;
                    } else if self.all_max_ring_fatal()? != 0 {
                        return Err("GROUP_LSA_TRANSPORT_FATAL".into());
                    }
                } else if device_epoch {
                    check(observed_native!(unsafe {
                        cudaStreamWaitEvent(s, self.exchange_done.0, 0)
                    }))?;
                } else {
                    self.wait_comm_stream(s)?;
                }
            }
            let local_states: *const u8 = unsafe {
                self.route_banks[self.active_route_bank]
                    .packed_states
                    .at(local_offset as usize * packet_stride)
                    .cast()
            };
            let local_hashes: *const c_void = unsafe {
                self.route_banks[self.active_route_bank]
                    .sorted_hashes
                    .at(local_offset as usize * 16)
            };
            let (remote_states, remote_hashes, remote_count): (
                *const u8,
                *const c_void,
                *const u32,
            ) = if let Some(view) = lsa {
                (view.states, view.hashes, view.count)
            } else {
                (
                    self.recv_states
                        .as_ref()
                        .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                        .ptr
                        .cast(),
                    self.recv_hashes
                        .as_ref()
                        .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                        .ptr,
                    self.recv_count
                        .as_ref()
                        .ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?
                        .ptr
                        .cast(),
                )
            };
            let remote_ready = self.exchange_done.0;
            let world = self.cfg.world;
            if trace_route {
                eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} round={round} stage=owner_begin received={received}", self.cfg.rank, self.depth);
            }
            let mut batch_error = process_owner_pair(
                if device_epoch {
                    OwnerFailurePolicy::CancelGroup
                } else {
                    OwnerFailurePolicy::CollectiveVote
                },
                (
                    0,
                    (
                        local_states,
                        local_hashes,
                        if round == 1 { local_rows } else { 0 },
                    ),
                ),
                (1, (remote_states, remote_hashes, received)),
                |(group, (states, hashes, rows))| {
                    // Inject at the shared owner boundary, including the
                    // host-sized HASH_FIRST owner path. Existing error policy
                    // still owns collective failure or cancellation/abort.
                    #[cfg(debug_assertions)]
                    // Multi-rank injection remains on the first remote
                    // owner job. World=1 has no such job, so exercise
                    // the same host-error cancellation on its local job.
                    if round == 1
                        && ((group == 1 && scheduled_rounds > 1) || (world == 1 && group == 0))
                        && TEST_OWNER_HOST_FAULT.with(|flag| flag.replace(false))
                    {
                        return Err("TEST_INJECTED_OWNER_HOST_ERROR".into());
                    }
                    if rank_mode {
                        if !crate::route_count::rank_owner_group_active(world, round, group)? {
                            return Ok(());
                        }
                        let window = self.owner_window.as_ref().ok_or("OWNER_WINDOW_MISSING")?;
                        let (states, hashes, begin, rows, source_rows) = if group == 0 {
                            (
                                self.route_banks[self.active_route_bank].packed_states.ptr
                                    as *const u8,
                                self.route_banks[self.active_route_bank].sorted_hashes.ptr
                                    as *const c_void,
                                window.ptr as *const u32,
                                unsafe { window.at(4) } as *const u32,
                                self.route_banks[self.active_route_bank].route_count.ptr
                                    as *const u32,
                            )
                        } else {
                            (
                                remote_states,
                                remote_hashes,
                                unsafe { window.at(8) } as *const u32,
                                remote_count,
                                remote_count,
                            )
                        };
                        if self.native_rank.is_some() {
                            return self.commit_rank_native_batch(
                                states,
                                hashes,
                                begin,
                                rows,
                                source_rows,
                                group,
                                target_depth,
                            );
                        }
                        #[cfg(feature = "library-owner")]
                        return self.commit_rank_library_batch(
                            states,
                            hashes,
                            begin,
                            rows,
                            source_rows,
                            group,
                            target_depth,
                        );
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
                        check(observed_native!(unsafe {
                            cudaStreamWaitEvent(s, remote_ready, 0)
                        }))?;
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
                if let Some(error) = batch_error {
                    return Err(error);
                }
                // All ranks issue the post-owner vote, including an empty
                // peer round. Its device result predicates the next LSA
                // rendezvous without returning a count to the host.
                self.queue_owner_fatal_gate(s)?;
            }
            if (lsa.is_some() || device_epoch)
                && self.hash_first.is_none()
                && (batch_error.is_none() || device_epoch)
            {
                // Every rank waits for its own receive-slot consumer before
                // entering the next all-rank LSA barrier. A record failure
                // must participate in the existing group error vote.
                let recorded = self
                    .owner_consumed
                    .as_ref()
                    .ok_or_else(|| "LSA_OWNER_EVENT_MISSING".to_string())
                    .and_then(|event| {
                        check(observed_native!(unsafe { cudaEventRecord(event.0, s) }))
                    });
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
                if lsa.is_some() || device_epoch {
                    check(observed_native!(unsafe {
                        cudaEventRecord(
                            self.owner_consumed
                                .as_ref()
                                .ok_or("LSA_OWNER_EVENT_MISSING")?
                                .0,
                            s,
                        )
                    }))?;
                    lsa_owner_recorded = true;
                }
            }
        }
        Ok(lsa_owner_recorded)
    }
    /// Original-depth weighted BFS in the existing rank runtime. The first
    /// admitted driver is one-rank DENSE/native; unsupported transports and
    /// profiles fail before mutation, never reinterpret macro moves as unit cost.
    fn advance_weighted_inner(
        &mut self,
        mut archive: Option<&mut crate::pinned_archive::PinnedArchive>,
    ) -> Result<bool> {
        if self.hash_first.is_some() || !self.rank_owner_mode() {
            return Err("WEIGHTED_DRIVER_MODE_NOT_READY".into());
        }
        let trace_ranges = std::env::var_os("MGBFS_TRACE_RANGES").is_some();
        if trace_ranges && unsafe { mgbfs_trace_ranges_available() } == 0 {
            return Err("PROFILE_NVTX_NOT_BUILT".into());
        }
        if !self.prefetched.is_empty() || !self.epoch_outstanding.is_empty() {
            return Err("WEIGHTED_DEPTH_LEASE_LEAK".into());
        }
        let s = self.stream.0;
        // Native compare consumes directory views. CUCO weighted targets use
        // empty-history membership and the shared original-depth settle window.
        if let Some(rank) = self.native_rank.as_ref() {
            rank.previous.put(&self.prev_dir)?;
            rank.current.put(&self.curr_dir)?;
        }
        check(observed_native!(unsafe {
            cudaMemsetAsync(
                self.owner_window
                    .as_ref()
                    .ok_or("OWNER_WINDOW_MISSING")?
                    .ptr,
                0,
                4,
                s,
            )
        }))?;
        if let Some(a) = archive.as_ref() {
            if self.archived_depth == Some(self.depth) {
                return Err("ARCHIVE_DEPTH_ALREADY_SUBMITTED".into());
            }
            if a.width != self.width && self.permutation_n != u32::try_from(a.width).ok() {
                return Err("ARCHIVE_STATE_WIDTH".into());
            }
        }
        let archive_live = archive.is_some() || self.archived_depth == Some(self.depth);
        // Agree at the semantic boundary only. Exhausted ranks still issue every
        // weight/peer epoch, without healthy-batch counts returned to the host.
        let local_rounds = ParentCursor::round_count(&self.front, self.cfg.batch)?;
        let scheduled_rounds = self.all_max(local_rounds)?;
        check(observed_native!(unsafe {
            cudaMemsetAsync(self.collective_recv.ptr, 0, 4, s)
        }))?;
        let mut lsa_owner_recorded = false;
        let mut archive_released = [false; 2];
        let mut producer_cursor = ParentCursor::default();
        let mut produced = 0usize;
        for _ in 0..self.route_banks.len() {
            if !self.enqueue_route_prefetch(&mut producer_cursor, &mut produced)? {
                break;
            }
        }
        let mut epoch_serial = 0usize;
        for scheduled_round in 0..scheduled_rounds {
            let (batch, sequence, bank) = if let Some(packet) = self.prefetched.pop_front() {
                packet
            } else {
                let bank = scheduled_round as usize % self.route_banks.len();
                let batch = ParentBatch {
                    extent: 0,
                    offset: 0,
                    begin: 0,
                    sequence: 0,
                    count: 0,
                };
                let sequence = self.enqueue_frontier_generation(batch, bank)?;
                (batch, sequence, bank)
            };
            let _batch_range = TraceRange::new(trace_ranges, b"mgbfs.weighted_batch\0");
            self.ensure_not_cancelled()?;
            if self.epoch_outstanding.len() == self.epoch_completed.len() {
                let slot = *self.epoch_outstanding.front().ok_or("EPOCH_CREDIT_EMPTY")?;
                self.wait_epoch_credit(slot)?;
                self.epoch_outstanding.pop_front();
            }
            self.active_route_bank = bank;
            if let Some(a) = archive.as_deref_mut() {
                if batch.count != 0 {
                    self.archive_range(a, batch.begin, u64::from(batch.count), batch.extent)?;
                }
                self.queue_owner_fatal_gate(s)?;
            }
            let runs = self
                .weighted_runs
                .as_ref()
                .ok_or("WEIGHTED_SCHEDULE_MISSING")?
                .len();
            for run in 0..runs {
                let weight = self
                    .weighted_runs
                    .as_ref()
                    .ok_or("WEIGHTED_SCHEDULE_MISSING")?[run]
                    .0;
                let target = self
                    .depth
                    .checked_add(weight)
                    .ok_or("WEIGHTED_TARGET_OVERFLOW")?;
                let (_, candidate_count) =
                    self.enqueue_weighted_route_packet(bank, sequence, batch.count, weight)?;
                if self.cfg.world == 1 {
                    let packet = &self.route_banks[bank];
                    let (states, hashes, rows, source_rows) = (
                        packet.packed_states.ptr.cast(),
                        packet.sorted_hashes.ptr.cast_const(),
                        packet.owner_counts.ptr.cast(),
                        packet.route_count.ptr.cast(),
                    );
                    let begin = self
                        .owner_window
                        .as_ref()
                        .ok_or("OWNER_WINDOW_MISSING")?
                        .ptr
                        .cast();
                    if self.native_rank.is_some() {
                        self.commit_rank_native_batch(
                            states,
                            hashes,
                            begin,
                            rows,
                            source_rows,
                            0,
                            Some(target),
                        )?;
                    } else {
                        #[cfg(feature = "library-owner")]
                        self.commit_rank_library_batch(
                            states,
                            hashes,
                            begin,
                            rows,
                            source_rows,
                            0,
                            Some(target),
                        )?;
                        #[cfg(not(feature = "library-owner"))]
                        return Err("WEIGHTED_LIBRARY_NOT_BUILT".into());
                    }
                    self.queue_owner_fatal_gate(s)?;
                } else {
                    if self.lsa_view.is_some() && lsa_owner_recorded {
                        check(observed_native!(unsafe {
                            cudaStreamWaitEvent(
                                self.exchange_stream.0,
                                self.owner_consumed
                                    .as_ref()
                                    .ok_or("LSA_OWNER_EVENT_MISSING")?
                                    .0,
                                0,
                            )
                        }))?;
                    }
                    lsa_owner_recorded = self.consume_owner_packet(
                        OwnerPacketContext {
                            generation: None,
                            candidate_count,
                            target_depth: Some(target),
                            packet_stride: self.stride,
                            batch_index: u64::from(scheduled_round),
                            scheduled_rounds,
                            trace_route: false,
                            parent: None,
                            extent_offset: 0,
                            parents: batch.count,
                            extent_index: batch.extent,
                            archive_live,
                            archive_dependency_imported: false,
                        },
                        &mut archive_released,
                        lsa_owner_recorded,
                    )?;
                }
                self.release_weighted_route_packet(bank, sequence)?;
            }
            // Generation/owner readers are joined on s. Archive has its own
            // final-reader event, recorded after every range for this extent.
            if batch.count != 0 {
                if archive_live {
                    check(observed_native!(unsafe {
                        cudaStreamWaitEvent(s, self.archive_done[batch.extent].0, 0)
                    }))?;
                }
                let mut live = self.front[batch.extent];
                live.sequence = batch.sequence;
                live.begin = batch.begin;
                live.count -= batch.offset;
                live.granted_rows = live.count as u32;
                self.enqueue_weighted_extent_retire_after_readers(live, u64::from(batch.count))?;
            }
            self.queue_owner_fatal_gate(s)?;
            let slot = epoch_serial % self.epoch_completed.len();
            check(observed_native!(unsafe {
                cudaEventRecord(self.epoch_completed[slot].0, s)
            }))?;
            self.epoch_outstanding.push_back(slot);
            epoch_serial = epoch_serial.checked_add(1).ok_or("EPOCH_SEQUENCE")?;
            self.enqueue_route_prefetch(&mut producer_cursor, &mut produced)?;
        }
        if let Some(a) = archive.as_deref_mut() {
            a.layer(u64::from(self.depth), u64::from(self.current_count))?;
            self.archived_depth = Some(self.depth);
        }
        let _finalize = TraceRange::new(trace_ranges, b"mgbfs.FinalizeDepth\0");
        while let Some(&slot) = self.epoch_outstanding.front() {
            self.wait_epoch_credit(slot)?;
            self.epoch_outstanding.pop_front();
        }
        // All original-depth producers have completed. Empty current layers
        // drain pending future targets before COMPLETE, not before each batch.
        loop {
            let target = self.depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;
            self.queue_weighted_promotion(target)?;
            let refs = self
                .weighted_owner_refs
                .as_ref()
                .ok_or("WEIGHTED_OWNER_REFS_MISSING")?;
            let history_slot = target as usize % (refs.targets.len() * 2);
            unsafe {
                check(observed_native!(cudaMemcpyAsync(
                    refs.history
                        .at(history_slot * self.cfg.layer_capacity as usize * 16),
                    refs.survivor_hashes.ptr,
                    self.cfg.layer_capacity as usize * 16,
                    3,
                    s,
                )))?;
                check(observed_native!(cudaMemcpyAsync(
                    refs.history_counts.at(history_slot * 4),
                    refs.survivor_count.ptr,
                    4,
                    3,
                    s,
                )))?;
                check(observed_native!(cudaMemcpyAsync(
                    self.prev.ptr,
                    refs.survivor_hashes.ptr,
                    self.cfg.layer_capacity as usize * 16,
                    3,
                    s,
                )))?;
                check(observed_native!(rank_directory(
                    self.cfg.world,
                    self.prev.ptr,
                    refs.survivor_count.ptr.cast(),
                    self.cfg.layer_capacity,
                    self.cfg.buckets,
                    self.cfg
                        .logical_owner_to_rank
                        .iter()
                        .position(|&rank| rank == self.cfg.rank)
                        .ok_or("OWNER_MAP")? as u32,
                    self.directory.ptr.cast(),
                    self.fatal.ptr.cast(),
                    s,
                )))?;
                check(observed_native!(traced_stream_synchronize(s)))?;
            }
            let settled = refs.settle_state.one::<MacroSettleState>()?;
            let control = self.control.one::<Control>()?;
            let ring = self.ring.one::<Ring>()?;
            let extent = self.extent.one::<Extent>()?;
            if settled.fatal != 0
                || control.error != 0
                || ring.fatal != 0
                || self.fatal.one::<u32>()? != 0
                || settled.last_epoch != u64::from(target) + 1
                || settled.count != control.survivors
                || extent.count != u64::from(settled.count)
                || extent.ready != 1
            {
                return Err(format!(
                    "WEIGHTED_FINALIZE_FATAL_{}_{}_{}",
                    settled.fatal, control.error, ring.fatal
                ));
            }
            let slot = target as usize % refs.targets.len();
            unsafe {
                if let Some(owner) = self.owner.as_ref() {
                    check(observed_native!(cudaMemsetAsync(
                        owner.lengths.at(slot * self.cfg.buckets as usize * 4),
                        0,
                        self.cfg.buckets as usize * 4,
                        s,
                    )))?;
                } else {
                    #[cfg(feature = "library-owner")]
                    {
                        let library = self.library_owner.as_ref().ok_or("LIBRARY_OWNER_MISSING")?;
                        let handle = *library
                            .weighted_ranks
                            .get(slot)
                            .ok_or("WEIGHTED_LIBRARY_TARGET_SLOT")?;
                        // Promotion and every provisional StateRef reader have
                        // completed at this semantic depth boundary.
                        check(observed_native!(mgbfs_library_rank_reset_weighted_v1(
                            handle
                        )))?;
                    }
                    #[cfg(not(feature = "library-owner"))]
                    return Err("WEIGHTED_LIBRARY_NOT_BUILT".into());
                }
                check(observed_native!(cudaMemsetAsync(
                    self.layer_count.at(slot * 4),
                    0,
                    4,
                    s
                )))?;
                check(observed_native!(traced_stream_synchronize(s)))?;
            }
            self.directory.read(&mut self.prev_dir)?;
            std::mem::swap(&mut self.prev_dir, &mut self.curr_dir);
            std::mem::swap(&mut self.prev, &mut self.curr);
            self.front.clear();
            if settled.count != 0 {
                self.front.push(extent);
            }
            self.prev_count = self.current_count;
            self.current_count = settled.count;
            self.depth = target;
            let refs = self
                .weighted_owner_refs
                .as_mut()
                .ok_or("WEIGHTED_OWNER_REFS_MISSING")?;
            refs.targets[slot] = None;
            self.layer_count.read(&mut refs.counts_host)?;
            let pending = refs.counts_host.iter().any(|&count| count != 0);
            if self.all_max(settled.count)? != 0 {
                return Ok(true);
            }
            if self.all_max(u32::from(pending))? == 0 {
                return Ok(false);
            }
        }
    }
    fn enqueue_compact_frontier_generation(&mut self, batch: ParentBatch) -> Result<u64> {
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
                    mgbfs_generate_hash_only_tc
                } else { mgbfs_generate_hash_only };
                check(generate(h.n, self.moves, h.modulus, self.stride as u32,
                    self.cfg.batch, self.candidates, self.cfg.rank, batch.sequence,
                    self.states.at(batch.begin as usize * self.stride).cast(),
                    h.generators.ptr.cast(), h.coefficients.ptr.cast(), h.offsets.ptr.cast(),
                    h.parent_count.ptr.cast(), self.route_banks[0].child_hashes.ptr.cast(), self.route_banks[0].children.ptr.cast(),
                    // Producer count must not overwrite the current owner's
                    // routed-count before its selected origins are committed.
                    self.route_banks[0].generation_control.at(4).cast(), self.route_banks[0].generation_control.ptr.cast(), s))?;
            } else if let Some(plan)=self.compact_direct_hash.as_ref(){
                check(mgbfs_compact_hash_run(plan.0,self.states.at(batch.begin as usize*self.stride).cast(),self.route_banks[0].child_hashes.ptr.cast(),batch.count,s))?;
            } else {
            check(mgbfs_generate_run(
                self.generate.as_ref().ok_or("DENSE_GENERATOR_MISSING")?.0,
                self.states.at(batch.begin as usize * self.stride).cast(),
                self.route_banks[0].children.ptr.cast(),
                batch.count,
                s,
            ))?;
            check(mgbfs_hash_run(
                self.hash.as_ref().ok_or("DENSE_HASH_MISSING")?.0,
                self.route_banks[0].children.ptr.cast(),
                self.route_banks[0].child_hashes.ptr.cast(),
                batch.count * self.moves,
                s,
            ))?;
            }
            self.route_banks[0].generation_done.record(sequence, s)?;
        }
        self.generation_sequence = sequence;
        Ok(sequence)
    }
    fn advance_shard_ab_inner(
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
        if self.shard_ab.is_some() {
            let largest = self.all_max(self.current_count)?;
            let generated = u64::from(largest).checked_mul(u64::from(self.cfg.world))
                .and_then(|n| n.checked_mul(u64::from(self.moves))).ok_or("SHARD_AB_INPUT_BOUND")?.max(1);
            let ab = self.shard_ab.as_mut().ok_or("SHARD_AB_MISSING")?;
            let shape = ab.shape;
            // Uniform owner hashing: largest rank frontier bounds average input;
            // retain 2x skew headroom without multiplying logical shards by
            // every GPU. Absolute input bound and explicit overflow stay intact.
            let expected = u64::from(largest).checked_mul(u64::from(self.moves))
                .and_then(|v|v.checked_mul(u64::from(self.cfg.world.min(2))))
                .ok_or("SHARD_AB_INPUT_BOUND")?.max(1);
            // A large producer portion must also fit the bounded A/B staging arena.
            // Choose this once per depth; no host counts or reallocations per portion.
            let stage = u64::from(if shape.capacity < 256 { 256 } else { shape.capacity.min(131072) });
            let portion = u64::from(largest.min(self.cfg.batch)).checked_mul(u64::from(self.moves))
                .and_then(|v|v.checked_mul(u64::from(self.cfg.world.min(2))))
                .ok_or("SHARD_AB_INPUT_BOUND")?.max(1);
            let estimate = ((expected+u64::from(shape.capacity)-1)/u64::from(shape.capacity))
                .max((portion+stage-1)/stage)
                .max(1).next_power_of_two().min(u64::from(shape.maximum)) as u32;
            let active = shape.forced.unwrap_or(estimate);
            let bound = generated.min(u64::from(shape.capacity)).next_power_of_two() as u32;
            ab.input_bound = self.candidates.min(u32::try_from(generated.min(u64::from(u32::MAX)))
                .map_err(|_| "SHARD_AB_INPUT_BOUND")?).max(1);
            let owner = self.cfg.logical_owner_to_rank.iter().position(|&rank|rank==self.cfg.rank).ok_or("OWNER_MAP")? as u32;
            check(unsafe { mgbfs_shard_ab_indexed_bind(ab.handle,self.ring.ptr,self.control.ptr,self.extent.ptr,
                self.states.ptr.cast(),self.layer_count.ptr.cast(),self.cfg.layer_capacity,
                self.next_extent_count.as_ref().ok_or("NEXT_EXTENT_COUNT_MISSING")?.ptr.cast(),
                self.next_extents.as_ref().ok_or("NEXT_EXTENTS_MISSING")?.ptr) })?;
            check(unsafe { mgbfs_shard_ab_pipeline_begin(ab.handle,active,owner,self.cfg.world,
                self.prev.ptr,self.prev_count,self.curr.ptr,self.current_count,bound,
                self.ring.at(48).cast()) })?;
            eprintln!("MGBFS_SHARD_AB_DEPTH rank={} depth={} shards={} slots={} job_bound={} capacity={}",
                self.cfg.rank,self.depth,active,shape.slots,bound,shape.capacity);
        }
        let mut cursor = ParentCursor::default();
        let mut prefetched: Option<(ParentBatch, u64)> = None;
        let mut archive_released = [false; 2];
        let mut lsa_owner_recorded = false;
        let device_epoch = self.lsa_view.is_some() && self.rank_owner_mode();
        let graph_requested = self.batch_graph.is_some();
        if self.shard_ab.is_some() && graph_requested
            && std::env::var_os("MGBFS_TEST_SHARD_AB_GRAPH").is_none() {
            return Err("SHARD_AB_GRAPH_NOT_YET_VALIDATED".into());
        }
        if graph_requested && (!device_epoch || self.hash_first.is_some() || trace_sync
            || std::env::var_os("MGBFS_TEST_OWNER_DAG_CAPTURE").is_some()) {
            return Err("BATCH_GRAPH_REQUIRES_DENSE_DEVICE_COUNT_RANK_OWNER".into());
        }
        // Small frontiers cannot fill one window. Keep their existing direct
        // launch path instead of capturing/reinstantiating a short graph at
        // every early/late depth. This decision is shared once per layer.
        let graph_mode = graph_requested && scheduled_rounds >= 32;
        if graph_mode {
            if let Some(a) = archive.as_deref() {
                let credits = crate::reference_launch::graph_archive_credits(
                    scheduled_rounds, self.cfg.batch, a.rows, self.epoch_completed.len())?;
                if a.credit_capacity() < credits {
                    return Err(format!("GRAPH_ARCHIVE_CREDIT_BUDGET required={credits} available={}",
                        a.credit_capacity()));
                }
            }
        }
        // The host-sized depth schedule above reuses collective_recv for a
        // nonzero round count. LSA reads this word as its device group-fatal
        // predicate before the first owner vote, so start each depth clean.
        if device_epoch {
            check(unsafe { cudaMemsetAsync(self.collective_recv.ptr, 0, 4, s) })?;
        }
        let mut epoch_serial = 0usize;
        for scheduled_round in 0..scheduled_rounds {
            let graph_mode = graph_mode && scheduled_round < (scheduled_rounds / 32) * 32;
            let _batch_range = TraceRange::new(trace_ranges, b"mgbfs.batch\0");
            self.ensure_not_cancelled()?;
            if device_epoch && self.epoch_outstanding.len() == self.epoch_completed.len() {
                let slot = *self.epoch_outstanding.front().ok_or("EPOCH_CREDIT_EMPTY")?;
                self.wait_epoch_credit(slot)?;
                self.epoch_outstanding.pop_front();
            }
            if graph_mode && scheduled_round % 32 == 0 {
                // The archive worker observes real D2H events, never captured
                // events. Stage only this bounded window while parents remain
                // live; owner retirement waits on those copies inside the graph.
                if let Some(a) = archive.as_deref_mut() {
                    let mut staged = cursor;
                    for _ in 0..(scheduled_rounds-scheduled_round).min(32) {
                        if let Some(batch) = staged.take(&self.front, self.cfg.batch)? {
                            self.archive_range(a, batch.begin, u64::from(batch.count), batch.extent)?;
                        }
                    }
                }
                unsafe {
                    self.batch_graph.as_mut().ok_or("BATCH_GRAPH_MISSING")?.begin(
                        s, self.generation_stream.0, self.exchange_stream.0)?;
                    // Rebind the prior-consumer dependency into this capture.
                    // Earlier launches are already ordered on the owner stream.
                    check(cudaEventRecord(self.owner_consumed.as_ref()
                        .ok_or("LSA_OWNER_EVENT_MISSING")?.0, s))?;
                }
                lsa_owner_recorded = true;
            }
            let work = cursor.take(&self.front, self.cfg.batch)?;
            let extent_index = work.map(|b| b.extent).unwrap_or(0);
            let extent_offset = work.map(|b| b.offset).unwrap_or(0);
            let parent = work.map(|b| self.front[b.extent]);
            let parents = work.map(|b| b.count).unwrap_or(0);
            let next_work = if graph_mode && (scheduled_round+1)%32 == 0 {
                None // Never export a captured generation into the next window.
            } else { cursor.peek(&self.front, self.cfg.batch)? };
            let mut generation = None;
            let candidate_count = parents * self.moves;
            if trace_route {
                eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=batch_begin parents={parents} candidates={candidate_count}", self.cfg.rank, self.depth);
            }
            if let Some(a) = archive.as_deref_mut() {
                if trace_route {
                    eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=archive_begin", self.cfg.rank, self.depth);
                }
                let error = if graph_mode { None } else if let Some(extent) = parent {
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
                    None => self.enqueue_compact_frontier_generation(batch)?,
                };
                unsafe { self.route_banks[0].generation_done.wait(sequence, s)?; }
                generation = Some(sequence);
                let fatal = self.route_banks[0].generation_control.ptr.cast();
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
                        mgbfs_generate_hash_only_tc
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
                        self.route_banks[0].child_hashes.ptr.cast(),
                        self.route_banks[0].children.ptr.cast(),
                        self.route_banks[0].route_count.ptr.cast(),
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
                    None => self.enqueue_compact_frontier_generation(batch)?,
                };
                unsafe {
                    self.route_banks[0].generation_done.wait(sequence, s)?;
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
                if let Some(ab) = self.shard_ab.as_ref() {
                    if ab.sorted_merge {
                        // Full key order also preserves owner/shard prefixes.
                        // No pre-dedup: states and reference indices remain paired.
                        check(mgbfs_route_run(self.route.0,self.route_banks[0].child_hashes.ptr,
                            self.identity_refs.ptr.cast(),self.route_banks[0].sorted_hashes.ptr,
                            self.route_banks[0].sorted_refs.ptr.cast(),self.route_banks[0].route_count.ptr.cast(),candidate_count,0,s))?;
                    } else {
                    check(mgbfs_route_run_sharded(self.route.0,self.route_banks[0].child_hashes.ptr,
                        self.identity_refs.ptr.cast(),self.route_banks[0].sorted_hashes.ptr,
                        self.route_banks[0].sorted_refs.ptr.cast(),self.route_banks[0].route_count.ptr.cast(),
                        candidate_count,self.cfg.world*ab.shape.maximum,s))?;
                    }
                } else {
                check(mgbfs_route_run(
                    self.route.0,
                    self.route_banks[0].child_hashes.ptr,
                    self.identity_refs.ptr.cast(),
                    self.route_banks[0].sorted_hashes.ptr,
                    self.route_banks[0].sorted_refs.ptr.cast(),
                    self.route_banks[0].route_count.ptr.cast(),
                    candidate_count,
                    self.cfg.prededup as i32,
                    s,
                ))?;
                }
            }
            let packet_stride = if self.hash_first.is_some() {
                16
            } else {
                self.stride
            };
            if self.shard_key_first.is_some() {
                check(unsafe {mgbfs_exchange_count_device_n(self.cfg.world,self.candidates,self.route_banks[0].sorted_hashes.ptr,self.route_banks[0].route_count.ptr.cast(),self.route_banks[0].owner_counts.ptr.cast(),s)})?;
            } else {
            check(unsafe {
                mgbfs_exchange_pack_device_n(
                    self.cfg.world,
                    packet_stride as u32,
                    self.candidates,
                    self.route_banks[0].children.ptr.cast(),
                    candidate_count,
                    self.route_banks[0].sorted_hashes.ptr,
                    self.route_banks[0].sorted_refs.ptr.cast(),
                    self.route_banks[0].route_count.ptr.cast(),
                    self.route_banks[0].packed_states.ptr.cast(),
                    self.route_banks[0].owner_counts.ptr.cast(),
                    s,
                )
            })?;
            }
            if self.rank_owner_mode() {
                let window = self.owner_window.as_ref().ok_or("OWNER_WINDOW_MISSING")?;
                let logical_owner = self.cfg.logical_owner_to_rank.iter()
                    .position(|&rank| rank == self.cfg.rank).ok_or("OWNER_MAP")? as u32;
                check(unsafe {
                    mgbfs_owner_window_from_counts(
                        self.cfg.world, self.candidates, logical_owner,
                        self.route_banks[0].owner_counts.ptr.cast(), self.route_banks[0].route_count.ptr.cast(),
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
                let routed = self.route_banks[0].route_count.one::<u32>()?;
                let mut owner_counts = [0u32; 8];
                self.route_banks[0].owner_counts.read(&mut owner_counts[..self.cfg.world as usize])?;
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
                        self.route_banks[0].generation_done.retire_after_device_barrier(
                            sequence, self.generation_stream.0, self.pack_done.0,
                        )?;
                    }
                } else {
                    if !self.route_banks[0].generation_done.poll(sequence)? {
                        return Err("GENERATION_PACK_ORDER".into());
                    }
                    self.route_banks[0].generation_done.retire(sequence)?;
                }
                if let Some(next) = next_work {
                    let sequence = self.enqueue_compact_frontier_generation(next)?;
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
                            self.comm.0, self.route_banks[0].sorted_hashes.ptr,
                            if self.shard_key_first.is_some(){std::ptr::null()}else{self.route_banks[0].packed_states.ptr}, self.route_banks[0].owner_counts.ptr.cast(),
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
                    let received = if let Some(h)=self.shard_key_first.as_ref().filter(|h|self.cfg.world==2 && h.peer_metadata_enabled) {
                        // Both peers inspect the same two counts and sticky status
                        // words before posting any variable-sized payload. The
                        // fixed metadata lease drains before its regeneration reuse.
                        check(unsafe{mgbfs_device_store_u32(h.peer_metadata.ptr.cast(),remote_rows,communication)})?;
                        check(unsafe{cudaMemcpyAsync(h.peer_metadata.at(4),self.ring.at(48),4,3,communication)})?;
                        check(unsafe{mgbfs_nccl_send_recv(self.comm.0,h.peer_metadata.ptr,8,exchange_peer,h.peer_metadata.at(8),8,communication)})?;
                        self.wait_comm_stream(communication)?;
                        let m=h.peer_metadata.one::<[u32;4]>()?;
                        let fatal=m[1].max(m[3]);
                        if fatal!=0{return Err(format!("GROUP_INITIAL_EXCHANGE_FATAL_{fatal}"));}
                        if m[0]>self.candidates || m[2]>self.candidates{return Err("GROUP_INITIAL_EXCHANGE_COUNT_CAPACITY".into());}
                        let incoming=m[2];
                        check(unsafe{mgbfs_device_store_u32(self.recv_count.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr.cast(),incoming,communication)})?;
                        incoming
                    } else {
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
                    received
                    };
                    check(unsafe {
                        mgbfs_nccl_send_recv(
                            self.comm.0,
                            self.route_banks[0].sorted_hashes.at(remote_offset as usize * 16),
                            u64::from(remote_rows) * 16,
                            exchange_peer,
                            self.recv_hashes.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr,
                            u64::from(received) * 16,
                            communication,
                        )
                    })?;
                    if self.shard_key_first.is_none() {
                    check(unsafe {
                        mgbfs_nccl_send_recv(
                            self.comm.0,
                            self.route_banks[0].packed_states
                                .at(remote_offset as usize * packet_stride),
                            u64::from(remote_rows) * packet_stride as u64,
                            exchange_peer,
                            self.recv_states.as_ref().ok_or("LEGACY_RECEIVE_BUFFER_MISSING")?.ptr,
                            u64::from(received) * packet_stride as u64,
                            communication,
                        )
                    })?;
                    }
                    check(unsafe { cudaEventRecord(self.exchange_done.0, communication) })?;
                    received
                };
                if let Some(parent_extent) =
                    parent.filter(|_| round == 1 && self.hash_first.is_none() && self.shard_key_first.is_none())
                {
                    if archive.is_some()
                        || (self.archived_depth == Some(self.depth)
                            && !archive_released[extent_index])
                    {
                        check(unsafe {
                            cudaStreamWaitEvent(s, self.archive_done[extent_index].0,
                                u32::from(graph_mode)) // cudaEventWaitExternal during capture
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
                // Key-first keeps parents live through materialization. The fixed
                // initial count/fatal packet has already gated hash exchange;
                // selection metadata and the post-owner vote gate regeneration
                // and retirement. No retirement occurs in this pre-owner block.
                // Keep the legacy vote for all other transports/configurations.
                if round == 1 && self.hash_first.is_none()
                    && !(self.lsa_view.is_none() && self.shard_key_first.as_ref()
                        .is_some_and(|h|h.reuse_preowner_status)) {
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
                    } else {
                        let code = self.all_max_ring_fatal()?;
                        if code != 0 {
                            return Err(format!("GROUP_STATE_RING_RETIRE_FATAL_{code}"));
                        }
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
                    self.route_banks[0].packed_states
                        .at(local_offset as usize * packet_stride)
                        .cast()
                };
                let local_hashes: *const c_void = unsafe {
                    self.route_banks[0].sorted_hashes.at(local_offset as usize * 16)
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
                                (self.route_banks[0].packed_states.ptr as *const u8,
                                 self.route_banks[0].sorted_hashes.ptr as *const c_void,
                                 window.ptr as *const u32,
                                 unsafe { window.at(4) } as *const u32,
                                 self.route_banks[0].route_count.ptr as *const u32)
                            } else {
                                (remote_states, remote_hashes,
                                 unsafe { window.at(8) } as *const u32,
                                 remote_count, remote_count)
                            };
                            if let Some(ab) = self.shard_ab.as_ref() {
                                if self.shard_key_first.is_some() {
                                    let handle=ab.handle;let bound=ab.input_bound;
                                    check(unsafe {mgbfs_shard_ab_pipeline_select(handle,hashes,begin,rows,source_rows,bound)})?;
                                    return self.materialize_shard_selected(parent,extent_offset,parents,round,group,remote_offset);
                                }
                                return check(unsafe { mgbfs_shard_ab_pipeline_push(
                                    ab.handle, hashes, states, begin, rows, source_rows,
                                    ab.input_bound) });
                            }
                            if self.native_rank.is_some() {
                                return self.commit_rank_native_batch(states, hashes, begin, rows, source_rows, group, None);
                            }
                            #[cfg(feature = "library-owner")]
                            return self.commit_rank_library_batch(states, hashes, begin, rows, source_rows, group, None);
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
                    if self.lsa_view.is_none() && self.shard_key_first.as_ref().is_some_and(|h|h.combined_status) {
                        // Fuse host enqueue status with sticky GPU regeneration failure.
                        // This all-rank gate is before current-parent retirement and
                        // before any publication. Failed peer responses can affect only
                        // the unfinished future; parents remain available for INCOMPLETE.
                        self.collective_send.put_u32(u32::from(batch_error.is_some()))?;
                        check(unsafe {mgbfs_owner_import_transport_fatal(self.collective_send.ptr.cast(),self.ring.ptr.cast(),self.control.ptr.cast(),s)})?;
                        let code=self.all_max_ring_fatal()?;
                        if code!=0 {return Err(batch_error.unwrap_or_else(||format!("GROUP_STATE_RING_RETIRE_FATAL_{code}")));}
                    } else {
                    vote_group_error(
                        batch_error.map_or(Ok(()), Err),
                        |failed| Ok(self.all_max(u32::from(failed))? != 0),
                        "REMOTE_OWNER_BATCH_FATAL".into(),
                    )?;
                    }
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
            if self.hash_first.is_some() || self.shard_key_first.is_some() {
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
                } else {
                    let retirement_error=self.all_max_ring_fatal()?;
                    if retirement_error!=0{return Err(format!("GROUP_STATE_RING_RETIRE_FATAL_{retirement_error}"));}
                }
            }
            if graph_mode && ((scheduled_round+1)%32 == 0 || scheduled_round+1 == scheduled_rounds) {
                self.batch_graph.as_mut().ok_or("BATCH_GRAPH_MISSING")?
                    .submit(scheduled_round%32+1)?;
                // Captured events encode graph edges; direct tail batches need
                // an ordinary completion record after the launched window.
                check(unsafe { cudaEventRecord(self.owner_consumed.as_ref()
                    .ok_or("LSA_OWNER_EVENT_MISSING")?.0, s) })?;
                if scheduled_round + 1 == (scheduled_rounds / 32) * 32
                    && scheduled_round + 1 < scheduled_rounds {
                    // A graph launch does not enqueue its captured kernels on
                    // the original generation stream. Its first direct tail
                    // producer must not overwrite shared children/hash buffers
                    // while the launched window still reads them.
                    check(unsafe { cudaStreamWaitEvent(self.generation_stream.0,
                        self.owner_consumed.as_ref()
                            .ok_or("LSA_OWNER_EVENT_MISSING")?.0, 0) })?;
                }
            }
            if device_epoch && (!graph_mode || (scheduled_round+1)%32 == 0
                || scheduled_round+1 == scheduled_rounds) {
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
                    self.route_banks[0].route_count.ptr.cast(),
                    self.fatal.ptr.cast(),
                    s,
                ))?;
                check(cudaStreamSynchronize(s))?;
            }
        }
        if let Some(ab) = self.shard_ab.as_ref() {
            let mut keys = std::ptr::null_mut();let mut states = std::ptr::null_mut();let mut count = std::ptr::null();
            check(unsafe { mgbfs_shard_ab_pipeline_finalize(ab.handle,&mut keys,&mut states,&mut count) })?;
            check(unsafe {mgbfs_shard_ab_pipeline_prepare_publish(ab.handle,self.ring.ptr,
                self.control.ptr,self.extent.ptr,self.layer_count.ptr.cast(),self.cfg.layer_capacity)})?;
            check(unsafe {cudaStreamSynchronize(s)})?;
            let prepared_control=self.control.one::<Control>()?;
            let prepared_ring=self.ring.one::<Ring>()?;
            let prepared_error=self.all_max(prepared_control.error.max(prepared_ring.fatal))?;
            if prepared_error!=0{return Err(format!("SHARD_AB_PREPARE_FATAL_{prepared_error}"));}
            check(unsafe { mgbfs_shard_ab_pipeline_publish(ab.handle,self.ring.ptr,self.control.ptr,self.extent.ptr,
                self.states.ptr.cast(),self.prev.ptr,self.layer_count.ptr.cast(),self.cfg.layer_capacity,
                self.next_extent_count.as_ref().ok_or("NEXT_EXTENT_COUNT_MISSING")?.ptr.cast(),
                self.next_extents.as_ref().ok_or("NEXT_EXTENTS_MISSING")?.ptr,self.route_banks[0].route_count.ptr.cast()) })?;
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
            if self.shard_ab.is_some()&&std::env::var_os("MGBFS_TEST_VERIFY_STATE_HASH_ALIGNMENT").is_some(){
                if std::env::var("MGBFS_SHARD_AB_DEDUP").ok().as_deref()==Some("SORT_MERGE"){return Err("HASH_ALIGNMENT_REQUIRES_HASH_ORDER".into());}
                let mut ordinal=0usize;
                for extent in &self.next {
                    let mut at=0u64;
                    while at<extent.count {
                        let n=(extent.count-at).min(u64::from(self.cfg.batch)) as u32;
                        check(unsafe{mgbfs_hash_run(self.archive_hash.0,self.states.at((extent.begin+at) as usize*self.stride).cast(),self.archive_hashes.ptr.cast(),n,s)})?;
                        check(unsafe{mgbfs_debug_hashes_equal(self.prev.at(ordinal*16),self.archive_hashes.ptr,n,self.ring.at(48).cast(),s)})?;
                        at+=u64::from(n);ordinal+=n as usize;
                    }
                }
                check(unsafe{cudaStreamSynchronize(s)})?;let error=self.all_max(self.ring.one::<Ring>()?.fatal)?;
                if error!=0{return Err(format!("STATE_HASH_ALIGNMENT_FATAL_{error}"));}
            }



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
            self.route_banks[0].route_count.put(&[count])?;
        }
        if self.fatal.one::<u32>()? != 0 {
            return Err("FINALIZE_FATAL".into());
        }
        let count = self.route_banks[0].route_count.one::<u32>()?;
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
    fn advance_inner(
        &mut self,
        mut archive: Option<&mut crate::pinned_archive::PinnedArchive>,
    ) -> Result<bool> {
        if self.shard_ab.is_some() { return self.advance_shard_ab_inner(archive); }

        let _native_scope = NativeCallMarker::enter(line!());
        self.ensure_not_cancelled()?;
        if self.weighted_runs.is_some() {
            return self.advance_weighted_inner(archive);
        }
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
                check(observed_native!(cudaMemsetAsync(
                    owner.lengths.ptr,
                    0,
                    self.cfg.buckets as usize * 4,
                    s,
                )))?;
            }
            check(observed_native!(cudaMemsetAsync(
                self.layer_count.ptr,
                0,
                4,
                s
            )))?;
            if self.rank_owner_mode() {
                check(observed_native!(cudaMemsetAsync(
                    self.next_extent_count
                        .as_ref()
                        .ok_or("NEXT_EXTENT_COUNT_MISSING")?
                        .ptr,
                    0,
                    4,
                    s,
                )))?;
            }
        }
        #[cfg(feature = "library-owner")]
        if let Some(library) = self.library_owner.as_mut() {
            if library.closed {
                if library.rank_mode {
                    library.rank = unsafe {
                        create_rank_owner(
                            library,
                            &self.prev,
                            &self.curr,
                            self.cfg.shards,
                            self.candidates,
                            self.cfg.world,
                            s,
                        )?
                    };
                } else {
                    for shard in 0..library.shards.len() {
                        unsafe {
                            library.shards[shard].reopen_window(
                                history_view(
                                    &self.prev,
                                    library.plane_words,
                                    &library.previous[shard],
                                ),
                                history_view(
                                    &self.curr,
                                    library.plane_words,
                                    &library.current[shard],
                                ),
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
        #[cfg(debug_assertions)]
        let batch_capture_requested = std::env::var_os("MGBFS_TEST_BATCH_DAG_CAPTURE").is_some();
        #[cfg(debug_assertions)]
        if batch_capture_requested {
            // Admit only the shared device-resident owner/transport path.
            // HASH_FIRST also requires device materialization descriptors; the
            // legacy host-sized path and nested capture remain inadmissible.
            let host_materialization = self
                .hash_first
                .as_ref()
                .is_some_and(|state| state.device.is_none());
            if self.cfg.world != 2
                || self.lsa_view.is_none()
                || host_materialization
                || !self.rank_owner_mode()
                || trace_route
                || std::env::var_os("MGBFS_TEST_OWNER_DAG_CAPTURE").is_some()
            {
                return Err("BATCH_DAG_CAPTURE_REQUIRES_LSA_DEVICE_OWNER".into());
            }
            if self.depth == 3 && scheduled_rounds < 2 {
                return Err("BATCH_DAG_CAPTURE_REQUIRES_TWO_ROUNDS".into());
            }
        }
        let mut cursor = ParentCursor::default();
        if !self.prefetched.is_empty() {
            return Err("ROUTE_PREFETCH_DEPTH_LEAK".into());
        }
        let mut producer_cursor = ParentCursor::default();
        let mut produced = 0usize;
        // Generation/route packets do not require LSA materialization storage.
        // Both transports share the same bounded producer banks and reader leases.
        for _ in 0..self.route_banks.len() {
            if !self.enqueue_route_prefetch(&mut producer_cursor, &mut produced)? {
                break;
            }
        }
        let mut archive_released = [false; 2];
        let mut lsa_owner_recorded = false;
        let device_epoch = self.rank_owner_mode();
        #[cfg(debug_assertions)]
        if std::env::var_os("MGBFS_TEST_REQUIRE_HOST_DEVICE_EPOCH").is_some()
            && self.cfg.transport == mgbfs_core::config::ReferenceTransport::HostSizedNccl
            && self.hash_first.is_none()
            && self.rank_owner_mode()
        {
            if !device_epoch {
                return Err("HOST_DEVICE_EPOCH_UNAVAILABLE".into());
            }
            eprintln!(
                "MGBFS_TEST_HOST_DEVICE_EPOCH_ENABLED rank={} depth={}",
                self.cfg.rank, self.depth
            );
        }

        // The host-sized depth schedule above reuses collective_recv for a
        // nonzero round count. LSA reads this word as its device group-fatal
        // predicate before the first owner vote, so start each depth clean.
        if device_epoch {
            check(observed_native!(unsafe {
                cudaMemsetAsync(self.collective_recv.ptr, 0, 4, s)
            }))?;
        }
        #[cfg(debug_assertions)]
        let require_dense_prefetch =
            std::env::var_os("MGBFS_TEST_REQUIRE_DENSE_PREFETCH_BEFORE_SIZING").is_some();
        let mut epoch_serial = 0usize;
        for scheduled_round in 0..scheduled_rounds {
            let _batch_range = TraceRange::new(trace_ranges, b"mgbfs.batch\0");
            self.ensure_not_cancelled()?;
            if device_epoch && self.epoch_outstanding.len() == self.epoch_completed.len() {
                let slot = *self.epoch_outstanding.front().ok_or("EPOCH_CREDIT_EMPTY")?;
                self.wait_epoch_credit(slot)?;
                self.epoch_outstanding.pop_front();
            }
            self.active_route_bank = scheduled_round as usize % self.route_banks.len();
            let work = cursor.take(&self.front, self.cfg.batch)?;
            let extent_index = work.map(|b| b.extent).unwrap_or(0);
            let extent_offset = work.map(|b| b.offset).unwrap_or(0);
            let parent = work.map(|b| self.front[b.extent]);
            let parents = work.map(|b| b.count).unwrap_or(0);
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
                    if let Some(error) = error {
                        return Err(error);
                    }
                    self.queue_owner_fatal_gate(s)?;
                    if trace_route {
                        eprintln!("MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=archive_vote_queued", self.cfg.rank, self.depth);
                    }
                } else if self.all_max(u32::from(error.is_some()))? != 0 {
                    return Err(error.unwrap_or_else(|| "REMOTE_ARCHIVE_FATAL".into()));
                }
            }
            if trace_route {
                eprintln!(
                    "MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=generate_begin",
                    self.cfg.rank, self.depth
                );
            }
            if self.hash_first.is_some() && work.is_some() {
                let batch = work.ok_or("GENERATION_BATCH_MISSING")?;
                let sequence = match self.prefetched.pop_front() {
                    Some((expected, sequence, bank))
                        if expected == batch && bank == self.active_route_bank =>
                    {
                        sequence
                    }
                    Some(_) => return Err("GENERATION_BATCH_IDENTITY".into()),
                    None => return Err("GENERATION_BATCH_MISSING".into()),
                };
                unsafe {
                    self.route_banks[self.active_route_bank]
                        .generation_done
                        .wait(sequence, s)?;
                }
                generation = Some(sequence);
                let fatal = self.route_banks[self.active_route_bank]
                    .generation_control
                    .ptr
                    .cast();
                check(observed_native!(unsafe {
                    mgbfs_owner_import_transport_fatal(
                        fatal,
                        self.ring.ptr.cast(),
                        self.control.ptr.cast(),
                        s,
                    )
                }))?;
                if device_epoch {
                    self.queue_owner_fatal_gate(s)?;
                } else if self.all_max_ring_fatal()? != 0 {
                    return Err("HASH_FIRST_GENERATION_FATAL".into());
                }
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
                    check(observed_native!(generate(
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
                        self.route_banks[self.active_route_bank]
                            .child_hashes
                            .ptr
                            .cast(),
                        self.route_banks[self.active_route_bank].children.ptr.cast(),
                        self.route_banks[self.active_route_bank]
                            .route_count
                            .ptr
                            .cast(),
                        h.local_fatal.ptr.cast(),
                        s,
                    )))?;
                    if !device_epoch {
                        check(observed_native!(traced_stream_synchronize(s)))?;
                    }
                }
                if device_epoch {
                    check(observed_native!(unsafe {
                        mgbfs_owner_import_transport_fatal(
                            h.local_fatal.ptr.cast(),
                            self.ring.ptr.cast(),
                            self.control.ptr.cast(),
                            s,
                        )
                    }))?;
                    self.queue_owner_fatal_gate(s)?;
                } else if self.all_max(h.local_fatal.one::<u32>()?)? != 0 {
                    return Err("HASH_FIRST_GENERATION_FATAL".into());
                }
            } else if let Some(batch) = work {
                let sequence = match self.prefetched.pop_front() {
                    Some((expected, sequence, bank))
                        if expected == batch && bank == self.active_route_bank =>
                    {
                        sequence
                    }
                    Some(_) => return Err("GENERATION_BATCH_IDENTITY".into()),
                    None => return Err("GENERATION_BATCH_MISSING".into()),
                };
                unsafe {
                    self.route_banks[self.active_route_bank]
                        .generation_done
                        .wait(sequence, s)?;
                }
                generation = Some(sequence);
            }
            // Receive-slot reuse depends on the previous epoch's owner, not
            // on this packet. Import that external event before starting the
            // prepared-packet capture. The current pack_done fork below still
            // joins exchange to this owner DAG; no host wait/readback is added.
            if self.lsa_view.is_some() && lsa_owner_recorded {
                check(observed_native!(unsafe {
                    cudaStreamWaitEvent(
                        self.exchange_stream.0,
                        self.owner_consumed
                            .as_ref()
                            .ok_or("LSA_OWNER_EVENT_MISSING")?
                            .0,
                        0,
                    )
                }))?;
            }
            // Archive D2H was submitted outside this prepared-packet DAG.
            // Import its retirement dependency before capture, not as an
            // illegal external-event wait in the captured retirement node.
            // Normal execution keeps the later wait to preserve overlap.
            #[allow(unused_mut)]
            let mut archive_dependency_imported = false;
            #[cfg(debug_assertions)]
            if batch_capture_requested
                && self.depth == 3
                && scheduled_round < 2
                && parent.is_some()
                && (archive.is_some()
                    || (self.archived_depth == Some(self.depth) && !archive_released[extent_index]))
            {
                check(observed_native!(unsafe {
                    cudaStreamWaitEvent(s, self.archive_done[extent_index].0, 0)
                }))?;
                archive_released[extent_index] = true;
                archive_dependency_imported = true;
            }
            // The probe admits prepared packets, not the producer DAG. Import
            // its existing ready-event dependency before entering capture:
            // CUDA forbids a captured stream waiting on an event recorded by
            // a noncaptured producer. This remains a GPU wait, never a drain.
            #[cfg(debug_assertions)]
            let batch_capture = OwnerCaptureProbe::begin_enabled(
                s,
                batch_capture_requested && self.depth == 3 && scheduled_round < 2,
            )?;
            if trace_sync {
                check(observed_native!(unsafe { traced_stream_synchronize(s) }))?;
            }
            if trace_route {
                eprintln!(
                    "MGBFS_ROUTE_TRACE rank={} depth={} batch={batch_index} stage=generation_end",
                    self.cfg.rank, self.depth
                );
            }
            // Both profiles prepare nonempty immutable packets on the sole
            // producer. HASH_FIRST retains parents until materialization ends.
            // Empty HASH_FIRST generation remains on s; order its packet work
            // on the producer too, so radix scratch never has concurrent users.
            let packet_stride = if self.hash_first.is_some() {
                16
            } else {
                self.stride
            };
            if generation.is_none() {
                let route_stream = self.generation_stream.0;
                // Empty epochs still touch the reusable route bank. Fork the
                // producer from the owner before packet construction, then join
                // below. Both profiles need this ordering; DENSE previously
                // imported a noncaptured producer event into a captured owner
                // stream (CUDA_STATUS_905 on an uneven rank frontier).
                check(observed_native!(unsafe {
                    cudaEventRecord(self.pack_done.0, s)
                }))?;
                check(observed_native!(unsafe {
                    cudaStreamWaitEvent(route_stream, self.pack_done.0, 0)
                }))?;
                self.enqueue_route_packet(self.active_route_bank, candidate_count, route_stream)?;
                check(observed_native!(unsafe {
                    cudaEventRecord(self.pack_done.0, route_stream)
                }))?;
                check(observed_native!(unsafe {
                    cudaStreamWaitEvent(s, self.pack_done.0, 0)
                }))?;
            }
            // A prepared packet owns sorted hashes and packed states until
            // transport AND owner consume it. Retire/re-prefetch at batch end,
            // not after pack; the other prefilled banks provide lookahead.
            #[cfg(debug_assertions)]
            if require_dense_prefetch
                && self.hash_first.is_none()
                && generation.is_some()
                && produced < ParentCursor::round_count(&self.front, self.cfg.batch)? as usize
            {
                // This assertion checks actual production admission, before
                // the first host count wait. No GPU readback in the test hook.
                if !self
                    .prefetched
                    .iter()
                    .any(|(_, sequence, _bank)| Some(*sequence) > generation)
                {
                    return Err("DENSE_PREFETCH_BEFORE_SIZING_MISSING".into());
                }
                eprintln!(
                    "MGBFS_TEST_DENSE_PREFETCH_BEFORE_SIZING rank={} depth={} bank={}",
                    self.cfg.rank, self.depth, self.active_route_bank
                );
            }
            lsa_owner_recorded = self.consume_owner_packet(
                OwnerPacketContext {
                    generation,
                    candidate_count,
                    target_depth: None,
                    packet_stride,
                    batch_index,
                    scheduled_rounds,
                    trace_route,
                    parent,
                    extent_offset,
                    parents,
                    extent_index,
                    archive_live: archive.is_some(),
                    archive_dependency_imported,
                },
                &mut archive_released,
                lsa_owner_recorded,
            )?;
            if self.hash_first.is_some() {
                if let Some(parent_extent) = parent {
                    if !archive_dependency_imported
                        && (archive.is_some()
                            || (self.archived_depth == Some(self.depth)
                                && !archive_released[extent_index]))
                    {
                        check(observed_native!(unsafe {
                            cudaStreamWaitEvent(s, self.archive_done[extent_index].0, 0)
                        }))?;
                        archive_released[extent_index] = true;
                    }
                    let mut live = parent_extent;
                    live.sequence += extent_offset;
                    live.begin = live.sequence % u64::from(self.cfg.state_ring_capacity);
                    live.count -= extent_offset;
                    live.granted_rows = live.count as u32;
                    check(observed_native!(unsafe {
                        mgbfs_state_retire_dense_prefix_value(
                            self.ring.ptr.cast(),
                            live,
                            u64::from(parents),
                            s,
                        )
                    }))?;
                }
                if device_epoch {
                    self.queue_owner_fatal_gate(s)?;
                } else if self.all_max_ring_fatal()? != 0 {
                    return Err("HASH_FIRST_RETIRE_FATAL".into());
                }
            }
            #[cfg(debug_assertions)]
            if let Some(probe) = batch_capture {
                // Transport streams have joined the owner stream and parent
                // retirement is queued. Launch before producer re-prefetch:
                // joining a producer here would capture an unconsumed future
                // packet fork. No host count read or stream drain is inserted.
                probe.launch_named("MGBFS_BATCH_DAG_CAPTURE")?;
                // The captured owner event encodes an internal graph edge.
                // Publish an ordinary completion record after the launched DAG
                // before the next epoch waits on this receive-slot lease. The
                // owner stream orders it after all graph readers/retirement;
                // no host completion query or synchronization is needed.
                if lsa_owner_recorded {
                    check(observed_native!(unsafe {
                        cudaEventRecord(
                            self.owner_consumed
                                .as_ref()
                                .ok_or("LSA_OWNER_EVENT_MISSING")?
                                .0,
                            s,
                        )
                    }))?;
                }
                eprintln!("MGBFS_BATCH_DAG_CAPTURE scope=prepared_packet_transport_owner_retirement rank={} depth={} batch={} archive_submitted_before_capture={}",
                    self.cfg.rank, self.depth, scheduled_round, archive.is_some());
            }
            if let Some(sequence) = generation {
                // Both profiles retain complete route packets through their
                // transport/owner/materialization readers. The owner stream
                // has joined every peer exchange before publishing this event.
                // Reuse is a GPU dependency, not a host snapshot of counts.
                self.release_route_generation(self.active_route_bank, sequence)?;
                self.enqueue_route_prefetch(&mut producer_cursor, &mut produced)?;
            }
            if device_epoch {
                let slot = epoch_serial % self.epoch_completed.len();
                check(observed_native!(unsafe {
                    cudaEventRecord(self.epoch_completed[slot].0, s)
                }))?;
                self.epoch_outstanding.push_back(slot);
                epoch_serial = epoch_serial.checked_add(1).ok_or("EPOCH_SEQUENCE")?;
            }
            if trace_route {
                batch_index = batch_index.checked_add(1).ok_or("TRACE_BATCH_OVERFLOW")?;
            }
        }
        let _finalize_range = TraceRange::new(trace_ranges, b"mgbfs.FinalizeDepth\0");
        if !self.prefetched.is_empty() {
            return Err("ROUTE_PREFETCH_FINALIZE_LEAK".into());
        }
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
                check(observed_native!(mgbfs_compact_hash_layer(
                    owner.accepted.ptr,
                    owner.lengths.ptr.cast(),
                    self.cfg.buckets,
                    self.cfg.bucket_capacity,
                    self.prev.ptr,
                    self.cfg.layer_capacity,
                    self.directory.ptr.cast(),
                    self.route_banks[self.active_route_bank]
                        .route_count
                        .ptr
                        .cast(),
                    self.fatal.ptr.cast(),
                    s,
                )))?;
                check(observed_native!(traced_stream_synchronize(s)))?;
            }
        }
        if self.rank_owner_mode() {
            let ready = (|| -> Result<Vec<Extent>> {
                check(observed_native!(unsafe { traced_stream_synchronize(s) }))?;
                let control = self.control.one::<Control>()?;
                let ring = self.ring.one::<Ring>()?;
                if control.error != 0 || ring.fatal != 0 {
                    return Err(format!(
                        "LIBRARY_RANK_DEPTH_FATAL_{}_{}",
                        control.error, ring.fatal
                    ));
                }
                let count = self
                    .next_extent_count
                    .as_ref()
                    .ok_or("NEXT_EXTENT_COUNT_MISSING")?
                    .one::<u32>()? as usize;
                if count > 2 || !self.next.is_empty() {
                    return Err("NEXT_EXTENT_CAPACITY".into());
                }
                let mut extents = vec![Extent::default(); count];
                self.next_extents
                    .as_ref()
                    .ok_or("NEXT_EXTENTS_MISSING")?
                    .read(&mut extents)?;
                if extents
                    .iter()
                    .any(|e| e.ready != 1 || e.count == 0 || e.granted_rows as u64 != e.count)
                    || extents.iter().map(|e| e.count).sum::<u64>()
                        != u64::from(self.layer_count.one::<u32>()?)
                {
                    return Err("NEXT_EXTENT_MISMATCH".into());
                }
                Ok(extents)
            })();
            if self.all_max(u32::from(ready.is_err()))? != 0 {
                return Err(ready
                    .err()
                    .unwrap_or_else(|| "REMOTE_NEXT_EXTENT_FATAL".into()));
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
            self.route_banks[self.active_route_bank]
                .route_count
                .put(&[count])?;
        }
        if self.fatal.one::<u32>()? != 0 {
            return Err("FINALIZE_FATAL".into());
        }
        let count = self.route_banks[self.active_route_bank]
            .route_count
            .one::<u32>()?;
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
        let _native_scope = NativeCallMarker::enter(line!());
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
        self.archive_range_inner(archive,begin,count,extent_index,false)
    }
    fn archive_range_inner(
        &mut self,
        archive: &mut crate::pinned_archive::PinnedArchive,
        begin: u64,
        count: u64,
        extent_index: usize,
        snapshot: bool,
    ) -> Result<()> {
        let compact_permutation = self.permutation_n == u32::try_from(archive.width).ok();
        let _archive_range = TraceRange::new(
            std::env::var_os("MGBFS_TRACE_RANGES").is_some(), b"mgbfs.archive_d2h\0");
        let s = self.archive_stream.0;
        let mut offset = 0u64;
        while offset < count {
            // The compact u8 state plane can copy large contiguous blocks
            // independently of the compute batch. Matrix conversion still
            // uses its batch-sized device scratch buffer.
            let rows = if archive.state_only && self.width == archive.width {
                archive.rows
            } else { archive.rows.min(self.cfg.batch) };
            let n = u64::from(rows).min(count - offset) as u32;
            let slot = archive.acquire_cancellable(||
                !snapshot && (self.ensure_not_cancelled().is_err() ||
                self.failure_report.as_ref().is_some_and(|flag|
                    flag.load(std::sync::atomic::Ordering::Acquire)==2)))?;
            let copied = (|| unsafe {
                let states = self.states.at((begin + offset) as usize * self.stride);
                if !archive.state_only { check(mgbfs_hash_run(
                    self.archive_hash.0,
                    states.cast(),
                    self.archive_hashes.ptr.cast(),
                    n,
                    s,
                ))?; }
                if compact_permutation && self.width != archive.width && !snapshot {
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
                        archive.width,
                        states,
                        self.stride,
                        archive.width,
                        n as usize,
                        2,
                        s,
                    ))?;
                }
                if !archive.state_only { check(cudaMemcpyAsync(
                    slot.ptr.cast::<u8>().add(n as usize * archive.width).cast(),
                    self.archive_hashes.ptr,
                    n as usize * 16,
                    2,
                    s,
                ))?; }
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
        let _native_scope = NativeCallMarker::enter(line!());
        check(observed_native!(unsafe {
            traced_stream_synchronize(self.stream.0)
        }))?;
        let mut result = Vec::with_capacity(self.current_count as usize);
        for extent in &self.front {
            let mut bytes = vec![0u8; extent.count as usize * self.stride];
            check(observed_native!(unsafe {
                traced_device_copy(
                    bytes.as_mut_ptr().cast(),
                    self.states.at(extent.begin as usize * self.stride),
                    bytes.len(),
                    2,
                )
            }))?;
            result.extend(
                bytes
                    .chunks_exact(self.stride)
                    .map(|x| x[..self.width].to_vec()),
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod native_call_snapshot_tests {
    use super::*;
    #[test]
    fn native_call_snapshot_nested_disabled_and_unwind_restore() {
        NATIVE_CALL_SITE.store(0, std::sync::atomic::Ordering::Release);
        {
            let _outer = NativeCallMarker::enter_enabled(11, true);
            assert_eq!(
                NATIVE_CALL_SITE.load(std::sync::atomic::Ordering::Acquire),
                11
            );
            {
                let _inner = NativeCallMarker::enter_enabled(22, true);
                assert_eq!(
                    NATIVE_CALL_SITE.load(std::sync::atomic::Ordering::Acquire),
                    22
                );
            }
            assert_eq!(
                NATIVE_CALL_SITE.load(std::sync::atomic::Ordering::Acquire),
                11
            );
            let _disabled = NativeCallMarker::enter_enabled(33, false);
            assert_eq!(
                NATIVE_CALL_SITE.load(std::sync::atomic::Ordering::Acquire),
                11
            );
        }
        assert_eq!(
            NATIVE_CALL_SITE.load(std::sync::atomic::Ordering::Acquire),
            0
        );
        let result = std::panic::catch_unwind(|| {
            let _guard = NativeCallMarker::enter_enabled(44, true);
            panic!("injected");
        });
        assert!(result.is_err());
        assert_eq!(
            NATIVE_CALL_SITE.load(std::sync::atomic::Ordering::Acquire),
            0
        );
    }
}

#[cfg(test)]
mod weighted_producer_tests {
    use super::*;
    use mgbfs_core::{hash::Hash128, macro_generators::MacroGeneratorSet};

    fn fixture_config(variant: u32, prededup: bool) -> DistributedConfig {
        DistributedConfig {
            route_banks: 2,
            epoch_window: 2,
            rank: 0,
            world: 1,
            logical_owner_to_rank: vec![0, 0],
            transport: mgbfs_core::config::ReferenceTransport::HostSizedNccl,
            batch: 4,
            layer_capacity: 128,
            state_ring_capacity: 256,
            state_descriptor_capacity: 256,
            buckets: 8,
            shards: 2,
            job_buckets: 4,
            bucket_capacity: 128,
            prededup,
            generation_variant: variant,
            untouched_vram_reserve: 1 << 30,
        }
    }

    #[test]
    fn weighted_advance_preserves_original_depth_full_state_layers() {
        // Calling the real public advance path catches a missing settlement
        // driver, treating composed moves as unit cost, and future arrivals
        // incorrectly suppressing an earlier shortest-path discovery.
        let graph = MatrixGroup::symmetric_permutation_matrices(4).unwrap();
        let oracle = graph.exact_layers(24).unwrap();
        assert_eq!(oracle.iter().map(Vec::len).sum::<usize>(), 24);
        for (macro_depth, backend, backend_name, seed_id) in [
            (2, OwnerBackend::CubSortMerge, "cub", 0u128),
            (10, OwnerBackend::CubSortMerge, "cub", 0u128),
            (2, OwnerBackend::BmmaBucket, "bmma", 0u128),
            (10, OwnerBackend::BmmaBucket, "bmma", 0u128),
            (2, OwnerBackend::CubSortMerge, "cub", 1u128),
            (10, OwnerBackend::CubSortMerge, "cub", 1u128),
            (2, OwnerBackend::BmmaBucket, "bmma", 1u128),
            (10, OwnerBackend::BmmaBucket, "bmma", 1u128),
            (2, OwnerBackend::CubSortMerge, "cub", 20260828u128),
            (10, OwnerBackend::CubSortMerge, "cub", 20260828u128),
            (2, OwnerBackend::BmmaBucket, "bmma", 20260828u128),
            (10, OwnerBackend::BmmaBucket, "bmma", 20260828u128),
        ] {
            let seed = seed_id.to_le_bytes();
            let schedule = MacroGeneratorSet::compile_bounded(&graph, macro_depth, 128).unwrap();
            for prededup in [false, true] {
                let path = std::env::temp_dir().join(format!(
                    "mgbfs-weighted-archive-{}-{backend_name}-{macro_depth}-{prededup}-{seed_id}.bin",
                    std::process::id()
                ));
                let mut archive = crate::pinned_archive::PinnedArchive::new(
                    crate::archive::FileExtent::create_new(&path).unwrap(),
                    65536,
                    graph.start.len(),
                    [43; 32],
                    4,
                    128,
                )
                .unwrap();
                let mut id = [0; 128];
                assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
                let mut bfs = DistributedNativeBfs::new_weighted_with_owner_and_cancel(
                    &graph,
                    &schedule,
                    seed,
                    id,
                    fixture_config(1, prededup),
                    ReferenceOwner::Native(backend),
                    None,
                    256,
                    None,
                    None,
                )
                .unwrap();
                for (depth, expected) in oracle.iter().enumerate() {
                    assert_eq!(bfs.depth as usize, depth);
                    assert_eq!(bfs.current_count as usize, expected.len());
                    let mut storage = vec![0u8; bfs.cfg.state_ring_capacity as usize * bfs.stride];
                    // Test-only observation after FinalizeDepth, never a batch
                    // decision or an extra production state arena.
                    bfs.states.read(&mut storage).unwrap();
                    let mut actual = std::collections::BTreeSet::new();
                    let mut rows = 0usize;
                    for extent in &bfs.front {
                        assert_eq!(extent.ready, 1);
                        assert_eq!(extent.padding[1], extent.descriptor);
                        for row in 0..extent.count as usize {
                            let offset = (extent.begin as usize + row) * bfs.stride;
                            assert!(actual.insert(storage[offset..offset + bfs.width].to_vec()));
                            rows += 1;
                        }
                    }
                    let expected: std::collections::BTreeSet<_> =
                        expected.iter().cloned().collect();
                    assert_eq!(
                        actual, expected,
                        "K={macro_depth} depth={depth} pre={prededup}"
                    );
                    assert_eq!(rows, bfs.current_count as usize);
                    let live = bfs.advance_archived(&mut archive).unwrap_or_else(|error| {
                        let ring = bfs.ring.one::<Ring>();
                        let control = bfs.control.one::<Control>();
                        panic!(
                            "WEIGHTED_ADVANCE_FATAL seed={seed_id} backend={backend_name} K={macro_depth} depth={depth} pre={prededup} error={error} ring={ring:?} control={control:?}"
                        );
                    });
                    assert_eq!(live, depth + 1 < oracle.len());
                }
                assert_eq!(bfs.current_count, 0);
                assert!(bfs.front.is_empty());
                archive.finish().unwrap();
                let bytes = std::fs::read(&path).unwrap();
                crate::archive::verify(&bytes).unwrap();
                let hash = GemmHash::from_seed(bfs.width, seed).unwrap();
                let mut archived = vec![std::collections::BTreeSet::new(); oracle.len()];
                let mut offset = 48usize;
                loop {
                    let frame = &bytes[offset..offset + 80];
                    let word = |at: usize| {
                        u64::from_le_bytes(frame[at..at + 8].try_into().unwrap()) as usize
                    };
                    let (kind, depth, count, size) = (word(8), word(16), word(24), word(32));
                    let payload = &bytes[offset + 80..offset + 80 + size];
                    if kind == 1 {
                        assert!(depth < oracle.len());
                        for row in 0..count {
                            let state = &payload[row * bfs.width..(row + 1) * bfs.width];
                            let begin = count * bfs.width + row * 16;
                            let expected = hash.hash(state).unwrap();
                            for limb in 0..4 {
                                let at = begin + limb * 4;
                                assert_eq!(
                                    u32::from_le_bytes(payload[at..at + 4].try_into().unwrap()),
                                    expected.0[limb]
                                );
                            }
                            assert!(archived[depth].insert(state.to_vec()));
                        }
                    }
                    offset += 80 + size + 32;
                    if kind == 3 {
                        assert_eq!(depth, oracle.len());
                        assert_eq!(count, 24);
                        break;
                    }
                }
                for (actual, expected) in archived.iter().zip(&oracle) {
                    let expected: std::collections::BTreeSet<_> =
                        expected.iter().cloned().collect();
                    assert_eq!(*actual, expected);
                }
                std::fs::remove_file(&path).unwrap();
            }
        }
    }

    #[test]
    fn weighted_owner_registers_provisional_state_after_commit() {
        // The real route -> rank owner -> reserve -> materialize path must
        // publish a provisional descriptor. Missing registry integration makes
        // later settlement discard/reclamation unable to find these states.
        let graph = MatrixGroup::symmetric_permutation_matrices(4).unwrap();
        let schedule = MacroGeneratorSet::compile_bounded(&graph, 2, 128).unwrap();
        let mut id = [0; 128];
        assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
        let mut bfs = DistributedNativeBfs::new_weighted_with_owner_and_cancel(
            &graph,
            &schedule,
            [29; 16],
            id,
            fixture_config(1, false),
            ReferenceOwner::Native(OwnerBackend::CubSortMerge),
            None,
            256,
            None,
            None,
        )
        .unwrap();
        let begin = bfs.owner_window.as_ref().unwrap().ptr.cast();
        bfs.owner_window.as_ref().unwrap().put(&[0u32; 3]).unwrap();
        let sequence = bfs
            .enqueue_frontier_generation(
                ParentBatch {
                    extent: 0,
                    offset: 0,
                    begin: 0,
                    sequence: 0,
                    count: 1,
                },
                0,
            )
            .unwrap();
        for weight in [1, 2] {
            bfs.enqueue_weighted_route_packet(0, sequence, 1, weight)
                .unwrap();
            let packet = &bfs.route_banks[0];
            let (states, hashes, rows, source_rows) = (
                packet.packed_states.ptr.cast(),
                packet.sorted_hashes.ptr.cast_const(),
                packet.owner_counts.ptr.cast(),
                packet.route_count.ptr.cast(),
            );
            bfs.commit_rank_native_batch(states, hashes, begin, rows, source_rows, 0, Some(weight))
                .unwrap();
            bfs.release_weighted_route_packet(0, sequence).unwrap();
        }
        // Observation only after the entire real owner DAG has been queued.
        check(unsafe { traced_stream_synchronize(bfs.stream.0) }).unwrap();
        let ring = bfs.ring.one::<Ring>().unwrap();
        assert_eq!(ring.fatal, 0);
        assert_eq!(ring.descriptor_tail, 3);
        assert_eq!(
            bfs.next_extent_count
                .as_ref()
                .unwrap()
                .one::<u32>()
                .unwrap(),
            0
        );
        let descriptors = bfs.cfg.state_descriptor_capacity as usize;
        let mut records = vec![WeightedExtent::default(); descriptors];
        bfs.weighted_extents
            .as_ref()
            .unwrap()
            .read(&mut records)
            .unwrap();
        for descriptor in 1..ring.descriptor_tail {
            let record = records[(descriptor % ring.descriptor_capacity) as usize];
            assert_eq!(record.descriptor, descriptor);
            assert_eq!(record.phase, 2);
            assert_eq!(record.target_depth, descriptor as u32);
            assert!(record.count > 0);
        }
        // Independent target-depth oracle after both real commits. A global
        // accepted table incorrectly mixes future depths even if refs are valid.
        let owner = bfs.owner.as_ref().unwrap();
        let refs = bfs.weighted_owner_refs.as_ref().unwrap();
        let records_per_target = (bfs.cfg.buckets * bfs.cfg.bucket_capacity) as usize;
        let mut lengths = vec![0u32; bfs.cfg.buckets as usize * 2];
        let mut keys = vec![Hash128([0; 4]); records_per_target * 2];
        let mut sequences = vec![0u64; records_per_target * 2];
        let mut states = vec![0u8; bfs.cfg.state_ring_capacity as usize * bfs.stride];
        owner.lengths.read(&mut lengths).unwrap();
        owner.accepted.read(&mut keys).unwrap();
        refs.accepted.read(&mut sequences).unwrap();
        bfs.states.read(&mut states).unwrap();
        let hash = GemmHash::from_seed(bfs.width, [29; 16]).unwrap();
        for target in [1u32, 2] {
            let slot = target as usize % 2;
            let mut expected = std::collections::BTreeSet::new();
            for transition in &schedule.transitions {
                if transition.weight == target {
                    let state = graph.apply_left(&transition.matrix, &graph.start).unwrap();
                    expected.insert(hash.hash(&state).unwrap());
                }
            }
            let mut actual = std::collections::BTreeSet::new();
            for bucket in 0..bfs.cfg.buckets as usize {
                let length = lengths[slot * bfs.cfg.buckets as usize + bucket];
                for row in 0..length as usize {
                    let index =
                        slot * records_per_target + bucket * bfs.cfg.bucket_capacity as usize + row;
                    let sequence = sequences[index];
                    assert!(sequence > 0 && sequence < ring.tail, "OWNER_STATE_REF_LIVE");
                    assert!(
                        records.iter().any(|record| {
                            record.phase == 2
                                && record.target_depth == target
                                && sequence >= record.sequence
                                && sequence - record.sequence < record.count
                        }),
                        "WEIGHTED_TARGET_OWNER_REF_DEPTH"
                    );
                    let begin = (sequence % ring.capacity) as usize * bfs.stride;
                    assert_eq!(
                        hash.hash(&states[begin..begin + bfs.width]).unwrap(),
                        keys[index]
                    );
                    assert!(actual.insert(keys[index]));
                }
            }
            assert_eq!(
                actual, expected,
                "WEIGHTED_TARGET_OWNER_KEYS target={target}"
            );
        }
    }

    // Test-owned fixture storage stays alive through the complete submitted DAG.
    // It never enters the production runtime or contributes extra state arenas.
    struct WeightedRingFixture {
        bfs: DistributedNativeBfs,
        input: Buffer,
        selected: Buffer,
        reserve: Vec<Buffer>,
        materialize: Vec<Buffer>,
        extents: Vec<Buffer>,
        checkpoint: Buffer,
        padded: Vec<u8>,
    }
    impl WeightedRingFixture {
        fn new(ring_capacity: u32, descriptor_capacity: u32) -> Self {
            let graph = MatrixGroup::symmetric_permutation_matrices(4).unwrap();
            let schedule = MacroGeneratorSet::compile_bounded(&graph, 2, 128).unwrap();
            let mut id = [0; 128];
            assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
            let mut cfg = fixture_config(5, false);
            cfg.state_ring_capacity = ring_capacity;
            cfg.state_descriptor_capacity = descriptor_capacity;
            cfg.layer_capacity = ring_capacity;
            let bfs = DistributedNativeBfs::new_weighted_with_owner_and_cancel(
                &graph,
                &schedule,
                [29; 16],
                id,
                cfg,
                ReferenceOwner::Native(OwnerBackend::CubSortMerge),
                None,
                256,
                None,
                None,
            )
            .unwrap();
            assert_eq!(bfs.current_count, 1);
            let s = bfs.stream.0;
            let checkpoint = Buffer::new(64, s).unwrap();
            let selected = Buffer::new(12, s).unwrap();
            selected.put(&[0u32, 1, 2]).unwrap();
            let input = Buffer::new(3 * bfs.stride, s).unwrap();
            let state = encode_permutation_matrix(&graph.start, graph.rows).unwrap();
            let mut padded = vec![0u8; 3 * bfs.stride];
            for row in 0..3 {
                padded[row * bfs.stride..row * bfs.stride + state.len()].copy_from_slice(&state);
            }
            input.put(&padded).unwrap();
            let mut reserve = Vec::new();
            let mut materialize = Vec::new();
            let mut extents = Vec::new();
            for rows in [3u32, 1, 1, 3] {
                let control = Buffer::new(64, s).unwrap();
                control
                    .put(&[Control {
                        stage: 1,
                        survivors: rows,
                        ..Control::default()
                    }])
                    .unwrap();
                reserve.push(control);
                let control = Buffer::new(64, s).unwrap();
                control
                    .put(&[Control {
                        stage: 2,
                        survivors: rows,
                        ..Control::default()
                    }])
                    .unwrap();
                materialize.push(control);
                extents.push(Buffer::new(64, s).unwrap());
            }
            Self {
                bfs,
                input,
                selected,
                reserve,
                materialize,
                extents,
                checkpoint,
                padded,
            }
        }
        fn enqueue_allocation(&self, index: usize, target: u32, provisional: bool) {
            let bfs = &self.bfs;
            unsafe {
                check(mgbfs_state_reserve(
                    bfs.ring.ptr.cast(),
                    self.reserve[index].ptr.cast(),
                    self.extents[index].ptr.cast(),
                    bfs.stream.0,
                ))
                .unwrap();
                check(mgbfs_state_materialize_packed(
                    self.input.ptr.cast(),
                    3,
                    self.selected.ptr.cast(),
                    3,
                    bfs.stride as u32,
                    bfs.states.ptr.cast(),
                    bfs.ring.ptr.cast(),
                    self.materialize[index].ptr.cast(),
                    self.extents[index].ptr.cast(),
                    bfs.stream.0,
                ))
                .unwrap();
            }
            bfs.enqueue_weighted_extent_register(
                self.extents[index].ptr.cast(),
                self.materialize[index].ptr.cast(),
                target,
                provisional,
            )
            .unwrap();
        }
        fn enqueue_near_depth_release(&self) {
            for index in 0..3 {
                self.enqueue_allocation(index, [3, 1, 1][index], index != 2);
            }
            self.bfs
                .enqueue_weighted_extent_retire_after_readers(self.bfs.front[0], 1)
                .unwrap();
            self.bfs
                .enqueue_weighted_discard_depth_after_readers(1)
                .unwrap();
            let settled = Extent {
                sequence: 5,
                begin: 5,
                count: 1,
                descriptor: 3,
                granted_rows: 1,
                ready: 1,
                padding: [0, 3, 0],
            };
            self.bfs
                .enqueue_weighted_extent_retire_after_readers(settled, 1)
                .unwrap();
        }
        fn observe(&self) -> Ring {
            check(unsafe { traced_stream_synchronize(self.bfs.stream.0) }).unwrap();
            self.bfs.ring.one::<Ring>().unwrap()
        }
    }

    #[test]
    fn weighted_reclamation_keeps_an_older_future_allocation_live() {
        // Missing live-prefix protection, target filtering, or the independent
        // reader join breaks these hand-derived offsets and actual state bytes.
        for capturing in [false, true] {
            let f = WeightedRingFixture::new(256, 256);
            let live = Buffer::new(3 * f.bfs.stride, f.bfs.stream.0).unwrap();
            let ready = Event::new().unwrap();
            let consumed = Event::new().unwrap();
            #[cfg(debug_assertions)]
            let capture = OwnerCaptureProbe::begin_enabled(f.bfs.stream.0, capturing).unwrap();
            f.enqueue_near_depth_release();
            unsafe {
                check(cudaMemcpyAsync(
                    f.checkpoint.ptr,
                    f.bfs.ring.ptr,
                    64,
                    3,
                    f.bfs.stream.0,
                ))
                .unwrap();
                check(cudaEventRecord(ready.0, f.bfs.stream.0)).unwrap();
                check(cudaStreamWaitEvent(f.bfs.exchange_stream.0, ready.0, 0)).unwrap();
                check(cudaMemcpyAsync(
                    live.ptr,
                    f.bfs.states.ptr.cast::<u8>().add(f.bfs.stride).cast(),
                    3 * f.bfs.stride,
                    3,
                    f.bfs.exchange_stream.0,
                ))
                .unwrap();
                check(cudaEventRecord(consumed.0, f.bfs.exchange_stream.0)).unwrap();
                check(cudaStreamWaitEvent(f.bfs.stream.0, consumed.0, 0)).unwrap();
            }
            f.bfs
                .enqueue_weighted_discard_depth_after_readers(3)
                .unwrap();
            #[cfg(debug_assertions)]
            if let Some(capture) = capture {
                capture
                    .launch_named("MGBFS_WEIGHTED_RING_READER_DAG_CAPTURE")
                    .unwrap();
            }
            // Only the final oracle observes device state; capture/submission
            // above contain no host count/control readback or stream drain.
            let final_ring = f.observe();
            let ring = f.checkpoint.one::<Ring>().unwrap();
            assert_eq!(
                ring.fatal, 0,
                "weighted release must not require allocation FIFO order"
            );
            assert_eq!(
                (
                    ring.head,
                    ring.tail,
                    ring.descriptor_head,
                    ring.descriptor_tail
                ),
                (1, 6, 1, 4),
                "depth-3 states are still live despite depth-1 release"
            );
            let mut observed = vec![0u8; 3 * f.bfs.stride];
            live.read(&mut observed).unwrap();
            assert_eq!(observed, f.padded);
            assert_eq!(final_ring.fatal, 0);
            assert_eq!(
                (
                    final_ring.head,
                    final_ring.tail,
                    final_ring.descriptor_head,
                    final_ring.descriptor_tail
                ),
                (6, 6, 4, 4)
            );
        }
    }

    #[test]
    fn weighted_capacity_rejection_never_reuses_an_older_future_extent() {
        // Physical wrap needs padding. Logical release of later allocations
        // cannot make the earlier live states available to reservation.
        let f = WeightedRingFixture::new(8, 4);
        f.enqueue_near_depth_release();
        check(unsafe {
            mgbfs_state_reserve(
                f.bfs.ring.ptr.cast(),
                f.reserve[3].ptr.cast(),
                f.extents[3].ptr.cast(),
                f.bfs.stream.0,
            )
        })
        .unwrap();
        let ring = f.observe();
        assert_eq!(ring.fatal, 11);
        assert_eq!(
            (
                ring.head,
                ring.tail,
                ring.descriptor_head,
                ring.descriptor_tail
            ),
            (1, 6, 1, 4)
        );
        let rejected = f.extents[3].one::<Extent>().unwrap();
        assert_eq!((rejected.count, rejected.ready), (0, 0));
        let mut live = vec![0u8; 3 * f.bfs.stride];
        check(unsafe {
            cudaMemcpy(
                live.as_mut_ptr().cast(),
                f.bfs.states.ptr.cast::<u8>().add(f.bfs.stride).cast(),
                live.len(),
                2,
            )
        })
        .unwrap();
        assert_eq!(live, f.padded);
    }

    #[test]
    fn weighted_wrapped_descriptor_reuse_rejects_a_stale_current_release() {
        // Reusing descriptor slot zero must not let an old descriptor-zero
        // snapshot release its new generation (absolute descriptor four).
        let f = WeightedRingFixture::new(8, 4);
        f.enqueue_near_depth_release();
        f.bfs
            .enqueue_weighted_discard_depth_after_readers(3)
            .unwrap();
        f.enqueue_allocation(3, 4, false);
        f.bfs
            .enqueue_weighted_extent_retire_after_readers(f.bfs.front[0], 1)
            .unwrap();
        let ring = f.observe();
        assert_eq!(ring.fatal, 28);
        assert_eq!(
            (
                ring.head,
                ring.tail,
                ring.descriptor_head,
                ring.descriptor_tail
            ),
            (8, 11, 4, 5)
        );
        let mut current = vec![0u8; 3 * f.bfs.stride];
        f.bfs.states.read(&mut current).unwrap();
        assert_eq!(current, f.padded);
    }

    #[test]
    fn weighted_producer_routes_exact_states_and_hashes_for_partial_and_empty_batches() {
        // Wrong span origin, parent-major output, hash seed, packed reference,
        // or prededup count must disagree with this independent matrix oracle.
        for (graph, variant) in [
            (MatrixGroup::unitriangular(3, 3).unwrap(), 1),
            (MatrixGroup::symmetric_permutation_matrices(4).unwrap(), 5),
        ] {
            let mut parents = vec![graph.successor(&graph.start, 0).unwrap()];
            parents.push(graph.successor(&parents[0], 1).unwrap());
            parents.push(graph.successor(&parents[1], 0).unwrap());
            for depth in [2, 10] {
                let schedule = MacroGeneratorSet::compile_bounded(&graph, depth, 128).unwrap();
                for prededup in [false, true] {
                    let mut id = [0; 128];
                    assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
                    let cfg = fixture_config(variant, prededup);
                    let seed = [29; 16];
                    let mut bfs = DistributedNativeBfs::new_weighted_with_owner_and_cancel(
                        &graph,
                        &schedule,
                        seed,
                        id,
                        cfg,
                        ReferenceOwner::Native(OwnerBackend::CubSortMerge),
                        None,
                        256,
                        None,
                        None,
                    )
                    .unwrap();
                    let encode = |state: Vec<u8>| {
                        if variant == 5 {
                            encode_permutation_matrix(&state, graph.rows).unwrap()
                        } else {
                            state
                        }
                    };
                    let hash = GemmHash::from_seed(bfs.width, seed).unwrap();
                    let mut padded = vec![0u8; parents.len() * bfs.stride];
                    for (row, parent) in parents.iter().enumerate() {
                        padded[row * bfs.stride..row * bfs.stride + bfs.width]
                            .copy_from_slice(&encode(parent.clone()));
                    }
                    bfs.states.put(&padded).unwrap();
                    for parent_count in [3u32, 0, 1, 3] {
                        let bank = (bfs.generation_sequence % 2) as usize;
                        let sequence = bfs
                            .enqueue_frontier_generation(
                                ParentBatch {
                                    extent: 0,
                                    offset: 0,
                                    begin: 0,
                                    sequence: 0,
                                    count: parent_count,
                                },
                                bank,
                            )
                            .unwrap();
                        for weight in 1..=schedule.effective_depth {
                            let mut expected = Vec::new();
                            for transition in &schedule.transitions {
                                if transition.weight == weight {
                                    for parent in parents.iter().take(parent_count as usize) {
                                        let state = encode(
                                            graph.apply_left(&transition.matrix, parent).unwrap(),
                                        );
                                        expected.push((hash.hash(&state).unwrap(), state));
                                    }
                                }
                            }
                            expected.sort_by_key(|row| row.0);
                            if prededup {
                                expected.dedup_by_key(|row| row.0);
                            }
                            bfs.enqueue_weighted_route_packet(bank, sequence, parent_count, weight)
                                .unwrap();
                            // A packet has transport/owner readers until explicitly
                            // released. Reissuing it must fail before overwriting output.
                            let overwrite = bfs.enqueue_weighted_route_packet(
                                bank,
                                sequence,
                                parent_count,
                                weight,
                            );
                            assert_eq!(
                                overwrite.unwrap_err(),
                                "WEIGHTED_ROUTE_READER_NOT_RELEASED"
                            );
                            // Observation only: production routing uses a GPU event wait.
                            check(unsafe { traced_stream_synchronize(bfs.stream.0) }).unwrap();
                            let packet = &bfs.route_banks[bank];
                            let count = packet.route_count.one::<u32>().unwrap();
                            assert_eq!(count as usize, expected.len());
                            assert_eq!(packet.owner_counts.one::<u32>().unwrap(), count);
                            let mut hashes = vec![Hash128([0; 4]); count as usize];
                            let mut states = vec![0u8; count as usize * bfs.stride];
                            packet.sorted_hashes.read(&mut hashes).unwrap();
                            packet.packed_states.read(&mut states).unwrap();
                            let actual: Vec<_> = hashes
                                .into_iter()
                                .zip(states.chunks_exact(bfs.stride))
                                .map(|(h, state)| (h, state[..bfs.width].to_vec()))
                                .collect();
                            assert_eq!(actual, expected, "K={depth} w={weight} parents={parent_count} variant={variant} pre={prededup}");
                            bfs.release_weighted_route_packet(bank, sequence).unwrap();
                        }
                        assert!(bfs.route_banks[bank].weighted_cursor.is_none());
                    }
                }
            }
        }
    }

    #[test]
    fn weighted_bank_reuse_preserves_queued_transport_readers_without_host_readbacks() {
        // Distinct parents expose premature producer/packed-bank reuse. Keep
        // two raw banks prefetched and join a real independent GPU reader.
        let graph = MatrixGroup::symmetric_permutation_matrices(4).unwrap();
        let schedule = MacroGeneratorSet::compile(&graph, 2).unwrap();
        let parents: Vec<_> = graph
            .exact_layers(24)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let mut id = [0; 128];
        assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
        let seed = [47; 16];
        let mut bfs = DistributedNativeBfs::new_weighted_with_owner_and_cancel(
            &graph,
            &schedule,
            seed,
            id,
            fixture_config(5, false),
            ReferenceOwner::Native(OwnerBackend::CubSortMerge),
            None,
            256,
            None,
            None,
        )
        .unwrap();
        let batches = 64usize;
        let runs = schedule.effective_depth as usize;
        let records = batches * runs;
        let capacity = bfs.candidates as usize;
        let hash = GemmHash::from_seed(bfs.width, seed).unwrap();
        let mut input = vec![0; batches * bfs.stride];
        for row in 0..batches {
            input[row * bfs.stride..row * bfs.stride + bfs.width].copy_from_slice(
                &encode_permutation_matrix(&parents[row % parents.len()], graph.rows).unwrap(),
            );
        }
        bfs.states.put(&input).unwrap();
        // Test observation storage is allocated before submission, never a
        // runtime payload slot. Every consumer copy has a fixed oracle bound.
        let states = Buffer::new(records * capacity * bfs.stride, bfs.stream.0).unwrap();
        let hashes = Buffer::new(records * capacity * 16, bfs.stream.0).unwrap();
        let counts = Buffer::new(records * 4, bfs.stream.0).unwrap();
        let packet_ready = Event::new().unwrap();
        let transport_done = Event::new().unwrap();
        let parent_batch = |row: usize| ParentBatch {
            extent: 0,
            offset: row as u64,
            begin: row as u64,
            sequence: row as u64,
            count: 1,
        };
        let mut generations = [0; 2];
        for (bank, sequence) in generations.iter_mut().enumerate() {
            *sequence = bfs
                .enqueue_frontier_generation(parent_batch(bank), bank)
                .unwrap();
        }
        let mut expected = Vec::with_capacity(records);
        for row in 0..batches {
            let bank = row % 2;
            let sequence = generations[bank];
            for weight in 1..=schedule.effective_depth {
                let record = row * runs + (weight as usize - 1);
                let mut wanted = Vec::new();
                for transition in &schedule.transitions {
                    if transition.weight == weight {
                        let state = encode_permutation_matrix(
                            &graph
                                .apply_left(&transition.matrix, &parents[row % parents.len()])
                                .unwrap(),
                            graph.rows,
                        )
                        .unwrap();
                        wanted.push((hash.hash(&state).unwrap(), state));
                    }
                }
                wanted.sort_by_key(|record| record.0);
                bfs.enqueue_weighted_route_packet(bank, sequence, 1, weight)
                    .unwrap();
                let packet = &bfs.route_banks[bank];
                unsafe {
                    check(cudaEventRecord(packet_ready.0, bfs.stream.0)).unwrap();
                    check(cudaStreamWaitEvent(
                        bfs.exchange_stream.0,
                        packet_ready.0,
                        0,
                    ))
                    .unwrap();
                    check(cudaMemcpyAsync(
                        states.at(record * capacity * bfs.stride),
                        packet.packed_states.ptr,
                        wanted.len() * bfs.stride,
                        3,
                        bfs.exchange_stream.0,
                    ))
                    .unwrap();
                    check(cudaMemcpyAsync(
                        hashes.at(record * capacity * 16),
                        packet.sorted_hashes.ptr,
                        wanted.len() * 16,
                        3,
                        bfs.exchange_stream.0,
                    ))
                    .unwrap();
                    check(cudaMemcpyAsync(
                        counts.at(record * 4),
                        packet.route_count.ptr,
                        4,
                        3,
                        bfs.exchange_stream.0,
                    ))
                    .unwrap();
                    check(cudaEventRecord(transport_done.0, bfs.exchange_stream.0)).unwrap();
                    check(cudaStreamWaitEvent(bfs.stream.0, transport_done.0, 0)).unwrap();
                }
                // No EventQuery, GPU-count read or host synchronization before
                // the next weight/batch. Reader joins and producer reuse are GPU edges.
                bfs.release_weighted_route_packet(bank, sequence).unwrap();
                expected.push(wanted);
            }
            if row + 2 < batches {
                generations[bank] = bfs
                    .enqueue_frontier_generation(parent_batch(row + 2), bank)
                    .unwrap();
            }
        }
        // Single final observation, outside the queued pipeline.
        check(unsafe { traced_stream_synchronize(bfs.stream.0) }).unwrap();
        let mut actual_counts = vec![0u32; records];
        let mut actual_hashes = vec![Hash128([0; 4]); records * capacity];
        let mut actual_states = vec![0u8; records * capacity * bfs.stride];
        counts.read(&mut actual_counts).unwrap();
        hashes.read(&mut actual_hashes).unwrap();
        states.read(&mut actual_states).unwrap();
        for (record, wanted) in expected.iter().enumerate() {
            assert_eq!(actual_counts[record] as usize, wanted.len());
            let actual: Vec<_> = (0..wanted.len())
                .map(|row| {
                    let index = record * capacity + row;
                    (
                        actual_hashes[index],
                        actual_states[index * bfs.stride..index * bfs.stride + bfs.width].to_vec(),
                    )
                })
                .collect();
            assert_eq!(&actual, wanted, "record={record}");
        }
    }
}
