//! Native single-rank DENSE executor. States never leave GPU except explicit
//! snapshots/archive callbacks. Host dispatch consumes only bucket metadata.
use crate::jobs::{split, JobSpan};
use mgbfs_core::{hash::GemmHash, matrix::MatrixGroup, Result};
use mgbfs_cuda::{ffi::*, native_owner::*};
use std::ffi::{c_void, CStr};
#[derive(Clone, Copy)]
pub struct NativeConfig {
    pub batch: u32,
    pub layer_capacity: u32,
    pub buckets: u32,
    pub shards: u32,
    pub job_buckets: u32,
    pub bucket_capacity: u32,
    pub prededup: bool,
}
fn check(x: i32) -> Result<()> {
    if x == 0 {
        Ok(())
    } else {
        Err(format!("CUDA_STATUS_{x}"))
    }
}
struct Buffer {
    p: *mut c_void,
    bytes: usize,
    stream: *mut c_void,
}
impl Buffer {
    fn new(bytes: usize, stream: *mut c_void) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut p, bytes.max(1)) })?;
        let b = Self { p, bytes, stream };
        check(unsafe { cudaMemsetAsync(p, 0, bytes.max(1), stream) })?;
        Ok(b)
    }
    fn put<T: Copy>(&self, v: &[T]) -> Result<()> {
        if std::mem::size_of_val(v) > self.bytes {
            return Err("UPLOAD_CAPACITY".into());
        }
        // Order pageable uploads with the nonblocking consumer stream and keep
        // the source alive until DMA completes; default-stream staging is insufficient.
        check(unsafe {
            cudaMemcpyAsync(
                self.p,
                v.as_ptr().cast(),
                std::mem::size_of_val(v),
                1,
                self.stream,
            )
        })?;
        check(unsafe { cudaStreamSynchronize(self.stream) })
    }
    fn read_into<T: Copy>(&self, v: &mut [T]) -> Result<()> {
        if std::mem::size_of_val(v) > self.bytes {
            return Err("READ_CAPACITY".into());
        }
        check(unsafe { cudaMemcpy(v.as_mut_ptr().cast(), self.p, std::mem::size_of_val(v), 2) })
    }
    fn one<T: Copy + Default>(&self) -> Result<T> {
        let mut v = [T::default()];
        self.read_into(&mut v)?;
        Ok(v[0])
    }
    unsafe fn at(&self, offset: usize) -> *mut c_void {
        self.p.cast::<u8>().add(offset).cast()
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.p);
        }
    }
}
struct Plan(*mut c_void, unsafe extern "C" fn(*mut c_void));
impl Plan {
    fn new(
        destroy: unsafe extern "C" fn(*mut c_void),
        f: impl FnOnce(*mut *mut c_void, *mut i8) -> i32,
    ) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        let mut e = [0i8; 512];
        if f(&mut p, e.as_mut_ptr()) != 0 {
            return Err(unsafe { CStr::from_ptr(e.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        Ok(Self(p, destroy))
    }
}
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe { self.1(self.0) }
    }
}
struct Stream(*mut c_void);
struct Event(*mut c_void);
impl Event {
    fn new() -> Result<Self> {
        let mut event = std::ptr::null_mut();
        check(unsafe { cudaEventCreateWithFlags(&mut event, 2) })?;
        Ok(Self(event))
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            cudaEventDestroy(self.0);
        }
    }
}
struct ProducerBuffers {
    children: Buffer,
    hashes: Buffer,
    sorted: Buffer,
    refs: Buffer,
    routed_count: Buffer,
    directory: Buffer,
    fatal: Buffer,
    ready: Event,
}
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            cudaStreamSynchronize(self.0);
            cudaStreamDestroy(self.0);
        }
    }
}
pub struct NativeBfs {
    cfg: NativeConfig,
    width: usize,
    stride: usize,
    moves: u32,
    candidates: u32,
    depth: u32,
    failed: bool,
    generate: Plan,
    hash: Plan,
    archive_hash: Plan,
    archive_hashes: Buffer,
    archive_done: [Event; 2],
    archived_depth: Option<u32>,
    archive_stream: Stream,
    route: Plan,
    owner: Plan,
    states: Buffer,
    prev: Buffer,
    curr: Buffer,
    accepted: Buffer,
    lengths: Buffer,
    children: Buffer,
    hashes: Buffer,
    origins: Buffer,
    sorted: Buffer,
    refs: Buffer,
    routed_count: Buffer,
    directory: Buffer,
    fatal: Buffer,
    jobs_gpu: Buffer,
    counts: Buffer,
    control: Buffer,
    selected: Buffer,
    ring: Buffer,
    extent: Buffer,
    layer_count: Buffer,
    incoming_dir: Vec<Range>,
    prev_dir: Vec<Range>,
    curr_dir: Vec<Range>,
    descriptors: Vec<BucketJob>,
    spans: Vec<JobSpan>,
    front: Vec<Extent>,
    next: Vec<Extent>,
    front_count: u32,
    prev_count: u32,
    plan_bytes: u64,
    alternate: ProducerBuffers,
    ready: Event,
    producer_stream: Stream,
    stream: Stream,
}
impl Drop for NativeBfs {
    fn drop(&mut self) {
        unsafe {
            cudaStreamSynchronize(self.producer_stream.0);
            cudaStreamSynchronize(self.stream.0);
            cudaStreamSynchronize(self.archive_stream.0);
        }
    }
}
impl NativeBfs {
    pub fn new(g: &MatrixGroup, seed: [u8; 16], cfg: NativeConfig) -> Result<Self> {
        Self::new_with_reserve(g, seed, cfg, 1 << 30)
    }
    pub fn new_with_reserve(
        g: &MatrixGroup,
        seed: [u8; 16],
        cfg: NativeConfig,
        reserve: u64,
    ) -> Result<Self> {
        Self::new_with_generation_and_reserve(g, seed, cfg, 0, reserve)
    }
    pub fn new_with_generation(
        g: &MatrixGroup,
        seed: [u8; 16],
        cfg: NativeConfig,
        generation: u32,
    ) -> Result<Self> {
        Self::new_with_generation_and_reserve(g, seed, cfg, generation, 1 << 30)
    }
    pub fn new_with_generation_and_reserve(
        g: &MatrixGroup,
        seed: [u8; 16],
        cfg: NativeConfig,
        generation: u32,
        reserve: u64,
    ) -> Result<Self> {
        g.validate()?;
        let moves = u32::try_from(g.generators.len()).map_err(|_| "MOVES")?;
        let c = cfg.batch.checked_mul(moves).ok_or("CANDIDATE_OVERFLOW")?;
        if c == 0
            || c > i32::MAX as u32
            || cfg.layer_capacity == 0
            || cfg.layer_capacity > i32::MAX as u32
            || !cfg.buckets.is_power_of_two()
            || !cfg.shards.is_power_of_two()
            || cfg.shards > cfg.buckets
            || cfg.job_buckets == 0
            || cfg.job_buckets > cfg.buckets / cfg.shards
            || cfg.job_buckets > c
            || cfg.bucket_capacity == 0
        {
            return Err("NATIVE_CONFIG".into());
        }
        let width = g.start.len();
        let stride = (width + 15) & !15;
        let h = GemmHash::from_seed(width, seed)?;
        let limbs = h.limbs();
        let gens: Vec<_> = g.generators.iter().flatten().copied().collect();
        let b = cfg.buckets as usize;
        let f = cfg.layer_capacity as usize;
        let cap = c as usize;
        // I equals this producer slot's maximum row count; a bucket therefore
        // appears at most once in the batch directory/job descriptor pool.
        let slots = b + 1;
        let ring_capacity = f
            .checked_mul(2)
            .and_then(|v| v.checked_add(cap))
            .ok_or("RING_OVERFLOW")?;
        unsafe {
            let (mut gb, mut hb, mut rb) = (
                GenerateBytes::default(),
                HashBytes::default(),
                RouteBytes::default(),
            );
            check(mgbfs_generate_query(
                g.rows as u32,
                moves,
                g.modulus as u32,
                cfg.batch,
                generation,
                &mut gb,
            ))?;
            check(mgbfs_hash_query(width as u32, c, &mut hb))?;
            let mut ahb = HashBytes::default();
            check(mgbfs_hash_query(width as u32, cfg.batch, &mut ahb))?;
            check(mgbfs_route_query(c, &mut rb))?;
            let plan_bytes = [
                gb.generators,
                gb.packed_parents,
                gb.products_s32,
                gb.workspace,
                hb.weights,
                hb.offsets,
                hb.partials_s32,
                hb.workspace,
                ahb.weights,
                ahb.offsets,
                ahb.partials_s32,
                ahb.workspace,
                rb.sorted,
                rb.refs,
                rb.indices,
                rb.selected,
                rb.flags,
                rb.scratch,
            ]
            .iter()
            .map(|&n| u128::from(n))
            .sum::<u128>()
                + u128::from(c) * 5
                + u128::from(cfg.job_buckets) * u128::from(cfg.bucket_capacity) * 16;
            let buffers = ring_capacity as u128 * stride as u128
                + f as u128 * 32
                + b as u128 * u128::from(cfg.bucket_capacity) * 16
                + 2 * cap as u128 * stride as u128
                + cap as u128 * 92
                + u128::from(cfg.batch) * 16
                + b as u128 * 36
                + slots as u128 * 64
                + u128::from(cfg.job_buckets) * 32
                + 212;
            let requested =
                u64::try_from(plan_bytes + buffers).map_err(|_| "VRAM_PLAN_OVERFLOW")?;
            let plan_bytes = u64::try_from(plan_bytes).map_err(|_| "VRAM_PLAN_OVERFLOW")?;
            let (mut free, mut total) = (0usize, 0usize);
            check(cudaMemGetInfo(&mut free, &mut total))?;
            if u128::from(requested) + u128::from(reserve) > free as u128 {
                return Err(format!(
                    "VRAM_PREFLIGHT requested={requested} reserve={reserve} free={free}"
                ));
            }
            let mut s = std::ptr::null_mut();
            check(cudaStreamCreateWithFlags(&mut s, 1))?;
            let stream = Stream(s);
            let mut producer = std::ptr::null_mut();
            check(cudaStreamCreateWithFlags(&mut producer, 1))?;
            let producer_stream = Stream(producer);
            let mut archive_lane = std::ptr::null_mut();
            check(cudaStreamCreateWithFlags(&mut archive_lane, 1))?;
            let archive_stream = Stream(archive_lane);
            let generate = Plan::new(mgbfs_generate_destroy, |p, e| {
                mgbfs_generate_create_variant(
                    g.rows as u32,
                    moves,
                    g.modulus as u32,
                    cfg.batch,
                    gens.as_ptr(),
                    generation,
                    p,
                    e,
                    512,
                )
            })?;
            let hash = Plan::new(mgbfs_hash_destroy, |p, e| {
                mgbfs_hash_create(
                    width as u32,
                    c,
                    limbs.as_ptr(),
                    h.offsets.as_ptr(),
                    p,
                    e,
                    512,
                )
            })?;
            let route = Plan::new(mgbfs_route_destroy, |p, e| mgbfs_route_create(c, p, e, 512))?;
            let archive_hash = Plan::new(mgbfs_hash_destroy, |p, e| {
                mgbfs_hash_create(
                    width as u32,
                    cfg.batch,
                    limbs.as_ptr(),
                    h.offsets.as_ptr(),
                    p,
                    e,
                    512,
                )
            })?;
            let owner = Plan::new(mgbfs_bounded_owner_destroy, |p, _| {
                mgbfs_bounded_owner_create(c, cfg.job_buckets, cfg.bucket_capacity, p)
            })?;
            let buffer = |bytes| Buffer::new(bytes, s);
            let mut result = Self {
                cfg,
                width,
                stride,
                moves,
                candidates: c,
                depth: 0,
                failed: false,
                generate,
                hash,
                archive_hash,
                archive_hashes: buffer(cfg.batch as usize * 16)?,
                archive_done: [Event::new()?, Event::new()?],
                archived_depth: None,
                archive_stream,
                route,
                owner,
                states: buffer(
                    ring_capacity
                        .checked_mul(stride)
                        .ok_or("STATE_BYTES_OVERFLOW")?,
                )?,
                prev: buffer(f * 16)?,
                curr: buffer(f * 16)?,
                accepted: buffer(
                    b.checked_mul(cfg.bucket_capacity as usize)
                        .and_then(|v| v.checked_mul(16))
                        .ok_or("HASH_BYTES_OVERFLOW")?,
                )?,
                lengths: buffer(b * 4)?,
                children: buffer(cap * stride)?,
                hashes: buffer(cap * 16)?,
                origins: buffer(cap * 8)?,
                sorted: buffer(cap * 16)?,
                refs: buffer(cap * 8)?,
                routed_count: buffer(4)?,
                directory: buffer(b * 16)?,
                fatal: buffer(4)?,
                jobs_gpu: buffer(slots * 64)?,
                counts: buffer(cfg.job_buckets as usize * 32)?,
                control: buffer(64)?,
                selected: buffer(cap * 4)?,
                ring: buffer(64)?,
                extent: buffer(64)?,
                layer_count: buffer(4)?,
                incoming_dir: vec![Range::default(); b],
                prev_dir: vec![Range::default(); b],
                curr_dir: vec![Range::default(); b],
                descriptors: vec![BucketJob::default(); slots],
                spans: vec![JobSpan::default(); slots],
                front: Vec::with_capacity(2),
                next: Vec::with_capacity(2),
                front_count: 1,
                prev_count: 0,
                plan_bytes,
                alternate: ProducerBuffers {
                    children: buffer(cap * stride)?,
                    hashes: buffer(cap * 16)?,
                    sorted: buffer(cap * 16)?,
                    refs: buffer(cap * 8)?,
                    routed_count: buffer(4)?,
                    directory: buffer(b * 16)?,
                    fatal: buffer(4)?,
                    ready: Event::new()?,
                },
                ready: Event::new()?,
                producer_stream,
                stream,
            };
            if result.requested_device_bytes() != requested {
                return Err("VRAM_LEDGER_MISMATCH".into());
            }
            check(cudaDeviceSynchronize())?;
            check(cudaMemGetInfo(&mut free, &mut total))?;
            if (free as u128) < u128::from(reserve) {
                return Err("VRAM_RESERVE_AFTER_ALLOCATION".into());
            }
            let mut start = vec![0; stride];
            start[..width].copy_from_slice(&g.start);
            result.states.put(&start)?;
            result.origins.put(&(0..u64::from(c)).collect::<Vec<_>>())?;
            result.ring.put(&[Ring {
                tail: 1,
                descriptor_tail: 1,
                capacity: ring_capacity as u64,
                descriptor_capacity: (f * 2 + cap) as u64,
                ..Ring::default()
            }])?;
            result.front.push(Extent {
                count: 1,
                granted_rows: 1,
                ready: 1,
                padding: [0, 0, 0],
                ..Extent::default()
            });
            result.layer_count.put(&[1u32])?;
            check(mgbfs_hash_run(
                result.hash.0,
                result.states.p.cast(),
                result.curr.p.cast(),
                1,
                s,
            ))?;
            check(mgbfs_bucket_directory(
                result.curr.p,
                result.layer_count.p.cast(),
                cfg.layer_capacity,
                cfg.buckets,
                result.directory.p.cast(),
                result.fatal.p.cast(),
                s,
            ))?;
            check(cudaStreamSynchronize(s))?;
            result.directory.read_into(&mut result.curr_dir)?;
            Ok(result)
        }
    }
    pub fn frontier_len(&self) -> u32 {
        self.front_count
    }
    /// Sum of cudaMalloc request sizes, excluding driver/context rounding.
    pub fn requested_device_bytes(&self) -> u64 {
        self.plan_bytes
            + [
                &self.states,
                &self.prev,
                &self.curr,
                &self.accepted,
                &self.lengths,
                &self.children,
                &self.hashes,
                &self.archive_hashes,
                &self.origins,
                &self.sorted,
                &self.refs,
                &self.routed_count,
                &self.directory,
                &self.fatal,
                &self.jobs_gpu,
                &self.counts,
                &self.control,
                &self.selected,
                &self.ring,
                &self.extent,
                &self.layer_count,
                &self.alternate.children,
                &self.alternate.hashes,
                &self.alternate.sorted,
                &self.alternate.refs,
                &self.alternate.routed_count,
                &self.alternate.directory,
                &self.alternate.fatal,
            ]
            .iter()
            .map(|b| b.bytes.max(1) as u64)
            .sum::<u64>()
    }
    pub fn frontier_extents(&self) -> usize {
        self.front.len()
    }
    /// Enqueue bounded D2H runs without a host wait. Disk consumes ready slots;
    /// advance overlaps generation/owner work and retires each parent only after D2H.
    /// V1 archive hashes are recomputed on GPU in state order (not CPU hashed).
    pub fn archive_current(
        &mut self,
        archive: &mut crate::pinned_archive::PinnedArchive,
    ) -> Result<()> {
        if self.failed {
            return Err("NATIVE_FAILED".into());
        }
        let result = self.archive_inner(archive);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn archive_inner(&mut self, archive: &mut crate::pinned_archive::PinnedArchive) -> Result<()> {
        if archive.width != self.width {
            return Err("ARCHIVE_STATE_WIDTH".into());
        }
        if self.archived_depth == Some(self.depth) {
            return Err("ARCHIVE_DEPTH_ALREADY_SUBMITTED".into());
        }
        let s = self.archive_stream.0;
        for (fi, extent) in self.front.iter().enumerate() {
            let mut offset = 0;
            while offset < extent.count {
                let n =
                    u64::from(archive.rows.min(self.cfg.batch)).min(extent.count - offset) as u32;
                let slot = archive.acquire()?;
                let copied = (|| unsafe {
                    let states = self
                        .states
                        .at((extent.begin + offset) as usize * self.stride);
                    check(mgbfs_hash_run(
                        self.archive_hash.0,
                        states.cast(),
                        self.archive_hashes.p.cast(),
                        n,
                        s,
                    ))?;
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
                    check(cudaMemcpyAsync(
                        slot.ptr.cast::<u8>().add(n as usize * self.width).cast(),
                        self.archive_hashes.p,
                        n as usize * 16,
                        2,
                        s,
                    ))?;
                    check(cudaEventRecord(slot.ready, s))
                })();
                // A failed enqueue might not have recorded the slot event.
                // Drain that error path before its pinned storage can be freed.
                if let Err(error) = copied {
                    unsafe {
                        cudaStreamSynchronize(s);
                    }
                    return Err(error);
                }
                archive.submit(slot, u64::from(self.depth), n)?;
                offset += u64::from(n);
            }
            check(unsafe { cudaEventRecord(self.archive_done[fi].0, s) })?;
        }
        archive.layer(u64::from(self.depth), u64::from(self.front_count))?;
        self.archived_depth = Some(self.depth);
        Ok(())
    }
    pub fn snapshot(&self) -> Result<Vec<Vec<u8>>> {
        if self.failed {
            return Err("NATIVE_FAILED".into());
        }
        check(unsafe { cudaStreamSynchronize(self.stream.0) })?;
        let mut result = Vec::with_capacity(self.front_count as usize);
        for e in &self.front {
            let mut bytes = vec![0u8; e.count as usize * self.stride];
            check(unsafe {
                cudaMemcpy(
                    bytes.as_mut_ptr().cast(),
                    self.states.at(e.begin as usize * self.stride),
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
    pub fn advance(&mut self) -> Result<bool> {
        if self.failed {
            return Err("NATIVE_FAILED".into());
        }
        if self.front_count == 0 {
            return Ok(false);
        }
        let r = self.advance_inner();
        if r.is_err() {
            self.failed = true;
        }
        r
    }
    fn swap_producer_banks(&mut self) {
        std::mem::swap(&mut self.children, &mut self.alternate.children);
        std::mem::swap(&mut self.hashes, &mut self.alternate.hashes);
        std::mem::swap(&mut self.sorted, &mut self.alternate.sorted);
        std::mem::swap(&mut self.refs, &mut self.alternate.refs);
        std::mem::swap(&mut self.routed_count, &mut self.alternate.routed_count);
        std::mem::swap(&mut self.directory, &mut self.alternate.directory);
        std::mem::swap(&mut self.fatal, &mut self.alternate.fatal);
        std::mem::swap(&mut self.ready, &mut self.alternate.ready);
    }
    fn produce(&self, begin: u64, rows: u32, alternate: bool) -> Result<()> {
        let (children, hashes, sorted, refs, count, directory, fatal, ready) = if alternate {
            let a = &self.alternate;
            (
                &a.children,
                &a.hashes,
                &a.sorted,
                &a.refs,
                &a.routed_count,
                &a.directory,
                &a.fatal,
                &a.ready,
            )
        } else {
            (
                &self.children,
                &self.hashes,
                &self.sorted,
                &self.refs,
                &self.routed_count,
                &self.directory,
                &self.fatal,
                &self.ready,
            )
        };
        let n = rows * self.moves;
        let s = self.producer_stream.0;
        unsafe {
            check(mgbfs_generate_run(
                self.generate.0,
                self.states.at(begin as usize * self.stride).cast(),
                children.p.cast(),
                rows,
                s,
            ))?;
            check(mgbfs_hash_run(
                self.hash.0,
                children.p.cast(),
                hashes.p.cast(),
                n,
                s,
            ))?;
            check(mgbfs_route_run(
                self.route.0,
                hashes.p,
                self.origins.p.cast(),
                sorted.p,
                refs.p.cast(),
                count.p.cast(),
                n,
                self.cfg.prededup as i32,
                s,
            ))?;
            check(mgbfs_bucket_directory(
                sorted.p,
                count.p.cast(),
                self.candidates,
                self.cfg.buckets,
                directory.p.cast(),
                fatal.p.cast(),
                s,
            ))?;
            check(cudaEventRecord(ready.0, s))
        }
    }
    fn advance_inner(&mut self) -> Result<bool> {
        unsafe {
            let s = self.stream.0;
            check(cudaMemsetAsync(
                self.lengths.p,
                0,
                self.cfg.buckets as usize * 4,
                s,
            ))?;
            check(cudaMemsetAsync(self.layer_count.p, 0, 4, s))?;
            self.next.clear();
            let first = self.front[0];
            self.produce(
                first.begin,
                u64::from(self.cfg.batch).min(first.count) as u32,
                false,
            )?;
            for fi in 0..self.front.len() {
                let parent = self.front[fi];
                let mut live_parent = parent;
                let mut archive_released = false;
                let mut offset = 0;
                while offset < parent.count {
                    let n = u64::from(self.cfg.batch).min(parent.count - offset) as u32;
                    let children = n * self.moves;
                    let next_position = if offset + u64::from(n) < parent.count {
                        Some((
                            parent.begin + offset + u64::from(n),
                            u64::from(self.cfg.batch).min(parent.count - offset - u64::from(n))
                                as u32,
                        ))
                    } else {
                        self.front
                            .get(fi + 1)
                            .map(|e| (e.begin, u64::from(self.cfg.batch).min(e.count) as u32))
                    };
                    // Fill the other bank while this bank is consumed by owner jobs.
                    if let Some((begin, rows)) = next_position {
                        self.produce(begin, rows, true)?;
                    }
                    check(cudaEventSynchronize(self.ready.0))?;
                    if self.fatal.one::<u32>()? != 0 {
                        return Err("DIRECTORY_FATAL".into());
                    }
                    // Generation has copied this parent batch into an independent
                    // candidate slot. Once archive DMA is also done, the prefix
                    // is dead and owner commits may immediately reuse it.
                    if self.archived_depth == Some(self.depth) && !archive_released {
                        check(cudaStreamWaitEvent(s, self.archive_done[fi].0, 0))?;
                        archive_released = true;
                    }
                    self.extent.put(&[live_parent])?;
                    check(mgbfs_state_retire_dense_prefix(
                        self.ring.p.cast(),
                        self.extent.p.cast(),
                        u64::from(n),
                        s,
                    ))?;
                    check(cudaStreamSynchronize(s))?;
                    let ring_state = self.ring.one::<Ring>()?;
                    if ring_state.fatal != 0 {
                        return Err(format!("STATE_RING_RETIRE_FATAL_{}", ring_state.fatal));
                    }
                    live_parent.sequence += u64::from(n);
                    live_parent.begin = live_parent.sequence % ring_state.capacity;
                    live_parent.count -= u64::from(n);
                    live_parent.granted_rows = live_parent.count as u32;
                    if live_parent.count == 0 {
                        live_parent.ready = 0;
                    }
                    self.directory.read_into(&mut self.incoming_dir)?;
                    let (nd, ns) = split(
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
                    self.jobs_gpu.put(&self.descriptors[..nd])?;
                    for sj in 0..ns {
                        let job = self.spans[sj];
                        let p = self.jobs_gpu.at(job.first * 64).cast::<BucketJob>();
                        let hashes = self.sorted.at(job.source_begin as usize * 16);
                        let refs = self.refs.at(job.source_begin as usize * 8).cast::<u64>();
                        let lane = self.descriptors[job.first].lane;
                        check(mgbfs_bind_owner_jobs(
                            p,
                            job.buckets,
                            self.lengths.p.cast(),
                            self.cfg.buckets,
                            s,
                        ))?;
                        check(mgbfs_bounded_owner_compare(
                            self.owner.0,
                            p,
                            job.buckets,
                            job.rows,
                            hashes,
                            self.prev.p,
                            u64::from(self.prev_count),
                            self.curr.p,
                            u64::from(self.front_count),
                            self.accepted.p,
                            self.lengths.p.cast(),
                            self.cfg.buckets,
                            self.cfg.buckets / self.cfg.shards,
                            lane,
                            self.depth,
                            self.counts.p.cast(),
                            self.control.p.cast(),
                            s,
                        ))?;
                        check(mgbfs_state_reserve_layer(
                            self.ring.p.cast(),
                            self.control.p.cast(),
                            self.extent.p.cast(),
                            self.layer_count.p.cast(),
                            self.cfg.layer_capacity,
                            s,
                        ))?;
                        let ep = self.extent.p.cast::<Extent>();
                        check(mgbfs_bounded_owner_commit(
                            self.owner.0,
                            p,
                            job.buckets,
                            hashes,
                            self.accepted.p,
                            self.lengths.p.cast(),
                            self.counts.p.cast(),
                            self.control.p.cast(),
                            std::ptr::addr_of!((*ep).granted_rows),
                            self.selected.p.cast(),
                            s,
                        ))?;
                        check(mgbfs_state_materialize(
                            self.children.p.cast(),
                            children,
                            refs,
                            job.rows,
                            self.selected.p.cast(),
                            self.candidates,
                            self.stride as u32,
                            self.states.p.cast(),
                            self.ring.p.cast(),
                            self.control.p.cast(),
                            ep,
                            s,
                        ))?;
                        check(cudaStreamSynchronize(s))?;
                        let control = self.control.one::<Control>()?;
                        if control.error != 0 {
                            return Err(format!("NATIVE_OWNER_FATAL_{}", control.error));
                        }
                        let mut e = self.extent.one::<Extent>()?;
                        if e.ready != 1 {
                            return Err("STATE_NOT_READY".into());
                        }
                        if e.count != 0 {
                            e.padding[1] = e.descriptor;
                            let adjacent = self.next.last().is_some_and(|last| {
                                last.begin + last.count == e.begin
                                    && last.sequence + last.count == e.sequence
                            });
                            if adjacent {
                                let last = self.next.last_mut().unwrap();
                                last.count += e.count;
                                last.granted_rows = last.count as u32;
                                // One contiguous generation span retires all its
                                // allocation descriptors through the last ticket.
                                last.padding[1] = e.descriptor;
                            } else {
                                if self.next.len() == self.next.capacity() {
                                    return Err("HOST_EXTENT_CAPACITY".into());
                                }
                                self.next.push(e);
                            }
                        }
                    }
                    self.swap_producer_banks();
                    offset += u64::from(n);
                }
            }
            check(mgbfs_compact_hash_layer(
                self.accepted.p,
                self.lengths.p.cast(),
                self.cfg.buckets,
                self.cfg.bucket_capacity,
                self.prev.p,
                self.cfg.layer_capacity,
                self.directory.p.cast(),
                self.routed_count.p.cast(),
                self.fatal.p.cast(),
                s,
            ))?;
            check(cudaStreamSynchronize(s))?;
            if self.fatal.one::<u32>()? != 0 {
                return Err("FINALIZE_FATAL".into());
            }
            let next_count = self.routed_count.one::<u32>()?;
            if self.layer_count.one::<u32>()? != next_count {
                return Err("LAYER_COUNT_MISMATCH".into());
            }
            self.directory.read_into(&mut self.prev_dir)?;
            std::mem::swap(&mut self.prev_dir, &mut self.curr_dir);
            std::mem::swap(&mut self.prev, &mut self.curr);
            std::mem::swap(&mut self.front, &mut self.next);
            self.prev_count = self.front_count;
            self.front_count = next_count;
            self.depth = self.depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;
            Ok(next_count != 0)
        }
    }
}
