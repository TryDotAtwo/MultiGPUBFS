//! Native two-rank NCCL DENSE BFS reference. Torchrun supplies only rank env.
use crate::failure::attempt_all;
use crate::jobs::{split, JobSpan};
use mgbfs_core::{
    hash::GemmHash,
    matrix::{encode_permutation_matrix, MatrixGroup},
    Result,
};
use mgbfs_cuda::{ffi::*, native_owner::*};
use std::ffi::{c_void, CStr};

#[derive(Clone, Copy)]
pub struct DistributedConfig {
    pub rank: u32,
    pub world: u32,
    pub logical_owner_to_rank: [u32; 2],
    pub batch: u32,
    pub layer_capacity: u32,
    pub state_ring_capacity: u32,
    pub buckets: u32,
    pub shards: u32,
    pub job_buckets: u32,
    pub bucket_capacity: u32,
    pub prededup: bool,
    pub generation_variant: u32,
}
fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("CUDA_STATUS_{status}"))
    }
}
struct Buffer {
    ptr: *mut c_void,
    bytes: usize,
    stream: *mut c_void,
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
impl Plan {
    fn new(
        drop: unsafe extern "C" fn(*mut c_void),
        create: impl FnOnce(*mut *mut c_void, *mut i8) -> i32,
    ) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        let mut e = [0i8; 512];
        if create(&mut p, e.as_mut_ptr()) != 0 {
            return Err(unsafe { CStr::from_ptr(e.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        Ok(Self(p, drop))
    }
}
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe { self.1(self.0) }
    }
}
struct Stream(*mut c_void);
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            cudaStreamSynchronize(self.0);
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
struct Comm(*mut c_void);
impl Drop for Comm {
    fn drop(&mut self) {
        unsafe { mgbfs_nccl_destroy(self.0) }
    }
}

pub struct DistributedNativeBfs {
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
    archive_stream: Stream,
    archive_done: [Event; 2],
    archived_depth: Option<u32>,
    comm: Comm,
    generate: Plan,
    hash: Plan,
    archive_hash: Plan,
    route: Plan,
    owner: Plan,
    states: Buffer,
    prev: Buffer,
    curr: Buffer,
    accepted: Buffer,
    lengths: Buffer,
    children: Buffer,
    child_hashes: Buffer,
    archive_hashes: Buffer,
    archive_states: Buffer,
    sorted_hashes: Buffer,
    sorted_refs: Buffer,
    route_count: Buffer,
    packed_states: Buffer,
    owner_counts: Buffer,
    recv_states: Buffer,
    recv_hashes: Buffer,
    recv_count: Buffer,
    identity_refs: Buffer,
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
    collective_send: Buffer,
    collective_recv: Buffer,
}
impl DistributedNativeBfs {
    pub fn new(
        graph: &MatrixGroup,
        seed: [u8; 16],
        id: [u8; 128],
        mut cfg: DistributedConfig,
    ) -> Result<Self> {
        graph.validate()?;
        if cfg.world != 2
            || cfg.rank >= 2
            || cfg.logical_owner_to_rank[0] == cfg.logical_owner_to_rank[1]
            || cfg.logical_owner_to_rank.iter().any(|&x| x >= 2)
            || cfg.batch == 0
            || cfg.layer_capacity == 0
            || cfg.state_ring_capacity == 0
            || !cfg.buckets.is_power_of_two()
            || !cfg.shards.is_power_of_two()
            || cfg.shards < 2
            || cfg.shards > cfg.buckets
            || cfg.job_buckets == 0
            || cfg.job_buckets > cfg.buckets / cfg.shards
            || cfg.bucket_capacity == 0
        {
            return Err("DISTRIBUTED_CONFIG".into());
        }
        // Public config retains global prefix geometry. Persistent storage and
        // all owner jobs use only this owner's contiguous half of that space.
        cfg.buckets /= 2;
        cfg.shards /= 2;
        check(unsafe { cudaSetDevice(cfg.rank as i32) })?;
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
            encode_permutation_matrix(&graph.start, graph.rows)?
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
        let mut raw = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw, 1) })?;
        let stream = Stream(raw);
        let mut raw_archive = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw_archive, 1) })?;
        let archive_stream = Stream(raw_archive);
        let mut comm = std::ptr::null_mut();
        let mut error = [0i8; 512];
        if unsafe {
            mgbfs_nccl_create(
                cfg.rank,
                cfg.world,
                cfg.rank,
                id.as_ptr().cast(),
                &mut comm,
                error.as_mut_ptr(),
                512,
            )
        } != 0
        {
            return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        let comm = Comm(comm);
        let contract = GemmHash::from_seed(width, seed)?;
        let limbs = contract.limbs();
        let matrices: Vec<u8> = graph.generators.iter().flatten().copied().collect();
        let weights = vec![1u32; moves as usize];
        let generate = Plan::new(mgbfs_generate_destroy, |out, e| unsafe {
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
        })?;
        let hash = Plan::new(mgbfs_hash_destroy, |out, e| unsafe {
            mgbfs_hash_create(
                width as u32,
                candidates,
                limbs.as_ptr(),
                contract.offsets.as_ptr(),
                out,
                e,
                512,
            )
        })?;
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
        let owner = Plan::new(mgbfs_bounded_owner_destroy, |out, _| unsafe {
            mgbfs_bounded_owner_create(candidates, cfg.job_buckets, cfg.bucket_capacity, out)
        })?;
        let b = |n| Buffer::new(n, raw);
        let state_bytes = cfg.state_ring_capacity as usize * stride;
        let states = b(state_bytes)?;
        let prev = b(cfg.layer_capacity as usize * 16)?;
        let curr = b(cfg.layer_capacity as usize * 16)?;
        let start_hash = contract.hash(&start_state)?;
        let start_owner = (start_hash.0[3] >> 31) as usize;
        let start_rank = cfg.logical_owner_to_rank[start_owner];
        let current_count = (start_rank == cfg.rank) as u32;
        if current_count == 1 {
            let mut start = vec![0u8; stride];
            start[..width].copy_from_slice(&start_state);
            states.put(&start)?;
            curr.put(&[start_hash.to_le_bytes()])?;
        }
        let identity_refs = b(candidates as usize * 8)?;
        identity_refs.put(&(0..u64::from(candidates)).collect::<Vec<_>>())?;
        let archive_done = [Event::new()?, Event::new()?];
        check(unsafe { cudaEventRecord(archive_done[0].0, raw) })?;
        check(unsafe { cudaEventRecord(archive_done[1].0, raw) })?;
        check(unsafe { cudaStreamSynchronize(raw) })?;
        let buckets = cfg.buckets as usize;
        let slots = buckets + 1;
        let directory = b(buckets * std::mem::size_of::<Range>())?;
        let fatal = b(4)?;
        let route_count = b(4)?;
        route_count.put(&[current_count])?;
        check(unsafe {
            mgbfs_owner_bucket_directory(
                curr.ptr,
                route_count.ptr.cast(),
                cfg.layer_capacity,
                cfg.buckets,
                u32::from(cfg.logical_owner_to_rank[1] == cfg.rank),
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
        let ring = b(std::mem::size_of::<Ring>())?;
        ring.put(&[Ring {
            tail: u64::from(current_count),
            descriptor_tail: u64::from(current_count),
            capacity: u64::from(cfg.state_ring_capacity),
            descriptor_capacity: u64::from(cfg.state_ring_capacity),
            ..Ring::default()
        }])?;
        let result = Self {
            cfg,
            width,
            stride,
            permutation_n,
            moves,
            candidates,
            depth: 0,
            current_count,
            prev_count: 0,
            failed: false,
            stream,
            archive_stream,
            archive_done,
            archived_depth: None,
            comm,
            generate,
            hash,
            archive_hash,
            route,
            owner,
            states,
            prev,
            curr,
            accepted: b(buckets
                .checked_mul(cfg.bucket_capacity as usize)
                .and_then(|n| n.checked_mul(16))
                .ok_or("ACCEPTED_BYTES_OVERFLOW")?)?,
            lengths: b(buckets * 4)?,
            children: b(candidates as usize * stride)?,
            child_hashes: b(candidates as usize * 16)?,
            archive_hashes: b(cfg.batch as usize * 16)?,
            archive_states: b(cfg.batch as usize * permutation_n.unwrap_or(1) as usize)?,
            sorted_hashes: b(candidates as usize * 16)?,
            sorted_refs: b(candidates as usize * 8)?,
            route_count,
            packed_states: b(candidates as usize * stride)?,
            owner_counts: b(8)?,
            recv_states: b(candidates as usize * stride)?,
            recv_hashes: b(candidates as usize * 16)?,
            recv_count: b(4)?,
            identity_refs,
            directory,
            fatal,
            jobs_gpu: b(slots * std::mem::size_of::<BucketJob>())?,
            counts: b(cfg.job_buckets as usize * std::mem::size_of::<Counts>())?,
            control: b(std::mem::size_of::<Control>())?,
            selected: b(candidates as usize * 4)?,
            ring,
            extent: b(std::mem::size_of::<Extent>())?,
            layer_count: b(4)?,
            incoming_dir: vec![Range::default(); buckets],
            prev_dir: vec![Range::default(); buckets],
            curr_dir,
            descriptors: vec![BucketJob::default(); slots],
            spans: vec![JobSpan::default(); slots],
            front,
            next: Vec::with_capacity(2),
            collective_send: b(4)?,
            collective_recv: b(8)?,
        };
        result.all_max(0)?;
        Ok(result)
    }
    pub fn depth(&self) -> u32 {
        self.depth
    }
    pub fn frontier_len(&self) -> u32 {
        self.current_count
    }
    fn all_max(&self, value: u32) -> Result<u32> {
        self.collective_send.put(&[value])?;
        check(unsafe {
            mgbfs_nccl_all_reduce_max_u32(
                self.comm.0,
                self.collective_send.ptr.cast(),
                self.collective_recv.ptr.cast(),
                self.stream.0,
            )
        })?;
        check(unsafe { cudaStreamSynchronize(self.stream.0) })?;
        self.collective_recv.one()
    }
    fn commit_owner_batch(
        &mut self,
        source_states: *const u8,
        source_hashes: *const c_void,
        rows: u32,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let s = self.stream.0;
        self.route_count.put(&[rows])?;
        unsafe {
            check(mgbfs_owner_bucket_directory(
                source_hashes,
                self.route_count.ptr.cast(),
                self.candidates,
                self.cfg.buckets,
                u32::from(self.cfg.logical_owner_to_rank[1] == self.cfg.rank),
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
                    self.lengths.ptr.cast(),
                    self.cfg.buckets,
                    s,
                ))?;
                check(mgbfs_bounded_owner_compare(
                    self.owner.0,
                    jobs,
                    span.buckets,
                    span.rows,
                    hashes.cast(),
                    self.prev.ptr,
                    u64::from(self.prev_count),
                    self.curr.ptr,
                    u64::from(self.current_count),
                    self.accepted.ptr,
                    self.lengths.ptr.cast(),
                    self.cfg.buckets,
                    self.cfg.buckets / self.cfg.shards,
                    lane,
                    self.depth,
                    self.counts.ptr.cast(),
                    self.control.ptr.cast(),
                    s,
                ))?;
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
                    self.owner.0,
                    jobs,
                    span.buckets,
                    hashes.cast(),
                    self.accepted.ptr,
                    self.lengths.ptr.cast(),
                    self.counts.ptr.cast(),
                    self.control.ptr.cast(),
                    std::ptr::addr_of!((*extent).granted_rows),
                    self.selected.ptr.cast(),
                    s,
                ))?;
                check(mgbfs_state_materialize(
                    source_states,
                    rows,
                    refs,
                    span.rows,
                    self.selected.ptr.cast(),
                    self.candidates,
                    self.stride as u32,
                    self.states.ptr.cast(),
                    self.ring.ptr.cast(),
                    self.control.ptr.cast(),
                    extent,
                    s,
                ))?;
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
            let mut extent = self.extent.one::<Extent>()?;
            if extent.ready != 1 {
                return Err("STATE_NOT_READY".into());
            }
            if extent.count != 0 {
                extent.padding[1] = extent.descriptor;
                if self.next.last().is_some_and(|last| {
                    last.begin + last.count == extent.begin
                        && last.sequence + last.count == extent.sequence
                }) {
                    let last = self.next.last_mut().unwrap();
                    last.count += extent.count;
                    last.granted_rows = last.count as u32;
                    last.padding[1] = extent.descriptor;
                } else {
                    if self.next.len() == self.next.capacity() {
                        return Err("HOST_EXTENT_CAPACITY".into());
                    }
                    self.next.push(extent);
                }
            }
        }
        Ok(())
    }
    pub fn advance(&mut self) -> Result<bool> {
        if self.failed {
            return Err("DISTRIBUTED_FAILED".into());
        }
        let result = self.advance_inner();
        if result.is_err() {
            self.failed = true
        }
        result
    }
    fn advance_inner(&mut self) -> Result<bool> {
        let s = self.stream.0;
        unsafe {
            check(cudaMemsetAsync(
                self.lengths.ptr,
                0,
                self.cfg.buckets as usize * 4,
                s,
            ))?;
            check(cudaMemsetAsync(self.layer_count.ptr, 0, 4, s))?;
        }
        self.next.clear();
        let local_owner = self
            .cfg
            .logical_owner_to_rank
            .iter()
            .position(|&rank| rank == self.cfg.rank)
            .ok_or("OWNER_MAP")?;
        let remote_owner = local_owner ^ 1;
        let mut extent_index = 0usize;
        let mut extent_offset = 0u64;
        let mut archive_released = [false; 2];
        loop {
            let parent = self.front.get(extent_index).copied();
            let parents = parent
                .map(|extent| u64::from(self.cfg.batch).min(extent.count - extent_offset) as u32)
                .unwrap_or(0);
            let candidate_count = parents * self.moves;
            if let Some(extent) = parent {
                unsafe {
                    check(mgbfs_generate_run(
                        self.generate.0,
                        self.states
                            .at((extent.begin + extent_offset) as usize * self.stride)
                            .cast(),
                        self.children.ptr.cast(),
                        parents,
                        s,
                    ))?;
                    check(mgbfs_hash_run(
                        self.hash.0,
                        self.children.ptr.cast(),
                        self.child_hashes.ptr.cast(),
                        candidate_count,
                        s,
                    ))?;
                }
            }
            unsafe {
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
                check(cudaStreamSynchronize(s))?;
            }
            let routed = self.route_count.one::<u32>()?;
            check(unsafe {
                mgbfs_exchange_pack(
                    self.stride as u32,
                    self.candidates,
                    self.children.ptr.cast(),
                    candidate_count,
                    self.sorted_hashes.ptr,
                    self.sorted_refs.ptr.cast(),
                    routed,
                    self.packed_states.ptr.cast(),
                    self.owner_counts.ptr.cast(),
                    s,
                )
            })?;
            check(unsafe { cudaStreamSynchronize(s) })?;
            let mut owner_counts = [0u32; 2];
            self.owner_counts.read(&mut owner_counts)?;
            if owner_counts[0] == u32::MAX {
                return Err("EXCHANGE_SOURCE_REF".into());
            }
            let local_offset = if local_owner == 0 { 0 } else { owner_counts[0] };
            let remote_offset = if remote_owner == 0 {
                0
            } else {
                owner_counts[0]
            };
            self.collective_send.put(&[owner_counts[remote_owner]])?;
            check(unsafe {
                mgbfs_nccl_send_recv(
                    self.comm.0,
                    self.collective_send.ptr,
                    4,
                    self.cfg.rank ^ 1,
                    self.recv_count.ptr,
                    4,
                    s,
                )
            })?;
            check(unsafe { cudaStreamSynchronize(s) })?;
            let received = self.recv_count.one::<u32>()?;
            if received > self.candidates {
                return Err("EXCHANGE_CAPACITY".into());
            }
            check(unsafe {
                mgbfs_nccl_send_recv(
                    self.comm.0,
                    self.sorted_hashes.at(remote_offset as usize * 16),
                    u64::from(owner_counts[remote_owner]) * 16,
                    self.cfg.rank ^ 1,
                    self.recv_hashes.ptr,
                    u64::from(received) * 16,
                    s,
                )
            })?;
            check(unsafe {
                mgbfs_nccl_send_recv(
                    self.comm.0,
                    self.packed_states.at(remote_offset as usize * self.stride),
                    u64::from(owner_counts[remote_owner]) * self.stride as u64,
                    self.cfg.rank ^ 1,
                    self.recv_states.ptr,
                    u64::from(received) * self.stride as u64,
                    s,
                )
            })?;
            if let Some(parent_extent) = parent {
                if self.archived_depth == Some(self.depth) && !archive_released[extent_index] {
                    check(unsafe { cudaStreamWaitEvent(s, self.archive_done[extent_index].0, 0) })?;
                    archive_released[extent_index] = true;
                }
                let mut live = parent_extent;
                live.sequence += extent_offset;
                live.begin = live.sequence % u64::from(self.cfg.state_ring_capacity);
                live.count -= extent_offset;
                live.granted_rows = live.count as u32;
                self.extent.put(&[live])?;
                check(unsafe {
                    mgbfs_state_retire_dense_prefix(
                        self.ring.ptr.cast(),
                        self.extent.ptr.cast(),
                        u64::from(parents),
                        s,
                    )
                })?;
                check(unsafe { cudaStreamSynchronize(s) })?;
                let ring = self.ring.one::<Ring>()?;
                if ring.fatal != 0 {
                    return Err(format!("STATE_RING_RETIRE_FATAL_{}", ring.fatal));
                }
            }
            check(unsafe { cudaStreamSynchronize(s) })?;
            let local_states = unsafe {
                self.packed_states
                    .at(local_offset as usize * self.stride)
                    .cast()
            };
            let local_hashes = unsafe { self.sorted_hashes.at(local_offset as usize * 16) };
            let batch_error = attempt_all(
                [
                    (local_states, local_hashes, owner_counts[local_owner]),
                    (self.recv_states.ptr.cast(), self.recv_hashes.ptr, received),
                ],
                |(states, hashes, rows)| self.commit_owner_batch(states, hashes, rows),
            )
            .err();
            if self.all_max(u32::from(batch_error.is_some()))? != 0 {
                return Err(batch_error.unwrap_or_else(|| "REMOTE_OWNER_BATCH_FATAL".into()));
            }
            if let Some(extent) = parent {
                extent_offset += u64::from(parents);
                if extent_offset == extent.count {
                    extent_index += 1;
                    extent_offset = 0;
                }
            }
            let more = (extent_index < self.front.len()) as u32;
            if self.all_max(more)? == 0 {
                break;
            }
        }
        unsafe {
            check(mgbfs_compact_hash_layer(
                self.accepted.ptr,
                self.lengths.ptr.cast(),
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
        if self.fatal.one::<u32>()? != 0 {
            return Err("FINALIZE_FATAL".into());
        }
        let count = self.route_count.one::<u32>()?;
        if self.layer_count.one::<u32>()? != count {
            return Err("LAYER_COUNT_MISMATCH".into());
        }
        self.directory.read(&mut self.prev_dir)?;
        std::mem::swap(&mut self.prev_dir, &mut self.curr_dir);
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
        let s = self.archive_stream.0;
        for (extent_index, extent) in self.front.iter().enumerate() {
            let mut offset = 0u64;
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
                if let Err(e) = copied {
                    unsafe {
                        cudaStreamSynchronize(s);
                    }
                    self.failed = true;
                    return Err(e);
                }
                archive.submit(slot, u64::from(self.depth), n)?;
                offset += u64::from(n);
            }
            check(unsafe { cudaEventRecord(self.archive_done[extent_index].0, s) })?;
        }
        archive.layer(u64::from(self.depth), u64::from(self.current_count))?;
        self.archived_depth = Some(self.depth);
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
