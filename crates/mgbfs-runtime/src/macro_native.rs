//! Single-rank CUDA reference for exact weighted macro lookahead.
//! All device arenas are allocated before depth zero; CPU performs orchestration only.
use mgbfs_core::{
    hash::GemmHash, macro_generators::MacroGeneratorSet, matrix::MatrixGroup, Result,
};
use mgbfs_cuda::{ffi::*, native_owner::*};
use std::ffi::{c_void, CStr};

#[derive(Clone, Copy)]
pub struct MacroNativeConfig {
    pub macro_depth: u32,
    pub batch: u32,
    pub layer_capacity: u32,
    pub future_capacity_per_depth: u32,
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
        let value = Self { ptr, bytes, stream };
        check(unsafe { cudaMemsetAsync(ptr, 0, bytes.max(1), stream) })?;
        Ok(value)
    }
    fn put<T: Copy>(&self, values: &[T]) -> Result<()> {
        if std::mem::size_of_val(values) > self.bytes {
            return Err("UPLOAD_CAPACITY".into());
        }
        check(unsafe {
            cudaMemcpyAsync(
                self.ptr,
                values.as_ptr().cast(),
                std::mem::size_of_val(values),
                1,
                self.stream,
            )
        })
    }
    fn one<T: Copy + Default>(&self) -> Result<T> {
        let mut out = [T::default()];
        check(unsafe {
            cudaMemcpy(
                out.as_mut_ptr().cast(),
                self.ptr,
                std::mem::size_of::<T>(),
                2,
            )
        })?;
        Ok(out[0])
    }
    unsafe fn at(&self, offset: usize) -> *mut c_void {
        self.ptr.cast::<u8>().add(offset).cast()
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
        destroy: unsafe extern "C" fn(*mut c_void),
        create: impl FnOnce(*mut *mut c_void, *mut i8) -> i32,
    ) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let mut error = [0i8; 512];
        if create(&mut ptr, error.as_mut_ptr()) != 0 {
            return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        Ok(Self(ptr, destroy))
    }
}
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            self.1(self.0);
        }
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
        let mut event = std::ptr::null_mut();
        check(unsafe { cudaEventCreateWithFlags(&mut event, 2) })?;
        Ok(Self(event))
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.0); }
    }
}
struct FutureSlot {
    depth: Option<u32>,
    states: Buffer,
    hashes: Buffer,
    state: Buffer,
    count: u32,
}

pub struct MacroNativeBfs {
    cfg: MacroNativeConfig,
    width: usize,
    stride: usize,
    depth: u32,
    current_count: u32,
    failed: bool,
    weight_runs: Vec<(u32, u32, u32)>,
    effective_depth: u32,
    stream: Stream,
    archive_stream: Stream,
    archive_done: [Event; 2],
    archive_hash: Plan,
    archive_hashes: Buffer,
    archived_depth: Option<u32>,
    current_bank: usize,
    generate: Plan,
    hash: Plan,
    route: Plan,
    materialize: Plan,
    settle: Plan,
    current_states: Buffer,
    next_states: Buffer,
    next_hashes: Buffer,
    next_state: Buffer,
    children: Buffer,
    child_hashes: Buffer,
    identity_refs: Buffer,
    sorted_hashes: Buffer,
    sorted_refs: Buffer,
    route_count: Buffer,
    survivor_hashes: Buffer,
    survivor_refs: Buffer,
    survivor_count: Buffer,
    settle_state: Buffer,
    history: Buffer,
    history_counts_gpu: Buffer,
    history_counts: Vec<u32>,
    future: Vec<FutureSlot>,
}

impl MacroNativeBfs {
    pub fn new(graph: &MatrixGroup, seed: [u8; 16], cfg: MacroNativeConfig) -> Result<Self> {
        graph.validate()?;
        let macros = MacroGeneratorSet::compile(graph, cfg.macro_depth)?;
        let moves = u32::try_from(macros.transitions.len()).map_err(|_| "MACRO_MOVES")?;
        let candidates = cfg
            .batch
            .checked_mul(moves)
            .ok_or("MACRO_CANDIDATE_OVERFLOW")?;
        if cfg.batch == 0
            || cfg.layer_capacity == 0
            || cfg.future_capacity_per_depth == 0
            || candidates == 0
            || candidates > i32::MAX as u32
        {
            return Err("MACRO_NATIVE_CONFIG".into());
        }
        let width = graph.start.len();
        let stride = (width + 15) & !15;
        let effective = macros.effective_depth;
        let history_layers = effective.checked_mul(2).ok_or("MACRO_HISTORY_OVERFLOW")?;
        let max_records = candidates
            .max(cfg.future_capacity_per_depth)
            .max(cfg.layer_capacity);
        if max_records > i32::MAX as u32 {
            return Err("MACRO_NATIVE_CAPACITY".into());
        }
        let mut raw_stream = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw_stream, 1) })?;
        let stream = Stream(raw_stream);
        let mut raw_archive_stream = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut raw_archive_stream, 1) })?;
        let archive_stream = Stream(raw_archive_stream);
        let matrices: Vec<u8> = macros
            .transitions
            .iter()
            .flat_map(|item| item.matrix.iter().copied())
            .collect();
        let weights: Vec<u32> = macros.transitions.iter().map(|item| item.weight).collect();
        let mut weight_runs = Vec::new();
        let mut begin = 0usize;
        while begin < weights.len() {
            let weight = weights[begin];
            let mut end = begin + 1;
            while end < weights.len() && weights[end] == weight {
                end += 1;
            }
            weight_runs.push((weight, begin as u32, (end - begin) as u32));
            begin = end;
        }
        let hash_contract = GemmHash::from_seed(width, seed)?;
        let limbs = hash_contract.limbs();
        let generate = Plan::new(mgbfs_generate_destroy, |out, error| unsafe {
            mgbfs_generate_create_macro_variant(
                graph.rows as u32,
                moves,
                graph.modulus as u32,
                cfg.batch,
                matrices.as_ptr(),
                weights.as_ptr(),
                cfg.generation_variant,
                out,
                error,
                512,
            )
        })?;
        let hash = Plan::new(mgbfs_hash_destroy, |out, error| unsafe {
            mgbfs_hash_create(
                width as u32,
                candidates,
                limbs.as_ptr(),
                hash_contract.offsets.as_ptr(),
                out,
                error,
                512,
            )
        })?;
        let archive_hash = Plan::new(mgbfs_hash_destroy, |out, error| unsafe {
            mgbfs_hash_create(
                width as u32,
                cfg.batch,
                limbs.as_ptr(),
                hash_contract.offsets.as_ptr(),
                out,
                error,
                512,
            )
        })?;
        let route = Plan::new(mgbfs_route_destroy, |out, error| unsafe {
            mgbfs_route_create(max_records, out, error, 512)
        })?;
        let materialize = Plan::new(mgbfs_materialize_destroy, |out, error| unsafe {
            mgbfs_materialize_create(
                stride as u32,
                max_records,
                cfg.future_capacity_per_depth.max(cfg.layer_capacity),
                out,
                error,
                512,
            )
        })?;
        let settle = Plan::new(mgbfs_macro_settle_destroy, |out, error| unsafe {
            mgbfs_macro_settle_create(
                cfg.future_capacity_per_depth,
                history_layers,
                cfg.layer_capacity,
                out,
                error,
                512,
            )
        })?;
        let buffer = |bytes| Buffer::new(bytes, raw_stream);
        let state_bytes = cfg.layer_capacity as usize * stride;
        let future_state_bytes = cfg.future_capacity_per_depth as usize * stride;
        let current_states = buffer(state_bytes)?;
        let next_states = buffer(state_bytes)?;
        let next_hashes = buffer(cfg.layer_capacity as usize * 16)?;
        let next_state = buffer(std::mem::size_of::<FrontierState>())?;
        let children = buffer(candidates as usize * stride)?;
        let child_hashes = buffer(candidates as usize * 16)?;
        let archive_hashes = Buffer::new(cfg.batch as usize * 16, raw_archive_stream)?;
        let identity_refs = buffer(max_records as usize * 8)?;
        let sorted_hashes = buffer(max_records as usize * 16)?;
        let sorted_refs = buffer(max_records as usize * 8)?;
        let route_count = buffer(4)?;
        let survivor_hashes = buffer(cfg.future_capacity_per_depth as usize * 16)?;
        let survivor_refs = buffer(cfg.future_capacity_per_depth as usize * 8)?;
        let survivor_count = buffer(4)?;
        let settle_state = buffer(std::mem::size_of::<MacroSettleState>())?;
        let history = buffer(history_layers as usize * cfg.layer_capacity as usize * 16)?;
        let history_counts_gpu = buffer(history_layers as usize * 4)?;
        let mut future = Vec::with_capacity(effective as usize);
        for _ in 0..effective {
            future.push(FutureSlot {
                depth: None,
                states: buffer(future_state_bytes)?,
                hashes: buffer(cfg.future_capacity_per_depth as usize * 16)?,
                state: buffer(std::mem::size_of::<FrontierState>())?,
                count: 0,
            });
        }
        let mut start = vec![0u8; stride];
        start[..width].copy_from_slice(&graph.start);
        current_states.put(&start)?;
        identity_refs.put(&(0..u64::from(max_records)).collect::<Vec<_>>())?;
        let mut history_counts = vec![0u32; history_layers as usize];
        history_counts[0] = 1;
        history_counts_gpu.put(&history_counts)?;
        check(unsafe {
            mgbfs_hash_run(
                hash.0,
                current_states.ptr.cast(),
                history.ptr.cast(),
                1,
                raw_stream,
            )
        })?;
        let archive_done = [Event::new()?, Event::new()?];
        check(unsafe { cudaEventRecord(archive_done[0].0, raw_stream) })?;
        check(unsafe { cudaEventRecord(archive_done[1].0, raw_stream) })?;
        check(unsafe { cudaStreamSynchronize(raw_stream) })?;
        Ok(Self {
            cfg,
            width,
            stride,
            depth: 0,
            current_count: 1,
            failed: false,
            weight_runs,
            effective_depth: effective,
            stream,
            archive_stream,
            archive_done,
            archive_hash,
            archive_hashes,
            archived_depth: None,
            current_bank: 0,
            generate,
            hash,
            route,
            materialize,
            settle,
            current_states,
            next_states,
            next_hashes,
            next_state,
            children,
            child_hashes,
            identity_refs,
            sorted_hashes,
            sorted_refs,
            route_count,
            survivor_hashes,
            survivor_refs,
            survivor_count,
            settle_state,
            history,
            history_counts_gpu,
            history_counts,
            future,
        })
    }
    pub fn frontier_len(&self) -> u32 {
        self.current_count
    }
    pub fn snapshot(&self) -> Result<Vec<Vec<u8>>> {
        if self.failed {
            return Err("MACRO_NATIVE_FAILED".into());
        }
        check(unsafe { cudaStreamSynchronize(self.stream.0) })?;
        let mut packed = vec![0u8; self.current_count as usize * self.stride];
        check(unsafe {
            cudaMemcpy(
                packed.as_mut_ptr().cast(),
                self.current_states.ptr,
                packed.len(),
                2,
            )
        })?;
        Ok(packed
            .chunks_exact(self.stride)
            .map(|row| row[..self.width].to_vec())
            .collect())
    }
    /// Enqueues dense state/hash records to the pinned archive lane. The next
    /// reuse of this state bank waits on the D2H event, not on disk IO.
    pub fn archive_current(
        &mut self,
        archive: &mut crate::pinned_archive::PinnedArchive,
    ) -> Result<()> {
        if self.failed {
            return Err("MACRO_NATIVE_FAILED".into());
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
        let stream = self.archive_stream.0;
        let mut offset = 0u32;
        while offset < self.current_count {
            let count = archive.rows.min(self.cfg.batch).min(self.current_count - offset);
            let slot = archive.acquire()?;
            let copied = (|| unsafe {
                let states = self.current_states.at(offset as usize * self.stride);
                check(mgbfs_hash_run(
                    self.archive_hash.0,
                    states.cast(),
                    self.archive_hashes.ptr.cast(),
                    count,
                    stream,
                ))?;
                check(cudaMemcpy2DAsync(
                    slot.ptr,
                    self.width,
                    states,
                    self.stride,
                    self.width,
                    count as usize,
                    2,
                    stream,
                ))?;
                check(cudaMemcpyAsync(
                    slot.ptr.cast::<u8>().add(count as usize * self.width).cast(),
                    self.archive_hashes.ptr,
                    count as usize * 16,
                    2,
                    stream,
                ))?;
                check(cudaEventRecord(slot.ready, stream))
            })();
            if let Err(error) = copied {
                unsafe { cudaStreamSynchronize(stream); }
                return Err(error);
            }
            archive.submit(slot, u64::from(self.depth), count)?;
            offset += count;
        }
        check(unsafe { cudaEventRecord(self.archive_done[self.current_bank].0, stream) })?;
        archive.layer(u64::from(self.depth), u64::from(self.current_count))?;
        self.archived_depth = Some(self.depth);
        Ok(())
    }
    fn produce(&mut self) -> Result<()> {
        let mut parent_offset = 0u32;
        while parent_offset < self.current_count {
            let parents = self.cfg.batch.min(self.current_count - parent_offset);
            let all = parents
                .checked_mul(self.weight_runs.iter().map(|run| run.2).sum::<u32>())
                .ok_or("COUNT_OVERFLOW")?;
            unsafe {
                check(mgbfs_generate_run(
                    self.generate.0,
                    self.current_states
                        .at(parent_offset as usize * self.stride)
                        .cast(),
                    self.children.ptr.cast(),
                    parents,
                    self.stream.0,
                ))?;
                check(mgbfs_hash_run(
                    self.hash.0,
                    self.children.ptr.cast(),
                    self.child_hashes.ptr.cast(),
                    all,
                    self.stream.0,
                ))?;
            }
            for &(weight, move_begin, move_count) in &self.weight_runs {
                let count = parents.checked_mul(move_count).ok_or("COUNT_OVERFLOW")?;
                let row_begin = move_begin as usize * parents as usize;
                let target = self.depth.checked_add(weight).ok_or("DEPTH_OVERFLOW")?;
                let slot_index = (target % self.effective_depth) as usize;
                let slot = &mut self.future[slot_index];
                if slot.depth.is_some() && slot.depth != Some(target) {
                    return Err("FUTURE_SLOT_ALIAS".into());
                }
                slot.depth = Some(target);
                unsafe {
                    check(mgbfs_route_run(
                        self.route.0,
                        self.child_hashes.at(row_begin * 16),
                        self.identity_refs.ptr.cast(),
                        self.sorted_hashes.ptr,
                        self.sorted_refs.ptr.cast(),
                        self.route_count.ptr.cast(),
                        count,
                        self.cfg.prededup as i32,
                        self.stream.0,
                    ))?;
                    check(mgbfs_materialize_run(
                        self.materialize.0,
                        self.children.at(row_begin * self.stride).cast(),
                        count,
                        self.sorted_hashes.ptr,
                        self.sorted_refs.ptr.cast(),
                        self.route_count.ptr.cast(),
                        slot.states.ptr.cast(),
                        slot.hashes.ptr,
                        &mut *(slot.state.ptr.cast::<FrontierState>()),
                        self.stream.0,
                    ))?;
                }
            }
            parent_offset += parents;
        }
        check(unsafe { cudaStreamSynchronize(self.stream.0) })?;
        for slot in &mut self.future {
            if slot.depth.is_some() {
                let state = slot.state.one::<FrontierState>()?;
                if state.fatal != 0 {
                    return Err(format!("FUTURE_CAPACITY_{}", state.fatal));
                }
                slot.count = state.count;
            }
        }
        Ok(())
    }
    fn settle_depth(&mut self, target: u32) -> Result<u32> {
        check(unsafe {
            cudaStreamWaitEvent(self.stream.0, self.archive_done[self.current_bank ^ 1].0, 0)
        })?;
        self.next_state.put(&[FrontierState::default()])?;
        let slot_index = (target % self.effective_depth) as usize;
        let (source_states, source_hashes, source_count) =
            if self.future[slot_index].depth == Some(target) {
                (
                    &self.future[slot_index].states,
                    &self.future[slot_index].hashes,
                    self.future[slot_index].count,
                )
            } else {
                (
                    &self.future[slot_index].states,
                    &self.future[slot_index].hashes,
                    0,
                )
            };
        self.route_count.put(&[source_count])?;
        unsafe {
            check(mgbfs_route_run(
                self.route.0,
                source_hashes.ptr,
                self.identity_refs.ptr.cast(),
                self.sorted_hashes.ptr,
                self.sorted_refs.ptr.cast(),
                self.route_count.ptr.cast(),
                source_count,
                0,
                self.stream.0,
            ))?;
            check(mgbfs_macro_settle_run(
                self.settle.0,
                self.sorted_hashes.ptr,
                self.sorted_refs.ptr.cast(),
                self.route_count.ptr.cast(),
                self.history.ptr,
                self.history_counts_gpu.ptr.cast(),
                self.survivor_hashes.ptr,
                self.survivor_refs.ptr.cast(),
                self.survivor_count.ptr.cast(),
                self.settle_state.ptr.cast(),
                u64::from(target) + 1,
                self.stream.0,
            ))?;
            check(mgbfs_materialize_run(
                self.materialize.0,
                source_states.ptr.cast(),
                source_count,
                self.survivor_hashes.ptr,
                self.survivor_refs.ptr.cast(),
                self.survivor_count.ptr.cast(),
                self.next_states.ptr.cast(),
                self.next_hashes.ptr,
                self.next_state.ptr.cast(),
                self.stream.0,
            ))?;
            check(cudaStreamSynchronize(self.stream.0))?;
        }
        let settled = self.settle_state.one::<MacroSettleState>()?;
        let next = self.next_state.one::<FrontierState>()?;
        if settled.fatal != 0 || next.fatal != 0 || settled.count != next.count {
            return Err(format!(
                "MACRO_SETTLE_FATAL_{}_{}",
                settled.fatal, next.fatal
            ));
        }
        let history_slot = (target % (self.effective_depth * 2)) as usize;
        unsafe {
            check(cudaMemcpyAsync(
                self.history
                    .at(history_slot * self.cfg.layer_capacity as usize * 16),
                self.survivor_hashes.ptr,
                next.count as usize * 16,
                3,
                self.stream.0,
            ))?;
        }
        self.history_counts[history_slot] = next.count;
        self.history_counts_gpu.put(&self.history_counts)?;
        if self.future[slot_index].depth == Some(target) {
            self.future[slot_index]
                .state
                .put(&[FrontierState::default()])?;
            self.future[slot_index].depth = None;
            self.future[slot_index].count = 0;
        }
        check(unsafe { cudaStreamSynchronize(self.stream.0) })?;
        Ok(next.count)
    }
    pub fn advance(&mut self) -> Result<bool> {
        if self.failed {
            return Err("MACRO_NATIVE_FAILED".into());
        }
        let result = self.advance_inner();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn advance_inner(&mut self) -> Result<bool> {
        if self.current_count == 0 {
            return Ok(false);
        }
        self.produce()?;
        let target = self.depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;
        let count = self.settle_depth(target)?;
        self.depth = target;
        if count > 0 {
            std::mem::swap(&mut self.current_states, &mut self.next_states);
            self.current_bank ^= 1;
            self.current_count = count;
            return Ok(true);
        }
        self.current_count = 0;
        while self
            .future
            .iter()
            .any(|slot| slot.depth.is_some() && slot.count > 0)
        {
            let target = self.depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;
            let count = self.settle_depth(target)?;
            self.depth = target;
            if count != 0 {
                return Err("MACRO_EMPTY_LAYER_GAP".into());
            }
        }
        Ok(false)
    }
}
