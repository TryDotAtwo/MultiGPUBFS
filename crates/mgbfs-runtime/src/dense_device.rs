//! Development single-bucket GPU stepper; not the production archived runtime.
//! Serial or two-slot producer/consumer mode, two state arenas and three hash
//! arenas. All device allocation precedes depth zero. Only depth finalization
//! reads counts on the host. Snapshot is an explicit verification-only readback.
//! No archive, StateRing reclamation, reserve planner or rank exchange:
//! this API intentionally does not issue a RunCommit or expose production `run`.
use mgbfs_core::{hash::GemmHash, matrix::MatrixGroup, Result};
use mgbfs_cuda::ffi::*;
use std::{
    ffi::{c_void, CStr},
    mem::size_of,
};
fn check(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("NATIVE_CUDA_STATUS_{code}"))
    }
}
struct Buffer(*mut c_void);
impl Buffer {
    fn new(bytes: usize) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut p, bytes.max(1)) })?;
        Ok(Self(p))
    }
    fn upload<T: Copy>(&self, v: &[T]) -> Result<()> {
        check(unsafe { cudaMemcpy(self.0, v.as_ptr().cast(), std::mem::size_of_val(v), 1) })
    }
    fn read<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        if n > 0 {
            check(unsafe { cudaMemcpy(v.as_mut_ptr().cast(), self.0, n * size_of::<T>(), 2) })?;
        }
        Ok(v)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.0);
        }
    }
}
struct Plan(*mut c_void, unsafe extern "C" fn(*mut c_void));
impl Plan {
    fn create(
        destroy: unsafe extern "C" fn(*mut c_void),
        f: impl FnOnce(*mut *mut c_void, *mut i8) -> i32,
    ) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        let mut err = [0i8; 512];
        if f(&mut p, err.as_mut_ptr()) != 0 {
            return Err(unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        Ok(Self(p, destroy))
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
impl Stream {
    fn new() -> Result<Self> {
        let mut p = std::ptr::null_mut();
        check(unsafe { cudaStreamCreateWithFlags(&mut p, 1) })?;
        Ok(Self(p))
    }
}
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
struct Pipeline {
    producer: Stream,
    ready: [Event; 2],
    released: [Event; 2],
}
impl Pipeline {
    fn new() -> Result<Self> {
        Ok(Self {
            producer: Stream::new()?,
            ready: [Event::new()?, Event::new()?],
            released: [Event::new()?, Event::new()?],
        })
    }
}
pub struct DenseDeviceStepper {
    stream: Stream,
    pipeline: Option<Pipeline>,
    slot_capacity: usize,
    generate: Plan,
    hash: Plan,
    route: Plan,
    owner: Plan,
    materialize: Plan,
    current: Buffer,
    next: Buffer,
    previous_hashes: Buffer,
    current_hashes: Buffer,
    next_hashes: Buffer,
    materialized_hashes: Buffer,
    children: Buffer,
    hashes: Buffer,
    origins: Buffer,
    sorted: Buffer,
    sorted_refs: Buffer,
    routed_count: Buffer,
    owner_state: Buffer,
    survivors: Buffer,
    survivor_refs: Buffer,
    survivor_count: Buffer,
    frontier_state: Buffer,
    width: usize,
    stride: usize,
    batch: u32,
    moves: u32,
    current_count: u32,
    previous_count: u32,
    prededup: bool,
    failed: bool,
}
impl Drop for DenseDeviceStepper {
    fn drop(&mut self) {
        unsafe {
            cudaStreamSynchronize(self.stream.0);
            if let Some(p) = &self.pipeline {
                cudaStreamSynchronize(p.producer.0);
            }
        }
    }
}
impl DenseDeviceStepper {
    pub fn new_pipelined(
        g: &MatrixGroup,
        seed: [u8; 16],
        batch: u32,
        capacity: u32,
        prededup: bool,
    ) -> Result<Self> {
        Self::new_mode(g, seed, batch, capacity, prededup, true, 0)
    }
    /// Explicit experimental generation configuration, fixed before allocation.
    pub fn new_pipelined_with_generation(
        g: &MatrixGroup,
        seed: [u8; 16],
        batch: u32,
        capacity: u32,
        prededup: bool,
        generation_variant: u32,
    ) -> Result<Self> {
        Self::new_mode(g, seed, batch, capacity, prededup, true, generation_variant)
    }
    pub fn new(
        g: &MatrixGroup,
        seed: [u8; 16],
        batch: u32,
        capacity: u32,
        prededup: bool,
    ) -> Result<Self> {
        Self::new_mode(g, seed, batch, capacity, prededup, false, 0)
    }
    fn new_mode(
        g: &MatrixGroup,
        seed: [u8; 16],
        batch: u32,
        capacity: u32,
        prededup: bool,
        pipelined: bool,
        generation_variant: u32,
    ) -> Result<Self> {
        g.validate()?;
        let moves = g.generators.len() as u32;
        let candidates = batch.checked_mul(moves).ok_or("CANDIDATE_COUNT_OVERFLOW")?;
        if batch == 0 || capacity == 0 || candidates > i32::MAX as u32 {
            return Err("STEPPER_CAPACITY".into());
        }
        let width = g.start.len();
        let stride = (width + 15) & !15;
        let hashing = GemmHash::from_seed(width, seed)?;
        let limbs = hashing.limbs();
        let generators: Vec<u8> = g.generators.iter().flatten().copied().collect();
        unsafe {
            let mut stream = std::ptr::null_mut();
            check(cudaStreamCreateWithFlags(&mut stream, 1))?;
            let stream = Stream(stream);
            let generate = Plan::create(mgbfs_generate_destroy, |p, e| {
                mgbfs_generate_create_variant(
                    g.rows as u32,
                    moves,
                    g.modulus as u32,
                    batch,
                    generators.as_ptr(),
                    generation_variant,
                    p,
                    e,
                    512,
                )
            })?;
            let hash = Plan::create(mgbfs_hash_destroy, |p, e| {
                mgbfs_hash_create(
                    width as u32,
                    candidates,
                    limbs.as_ptr(),
                    hashing.offsets.as_ptr(),
                    p,
                    e,
                    512,
                )
            })?;
            let route = Plan::create(mgbfs_route_destroy, |p, e| {
                mgbfs_route_create(candidates, p, e, 512)
            })?;
            let owner = Plan::create(mgbfs_owner_destroy, |p, e| {
                mgbfs_owner_create(candidates, capacity, p, e, 512)
            })?;
            let materialize = Plan::create(mgbfs_materialize_destroy, |p, e| {
                mgbfs_materialize_create(stride as u32, candidates, capacity, p, e, 512)
            })?;
            let c = candidates as usize;
            let f = capacity as usize;
            let result = Self {
                stream,
                pipeline: if pipelined {
                    Some(Pipeline::new()?)
                } else {
                    None
                },
                slot_capacity: c,
                generate,
                hash,
                route,
                owner,
                materialize,
                current: Buffer::new(f * stride)?,
                next: Buffer::new(f * stride)?,
                previous_hashes: Buffer::new(f * 16)?,
                current_hashes: Buffer::new(f * 16)?,
                next_hashes: Buffer::new(f * 16)?,
                materialized_hashes: Buffer::new(f * 16)?,
                children: Buffer::new(c * stride * if pipelined { 2 } else { 1 })?,
                hashes: Buffer::new(c * 16 * if pipelined { 2 } else { 1 })?,
                origins: Buffer::new(c * 8)?,
                sorted: Buffer::new(c * 16)?,
                sorted_refs: Buffer::new(c * 8)?,
                routed_count: Buffer::new(4)?,
                owner_state: Buffer::new(size_of::<OwnerState>())?,
                survivors: Buffer::new(c * 16)?,
                survivor_refs: Buffer::new(c * 8)?,
                survivor_count: Buffer::new(4)?,
                frontier_state: Buffer::new(8)?,
                width,
                stride,
                batch,
                moves,
                current_count: 1,
                previous_count: 0,
                prededup,
                failed: false,
            };
            let mut start = vec![0u8; stride];
            start[..width].copy_from_slice(&g.start);
            result.current.upload(&start)?;
            result.origins.upload(&(0..c as u64).collect::<Vec<_>>())?;
            check(mgbfs_hash_run(
                result.hash.0,
                result.current.0.cast(),
                result.current_hashes.0.cast(),
                1,
                result.stream.0,
            ))?;
            check(cudaStreamSynchronize(result.stream.0))?;
            Ok(result)
        }
    }
    /// Published depth-finalization count; no device readback.
    pub fn frontier_len(&self) -> u32 {
        self.current_count
    }
    pub fn snapshot(&self) -> Result<Vec<Vec<u8>>> {
        if self.failed {
            return Err("STEPPER_FAILED".into());
        }
        check(unsafe { cudaStreamSynchronize(self.stream.0) })?;
        let bytes = self
            .current
            .read::<u8>(self.current_count as usize * self.stride)?;
        Ok(bytes
            .chunks_exact(self.stride)
            .map(|s| s[..self.width].to_vec())
            .collect())
    }
    pub fn advance(&mut self) -> Result<bool> {
        if self.failed {
            return Err("STEPPER_FAILED".into());
        }
        if self.current_count == 0 {
            return Ok(false);
        }
        let result = self.advance_inner();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn advance_inner(&mut self) -> Result<bool> {
        unsafe {
            let s = self.stream.0;
            check(cudaMemsetAsync(
                self.owner_state.0,
                0,
                size_of::<OwnerState>(),
                s,
            ))?;
            check(cudaMemsetAsync(self.frontier_state.0, 0, 8, s))?;
            for (epoch, offset) in (0..self.current_count)
                .step_by(self.batch as usize)
                .enumerate()
            {
                let n = self.batch.min(self.current_count - offset);
                let children = n * self.moves;
                let bank = if self.pipeline.is_some() {
                    epoch % 2
                } else {
                    0
                };
                let child_ptr = self
                    .children
                    .0
                    .cast::<u8>()
                    .add(bank * self.slot_capacity * self.stride);
                let hash_ptr = self
                    .hashes
                    .0
                    .cast::<u8>()
                    .add(bank * self.slot_capacity * 16);
                let producer = if let Some(p) = &self.pipeline {
                    // Capture the previously recorded release before re-recording
                    // it for this epoch. No producer waits for a future record.
                    if epoch >= 2 {
                        check(cudaStreamWaitEvent(p.producer.0, p.released[bank].0, 0))?;
                    }
                    p.producer.0
                } else {
                    s
                };
                check(mgbfs_generate_run(
                    self.generate.0,
                    self.current
                        .0
                        .cast::<u8>()
                        .add(offset as usize * self.stride),
                    child_ptr,
                    n,
                    producer,
                ))?;
                check(mgbfs_hash_run(
                    self.hash.0,
                    child_ptr,
                    hash_ptr.cast(),
                    children,
                    producer,
                ))?;
                if let Some(p) = &self.pipeline {
                    check(cudaEventRecord(p.ready[bank].0, producer))?;
                    check(cudaStreamWaitEvent(s, p.ready[bank].0, 0))?;
                }
                check(mgbfs_route_run(
                    self.route.0,
                    hash_ptr.cast(),
                    self.origins.0.cast(),
                    self.sorted.0,
                    self.sorted_refs.0.cast(),
                    self.routed_count.0.cast(),
                    children,
                    self.prededup as i32,
                    s,
                ))?;
                check(mgbfs_owner_run(
                    self.owner.0,
                    self.previous_hashes.0,
                    self.previous_count,
                    self.current_hashes.0,
                    self.current_count,
                    self.next_hashes.0,
                    self.owner_state.0.cast(),
                    self.sorted.0,
                    self.sorted_refs.0.cast(),
                    self.routed_count.0.cast(),
                    self.survivors.0,
                    self.survivor_refs.0.cast(),
                    self.survivor_count.0.cast(),
                    epoch as u64,
                    s,
                ))?;
                check(mgbfs_materialize_run(
                    self.materialize.0,
                    child_ptr,
                    children,
                    self.survivors.0,
                    self.survivor_refs.0.cast(),
                    self.survivor_count.0.cast(),
                    self.next.0.cast(),
                    self.materialized_hashes.0,
                    self.frontier_state.0.cast(),
                    s,
                ))?;
                if let Some(p) = &self.pipeline {
                    check(cudaEventRecord(p.released[bank].0, s))?;
                }
            }
            check(cudaStreamSynchronize(s))?;
            let owner = self.owner_state.read::<OwnerState>(1)?[0];
            let frontier = self.frontier_state.read::<FrontierState>(1)?[0];
            if owner.fatal != 0 {
                return Err(format!("OWNER_FATAL_{}", owner.fatal));
            }
            if frontier.fatal != 0 {
                return Err(format!("MATERIALIZE_FATAL_{}", frontier.fatal));
            }
            if owner.count != frontier.count {
                return Err("MATERIALIZE_COUNT_MISMATCH".into());
            }
            // No previous layer's states survive this swap. Only hashes are retained
            // for d-1 and d; inverse closure is validated before using this window.
            std::mem::swap(&mut self.current, &mut self.next);
            std::mem::swap(&mut self.previous_hashes, &mut self.current_hashes);
            std::mem::swap(&mut self.current_hashes, &mut self.next_hashes);
            self.previous_count = self.current_count;
            self.current_count = frontier.count;
            Ok(self.current_count != 0)
        }
    }
}
