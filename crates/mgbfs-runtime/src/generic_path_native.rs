//! Exact nearest search and window lineage on the append-only native path arenas.
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::Ordering;
use std::time::Instant;

use mgbfs_core::{
    graph_definition::{GraphAction, GraphDefinitionV2},
    Result,
};
use mgbfs_cuda::{
    ffi::{cudaMemcpyAsync, cudaStreamSynchronize},
    generic_graph::mgbfs_generic_gather_storage,
    native_owner::{cudaMemGetInfo, cudaSetDevice},
};
use serde::Deserialize;

use crate::generic_memory::{self, GenericMemoryPlan};
use crate::generic_native::{checked, Buffer, GenericAdvance, GenericNativeBfs};
use crate::generic_run::{Signals, CANCELLED};
use crate::path_meeting::{decode_meeting, LayerRange, Meeting};

mod nearest_witness;
use nearest_witness::{walk_parents, OwnerRow, Receipt, Trace};

#[repr(C)]
struct Scan {
    state_bytes: u32,
    kind: u32,
    elements: u32,
    n: u32,
    m: u32,
    generators: u32,
    count: u32,
    target_count: u32,
    transitions: u32,
    begin: u64,
    stride: u64,
    states: *const c_void,
    permutations: *const u32,
    matrices: *const i64,
    moduli: *const u32,
    targets: *const i64,
}

type ScanFn = unsafe extern "C" fn(*const Scan, *mut u64, *mut c_void) -> i32;

#[repr(C)]
struct TargetLookup {
    state_bytes: u32,
    elements: u32,
    hash_bits: u32,
    shared_bucket: u32,
    begin: u32,
    count: u32,
    target_count: u32,
    seed: u64,
    stride: u64,
    slot_count: u64,
    states: *const c_void,
    targets: *const i64,
    slots: *const u64,
}

type TargetLookupFn = unsafe extern "C" fn(*const TargetLookup, *mut u64, *mut c_void) -> i32;

fn target_lookup_kernel() -> Option<TargetLookupFn> {
    let symbol = unsafe {
        libc::dlsym(
            libc::RTLD_DEFAULT,
            b"mgbfs_path_find_targets_v1\0".as_ptr().cast(),
        )
    };
    (!symbol.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, TargetLookupFn>(symbol) })
}

fn scanner() -> Option<ScanFn> {
    let symbol =
        unsafe { libc::dlsym(libc::RTLD_DEFAULT, b"mgbfs_path_scan_v2\0".as_ptr().cast()) };
    if symbol.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute::<*mut c_void, ScanFn>(symbol) })
    }
}

#[repr(C)]
struct Meet {
    state_bytes: u32,
    elements: u32,
    hash_bits: u32,
    shared_bucket: u32,
    left_begin: u32,
    left_count: u32,
    right_count: u32,
    seed: u64,
    left_stride: u64,
    right_stride: u64,
    slot_count: u64,
    left: *const c_void,
    right: *const c_void,
    slots: *const u64,
}

type MeetFn = unsafe extern "C" fn(*const Meet, *mut u64, *mut c_void) -> i32;

fn meeting_kernel() -> Result<MeetFn> {
    let symbol =
        unsafe { libc::dlsym(libc::RTLD_DEFAULT, b"mgbfs_path_meet_v1\0".as_ptr().cast()) };
    if symbol.is_null() {
        return Err("NATIVE_PATH_MEET_CUDA_ABI_UNAVAILABLE".into());
    }
    Ok(unsafe { std::mem::transmute::<*mut c_void, MeetFn>(symbol) })
}

type ResolveChildrenFn = unsafe extern "C" fn(*const Meet, *mut u32, u32, *mut c_void) -> i32;

fn resolve_children_kernel() -> Result<ResolveChildrenFn> {
    let symbol = unsafe {
        libc::dlsym(
            libc::RTLD_DEFAULT,
            b"mgbfs_path_resolve_children_v1\0".as_ptr().cast(),
        )
    };
    if symbol.is_null() {
        return Err("NATIVE_WINDOW_JOIN_CUDA_ABI_UNAVAILABLE".into());
    }
    Ok(unsafe { std::mem::transmute::<*mut c_void, ResolveChildrenFn>(symbol) })
}

pub fn window_bidirectional_available() -> bool {
    bidirectional_available() && resolve_children_kernel().is_ok()
}

pub fn bidirectional_available() -> bool {
    available() && meeting_kernel().is_ok()
}

pub fn available() -> bool {
    scanner().is_some()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    graph: GraphDefinitionV2,
    targets: Vec<Vec<i64>>,
    device: u32,
    max_seconds: Option<f64>,
}

struct QueryTime {
    started: Instant,
    seconds: Option<f64>,
    resident: bool,
}

impl QueryTime {
    fn stop_reason(&self) -> Option<&'static str> {
        if self
            .seconds
            .is_some_and(|limit| limit > 0.0 && self.started.elapsed().as_secs_f64() >= limit)
        {
            return Some("TIME_LIMIT");
        }
        if CANCELLED.load(Ordering::Relaxed) || (self.resident && crate::job_phase::cancelled()) {
            return Some("CANCELLED");
        }
        None
    }
}

enum Match {
    Hit { row: u32, label: u32 },
    Missing,
    Stopped(&'static str),
}

struct QueryGpu {
    targets: Buffer,
    found: Buffer,
    row_index: Buffer,
    witness_row: Buffer,
    scan: ScanFn,
    target_lookup: Option<TargetLookupFn>,
}

impl QueryGpu {
    fn new(plan: &GenericMemoryPlan, tile_bytes: usize, device: i32, scan: ScanFn) -> Result<Self> {
        Ok(Self {
            targets: Buffer::new(tile_bytes, device)?,
            found: Buffer::new(8, device)?,
            row_index: Buffer::new(4, device)?,
            witness_row: Buffer::new(plan.elements as usize * plan.state_bytes as usize, device)?,
            scan,
            target_lookup: target_lookup_kernel(),
        })
    }

    fn read_best(&self, stream: *mut c_void) -> Result<u64> {
        let mut best = u64::MAX;
        let copied = checked!(cudaMemcpyAsync, unsafe {
            cudaMemcpyAsync((&mut best as *mut u64).cast(), self.found.ptr, 8, 2, stream)
        });
        let drained = checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        });
        copied.and(drained)?;
        Ok(best)
    }

    fn find_committed(
        &self,
        bfs: &GenericNativeBfs,
        range: LayerRange,
        target_count: u32,
        shared_bucket: bool,
        time: &QueryTime,
    ) -> Result<Match> {
        let Some(lookup) = self.target_lookup else {
            return self.scan_layer(bfs, range, target_count, false, time);
        };
        let (begin, count) = range;
        let target_bytes = bfs.plan.elements as usize * std::mem::size_of::<i64>();
        if target_count == 0
            || target_count as usize > self.targets.bytes / target_bytes
            || begin > bfs.visited_used
            || count > bfs.visited_used - begin
        {
            return Err("NATIVE_PATH_COMMITTED_LAYER_BOUNDS".into());
        }
        if count == 0 {
            return Ok(Match::Missing);
        }

        const TARGETS_PER_LAUNCH: u32 = 4096;
        let stream = bfs.stream.ptr;
        let mut offset = 0;
        while offset < target_count {
            if let Some(reason) = time.stop_reason() {
                return Ok(Match::Stopped(reason));
            }
            let tile = (target_count - offset).min(TARGETS_PER_LAUNCH);
            let request = TargetLookup {
                state_bytes: bfs.plan.state_bytes,
                elements: bfs.plan.elements,
                hash_bits: bfs.hash_bits,
                shared_bucket: u32::from(shared_bucket),
                begin,
                count,
                target_count: tile,
                seed: bfs.seed,
                stride: u64::from(bfs.arena_stride),
                slot_count: bfs.plan.table_slots,
                states: bfs.visited.ptr,
                targets: self.targets.at(offset as usize * target_bytes),
                slots: bfs.slots.ptr.cast(),
            };
            checked!(mgbfs_path_find_targets_v1, unsafe {
                lookup(&request, self.found.ptr.cast(), stream)
            })?;
            let best = self.read_best(stream)?;
            if let Some(reason) = time.stop_reason() {
                return Ok(Match::Stopped(reason));
            }
            if best != u64::MAX {
                let label = (best >> 32) as u32;
                let row = best as u32;
                if label >= tile {
                    return Err("NATIVE_PATH_LOOKUP_RESULT_BOUNDS".into());
                }
                if row == u32::MAX {
                    return self.scan_layer(bfs, range, target_count, false, time);
                }
                if row < begin || row - begin >= count {
                    return Err("NATIVE_PATH_LOOKUP_RESULT_BOUNDS".into());
                }
                return Ok(Match::Hit { row, label: offset + label });
            }
            offset += tile;
        }
        Ok(Match::Missing)
    }

    fn scan_layer(
        &self,
        bfs: &GenericNativeBfs,
        (start, count): LayerRange,
        target_count: u32,
        transitions: bool,
        time: &QueryTime,
    ) -> Result<Match> {
        const PAIRS: u32 = 1 << 16;
        let labels = if transitions {
            bfs.generators
        } else {
            target_count
        };
        if labels == 0 || start > bfs.visited_used || count > bfs.visited_used - start {
            return Err("NATIVE_PATH_COMMITTED_LAYER_BOUNDS".into());
        }
        let mut hit = Match::Missing;
        let mut best_label = labels;
        let mut offset = 0;
        while offset < count {
            if let Some(reason) = time.stop_reason() {
                return Ok(Match::Stopped(reason));
            }
            // A state match only needs to improve the best target in
            // subsequent row tiles. Transition scans retain generator order.
            let batch = (PAIRS / best_label).max(1);
            let scan = Scan {
                state_bytes: bfs.plan.state_bytes,
                kind: bfs.kind,
                elements: bfs.plan.elements,
                n: bfs.rows,
                m: bfs.cols,
                generators: bfs.generators,
                count: (count - offset).min(batch),
                target_count: if transitions { target_count } else { best_label },
                transitions: u32::from(transitions),
                begin: u64::from(start) + u64::from(offset),
                stride: u64::from(bfs.arena_stride),
                states: bfs.visited.ptr,
                permutations: bfs.perms.ptr.cast(),
                matrices: bfs.matrices.ptr.cast(),
                moduli: bfs.moduli.ptr.cast(),
                targets: self.targets.ptr.cast(),
            };
            let stream = bfs.stream.ptr;
            checked!(mgbfs_path_scan_v2, unsafe {
                (self.scan)(&scan, self.found.ptr.cast(), stream)
            })?;
            let best = self.read_best(stream)?;
            if let Some(reason) = time.stop_reason() {
                return Ok(Match::Stopped(reason));
            }
            if best != u64::MAX {
                let (parent, label) = if transitions {
                    (best / u64::from(labels), best % u64::from(labels))
                } else {
                    (best % u64::from(scan.count), best / u64::from(scan.count))
                };
                let label_count = if transitions { labels } else { scan.target_count };
                if parent >= u64::from(scan.count) || label >= u64::from(label_count) {
                    return Err("NATIVE_PATH_SCAN_RESULT_BOUNDS".into());
                }
                let label = label as u32;
                let candidate = Match::Hit {
                    row: start + offset + parent as u32,
                    label,
                };
                if transitions || label == 0 {
                    return Ok(candidate);
                }
                hit = candidate;
                best_label = label;
            }
            offset += scan.count;
        }
        Ok(hit)
    }

    fn meet(
        &self,
        left: &GenericNativeBfs,
        range: LayerRange,
        right: &GenericNativeBfs,
        right_layers: &[LayerRange],
        shared_bucket: bool,
    ) -> Result<Option<Meeting>> {
        let (begin, count) = range;
        if left.device != right.device
            || left.plan.elements != right.plan.elements
            || left.plan.state_bytes != right.plan.state_bytes
            || begin > left.visited_used
            || count > left.visited_used - begin
        {
            return Err("NATIVE_PATH_MEET_ARENA_LAYOUT".into());
        }
        let right_count = match right_layers.last() {
            Some(&(start, count)) => start
                .checked_add(count)
                .ok_or("NATIVE_PATH_MEET_RIGHT_BOUNDS")?,
            None => 0,
        };
        if right_count > right.visited_used {
            return Err("NATIVE_PATH_MEET_RIGHT_BOUNDS".into());
        }
        if count == 0 || right_count == 0 {
            return Ok(None);
        }
        let kernel = meeting_kernel()?;
        let query = Meet {
            state_bytes: left.plan.state_bytes,
            elements: left.plan.elements,
            hash_bits: right.hash_bits,
            shared_bucket: u32::from(shared_bucket),
            left_begin: begin,
            left_count: count,
            right_count,
            seed: right.seed,
            left_stride: u64::from(left.arena_stride),
            right_stride: u64::from(right.arena_stride),
            slot_count: right.plan.table_slots,
            left: left.visited.ptr,
            right: right.visited.ptr,
            slots: right.slots.ptr.cast(),
        };
        // Readers run only after the scheduler has committed both inputs.
        let stream = left.stream.ptr;
        checked!(mgbfs_path_meet_v1, unsafe {
            kernel(&query, self.found.ptr.cast(), stream)
        })?;
        let mut best = u64::MAX;
        checked!(cudaMemcpyAsync, unsafe {
            cudaMemcpyAsync((&mut best as *mut u64).cast(), self.found.ptr, 8, 2, stream)
        })?;
        checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        })?;
        decode_meeting(best, range, right_layers)
    }

    fn row_values(&self, bfs: &GenericNativeBfs, row: u32) -> Result<Vec<i64>> {
        if row >= bfs.visited_used {
            return Err("NATIVE_PATH_ROW_BOUNDS".into());
        }
        let bytes = bfs.plan.state_bytes as usize;
        let width = bfs.plan.elements as usize;
        let mut metadata = vec![0u8; width.checked_mul(bytes).ok_or("NATIVE_PATH_ROW_BYTES")?];
        self.row_index.upload(&[row])?;
        checked!(mgbfs_generic_gather_storage, unsafe {
            mgbfs_generic_gather_storage(
                bfs.plan.state_bytes,
                bfs.plan.elements,
                bfs.visited.ptr.cast(),
                bfs.arena_stride,
                self.row_index.ptr.cast(),
                1,
                self.witness_row.ptr.cast(),
                1,
                bfs.stream.ptr,
            )
        })?;
        checked!(cudaMemcpyAsync, unsafe {
            cudaMemcpyAsync(
                metadata.as_mut_ptr().cast(),
                self.witness_row.ptr,
                metadata.len(),
                2,
                bfs.stream.ptr,
            )
        })?;
        checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(bfs.stream.ptr)
        })?;
        if bytes == 1 {
            return Ok(metadata.into_iter().map(i64::from).collect());
        }
        Ok(metadata
            .chunks_exact(8)
            .map(|cell| i64::from_ne_bytes(cell.try_into().unwrap()))
            .collect())
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Layout {
    elements: u32,
    state_bytes: u32,
    action_bytes: [u64; 3],
    upper: Option<u64>,
    reserve: u64,
}

impl Layout {
    fn new(graph: &GraphDefinitionV2) -> Result<Self> {
        let elements = u32::try_from(graph.start.len()).map_err(|_| "GENERIC_STATE_WIDTH")?;
        let generators = graph.generator_count() as u64;
        let action_bytes = match &graph.action {
            GraphAction::Permutation { degree, .. } => [u64::from(*degree) * generators * 4, 1, 1],
            GraphAction::Matrix { rows, .. } => {
                let side = u64::from(*rows);
                let matrices = side
                    .checked_mul(side)
                    .and_then(|cells| cells.checked_mul(generators))
                    .and_then(|cells| cells.checked_mul(8))
                    .ok_or("GENERIC_ARENA_OVERFLOW")?;
                [1, matrices, generators * 4]
            }
        };
        Ok(Self {
            elements,
            state_bytes: generic_memory::preferred_state_bytes(graph),
            action_bytes,
            upper: generic_memory::state_space_bound(graph),
            reserve: generic_memory::admission_reserve_bytes()?,
        })
    }

    fn target_tile(&self) -> (usize, usize) {
        let row_bytes = self.elements as usize * 8;
        let rows = ((256 << 10) / row_bytes).max(1);
        (rows, rows * row_bytes)
    }

    fn plan(&self) -> Result<GenericMemoryPlan> {
        let mut free = 0;
        let mut total = 0;
        checked!(cudaMemGetInfo, unsafe {
            cudaMemGetInfo(&mut free, &mut total)
        })?;
        let scratch = self
            .action_bytes
            .iter()
            .copied()
            .try_fold(
                self.target_tile().1 as u64
                    + 12
                    + u64::from(self.elements) * u64::from(self.state_bytes),
                u64::checked_add,
            )
            .ok_or("GENERIC_ARENA_OVERFLOW")?;
        let available = (free as u64)
            .checked_sub(scratch)
            .ok_or("GENERIC_VRAM_HEADROOM")?;
        GenericMemoryPlan::automatic_storage(self.elements, available, self.upper, self.state_bytes)
    }
}

#[derive(Default)]
pub(crate) struct ResidentNearest {
    layout: Option<(Layout, GenericMemoryPlan)>,
}

impl ResidentNearest {
    pub(crate) fn clear(&mut self) {
        self.layout = None;
    }

    fn trim(&mut self) -> Result<()> {
        self.clear();
        crate::session_cache::release_idle_storage();
        crate::generic_pool::trim_idle()
    }

    fn prepare(&mut self, layout: Layout) -> Result<(GenericMemoryPlan, bool)> {
        let reused = self
            .layout
            .as_ref()
            .is_some_and(|(previous, _)| previous == &layout);
        if !reused {
            self.trim()?;
            let plan = layout.plan()?;
            self.layout = Some((layout, plan));
        }
        // This dedicated worker never creates an NCCL communicator. Here the
        // existing cache gate only parks raw allocations and synchronized streams.
        crate::session_cache::begin("path-nearest-v1".into());
        crate::session_cache::permit_comm(true);
        Ok((self.layout.as_ref().unwrap().1.clone(), reused))
    }
}

fn search(
    query: &Query,
    time: &QueryTime,
    scan: ScanFn,
    plan: GenericMemoryPlan,
    (tile_rows, tile_bytes): (usize, usize),
    checked_through: &mut Option<u64>,
) -> Result<Receipt> {
    if let Some(reason) = time.stop_reason() {
        return Ok(Receipt::incomplete(reason, None));
    }
    let device = i32::try_from(query.device).map_err(|_| "NATIVE_PATH_DEVICE")?;
    let kernels = QueryGpu::new(&plan, tile_bytes, device, scan)?;
    let mut bfs = GenericNativeBfs::new(&query.graph, query.device, plan, 0x19f856a2, 64)?;
    let mut tile_data = Vec::new();
    let mut layers = vec![(bfs.current_start, bfs.frontier_len())];
    let checkpoint = || time.stop_reason().map_or(Ok(()), |reason| Err(reason.into()));

    let (target_index, mut final_edge) = 'layers: loop {
        let depth = layers.len() as u64 - 1;
        crate::native_fault::layer(depth, depth);
        for (tile, targets) in query.targets.chunks(tile_rows).enumerate() {
            if let Some(reason) = time.stop_reason() {
                return Ok(Receipt::incomplete(reason, None));
            }
            tile_data.clear();
            tile_data.extend(targets.iter().flatten().copied());
            kernels.targets.upload(&tile_data)?;
            match kernels.find_committed(
                &bfs,
                *layers.last().unwrap(),
                targets.len() as u32,
                false,
                time,
            )? {
                Match::Hit { label, .. } => {
                    *checked_through = Some(depth);
                    break 'layers ((tile * tile_rows) as u32 + label, None);
                }
                Match::Stopped(reason) => return Ok(Receipt::incomplete(reason, None)),
                Match::Missing => {}
            }
        }
        *checked_through = Some(depth);
        match bfs.advance_with_checkpoint(&checkpoint) {
            Ok(GenericAdvance::Layer { count }) => {
                layers.push((bfs.current_start, count));
                crate::native_fault::completed(depth + 1);
            }
            Ok(GenericAdvance::Complete) => return Ok(Receipt::unreachable()),
            Ok(GenericAdvance::Resource { fatal }) if fatal & !7 == 0 => {
                // Capacity failure leaves the committed append-only prefix intact.
                // Every layer through `depth` was checked against all goals, so
                // a direct transition from this frontier proves distance depth + 1.
                let committed = layers[depth as usize];
                for (index, target) in query.targets.iter().enumerate() {
                    if let Some(reason) = time.stop_reason() {
                        return Ok(Receipt::incomplete(reason, None));
                    }
                    kernels.targets.upload(target)?;
                    match kernels.scan_layer(&bfs, committed, 1, true, time)? {
                        Match::Hit { row, label } => {
                            *checked_through = Some(depth + 1);
                            break 'layers (index as u32, Some((row, label)));
                        }
                        Match::Stopped(reason) => return Ok(Receipt::incomplete(reason, None)),
                        Match::Missing => {}
                    }
                }
                *checked_through = Some(depth + 1);
                return Ok(Receipt::incomplete("CUDA_MEMORY", None));
            }
            Ok(GenericAdvance::Resource { fatal }) => {
                return Err(format!("NATIVE_PATH_OWNER_RESOURCE_{fatal}"))
            }
            Err(error) => match query_stop_reason(&error) {
                Some(reason) => return Ok(Receipt::incomplete(reason, None)),
                None => return Err(error),
            },
        }
    };

    let distance = layers.len() - 1 + usize::from(final_edge.is_some());
    let mut labels = Vec::with_capacity(distance);
    if distance != 0 {
        let mut child = query.targets[target_index as usize].clone();
        for depth in (0..distance).rev() {
            let predecessor = if let Some((row, label)) = final_edge.take() {
                Match::Hit { row, label }
            } else {
                kernels.targets.upload(&child)?;
                kernels.scan_layer(&bfs, layers[depth], 1, true, time)?
            };
            match predecessor {
                Match::Hit { row, label } => {
                    labels.push(label);
                    if depth != 0 {
                        child = kernels.row_values(&bfs, row)?;
                    }
                }
                Match::Stopped(reason) => {
                    return Ok(Receipt::incomplete(reason, Some(distance as u64)))
                }
                Match::Missing => return Err("NATIVE_PATH_PREDECESSOR_NOT_FOUND".into()),
            }
        }
        labels.reverse();
    }
    Ok(Receipt::complete(distance as u64, target_index, labels))
}

pub(crate) fn distributed_workspace(
    width: u64,
    state_bytes: u32,
    world: usize,
    bidirectional: bool,
) -> Result<Vec<usize>> {
    let row = width.checked_mul(8).ok_or("NATIVE_PATH_GATHER_SIZE")?;
    let targets = if bidirectional {
        row
    } else {
        row.max(256 << 10)
    };
    let witness = width
        .checked_mul(u64::from(state_bytes))
        .ok_or("NATIVE_PATH_GATHER_SIZE")?;
    let tx = row
        .checked_add(if bidirectional { 24 } else { 16 })
        .ok_or("NATIVE_PATH_GATHER_SIZE")?;
    let rx = tx
        .checked_mul(world as u64)
        .ok_or("NATIVE_PATH_GATHER_SIZE")?;
    [targets, 8, 4, witness, tx, rx]
        .into_iter()
        .map(|size| usize::try_from(size).map_err(|_| "NATIVE_PATH_GATHER_SIZE".into()))
        .collect()
}

pub(crate) const PARENT_BYTES: usize =
    std::mem::size_of::<mgbfs_cuda::generic_graph::GenericPathParent>();

pub(crate) struct DistributedPathResult {
    pub(crate) receipt: serde_json::Value,
    // Retain the buffers through receipt commit. The worker can park them
    // only after the job has successfully quiesced and published.
    _workspace: (QueryGpu, Buffer, Option<Buffer>),
    _parents: Option<(Buffer, Option<Buffer>)>,
}

// Adapt the parent walk to collective reads. Every rank participates in
// each exchange, while only the owner reads the link from its GPU.
fn trace_parents(
    bfs: &crate::generic_distributed_native::GenericDistributedBfs,
    parents: &Buffer,
    layers: &[(u32, u32)],
    child: OwnerRow,
    tx: &Buffer,
    rx: &Buffer,
    gathered: &mut [i64],
    labels: &mut Vec<u32>,
    time: &QueryTime,
) -> Result<Trace<OwnerRow>> {
    use mgbfs_cuda::generic_graph::GenericPathParent;

    let arena = bfs.path_arena()?;
    let gathered = &mut gathered[..3 * bfs.path_world()];
    let read_parent = |(start, count), OwnerRow { owner, row }| {
        let mut local = [0; 3];
        let hit = (|| {
            if owner != bfs.path_rank() {
                return Ok(Match::Missing);
            }
            if row < start || row - start >= count || row >= arena.visited_used {
                return Err("NATIVE_PATH_PARENT_ROW".into());
            }
            if let Some(reason) = time.stop_reason() {
                return Ok(Match::Stopped(reason));
            }
            let mut parent = GenericPathParent {
                parent: 0,
                source_label: 0,
            };
            checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(
                    (&mut parent as *mut GenericPathParent).cast(),
                    parents.at::<c_void>(row as usize * PARENT_BYTES),
                    PARENT_BYTES,
                    2,
                    arena.stream.ptr,
                )
            })?;
            checked!(cudaStreamSynchronize, unsafe {
                cudaStreamSynchronize(arena.stream.ptr)
            })?;
            let rank = parent.source_label / arena.generators;
            let label = parent.source_label % arena.generators;
            if rank as usize >= bfs.path_world() || parent.parent == u32::MAX {
                return Err("NATIVE_PATH_PARENT_LINK".into());
            }
            local[2] = (u64::from(rank) << 32 | u64::from(parent.parent)) as i64;
            Ok(Match::Hit { row, label })
        })();
        encode_distributed_match(&mut local, &hit, time);
        bfs.gather_path_words(&local, tx, rx, gathered)?;
        if let Some(reason) = distributed_stop(gathered, 3, hit.as_ref().err())? {
            return Ok(Trace::Stopped(reason));
        }
        let link = &gathered[owner * 3..][..3];
        if link[0] != 1 {
            return Err("NATIVE_PATH_PARENT_LINK".into());
        }
        let target = link[2] as u64;
        let parent = OwnerRow {
            owner: (target >> 32) as usize,
            row: target as u32,
        };
        Ok(Trace::Found((parent, link[1] as u32)))
    };
    walk_parents(layers, child, labels, read_parent)
}

pub(crate) fn distributed_forward(
    bfs: &mut crate::generic_distributed_native::GenericDistributedBfs,
    goals: &[Vec<i64>],
    seconds: u64,
) -> Result<DistributedPathResult> {
    use crate::generic_distributed_native::DistributedAdvance;

    let arena = bfs.path_arena()?;
    let width = arena.plan.elements as usize;
    if goals.is_empty() || goals.len() > u32::MAX as usize || goals.iter().any(|g| g.len() != width)
    {
        return Err("NATIVE_PATH_TARGETS".into());
    }
    let tile_rows = ((256 << 10) / (width * 8)).max(1);
    let row_words = width + 2;
    let gathered_words = row_words
        .checked_mul(bfs.path_world())
        .ok_or("NATIVE_PATH_GATHER_SIZE")?;
    let setup = (|| {
        let scan = scanner().ok_or("NATIVE_PATH_CUDA_ABI_UNAVAILABLE")?;
        let parents = Buffer::new(arena.plan.capacity as usize * PARENT_BYTES, arena.device)?;
        let buffers = (
            QueryGpu::new(&arena.plan, (width * 8).max(256 << 10), arena.device, scan)?,
            Buffer::new(row_words * 8, arena.device)?,
            Buffer::new(gathered_words * 8, arena.device)?,
            parents,
        );
        crate::session_cache::generic_requirements_consumed()?;
        Ok::<_, String>(buffers)
    })();
    if bfs.collective_stop(u32::from(setup.is_err()))? != 0 {
        return setup
            .err()
            .map_or_else(|| Err("NATIVE_PATH_REMOTE_WORKSPACE".into()), Err);
    }
    let (kernel, tx, rx, parents) = setup?;
    let mut local = vec![0; row_words];
    let mut gathered = vec![0; gathered_words];
    let mut tile = Vec::with_capacity(tile_rows * width);
    let mut uploaded_tile = None;
    let time = QueryTime {
        started: Instant::now(),
        seconds: (seconds != 0).then_some(seconds as f64),
        resident: true,
    };
    let mut layers = Vec::new();
    let mut checked_through = None;
    let mut capacity_exhausted = false;
    let checkpoint = || query_checkpoint(&time);

    let receipt = loop {
        let arena = bfs.path_arena()?;
        let range = (arena.current_start, arena.frontier_len());
        if !capacity_exhausted {
            layers.push(range);
        }
        let depth = layers.len() - 1;
        crate::native_fault::layer(depth as u64, depth as u64);
        let mut incoming_label = 0;
        let hit = (|| {
            let scan_rows = if capacity_exhausted { 1 } else { tile_rows };
            for (part, chunk) in goals.chunks(scan_rows).enumerate() {
                if let Some(reason) = time.stop_reason() {
                    return Ok(Match::Stopped(reason));
                }
                // Most requests fit all goals in one tile. Keep it resident
                // across layers, but distinguish the singleton replay layout.
                let tile_key = (part, scan_rows);
                if uploaded_tile != Some(tile_key) {
                    tile.clear();
                    tile.extend(chunk.iter().flatten().copied());
                    kernel.targets.upload(&tile)?;
                    uploaded_tile = Some(tile_key);
                }
                let hit = if capacity_exhausted {
                    kernel.scan_layer(arena, range, chunk.len() as u32, true, &time)?
                } else {
                    kernel.find_committed(arena, range, chunk.len() as u32, true, &time)?
                };
                match hit {
                    Match::Hit { row, label } => {
                        if capacity_exhausted {
                            incoming_label = label;
                        }
                        return Ok(Match::Hit {
                            row,
                            label: (part * scan_rows) as u32
                                + if capacity_exhausted { 0 } else { label },
                        });
                    }
                    Match::Missing => {}
                    stopped => return Ok(stopped),
                }
            }
            Ok::<_, String>(Match::Missing)
        })();
        local.fill(0);
        if let Ok(Match::Hit { row, .. }) = hit {
            local[2] = (u64::from(incoming_label) << 32 | u64::from(row)) as i64;
        }
        encode_distributed_match(&mut local, &hit, &time);
        bfs.gather_path_words(&local[..3], &tx, &rx, &mut gathered[..3 * bfs.path_world()])?;
        let replies = &gathered[..3 * bfs.path_world()];
        if let Some(reason) = distributed_stop(replies, 3, hit.as_ref().err())? {
            break Receipt::incomplete(reason, None);
        }
        let search_depth = depth + usize::from(capacity_exhausted);
        checked_through = Some(search_depth as u64);
        if let Some((owner, winner)) = replies
            .chunks_exact(3)
            .enumerate()
            .filter(|(_, r)| r[0] == 1)
            .min_by_key(|(_, r)| r[1])
        {
            let index = u32::try_from(winner[1]).map_err(|_| "NATIVE_DISTRIBUTED_PATH_LABEL")?;
            let reference = winner[2] as u64;
            let row = reference as u32;
            let mut labels = Vec::with_capacity(search_depth);
            if capacity_exhausted {
                // The final edge came from the scan of the uncommitted
                // expansion; its source row is in the last committed layer.
                labels.push((reference >> 32) as u32);
            }
            if let Trace::Stopped(reason) = trace_parents(
                bfs,
                &parents,
                &layers[1..depth + 1],
                OwnerRow { owner, row },
                &tx,
                &rx,
                &mut gathered,
                &mut labels,
                &time,
            )? {
                break Receipt::incomplete(reason, Some(search_depth as u64));
            }
            labels.reverse();
            break Receipt::complete(search_depth as u64, index, labels);
        }
        if capacity_exhausted {
            break Receipt::incomplete("CUDA_MEMORY", None);
        }
        match bfs.advance_nearest(&parents, &checkpoint) {
            Ok(DistributedAdvance::Layer { .. }) => {
                crate::native_fault::completed((depth + 1) as u64)
            }
            Ok(DistributedAdvance::Complete) => break Receipt::unreachable(),
            Ok(DistributedAdvance::Resource { fatal }) if fatal & !7 == 0 => {
                // The failed expansion did not commit a layer. Test its edges
                // from the intact frontier to prove any target at depth + 1.
                capacity_exhausted = true;
            }
            Ok(DistributedAdvance::Resource { fatal }) => {
                return Err(format!("NATIVE_PATH_OWNER_RESOURCE_{fatal}"));
            }
            Err(error) => match query_stop_reason(&error) {
                Some(reason) => break Receipt::incomplete(reason, None),
                None => return Err(error),
            },
        }
    };
    let mut result = serde_json::to_value(receipt).map_err(|e| e.to_string())?;
    result["native_state_bytes"] = serde_json::json!(bfs.path_arena()?.plan.state_bytes);
    result["search_confirmed_through_depth"] = serde_json::json!(checked_through);
    Ok(DistributedPathResult {
        receipt: result,
        _workspace: (kernel, tx, Some(rx)),
        _parents: Some((parents, None)),
    })
}

/// The two arenas must use identical owner partitions and preserve generator
/// IDs across the inverse action. Construction and GPU admission are separate.
pub(crate) fn distributed_bidirectional(
    forward: &mut crate::generic_distributed_native::GenericDistributedBfs,
    reverse: &mut crate::generic_distributed_native::GenericDistributedBfs,
    goals: &[Vec<i64>],
    seconds: u64,
) -> Result<DistributedPathResult> {
    use crate::generic_distributed_native::DistributedAdvance;

    let world = forward.path_world();
    if reverse.path_world() != world {
        return Err("NATIVE_PATH_MEET_WORLD".into());
    }
    let arena = forward.path_arena()?;
    let width = arena.plan.elements as usize;
    let stride = width + 3;
    let words = stride.checked_mul(world).ok_or("NATIVE_PATH_GATHER_SIZE")?;
    let setup = (|| {
        let scan = scanner().ok_or("NATIVE_PATH_CUDA_ABI_UNAVAILABLE")?;
        let parents = Buffer::new(arena.plan.capacity as usize * PARENT_BYTES, arena.device)?;
        let other = reverse.path_arena()?;
        let reverse_parents =
            Buffer::new(other.plan.capacity as usize * PARENT_BYTES, other.device)?;
        let buffers = (
            QueryGpu::new(&arena.plan, width * 8, arena.device, scan)?,
            Buffer::new(stride * 8, arena.device)?,
            Buffer::new(words * 8, arena.device)?,
            parents,
            reverse_parents,
        );
        crate::session_cache::generic_requirements_consumed()?;
        Ok::<_, String>(buffers)
    })();
    if forward.collective_stop(u32::from(setup.is_err()))? != 0 {
        return Err(setup
            .err()
            .unwrap_or_else(|| "NATIVE_PATH_REMOTE_WORKSPACE".into()));
    }
    let (kernel, tx, rx, parents, reverse_parents) = setup?;
    let mut local = vec![0; stride];
    let mut gathered = vec![0; words];
    let time = QueryTime {
        started: Instant::now(),
        seconds: (seconds != 0).then_some(seconds as f64),
        resident: true,
    };
    let sides = [forward, reverse];
    let range = |arena: &GenericNativeBfs| (arena.current_start, arena.frontier_len());
    let mut layers = [
        vec![range(sides[0].path_arena()?)],
        vec![range(sides[1].path_arena()?)],
    ];
    let mut active = 0;
    let mut checked_depth = None;
    let checkpoint = || query_checkpoint(&time);
    let receipt = 'search: loop {
        let opposite = 1 - active;
        let depths = [layers[0].len() - 1, layers[1].len() - 1];
        let depth = depths[active];
        let hit = (|| {
            let left = sides[active].path_arena()?;
            let right = sides[opposite].path_arena()?;
            let meeting = if time.stop_reason().is_none() {
                kernel.meet(left, layers[active][depth], right, &layers[opposite], true)?
            } else {
                None
            };
            if let Some(meeting) = meeting {
                local[2] = meeting.right_depth as i64;
                local[3] =
                    (u64::from(meeting.right_row) << 32 | u64::from(meeting.left_row)) as i64;
                Ok(Match::Hit {
                    row: meeting.left_row,
                    label: 0,
                })
            } else {
                Ok(Match::Missing)
            }
        })();
        encode_distributed_match(&mut local, &hit, &time);
        sides[0].gather_path_words(&local, &tx, &rx, &mut gathered)?;
        if let Some(reason) = distributed_stop(&gathered, stride, hit.as_ref().err())? {
            break Receipt::incomplete(reason, None);
        }
        checked_depth = Some((depths[0] + depths[1]) as u64);
        if let Some((owner, winner)) = gathered
            .chunks_exact(stride)
            .enumerate()
            .filter(|(_, row)| row[0] == 1)
            .min_by_key(|(_, row)| row[2])
        {
            let mut witness_depths = depths;
            witness_depths[opposite] =
                usize::try_from(winner[2]).map_err(|_| "NATIVE_PATH_MEET_DEPTH")?;
            if witness_depths[opposite] > depths[opposite] {
                return Err("NATIVE_PATH_MEET_DEPTH".into());
            }
            // Every path of length <= f+r intersects the two visited balls.
            // This is the first committed iteration with an intersection.
            let distance = (witness_depths[0] + witness_depths[1]) as u64;
            let pair = winner[3] as u64;
            let mut meeting_rows = [0; 2];
            meeting_rows[active] = pair as u32;
            meeting_rows[opposite] = (pair >> 32) as u32;
            let mut labels = [Vec::new(), Vec::new()];
            let mut roots = [OwnerRow { owner: 0, row: 0 }; 2];
            for side in 0..2 {
                match trace_parents(
                    sides[side],
                    [&parents, &reverse_parents][side],
                    &layers[side][1..witness_depths[side] + 1],
                    OwnerRow {
                        owner,
                        row: meeting_rows[side],
                    },
                    &tx,
                    &rx,
                    &mut gathered,
                    &mut labels[side],
                    &time,
                )? {
                    Trace::Found(root) => roots[side] = root,
                    Trace::Stopped(reason) => {
                        break 'search Receipt::incomplete(reason, Some(distance))
                    }
                }
            }
            let OwnerRow { owner, row } = roots[1];
            let target = (|| {
                if owner != sides[1].path_rank() {
                    return Ok(Match::Missing);
                }
                let (start, count) = layers[1][0];
                if row < start || row - start >= count {
                    return Err("NATIVE_PATH_REVERSE_ROOT".into());
                }
                if let Some(reason) = time.stop_reason() {
                    return Ok(Match::Stopped(reason));
                }
                let root = kernel.row_values(sides[1].path_arena()?, row)?;
                let label = goals
                    .iter()
                    .position(|goal| *goal == root)
                    .and_then(|index| u32::try_from(index).ok())
                    .ok_or("NATIVE_PATH_REVERSE_ROOT")?;
                Ok(Match::Hit { row, label })
            })();
            encode_distributed_match(&mut local, &target, &time);
            sides[0].gather_path_words(&local[..2], &tx, &rx, &mut gathered[..2 * world])?;
            if let Some(reason) =
                distributed_stop(&gathered[..2 * world], 2, target.as_ref().err())?
            {
                break Receipt::incomplete(reason, Some(distance));
            }
            let reply = &gathered[owner * 2..][..2];
            if reply[0] != 1 {
                return Err("NATIVE_PATH_REVERSE_ROOT".into());
            }
            let [mut prefix, suffix] = labels;
            prefix.reverse();
            prefix.extend(suffix);
            break Receipt::complete(distance, reply[1] as u32, prefix);
        }

        // The busiest rank determines how many synchronized exchange rounds
        // the layer needs. Break ties by total generation work, then depth.
        active = usize::from(
            (sides[1].path_layer_work(), depths[1]) < (sides[0].path_layer_work(), depths[0]),
        );
        match sides[active].advance_nearest([&parents, &reverse_parents][active], &checkpoint) {
            Ok(DistributedAdvance::Layer { .. }) => {
                layers[active].push(range(sides[active].path_arena()?));
            }
            Ok(DistributedAdvance::Complete) => break Receipt::unreachable(),
            Ok(DistributedAdvance::Resource { fatal }) if fatal & !7 == 0 => {
                break Receipt::incomplete("CUDA_MEMORY", None);
            }
            Ok(DistributedAdvance::Resource { fatal }) => {
                return Err(format!("NATIVE_PATH_OWNER_RESOURCE_{fatal}"));
            }
            Err(error) => match query_stop_reason(&error) {
                Some(reason) => break Receipt::incomplete(reason, None),
                None => return Err(error),
            },
        }
    };
    let mut result = serde_json::to_value(&receipt).map_err(|e| e.to_string())?;
    result["native_state_bytes"] = serde_json::json!(sides[0].path_arena()?.plan.state_bytes);
    result["search_confirmed_through_depth"] = serde_json::json!(receipt.d_min.or(checked_depth));
    Ok(DistributedPathResult {
        receipt: result,
        _workspace: (kernel, tx, Some(rx)),
        _parents: Some((parents, Some(reverse_parents))),
    })
}

fn process_peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // Linux reports ru_maxrss in KiB.
    u64::try_from(unsafe { usage.assume_init() }.ru_maxrss)
        .ok()?
        .checked_mul(1024)
}


/// Produce rank-local CSR and terminal-ID files through a proven window
/// bound. Only a depth agreed by all ranks is listed in the receipt.
pub(crate) fn distributed_window_forward(
    bfs: &mut crate::generic_distributed_native::GenericDistributedBfs,
    goals: &[Vec<i64>],
    max_length: u32,
    budget: u64,
    seconds: u64,
    output: &std::path::Path,
) -> Result<DistributedPathResult> {
    use crate::generic_distributed_native::{DistributedAdvance, WindowTile};
    use crate::path_csr::TransitionSpool;
    use std::io::{BufWriter, Write};

    let arena = bfs.path_arena()?;
    let width = arena.plan.elements as usize;
    let mut tile = WindowTile::new(bfs.window_plan());
    if goals.is_empty() || goals.iter().any(|goal| goal.len() != width) {
        return Err("NATIVE_PATH_TARGETS".into());
    }
    let setup = (|| {
        let scan = scanner().ok_or("NATIVE_PATH_CUDA_ABI_UNAVAILABLE")?;
        let buffers = (
            QueryGpu::new(&arena.plan, width * 8, arena.device, scan)?,
            Buffer::new(bfs.window_edge_bytes(), arena.device)?,
        );
        crate::session_cache::generic_requirements_consumed()?;
        Ok::<_, String>(buffers)
    })();
    agree_path_stop(bfs, &setup)?;
    let (kernel, ids) = setup?;
    let local_goals: Vec<_> = goals.iter().enumerate()
        .filter(|(_, goal)| bfs.owns_path_target(goal))
        .collect();
    let time = QueryTime {
        started: Instant::now(),
        seconds: (seconds != 0).then_some(seconds as f64),
        resident: true,
    };
    let mut sizes = Vec::new();
    let mut committed_bytes = 0u64;
    let mut committed_csr_bytes = 0u64;
    let mut peak_bytes = 0u64;
    let mut pending_csr = 0;
    let mut depth = 0;
    let (status, stop_reason) = loop {
        if let Some(reason) = agree_path_stop(bfs, &query_checkpoint(&time))? {
            break ("INCOMPLETE", Some(reason));
        }
        let terminal_path = output.join(format!("{depth}.targets"));
        let local = (|| {
            let arena = bfs.path_arena()?;
            let range = (arena.current_start, arena.frontier_len());
            let mut terminals = BufWriter::new(
                std::fs::File::create(&terminal_path)
                    .map_err(crate::path_csr::io_error)?,
            );
            let mut bytes = 0u64;
            for &(target_id, goal) in &local_goals {
                kernel.targets.upload(goal)?;
                match kernel.find_committed(arena, range, 1, true, &time)? {
                    Match::Hit { row, .. } => {
                        bytes = bytes.checked_add(16).ok_or("PATH_LINEAGE_SIZE")?;
                        if bytes > budget - committed_bytes - pending_csr {
                            return Err("LINEAGE_LIMIT".into());
                        }
                        peak_bytes = peak_bytes.max(committed_bytes + pending_csr + bytes);
                        terminals
                            .write_all(&(target_id as u64).to_le_bytes())
                            .map_err(crate::path_csr::io_error)?;
                        terminals
                            .write_all(&u64::from(row - range.0).to_le_bytes())
                            .map_err(crate::path_csr::io_error)?;
                    }
                    Match::Missing => {}
                    Match::Stopped(reason) => return Err(reason.into()),
                }
            }
            terminals.flush().map_err(crate::path_csr::io_error)?;
            Ok((range.1, bytes))
        })();
        if let Some(reason) = agree_path_stop(bfs, &local)? {
            let _ = std::fs::remove_file(&terminal_path);
            break ("INCOMPLETE", Some(reason));
        }
        let (rows, terminal_bytes) = local?;
        sizes.push(rows);
        committed_csr_bytes += pending_csr;
        committed_bytes += pending_csr + terminal_bytes;
        pending_csr = 0;
        peak_bytes = peak_bytes.max(committed_bytes);
        if depth == max_length {
            break ("COMPLETE", None);
        }
        let next = depth + 1;
        let spool = TransitionSpool::new(
            output.join(format!("{next}.runs")),
            budget - committed_bytes,
        );
        if let Some(reason) = agree_path_stop(bfs, &spool)? {
            break ("INCOMPLETE", Some(reason));
        }
        let mut spool = spool?;
        let advanced = bfs.advance_window(&mut spool, &ids, &mut tile, &|| query_checkpoint(&time));
        peak_bytes = peak_bytes.max(committed_bytes + spool.peak_reserved_bytes());
        match advanced {
            Ok(DistributedAdvance::Complete) => break ("COMPLETE", None),
            Ok(DistributedAdvance::Resource { fatal }) if fatal & !7 == 0 => {
                break ("INCOMPLETE", Some("CUDA_MEMORY"));
            }
            Ok(DistributedAdvance::Resource { fatal }) => {
                return Err(format!("NATIVE_WINDOW_OWNER_RESOURCE_{fatal}"));
            }
            Err(error) if window_stop_reason(&error).is_some() => {
                break ("INCOMPLETE", window_stop_reason(&error));
            }
            Err(error) => return Err(error),
            Ok(DistributedAdvance::Layer { local, .. }) => {
                let csr_path = output.join(format!("{next}.csr"));
                let written = spool.finish(
                    &output.join(format!("{next}.staging")),
                    &csr_path,
                    local,
                    || query_checkpoint(&time),
                );
                peak_bytes = peak_bytes.max(committed_bytes + spool.peak_reserved_bytes());
                drop(spool);
                if let Some(reason) = agree_path_stop(bfs, &written)? {
                    discard_window_csr(&csr_path);
                    break ("INCOMPLETE", Some(reason));
                }
                let (bytes, _) = written?;
                pending_csr = bytes;
            }
        }
        depth = next;
    };
    if pending_csr != 0 {
        discard_window_csr(&output.join(format!("{depth}.csr")));
    }
    let completed_depth = if status == "COMPLETE" {
        Some(u64::from(max_length))
    } else {
        sizes.len().checked_sub(1).map(|depth| depth as u64)
    };
    let receipt = serde_json::json!({
        "status": status, "stop_reason": stop_reason, "layer_rows": sizes,
        "graph_completed_through_depth": completed_depth,
        "native_state_bytes": bfs.path_arena()?.plan.state_bytes,
        "lineage_bytes": committed_bytes, "lineage_csr_bytes": committed_csr_bytes,
        "lineage_peak_bytes": peak_bytes,
        "lineage_peak_scope": "admitted disk bytes including interrupted transition spools",
        "native_process_peak_rss_bytes": process_peak_rss_bytes(),
    });
    Ok(DistributedPathResult {
        receipt,
        _workspace: (kernel, ids, None),
        _parents: None,
    })
}

pub(crate) fn distributed_window_bidirectional(
    forward: &mut crate::generic_distributed_native::GenericDistributedBfs,
    reverse: &mut crate::generic_distributed_native::GenericDistributedBfs,
    goals: &[Vec<i64>],
    min_length: u32,
    max_length: u32,
    budget: u64,
    seconds: u64,
    output: &std::path::Path,
) -> Result<DistributedPathResult> {
    use crate::generic_distributed_native::{DistributedAdvance, WindowTile};
    use crate::path_csr::TransitionSpool;
    use std::io::{BufWriter, Write};

    let arena = forward.path_arena()?;
    let width = arena.plan.elements as usize;
    let mut tile = WindowTile::new(forward.window_plan());
    let setup = (|| {
        let scan = scanner().ok_or("NATIVE_PATH_CUDA_ABI_UNAVAILABLE")?;
        let resolve = resolve_children_kernel()?;
        let buffers = (
            QueryGpu::new(&arena.plan, width * 8, arena.device, scan)?,
            // The owner tile and cut join are sequential and drain before reuse.
            Buffer::new(forward.window_edge_bytes().max(WINDOW_JOIN_ROWS * 4), arena.device)?,
        );
        crate::session_cache::generic_requirements_consumed()?;
        Ok::<_, String>((buffers, resolve))
    })();
    agree_path_stop(forward, &setup)?;
    let ((kernel, ids), resolve) = setup?;
    let mut host_matches = vec![u32::MAX; WINDOW_JOIN_ROWS];
    let sides = [forward, reverse];
    let roots = [output.join("forward"), output.join("reverse")];
    let time = QueryTime {
        started: Instant::now(),
        seconds: (seconds != 0).then_some(seconds as f64),
        resident: true,
    };
    let mut sizes = [Vec::new(), Vec::new()];
    let mut committed_bytes = 0u64;
    let mut csr_bytes = 0u64;
    let mut peak_bytes = 0u64;
    let mut completed = None;
    let mut last_cut = None;
    let setup_roots = (|| {
        for root in &roots {
            std::fs::create_dir(root).map_err(crate::path_csr::io_error)?;
        }
        let arena = sides[1].path_arena()?;
        let range = (arena.current_start, arena.frontier_len());
        let file = std::fs::File::create(roots[1].join("0.targets")).map_err(crate::path_csr::io_error)?;
        let mut writer = BufWriter::new(file);
        let mut bytes = 0u64;
        for (index, goal) in goals.iter().enumerate() {
            kernel.targets.upload(goal)?;
            match kernel.find_committed(arena, range, 1, true, &time)? {
                Match::Hit { row, .. } => {
                    bytes = bytes.checked_add(16).ok_or("PATH_LINEAGE_SIZE")?;
                    if bytes > budget {
                        return Err("LINEAGE_LIMIT".into());
                    }
                    writer
                        .write_all(&(index as u64).to_le_bytes())
                        .map_err(crate::path_csr::io_error)?;
                    writer
                        .write_all(&u64::from(row - range.0).to_le_bytes())
                        .map_err(crate::path_csr::io_error)?;
                }
                Match::Missing => {}
                Match::Stopped(reason) => return Err(reason.into()),
            }
        }
        writer.flush().map_err(crate::path_csr::io_error)?;
        Ok(bytes)
    })();

    let (status, stop_reason) = 'window: {
        if let Some(reason) = agree_path_stop(sides[0], &setup_roots)? {
            break 'window ("INCOMPLETE", Some(reason));
        }
        committed_bytes = setup_roots?;
        peak_bytes = committed_bytes;
        for side in 0..2 {
            sizes[side].push(sides[side].path_arena()?.frontier_len());
        }
        for length in 0..=max_length {
            if let Some(reason) = agree_path_stop(sides[0], &query_checkpoint(&time))? {
                break 'window ("INCOMPLETE", Some(reason));
            }
            if length != 0 {
                // Odd lengths grow reverse; even lengths grow forward.
                let side = (length % 2) as usize;
                let depth = sizes[side].len();
                let root = &roots[side];
                let spool = TransitionSpool::new(
                    root.join(format!("{depth}.runs")),
                    budget - committed_bytes,
                );
                if let Some(reason) = agree_path_stop(sides[0], &spool)? {
                    break 'window ("INCOMPLETE", Some(reason));
                }
                let mut spool = spool?;
                let advanced = sides[side].advance_window(&mut spool, &ids, &mut tile, &|| query_checkpoint(&time));
                peak_bytes = peak_bytes.max(committed_bytes + spool.peak_reserved_bytes());
                match advanced {
                    Ok(DistributedAdvance::Complete) => {
                        completed = Some(max_length);
                        break 'window ("COMPLETE", None);
                    }
                    Ok(DistributedAdvance::Resource { fatal }) if fatal & !7 == 0 => {
                        break 'window ("INCOMPLETE", Some("CUDA_MEMORY"));
                    }
                    Ok(DistributedAdvance::Resource { fatal }) => {
                        return Err(format!("NATIVE_WINDOW_OWNER_RESOURCE_{fatal}"));
                    }
                    Err(error) if window_stop_reason(&error).is_some() => {
                        break 'window ("INCOMPLETE", window_stop_reason(&error));
                    }
                    Err(error) => return Err(error),
                    Ok(DistributedAdvance::Layer { local, .. }) => {
                        let csr_path = root.join(format!("{depth}.csr"));
                        let written = spool.finish(
                            &root.join(format!("{depth}.staging")),
                            &csr_path,
                            local,
                            || query_checkpoint(&time),
                        );
                        peak_bytes = peak_bytes.max(committed_bytes + spool.peak_reserved_bytes());
                        drop(spool);
                        if let Some(reason) = agree_path_stop(sides[0], &written)? {
                            discard_window_csr(&csr_path);
                            break 'window ("INCOMPLETE", Some(reason));
                        }
                        let (bytes, _) = written?;
                        committed_bytes += bytes;
                        csr_bytes += bytes;
                        sizes[side].push(local as u32);
                    }
                }
            }
            if length >= min_length {
                let cut_path = output.join(format!("{length}.matches"));
                let written = write_window_cut(
                    sides[0].path_arena()?,
                    sides[1].path_arena()?,
                    &ids,
                    &mut host_matches,
                    resolve,
                    &time,
                    &cut_path,
                    budget - committed_bytes,
                );
                if let Some(reason) = agree_path_stop(sides[0], &written)? {
                    if let Ok(meta) = cut_path.metadata() {
                        peak_bytes = peak_bytes.max(committed_bytes + meta.len());
                    }
                    let _ = std::fs::remove_file(&cut_path);
                    break 'window ("INCOMPLETE", Some(reason));
                }
                committed_bytes += written?;
                peak_bytes = peak_bytes.max(committed_bytes);
                last_cut = Some(length);
            }
            completed = Some(length);
        }
        ("COMPLETE", None)
    };
    let receipt = serde_json::json!({
        "status": status, "stop_reason": stop_reason,
        "side_layer_rows": sizes, "first_cut_length": min_length, "last_cut_length": last_cut,
        "graph_completed_through_depth": completed, "lineage_bytes": committed_bytes,
        "native_state_bytes": sides[0].path_arena()?.plan.state_bytes,
        "lineage_csr_bytes": csr_bytes, "lineage_peak_bytes": peak_bytes,
        "lineage_peak_scope": "admitted disk bytes including interrupted transition spools",
        "native_process_peak_rss_bytes": process_peak_rss_bytes(),
    });
    Ok(DistributedPathResult {
        receipt,
        _workspace: (kernel, ids, None),
        _parents: None,
    })
}

pub(crate) const WINDOW_JOIN_ROWS: usize = 8192;

/// Join the two current depth layers on their common owner. Window expansion
/// leaves the reverse hash table scoped to its latest committed layer, so the
/// alternating cuts can reuse it without allocating a third table.
fn write_window_cut(
    left: &GenericNativeBfs,
    right: &GenericNativeBfs,
    matches: &Buffer,
    ids: &mut [u32],
    kernel: ResolveChildrenFn,
    time: &QueryTime,
    output: &std::path::Path,
    budget: u64,
) -> Result<u64> {
    use std::io::{BufWriter, Write};

    if left.device != right.device
        || left.plan.elements != right.plan.elements
        || left.plan.state_bytes != right.plan.state_bytes
        || matches.device != left.device
        || matches.bytes < WINDOW_JOIN_ROWS * 4
        || ids.len() < WINDOW_JOIN_ROWS
    {
        return Err("NATIVE_WINDOW_JOIN_LAYOUT".into());
    }
    let left_begin = left.current_start;
    let right_begin = right.current_start;
    let right_end = right_begin
        .checked_add(right.frontier_len())
        .ok_or("NATIVE_WINDOW_JOIN_BOUNDS")?;
    let stream = left.stream.ptr;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(crate::path_csr::io_error)?;
    let mut writer = BufWriter::with_capacity(64 << 10, file);
    let mut bytes = 0u64;
    for offset in (0..left.frontier_len()).step_by(WINDOW_JOIN_ROWS) {
        query_checkpoint(time)?;
        let count = (left.frontier_len() - offset).min(WINDOW_JOIN_ROWS as u32);
        let request = Meet {
            state_bytes: left.plan.state_bytes,
            elements: left.plan.elements,
            hash_bits: right.hash_bits,
            shared_bucket: 1,
            left_begin: left_begin + offset,
            left_count: count,
            right_count: right_end,
            seed: right.seed,
            left_stride: u64::from(left.arena_stride),
            right_stride: u64::from(right.arena_stride),
            slot_count: right.plan.table_slots,
            left: left.visited.ptr,
            right: right.visited.ptr,
            slots: right.slots.ptr.cast(),
        };
        checked!(mgbfs_path_resolve_children_v1, unsafe {
            kernel(
                &request,
                matches.ptr.cast(),
                WINDOW_JOIN_ROWS as u32,
                stream,
            )
        })?;
        checked!(cudaMemcpyAsync, unsafe {
            cudaMemcpyAsync(
                ids.as_mut_ptr().cast(),
                matches.ptr,
                count as usize * 4,
                2,
                stream,
            )
        })?;
        checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        })?;
        for (index, &row) in ids[..count as usize].iter().enumerate() {
            if row == u32::MAX {
                continue;
            }
            if row < right_begin || row >= right_end {
                return Err("NATIVE_WINDOW_JOIN_BOUNDS".into());
            }
            bytes = bytes.checked_add(16).ok_or("PATH_LINEAGE_SIZE")?;
            if bytes > budget {
                return Err("LINEAGE_LIMIT".into());
            }
            let left_row = u64::from(offset) + index as u64;
            let right_row = u64::from(row - right_begin);
            writer
                .write_all(&left_row.to_le_bytes())
                .map_err(crate::path_csr::io_error)?;
            writer
                .write_all(&right_row.to_le_bytes())
                .map_err(crate::path_csr::io_error)?;
        }
    }
    writer.flush().map_err(crate::path_csr::io_error)?;
    Ok(bytes)
}

fn discard_window_csr(path: &std::path::Path) {
    let _ = std::fs::remove_file(path.join("offsets"));
    let _ = std::fs::remove_file(path.join("edges"));
    let _ = std::fs::remove_dir(path);
}

fn query_stop_reason(error: &str) -> Option<&'static str> {
    match error {
        "CANCELLED" => Some("CANCELLED"),
        "TIME_LIMIT" => Some("TIME_LIMIT"),
        _ => None,
    }
}

fn window_stop_reason(error: &str) -> Option<&'static str> {
    match error {
        "LINEAGE_LIMIT" => Some("LINEAGE_LIMIT"),
        "LINEAGE_DISK_FULL" => Some("LINEAGE_DISK_FULL"),
        _ => query_stop_reason(error),
    }
}

fn query_checkpoint(time: &QueryTime) -> Result<()> {
    time.stop_reason()
        .map_or(Ok(()), |reason| Err(reason.into()))
}

pub(crate) fn agree_path_stop<T>(
    bfs: &crate::generic_distributed_native::GenericDistributedBfs,
    local: &Result<T>,
) -> Result<Option<&'static str>> {
    let code = match local.as_ref().err().map(String::as_str) {
        None => 0,
        Some("CANCELLED") => 1,
        Some("TIME_LIMIT") => 2,
        Some("LINEAGE_LIMIT") => 3,
        Some("LINEAGE_DISK_FULL") => 4,
        Some(_) => 5,
    };
    match bfs.path_vote(code)? {
        0 => Ok(None),
        1 => Ok(Some("CANCELLED")),
        2 => Ok(Some("TIME_LIMIT")),
        3 => Ok(Some("LINEAGE_LIMIT")),
        4 => Ok(Some("LINEAGE_DISK_FULL")),
        _ => Err(local
            .as_ref()
            .err()
            .filter(|error| window_stop_reason(error).is_none())
            .cloned()
            .unwrap_or_else(|| "NATIVE_WINDOW_REMOTE_LINEAGE".into())),
    }
}

fn encode_distributed_match(words: &mut [i64], hit: &Result<Match>, time: &QueryTime) {
    let (status, label) = match (hit, time.stop_reason()) {
        (Err(_), _) => (4, 0),
        (Ok(Match::Stopped("TIME_LIMIT")), _) | (_, Some("TIME_LIMIT")) => (3, 0),
        (Ok(Match::Stopped(_)), _) | (_, Some(_)) => (2, 0),
        (Ok(Match::Hit { label, .. }), None) => (1, i64::from(*label)),
        (Ok(Match::Missing), None) => (0, 0),
    };
    words[..2].copy_from_slice(&[status, label]);
}

fn distributed_stop(
    replies: &[i64],
    stride: usize,
    local_error: Option<&String>,
) -> Result<Option<&'static str>> {
    match replies.chunks_exact(stride).map(|r| r[0]).max() {
        Some(4) => Err(local_error
            .cloned()
            .unwrap_or_else(|| "NATIVE_PATH_REMOTE_SCAN".into())),
        Some(3) => Ok(Some("TIME_LIMIT")),
        Some(2) => Ok(Some("CANCELLED")),
        Some(0 | 1) => Ok(None),
        _ => Err("NATIVE_DISTRIBUTED_PATH_REPLY".into()),
    }
}

fn read_query(args: &[String]) -> Result<Query> {
    if args.len() != 1 {
        return Err("CLI_USAGE: mgbfs graph-path-nearest QUERY.json".into());
    }
    let query: Query =
        serde_json::from_reader(std::fs::File::open(&args[0]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    query.graph.validate()?;
    let mut distinct = HashSet::with_capacity(query.targets.len());
    if query.targets.is_empty()
        || u32::try_from(query.targets.len()).is_err()
        || query
            .targets
            .iter()
            .any(|row| row.len() != query.graph.start.len() || !distinct.insert(row))
    {
        return Err("NATIVE_PATH_TARGETS".into());
    }
    if query
        .max_seconds
        .is_some_and(|seconds| !seconds.is_finite() || seconds < 0.0)
    {
        return Err("NATIVE_PATH_MAX_SECONDS".into());
    }
    Ok(query)
}

fn resource_error(error: &str) -> bool {
    error.starts_with("GENERIC_CUDA_2 ")
        || matches!(
            error,
            "GENERIC_VRAM_HEADROOM" | "GENERIC_VRAM_CAPACITY" | "GENERIC_ARENA_OVERFLOW"
        )
}

fn execute(
    query: &Query,
    started: Instant,
    mut resident: Option<&mut ResidentNearest>,
) -> Result<serde_json::Value> {
    let scan = scanner().ok_or("NATIVE_PATH_CUDA_ABI_UNAVAILABLE")?;
    let _signals = Signals::new();
    let time = QueryTime {
        started,
        seconds: query.max_seconds,
        resident: resident.is_some(),
    };
    let mut workspace = None;
    let mut checked_through = None;
    let result = (|| {
        if let Some(reason) = time.stop_reason() {
            return Ok(Receipt::incomplete(reason, None));
        }
        let device = i32::try_from(query.device).map_err(|_| "NATIVE_PATH_DEVICE")?;
        checked!(cudaSetDevice, unsafe { cudaSetDevice(device) })?;
        crate::cuda_loading::verify_driver_before_allocations()?;
        let layout = Layout::new(&query.graph)?;
        let tile = layout.target_tile();
        let plan = if let Some(pool) = resident.as_deref_mut() {
            let (plan, reused) = pool.prepare(layout)?;
            workspace = Some((plan.capacity, reused));
            plan
        } else {
            layout.plan()?
        };
        search(query, &time, scan, plan, tile, &mut checked_through)
    })();
    let receipt = match result {
        Err(error) if resource_error(&error) => {
            if let Some(pool) = resident {
                pool.trim()?;
                workspace = None;
            }
            Receipt::incomplete(time.stop_reason().unwrap_or("CUDA_MEMORY"), None)
        }
        result => result?,
    };
    let mut output = serde_json::to_value(receipt).map_err(|e| e.to_string())?;
    output["search_confirmed_through_depth"] = serde_json::json!(checked_through);
    if let Some((capacity, reused)) = workspace {
        let stats = crate::session_cache::stats();
        output["workspace"] = serde_json::json!({
            "worker_pid": std::process::id(),
            "arena_capacity": capacity,
            "layout_reused": reused,
            "buffer_hits": stats["buffer_hits"],
            "cached_buffers": stats["cached_buffers"],
        });
    }
    Ok(output)
}

pub(crate) fn run_resident(
    args: &[String],
    device: u32,
    state: &mut ResidentNearest,
) -> Result<serde_json::Value> {
    let started = Instant::now();
    let query = read_query(args)?;
    if query.device != device {
        return Err("NATIVE_PATH_SESSION_DEVICE".into());
    }
    execute(&query, started, Some(state))
}

pub fn run(args: &[String]) -> Result<()> {
    if std::env::var("WORLD_SIZE")
        .ok()
        .is_some_and(|value| value != "1")
    {
        return Err("NATIVE_PATH_REQUIRES_ONE_GPU".into());
    }
    let started = Instant::now();
    let query = read_query(args)?;
    crate::native_fault::reset(Some(0), Some(query.device));
    let output = execute(&query, started, None)?;
    println!("{output}");
    Ok(())
}
