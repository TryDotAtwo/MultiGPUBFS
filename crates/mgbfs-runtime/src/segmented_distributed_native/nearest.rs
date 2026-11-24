//! Append-only path history. Logical IDs never embed a vector index or GPU address.

use super::{
    mix, BatchHashRun, BatchHashSegment, Buffer, DistributedAdvance, HashSeriesDirectory, Range,
    Segment, SegmentedDistributedBfs,
};
use crate::generic_path_native::nearest_witness::Receipt;
use mgbfs_core::{graph_definition::GraphDefinitionV2, Result};
use mgbfs_cuda::ffi::{cudaMemcpy, cudaMemcpyAsync, cudaMemsetAsync, cudaStreamSynchronize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{size_of, size_of_val};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

mod bidirectional;
pub(super) use bidirectional::available as bidirectional_abi_ready;
mod roots;
pub(crate) use roots::RootSpec;

const QUERY_BYTES: usize = 256 * 1024;
const NO_ITEM: u64 = u64::MAX;
const RESOURCE: u64 = 3;
const FAILED: u64 = 4;
type Reply = [u64; 4];

extern "C" {
    fn mgbfs_nccl_all_gather_bytes(
        comm: *mut c_void,
        send: *const c_void,
        recv: *mut c_void,
        bytes: u64,
        stream: *mut c_void,
    ) -> i32;
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ParentLink {
    pub(super) parent: u64,
    pub(super) source: u32,
    pub(super) generator: u32,
}

pub(crate) const PATH_PARENT_BYTES: u64 = size_of::<ParentLink>() as u64;
const _: [(); 16] = [(); size_of::<ParentLink>()];

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct TargetHit {
    pub(super) segment: u32,
    pub(super) bank: u32,
    pub(super) row: u32,
    pub(super) found: u32,
}

pub(super) type AcceptParents = unsafe extern "C" fn(
    u32,
    u32,
    u32,
    *const c_void,
    *const c_void,
    *const super::GenericRouteRecord,
    *const super::GenericRouteRecord,
    *const u32,
    *const u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *const u32,
    u32,
    u32,
    *const BatchHashSegment,
    u32,
    u32,
    u32,
    *mut BatchHashRun,
    *mut u32,
    u32,
    *mut u64,
    *mut u32,
    *mut u32,
    *mut u32,
    *mut u64,
    u64,
    *mut c_void,
    usize,
    *mut c_void,
    *mut c_void,
    *mut u64,
    *mut c_void,
    *const HashSeriesDirectory,
    u32,
    *mut ParentLink,
    u32,
    u32,
) -> i32;

pub(super) type FindTargets = unsafe extern "C" fn(
    u32,
    u32,
    *const i64,
    *const u64,
    u32,
    *const BatchHashSegment,
    u32,
    *const BatchHashRun,
    *const u32,
    u32,
    *mut TargetHit,
    *mut u32,
    *mut c_void,
    *const HashSeriesDirectory,
    u32,
) -> i32;

#[derive(Clone, Copy)]
pub(super) struct NearestFns {
    pub(super) accept: AcceptParents,
    pub(super) find: FindTargets,
}

impl NearestFns {
    pub(super) fn load() -> Option<Self> {
        unsafe {
            let accept = libc::dlsym(
                libc::RTLD_DEFAULT,
                b"mgbfs_hash_batch_accept_path_v1\0".as_ptr().cast(),
            );
            let find = libc::dlsym(
                libc::RTLD_DEFAULT,
                b"mgbfs_hash_batch_find_targets_v1\0".as_ptr().cast(),
            );
            if accept.is_null() || find.is_null() {
                return None;
            }
            Some(Self {
                accept: std::mem::transmute(accept),
                find: std::mem::transmute(find),
            })
        }
    }
}

#[derive(Clone, Copy)]
struct Extent {
    logical: u64,
    len: u64,
    physical: u64,
    parents: usize,
    depth: u64,
}

pub(super) struct NearestState {
    pub(super) fns: NearestFns,
    owner_map: Vec<u32>,
    cuts: Option<Vec<u64>>,
    workspace: Option<Workspace>,
    extents: Vec<Extent>,
    index_by_segment: HashMap<usize, Vec<usize>>,
    root_targets: Vec<u32>,
    end: u64,
}

impl NearestState {
    pub(super) fn new(
        fns: NearestFns,
        owner_map: Vec<u32>,
        cuts: Option<Vec<u64>>,
        workspace: Option<Workspace>,
    ) -> Self {
        Self {
            fns,
            owner_map,
            cuts,
            workspace,
            extents: Vec::new(),
            index_by_segment: HashMap::new(),
            root_targets: Vec::new(),
            end: 0,
        }
    }

    pub(super) fn check_layer_size(&self, rows: u64) -> Result<()> {
        self.end
            .checked_add(rows)
            .ok_or_else(|| "SEGMENTED_PATH_ROW_OVERFLOW".into())
            .map(|_| ())
    }

    pub(super) fn publish(&mut self, ranges: &mut [Range], segments: &[Vec<Segment>], depth: u64) {
        // Retained segments with live rows remain allocated even when empty
        // neighbors are reclaimed and descriptor indices change. The parent
        // column's address is stable until this whole arena is retired.
        for range in ranges {
            let segment = &segments[range.shard][range.segment];
            debug_assert_eq!(range.bank, 0);
            debug_assert!(!segment.parents.is_null());
            let parents = segment.parents as usize;
            let len = range.end - range.begin;
            range.logical_start = self.end;
            self.index_by_segment
                .entry(parents)
                .or_default()
                .push(self.extents.len());
            self.extents.push(Extent {
                logical: self.end,
                len,
                physical: range.begin,
                parents,
                depth,
            });
            self.end += len;
        }
    }

    fn logical_row(&self, segment: &Segment, row: u32, depth: u64) -> Result<u64> {
        let ids = self
            .index_by_segment
            .get(&(segment.parents as usize))
            .ok_or("SEGMENTED_PATH_TARGET_SEGMENT")?;
        let row = u64::from(row);
        let position = ids.partition_point(|&index| self.extents[index].physical <= row);
        let index = position.checked_sub(1).ok_or("SEGMENTED_PATH_TARGET_ROW")?;
        let extent = self.extents[ids[index]];
        let offset = row - extent.physical;
        if offset >= extent.len || extent.depth != depth {
            return Err("SEGMENTED_PATH_TARGET_ROW".into());
        }
        Ok(extent.logical + offset)
    }

    pub(super) fn parent_ptr(&self, row: u64, depth: u64) -> Result<*const ParentLink> {
        let position = self.extents.partition_point(|extent| extent.logical <= row);
        let extent = self.extents[position.checked_sub(1).ok_or("SEGMENTED_PATH_PARENT_ROW")?];
        let row = row - extent.logical;
        if depth != extent.depth || row >= extent.len {
            return Err("SEGMENTED_PATH_PARENT_ROW".into());
        }
        let physical =
            usize::try_from(extent.physical + row).map_err(|_| "SEGMENTED_PATH_PARENT_ROW")?;
        Ok((extent.parents as *const ParentLink).wrapping_add(physical))
    }

    fn rank_for(&self, hash: u64) -> u32 {
        let high = hash >> 32;
        let owner = match &self.cuts {
            Some(cuts) => cuts.partition_point(|&cut| cut <= high) - 1,
            None => ((high * self.owner_map.len() as u64) >> 32) as usize,
        };
        self.owner_map[owner]
    }
}

pub(super) fn state_hash(state: &[i64], seed: u64, bits: u32) -> u64 {
    let mut hash = seed;
    for (index, &value) in state.iter().enumerate() {
        hash = mix(hash ^ value as u64 ^ index as u64);
    }
    mix(hash) & u64::MAX.checked_shr(64 - bits).unwrap_or(0)
}

#[derive(Clone, Copy)]
struct Target {
    index: u32,
    hash: u64,
}

struct Targets<'a> {
    states: &'a [Vec<i64>],
    local: Vec<Vec<Target>>,
}

pub(super) struct Workspace {
    coordinates: Buffer,
    hashes: Buffer,
    hits: Buffer,
    error: Buffer,
    tx: Buffer,
    rx: Buffer,
    coordinates_host: Vec<i64>,
    hashes_host: Vec<u64>,
    hits_host: Vec<TargetHit>,
    replies: Vec<Reply>,
}

impl Workspace {
    pub(super) fn new(elements: u32, world: u32, device: i32) -> Result<Self> {
        let state_bytes = (elements as usize)
            .checked_mul(size_of::<i64>())
            .filter(|&bytes| bytes > 0)
            .ok_or("SEGMENTED_NEAREST_TARGET_BYTES")?;
        let tile = (QUERY_BYTES / (state_bytes + size_of::<u64>() + size_of::<TargetHit>())).max(1);
        Ok(Self {
            coordinates: Buffer::new(tile * state_bytes, device)?,
            hashes: Buffer::new(tile * size_of::<u64>(), device)?,
            hits: Buffer::new(tile * size_of::<TargetHit>(), device)?,
            error: Buffer::new(size_of::<u32>(), device)?,
            tx: Buffer::new(size_of::<Reply>(), device)?,
            rx: Buffer::new(size_of::<Reply>() * world as usize, device)?,
            coordinates_host: Vec::with_capacity(tile * elements as usize),
            hashes_host: Vec::with_capacity(tile),
            hits_host: vec![TargetHit::default(); tile],
            replies: vec![[0; 4]; world as usize],
        })
    }
}

fn stopped(code: u64) -> Option<&'static str> {
    match code {
        1 => Some("CANCELLED"),
        2 => Some("TIME_LIMIT"),
        _ => None,
    }
}

impl SegmentedDistributedBfs {
    pub(crate) fn solve_nearest(
        &mut self,
        graph: &GraphDefinitionV2,
        targets: &[Vec<i64>],
        max_depth: Option<u32>,
        max_seconds: u64,
    ) -> Result<Value> {
        let state = self
            .nearest
            .as_mut()
            .ok_or("SEGMENTED_NEAREST_UNAVAILABLE")?;
        let mut workspace = state
            .workspace
            .take()
            .ok_or("SEGMENTED_NEAREST_ALREADY_RUN")?;
        let start = Instant::now();
        let stop = || {
            if crate::generic_run::CANCELLED.load(Ordering::Relaxed)
                || crate::job_phase::cancelled()
            {
                1
            } else if max_seconds != 0 && start.elapsed() >= Duration::from_secs(max_seconds) {
                2
            } else {
                0
            }
        };
        let root = self
            .counts
            .iter()
            .position(|&rows| rows == 1)
            .filter(|_| self.depth == 0)
            .ok_or("SEGMENTED_NEAREST_ROOT")?;
        let mut local = vec![Vec::new(); self.plan.shards as usize];
        for (index, target) in targets.iter().enumerate() {
            if target.len() != graph.start.len() {
                return Err("SEGMENTED_NEAREST_TARGET_SHAPE".into());
            }
            let index = u32::try_from(index).map_err(|_| "SEGMENTED_NEAREST_TARGET_COUNT")?;
            let hash = state_hash(target, self.seed, self.bits);
            if state.rank_for(hash) == self.rank {
                let shard = ((u64::from(hash as u32) * u64::from(self.plan.shards)) >> 32) as usize;
                local[shard].push(Target { index, hash });
            }
        }
        let targets = Targets {
            states: targets,
            local,
        };
        let (receipt, checked) =
            self.search_nearest(&targets, max_depth, root, &stop, &mut workspace)?;
        let mut output = serde_json::to_value(receipt).map_err(|e| e.to_string())?;
        output["search_confirmed_through_depth"] = json!(checked);
        output["native_exchange"] = json!("COMPACT_PACKET");
        output["native_storage"] = json!("SEGMENTED_BATCH_HASH");
        output["native_state_bytes"] = json!(self.plan.state_bytes);
        self.nearest.as_mut().unwrap().workspace = Some(workspace);
        Ok(output)
    }

    fn search_nearest(
        &mut self,
        targets: &Targets<'_>,
        max_depth: Option<u32>,
        root: usize,
        stop: &impl Fn() -> u32,
        workspace: &mut Workspace,
    ) -> Result<(Receipt, Option<u64>)> {
        let mut checked = None;
        let mut boundary_stop = self.collective_stop(stop())?;
        loop {
            if let Some(reason) = stopped(u64::from(boundary_stop)) {
                return Ok((Receipt::incomplete(reason, None), checked));
            }

            let local = self.find_local_target(targets, workspace, stop);
            let words = match &local {
                Ok(LocalTarget::Hit(index, row)) => [u64::from(stop()), u64::from(*index), *row, 0],
                Ok(LocalTarget::Miss) => [u64::from(stop()), NO_ITEM, 0, 0],
                Ok(LocalTarget::Stopped(code)) => [*code, NO_ITEM, 0, 1],
                Err(_) => [FAILED, NO_ITEM, 0, 0],
            };
            let code = self.nearest_exchange(words, workspace)?;
            if code == FAILED {
                return Err(local
                    .err()
                    .unwrap_or_else(|| "SEGMENTED_NEAREST_PEER_LOOKUP".into()));
            }
            if code == RESOURCE {
                return Ok((Receipt::incomplete("CUDA_MEMORY", None), checked));
            }
            let incomplete_probe = workspace.replies.iter().any(|r| r[3] != 0);
            if let Some(reason) = stopped(code) {
                if !incomplete_probe {
                    checked = Some(self.depth);
                }
                let d_min = (!incomplete_probe
                    && workspace.replies.iter().any(|r| r[1] != NO_ITEM))
                .then_some(self.depth);
                return Ok((Receipt::incomplete(reason, d_min), checked));
            }

            checked = Some(self.depth);
            if let Some((owner, reply)) = workspace
                .replies
                .iter()
                .enumerate()
                .filter(|(_, reply)| reply[1] != NO_ITEM)
                .min_by_key(|&(rank, reply)| (reply[1], rank))
            {
                let index =
                    u32::try_from(reply[1]).map_err(|_| "SEGMENTED_NEAREST_TARGET_INDEX")?;
                let receipt = self.trace_nearest(owner, reply[2], index, root, stop, workspace)?;
                return Ok((receipt, checked));
            }
            if max_depth.is_some_and(|limit| self.depth >= u64::from(limit)) {
                return Ok((Receipt::incomplete("DEPTH_LIMIT", None), checked));
            }

            crate::native_fault::layer(self.depth + 1, self.depth);
            match self.advance_with_stop(stop, &mut boundary_stop)? {
                DistributedAdvance::Layer { .. } => crate::native_fault::completed(self.depth),
                DistributedAdvance::Complete => {
                    return Ok((Receipt::unreachable(), checked));
                }
                DistributedAdvance::Resource { .. } => {
                    let reason = stopped(u64::from(boundary_stop)).unwrap_or("CUDA_MEMORY");
                    return Ok((Receipt::incomplete(reason, None), checked));
                }
            }
        }
    }

    fn find_local_target(
        &mut self,
        targets: &Targets<'_>,
        workspace: &mut Workspace,
        stop: &impl Fn() -> u32,
    ) -> Result<LocalTarget> {
        let limit = self
            .hash_lookup_limit
            .ok_or("SEGMENTED_NEAREST_LOOKUP_BOUND")?;
        let find = self.nearest.as_ref().unwrap().fns.find;
        let mut best: Option<(u32, u64)> = None;
        for (shard, local) in targets.local.iter().enumerate() {
            if local.is_empty() || self.segments[shard].is_empty() {
                continue;
            }
            let stop_code = stop();
            if stop_code != 0 {
                return Ok(LocalTarget::Stopped(u64::from(stop_code)));
            }
            let Some(view) = self.hash_view(shard, false)? else {
                return Ok(LocalTarget::Stopped(RESOURCE));
            };
            if self.hash_shards[shard]
                .as_ref()
                .is_some_and(|item| item.searched_series() > limit)
            {
                self.refresh_hash_series_snapshots(2, Some(shard))?;
            }
            let catalog = self.hash_shards[shard]
                .as_ref()
                .ok_or("SEGMENTED_NEAREST_LOOKUP_MISSING")?;
            if catalog.searched_series() > limit {
                return Ok(LocalTarget::Stopped(RESOURCE));
            }
            let directory = catalog.snapshot_directory();
            let stream = self.base.stream.ptr;
            for tile in local.chunks(workspace.hits_host.len()) {
                if best.is_some_and(|(index, _)| tile[0].index >= index) {
                    break;
                }
                let stop_code = stop();
                if stop_code != 0 {
                    return Ok(LocalTarget::Stopped(u64::from(stop_code)));
                }
                workspace.coordinates_host.clear();
                workspace.hashes_host.clear();
                for target in tile {
                    workspace
                        .coordinates_host
                        .extend_from_slice(&targets.states[target.index as usize]);
                    workspace.hashes_host.push(target.hash);
                }
                workspace.coordinates.upload(&workspace.coordinates_host)?;
                workspace.hashes.upload(&workspace.hashes_host)?;
                let mut error = 0u32;
                let queued = (|| -> Result<()> {
                    crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                        cudaMemsetAsync(workspace.error.ptr, 0, size_of::<u32>(), stream)
                    })?;
                    crate::generic_native::checked!(mgbfs_hash_batch_find_targets_v1, unsafe {
                        find(
                            self.plan.state_bytes,
                            self.plan.elements,
                            workspace.coordinates.ptr.cast(),
                            workspace.hashes.ptr.cast(),
                            tile.len() as u32,
                            view.hash_segments,
                            view.hash_count,
                            view.hash_runs,
                            view.hash_run_count,
                            view.hash_run_capacity,
                            workspace.hits.ptr.cast(),
                            workspace.error.ptr.cast(),
                            stream,
                            &directory,
                            limit,
                        )
                    })?;
                    crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                        cudaMemcpyAsync(
                            (&mut error as *mut u32).cast(),
                            workspace.error.ptr,
                            size_of::<u32>(),
                            2,
                            stream,
                        )
                    })?;
                    crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                        cudaMemcpyAsync(
                            workspace.hits_host.as_mut_ptr().cast(),
                            workspace.hits.ptr,
                            tile.len() * size_of::<TargetHit>(),
                            2,
                            stream,
                        )
                    })
                })();
                let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                    cudaStreamSynchronize(stream)
                });
                queued.and(drained)?;
                if error != 0 {
                    return Err(format!("SEGMENTED_NEAREST_LOOKUP_{error}"));
                }
                for (target, hit) in tile.iter().zip(&workspace.hits_host) {
                    if hit.found == 0 {
                        continue;
                    }
                    if hit.found != 1 || hit.bank != 0 {
                        return Err("SEGMENTED_NEAREST_TARGET_BANK".into());
                    }
                    let segment = self.segments[shard]
                        .get(hit.segment as usize)
                        .ok_or("SEGMENTED_NEAREST_TARGET_SEGMENT")?;
                    let row = self
                        .nearest
                        .as_ref()
                        .unwrap()
                        .logical_row(segment, hit.row, self.depth)?;
                    if best.map_or(true, |(index, _)| target.index < index) {
                        best = Some((target.index, row));
                    }
                    break;
                }
            }
        }
        Ok(best.map_or(LocalTarget::Miss, |(index, row)| {
            LocalTarget::Hit(index, row)
        }))
    }

    fn nearest_exchange(&mut self, words: Reply, workspace: &mut Workspace) -> Result<u64> {
        if self.plan.world == 1 {
            workspace.replies[0] = words;
            return Ok(words[0]);
        }
        let stream = self.base.stream.ptr;
        let queued = (|| -> Result<()> {
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(
                    workspace.tx.ptr,
                    words.as_ptr().cast(),
                    size_of::<Reply>(),
                    1,
                    stream,
                )
            })?;
            crate::generic_native::checked!(mgbfs_nccl_all_gather_bytes, unsafe {
                mgbfs_nccl_all_gather_bytes(
                    self.comm.ptr,
                    workspace.tx.ptr,
                    workspace.rx.ptr,
                    size_of::<Reply>() as u64,
                    stream,
                )
            })?;
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(
                    workspace.replies.as_mut_ptr().cast(),
                    workspace.rx.ptr,
                    size_of_val(workspace.replies.as_slice()),
                    2,
                    stream,
                )
            })
        })();
        let drained = self.wait_boundary_stream();
        queued.and(drained)?;
        Ok(workspace
            .replies
            .iter()
            .map(|reply| reply[0])
            .max()
            .unwrap_or(0))
    }

    fn trace_nearest(
        &mut self,
        owner: usize,
        row: u64,
        target: u32,
        root: usize,
        stop: &impl Fn() -> u32,
        workspace: &mut Workspace,
    ) -> Result<Receipt> {
        match self.trace_parent_chain(owner, row, stop, workspace)? {
            Chain::Stopped(reason) => Ok(Receipt::incomplete(reason, Some(self.depth))),
            Chain::Root {
                owner,
                row: 0,
                mut labels,
            } if owner == root => {
                labels.reverse();
                Ok(Receipt::complete(self.depth, target, labels))
            }
            Chain::Root { .. } => Err("SEGMENTED_NEAREST_WITNESS_ROOT".into()),
        }
    }

    fn trace_parent_chain(
        &mut self,
        mut owner: usize,
        mut row: u64,
        stop: &impl Fn() -> u32,
        workspace: &mut Workspace,
    ) -> Result<Chain> {
        let mut labels = Vec::new();
        for depth in (1..=self.depth).rev() {
            let local_stop = stop();
            let parent: Result<Option<ParentLink>> = (|| {
                if owner != self.rank as usize || local_stop != 0 {
                    return Ok(None);
                }
                let source = self.nearest.as_ref().unwrap().parent_ptr(row, depth)?;
                let mut link = ParentLink::default();
                crate::generic_native::checked!(cudaMemcpy, unsafe {
                    cudaMemcpy(
                        (&mut link as *mut ParentLink).cast(),
                        source.cast(),
                        size_of::<ParentLink>(),
                        2,
                    )
                })?;
                if link.source >= self.plan.world || link.generator >= self.base.generators {
                    return Err("SEGMENTED_NEAREST_PARENT_LINK".into());
                }
                Ok(Some(link))
            })();
            let words = match &parent {
                Ok(Some(link)) => [
                    u64::from(local_stop),
                    link.parent,
                    u64::from(link.source),
                    u64::from(link.generator),
                ],
                Ok(None) => [u64::from(local_stop), 0, NO_ITEM, 0],
                Err(_) => [FAILED, 0, NO_ITEM, 0],
            };
            let code = self.nearest_exchange(words, workspace)?;
            if code == FAILED {
                return Err(parent
                    .err()
                    .unwrap_or_else(|| "SEGMENTED_NEAREST_PEER_PARENT".into()));
            }
            if let Some(reason) = stopped(code) {
                return Ok(Chain::Stopped(reason));
            }
            let [_, parent, source, label] = workspace.replies[owner];
            if source >= u64::from(self.plan.world) || label >= u64::from(self.base.generators) {
                return Err("SEGMENTED_NEAREST_PARENT_LINK".into());
            }
            row = parent;
            owner = source as usize;
            labels.push(label as u32);
        }
        Ok(Chain::Root { owner, row, labels })
    }
}

enum Chain {
    Root {
        owner: usize,
        row: u64,
        labels: Vec<u32>,
    },
    Stopped(&'static str),
}

enum LocalTarget {
    Hit(u32, u64),
    Miss,
    Stopped(u64),
}
