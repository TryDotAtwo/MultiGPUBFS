//! Layer-synchronous distributed search over two append-only owner histories.

use super::{
    stopped, Buffer, Chain, DistributedAdvance, HashSeriesDirectory, Workspace, FAILED, NO_ITEM,
    RESOURCE,
};
use crate::generic_path_native::nearest_witness::Receipt;
use crate::segmented_distributed_native::{
    BatchHashRun, BatchHashSegment, SegmentedDistributedBfs,
};
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{cudaMemcpyAsync, cudaMemsetAsync, cudaStreamSynchronize};
use serde_json::{json, Value};
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const PROBE_ROWS: u64 = 1 << 20;
const PROBE_LAUNCH_WINDOW: usize = 8;

#[repr(C)]
#[derive(Default)]
struct Meeting {
    source_range: u32,
    source_row: u32,
    opposite_segment: u32,
    opposite_bank: u32,
    opposite_row: u32,
    found: u32,
}

type MeetFrontier = unsafe extern "C" fn(
    u32,
    u32,
    *const c_void,
    u64,
    u64,
    u64,
    u32,
    u64,
    u32,
    *const BatchHashSegment,
    u32,
    *const BatchHashRun,
    *const u32,
    u32,
    *mut Meeting,
    *mut u32,
    *mut c_void,
    *const HashSeriesDirectory,
    u32,
) -> i32;

fn meeting_kernel() -> Option<MeetFrontier> {
    unsafe {
        let symbol = libc::dlsym(
            libc::RTLD_DEFAULT,
            b"mgbfs_hash_batch_meet_frontier_v1\0".as_ptr().cast(),
        );
        (!symbol.is_null()).then(|| std::mem::transmute::<*mut c_void, MeetFrontier>(symbol))
    }
}

pub(crate) fn available() -> bool {
    meeting_kernel().is_some()
}

enum LocalMeeting {
    Found { source: u64, opposite: u64 },
    Missing,
    Stopped(u64),
}

impl SegmentedDistributedBfs {
    pub(crate) fn solve_nearest_bidirectional(
        forward: &mut Self,
        reverse: &mut Self,
        max_depth: Option<u32>,
        max_seconds: u64,
    ) -> Result<Value> {
        let started = Instant::now();
        let stop = || {
            if crate::generic_run::CANCELLED.load(Ordering::Relaxed)
                || crate::job_phase::cancelled()
            {
                1
            } else if max_seconds != 0 && started.elapsed() >= Duration::from_secs(max_seconds) {
                2
            } else {
                0
            }
        };
        let setup = (|| {
            if forward.packet.is_none() || reverse.packet.is_some() {
                return Err("SEGMENTED_MEET_PACKET_OWNERSHIP".into());
            }
            let kernel = meeting_kernel().ok_or("SEGMENTED_MEET_CUDA_ABI_UNAVAILABLE")?;
            if reverse.root_counts().iter().all(|&count| count == 0) {
                return Err("SEGMENTED_MEET_REVERSE_ROOT".into());
            }
            let root = forward
                .root_counts()
                .iter()
                .position(|&count| count == 1)
                .ok_or("SEGMENTED_NEAREST_ROOT")?;
            let hit = Buffer::new(size_of::<Meeting>(), forward.base.device)?;
            let error = Buffer::new(size_of::<u32>(), forward.base.device)?;
            let workspace = forward
                .nearest
                .as_mut()
                .and_then(|state| state.workspace.take())
                .ok_or("SEGMENTED_NEAREST_ALREADY_RUN")?;
            Ok::<_, String>((root, kernel, workspace, hit, error))
        })();
        if forward.collective_stop(u32::from(setup.is_err()))? != 0 {
            return Err(setup
                .err()
                .unwrap_or_else(|| "SEGMENTED_MEET_PEER_SETUP".into()));
        }
        let (root, kernel, mut workspace, hit, error) = setup?;
        forward.connect_packet_control()?;

        let mut checked = None;
        let mut active = 0;
        let receipt = loop {
            let depths = [forward.depth, reverse.depth];
            let depth = depths[0]
                .checked_add(depths[1])
                .ok_or("SEGMENTED_MEET_DEPTH_OVERFLOW")?;
            let local = if active == 0 {
                forward.find_local_meeting(reverse, kernel, &hit, &error, &stop)
            } else {
                reverse.find_local_meeting(forward, kernel, &hit, &error, &stop)
            };
            let words = match &local {
                Ok(LocalMeeting::Found { source, opposite }) if active == 0 => {
                    [u64::from(stop()), *source, *opposite, 1]
                }
                Ok(LocalMeeting::Found { source, opposite }) => {
                    [u64::from(stop()), *opposite, *source, 1]
                }
                Ok(LocalMeeting::Missing) => [u64::from(stop()), 0, 0, 0],
                Ok(LocalMeeting::Stopped(code)) => [*code, 0, 0, 0],
                Err(_) => [FAILED, 0, 0, 0],
            };
            match forward.nearest_exchange(words, &mut workspace)? {
                FAILED => {
                    return Err(local
                        .err()
                        .unwrap_or_else(|| "SEGMENTED_MEET_PEER_LOOKUP".into()))
                }
                RESOURCE => break Receipt::incomplete("CUDA_MEMORY", None),
                code => {
                    if let Some(reason) = stopped(code) {
                        break Receipt::incomplete(reason, None);
                    }
                }
            }

            checked = Some(depth);
            if let Some((owner, &[_, left, right, _])) = workspace
                .replies
                .iter()
                .enumerate()
                .find(|(_, reply)| reply[3] == 1)
            {
                // Every path of length <= f + r meets the committed balls.
                // Their first intersection therefore lies at this depth sum.
                break Self::trace_meeting(
                    forward,
                    reverse,
                    &mut workspace,
                    root,
                    owner,
                    [left, right],
                    depth,
                    &stop,
                )?;
            }
            if max_depth.is_some_and(|limit| depth >= u64::from(limit)) {
                break Receipt::incomplete("DEPTH_LIMIT", None);
            }

            // The busiest rank accounts for the slowest producer on that side.
            let work = |side: &Self| {
                (
                    side.counts.iter().copied().max().unwrap_or(0),
                    side.global_frontier(),
                    side.depth,
                )
            };
            active = usize::from(work(reverse) < work(forward));
            let mut boundary_stop = 0;
            let next_depth = depth
                .checked_add(1)
                .ok_or("SEGMENTED_MEET_DEPTH_OVERFLOW")?;
            crate::native_fault::layer(next_depth, depth);
            let advance = if active == 0 {
                forward.advance_with_stop(&stop, &mut boundary_stop)
            } else {
                Self::with_reverse_packet(forward, reverse, |side| {
                    side.advance_with_stop(&stop, &mut boundary_stop)
                })
            }?;
            match advance {
                DistributedAdvance::Layer { .. } => {
                    crate::native_fault::completed(next_depth);
                    if let Some(reason) = stopped(u64::from(boundary_stop)) {
                        break Receipt::incomplete(reason, None);
                    }
                }
                DistributedAdvance::Complete => break Receipt::unreachable(),
                DistributedAdvance::Resource { .. } => {
                    break Receipt::incomplete(
                        stopped(u64::from(boundary_stop)).unwrap_or("CUDA_MEMORY"),
                        None,
                    );
                }
            }
        };

        let mut report = serde_json::to_value(receipt).map_err(|e| e.to_string())?;
        report["search_confirmed_through_depth"] = json!(checked);
        report["native_exchange"] = json!("COMPACT_PACKET");
        report["native_storage"] = json!("SEGMENTED_BATCH_HASH");
        report["native_state_bytes"] = json!(forward.plan.state_bytes);
        report["native_search_depths"] = json!([forward.depth, reverse.depth]);
        forward.nearest.as_mut().unwrap().workspace = Some(workspace);
        Ok(report)
    }

    fn with_reverse_packet<T>(
        forward: &mut Self,
        reverse: &mut Self,
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        std::mem::swap(&mut forward.packet, &mut reverse.packet);
        let result = run(reverse);
        std::mem::swap(&mut forward.packet, &mut reverse.packet);
        result
    }

    fn find_local_meeting(
        &mut self,
        opposite: &mut Self,
        kernel: MeetFrontier,
        hit: &Buffer,
        error: &Buffer,
        stop: &impl Fn() -> u32,
    ) -> Result<LocalMeeting> {
        let stopped = stop();
        if stopped != 0 {
            return Ok(LocalMeeting::Stopped(u64::from(stopped)));
        }
        let limit = opposite
            .hash_lookup_limit
            .ok_or("SEGMENTED_MEET_LOOKUP_BOUND")?;
        let stream = self.base.stream.ptr;
        let mut first = 0;
        while first < self.current.len() {
            let shard = self.current[first].shard;
            let past = first + self.current[first..].partition_point(|range| range.shard == shard);
            if opposite.segments[shard].is_empty() {
                first = past;
                continue;
            }
            let Some(view) = opposite.hash_view(shard, false)? else {
                return Ok(LocalMeeting::Stopped(RESOURCE));
            };
            if opposite.hash_shards[shard]
                .as_ref()
                .is_some_and(|owner| owner.searched_series() > limit)
            {
                opposite.refresh_hash_series_snapshots(2, Some(shard))?;
            }
            let catalog = opposite.hash_shards[shard]
                .as_ref()
                .ok_or("SEGMENTED_MEET_LOOKUP_MISSING")?;
            if catalog.searched_series() > limit {
                return Ok(LocalMeeting::Stopped(RESOURCE));
            }
            let directory = catalog.snapshot_directory();
            let mut result = Meeting::default();
            let mut status = 0u32;
            let queued = (|| {
                crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                    cudaMemsetAsync(hit.ptr, 0, size_of::<Meeting>(), stream)
                })?;
                crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                    cudaMemsetAsync(error.ptr, 0, size_of::<u32>(), stream)
                })?;
                let mut launches = 0;
                for index in first..past {
                    let range = self.current[index];
                    let source = &self.segments[shard][range.segment];
                    if range.bank != 0 || source.desc.banks != 1 {
                        return Err("SEGMENTED_MEET_BANK".into());
                    }
                    let tag = u32::try_from(index).map_err(|_| "SEGMENTED_MEET_RANGE_ID")?;
                    let mut begin = range.begin;
                    while begin < range.end {
                        let code = stop();
                        if code != 0 {
                            return Ok(u64::from(code));
                        }
                        let end = (begin + PROBE_ROWS).min(range.end);
                        crate::generic_native::checked!(
                            mgbfs_hash_batch_meet_frontier_v1,
                            unsafe {
                                kernel(
                                    self.plan.state_bytes,
                                    self.plan.elements,
                                    source.desc.states,
                                    source.desc.capacity,
                                    begin,
                                    end,
                                    tag,
                                    self.seed,
                                    self.bits,
                                    view.hash_segments,
                                    view.hash_count,
                                    view.hash_runs,
                                    view.hash_run_count,
                                    view.hash_run_capacity,
                                    hit.ptr.cast(),
                                    error.ptr.cast(),
                                    stream,
                                    &directory,
                                    limit,
                                )
                            }
                        )?;
                        begin = end;
                        launches += 1;
                        if launches == PROBE_LAUNCH_WINDOW {
                            crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                                cudaStreamSynchronize(stream)
                            })?;
                            launches = 0;
                        }
                    }
                }
                crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                    cudaMemcpyAsync(
                        (&mut status as *mut u32).cast(),
                        error.ptr,
                        size_of::<u32>(),
                        2,
                        stream,
                    )
                })?;
                crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                    cudaMemcpyAsync(
                        (&mut result as *mut Meeting).cast(),
                        hit.ptr,
                        size_of::<Meeting>(),
                        2,
                        stream,
                    )
                })?;
                Ok::<_, String>(0)
            })();
            let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                cudaStreamSynchronize(stream)
            });
            let code = queued?;
            drained?;
            if code != 0 {
                return Ok(LocalMeeting::Stopped(code));
            }
            if status != 0 {
                return Err(format!("SEGMENTED_MEET_LOOKUP_{status}"));
            }
            if result.found == 0 {
                first = past;
                continue;
            }
            let index = result.source_range as usize;
            if result.found != 1 || result.opposite_bank != 0 || !(first..past).contains(&index) {
                return Err("SEGMENTED_MEET_REFERENCE".into());
            }
            let range = self.current[index];
            let row = u64::from(result.source_row);
            if !(range.begin..range.end).contains(&row) {
                return Err("SEGMENTED_MEET_SOURCE_ROW".into());
            }
            let segment = opposite.segments[shard]
                .get(result.opposite_segment as usize)
                .ok_or("SEGMENTED_MEET_OPPOSITE_SEGMENT")?;
            let other = opposite.nearest.as_ref().unwrap().logical_row(
                segment,
                result.opposite_row,
                opposite.depth,
            )?;
            return Ok(LocalMeeting::Found {
                source: range.logical_start + row - range.begin,
                opposite: other,
            });
        }
        Ok(LocalMeeting::Missing)
    }

    fn trace_meeting(
        forward: &mut Self,
        reverse: &mut Self,
        workspace: &mut Workspace,
        forward_root: usize,
        meeting_owner: usize,
        [left, right]: [u64; 2],
        depth: u64,
        stop: &impl Fn() -> u32,
    ) -> Result<Receipt> {
        let mut prefix = match forward.trace_parent_chain(meeting_owner, left, stop, workspace)? {
            Chain::Stopped(reason) => return Ok(Receipt::incomplete(reason, Some(depth))),
            Chain::Root {
                owner,
                row: 0,
                labels,
            } if owner == forward_root => labels,
            Chain::Root { .. } => return Err("SEGMENTED_MEET_FORWARD_ROOT".into()),
        };
        let (root, row, suffix) = match Self::with_reverse_packet(forward, reverse, |side| {
            side.trace_parent_chain(meeting_owner, right, stop, workspace)
        })? {
            Chain::Stopped(reason) => return Ok(Receipt::incomplete(reason, Some(depth))),
            Chain::Root { owner, row, labels } => (owner, row, labels),
        };
        let stopped_here = stop();
        let target = if root == reverse.rank as usize && stopped_here == 0 {
            reverse.root_target(row).map(|target| u64::from(target))
        } else {
            Ok(NO_ITEM)
        };
        let words = match &target {
            Ok(target) => [u64::from(stopped_here), *target, 0, 0],
            Err(_) => [FAILED, NO_ITEM, 0, 0],
        };
        let code = forward.nearest_exchange(words, workspace)?;
        if code == FAILED {
            return Err(target
                .err()
                .unwrap_or_else(|| "SEGMENTED_MEET_REVERSE_ROOT".into()));
        }
        if let Some(reason) = stopped(code) {
            return Ok(Receipt::incomplete(reason, Some(depth)));
        }
        let target =
            u32::try_from(workspace.replies[root][1]).map_err(|_| "SEGMENTED_MEET_TARGET_ID")?;
        prefix.reverse();
        prefix.extend(suffix);
        Ok(Receipt::complete(depth, target, prefix))
    }
}
