//! Sorted Batch hash run directories and exact snapshot coverage.

use super::{
    BatchHashRun, BatchHashSegment, BatchHashShard, DescriptorCache, HashSeriesSnapshot,
    HashSeriesView, OwnerPrepared, RawBuffer, Segment, SegmentedDistributedBfs,
};
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{cudaMemcpy, cudaMemcpyAsync, cudaMemsetAsync, cudaStreamSynchronize};
use std::{ffi::c_void, ptr};

pub(super) fn initial_hash_run_capacity() -> Result<u32> {
    let capacity = std::env::var("MGBFS_SEGMENT_TEST_HASH_RUNS")
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|_| "SEGMENTED_HASH_RUN_CAP".to_owned())
        })
        .transpose()?
        .unwrap_or(8);
    if capacity == 0 {
        return Err("SEGMENTED_HASH_RUN_CAP".into());
    }
    Ok(capacity)
}

pub(super) fn hash_directory_bytes(capacity: u32) -> Result<usize> {
    (capacity as usize)
        .checked_mul(std::mem::size_of::<BatchHashRun>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u32>()))
        .ok_or_else(|| "SEGMENTED_HASH_DIRECTORY_SIZE".into())
}
extern "C" {
    fn mgbfs_hash_extent_bank_summary_v1(
        runs: *const BatchHashRun,
        count: u32,
        result: *mut u64,
        stream: *mut c_void,
    ) -> i32;
    fn mgbfs_hash_extent_gather_v1(
        segments: *const BatchHashSegment,
        segment_count: u32,
        runs: *const BatchHashRun,
        run_count: u32,
        rows: u32,
        run_offsets: *mut u32,
        keys: *mut u64,
        refs: *mut c_void,
        ref_bytes: u32,
        row_bits: u32,
        error: *mut u32,
        stream: *mut c_void,
    ) -> i32;
    fn mgbfs_hash_extent_sort_scratch_v1(
        state_bytes: u32,
        ref_bytes: u32,
        rows: u32,
        bytes: *mut usize,
    ) -> i32;
    fn mgbfs_hash_extent_sort_v1(
        state_bytes: u32,
        elements: u32,
        segments: *const BatchHashSegment,
        rows: u32,
        keys: *const u64,
        refs: *const c_void,
        ref_bytes: u32,
        row_bits: u32,
        out_keys: *mut u64,
        out_refs: *mut c_void,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    fn mgbfs_hash_extent_prefixes_v1(
        keys: *const u64,
        count: u32,
        offsets: *mut u32,
        stream: *mut c_void,
    ) -> i32;
}

impl SegmentedDistributedBfs {
    // Every caller is between drained owner waves. Directory replacement is
    // transactional: old and new coexist until the copy finishes. The input queue
    // and filtered spill remain leased until a failed tile is replayed.
    // Raw runs remain complete after eviction. Keep an index resident if
    // removing it would exceed the lookup bound and host migration is unavailable.
    pub(super) fn reclaim_snapshots_for_storage(
        &mut self,
        requested: u64,
        free: u64,
    ) -> Result<bool> {
        let available = self
            .remaining_bytes
            .min(free.saturating_sub(self.reserve_bytes));
        if requested <= available {
            return Ok(false);
        }
        let needed = requested.saturating_sub(self.remaining_bytes).max(
            requested
                .saturating_add(self.reserve_bytes)
                .saturating_sub(free),
        );
        let mut candidates = self
            .hash_shards
            .iter()
            .enumerate()
            .filter_map(|(shard, item)| {
                let snapshots = &item.as_ref()?.snapshots;
                let bytes = snapshots
                    .iter()
                    .map(|s| s.storage.device_bytes())
                    .sum::<u64>();
                (bytes != 0).then_some((bytes, shard))
            })
            .collect::<Vec<_>>();
        if candidates.iter().map(|(bytes, _)| *bytes).sum::<u64>() < needed {
            return Ok(false);
        }
        // Snapshots cover a contiguous prefix per shard. Select whole prefixes,
        // releasing the fewest shards sufficient to fund this allocation.
        candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let mut released = 0;
        for (bytes, shard) in candidates {
            let item = self.hash_shards[shard]
                .as_ref()
                .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
            let can_drop = self
                .hash_lookup_limit
                .map_or(true, |limit| item.host_count <= limit);
            // Cheap prefixes do not warrant pinning a second copy just to avoid
            // at most two extra probes. Keep enough headroom to avoid an eager rebuild.
            let cheap_to_drop = self.hash_lookup_limit.is_some_and(|limit| {
                item.host_count < (limit / 2).min(16)
                    && item.host_count - item.searched_series() <= 2
            });
            let migrated = if cheap_to_drop {
                None
            } else {
                self.migrate_hash_snapshots(shard, bytes)?
            };
            if let Some(moved) = migrated {
                self.hash_snapshot_mapped_host_installations += moved;
            } else if can_drop {
                let item = self.hash_shards[shard]
                    .as_mut()
                    .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
                self.hash_snapshot_storage_evictions += item.snapshots.len() as u64;
                item.snapshots.clear();
            } else {
                continue;
            }
            self.remaining_bytes += bytes;
            released += bytes;
            if released >= needed {
                break;
            }
        }
        Ok(released != 0)
    }

    fn hash_mapped_snapshot_headroom(&self) -> Result<u64> {
        let limit = std::env::var("MGBFS_SEGMENT_HASH_HOST_BYTES")
            .ok()
            .map(|v| v.parse::<u64>().map_err(|_| "SEGMENTED_HASH_HOST_BUDGET"))
            .transpose()?
            .unwrap_or(512 << 20);
        let live = self
            .hash_shards
            .iter()
            .flatten()
            .flat_map(|item| &item.snapshots)
            .map(|snapshot| snapshot.storage.mapped_bytes())
            .sum::<u64>();
        Ok(limit.saturating_sub(live))
    }

    fn migrate_hash_snapshots(&mut self, shard: usize, bytes: u64) -> Result<Option<u64>> {
        if bytes > self.hash_mapped_snapshot_headroom()? {
            return Ok(None);
        }
        let item = self.hash_shards[shard]
            .as_mut()
            .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
        let mut replacements = Vec::with_capacity(item.snapshots.len());
        for (index, snapshot) in item.snapshots.iter().enumerate() {
            if snapshot.storage.device_bytes() == 0 {
                continue;
            }
            let Some(storage) = RawBuffer::try_mapped(snapshot.storage.bytes, self.base.device)?
            else {
                return Ok(None);
            };
            // The copy is synchronous and both allocations stay alive until every
            // view of this shard can be replaced without losing prefix coverage.
            crate::generic_native::checked!(cudaMemcpy, unsafe {
                cudaMemcpy(storage.host, snapshot.storage.ptr, storage.bytes, 2)
            })?;
            let mut view = snapshot.view;
            let references = u64::from(view.length) * 8;
            let offsets = references + u64::from(view.length) * u64::from(view.reference_bytes);
            view.keys = storage.ptr.cast();
            view.references = storage.at(references as usize);
            view.prefix_offsets = storage.at(offsets as usize);
            replacements.push((
                index,
                HashSeriesSnapshot {
                    storage,
                    view,
                    bank: snapshot.bank,
                },
            ));
        }
        let moved = replacements.len() as u64;
        for (index, replacement) in replacements {
            item.snapshots[index] = replacement;
        }
        Ok(Some(moved))
    }

    pub(super) fn refresh_hash_series_snapshots(
        &mut self,
        minimum_tail_runs: u32,
        only_shard: Option<usize>,
    ) -> Result<()> {
        if self.hash_lookup_limit.is_none() {
            return Ok(());
        }
        let rolling = self.layout.banks == 3;
        let shards = only_shard.map_or(0..self.hash_shards.len(), |shard| shard..shard + 1);
        for shard in shards {
            loop {
                let Some(item) = self.hash_shards[shard].as_ref() else {
                    break;
                };
                let replaced = item.replaced_runs();
                if item.host_count.saturating_sub(replaced) < minimum_tail_runs {
                    break;
                }
                if std::env::var("MGBFS_SEGMENT_TEST_BATCH").as_deref() == Ok("1")
                    && std::env::var("MGBFS_SEGMENT_TEST_HASH_SNAPSHOT_FAIL_RANK")
                        .ok()
                        .and_then(|rank| rank.parse::<u32>().ok())
                        == Some(self.rank)
                {
                    return Err("SEGMENTED_TEST_HASH_SNAPSHOT_FAILURE".into());
                }
                let started = std::time::Instant::now();
                let mut first = 0;
                let mut run_count = item.host_count;
                let mut bank = 0;
                let mut extend_last = false;
                let mut rows;
                if rolling {
                    first = replaced;
                    let count = run_count - first;
                    (bank, run_count, rows) = self.hash_prefix_summary(shard, first, count)?;
                    // A rebuild while processing the current layer may index a bank
                    // before that bank has received its final run.
                    if let Some(last) = item.snapshots.last().filter(|last| last.bank == bank) {
                        extend_last = true;
                        first -= last.view.replaced_runs;
                        run_count = run_count
                            .checked_add(last.view.replaced_runs)
                            .ok_or("SEGMENTED_HASH_SERIES_RUNS")?;
                        rows = rows
                            .checked_add(u64::from(last.view.length))
                            .ok_or("SEGMENTED_HASH_SERIES_RUNS")?;
                    } else if item.snapshots.iter().any(|s| s.bank == bank) {
                        return Err("SEGMENTED_HASH_SERIES_BANK_ORDER".into());
                    }
                } else {
                    rows = self.segments[shard].iter().map(|s| s.banks[0]).sum::<u64>();
                }
                let snapshot = if rows == 0 || rows >= 0x7fffffff {
                    None
                } else {
                    self.build_hash_series_snapshot(
                        shard,
                        rows as u32,
                        first,
                        run_count,
                        bank,
                        true,
                    )?
                };
                let replace_append_prefix = !rolling && snapshot.is_some();
                let snapshot = if !rolling && snapshot.is_none() {
                    self.build_append_tail_snapshot(shard)?
                } else {
                    snapshot
                };
                let item = self.hash_shards[shard].as_mut().unwrap();
                let Some(snapshot) = snapshot else {
                    item.snapshot_deferred += 1;
                    item.snapshot_deferred_seconds += started.elapsed().as_secs_f64();
                    break;
                };
                self.remaining_bytes -= snapshot.storage.device_bytes();
                item.snapshot_rebuilds += 1;
                self.hash_snapshot_installations += 1;
                self.hash_snapshot_mapped_host_installations +=
                    u64::from(snapshot.storage.mapped_bytes() != 0);
                if replace_append_prefix {
                    for old in item.snapshots.drain(..) {
                        self.remaining_bytes += old.storage.device_bytes();
                    }
                } else if extend_last
                    && item
                        .snapshots
                        .last()
                        .is_some_and(|previous| previous.bank == bank)
                {
                    let previous = item
                        .snapshots
                        .pop()
                        .ok_or("SEGMENTED_HASH_SERIES_BANK_ORDER")?;
                    self.remaining_bytes += previous.storage.device_bytes();
                }
                item.snapshots.push(snapshot);
                item.snapshot_rebuild_seconds += started.elapsed().as_secs_f64();
                if !rolling {
                    break;
                }
            }
        }
        Ok(())
    }

    fn hash_prefix_summary(&self, shard: usize, first: u32, count: u32) -> Result<(u32, u32, u64)> {
        let item = self.hash_shards[shard]
            .as_ref()
            .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
        let input = unsafe { item.dir.ptr.cast::<BatchHashRun>().add(first as usize) };
        let stream = self.base.stream.ptr;
        let result = self.control.at::<u64>(0);
        crate::generic_native::checked!(mgbfs_hash_extent_bank_summary_v1, unsafe {
            mgbfs_hash_extent_bank_summary_v1(input, count, result, stream)
        })?;
        let mut summary = [0u64; 3];
        crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
            cudaMemcpyAsync(
                summary.as_mut_ptr().cast(),
                result.cast(),
                std::mem::size_of_val(&summary),
                2,
                stream,
            )
        })?;
        crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        })?;
        if summary[0] >= 3 {
            return Err("SEGMENTED_HASH_SERIES_BANK_ORDER".into());
        }
        if summary[1] == 0 || summary[1] > u64::from(count) || summary[2] == u64::MAX {
            return Err("SEGMENTED_HASH_SERIES_RUNS".into());
        }
        Ok((summary[0] as u32, summary[1] as u32, summary[2]))
    }

    // A single sort of the entire append-only history can exceed the rebuild
    // budget or CUB's row limit even though the oldest uncovered runs still
    // fit. Adding one exact series preserves the indexed prefix and restores
    // the bound without paying for another copy of the indexed rows.
    fn build_append_tail_snapshot(&mut self, shard: usize) -> Result<Option<HashSeriesSnapshot>> {
        let item = self.hash_shards[shard]
            .as_ref()
            .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
        let limit = self
            .hash_lookup_limit
            .ok_or("SEGMENTED_HASH_SERIES_LIMIT")?;
        let first = item.replaced_runs();
        let runs = crate::segmented_history::append_tail_snapshot_candidates(
            first,
            item.host_count,
            item.snapshots.len() as u32,
            limit,
        );
        for count in runs.into_iter().flatten() {
            let (bank, covered, rows) = self.hash_prefix_summary(shard, first, count)?;
            if bank != 0 || covered != count {
                return Err("SEGMENTED_HASH_SERIES_BANK_ORDER".into());
            }
            if rows == 0 || rows >= 0x7fffffff {
                continue;
            }
            if let Some(snapshot) =
                self.build_hash_series_snapshot(shard, rows as u32, first, count, bank, false)?
            {
                return Ok(Some(snapshot));
            }
        }
        Ok(None)
    }

    fn build_hash_series_snapshot(
        &mut self,
        shard: usize,
        rows: u32,
        first: u32,
        run_count: u32,
        bank: u32,
        replace_append_prefix: bool,
    ) -> Result<Option<HashSeriesSnapshot>> {
        let item = self.hash_shards[shard]
            .as_ref()
            .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
        let view = item.view.as_ref().ok_or("SEGMENTED_HASH_VIEW_MISSING")?;
        let hash_segments = view.data.ptr.cast::<BatchHashSegment>();
        let hash_count = u32::try_from(view.count).map_err(|_| "SEGMENTED_DESCRIPTOR_SIZE")?;
        let hash_runs = item.dir.ptr.cast::<BatchHashRun>();
        let row_bits = 64 - (self.layout.rows_per_bank - 1).leading_zeros();
        let descriptor_count = u64::from(hash_count) * 3;
        let reference_bytes = if row_bits <= 32 && descriptor_count <= (1u64 << (32 - row_bits)) {
            4
        } else {
            8
        };
        let Some(keys_bytes) = (rows as usize).checked_mul(8) else {
            return Ok(None);
        };
        let Some(ref_bytes) = (rows as usize).checked_mul(reference_bytes as usize) else {
            return Ok(None);
        };
        const PREFIX_BYTES: usize = 258 * 4;
        let Some(storage_bytes) = keys_bytes
            .checked_add(ref_bytes)
            .and_then(|x| x.checked_add(PREFIX_BYTES))
        else {
            return Ok(None);
        };
        let mut sort_bytes = 0;
        crate::generic_native::checked!(mgbfs_hash_extent_sort_scratch_v1, unsafe {
            mgbfs_hash_extent_sort_scratch_v1(
                self.plan.state_bytes,
                reference_bytes,
                rows,
                &mut sort_bytes,
            )
        })?;
        let raw_bytes = keys_bytes + ref_bytes;
        let Some(sort_offset) = raw_bytes.checked_add(255).map(|x| x & !255usize) else {
            return Ok(None);
        };
        // Gather consumes these offsets before the sort starts. Both phases
        // can share the same tail of the allocation.
        let Some(gather_end) = raw_bytes.checked_add(run_count as usize * 4) else {
            return Ok(None);
        };
        let Some(sort_end) = sort_offset.checked_add(sort_bytes) else {
            return Ok(None);
        };
        let scratch_bytes = gather_end.max(sort_end);
        let Some(total) = storage_bytes.checked_add(scratch_bytes) else {
            return Ok(None);
        };
        let mut free = 0;
        let mut device_total = 0;
        crate::generic_native::checked!(cudaMemGetInfo, unsafe {
            mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free, &mut device_total)
        })?;
        // The rebuild reads the underlying runs. Under pressure it can give up
        // a rolling snapshot when its runs still fit the lookup cap on failure.
        let append = self.layout.banks == 1;
        let preserve_append_prefix = append
            && self
                .hash_lookup_limit
                .is_some_and(|limit| item.host_count > limit);
        let releasable = if append {
            if replace_append_prefix {
                item.snapshots
                    .iter()
                    .map(|s| s.storage.device_bytes())
                    .sum()
            } else {
                0
            }
        } else {
            item.snapshots
                .last()
                .filter(|old| {
                    old.bank == bank
                        && self.hash_lookup_limit.is_some_and(|limit| {
                            u64::from(item.searched_series()) + u64::from(old.view.replaced_runs)
                                - 1
                                <= u64::from(limit)
                        })
                })
                .map_or(0, |old| old.storage.device_bytes())
        };
        let snapshot_device_limit = std::env::var("MGBFS_SEGMENT_HASH_INDEX_VRAM_BYTES")
            .ok()
            .map(|v| {
                v.parse::<u64>()
                    .map_err(|_| "SEGMENTED_HASH_INDEX_VRAM_BUDGET")
            })
            .transpose()?
            .unwrap_or(u64::MAX);
        let snapshot_device_live = self
            .hash_shards
            .iter()
            .flatten()
            .flat_map(|item| &item.snapshots)
            .map(|snapshot| snapshot.storage.device_bytes())
            .sum::<u64>();
        let index_headroom = snapshot_device_limit.saturating_sub(snapshot_device_live);
        // Preserve the reusable extent when VRAM can hold the index and sort
        // workspace alongside it. Otherwise let the rebuild use the extent's budget.
        let physical_headroom = self
            .remaining_bytes
            .min((free as u64).saturating_sub(self.reserve_bytes));
        if self.spare_hash_segment.is_some() && total as u64 > physical_headroom {
            self.release_spare_hash_segment();
            crate::generic_native::checked!(cudaMemGetInfo, unsafe {
                mgbfs_cuda::native_owner::cudaMemGetInfo(&mut free, &mut device_total)
            })?;
        }
        let admission = crate::segmented_history::snapshot_rebuild_admission(
            total as u64,
            storage_bytes as u64,
            self.remaining_bytes,
            index_headroom,
            free as u64,
            self.reserve_bytes,
            releasable,
        );
        let mut buffers = None;
        let mut released = 0;
        if admission == Some(true) {
            if preserve_append_prefix {
                // If allocation or rebuild fails, leave the current lookup usable.
                // Migration reclaims only its device copy, keeping exact prefix coverage.
                if let Some(moved) = self.migrate_hash_snapshots(shard, releasable)? {
                    self.remaining_bytes += releasable;
                    self.hash_snapshot_mapped_host_installations += moved;
                    released = releasable;
                }
            } else {
                let item = self.hash_shards[shard]
                    .as_mut()
                    .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
                if append {
                    for old in item.snapshots.drain(..) {
                        self.remaining_bytes += old.storage.device_bytes();
                    }
                } else {
                    let old = item
                        .snapshots
                        .pop()
                        .ok_or("SEGMENTED_HASH_SERIES_BANK_ORDER")?;
                    self.remaining_bytes += old.storage.device_bytes();
                }
                item.snapshot_evicted_for_rebuild += 1;
                released = releasable;
            }
        }
        if admission == Some(false) || released != 0 {
            if let Some(storage) = RawBuffer::try_new(storage_bytes, self.base.device)? {
                if let Some(scratch) = RawBuffer::try_new(scratch_bytes, self.base.device)? {
                    buffers = Some((storage, scratch));
                }
            }
        }
        if buffers.is_none() {
            let mapped_headroom = self.hash_mapped_snapshot_headroom()?;
            // Scratch is temporary. Keep the published lookup index in VRAM
            // when the sort fits only with mapped host scratch.
            let device_headroom = self.remaining_bytes.min(
                (free as u64)
                    .saturating_add(released)
                    .saturating_sub(self.reserve_bytes),
            );
            let index_headroom = index_headroom.saturating_add(released);
            if (storage_bytes as u64) <= device_headroom.min(index_headroom)
                && (scratch_bytes as u64) <= mapped_headroom
            {
                if let Some(storage) = RawBuffer::try_new(storage_bytes, self.base.device)? {
                    if let Some(scratch) = RawBuffer::try_mapped(scratch_bytes, self.base.device)? {
                        buffers = Some((storage, scratch));
                    }
                }
            }
            // If the final index must live on the host, use available VRAM for
            // the transient sort instead of pinning both allocations.
            if buffers.is_none()
                && (scratch_bytes as u64) <= device_headroom
                && (storage_bytes as u64) <= mapped_headroom
            {
                if let Some(scratch) = RawBuffer::try_new(scratch_bytes, self.base.device)? {
                    if let Some(storage) = RawBuffer::try_mapped(storage_bytes, self.base.device)? {
                        buffers = Some((storage, scratch));
                    }
                }
            }
        }
        if buffers.is_none() {
            if total as u64 > self.hash_mapped_snapshot_headroom()? {
                return Ok(None);
            }
            let Some(storage) = RawBuffer::try_mapped(storage_bytes, self.base.device)? else {
                return Ok(None);
            };
            let Some(scratch) = RawBuffer::try_mapped(scratch_bytes, self.base.device)? else {
                return Ok(None);
            };
            buffers = Some((storage, scratch));
        }
        let (storage, scratch) = buffers.ok_or("SEGMENTED_HASH_SERIES_BUFFERS")?;
        let device_bytes = storage.device_bytes() + scratch.device_bytes();
        self.storage_peak_requested = self
            .storage_peak_requested
            .max(self.storage_budget - self.remaining_bytes + device_bytes);
        let existing_snapshots = self
            .hash_shards
            .iter()
            .flatten()
            .flat_map(|item| &item.snapshots)
            .map(|snapshot| snapshot.storage.device_bytes())
            .sum::<u64>();
        self.hash_snapshot_rebuild_peak_bytes = self
            .hash_snapshot_rebuild_peak_bytes
            .max(existing_snapshots + device_bytes);
        let stream = self.base.stream.ptr;
        let raw_keys = scratch.ptr.cast::<u64>();
        let raw_refs = unsafe { scratch.ptr.cast::<u8>().add(keys_bytes).cast::<c_void>() };
        let run_offsets = unsafe { scratch.ptr.cast::<u8>().add(raw_bytes).cast::<u32>() };
        let sort_scratch = unsafe { scratch.ptr.cast::<u8>().add(sort_offset).cast::<c_void>() };
        let error = self.control.at::<u32>(8);
        let out_keys = storage.ptr.cast::<u64>();
        let out_refs = unsafe { storage.ptr.cast::<u8>().add(keys_bytes).cast::<c_void>() };
        let offsets = unsafe {
            storage
                .ptr
                .cast::<u8>()
                .add(keys_bytes + ref_bytes)
                .cast::<u32>()
        };
        let mut code = 0u32;
        let queued = (|| -> Result<()> {
            crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                cudaMemsetAsync(error.cast(), 0, 4, stream)
            })?;
            crate::generic_native::checked!(mgbfs_hash_extent_gather_v1, unsafe {
                mgbfs_hash_extent_gather_v1(
                    hash_segments,
                    hash_count,
                    hash_runs.add(first as usize),
                    run_count,
                    rows,
                    run_offsets,
                    raw_keys,
                    raw_refs,
                    reference_bytes,
                    row_bits,
                    error,
                    stream,
                )
            })?;
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync((&mut code as *mut u32).cast(), error.cast(), 4, 2, stream)
            })?;
            crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                cudaStreamSynchronize(stream)
            })?;
            if code != 0 {
                return Err(format!("SEGMENTED_HASH_SERIES_GATHER:{code}").into());
            }
            crate::generic_native::checked!(mgbfs_hash_extent_sort_v1, unsafe {
                mgbfs_hash_extent_sort_v1(
                    self.plan.state_bytes,
                    self.plan.elements,
                    hash_segments,
                    rows,
                    raw_keys,
                    raw_refs,
                    reference_bytes,
                    row_bits,
                    out_keys,
                    out_refs,
                    sort_scratch,
                    sort_bytes,
                    stream,
                )
            })?;
            crate::generic_native::checked!(mgbfs_hash_extent_prefixes_v1, unsafe {
                mgbfs_hash_extent_prefixes_v1(out_keys, rows, offsets, stream)
            })?;
            Ok(())
        })();
        // A multi-kernel wrapper can fail after submitting some work.
        // Drain before dropping either device or mapped-host allocations.
        let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        });
        queued?;
        drained?;
        Ok(Some(HashSeriesSnapshot {
            storage,
            bank,
            view: HashSeriesView {
                keys: out_keys,
                references: out_refs,
                prefix_offsets: offsets,
                length: rows,
                replaced_runs: run_count,
                reference_bytes,
                row_bits,
            },
        }))
    }

    pub(super) fn hash_view(&mut self, shard: usize, grow: bool) -> Result<Option<OwnerPrepared>> {
        let n = self.segments[shard].len();
        if n == 0 {
            return Err("SEGMENTED_HASH_VIEW_EMPTY".into());
        }
        if self.hash_shards[shard].is_none() {
            let capacity = initial_hash_run_capacity()?;
            let bytes = hash_directory_bytes(capacity)?;
            let Some(dir) = self.storage_allocation(bytes, shard, "hash_directory_budget")? else {
                return Ok(None);
            };
            crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                cudaMemsetAsync(dir.at::<c_void>(bytes - 4), 0, 4, self.base.stream.ptr)
            })?;
            self.remaining_bytes -= bytes as u64;
            self.hash_shards[shard] = Some(BatchHashShard {
                snapshots: Vec::new(),
                snapshot_rebuilds: 0,
                snapshot_retirements: 0,
                snapshot_evicted_for_rebuild: 0,
                snapshot_deferred: 0,
                snapshot_rebuild_seconds: 0.,
                snapshot_deferred_seconds: 0.,
                lookup_max_series: 0,
                dir,
                capacity,
                host_count: 0,
                view: None,
                max_runs: 0,
                run_histogram: std::collections::BTreeMap::new(),
                timed_max_runs: 0,
                timed_run_histogram: std::collections::BTreeMap::new(),
                deferred_merges: 0,
                merged_runs: 0,
            });
        }
        if grow {
            let old = self.hash_shards[shard]
                .as_ref()
                .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
            let capacity = old
                .capacity
                .checked_mul(2)
                .ok_or("SEGMENTED_HASH_DIRECTORY_OVERFLOW")?;
            let bytes = hash_directory_bytes(capacity)?;
            let old_bytes = old.dir.bytes;
            let old_count = old.host_count;
            let Some(dir) =
                self.storage_allocation(bytes, shard, "hash_directory_growth_budget")?
            else {
                return Ok(None);
            };
            let old = self.hash_shards[shard].as_ref().unwrap();
            let live = (old_count as usize)
                .checked_mul(std::mem::size_of::<BatchHashRun>())
                .ok_or("SEGMENTED_HASH_DIRECTORY_SIZE")?;
            if old_count > old.capacity {
                return Err("SEGMENTED_HASH_DIRECTORY_CORRUPT".into());
            }
            if live != 0 {
                crate::generic_native::checked!(cudaMemcpy, unsafe {
                    cudaMemcpy(dir.ptr, old.dir.ptr, live, 3)
                })?;
            }
            crate::generic_native::checked!(cudaMemcpy, unsafe {
                cudaMemcpy(dir.at::<c_void>(bytes - 4), old.count_ptr().cast(), 4, 3)
            })?;
            let item = self.hash_shards[shard].as_mut().unwrap();
            item.dir = dir;
            item.capacity = capacity;
            self.remaining_bytes -=
                u64::try_from(bytes - old_bytes).map_err(|_| "SEGMENTED_HASH_DIRECTORY_SIZE")?;
        }
        let needs = self.hash_shards[shard]
            .as_ref()
            .unwrap()
            .view
            .as_ref()
            .map_or(true, |v| v.count != n);
        if needs {
            let item_size = std::mem::size_of::<BatchHashSegment>();
            let bytes = n
                .checked_mul(item_size)
                .ok_or("SEGMENTED_DESCRIPTOR_SIZE")?;
            let item = self.hash_shards[shard].as_mut().unwrap();
            if let Some(view) = item.view.as_mut().filter(|view| view.data.bytes >= bytes) {
                // Reclaim already rewrites the surviving prefix. Growth only appends
                // descriptors, so the cached prefix needs no host or device copy.
                // make_segment_prepared publishes one extent before trying another.
                if view.count != n - 1 {
                    return Err("SEGMENTED_HASH_VIEW_COUNT".into());
                }
                let descriptor = self.segments[shard][view.count].sorted_view();
                let offset = view.count * item_size;
                crate::generic_native::checked!(cudaMemcpy, unsafe {
                    cudaMemcpy(
                        view.data.at::<c_void>(offset),
                        (&descriptor as *const BatchHashSegment).cast(),
                        item_size,
                        1,
                    )
                })?;
                view.count = n;
            } else {
                let old = item.view.as_ref().map_or(0, |view| view.data.bytes);
                let Some(data) = self.storage_allocation(bytes, shard, "hash_view_budget")? else {
                    return Ok(None);
                };
                let descriptors = self.segments[shard]
                    .iter()
                    .map(Segment::sorted_view)
                    .collect::<Vec<_>>();
                crate::generic_native::checked!(cudaMemcpy, unsafe {
                    cudaMemcpy(data.ptr, descriptors.as_ptr().cast(), bytes, 1)
                })?;
                self.hash_shards[shard].as_mut().unwrap().view =
                    Some(DescriptorCache { data, count: n });
                self.remaining_bytes -=
                    u64::try_from(bytes - old).map_err(|_| "SEGMENTED_DESCRIPTOR_SIZE")?;
            }
        }
        let item = self.hash_shards[shard].as_ref().unwrap();
        let view = item.view.as_ref().ok_or("SEGMENTED_HASH_VIEW_MISSING")?;
        Ok(Some(OwnerPrepared {
            sealed: ptr::null(),
            descriptor_count: 0,
            hash_segments: view.data.ptr.cast(),
            hash_count: u32::try_from(n).map_err(|_| "SEGMENTED_DESCRIPTOR_SIZE")?,
            hash_runs: item.dir.ptr.cast(),
            hash_run_count: item.count_ptr(),
            hash_run_capacity: item.capacity,
        }))
    }
}
