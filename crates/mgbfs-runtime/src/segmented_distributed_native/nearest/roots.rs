//! Install the reverse frontier directly in the existing exact Batch hash arena.

use super::super::{
    mgbfs_nccl_all_gather_u64, BatchHashRun, BatchHashSegment, Range, SegmentedDistributedBfs,
};
use super::state_hash;
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{cudaMemcpyAsync, cudaStreamSynchronize};
use std::{
    ffi::c_void,
    mem::{size_of, size_of_val},
};

#[derive(Clone, Copy)]
pub(crate) enum RootSpec<'a> {
    Single,
    Targets(&'a [Vec<i64>]),
}

struct TargetRoot<'a> {
    hash: u64,
    coords: &'a [i64],
    target: u32,
}

const UPLOAD_BYTES: usize = 4 * 1024 * 1024;

impl SegmentedDistributedBfs {
    pub(in crate::segmented_distributed_native) fn install_target_roots(
        &mut self,
        targets: &[Vec<i64>],
    ) -> Result<()> {
        if targets.is_empty() || u32::try_from(targets.len()).is_err() {
            return Err("SEGMENTED_NEAREST_TARGET_COUNT".into());
        }
        let ownership = self.ownership.as_ref().ok_or("SEGMENTED_PATH_OWNERSHIP")?;
        let mut buckets: Vec<Vec<TargetRoot<'_>>> =
            (0..self.plan.shards).map(|_| Vec::new()).collect();
        for (index, coords) in targets.iter().enumerate() {
            if coords.len() != self.plan.elements as usize
                || (self.plan.state_bytes == 1 && coords.iter().any(|&v| u8::try_from(v).is_err()))
            {
                return Err("SEGMENTED_NEAREST_TARGET_SHAPE".into());
            }
            let hash = state_hash(coords, self.seed, self.bits);
            if ownership.rank_for(hash) == self.rank {
                let shard = ((u64::from(hash as u32) * u64::from(self.plan.shards)) >> 32) as usize;
                buckets[shard].push(TargetRoot {
                    hash,
                    coords,
                    target: index as u32,
                });
            }
        }
        let mut count = 0usize;
        for bucket in &mut buckets {
            bucket.sort_unstable_by(|a, b| {
                (a.hash, a.coords, a.target).cmp(&(b.hash, b.coords, b.target))
            });
            // Exact equality retains distinct colliding states and the first input ID.
            bucket.dedup_by(|a, b| a.coords == b.coords);
            count += bucket.len();
        }
        if self
            .capacity_limit
            .is_some_and(|limit| count as u64 > limit)
        {
            return Err("SEGMENTED_ROOT_ADMISSION_FAILED".into());
        }
        let mut ids = Vec::with_capacity(count);
        for (shard, bucket) in buckets.iter().enumerate() {
            let mut rows = bucket.as_slice();
            while !rows.is_empty() {
                let grow = self.hash_shards[shard]
                    .as_ref()
                    .is_some_and(|directory| directory.host_count == directory.capacity);
                let upper = self.layout.rows_per_bank.min(rows.len() as u64);
                let Some(view) = self.make_segment_prepared_bounded(shard, grow, 0, upper)? else {
                    return Err("SEGMENTED_ROOT_ADMISSION_FAILED".into());
                };
                let segment = self.segments[shard].len() - 1;
                let extent = &mut self.segments[shard][segment];
                let populated = rows.len().min(extent.desc.capacity as usize);
                let (batch, rest) = rows.split_at(populated);
                rows = rest;
                let stream = self.base.stream.ptr;
                let extent_view = extent.sorted_view();
                match self.plan.state_bytes {
                    1 => upload_extent(extent_view, batch, |v| v as u8, stream)?,
                    8 => upload_extent(extent_view, batch, |v| v, stream)?,
                    _ => return Err("SEGMENTED_NEAREST_ROOT_LAYOUT".into()),
                }
                let length = populated as u64;
                let run = root_run(
                    u32::try_from(segment).map_err(|_| "SEGMENTED_DESCRIPTOR_SIZE")?,
                    batch,
                );
                let directory = self.hash_shards[shard]
                    .as_mut()
                    .ok_or("SEGMENTED_HASH_DIRECTORY_MISSING")?;
                let slot = directory.host_count as usize;
                let run_count = directory.host_count + 1;
                let result = (|| {
                    enqueue(
                        view.hash_runs.wrapping_add(slot),
                        std::slice::from_ref(&run),
                        stream,
                    )?;
                    enqueue(
                        view.hash_run_count,
                        std::slice::from_ref(&run_count),
                        stream,
                    )?;
                    enqueue(
                        extent.counts.ptr.cast::<u64>(),
                        std::slice::from_ref(&length),
                        stream,
                    )
                })();
                finish_upload(stream, result)?;
                directory.observe(run_count, false);
                extent.banks[0] = length;
                self.current.push(Range {
                    shard,
                    segment,
                    bank: 0,
                    begin: 0,
                    end: length,
                    logical_start: 0,
                });
                // `NearestState::publish` numbers these ranges in this order.
                ids.extend(batch.iter().map(|root| root.target));
            }
        }
        let limit = self
            .hash_lookup_limit
            .ok_or("SEGMENTED_NEAREST_LOOKUP_BOUND")?;
        for shard in 0..self.hash_shards.len() {
            if self.hash_shards[shard]
                .as_ref()
                .is_some_and(|directory| directory.searched_series() > limit)
            {
                self.refresh_hash_series_snapshots(2, Some(shard))?;
            }
        }
        // Rebuilding another shard may evict a snapshot under pressure.
        if self
            .hash_shards
            .iter()
            .flatten()
            .any(|d| d.searched_series() > limit)
        {
            return Err("SEGMENTED_ROOT_ADMISSION_FAILED".into());
        }
        if let Some(nearest) = &mut self.nearest {
            nearest.root_targets = ids;
        } else {
            self.set_window_root_targets(ids)?;
        }
        self.bank_totals[0] = count as u64;
        self.peak_live_rows = count as u64;
        // Stage the rank count before the constructor agrees to enter NCCL.
        let stream = self.base.stream.ptr;
        let queued = enqueue(self.control.at(0), &self.bank_totals[..1], stream);
        finish_upload(stream, queued)
    }

    pub(in crate::segmented_distributed_native) fn gather_target_root_counts(
        &mut self,
    ) -> Result<()> {
        let stream = self.base.stream.ptr;
        crate::generic_native::checked!(mgbfs_nccl_all_gather_u64, unsafe {
            mgbfs_nccl_all_gather_u64(
                self.comm.ptr,
                self.control.at(0),
                self.control.at(64),
                stream,
            )
        })?;
        crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
            cudaMemcpyAsync(
                self.counts.as_mut_ptr().cast(),
                self.control.at::<c_void>(64),
                size_of_val(self.counts.as_slice()),
                2,
                stream,
            )
        })?;
        self.wait_boundary_stream()?;
        Ok(())
    }

    pub(crate) fn root_counts(&self) -> &[u64] {
        debug_assert_eq!(self.depth, 0);
        &self.counts
    }

    pub(crate) fn root_target(&self, logical_row: u64) -> Result<u32> {
        let logical_row = usize::try_from(logical_row).map_err(|_| "SEGMENTED_NEAREST_ROOT_ROW")?;
        self.nearest
            .as_ref()
            .and_then(|state| state.root_targets.get(logical_row).copied())
            .ok_or_else(|| "SEGMENTED_NEAREST_ROOT_ROW".into())
    }
}

fn enqueue<T>(destination: *mut T, values: &[T], stream: *mut c_void) -> Result<()> {
    crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
        cudaMemcpyAsync(
            destination.cast(),
            values.as_ptr().cast(),
            size_of_val(values),
            1,
            stream,
        )
    })
}

fn finish_upload(stream: *mut c_void, queued: Result<()>) -> Result<()> {
    // Keep pageable host staging alive until every successfully queued copy completes.
    let synced = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
        cudaStreamSynchronize(stream)
    });
    queued.and(synced)
}

fn upload_extent<T: Copy>(
    extent: BatchHashSegment,
    roots: &[TargetRoot<'_>],
    encode: impl Fn(i64) -> T,
    stream: *mut c_void,
) -> Result<()> {
    let stride = (extent.capacity * u64::from(extent.banks)) as usize;
    let elements = roots[0].coords.len();
    let row_bytes = elements * size_of::<T>() + size_of::<u64>() + size_of::<u32>();
    let tile_rows = (UPLOAD_BYTES / row_bytes).max(1).min(roots.len());
    let mut states = Vec::with_capacity(elements * tile_rows);
    let mut hashes = Vec::with_capacity(tile_rows);
    let mut rows = Vec::with_capacity(tile_rows);
    for (tile, chunk) in roots.chunks(tile_rows).enumerate() {
        let start = tile * tile_rows;
        states.clear();
        for coord in 0..elements {
            states.extend(chunk.iter().map(|root| encode(root.coords[coord])));
        }
        hashes.clear();
        hashes.extend(chunk.iter().map(|root| root.hash));
        rows.clear();
        rows.extend((start..start + chunk.len()).map(|row| row as u32));
        let queued = (|| {
            for (coord, column) in states.chunks_exact(chunk.len()).enumerate() {
                let offset = coord * stride + start;
                enqueue(
                    extent.states.cast::<T>().wrapping_add(offset),
                    column,
                    stream,
                )?;
            }
            enqueue(extent.keys.wrapping_add(start), &hashes, stream)?;
            enqueue(extent.rows.wrapping_add(start), &rows, stream)
        })();
        finish_upload(stream, queued)?;
    }
    Ok(())
}

fn root_run(segment: u32, roots: &[TargetRoot<'_>]) -> BatchHashRun {
    // Match segmented_hash_batch.cu::fill_fences. Short runs have a low-byte
    // mask, and long runs retain the seven quantile fences.
    let mut hash_fences = [0; 7];
    if roots.len() < 256 {
        for root in roots {
            hash_fences[((root.hash >> 6) & 3) as usize] |= 1u64 << (root.hash & 63);
        }
        hash_fences[4] = 1;
    } else {
        for (i, fence) in hash_fences.iter_mut().enumerate() {
            let index = (roots.len() as u64 * (i as u64 + 1) / 8) as usize;
            *fence = roots[index].hash;
        }
    }
    BatchHashRun {
        segment,
        bank: 0,
        begin: 0,
        length: roots.len() as u64,
        min_hash: roots[0].hash,
        max_hash: roots[roots.len() - 1].hash,
        hash_fences,
    }
}
