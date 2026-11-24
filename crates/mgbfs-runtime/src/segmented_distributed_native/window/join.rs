//! Exact owner-local intersection of two committed Window layers.

use super::super::nearest::TargetHit;
use super::super::{
    BatchHashRun, BatchHashSegment, Buffer, HashSeriesDirectory, OwnerPrepared, Segment,
    SegmentedDistributedBfs,
};
use super::write_pair;
use crate::path_csr::io_error;
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{cudaMemcpyAsync, cudaMemsetAsync, cudaStreamSynchronize};
use std::{
    ffi::c_void,
    fs::OpenOptions,
    io::{BufWriter, Write},
    mem::size_of,
    path::Path,
    rc::Rc,
};

const TILE_ROWS: u64 = 16 * 1024;
const DEVICE_TO_HOST: i32 = 2;
const STATUS_LOOKUP_LIMIT: u32 = 1024;
const WINDOW_BANK: u32 = 0;
const HIT_BYTES: usize = TILE_ROWS as usize * size_of::<TargetHit>();
const _: [(); 16] = [(); size_of::<TargetHit>()];

type Join = unsafe extern "C" fn(
    state_bytes: u32,
    elements: u32,
    left_states: *const c_void,
    left_stride: u64,
    begin: u64,
    end: u64,
    seed: u64,
    hash_bits: u32,
    expected_bank: u32,
    segments: *const BatchHashSegment,
    segment_count: u32,
    runs: *const BatchHashRun,
    run_count: *const u32,
    run_capacity: u32,
    hits: *mut TargetHit,
    hit_capacity: u32,
    error: *mut u32,
    stream: *mut c_void,
    view: *const HashSeriesDirectory,
    max_series: u32,
) -> i32;

pub(super) struct JoinTile {
    kernel: Join,
    device: Buffer,
    host: Vec<TargetHit>,
}

impl JoinTile {
    pub(super) const DEVICE_BYTES: usize = HIT_BYTES + size_of::<u32>();

    fn load() -> Option<Join> {
        let symbol = unsafe {
            libc::dlsym(
                libc::RTLD_DEFAULT,
                b"mgbfs_hash_batch_join_window_v1\0".as_ptr().cast(),
            )
        };
        (!symbol.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, Join>(symbol) })
    }

    pub(super) fn abi_ready() -> bool {
        Self::load().is_some()
    }

    pub(super) fn new(device: i32) -> Result<Self> {
        Ok(Self {
            kernel: Self::load().ok_or("SEGMENTED_WINDOW_JOIN_ABI_UNAVAILABLE")?,
            device: Buffer::new(Self::DEVICE_BYTES, device)?,
            host: vec![TargetHit::default(); TILE_ROWS as usize],
        })
    }

    // Artifact ownership and removal on failed agreement belong to the caller.
    pub(super) fn write(
        &mut self,
        left: &SegmentedDistributedBfs,
        right: &mut SegmentedDistributedBfs,
        output: &Path,
        budget: u64,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<u64> {
        if !Rc::ptr_eq(&left.comm, &right.comm)
            || left.rank != right.rank
            || left.base.device != right.base.device
            || left.base.device != self.device.device
            || left.plan.world != right.plan.world
            || left.plan.shards != right.plan.shards
            || left.plan.state_bytes != right.plan.state_bytes
            || left.plan.elements != right.plan.elements
            || left.seed != right.seed
            || left.bits != right.bits
            || !left
                .window
                .as_ref()
                .is_some_and(|window| window.expansion.is_none())
            || !right
                .window
                .as_ref()
                .is_some_and(|window| window.expansion.is_none())
        {
            return Err("SEGMENTED_WINDOW_JOIN_LAYOUT".into());
        }
        checkpoint()?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .map_err(io_error)?;
        let mut writer = BufWriter::new(file);
        let mut bytes = 0;
        let mut cursor = 0;
        while cursor < left.current.len() {
            checkpoint()?;
            let shard = left.current[cursor].shard;
            // Committed ranges are in (shard, extent) order on both sides.
            let past =
                cursor + left.current[cursor..].partition_point(|range| range.shard == shard);
            let ranges = &left.current[cursor..past];
            cursor = past;
            let right_extents = right
                .segments
                .get(shard)
                .ok_or("SEGMENTED_WINDOW_JOIN_SHARD")?;
            if right_extents.iter().all(|extent| extent.banks[0] == 0) {
                continue;
            }
            let catalog = right.window_catalog(shard)?;
            for range in ranges {
                let extent = left.segments[shard]
                    .get(range.segment)
                    .ok_or("SEGMENTED_WINDOW_JOIN_EXTENT")?;
                if range.bank != WINDOW_BANK
                    || extent.desc.banks != 1
                    || range.begin >= range.end
                    || range.end > extent.banks[0]
                {
                    return Err("SEGMENTED_WINDOW_JOIN_RANGE".into());
                }
                let mut begin = range.begin;
                while begin < range.end {
                    checkpoint()?;
                    let end = begin + (range.end - begin).min(TILE_ROWS);
                    let hits = self.probe(right, extent, begin, end, &catalog)?;
                    checkpoint()?;
                    for (offset, hit) in hits.iter().enumerate() {
                        if hit.found == 0 {
                            continue;
                        }
                        if hit.found != 1 || hit.bank != WINDOW_BANK {
                            return Err("SEGMENTED_WINDOW_JOIN_HIT".into());
                        }
                        let lhs = range
                            .logical_start
                            .checked_add(begin - range.begin + offset as u64)
                            .ok_or("SEGMENTED_WINDOW_ROW_OVERFLOW")?;
                        let rhs = right.window_logical_row(shard, hit.segment, hit.row)?;
                        write_pair(&mut writer, &mut bytes, budget, lhs, rhs)?;
                    }
                    begin = end;
                }
            }
        }
        checkpoint()?;
        writer.flush().map_err(io_error)?;
        Ok(bytes)
    }

    fn probe(
        &mut self,
        right: &SegmentedDistributedBfs,
        left: &Segment,
        begin: u64,
        end: u64,
        catalog: &(OwnerPrepared, HashSeriesDirectory, u32),
    ) -> Result<&[TargetHit]> {
        let (prepared, directory, limit) = catalog;
        let stream = right.base.stream.ptr;
        let device_error = self.device.at::<u32>(HIT_BYTES);
        let hits = &mut self.host[..(end - begin) as usize];
        let mut error = 0u32;
        let submitted = (|| -> Result<()> {
            crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                cudaMemsetAsync(device_error.cast(), 0, size_of::<u32>(), stream)
            })?;
            crate::generic_native::checked!(mgbfs_hash_batch_join_window_v1, unsafe {
                (self.kernel)(
                    right.plan.state_bytes,
                    right.plan.elements,
                    left.desc.states,
                    left.desc.capacity,
                    begin,
                    end,
                    right.seed,
                    right.bits,
                    WINDOW_BANK,
                    prepared.hash_segments,
                    prepared.hash_count,
                    prepared.hash_runs,
                    prepared.hash_run_count,
                    prepared.hash_run_capacity,
                    self.device.ptr.cast(),
                    TILE_ROWS as u32,
                    device_error,
                    stream,
                    directory,
                    *limit,
                )
            })?;
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(
                    hits.as_mut_ptr().cast(),
                    self.device.ptr,
                    std::mem::size_of_val(hits),
                    DEVICE_TO_HOST,
                    stream,
                )
            })?;
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(
                    (&mut error as *mut u32).cast(),
                    device_error.cast(),
                    size_of::<u32>(),
                    DEVICE_TO_HOST,
                    stream,
                )
            })
        })();
        // The host hit slice and status remain valid after any partial submission.
        let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        });
        submitted.and(drained)?;
        match error {
            0 => Ok(hits),
            STATUS_LOOKUP_LIMIT => Err("SEGMENTED_WINDOW_LOOKUP_RESOURCE".into()),
            _ => Err(format!("SEGMENTED_WINDOW_JOIN_STATUS_{error}")),
        }
    }
}
