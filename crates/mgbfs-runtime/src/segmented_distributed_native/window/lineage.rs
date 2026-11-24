//! Resolve every raw candidate against the accepted layer before releasing its packet.

use super::super::{
    BatchHashRun, BatchHashSegment, Buffer, GenericRouteRecord, HashSeriesDirectory, OwnerPrepared,
    PacketOwnerShard, Plan, Segment,
};
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{cudaMemcpyAsync, cudaMemsetAsync, cudaStreamSynchronize};
use std::{ffi::c_void, mem::size_of};

const TILE_ROWS: u32 = 16 * 1024;
const WINDOW_BANK: u32 = 0;
const DEVICE_TO_HOST: i32 = 2;
const STATUS_LOOKUP_LIMIT: u32 = 1024;

pub(super) const DEVICE_BYTES: usize =
    TILE_ROWS as usize * size_of::<ResolvedEdge>() + size_of::<u32>();

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ResolvedEdge {
    pub(super) parent: u64,
    pub(super) segment: u32,
    bank: u32,
    pub(super) row: u32,
    pub(super) source: u32,
    pub(super) generator: u32,
    found: u32,
}

const _: [(); 32] = [(); size_of::<ResolvedEdge>()];

type Resolve = unsafe extern "C" fn(
    state_bytes: u32,
    elements: u32,
    states: *const c_void,
    records: *const GenericRouteRecord,
    rows: u32,
    begin: u32,
    count: u32,
    source: u32,
    shard: u32,
    generators: u32,
    expected_bank: u32,
    segments: *const BatchHashSegment,
    segment_count: u32,
    runs: *const BatchHashRun,
    run_count: *const u32,
    run_capacity: u32,
    edges: *mut ResolvedEdge,
    output_capacity: u32,
    error: *mut u32,
    stream: *mut c_void,
    view: *const HashSeriesDirectory,
    max_series: u32,
) -> i32;

pub(super) struct LineageSource<'a> {
    pub(super) plan: &'a Plan,
    pub(super) span: PacketOwnerShard,
    pub(super) rank: u32,
    pub(super) parent_rows: u64,
    pub(super) generators: u32,
}

pub(super) struct LineageCatalog<'a> {
    pub(super) prepared: &'a OwnerPrepared,
    pub(super) segments: &'a [Segment],
    pub(super) directory: &'a HashSeriesDirectory,
    pub(super) max_series: u32,
}

pub(super) struct LineageTile {
    resolve: Resolve,
    device: Buffer,
    host: Vec<ResolvedEdge>,
}

impl LineageTile {
    fn load() -> Option<Resolve> {
        let symbol = unsafe {
            libc::dlsym(
                libc::RTLD_DEFAULT,
                b"mgbfs_hash_batch_resolve_window_v1\0".as_ptr().cast(),
            )
        };
        if symbol.is_null() {
            return None;
        }
        Some(unsafe { std::mem::transmute::<*mut c_void, Resolve>(symbol) })
    }

    pub(super) fn abi_ready() -> bool {
        Self::load().is_some()
    }

    pub(super) fn new(device: i32) -> Result<Self> {
        Ok(Self {
            resolve: Self::load().ok_or("SEGMENTED_WINDOW_LINEAGE_ABI_UNAVAILABLE")?,
            device: Buffer::new(DEVICE_BYTES, device)?,
            host: vec![ResolvedEdge::default(); TILE_ROWS as usize],
        })
    }

    // The caller holds the original span and an immutable destination catalog
    // until this returns. `append` writes only to the current layer's staging area.
    pub(super) fn resolve(
        &mut self,
        source: LineageSource<'_>,
        catalog: LineageCatalog<'_>,
        stream: *mut c_void,
        mut append: impl FnMut(&[ResolvedEdge]) -> Result<()>,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let LineageSource {
            plan,
            span,
            rank,
            parent_rows,
            generators,
        } = source;
        let LineageCatalog {
            prepared,
            segments,
            directory,
            max_series,
        } = catalog;
        if span.rows == 0 {
            return Ok(());
        }
        let shard = u32::try_from(span.shard).map_err(|_| "SEGMENTED_WINDOW_LINEAGE_SOURCE")?;
        if shard >= plan.shards
            || rank >= plan.world
            || generators == 0
            || parent_rows == 0
            || plan.packed_candidates()
            || span.byte_wire
            || span.queue.states.is_null()
            || span.queue.records.is_null()
        {
            return Err("SEGMENTED_WINDOW_LINEAGE_SOURCE".into());
        }
        if prepared.hash_count == 0
            || prepared.hash_count as usize != segments.len()
            || max_series == 0
        {
            return Err("SEGMENTED_WINDOW_LINEAGE_CATALOG".into());
        }

        let device_error = self
            .device
            .at::<u32>(TILE_ROWS as usize * size_of::<ResolvedEdge>());
        let mut begin = 0;
        while begin < span.rows {
            checkpoint()?;
            let count = (span.rows - begin).min(TILE_ROWS);
            let edges = &mut self.host[..count as usize];
            let mut error = 0u32;
            let submitted = (|| -> Result<()> {
                crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                    cudaMemsetAsync(device_error.cast(), 0, size_of::<u32>(), stream)
                })?;
                crate::generic_native::checked!(mgbfs_hash_batch_resolve_window_v1, unsafe {
                    (self.resolve)(
                        plan.state_bytes,
                        plan.elements,
                        span.queue.states,
                        span.queue.records.cast(),
                        span.rows,
                        begin,
                        count,
                        rank,
                        shard,
                        generators,
                        WINDOW_BANK,
                        prepared.hash_segments,
                        prepared.hash_count,
                        prepared.hash_runs,
                        prepared.hash_run_count,
                        prepared.hash_run_capacity,
                        self.device.ptr.cast(),
                        TILE_ROWS,
                        device_error,
                        stream,
                        directory,
                        max_series,
                    )
                })?;
                crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                    cudaMemcpyAsync(
                        edges.as_mut_ptr().cast(),
                        self.device.ptr,
                        std::mem::size_of_val(edges),
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
            // Even a partial submission may have queued a copy into these host buffers.
            let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                cudaStreamSynchronize(stream)
            });
            submitted.and(drained)?;
            if error == STATUS_LOOKUP_LIMIT {
                return Err("SEGMENTED_WINDOW_LOOKUP_RESOURCE".into());
            }
            if error != 0 {
                return Err(format!("SEGMENTED_WINDOW_LINEAGE_STATUS_{error}"));
            }
            for edge in edges.iter() {
                if edge.found != 1
                    || edge.bank != WINDOW_BANK
                    || edge.source != rank
                    || edge.parent >= parent_rows
                    || edge.generator >= generators
                {
                    return Err("SEGMENTED_WINDOW_EDGE_PROVENANCE".into());
                }
                let segment = segments
                    .get(edge.segment as usize)
                    .ok_or("SEGMENTED_WINDOW_EDGE_SEGMENT")?;
                if segment.desc.banks != 1
                    || u64::from(edge.row) >= segment.banks[WINDOW_BANK as usize]
                {
                    return Err("SEGMENTED_WINDOW_EDGE_ROW".into());
                }
            }
            checkpoint()?;
            append(edges)?;
            begin += count;
        }
        Ok(())
    }
}
