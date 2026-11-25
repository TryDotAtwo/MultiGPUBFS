//! Resolve every raw candidate against the accepted layer before releasing its packet.

use super::super::{
    packet::PacketEvent, BatchHashRun, BatchHashSegment, Buffer, GenericRouteRecord,
    HashSeriesDirectory, OwnerPrepared, PacketOwnerShard, Plan, Stream,
};
use crate::path_csr::{Transition, TransitionSpool, TransitionWriter};
use crate::pinned_archive::Slot;
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{
    cudaEventQuery, cudaEventRecord, cudaMemcpyAsync, cudaMemsetAsync, cudaStreamSynchronize,
};
use std::{
    collections::VecDeque,
    ffi::c_void,
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

const TILE_ROWS: u32 = 16 * 1024;
const WINDOW_BANK: u32 = 0;
const DEVICE_TO_HOST: i32 = 2;
const STATUS_LOOKUP_LIMIT: u32 = 1024;

const HOST_SLOTS: usize = 3;
const DEVICE_SLOTS: usize = 2;
const EDGE_BYTES: usize = TILE_ROWS as usize * size_of::<ResolvedEdge>();
const SLOT_BYTES: usize = EDGE_BYTES + size_of::<u64>();
pub(super) const DEVICE_BYTES: usize = DEVICE_SLOTS * SLOT_BYTES;

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
    pub(super) directory: &'a HashSeriesDirectory,
    pub(super) max_series: u32,
    pub(super) bases: Arc<Vec<u64>>,
    pub(super) last_rows: u64,
}

struct DeviceTile {
    buffer: Buffer,
    copied: PacketEvent,
}

pub(super) struct Batch {
    slot: Slot,
    count: u32,
    shard: u32,
    rank: u32,
    parent_rows: u64,
    generators: u32,
    bases: Arc<Vec<u64>>,
    last_rows: u64,
}

impl Batch {
    fn append(&self, spool: &mut TransitionSpool, edges: &mut Vec<Transition>) -> Result<()> {
        // The rank hands the slot to the writer only after D2H completes.
        let (status, records) = unsafe {
            let ptr = self.slot.ptr.cast::<u8>();
            (
                *ptr.add(EDGE_BYTES).cast::<u32>(),
                std::slice::from_raw_parts(ptr.cast::<ResolvedEdge>(), self.count as usize),
            )
        };
        if status == STATUS_LOOKUP_LIMIT {
            return Err("SEGMENTED_WINDOW_LOOKUP_RESOURCE".into());
        }
        if status != 0 {
            return Err(format!("SEGMENTED_WINDOW_LINEAGE_STATUS_{status}"));
        }
        edges.clear();
        for edge in records {
            if edge.found != 1
                || edge.bank != WINDOW_BANK
                || edge.source != self.rank
                || edge.parent >= self.parent_rows
                || edge.generator >= self.generators
            {
                return Err("SEGMENTED_WINDOW_EDGE_PROVENANCE".into());
            }
            let index = edge.segment as usize;
            let base = *self
                .bases
                .get(index)
                .ok_or("SEGMENTED_WINDOW_EDGE_SEGMENT")?;
            let rows = self
                .bases
                .get(index + 1)
                .map_or(self.last_rows, |end| end - base);
            if u64::from(edge.row) >= rows {
                return Err("SEGMENTED_WINDOW_EDGE_ROW".into());
            }
            edges.push(Transition {
                child: base
                    .checked_add(u64::from(edge.row))
                    .ok_or("SEGMENTED_WINDOW_ROW_OVERFLOW")?,
                parent_rank: edge.source,
                parent: edge.parent,
                generator: edge.generator,
            });
        }
        spool.add_shard(self.shard, edges)
    }
}

pub(super) type Writer = TransitionWriter<Batch>;

pub(super) struct FinishedSpool {
    pub(super) spool: TransitionSpool,
    pub(super) failure: Option<String>,
}

pub(super) fn start_writer(spool: TransitionSpool) -> Result<Writer> {
    let mut edges = Vec::with_capacity(TILE_ROWS as usize);
    Writer::new(spool, HOST_SLOTS, move |spool, batch| {
        batch.append(spool, &mut edges)
    })
}

pub(super) fn prepare_pinned_pool(sides: usize) {
    crate::session_cache::prepare_pinned(SLOT_BYTES, HOST_SLOTS * sides);
}

pub(super) struct LineageTile {
    resolve: Resolve,
    copy: Stream,
    produced: PacketEvent,
    device: [DeviceTile; DEVICE_SLOTS],
    next: usize,
    pending: VecDeque<Batch>,
    free: Vec<Slot>,
    buffer_wait: Duration,
    buffer_waits: u64,
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
        let tile = || -> Result<DeviceTile> {
            Ok(DeviceTile {
                buffer: Buffer::new(SLOT_BYTES, device)?,
                copied: PacketEvent::new(device)?,
            })
        };
        Ok(Self {
            resolve: Self::load().ok_or("SEGMENTED_WINDOW_LINEAGE_ABI_UNAVAILABLE")?,
            copy: Stream::new(device)?,
            produced: PacketEvent::new(device)?,
            device: [tile()?, tile()?],
            next: 0,
            pending: VecDeque::with_capacity(HOST_SLOTS),
            free: Slot::new_ring(SLOT_BYTES, HOST_SLOTS)?,
            buffer_wait: Duration::ZERO,
            buffer_waits: 0,
        })
    }

    pub(super) fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "pinned_slots": HOST_SLOTS, "pinned_bytes": HOST_SLOTS * SLOT_BYTES,
            "device_slots": DEVICE_SLOTS, "buffer_wait_seconds": self.buffer_wait.as_secs_f64(),
            "buffer_waits": self.buffer_waits,
        })
    }

    fn dispatch_copies(&mut self, writer: &mut Writer) -> Result<()> {
        while let Some(batch) = self.pending.front() {
            match unsafe { cudaEventQuery(batch.slot.ready) } {
                0 => {}
                600 => break,
                status => return crate::generic_native::checked!(cudaEventQuery, status),
            }
            let batch = self.pending.pop_front().unwrap();
            if let Err((error, rejected)) = writer.submit(batch) {
                self.free.push(rejected.slot);
                return Err(error);
            }
        }
        while let Some(batch) = writer.try_reclaim() {
            self.free.push(batch.slot);
        }
        writer.failure().map_or(Ok(()), Err)
    }

    fn acquire(
        &mut self,
        writer: &mut Writer,
        checkpoint: &mut impl FnMut() -> Result<()>,
    ) -> Result<Slot> {
        self.dispatch_copies(writer)?;
        if let Some(slot) = self.free.pop() {
            return Ok(slot);
        }
        let started = Instant::now();
        self.buffer_waits += 1;
        let result = (|| loop {
            checkpoint()?;
            if let Some(batch) = writer.reclaim_timeout(Duration::from_micros(100)) {
                self.free.push(batch.slot);
            }
            self.dispatch_copies(writer)?;
            if let Some(slot) = self.free.pop() {
                return Ok(slot);
            }
        })();
        self.buffer_wait += started.elapsed();
        result
    }

    pub(super) fn finish(
        &mut self,
        mut writer: Writer,
        complete: bool,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<FinishedSpool> {
        let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(self.copy.ptr)
        });
        let dispatch = if complete && drained.is_ok() {
            checkpoint().and_then(|()| self.dispatch_copies(&mut writer))
        } else {
            writer.discard();
            Ok(())
        };
        self.free
            .extend(self.pending.drain(..).map(|batch| batch.slot));
        let (spool, failure, finished) = writer.finish(checkpoint)?;
        self.free
            .extend(finished.into_iter().map(|batch| batch.slot));
        drained?;
        Ok(FinishedSpool {
            spool,
            failure: failure.or_else(|| dispatch.err()),
        })
    }

    // Return the source lease after its last lookup, independent of D2H and SSD.
    pub(super) fn resolve(
        &mut self,
        source: LineageSource<'_>,
        catalog: LineageCatalog<'_>,
        owner: &Stream,
        writer: &mut Writer,
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
            directory,
            max_series,
            bases,
            last_rows,
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
            || prepared.hash_count as usize != bases.len()
            || max_series == 0
        {
            return Err("SEGMENTED_WINDOW_LINEAGE_CATALOG".into());
        }

        let submitted = (|| -> Result<()> {
            let mut begin = 0;
            while begin < span.rows {
                checkpoint()?;
                let count = (span.rows - begin).min(TILE_ROWS);
                let slot = self.acquire(writer, &mut checkpoint)?;
                let tile = &self.device[self.next];
                let device_error = tile.buffer.at::<u32>(EDGE_BYTES);
                let queued = (|| -> Result<()> {
                    tile.copied.wait_on(owner)?;
                    crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                        cudaMemsetAsync(device_error.cast(), 0, size_of::<u32>(), owner.ptr)
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
                            tile.buffer.ptr.cast(),
                            TILE_ROWS,
                            device_error,
                            owner.ptr,
                            directory,
                            max_series,
                        )
                    })?;
                    self.produced.record(owner)?;
                    self.produced.wait_on(&self.copy)?;
                    crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                        cudaMemcpyAsync(
                            slot.ptr,
                            tile.buffer.ptr,
                            count as usize * size_of::<ResolvedEdge>(),
                            DEVICE_TO_HOST,
                            self.copy.ptr,
                        )
                    })?;
                    crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                        cudaMemcpyAsync(
                            slot.ptr.cast::<u8>().add(EDGE_BYTES).cast(),
                            device_error.cast(),
                            size_of::<u32>(),
                            DEVICE_TO_HOST,
                            self.copy.ptr,
                        )
                    })?;
                    tile.copied.record(&self.copy)?;
                    crate::generic_native::checked!(cudaEventRecord, unsafe {
                        cudaEventRecord(slot.ready, self.copy.ptr)
                    })
                })();
                if queued.is_err() {
                    // The event might not cover a partially submitted copy.
                    unsafe {
                        cudaStreamSynchronize(self.copy.ptr);
                    }
                    self.free.push(slot);
                    return queued;
                }
                self.next = (self.next + 1) % DEVICE_SLOTS;
                self.pending.push_back(Batch {
                    slot,
                    count,
                    shard,
                    rank,
                    parent_rows,
                    generators,
                    bases: Arc::clone(&bases),
                    last_rows,
                });
                begin += count;
            }
            Ok(())
        })();
        let released = self
            .produced
            .record(owner)
            .and_then(|()| self.produced.synchronize());
        if released.is_err() {
            // Keep the source span alive even if recording the event failed.
            unsafe {
                cudaStreamSynchronize(owner.ptr);
            }
        }
        submitted.and(released)?;
        self.dispatch_copies(writer)
    }
}
