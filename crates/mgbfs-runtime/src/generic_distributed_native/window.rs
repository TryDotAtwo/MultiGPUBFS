//! Bounded readback and CSR spooling for the generic native route owner.

use std::{collections::VecDeque, ffi::c_void, time::Duration};

use crate::path_csr::{Transition, TransitionSpool, TransitionWriter};
use crate::pinned_archive::Slot;
use mgbfs_core::Result;
use mgbfs_cuda::{ffi::*, generic_graph::GenericRouteRecord};

use super::{
    Buffer, DistributedAdvance, Event, GenericDistributedBfs, GenericDistributedMemoryPlan,
    GenericNativeBfs, QueueBank, Stream,
};

extern "C" {
    fn mgbfs_generic_resolve_incoming_edges_v3(
        state_bytes: u32,
        elements: u32,
        local: *const c_void,
        remote: *const c_void,
        local_meta: *const GenericRouteRecord,
        remote_meta: *const GenericRouteRecord,
        local_counts: *const u32,
        remote_counts: *const u32,
        rank: u32,
        world: u32,
        shard: u32,
        shards: u32,
        queue_capacity: u32,
        slots: *const u64,
        slot_count: u64,
        arena: *const c_void,
        stride: u32,
        first_child: u32,
        child_limit: u32,
        children: *mut WindowEdge,
        output_capacity: u32,
        begin: u32,
        length: u32,
        error: *mut u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Export only matched route metadata and child IDs. Candidate state vectors
/// remain on the device while the host writes the bounded transition runs.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct WindowEdge {
    parent: u64,
    child: u32,
    generator: u32,
}

const WINDOW_HOST_SLOTS: usize = 3;

struct WindowBatch {
    slot: Slot,
    senders: Vec<u32>,
    uniform_sender: u32,
    rows: usize,
    child_limit: u32,
    generators: u32,
    error_offset: Option<usize>,
}

pub(crate) struct WindowLineage {
    writer: Option<TransitionWriter<WindowBatch>>,
    spool: Option<TransitionSpool>,
}

impl WindowLineage {
    pub(crate) fn new(spool: TransitionSpool) -> Result<Self> {
        let mut transitions = Vec::with_capacity(GenericDistributedBfs::WINDOW_ROWS_PER_COPY);
        let writer = TransitionWriter::new(
            spool,
            WINDOW_HOST_SLOTS,
            move |spool, batch: &mut WindowBatch| {
                if let Some(offset) = batch.error_offset {
                    // The owner reports device overflow and resolver failures in the rank vote.
                    if unsafe { batch.slot.ptr.cast::<u8>().add(offset).cast::<u32>().read() } != 0
                    {
                        return Ok(());
                    }
                }
                let edges = unsafe {
                    std::slice::from_raw_parts(batch.slot.ptr.cast::<WindowEdge>(), batch.rows)
                };
                transitions.clear();
                for (index, edge) in edges.iter().enumerate() {
                    if edge.child >= batch.child_limit {
                        return Err("NATIVE_WINDOW_UNRESOLVED_EDGE".into());
                    }
                    if edge.generator >= batch.generators {
                        return Err("NATIVE_WINDOW_ROUTE_METADATA".into());
                    }
                    transitions.push(Transition {
                        child: u64::from(edge.child),
                        parent_rank: batch
                            .senders
                            .get(index)
                            .copied()
                            .unwrap_or(batch.uniform_sender),
                        parent: edge.parent,
                        generator: edge.generator,
                    });
                }
                spool.add(&mut transitions)
            },
        )?;
        Ok(Self {
            writer: Some(writer),
            spool: None,
        })
    }

    pub(crate) fn into_spool(
        self,
        advanced: &Result<DistributedAdvance>,
    ) -> Result<TransitionSpool> {
        self.spool.ok_or_else(|| {
            advanced
                .as_ref()
                .err()
                .cloned()
                .unwrap_or_else(|| "NATIVE_WINDOW_WRITER_UNFINISHED".into())
        })
    }
}

pub(super) fn finish_window_lineage(
    lineage: Option<(&mut WindowLineage, &Buffer, &mut WindowTile)>,
    checkpoint: Option<&dyn Fn() -> Result<()>>,
    path_error: &mut Option<String>,
) {
    if let Some((lineage, _, tile)) = lineage {
        if path_error.is_none() {
            *path_error = checkpoint.and_then(|check| check().err());
        }
        let result = tile.readback.finish(lineage, path_error.is_none());
        if path_error.is_none() {
            *path_error = result.err();
        }
    }
}

pub(super) enum WindowSenders<'a> {
    Uniform(u32),
    PerRow(&'a [u32]),
}

pub(super) struct WindowReadback {
    copy: Stream,
    produced: Event,
    copied: [Event; 2],
    pub(super) rows: usize,
    pending: VecDeque<WindowBatch>,
    free: Vec<WindowBatch>,
}

impl WindowReadback {
    fn new(rows: usize, device: i32) -> Result<Self> {
        // Reserve an aligned extra word for a compact owner's status snapshot.
        let bytes = rows * std::mem::size_of::<WindowEdge>() + std::mem::align_of::<WindowEdge>();
        crate::session_cache::prepare_pinned(bytes, WINDOW_HOST_SLOTS);
        let free = Slot::new_ring(bytes, WINDOW_HOST_SLOTS)?
            .into_iter()
            .map(|slot| WindowBatch {
                slot,
                senders: Vec::with_capacity(rows),
                uniform_sender: 0,
                rows: 0,
                child_limit: 0,
                generators: 0,
                error_offset: None,
            })
            .collect();
        Ok(Self {
            copy: Stream::new(device)?,
            produced: Event::new(device)?,
            copied: [Event::new(device)?, Event::new(device)?],
            rows,
            pending: VecDeque::with_capacity(WINDOW_HOST_SLOTS),
            free,
        })
    }

    pub(super) fn wait_for_scratch(&self, owner: *mut c_void, device_slot: usize) -> Result<()> {
        crate::generic_native::checked!(cudaStreamWaitEvent, unsafe {
            cudaStreamWaitEvent(owner, self.copied[device_slot].ptr, 0)
        })
    }

    fn service(&mut self, writer: &mut TransitionWriter<WindowBatch>) -> Result<()> {
        while let Some(batch) = self.pending.front() {
            match unsafe { cudaEventQuery(batch.slot.ready) } {
                0 => {}
                600 => break,
                status => return crate::generic_native::checked!(cudaEventQuery, status),
            }
            let batch = self.pending.pop_front().unwrap();
            if let Err((error, rejected)) = writer.submit(batch) {
                self.free.push(rejected);
                return Err(error);
            }
        }
        while let Some(batch) = writer.try_reclaim() {
            self.free.push(batch);
        }
        writer.failure().map_or(Ok(()), Err)
    }

    pub(super) fn enqueue(
        &mut self,
        lineage: &mut WindowLineage,
        ids: &Buffer,
        device_slot: usize,
        owner: *mut c_void,
        rows: usize,
        child_limit: u32,
        generators: u32,
        senders: WindowSenders<'_>,
        owner_error: Option<*const u32>,
        mut progress: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        if rows == 0
            || rows > self.rows
            || matches!(&senders, WindowSenders::PerRow(ranks) if ranks.len() != rows)
        {
            return Err("NATIVE_WINDOW_READBACK_ROWS".into());
        }
        let writer = lineage
            .writer
            .as_mut()
            .ok_or("NATIVE_WINDOW_WRITER_UNFINISHED")?;
        self.service(writer)?;
        while self.free.is_empty() {
            progress()?;
            if let Some(batch) = writer.reclaim_timeout(Duration::from_micros(100)) {
                self.free.push(batch);
            } else {
                std::thread::yield_now();
            }
            self.service(writer)?;
        }
        let mut batch = self.free.pop().unwrap();
        batch.rows = rows;
        batch.child_limit = child_limit;
        batch.generators = generators;
        batch.senders.clear();
        match senders {
            WindowSenders::Uniform(rank) => batch.uniform_sender = rank,
            WindowSenders::PerRow(ranks) => batch.senders.extend_from_slice(ranks),
        }
        let bytes = rows * std::mem::size_of::<WindowEdge>();
        batch.error_offset = owner_error.map(|_| bytes);
        let queued = (|| -> Result<()> {
            crate::generic_native::checked!(cudaEventRecord, unsafe {
                cudaEventRecord(self.produced.ptr, owner)
            })?;
            let copy = self.copy.ptr;
            crate::generic_native::checked!(cudaStreamWaitEvent, unsafe {
                cudaStreamWaitEvent(copy, self.produced.ptr, 0)
            })?;
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(batch.slot.ptr, ids.ptr, bytes, 2, copy)
            })?;
            if let Some(error) = owner_error {
                crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                    cudaMemcpyAsync(
                        batch.slot.ptr.cast::<u8>().add(bytes).cast(),
                        error.cast(),
                        4,
                        2,
                        copy,
                    )
                })?;
            }
            crate::generic_native::checked!(cudaEventRecord, unsafe {
                cudaEventRecord(self.copied[device_slot].ptr, copy)
            })?;
            crate::generic_native::checked!(cudaEventRecord, unsafe {
                cudaEventRecord(batch.slot.ready, copy)
            })
        })();
        if queued.is_err() {
            // A failed event record may leave a transfer not covered by slot.ready.
            unsafe { cudaStreamSynchronize(self.copy.ptr) };
            self.free.push(batch);
        } else {
            self.pending.push_back(batch);
        }
        queued
    }

    pub(super) fn finish(&mut self, lineage: &mut WindowLineage, complete: bool) -> Result<()> {
        let Some(mut writer) = lineage.writer.take() else {
            return Ok(());
        };
        let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(self.copy.ptr)
        });
        let submitted = if complete && drained.is_ok() {
            self.service(&mut writer)
        } else {
            writer.discard();
            Ok(())
        };
        self.free.extend(self.pending.drain(..));
        let (spool, failure, returned) = writer.finish()?;
        self.free.extend(returned);
        lineage.spool = Some(spool);
        drained.and(submitted)?;
        if complete {
            if let Some(error) = failure {
                return Err(error);
            }
        }
        Ok(())
    }
}

pub(crate) struct WindowTile {
    pub(super) readback: WindowReadback,
    pub(super) second_device: Option<Buffer>,
    pub(super) next_device: usize,
    senders: Vec<u32>,
    counts: Vec<u32>,
}
impl WindowTile {
    pub(super) fn matches_plan(&self, plan: &GenericDistributedMemoryPlan) -> bool {
        self.readback.rows == GenericDistributedBfs::window_tile_rows(plan)
            && self.second_device.is_some() == plan.compact_transport
            && self.counts.len() == plan.world as usize * plan.shards as usize
    }
    pub(crate) fn new(plan: &GenericDistributedMemoryPlan, device: i32) -> Result<Self> {
        let tile_rows = GenericDistributedBfs::window_tile_rows(plan);
        Ok(Self {
            readback: WindowReadback::new(tile_rows, device)?,
            second_device: plan
                .compact_transport
                .then(|| Buffer::new(GenericDistributedBfs::window_plan_edge_bytes(plan), device))
                .transpose()?,
            next_device: 0,
            senders: vec![0; tile_rows],
            counts: vec![0; plan.world as usize * plan.shards as usize],
        })
    }
    pub(super) fn write(
        &mut self,
        b: &GenericNativeBfs,
        p: &GenericDistributedMemoryPlan,
        rank: u32,
        source: &QueueBank,
        inbox: &QueueBank,
        exchanged_counts: Option<&[u32]>,
        ids: &Buffer,
        stream: *mut c_void,
        lineage: &mut WindowLineage,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        checkpoint()?;
        let mut counters = [0u32; 3];
        let q = p.queue_capacity as usize;
        let Self {
            readback,
            senders,
            counts,
            ..
        } = self;
        let tile_rows = readback.rows;
        let queues = counts.len();
        let local_begin = rank as usize * p.shards as usize;
        let local_end = local_begin + p.shards as usize;
        let copied = (|| -> Result<()> {
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(counters.as_mut_ptr().cast(), b.control.ptr, 12, 2, stream)
            })?;
            if let Some(exchanged) = exchanged_counts {
                let (outgoing, incoming) = exchanged.split_at(queues);
                counts[..local_begin].copy_from_slice(&incoming[..local_begin]);
                counts[local_begin..local_end].copy_from_slice(&outgoing[local_begin..local_end]);
                counts[local_end..].copy_from_slice(&incoming[local_end..]);
            } else {
                if p.world > 1 {
                    crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                        cudaMemcpyAsync(
                            counts.as_mut_ptr().cast(),
                            inbox.counts.ptr,
                            queues * 4,
                            2,
                            stream,
                        )
                    })?;
                }
                crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                    cudaMemcpyAsync(
                        counts.as_mut_ptr().add(local_begin).cast(),
                        source.counts.at::<c_void>(local_begin * 4),
                        p.shards as usize * 4,
                        2,
                        stream,
                    )
                })?;
            }
            Ok(())
        })();
        let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        });
        copied.and(drained)?;
        if counters[2] != 0 {
            return Ok(());
        }
        let child_count = counters[0]
            .checked_sub(b.visited_used)
            .ok_or("NATIVE_WINDOW_CHILD_COUNT")?;
        let written = (|| -> Result<()> {
            let mut used = 0;
            for shard in 0..p.shards {
                for sender in 0..p.world as usize {
                    let queue = sender * p.shards as usize + shard as usize;
                    let count = counts[queue] as usize;
                    if count > q {
                        return Err("NATIVE_WINDOW_ROUTE_COUNT".into());
                    }
                    let mut begin = 0;
                    while begin < count {
                        checkpoint()?;
                        if used == 0 {
                            readback.wait_for_scratch(stream, 0)?;
                        }
                        let n = (count - begin).min(tile_rows - used);
                        crate::generic_native::checked!(
                            mgbfs_generic_resolve_incoming_edges_v3,
                            unsafe {
                                mgbfs_generic_resolve_incoming_edges_v3(
                                    p.state_bytes,
                                    p.elements,
                                    source.states.ptr,
                                    inbox.states.ptr,
                                    source.records.ptr.cast(),
                                    inbox.records.ptr.cast(),
                                    source.counts.ptr.cast(),
                                    inbox.counts.ptr.cast(),
                                    rank,
                                    p.world,
                                    shard,
                                    p.shards,
                                    p.queue_capacity,
                                    b.slots.ptr.cast(),
                                    p.table_slots,
                                    b.visited.ptr,
                                    b.arena_stride,
                                    b.visited_used,
                                    counters[0],
                                    ids.at(used * GenericDistributedBfs::WINDOW_EDGE_BYTES),
                                    (tile_rows - used) as u32,
                                    (sender * q + begin) as u32,
                                    n as u32,
                                    b.control.at(8),
                                    stream,
                                )
                            }
                        )?;
                        senders[used..used + n].fill(sender as u32);
                        used += n;
                        begin += n;
                        if used == tile_rows {
                            readback.enqueue(
                                lineage,
                                ids,
                                0,
                                stream,
                                used,
                                child_count,
                                b.generators,
                                WindowSenders::PerRow(senders),
                                None,
                                checkpoint,
                            )?;
                            used = 0;
                        }
                    }
                }
            }
            if used != 0 {
                readback.enqueue(
                    lineage,
                    ids,
                    0,
                    stream,
                    used,
                    child_count,
                    b.generators,
                    WindowSenders::PerRow(&senders[..used]),
                    None,
                    checkpoint,
                )?;
            }
            checkpoint()?;
            Ok(())
        })();
        if written.is_err() {
            let _ = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                cudaStreamSynchronize(stream)
            });
        }
        written
    }
}
