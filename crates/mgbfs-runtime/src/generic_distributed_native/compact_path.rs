//! Native path candidate generation and owner consumption on compact packets.
use super::window::{WindowEdge, WindowSenders};
use super::{Event, GenericDistributedBfs, WindowLineage, WindowTile};
use crate::generic_distributed_memory::GenericDistributedMemoryPlan as Plan;
use crate::generic_native::{Buffer, GenericNativeBfs, Stream};
use crate::segmented_distributed_native::{
    packet::{compact_route_kernel, CompactRouteFn, PacketError, RawPacketTransport, RawShardRun},
    BitmapDeviceDomain,
};
use crate::segmented_packet_transport::{PacketEndpoint, ShardRun};
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{cudaEventRecord, cudaMemcpyAsync, cudaMemsetAsync, cudaStreamSynchronize};
use mgbfs_cuda::generic_graph::GenericRouteRecord;
use std::{ffi::c_void, mem::size_of, ptr};

const TARGET_CANDIDATES: u32 = 16_384;

extern "C" {
    fn mgbfs_segmented_batch_route_scratch_bytes(
        children: u32,
        buckets: u32,
        bytes: *mut usize,
    ) -> i32;
}

#[repr(C)]
struct Input {
    state_bytes: u32,
    elements: u32,
    generators: u32,
    rows: u32,
    source: u32,
    shard: u32,
    states: *const c_void,
    records: *const GenericRouteRecord,
}

#[repr(C)]
struct Owner {
    capacity: u32,
    slot_count: u64,
    slots: *mut u64,
    states: *mut c_void,
    visited_count: *mut u32,
    future: *mut u32,
    future_count: *mut u32,
    lineage: *mut c_void,
    error: *mut u32,
}

type Accept = unsafe extern "C" fn(*const Input, *const Owner, *mut c_void) -> i32;
type Resolve = unsafe extern "C" fn(
    *const Input,
    *const Owner,
    u32,
    u32,
    u32,
    *mut WindowEdge,
    u32,
    *mut c_void,
) -> i32;

#[derive(Clone, Copy)]
pub(super) struct OwnerKernels {
    accept: Accept,
    resolve: Resolve,
}

impl OwnerKernels {
    fn load() -> Option<Self> {
        let symbol = |name: &[u8]| unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr().cast()) };
        let accept = symbol(b"mgbfs_generic_accept_compact_path_v1\0");
        let resolve = symbol(b"mgbfs_generic_resolve_compact_window_edges_v1\0");
        if accept.is_null() || resolve.is_null() {
            return None;
        }
        Some(Self {
            accept: unsafe { std::mem::transmute::<*mut c_void, Accept>(accept) },
            resolve: unsafe { std::mem::transmute::<*mut c_void, Resolve>(resolve) },
        })
    }
}

pub(super) fn abi_ready() -> bool {
    OwnerKernels::load().is_some()
        && compact_route_kernel().is_ok()
        && crate::segmented_packet_transport::packet_wire_kernel().is_ok()
}

fn candidate_limit(generators: u32) -> u32 {
    TARGET_CANDIDATES.max(generators)
}

pub(super) fn route_scratch_bound(world: u32, shards: u32, generators: u32) -> Result<usize> {
    route_scratch_bytes(candidate_limit(generators), world, shards)
}

fn route_scratch_bytes(rows: u32, world: u32, shards: u32) -> Result<usize> {
    let buckets = world
        .checked_mul(shards)
        .ok_or("NATIVE_PATH_ROUTE_BUCKETS")?;
    let mut bytes = 0;
    crate::generic_native::checked!(mgbfs_segmented_batch_route_scratch_bytes, unsafe {
        mgbfs_segmented_batch_route_scratch_bytes(rows, buckets, &mut bytes)
    })?;
    Ok(bytes.max(1))
}

pub(super) struct Geometry {
    pub(super) packet_rows: u32,
    pub(super) parent_rows: u32,
    payload_width: usize,
    metadata_words: usize,
    scratch_bytes: usize,
}

impl Geometry {
    pub(super) fn new(plan: &Plan, generators: u32) -> Result<Self> {
        if !plan.compact_transport
            || plan.world < 2
            || plan.compact_packet_slots != 3
            || plan.compact_receive_slots != 3
            || plan.packed_candidates()
            || plan.parent_transport
        {
            return Err("NATIVE_PATH_COMPACT_GEOMETRY".into());
        }
        let packet_rows = plan.queue_capacity.min(candidate_limit(generators));
        let parent_rows = plan.batch.min(packet_rows / generators);
        let metadata_words = (plan.world * plan.shards) as usize + 1;
        let scratch_bytes = route_scratch_bytes(parent_rows * generators, plan.world, plan.shards)?;
        Ok(Self {
            packet_rows,
            parent_rows,
            payload_width: plan.queue_payload_bytes(),
            metadata_words,
            scratch_bytes,
        })
    }

    fn producer_requirements(&self) -> [usize; 4] {
        [
            self.packet_rows as usize * size_of::<GenericRouteRecord>(),
            self.packet_rows as usize * self.payload_width,
            self.metadata_words * size_of::<u32>(),
            self.scratch_bytes,
        ]
    }

    pub(super) fn buffer_requirements(&self, shards: u32) -> Result<Vec<usize>> {
        let mut sizes = Vec::from(self.producer_requirements());
        sizes.extend(RawPacketTransport::device_buffer_requirements(
            shards,
            self.packet_rows,
            self.payload_width,
        )?);
        Ok(sizes)
    }
}

pub(super) struct Producer {
    pub(super) stream: Stream,
    pub(super) geometry: Geometry,
    pub(super) records: Buffer,
    pub(super) states: Buffer,
    metadata: Buffer,
    scratch: Buffer,
    route: CompactRouteFn,
    host_metadata: Vec<u32>,
}

impl Producer {
    pub(super) fn new(geometry: Geometry, device: i32) -> Result<Self> {
        let [records, states, metadata, scratch] = geometry.producer_requirements();
        Ok(Self {
            stream: Stream::new(device)?,
            records: Buffer::new(records, device)?,
            states: Buffer::new(states, device)?,
            metadata: Buffer::new(metadata, device)?,
            scratch: Buffer::new(scratch, device)?,
            route: compact_route_kernel()?,
            host_metadata: vec![0; geometry.metadata_words],
            geometry,
        })
    }

    pub(super) fn route(
        &mut self,
        plan: &Plan,
        bfs: &GenericNativeBfs,
        rank: u32,
        owner_map: &Buffer,
        owner_cuts: Option<&Buffer>,
        window: bool,
        begin: u32,
        rows: u32,
    ) -> Result<()> {
        let (parents, origin) = if window {
            (&bfs.parents, begin)
        } else {
            (&bfs.visited, bfs.current_start + begin)
        };
        let last_word = self.geometry.metadata_words - 1;
        let error = self.metadata.at::<u32>(last_word * size_of::<u32>());
        let stream = self.stream.ptr;
        let queued = (|| -> Result<()> {
            crate::generic_native::checked!(cudaMemsetAsync, unsafe {
                cudaMemsetAsync(error.cast(), 0, size_of::<u32>(), stream)
            })?;
            crate::generic_native::checked!(mgbfs_segmented_compact_route_v1, unsafe {
                (self.route)(
                    0,
                    0,
                    BitmapDeviceDomain::EMPTY,
                    plan.state_bytes,
                    bfs.kind,
                    plan.elements,
                    bfs.rows,
                    bfs.cols,
                    bfs.generators,
                    parents.at::<c_void>(origin as usize * plan.state_bytes as usize),
                    rows,
                    u64::from(bfs.arena_stride),
                    bfs.perms.ptr.cast(),
                    bfs.matrices.ptr.cast(),
                    bfs.moduli.ptr.cast(),
                    bfs.seed,
                    bfs.hash_bits,
                    plan.world,
                    rank,
                    plan.shards,
                    self.geometry.packet_rows,
                    u64::from(origin),
                    owner_map.ptr.cast(),
                    owner_cuts.map_or(ptr::null(), |v| v.ptr.cast()),
                    self.records.ptr.cast(),
                    self.metadata.ptr.cast(),
                    error,
                    self.states.ptr,
                    self.scratch.ptr,
                    self.scratch.bytes,
                    stream,
                    0,
                    0,
                    self.geometry.packet_rows,
                )
            })?;
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(
                    self.host_metadata.as_mut_ptr().cast(),
                    self.metadata.ptr,
                    self.metadata.bytes,
                    2,
                    stream,
                )
            })
        })();
        let drained = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        });
        queued.and(drained)?;
        let error = self.host_metadata[last_word];
        if error != 0 {
            return Err(format!("NATIVE_PATH_COMPACT_ROUTE_{error}"));
        }
        let counts = &self.host_metadata[..last_word];
        if counts.iter().map(|&count| u64::from(count)).sum::<u64>()
            != u64::from(rows) * u64::from(bfs.generators)
        {
            return Err("NATIVE_PATH_COMPACT_ROUTE_COUNT".into());
        }
        Ok(())
    }

    fn counts(&self) -> &[u32] {
        &self.host_metadata[..self.geometry.metadata_words - 1]
    }
}

pub(super) struct OwnerBatch<'a> {
    pub(super) kernels: OwnerKernels,
    pub(super) plan: &'a Plan,
    pub(super) bfs: &'a GenericNativeBfs,
    pub(super) ready: &'a Event,
    pub(super) parents: Option<&'a Buffer>,
    pub(super) lineage: Option<(&'a mut WindowLineage, &'a Buffer, &'a mut WindowTile)>,
    pub(super) checkpoint: &'a dyn Fn() -> Result<()>,
    pub(super) path_error: &'a mut Option<String>,
}

impl OwnerBatch<'_> {
    pub(super) fn consume(
        &mut self,
        runs: &[RawShardRun],
        pump: &mut dyn FnMut(&[(usize, *mut c_void)]) -> Result<()>,
    ) -> Result<()> {
        if self.path_error.is_some() {
            return Ok(());
        }
        let b = self.bfs;
        let stream = b.stream.ptr;
        let owner = Owner {
            capacity: self.plan.capacity,
            slot_count: self.plan.table_slots,
            slots: b.slots.ptr.cast(),
            states: b.visited.ptr,
            visited_count: b.control.at(0),
            future: b.future.ptr.cast(),
            future_count: b.control.at(4),
            lineage: self.parents.map_or(ptr::null_mut(), |parents| parents.ptr),
            error: b.control.at(8),
        };
        for run in runs {
            let input = Input {
                state_bytes: self.plan.state_bytes,
                elements: self.plan.elements,
                generators: b.generators,
                rows: run.rows,
                source: run.source_rank,
                shard: run.shard as u32,
                states: run.states,
                records: run.records.cast(),
            };
            let queued = (|| -> Result<()> {
                crate::generic_native::checked!(mgbfs_generic_accept_compact_path_v1, unsafe {
                    (self.kernels.accept)(&input, &owner, stream)
                })?;
                if let Some((lineage, primary, tile)) = self.lineage.as_mut() {
                    let mut begin = 0;
                    while begin < run.rows {
                        if let Err(error) = (self.checkpoint)() {
                            *self.path_error = Some(error);
                            break;
                        }
                        let rows = (run.rows - begin).min(tile.readback.rows as u32);
                        let device_slot = tile.next_device;
                        let device = if device_slot == 0 {
                            *primary
                        } else {
                            tile.second_device.as_ref().ok_or("NATIVE_WINDOW_DEVICE_SLOT")?
                        };
                        tile.readback.wait_for_scratch(stream, device_slot)?;
                        crate::generic_native::checked!(
                            mgbfs_generic_resolve_compact_window_edges_v1,
                            unsafe {
                                (self.kernels.resolve)(
                                    &input,
                                    &owner,
                                    0,
                                    begin,
                                    rows,
                                    device.ptr.cast(),
                                    rows,
                                    stream,
                                )
                            }
                        )?;
                        // The copy stream reads tile-local status, so the next
                        // owner can update the shared error word before D2H ends.
                        let error_snapshot = device.at::<u32>(tile.readback.rows * size_of::<WindowEdge>());
                        crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                            cudaMemcpyAsync(error_snapshot.cast(), owner.error.cast(), size_of::<u32>(), 3, stream)
                        })?;
                        if let Err(error) = tile.readback.enqueue(
                            lineage, device, device_slot, stream, rows as usize, self.plan.capacity,
                            b.generators, WindowSenders::Uniform(run.source_rank),
                            Some(error_snapshot),
                            || {
                                (self.checkpoint)()?;
                                pump(&[(run.shard, stream)])
                            },
                        ) {
                            *self.path_error = Some(error);
                            break;
                        }
                        tile.next_device ^= 1;
                        begin += rows;
                    }
                }
                Ok(())
            })();
            // Release the transport lease after its last owner read. The
            // writer retains the independent pinned slot through SSD work.
            let drained = wait_owner(self.ready, run.shard, stream, pump);
            queued.and(drained)?;
            if self.path_error.is_some() {
                break;
            }
        }
        Ok(())
    }
}

fn wait_owner(
    ready: &Event,
    shard: usize,
    stream: *mut c_void,
    pump: &mut dyn FnMut(&[(usize, *mut c_void)]) -> Result<()>,
) -> Result<()> {
    let waited = (|| -> Result<()> {
        crate::generic_native::checked!(cudaEventRecord, unsafe {
            cudaEventRecord(ready.ptr, stream)
        })?;
        // Only the network producer runs while this stream owns the packet.
        loop {
            match unsafe { mgbfs_cuda::native_owner::cudaEventQuery(ready.ptr) } {
                0 => return Ok(()),
                600 => pump(&[(shard, stream)])?,
                status => return crate::generic_native::checked!(cudaEventQuery, status),
            }
            std::thread::yield_now();
        }
    })();
    if waited.is_err() {
        // Keep any in-flight D2H host pointers alive on the error path.
        unsafe { cudaStreamSynchronize(stream) };
    }
    waited
}

pub(super) fn owner_kernels() -> Result<OwnerKernels> {
    OwnerKernels::load().ok_or_else(|| "NATIVE_PATH_COMPACT_OWNER_ABI_UNAVAILABLE".into())
}

fn runs_for_peer(counts: &[u32], runs: &mut Vec<ShardRun>) -> u32 {
    runs.clear();
    let mut rows = 0;
    for (shard, &count) in counts.iter().enumerate() {
        if count != 0 {
            runs.push(ShardRun {
                shard: shard as u32,
                rows: count,
            });
            rows += count;
        }
    }
    rows
}

pub(super) struct Workspace {
    pub(super) transport: RawPacketTransport,
    producer: Producer,
    kernels: OwnerKernels,
}

impl Workspace {
    pub(super) fn new(
        plan: &Plan,
        generators: u32,
        rank: u32,
        id: &[u8; 128],
        endpoint: PacketEndpoint,
        device: i32,
    ) -> Result<Self> {
        let geometry = Geometry::new(plan, generators)?;
        let transport = RawPacketTransport::new(
            endpoint,
            id,
            plan.world,
            rank,
            plan.shards,
            geometry.packet_rows,
            geometry.payload_width,
            device,
        )?;
        Ok(Self {
            transport,
            producer: Producer::new(geometry, device)?,
            kernels: owner_kernels()?,
        })
    }

    pub(super) fn expand(
        &mut self,
        bfs: &GenericDistributedBfs,
        parents: Option<&Buffer>,
        lineage: Option<(&mut WindowLineage, &Buffer, &mut WindowTile)>,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<Option<String>> {
        let plan = &bfs.plan;
        let b = &bfs.bfs;
        let mut path_error = None;
        let window = lineage.is_some();
        // Both directions share the wire. A side's BFS depth can repeat
        // while the packet epoch advances once per drained expansion.
        let epoch = self.transport.epoch();
        let rank = bfs.rank;
        let Self {
            transport,
            producer,
            kernels,
            ..
        } = self;
        let mut owner = OwnerBatch {
            kernels: *kernels,
            plan,
            bfs: b,
            ready: &bfs.ready,
            parents,
            lineage,
            checkpoint,
            path_error: &mut path_error,
        };
        let mut runs = Vec::with_capacity(plan.shards as usize);
        let mut begin = 0;
        while begin < b.count {
            owner.checkpoint(checkpoint);
            if owner.path_error.is_some() {
                break;
            }
            while transport.has_pending_local(0)? {
                service(transport, epoch, &mut owner)?;
            }
            transport.wait_source_read(0, &producer.stream)?;
            let rows = (b.count - begin).min(producer.geometry.parent_rows);
            producer.route(
                plan,
                b,
                rank,
                &bfs.owner_map,
                bfs.owner_cuts.as_ref(),
                window,
                begin,
                rows,
            )?;
            let (records, states) = (&producer.records, &producer.states);
            let width = producer.geometry.payload_width;
            let mut offset = 0usize;
            for (peer, counts) in producer
                .counts()
                .chunks_exact(plan.shards as usize)
                .enumerate()
            {
                let count = runs_for_peer(counts, &mut runs);
                if count == 0 {
                    continue;
                }
                if peer as u32 == rank {
                    for run in &runs {
                        let span = RawShardRun {
                            shard: run.shard as usize,
                            source_rank: rank,
                            rows: run.rows,
                            records: records.at(offset * size_of::<GenericRouteRecord>()),
                            states: states.at(offset * width),
                        };
                        while !unsafe { transport.offer_local(0, span) }? {
                            service(transport, epoch, &mut owner)?;
                        }
                        offset += run.rows as usize;
                    }
                } else {
                    while !unsafe {
                        transport.stage(
                            epoch,
                            peer as u32,
                            &runs,
                            records.at(offset * size_of::<GenericRouteRecord>()),
                            states.at(offset * width),
                        )
                    }? {
                        service(transport, epoch, &mut owner)?;
                    }
                    offset += count as usize;
                }
            }
            transport.record_source_read(0)?;
            service(transport, epoch, &mut owner)?;
            begin += rows;
        }
        while !transport.seal(epoch).map_err(wire_error)? {
            service(transport, epoch, &mut owner)?;
        }
        while !transport.committed(epoch).map_err(wire_error)? {
            service(transport, epoch, &mut owner)?;
        }
        Ok(path_error)
    }
}

impl OwnerBatch<'_> {
    fn checkpoint(&mut self, check: &dyn Fn() -> Result<()>) {
        if self.path_error.is_none() {
            *self.path_error = check().err();
        }
    }
}

fn service(
    transport: &mut RawPacketTransport,
    depth: u64,
    owner: &mut OwnerBatch<'_>,
) -> Result<()> {
    let progressed = transport
        .progress(depth, &|| 0, 1, |runs, pump| owner.consume(runs, pump))
        .map_err(wire_error)?;
    if !progressed {
        std::thread::yield_now();
    }
    Ok(())
}

fn wire_error(error: PacketError) -> String {
    match error {
        PacketError::Failure(error) => error,
        PacketError::Terminal(code) => format!("NATIVE_PATH_COMPACT_TRANSPORT_{code}"),
    }
}
