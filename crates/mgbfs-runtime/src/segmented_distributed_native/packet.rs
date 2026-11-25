//! Segmented BFS packet staging, device ownership and NCCL rendezvous.
use super::{
    cudaMemcpy, cudaMemcpyAsync, cudaMemsetAsync, cudaStreamQuery, cudaStreamSynchronize,
    mgbfs_nccl_bind_cancel, BitmapDeviceDomain, BitmapPacketBatchFn, BitmapPacketOwnerFn, Buffer,
    ByteWireOwnerFn, GenericRouteRecord, OwnerInputs, Pack4PacketOwnerFn, Pack4RouteArgs,
    PinnedStatus, Plan, Queue, QueueSide, Range, SegmentStrategy, SegmentedDistributedBfs, Stream,
};
use crate::segmented_packet_transport::{
    decode_runs, encode_runs, packet_wire_kernel, Frame as PacketFrame, Packet, PacketBytes,
    PacketClient, PacketEndpoint, PacketJobToken, PacketWireFn, ShardRun, PACKET_CANCELLED,
    PACKET_DEADLINE, PACKET_LINEAGE_DISK_FULL, PACKET_LINEAGE_LIMIT, REMOTE_FAILURE,
    SHARD_DESCRIPTOR_BYTES,
};
use mgbfs_core::Result;
use mgbfs_cuda::ffi::{
    cudaEventCreateWithFlags, cudaEventDestroy, cudaEventRecord, cudaStreamWaitEvent,
};
use std::{
    collections::VecDeque,
    ffi::c_void,
    ptr,
    time::{Duration, Instant},
};

mod owner_banks;
mod raw;
mod receive;
pub(crate) use raw::{RawPacketTransport, RawShardRun};
use receive::{ReceiveCarousel, ReceiveSlot};

const PACKET_ROUTE_STATUS_WORDS: usize = 2;
const PACKET_TIMEOUT: Duration = Duration::from_secs(120);
pub(super) const PACKET_ROUTE_STATUS_BYTES: usize =
    PACKET_ROUTE_STATUS_WORDS * std::mem::size_of::<u32>();

#[derive(Clone, Copy)]
pub(super) struct PacketOwnerShard {
    pub(super) shard: usize,
    pub(super) queue: QueueSide,
    pub(super) rows: u32,
    pub(super) byte_wire: bool,
}

struct RouteSpan {
    queue: u32,
    rows: u32,
    tile_begin: u32,
}

fn read_route_counts(counts: &[u32], upper: u32, spans: &mut Vec<RouteSpan>) -> Result<(u32, u32)> {
    let (mut produced, mut maximum) = (0u32, 0);
    for (queue, &rows) in counts.iter().enumerate() {
        if rows == 0 {
            continue;
        }
        if rows > upper {
            return Err("SEGMENTED_ROUTE_COUNT_BOUND".into());
        }
        spans.push(RouteSpan {
            queue: queue as u32,
            rows,
            tile_begin: produced,
        });
        produced = produced
            .checked_add(rows)
            .ok_or("SEGMENT_PACKET_PRODUCER_SIZE")?;
        maximum = maximum.max(rows);
    }
    Ok((produced, maximum))
}

// Every span is nonempty and individually fits the queue. This always uses
// at least the first span of a nonempty peer and stops before its credit exceeds
// the configured window.
fn packet_window(
    spans: &[RouteSpan],
    shards: u32,
    capacity: u32,
    runs: &mut Vec<ShardRun>,
) -> (usize, u32) {
    runs.clear();
    let mut total = 0;
    for span in spans {
        if span.rows > capacity - total {
            break;
        }
        runs.push(ShardRun {
            shard: span.queue % shards,
            rows: span.rows,
        });
        total += span.rows;
    }
    (runs.len(), total)
}

type Pack4PacketRouteFn =
    unsafe extern "C" fn(Pack4RouteArgs, *mut c_void, usize, u32, *mut c_void) -> i32;
pub(crate) type CompactRouteFn = unsafe extern "C" fn(
    u32,
    u32,
    BitmapDeviceDomain,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *const c_void,
    u32,
    u64,
    *const u32,
    *const i64,
    *const u32,
    u64,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
    *const u32,
    *const u64,
    *mut GenericRouteRecord,
    *mut u32,
    *mut u32,
    *mut c_void,
    *mut c_void,
    usize,
    *mut c_void,
    u32,
    u64,
    u32,
) -> i32;
type PacketBinaryFn = unsafe extern "C" fn(
    *const c_void,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    u32,
    *mut c_void,
) -> i32;
type PacketRangeFn = unsafe extern "C" fn(
    *const c_void,
    u32,
    u32,
    u32,
    i64,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    u32,
    *mut c_void,
) -> i32;

#[derive(Clone, Copy)]
pub(super) enum PacketCodec {
    Binary(u32, PacketBinaryFn),
    Range16(i64, PacketRangeFn),
    Pack4(PacketBinaryFn),
}

impl PacketCodec {
    unsafe fn transform(
        self,
        header: *const c_void,
        runs: u32,
        rows: u32,
        elements: u32,
        states: *mut c_void,
        records: *mut c_void,
        packet: *mut c_void,
        decode: u32,
        stream: *mut c_void,
    ) -> i32 {
        match self {
            Self::Binary(_, codec) | Self::Pack4(codec) => unsafe {
                codec(
                    header, runs, rows, elements, states, records, packet, decode, stream,
                )
            },
            Self::Range16(base, codec) => unsafe {
                codec(
                    header, runs, rows, elements, base, states, records, packet, decode, stream,
                )
            },
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PacketConfig {
    pub(super) endpoint: PacketEndpoint,
    wire: PacketWireFn,
    pub(super) bitmap_owner: Option<BitmapPacketOwnerFn>,
    pub(super) bitmap_batch: Option<BitmapPacketBatchFn>,
    pub(super) pack4_owner: Option<Pack4PacketOwnerFn>,
    pub(super) pack4_wire_owner: Option<Pack4PacketOwnerFn>,
    pub(super) byte_wire_owner: Option<(u32, ByteWireOwnerFn)>,
    compact_route: Option<CompactRouteFn>,
    pack4_route: Option<Pack4PacketRouteFn>,
    pub(super) codec: Option<PacketCodec>,
}

pub(crate) fn compact_route_kernel() -> Result<CompactRouteFn> {
    let symbol = unsafe {
        libc::dlsym(
            libc::RTLD_DEFAULT,
            b"mgbfs_segmented_compact_route_v1\0".as_ptr().cast(),
        )
    };
    if symbol.is_null() {
        return Err("SEGMENT_PACKET_COMPACT_ROUTE_ABI_UNAVAILABLE".into());
    }
    Ok(unsafe { std::mem::transmute::<*mut c_void, CompactRouteFn>(symbol) })
}

impl PacketConfig {
    fn raw(endpoint: PacketEndpoint) -> Result<Self> {
        Ok(Self {
            endpoint,
            wire: packet_wire_kernel()?,
            bitmap_owner: None,
            bitmap_batch: None,
            pack4_owner: None,
            pack4_wire_owner: None,
            byte_wire_owner: None,
            compact_route: None,
            pack4_route: None,
            codec: None,
        })
    }

    pub(super) fn load(
        world: u32,
        rank: u32,
        bitmap: bool,
        pack4: bool,
        byte_bits: Option<u32>,
        byte_wire_requested: bool,
        range_base: Option<i64>,
        pack4_wire: bool,
        job_token: Option<PacketJobToken>,
    ) -> Result<Option<Self>> {
        let Some(endpoint) = PacketEndpoint::from_env(world, rank, job_token)? else {
            return Ok(None);
        };
        let wire = packet_wire_kernel()?;
        let mut bitmap_owner = None;
        let mut bitmap_batch = None;
        let mut pack4_owner = None;
        let mut pack4_route = None;
        if pack4 {
            let owner = unsafe {
                libc::dlsym(
                    libc::RTLD_DEFAULT,
                    b"mgbfs_pack4_owner_accept_packet_v1\0".as_ptr().cast(),
                )
            };
            let route = unsafe {
                libc::dlsym(
                    libc::RTLD_DEFAULT,
                    b"mgbfs_pack4_compact_route_v1\0".as_ptr().cast(),
                )
            };
            if owner.is_null() || route.is_null() {
                return Err("SEGMENT_PACKET_PACK4_ABI_UNAVAILABLE".into());
            }
            pack4_owner =
                Some(unsafe { std::mem::transmute::<*mut c_void, Pack4PacketOwnerFn>(owner) });
            pack4_route =
                Some(unsafe { std::mem::transmute::<*mut c_void, Pack4PacketRouteFn>(route) });
        } else if bitmap {
            let atomic = unsafe {
                libc::dlsym(
                    libc::RTLD_DEFAULT,
                    b"mgbfs_bitmap_accept_packet_v1\0".as_ptr().cast(),
                )
            };
            let batch = unsafe {
                libc::dlsym(
                    libc::RTLD_DEFAULT,
                    b"mgbfs_bitmap_accept_batch_packet_v1\0".as_ptr().cast(),
                )
            };
            if atomic.is_null() || batch.is_null() {
                return Err("SEGMENT_PACKET_BITMAP_OWNER_ABI_UNAVAILABLE".into());
            }
            bitmap_owner =
                Some(unsafe { std::mem::transmute::<*mut c_void, BitmapPacketOwnerFn>(atomic) });
            bitmap_batch =
                Some(unsafe { std::mem::transmute::<*mut c_void, BitmapPacketBatchFn>(batch) });
        }
        let compact_route = Some(compact_route_kernel()?);
        let pack4_codec_mode = std::env::var("MGBFS_SEGMENT_PACKET_PACK4");
        let pack4_codec = if pack4_wire && pack4_codec_mode.as_deref() != Ok("0") {
            let symbol = unsafe {
                libc::dlsym(
                    libc::RTLD_DEFAULT,
                    b"mgbfs_segmented_packet_pack4_v1\0".as_ptr().cast(),
                )
            };
            if symbol.is_null() && pack4_codec_mode.as_deref() == Ok("1") {
                return Err("SEGMENT_PACKET_PACK4_CODEC_ABI_UNAVAILABLE".into());
            }
            (!symbol.is_null())
                .then(|| unsafe { std::mem::transmute::<*mut c_void, PacketBinaryFn>(symbol) })
        } else {
            None
        };
        let codec = if let Some(encode) = pack4_codec {
            Some(PacketCodec::Pack4(encode))
        } else if let Some(bits) = byte_bits {
            let name: &[u8] = if bits == 1 {
                b"mgbfs_segmented_packet_binary_v2\0"
            } else {
                b"mgbfs_segmented_packet_quaternary_v1\0"
            };
            let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr().cast()) };
            if symbol.is_null() {
                return Err("SEGMENT_PACKET_BINARY_ABI_UNAVAILABLE".into());
            }
            Some(PacketCodec::Binary(bits, unsafe {
                std::mem::transmute::<*mut c_void, PacketBinaryFn>(symbol)
            }))
        } else if let Some(base) = range_base {
            let symbol = unsafe {
                libc::dlsym(
                    libc::RTLD_DEFAULT,
                    b"mgbfs_segmented_packet_range16_v1\0".as_ptr().cast(),
                )
            };
            if symbol.is_null()
                && std::env::var("MGBFS_SEGMENT_PACKET_RANGE16").as_deref() == Ok("1")
            {
                return Err("SEGMENT_PACKET_RANGE_ABI_UNAVAILABLE".into());
            }
            (!symbol.is_null()).then(|| {
                PacketCodec::Range16(base, unsafe {
                    std::mem::transmute::<*mut c_void, PacketRangeFn>(symbol)
                })
            })
        } else {
            None
        };
        let direct_mode = std::env::var("MGBFS_SEGMENT_PACKET_PACK4_DIRECT");
        let pack4_wire_owner = if pack4_codec.is_some() && direct_mode.as_deref() != Ok("0") {
            let symbol = unsafe {
                libc::dlsym(
                    libc::RTLD_DEFAULT,
                    b"mgbfs_pack4_owner_accept_wire_v1\0".as_ptr().cast(),
                )
            };
            if symbol.is_null() && direct_mode.as_deref() == Ok("1") {
                return Err("SEGMENT_PACKET_PACK4_WIRE_OWNER_ABI_UNAVAILABLE".into());
            }
            (!symbol.is_null())
                .then(|| unsafe { std::mem::transmute::<*mut c_void, Pack4PacketOwnerFn>(symbol) })
        } else {
            None
        };
        let byte_wire_owner =
            if let Some(PacketCodec::Binary(bits, _)) = codec.filter(|_| byte_wire_requested) {
                let symbol = unsafe {
                    libc::dlsym(
                        libc::RTLD_DEFAULT,
                        b"mgbfs_hash_accept_byte_packet_v1\0".as_ptr().cast(),
                    )
                };
                if symbol.is_null()
                    && std::env::var("MGBFS_SEGMENT_PACKET_BYTE_DIRECT").as_deref() == Ok("1")
                {
                    return Err("SEGMENT_PACKET_BYTE_WIRE_OWNER_ABI_UNAVAILABLE".into());
                }
                (!symbol.is_null()).then(|| {
                    (bits, unsafe {
                        std::mem::transmute::<*mut c_void, ByteWireOwnerFn>(symbol)
                    })
                })
            } else {
                None
            };
        Ok(Some(Self {
            endpoint,
            wire,
            bitmap_owner,
            bitmap_batch,
            pack4_owner,
            pack4_wire_owner,
            byte_wire_owner,
            compact_route,
            pack4_route,
            codec,
        }))
    }
}

#[derive(Default)]
struct PacketSourceScratch {
    pageable_metadata: Vec<u32>,
    spans: Vec<RouteSpan>,
    local_owner: Vec<PacketOwnerShard>,
    runs: Vec<ShardRun>,
}

pub(super) struct PacketEvent {
    ptr: *mut c_void,
    device: i32,
}

impl PacketEvent {
    pub(super) fn new(device: i32) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        const DISABLE_TIMING: u32 = 2;
        crate::generic_native::checked!(cudaEventCreateWithFlags, unsafe {
            cudaEventCreateWithFlags(&mut ptr, DISABLE_TIMING)
        })?;
        Ok(Self { ptr, device })
    }

    pub(super) fn record(&self, stream: &Stream) -> Result<()> {
        crate::generic_native::checked!(cudaEventRecord, unsafe {
            cudaEventRecord(self.ptr, stream.ptr)
        })
    }

    fn is_ready(&self) -> Result<bool> {
        match unsafe { mgbfs_cuda::native_owner::cudaEventQuery(self.ptr) } {
            0 => Ok(true),
            600 => Ok(false),
            status => Err(format!("SEGMENT_PACKET_EVENT_STATUS_{status}")),
        }
    }

    pub(super) fn wait_on(&self, stream: &Stream) -> Result<()> {
        crate::generic_native::checked!(cudaStreamWaitEvent, unsafe {
            cudaStreamWaitEvent(stream.ptr, self.ptr, 0)
        })
    }

    pub(super) fn synchronize(&self) -> Result<()> {
        crate::generic_native::checked!(cudaEventSynchronize, unsafe {
            mgbfs_cuda::native_owner::cudaEventSynchronize(self.ptr)
        })
    }
}

impl Drop for PacketEvent {
    fn drop(&mut self) {
        unsafe {
            mgbfs_cuda::native_owner::cudaSetDevice(self.device);
            cudaEventDestroy(self.ptr);
        }
    }
}

struct TransmitSlot {
    payload: Buffer,
    header: Option<PinnedStatus>,
    packed: Option<PacketEvent>,
}

#[derive(Clone, Copy)]
struct StagedSend {
    packet: Packet,
    slot: usize,
}

#[derive(Clone, Copy)]
enum TransmitProgress {
    Ready,
    Offered(Instant),
    Submitted(Instant),
}

enum TransmitPoll {
    Progressed,
    Waiting,
    PollHub(Option<StagedSend>),
}

struct TransmitCarousel {
    // Drop drains this stream before the slots' pinned headers or payloads
    // can be recycled. Each route slot also has a last-reader event.
    pack: Stream,
    source_read: [PacketEvent; 2],
    queue: VecDeque<StagedSend>,
    progress: TransmitProgress,
    next_slot: usize,
    peak_live: usize,
}

pub(super) struct PacketRuntime {
    config: PacketConfig,
    identity: [u8; 128],
    epoch: u64,
    pub(super) client: Option<PacketClient>,
    // Stream::drop drains outstanding copies before any packet buffer can
    // be returned to the session cache or freed.
    pub(super) stream: Stream,
    carousel: Option<TransmitCarousel>,
    tx: Vec<TransmitSlot>,
    receive: Option<ReceiveCarousel>,
    rx: Vec<ReceiveSlot>,
    wire_payload_bytes: usize,
    wire_record_bytes: usize,
    header: Vec<u8>,
    incoming_runs: Vec<ShardRun>,
    incoming_owner: Vec<PacketOwnerShard>,
    sent_packets: u64,
    sent_rows: u64,
    received_packets: u64,
    sent_bytes: u64,
    sent_header_bytes: u64,
    split_destination_windows: u64,
    source_scratch: PacketSourceScratch,
    route_metadata: Option<PinnedStatus>,
    route_receive_progress: bool,
    pub(super) producer_rows: u32,
    pub(super) route_lookahead: bool,
}

impl PacketRuntime {
    pub(super) fn new(
        config: PacketConfig,
        id: &[u8; 128],
        plan: &Plan,
        device: i32,
        generators: u32,
    ) -> Result<Self> {
        if generators == 0 || generators >= 0x7fff_ffff {
            return Err("SEGMENT_PACKET_GENERATOR_RANGE".into());
        }
        // Admission includes the bounded producer tile independently of TX/RX.
        let factor = crate::generic_distributed_memory::compact_producer_factor(plan.world);
        let row_budget = plan.queue_capacity.saturating_mul(factor).min(0x7fff_fffe);
        let producer_batch = (row_budget / generators).max(1).min(plan.batch);
        let producer_rows = producer_batch
            .checked_mul(generators)
            .ok_or("SEGMENT_PACKET_PRODUCER_SIZE")?;
        let metadata_words = (plan.world as usize)
            .checked_mul(plan.shards as usize)
            .and_then(|words| words.checked_add(PACKET_ROUTE_STATUS_WORDS))
            .ok_or("SEGMENT_PACKET_METADATA_SIZE")?;
        let metadata_bytes = metadata_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or("SEGMENT_PACKET_METADATA_SIZE")?;
        let route_metadata = PinnedStatus::try_new(metadata_bytes)?;
        let route_receive_progress = plan.compact_packet_slots == 3
            || (route_metadata.is_some()
                && (plan.compact_route_lookahead
                    || std::env::var("MGBFS_SEGMENT_PACKET_ROUTE_PROGRESS").as_deref() == Ok("1")));
        let mut transport = Self::new_exchange(
            config,
            id,
            plan.shards,
            plan.queue_capacity,
            plan.queue_payload_bytes(),
            [plan.compact_packet_slots, plan.compact_receive_slots],
            device,
        )?;
        transport.route_metadata = route_metadata;
        transport.producer_rows = producer_rows;
        transport.route_receive_progress = route_receive_progress;
        transport.route_lookahead = plan.compact_route_lookahead;
        Ok(transport)
    }

    fn new_exchange(
        config: PacketConfig,
        id: &[u8; 128],
        shards: u32,
        queue_capacity: u32,
        state_bytes: usize,
        slots: [u32; 2],
        device: i32,
    ) -> Result<Self> {
        let [tx_slots, rx_slots] = slots;
        let descriptors = shards.min(queue_capacity);
        let wire_payload_bytes = match config.codec {
            Some(PacketCodec::Binary(..)) => 8,
            Some(PacketCodec::Range16(..)) => state_bytes / 4,
            Some(PacketCodec::Pack4(_)) => 0,
            None => state_bytes,
        };
        let wire_record_bytes = match config.codec {
            Some(PacketCodec::Pack4(_)) => 16,
            Some(_) => 8,
            None => std::mem::size_of::<GenericRouteRecord>(),
        };
        let decoded_bytes = if config.codec.is_some()
            && config.pack4_wire_owner.is_none()
            && config.byte_wire_owner.is_none()
        {
            (queue_capacity as usize)
                .checked_mul(32 + state_bytes)
                .ok_or("SEGMENT_PACKET_BUFFER_SIZE")?
        } else {
            0
        };
        let packet = PacketBytes::new(
            descriptors,
            queue_capacity,
            wire_record_bytes,
            wire_payload_bytes,
        )?;
        let pipelined = tx_slots == 3;
        let carousel = if pipelined {
            Some(TransmitCarousel {
                pack: Stream::new(device)?,
                source_read: [PacketEvent::new(device)?, PacketEvent::new(device)?],
                queue: VecDeque::with_capacity(tx_slots as usize),
                progress: TransmitProgress::Ready,
                next_slot: 0,
                peak_live: 0,
            })
        } else {
            None
        };
        let mut tx = Vec::with_capacity(tx_slots as usize);
        for _ in 0..tx_slots {
            tx.push(TransmitSlot {
                payload: Buffer::new(packet.total, device)?,
                header: PinnedStatus::try_new(packet.header)?,
                packed: if pipelined {
                    Some(PacketEvent::new(device)?)
                } else {
                    None
                },
            });
        }
        let receive = if rx_slots == 3 {
            Some(ReceiveCarousel::new(shards as usize, device)?)
        } else {
            None
        };
        let rx = (0..rx_slots)
            .map(|_| {
                ReceiveSlot::new(
                    packet.total,
                    packet.header,
                    decoded_bytes,
                    receive.is_some(),
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            identity: *id,
            epoch: 0,
            client: None,
            carousel,
            tx,
            receive,
            rx,
            wire_payload_bytes,
            wire_record_bytes,
            stream: Stream::new(device)?,
            header: Vec::with_capacity(packet.header),
            incoming_runs: Vec::with_capacity(descriptors as usize),
            incoming_owner: Vec::with_capacity(descriptors as usize),
            source_scratch: Default::default(),
            sent_packets: 0,
            sent_rows: 0,
            received_packets: 0,
            sent_bytes: 0,
            sent_header_bytes: 0,
            split_destination_windows: 0,
            route_metadata: None,
            producer_rows: 0,
            route_receive_progress: false,
            route_lookahead: false,
        })
    }
    fn stage_payload(
        &mut self,
        slot: usize,
        runs: &[ShardRun],
        total_rows: u32,
        elements: u32,
        width: usize,
        input: QueueSide,
    ) -> Result<()> {
        let pack_stream = self
            .carousel
            .as_ref()
            .map_or(&self.stream, |ring| &ring.pack);
        let st = pack_stream.ptr;
        let transmit = &self.tx[slot];
        let layout = PacketBytes::new(
            runs.len() as u32,
            total_rows,
            self.wire_record_bytes,
            self.wire_payload_bytes,
        )?;
        if layout.total > transmit.payload.bytes {
            return Err("SEGMENT_PACKET_BUFFER_BOUNDS".into());
        }
        encode_runs(runs, &mut self.header);
        let header = layout.header;
        if let Some(pinned) = &transmit.header {
            // The host may format another packet while this header is being uploaded.
            unsafe {
                ptr::copy_nonoverlapping(self.header.as_ptr(), pinned.ptr.cast(), header);
            }
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(transmit.payload.ptr, pinned.ptr.cast(), header, 1, st)
            })?;
        } else {
            crate::generic_native::checked!(cudaMemcpy, unsafe {
                cudaMemcpy(transmit.payload.ptr, self.header.as_ptr().cast(), header, 1)
            })?;
        }
        let rows = total_rows as usize;
        if let Some(codec) = self.config.codec {
            crate::generic_native::checked!(segmented_packet_encode, unsafe {
                codec.transform(
                    transmit.payload.ptr,
                    runs.len() as u32,
                    total_rows,
                    elements,
                    input.states,
                    input.records,
                    transmit.payload.at(header),
                    0,
                    st,
                )
            })?;
        } else {
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(
                    transmit.payload.at(header),
                    input.records,
                    rows * self.wire_record_bytes,
                    3,
                    st,
                )
            })?;
            if width != 0 {
                crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                    cudaMemcpyAsync(
                        transmit.payload.at(layout.states),
                        input.states,
                        rows * width,
                        3,
                        st,
                    )
                })?;
            }
        }
        if let Some(packed) = &transmit.packed {
            packed.record(pack_stream)?;
        }
        Ok(())
    }

    fn tx_bytes(&self) -> usize {
        self.tx.iter().map(|slot| slot.payload.bytes).sum()
    }
    fn drain_after_abort(&self) -> Result<()> {
        let transfer = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(self.stream.ptr)
        });
        let packing = if let Some(ring) = &self.carousel {
            crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                cudaStreamSynchronize(ring.pack.ptr)
            })
        } else {
            Ok(())
        };
        let unpacking = self.receive.as_ref().map_or(Ok(()), ReceiveCarousel::drain);
        transfer.and(packing).and(unpacking)
    }
    fn rx_bytes(&self) -> usize {
        self.rx.iter().map(|slot| slot.payload.bytes).sum()
    }
    fn decoded_bytes(&self) -> usize {
        self.rx
            .iter()
            .filter_map(|slot| slot.decoded.as_ref())
            .map(|buffer| buffer.bytes)
            .sum()
    }
    pub(super) fn window_bytes(&self) -> usize {
        self.tx_bytes() + self.rx_bytes() + self.decoded_bytes()
    }
    pub(super) fn transport_totals(&self) -> (u64, u64, u64, u64) {
        (
            self.sent_bytes - self.sent_header_bytes,
            self.sent_bytes,
            self.sent_header_bytes,
            self.sent_packets,
        )
    }
    pub(super) fn submitted_packets(&self) -> u64 {
        self.sent_packets
    }

    pub(super) fn diagnostics(&self, producer_queue_bytes: usize) -> serde_json::Value {
        serde_json::json!({"protocol":if self.receive.is_some(){"nccl_compact_credit_v2"}else{"nccl_compact_credit_v1"},"packet_scope":"peer",
   "submitted_packets":self.sent_packets,"submitted_bytes":self.sent_bytes,
   "submitted_payload_bytes":self.sent_bytes-self.sent_header_bytes,"submitted_descriptor_bytes":self.sent_header_bytes,
   "tx_allocated_bytes":self.tx_bytes(),"rx_allocated_bytes":self.rx_bytes(),
   "tx_exchange_slots":self.tx.len(),
   "rx_exchange_slots":self.rx.len(),
   "rx_peak_live_slots":self.receive.as_ref().map_or(usize::from(self.received_packets != 0),|ring|ring.peak_live),
   "rx_unpack_stream_separate":self.receive.is_some(),
   "owner_ab_peak_shards":self.receive.as_ref().map_or(0,|ring|ring.peak_double),
   "owner_ab_fill_while_consuming":self.receive.as_ref().map_or(0,|ring|ring.overlap_fills),
   "owner_ab_wire_complete_while_consuming":self.receive.as_ref().map_or(0,|ring|ring.overlap_wire),
   "tx_peak_live_slots":self.carousel.as_ref().map_or(usize::from(self.sent_packets!=0),|ring| ring.peak_live),
   "tx_pack_stream_separate":self.carousel.is_some(),
   "decoded_rows_allocated_bytes":self.decoded_bytes(),
   "pack4_direct_owner":self.config.pack4_wire_owner.is_some(),
   "byte_direct_owner":self.config.byte_wire_owner.is_some(),
   "wire_record_bytes_per_row":self.wire_record_bytes,
   "wire_state_bytes_per_row":self.wire_payload_bytes,
   "rx_count_allocated_bytes":0,"max_concurrent_gpu_transfers_per_rank":1,
   "hub_peak_active_receive_credits":self.client.as_ref().and_then(PacketClient::hub_peak_active_credits),
   "producer_layout":"compact_tile","producer_rows":self.producer_rows,
   "producer_slots":if self.route_lookahead{2}else{1},
   "split_destination_windows":self.split_destination_windows,
   "pinned_route_metadata":self.route_metadata.is_some(),
   "route_receive_progress":self.route_receive_progress,
   "pinned_tx_header":self.tx.iter().all(|slot| slot.header.is_some()),
   "producer_queue_allocated_bytes":producer_queue_bytes,
   "chunk_collectives":0})
    }
    fn enqueue_transmit(&mut self, staged: StagedSend) -> Result<()> {
        let slots = self.tx.len();
        let ring = self
            .carousel
            .as_mut()
            .ok_or("SEGMENT_PACKET_CAROUSEL_MISSING")?;
        if ring.queue.len() >= slots || staged.slot != ring.next_slot {
            return Err("SEGMENT_PACKET_TX_SLOT".into());
        }
        ring.queue.push_back(staged);
        ring.next_slot = (staged.slot + 1) % slots;
        ring.peak_live = ring.peak_live.max(ring.queue.len());
        Ok(())
    }

    fn source_read_event(&self, slot: usize) -> Result<(&PacketEvent, &Stream)> {
        let ring = self
            .carousel
            .as_ref()
            .ok_or("SEGMENT_PACKET_CAROUSEL_MISSING")?;
        let event = ring
            .source_read
            .get(slot)
            .ok_or("SEGMENT_PACKET_LOCAL_SLOT")?;
        Ok((event, &ring.pack))
    }

    fn poll_transmit(&mut self) -> PacketResult<TransmitPoll> {
        let head = self
            .carousel
            .as_ref()
            .and_then(|tx| tx.queue.front().copied().map(|entry| (entry, tx.progress)));
        match head {
            Some((entry, TransmitProgress::Ready)) => {
                self.client()?.send(PacketFrame::Offer(entry.packet))?;
                self.set_transmit_progress(TransmitProgress::Offered(
                    Instant::now() + PACKET_TIMEOUT,
                ));
                Ok(TransmitPoll::Progressed)
            }
            Some((entry, TransmitProgress::Offered(deadline))) => {
                if Instant::now() >= deadline {
                    return Err("SEGMENT_PACKET_WAIT_TIMEOUT".into());
                }
                Ok(TransmitPoll::PollHub(Some(entry)))
            }
            Some((entry, TransmitProgress::Submitted(deadline))) => {
                match unsafe { cudaStreamQuery(self.stream.ptr) } {
                    0 => {
                        self.client()?.send(PacketFrame::Ack(entry.packet))?;
                        if let Some(tx) = &mut self.carousel {
                            tx.queue.pop_front();
                            tx.progress = TransmitProgress::Ready;
                        }
                        Ok(TransmitPoll::Progressed)
                    }
                    600 if Instant::now() < deadline => Ok(TransmitPoll::Waiting),
                    600 => Err("SEGMENT_PACKET_DEVICE_TIMEOUT".into()),
                    status => Err(format!("SEGMENT_PACKET_DEVICE_STATUS_{status}").into()),
                }
            }
            None => Ok(TransmitPoll::PollHub(None)),
        }
    }

    fn set_transmit_progress(&mut self, progress: TransmitProgress) {
        if let Some(tx) = &mut self.carousel {
            tx.progress = progress;
        }
    }

    fn client(&mut self) -> Result<&mut PacketClient> {
        self.client
            .as_mut()
            .ok_or_else(|| "SEGMENT_PACKET_CLIENT".into())
    }
}

#[derive(Default)]
pub(super) struct PacketLayerStats {
    pub(super) parents: u64,
    pub(super) waves: u64,
    pub(super) max_bucket: u32,
    pub(super) max_received_bucket: u32,
    pub(super) sent: Vec<u64>,
    pub(super) received: Vec<u64>,
}

fn packet_prefix_stop(code: u32) -> bool {
    matches!(
        code,
        10 | PACKET_CANCELLED | PACKET_DEADLINE | PACKET_LINEAGE_LIMIT | PACKET_LINEAGE_DISK_FULL
    )
}
#[derive(Debug)]
pub(crate) enum PacketError {
    Failure(String),
    Terminal(u32),
}
impl From<String> for PacketError {
    fn from(value: String) -> Self {
        Self::Failure(value)
    }
}
impl From<&str> for PacketError {
    fn from(value: &str) -> Self {
        Self::Failure(value.into())
    }
}
pub(crate) type PacketResult<T> = std::result::Result<T, PacketError>;

fn check_packet_stop(stop: &impl Fn() -> u32) -> PacketResult<()> {
    match stop() {
        1 => Err(PacketError::Terminal(PACKET_CANCELLED)),
        2 => Err(PacketError::Terminal(PACKET_DEADLINE)),
        _ => Ok(()),
    }
}

fn wait_packet_stream(
    client: &PacketClient,
    stream: *mut c_void,
    stop: &impl Fn() -> u32,
) -> PacketResult<()> {
    let started = Instant::now();
    loop {
        let terminal = client.terminal_code();
        if terminal != 0 {
            return Err(PacketError::Terminal(terminal));
        }
        match unsafe { cudaStreamQuery(stream) } {
            0 => return Ok(()),
            600 => {
                check_packet_stop(stop)?;
                let elapsed = started.elapsed();
                if elapsed >= PACKET_TIMEOUT {
                    return Err("SEGMENT_PACKET_DEVICE_TIMEOUT".into());
                }
                // Fast completions stay on the polling path. Under a delayed peer,
                // release the CPU so the control and NCCL progress threads can run.
                if elapsed < Duration::from_micros(100) {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(Duration::from_micros(50));
                }
            }
            status => return Err(format!("SEGMENT_PACKET_DEVICE_STATUS_{status}").into()),
        }
    }
}

extern "C" fn packet_cancel(context: *mut c_void) -> i32 {
    // The rank holds PacketClient, and thus the Arc, until NCCL is detached.
    let terminal = unsafe { &*context.cast::<std::sync::atomic::AtomicU32>() };
    i32::from(terminal.load(std::sync::atomic::Ordering::Acquire) != 0)
}

impl SegmentedDistributedBfs {
    pub(crate) fn signal_packet_failure(&mut self) {
        if let Some(client) = self
            .packet
            .as_mut()
            .and_then(|transport| transport.client.as_mut())
        {
            if client.terminal_code() == 0 {
                let _ = client.send(PacketFrame::Fail(REMOTE_FAILURE));
            }
        }
    }

    pub(super) fn wait_boundary_stream(&mut self) -> Result<()> {
        let stream = self.base.stream.ptr;
        if let Some(client) = self
            .packet
            .as_mut()
            .and_then(|transport| transport.client.as_mut())
        {
            let status = wait_packet_stream(client, stream, &|| 0);
            if status.is_err() {
                if client.terminal_code() == 0 {
                    let _ = client.send(PacketFrame::Fail(REMOTE_FAILURE));
                }
                // The queued D2H copy can still reference a caller's host output.
                // Abort and drain the stream before returning control to that caller.
                self.comm.abort_once();
                unsafe {
                    cudaStreamSynchronize(stream);
                }
            }
            return match status {
                Ok(()) => Ok(()),
                Err(PacketError::Failure(error)) => Err(error),
                Err(PacketError::Terminal(code)) => {
                    Err(format!("SEGMENT_PACKET_BOUNDARY_ABORT_{code}"))
                }
            };
        }
        crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(stream)
        })
    }
    fn start_packet_route(
        &self,
        transport: &PacketRuntime,
        slot: usize,
        range: Range,
        cursor: u64,
        batch: u32,
    ) -> Result<()> {
        let stream = self.route_stream.ptr;
        let p = &self.plan;
        let output = &self.send[slot];
        if let Some(ring) = &transport.carousel {
            ring.source_read[slot].wait_on(&self.route_stream)?;
        }
        // The route kernel writes only its world*shards counts. Keeping the
        // status at the tail lets the host read the whole tile metadata together.
        let error = output
            .counts
            .at::<u32>(output.counts.bytes - PACKET_ROUTE_STATUS_BYTES);
        crate::generic_native::checked!(cudaMemsetAsync, unsafe {
            cudaMemsetAsync(error.cast(), 0, PACKET_ROUTE_STATUS_BYTES, stream)
        })?;
        let segment = self.packet_parent_segment(range);
        let stride = segment
            .desc
            .capacity
            .checked_mul(u64::from(segment.desc.banks))
            .ok_or("SEGMENT_PACKET_PARENT_OFFSET")?;
        let row = u64::from(range.bank)
            .checked_mul(segment.desc.capacity)
            .and_then(|v| v.checked_add(cursor))
            .ok_or("SEGMENT_PACKET_PARENT_OFFSET")?;
        let word_bytes = if self.pack4.is_some() {
            8
        } else {
            u64::from(p.state_bytes)
        };
        let offset = usize::try_from(
            row.checked_mul(word_bytes)
                .ok_or("SEGMENT_PACKET_PARENT_OFFSET")?,
        )
        .map_err(|_| "SEGMENT_PACKET_PARENT_OFFSET")?;
        let parent = unsafe { segment.desc.states.cast::<u8>().add(offset) };
        let count = u32::try_from(u64::from(batch).min(range.end - cursor))
            .map_err(|_| "SEGMENT_PACKET_PARENT_RANGE")?;
        let parent_base = if self.ownership.is_some() {
            range
                .logical_start
                .checked_add(cursor - range.begin)
                .ok_or("SEGMENTED_PATH_ROW_OVERFLOW")?
        } else {
            0
        };
        let scratch = self
            .route_scratch
            .as_ref()
            .ok_or("SEGMENT_PACKET_ROUTE_SCRATCH")?;
        if let Some(packed) = self.pack4 {
            let bitmap = self.bitmap.as_ref().ok_or("BITMAP_PACK4_REQUIRES_BITMAP")?;
            let route = transport
                .config
                .pack4_route
                .ok_or("SEGMENT_PACKET_PACK4_ABI_UNAVAILABLE")?;
            let args = Pack4RouteArgs {
                dictionary: packed.dictionary,
                domain: bitmap.domain,
                parents: parent.cast(),
                permutations: self.base.perms.ptr.cast(),
                count,
                generators: self.base.generators,
                world: p.world,
                source: self.rank,
                shards: p.shards,
                capacity: p.queue_capacity,
                state_bytes: p.state_bytes,
                grouped: 0,
                map: self.map.ptr.cast(),
                records: output.records.ptr.cast(),
                payload: output.states.ptr,
                counts: output.counts.ptr.cast(),
                error,
                owner_page: bitmap.owner_page,
                owner_boundary: bitmap.owner_boundary,
            };
            crate::generic_native::checked!(mgbfs_pack4_compact_route_v1, unsafe {
                route(
                    args,
                    scratch.ptr,
                    scratch.bytes,
                    transport.producer_rows,
                    stream,
                )
            })?;
        } else {
            let (bitmap, domain, page, boundary) = self.bitmap.as_ref().map_or(
                (
                    0,
                    BitmapDeviceDomain {
                        symbols: ptr::null(),
                        counts: ptr::null(),
                        distinct: 0,
                        elements: 0,
                        total: 0,
                    },
                    0,
                    0,
                ),
                |bitmap| (1, bitmap.domain, bitmap.owner_page, bitmap.owner_boundary),
            );
            let word_shard = u32::from(bitmap != 0 && self.strategy == SegmentStrategy::Batch);
            if let Some((call, masks)) = self.bitmap.as_ref().and_then(|bitmap| bitmap.xor.as_ref())
            {
                crate::generic_native::checked!(mgbfs_segmented_compact_route_xor_v1, unsafe {
                    call(
                        bitmap,
                        word_shard,
                        domain,
                        p.state_bytes,
                        self.base.kind,
                        p.elements,
                        self.base.rows,
                        self.base.cols,
                        self.base.generators,
                        parent.cast(),
                        count,
                        stride,
                        self.base.perms.ptr.cast(),
                        self.base.matrices.ptr.cast(),
                        self.base.moduli.ptr.cast(),
                        self.seed,
                        self.bits,
                        p.world,
                        self.rank,
                        p.shards,
                        p.queue_capacity,
                        parent_base,
                        self.map.ptr.cast(),
                        self.cuts.as_ref().map_or(ptr::null(), |v| v.ptr.cast()),
                        output.records.ptr.cast(),
                        output.counts.ptr.cast(),
                        error,
                        output.states.ptr,
                        scratch.ptr,
                        scratch.bytes,
                        stream,
                        page,
                        boundary,
                        transport.producer_rows,
                        masks.ptr.cast(),
                    )
                })?;
            } else {
                crate::generic_native::checked!(mgbfs_segmented_compact_route_v1, unsafe {
                    (transport
                        .config
                        .compact_route
                        .ok_or("SEGMENT_PACKET_COMPACT_ROUTE_ABI_UNAVAILABLE")?)(
                        bitmap,
                        word_shard,
                        domain,
                        p.state_bytes,
                        self.base.kind,
                        p.elements,
                        self.base.rows,
                        self.base.cols,
                        self.base.generators,
                        parent.cast(),
                        count,
                        stride,
                        self.base.perms.ptr.cast(),
                        self.base.matrices.ptr.cast(),
                        self.base.moduli.ptr.cast(),
                        self.seed,
                        self.bits,
                        p.world,
                        self.rank,
                        p.shards,
                        p.queue_capacity,
                        parent_base,
                        self.map.ptr.cast(),
                        self.cuts.as_ref().map_or(ptr::null(), |v| v.ptr.cast()),
                        output.records.ptr.cast(),
                        output.counts.ptr.cast(),
                        error,
                        output.states.ptr,
                        scratch.ptr,
                        scratch.bytes,
                        stream,
                        page,
                        boundary,
                        transport.producer_rows,
                    )
                })?;
            }
        }
        Ok(())
    }

    fn stage_packet(
        &self,
        transport: &mut PacketRuntime,
        output: &Queue,
        slot: usize,
        destination: u32,
        runs: &[ShardRun],
        total_rows: u32,
        tile_begin: usize,
    ) -> Result<()> {
        let p = &self.plan;
        let first = runs.first().ok_or("SEGMENT_PACKET_EMPTY_RUNS")?;
        if runs.len() as u32 > p.shards.min(p.queue_capacity) {
            return Err("SEGMENT_PACKET_BUFFER_BOUNDS".into());
        }
        // Runs are consecutive nonempty shards in the sorted producer tile. Their
        // records and their SoA state planes form two contiguous source spans.
        let queue = destination as usize * p.shards as usize + first.shard as usize;
        let input = output.slice(queue, tile_begin, p);
        transport.stage_payload(
            slot,
            runs,
            total_rows,
            p.elements,
            p.queue_payload_bytes(),
            input,
        )
    }

    fn count_packet_owner_input(&mut self, shards: &[PacketOwnerShard]) -> Result<()> {
        if self.input_stats_active() {
            let rows = shards
                .iter()
                .map(|input| u64::from(input.rows))
                .sum::<u64>();
            self.input_valid = self
                .input_valid
                .checked_add(rows)
                .ok_or("SEGMENTED_INPUT_STATS_OVERFLOW")?;
        }
        Ok(())
    }

    fn packet_owner(
        &mut self,
        shards: &[PacketOwnerShard],
        future: u32,
        config: PacketConfig,
    ) -> PacketResult<()> {
        self.count_packet_owner_input(shards)?;
        let status = self.accept_input(OwnerInputs::Compact { shards, config }, future, None)?;
        match status {
            0 => Ok(()),
            10 => Err(PacketError::Terminal(10)),
            other => Err(format!("SEGMENTED_OWNER_{other}").into()),
        }
    }

    fn transfer_packet(
        &mut self,
        transport: &mut PacketRuntime,
        packet: Packet,
        outgoing: Option<StagedSend>,
        source_owner: &[PacketOwnerShard],
        future: u32,
        stats: &mut PacketLayerStats,
        stop: &impl Fn() -> u32,
    ) -> PacketResult<()> {
        let ctx = self.packet_wire_context(transport.epoch);
        let source = packet.source == ctx.rank;
        transport.submit_wire(ctx, packet, outgoing, 0)?;
        if source && transport.carousel.is_some() {
            // Keep the slot until progress observes NCCL completion and sends ACK.
            return Ok(());
        }
        if !source {
            transport.queue_unpack(ctx, packet, 0, &transport.stream)?;
        }
        // Local rows and the outgoing packet occupy independent buffers.
        if source && !source_owner.is_empty() {
            self.packet_owner(source_owner, future, transport.config)?;
        }
        let stream = transport.stream.ptr;
        wait_packet_stream(transport.client()?, stream, stop)?;
        if !source {
            transport.read_incoming(ctx, packet, 0, stats)?;
            self.packet_owner(
                &transport.incoming_owner,
                future,
                transport.remote_owner_config(),
            )?;
        }
        transport.client()?.send(PacketFrame::Ack(packet))?;
        Ok(())
    }

    pub(super) fn connect_packet_control(&mut self) -> Result<()> {
        if let Some(transport) = self
            .packet
            .as_mut()
            .filter(|transport| transport.client.is_none())
        {
            let p = &self.plan;
            let client = transport.config.endpoint.connect(
                self.rank,
                p.world,
                p.shards,
                p.queue_capacity,
                &transport.identity,
                p.compact_receive_slots,
            )?;
            crate::generic_native::checked!(mgbfs_nccl_bind_cancel, unsafe {
                mgbfs_nccl_bind_cancel(
                    self.comm.ptr,
                    Some(packet_cancel),
                    client.terminal_ptr().cast_mut().cast(),
                )
            })?;
            transport.client = Some(client);
        }
        Ok(())
    }

    fn service_source_grant(
        &mut self,
        transport: &mut PacketRuntime,
        future: u32,
        stats: &mut PacketLayerStats,
        stop: &impl Fn() -> u32,
    ) -> PacketResult<bool> {
        if transport.receive.is_some() {
            return self.service_receive_carousel(transport, future, stats, stop, false);
        }
        let terminal = transport.client()?.terminal_code();
        if terminal != 0 {
            return Err(PacketError::Terminal(terminal));
        }
        let offered = match transport.poll_transmit()? {
            TransmitPoll::Progressed => return Ok(true),
            TransmitPoll::Waiting => return Ok(false),
            TransmitPoll::PollHub(offered) => offered,
        };
        match transport.client()?.next(None) {
            Some(PacketFrame::Grant(packet)) => {
                self.transfer_packet(transport, packet, offered, &[], future, stats, stop)?;
                if packet.source == self.rank {
                    transport.set_transmit_progress(TransmitProgress::Submitted(
                        Instant::now() + PACKET_TIMEOUT,
                    ));
                }
                Ok(true)
            }
            Some(PacketFrame::Abort(code)) => Err(PacketError::Terminal(code)),
            Some(_) => Err("SEGMENT_PACKET_UNEXPECTED_SOURCE_FRAME".into()),
            None => Ok(false),
        }
    }

    fn wait_for_transmit_room(
        &mut self,
        transport: &mut PacketRuntime,
        limit: usize,
        future: u32,
        stats: &mut PacketLayerStats,
        stop: &impl Fn() -> u32,
    ) -> PacketResult<()> {
        while transport
            .carousel
            .as_ref()
            .is_some_and(|ring| ring.queue.len() >= limit)
        {
            check_packet_stop(stop)?;
            if !self.service_source_grant(transport, future, stats, stop)? {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        Ok(())
    }

    fn wait_for_source_route(
        &mut self,
        transport: &mut PacketRuntime,
        future: u32,
        stats: &mut PacketLayerStats,
        stop: &impl Fn() -> u32,
        receive_progress: bool,
    ) -> PacketResult<()> {
        let stream = self.route_stream.ptr;
        if !receive_progress {
            crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                cudaStreamSynchronize(stream)
            })?;
            return Ok(());
        }
        let started = std::time::Instant::now();
        loop {
            check_packet_stop(stop)?;
            match unsafe { cudaStreamQuery(stream) } {
                0 => break,
                600 => {}
                status => return Err(format!("SEGMENT_PACKET_ROUTE_STATUS_{status}").into()),
            }
            let elapsed = started.elapsed();
            if elapsed >= PACKET_TIMEOUT {
                return Err("SEGMENT_PACKET_ROUTE_TIMEOUT".into());
            }
            if !self.service_source_grant(transport, future, stats, stop)? {
                if elapsed < std::time::Duration::from_micros(100) {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
        }
        // Even short kernels give an incoming grant a chance to progress.
        self.service_source_grant(transport, future, stats, stop)?;
        Ok(())
    }

    fn advance_packets_inner(
        &mut self,
        transport: &mut PacketRuntime,
        work: &[Range],
        future: u32,
        batch: &mut u32,
        stats_on: bool,
        scratch: &mut PacketSourceScratch,
        stop: &impl Fn() -> u32,
    ) -> PacketResult<PacketLayerStats> {
        let p = self.plan.clone();
        let queues = p.world * p.shards;
        let queue_count = queues as usize;
        let metadata_words = queue_count + PACKET_ROUTE_STATUS_WORDS;
        let overlap_local_owner =
            std::env::var("MGBFS_SEGMENT_PACKET_OVERLAP_LOCAL").as_deref() != Ok("0");
        // Receive progress can overlap a route on the existing slot. Incoming
        // transfers and owner work do not touch the producer's pinned metadata.
        let progress_during_route = transport.route_receive_progress;
        let PacketSourceScratch {
            pageable_metadata,
            spans,
            local_owner,
            runs,
        } = scratch;
        let metadata_ptr = if let Some(pinned) = &transport.route_metadata {
            pinned.ptr
        } else {
            pageable_metadata.resize(metadata_words, 0u32);
            pageable_metadata.as_mut_ptr()
        };
        local_owner.clear();
        let mut stats = PacketLayerStats::default();
        if stats_on {
            stats.sent = vec![0; queue_count];
            stats.received = vec![0; queue_count];
        }
        let mut sequence = 0u64;
        for range in work {
            let mut cursor = range.begin;
            let mut prefetched = None;
            while cursor < range.end {
                // Only a source that has not sealed can stop a layer. Once the hub
                // commits, every rank must complete its boundary collective.
                check_packet_stop(stop)?;
                if !progress_during_route {
                    while self.service_source_grant(transport, future, &mut stats, stop)? {}
                }
                let slot = match prefetched.take() {
                    Some(slot) => slot,
                    None => {
                        self.start_packet_route(transport, 0, *range, cursor, *batch)?;
                        0
                    }
                };
                let st = self.route_stream.ptr;
                crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                    cudaMemcpyAsync(
                        metadata_ptr.cast(),
                        self.send[slot].counts.ptr,
                        metadata_words * std::mem::size_of::<u32>(),
                        2,
                        st,
                    )
                })?;
                self.wait_for_source_route(
                    transport,
                    future,
                    &mut stats,
                    stop,
                    progress_during_route,
                )?;
                // Reading is safe only after the route stream has completed the copy.
                let metadata = unsafe { std::slice::from_raw_parts(metadata_ptr, metadata_words) };
                let (counts, errors) = metadata.split_at(queue_count);
                match crate::segmented_history::classify_route_and_regeneration(
                    errors[0], errors[1],
                ) {
                    0 => {}
                    1 if *batch > 1 => {
                        *batch = (*batch / 2).max(1);
                        self.queue_batch_ceiling = self.queue_batch_ceiling.min(*batch);
                        self.source_retries += 1;
                        continue; // no owner publication or offer has happened for this tile
                    }
                    status => return Err(format!("SEGMENTED_ROUTE_{status}").into()),
                }
                let upper = (u64::from(*batch) * u64::from(self.base.generators))
                    .min(u64::from(p.queue_capacity)) as u32;
                let parents = (range.end - cursor).min(u64::from(*batch));
                spans.clear();
                let (produced, max_bucket) = read_route_counts(counts, upper, spans)?;
                if produced > transport.producer_rows
                    || u64::from(produced) != parents * u64::from(self.base.generators)
                {
                    return Err("SEGMENT_PACKET_PRODUCER_COUNT".into());
                }
                // After count validation the next tile can read the immutable source bank
                // while NCCL staging and the owner consume this tile. Both consumers
                // drain before cycling back to the slot. Route scratch stays on one stream.
                let next_cursor = cursor + parents;
                if transport.route_lookahead && next_cursor < range.end {
                    self.start_packet_route(transport, 1 - slot, *range, next_cursor, *batch)?;
                    prefetched = Some(1 - slot);
                    self.prefetched_chunks += 1;
                }
                stats.max_bucket = stats.max_bucket.max(max_bucket);
                stats.parents = stats
                    .parents
                    .checked_add(parents)
                    .ok_or("SEGMENT_PACKET_COUNTER_OVERFLOW")?;
                stats.waves = stats
                    .waves
                    .checked_add(1)
                    .ok_or("SEGMENT_PACKET_COUNTER_OVERFLOW")?;
                local_owner.clear();
                if stats_on {
                    for span in spans.iter() {
                        stats.sent[span.queue as usize] += u64::from(span.rows);
                    }
                }
                let rank = self.rank;
                let local_start = spans.partition_point(|span| span.queue / p.shards < rank);
                let local_end = spans.partition_point(|span| span.queue / p.shards <= rank);
                for span in &spans[local_start..local_end] {
                    let bucket = span.queue as usize;
                    if stats_on {
                        stats.received[bucket] += u64::from(span.rows);
                    }
                    stats.max_received_bucket = stats.max_received_bucket.max(span.rows);
                    let queue = self.send[slot].slice(bucket, span.tile_begin as usize, &p);
                    local_owner.push(PacketOwnerShard {
                        shard: (span.queue % p.shards) as usize,
                        queue,
                        rows: span.rows,
                        byte_wire: false,
                    });
                }
                let mut packed_tile = false;
                let mut peers = spans.as_slice();
                while let Some(first) = peers.first() {
                    let destination = first.queue / p.shards;
                    let end = peers.partition_point(|span| span.queue / p.shards == destination);
                    let (mut peer, remaining) = peers.split_at(end);
                    peers = remaining;
                    if destination == rank {
                        continue;
                    }
                    let last = peer.last().ok_or("SEGMENT_PACKET_EMPTY_PEER")?;
                    let destination_rows = last.tile_begin + last.rows - first.tile_begin;
                    if destination_rows > p.queue_capacity {
                        transport.split_destination_windows = transport
                            .split_destination_windows
                            .checked_add(1)
                            .ok_or("SEGMENT_PACKET_COUNTER_OVERFLOW")?;
                    }
                    // Each credit bounds one send window. Keep a successfully routed
                    // tile when a peer needs multiple windows; replay is only needed if
                    // an individual shard could not fit during routing.
                    while let Some(first) = peer.first() {
                        let tile_begin = first.tile_begin;
                        let (used, rows) = packet_window(peer, p.shards, p.queue_capacity, runs);
                        peer = &peer[used..];
                        let outgoing = Packet {
                            depth: transport.epoch,
                            sequence,
                            source: self.rank,
                            destination,
                            descriptors: runs.len() as u32,
                            rows,
                        };
                        sequence = sequence
                            .checked_add(1)
                            .ok_or("SEGMENT_PACKET_SEQUENCE_OVERFLOW")?;
                        let slots = transport.tx.len();
                        self.wait_for_transmit_room(transport, slots, future, &mut stats, stop)?;
                        let transmit_slot =
                            transport.carousel.as_ref().map_or(0, |ring| ring.next_slot);
                        self.stage_packet(
                            transport,
                            &self.send[slot],
                            transmit_slot,
                            destination,
                            runs,
                            rows,
                            tile_begin as usize,
                        )?;
                        let staged = StagedSend {
                            packet: outgoing,
                            slot: transmit_slot,
                        };
                        if transport.carousel.is_some() {
                            transport.enqueue_transmit(staged)?;
                            packed_tile = true;
                            self.service_source_grant(transport, future, &mut stats, stop)?;
                            continue;
                        }
                        if !overlap_local_owner && !local_owner.is_empty() {
                            self.packet_owner(local_owner, future, transport.config)?;
                            local_owner.clear();
                        }
                        transport.client()?.send(PacketFrame::Offer(outgoing))?;
                        let grant_deadline = Instant::now() + PACKET_TIMEOUT;
                        loop {
                            check_packet_stop(stop)?;
                            let remaining =
                                grant_deadline.saturating_duration_since(std::time::Instant::now());
                            if remaining.is_zero() {
                                return Err("SEGMENT_PACKET_WAIT_TIMEOUT".into());
                            }
                            let Some(frame) = transport
                                .client()?
                                .next(Some(remaining.min(std::time::Duration::from_millis(10))))
                            else {
                                continue;
                            };
                            match frame {
                                PacketFrame::Grant(packet) => {
                                    self.transfer_packet(
                                        transport,
                                        packet,
                                        Some(staged),
                                        local_owner,
                                        future,
                                        &mut stats,
                                        stop,
                                    )?;
                                    // Send completion is local. TCP orders its ACK before the next offer.
                                    if packet == outgoing {
                                        local_owner.clear();
                                        break;
                                    }
                                }
                                PacketFrame::Abort(code) => {
                                    return Err(PacketError::Terminal(code))
                                }
                                _ => return Err("SEGMENT_PACKET_UNEXPECTED_PENDING_FRAME".into()),
                            }
                        }
                    }
                }
                if packed_tile {
                    // Route reuse waits for all reads from this tile on the pack stream.
                    let (event, pack) = transport.source_read_event(slot)?;
                    event.record(pack)?;
                }
                if !local_owner.is_empty() {
                    if transport.receive.is_some() {
                        self.accept_local_leases(
                            transport,
                            slot,
                            local_owner,
                            future,
                            &mut stats,
                            stop,
                        )?;
                    } else {
                        self.packet_owner(local_owner, future, transport.config)?;
                    }
                }
                cursor = next_cursor;
            }
        }
        self.wait_for_transmit_room(transport, 1, future, &mut stats, stop)?;
        let epoch = transport.epoch;
        transport.client()?.send(PacketFrame::Seal(epoch))?;
        if transport.receive.is_some() {
            return self.finish_receive_carousel(transport, future, stats);
        }
        loop {
            match transport.client()?.next_after_seal() {
                PacketFrame::Grant(packet) => {
                    self.transfer_packet(transport, packet, None, &[], future, &mut stats, &|| 0)?
                }
                PacketFrame::Commit(depth) if depth == transport.epoch => {
                    self.note_sync(|counts| counts.packet_hub_layer_commits += 1);
                    return Ok(stats);
                }
                PacketFrame::Abort(code) => return Err(PacketError::Terminal(code)),
                _ => return Err("SEGMENT_PACKET_UNEXPECTED_SEALED_FRAME".into()),
            }
        }
    }

    pub(super) fn advance_packets(
        &mut self,
        work: &[Range],
        future: u32,
        batch: &mut u32,
        stats: bool,
        stop: &impl Fn() -> u32,
    ) -> Result<std::result::Result<PacketLayerStats, u32>> {
        let mut transport = self.packet.take().ok_or("SEGMENT_PACKET_CONFIG_MISSING")?;
        let mut scratch = std::mem::take(&mut transport.source_scratch);
        let execution = self.advance_packets_inner(
            &mut transport,
            work,
            future,
            batch,
            stats,
            &mut scratch,
            stop,
        );
        transport.source_scratch = scratch;
        self.remote_records_sent = transport.sent_rows;
        if execution.is_ok() {
            transport.epoch += 1;
        }
        let mut terminal = transport
            .client
            .as_ref()
            .map_or(0, PacketClient::terminal_code);
        if execution.is_err() {
            if let Some(client) = &mut transport.client {
                if terminal == 0 {
                    let code = if let Err(PacketError::Terminal(code)) = execution.as_ref() {
                        *code
                    } else {
                        REMOTE_FAILURE
                    };
                    let _ = client.send(PacketFrame::Fail(code));
                    if packet_prefix_stop(code) {
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(2);
                        while client.terminal_code() == 0 && std::time::Instant::now() < deadline {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                }
                terminal = client.terminal_code();
            }
            self.comm.abort_once();
            // Cancellation and grant errors can interrupt a route in either the
            // single-slot receive-progress path or the two-slot lookahead path.
            // Drain route work before publishing a prefix or releasing its buffers.
            let exchange = if transport.carousel.is_some() {
                transport.drain_after_abort()
            } else {
                Ok(())
            };
            let route = crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
                cudaStreamSynchronize(self.route_stream.ptr)
            });
            if let Err(error) = exchange.and(route) {
                self.packet = Some(transport);
                return Err(error);
            }
        }
        self.packet = Some(transport);
        match execution {
            Ok(stats) => Ok(Ok(stats)),
            Err(_) if packet_prefix_stop(terminal) => Ok(Err(terminal)),
            Err(PacketError::Terminal(code)) if packet_prefix_stop(code) => Err(format!(
                "SEGMENT_PACKET_PREFIX_STOP_UNCONFIRMED request={code} hub={terminal}"
            )),
            Err(PacketError::Terminal(code)) => {
                Err(format!("SEGMENT_PACKET_REMOTE_FAILURE_{code}"))
            }
            Err(PacketError::Failure(error)) => Err(error),
        }
    }
}
