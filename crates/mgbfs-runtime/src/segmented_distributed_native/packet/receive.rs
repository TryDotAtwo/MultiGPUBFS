//! Receive credits and per-shard input leases for compact packet exchange.

use super::owner_banks::{OwnerBanks, OwnerLease, OwnerSource};
use super::*;

#[derive(Clone, Copy)]
pub(super) struct WireContext {
    pub(super) rank: u32,
    pub(super) depth: u64,
    pub(super) world: u32,
    pub(super) shards: u32,
    pub(super) queue_capacity: u32,
    pub(super) elements: u32,
    pub(super) width: usize,
    pub(super) comm: *mut c_void,
}

pub(super) struct ReceiveSlot {
    pub(super) payload: Buffer,
    pub(super) decoded: Option<Buffer>,
    header: Option<PinnedStatus>,
    events: Option<(PacketEvent, PacketEvent)>,
    packet: Option<Packet>,
    deadline: Option<Instant>,
    to_unpack: bool,
    parked: VecDeque<PacketOwnerShard>,
    remaining: usize,
}

impl ReceiveSlot {
    pub(super) fn new(
        bytes: usize,
        header: usize,
        decoded: usize,
        ring: bool,
        device: i32,
    ) -> Result<Self> {
        Ok(Self {
            payload: Buffer::new(bytes, device)?,
            decoded: if decoded == 0 {
                None
            } else {
                Some(Buffer::new(decoded, device)?)
            },
            header: PinnedStatus::try_new(header)?,
            events: if ring {
                Some((PacketEvent::new(device)?, PacketEvent::new(device)?))
            } else {
                None
            },
            packet: None,
            deadline: None,
            to_unpack: false,
            parked: VecDeque::new(),
            remaining: 0,
        })
    }

    fn events(&self) -> Result<&(PacketEvent, PacketEvent)> {
        self.events
            .as_ref()
            .ok_or_else(|| "SEGMENT_PACKET_RX_EVENTS".into())
    }
}

pub(super) struct ReceiveCarousel {
    unpack: Stream,
    banks: OwnerBanks,
    wave: Vec<OwnerLease>,
    spans: Vec<PacketOwnerShard>,
    next_slot: usize,
    wire_slot: Option<usize>,
    live: usize,
    double_shards: usize,
    local_live: [usize; 2],
    commit: bool,
    pub(super) peak_live: usize,
    pub(super) peak_double: usize,
    pub(super) overlap_fills: u64,
    pub(super) overlap_wire: u64,
}

impl ReceiveCarousel {
    pub(super) fn new(shards: usize, device: i32) -> Result<Self> {
        Ok(Self {
            unpack: Stream::new(device)?,
            banks: OwnerBanks::new(shards),
            wave: Vec::new(),
            spans: Vec::new(),
            next_slot: 0,
            wire_slot: None,
            live: 0,
            double_shards: 0,
            local_live: [0; 2],
            commit: false,
            peak_live: 0,
            peak_double: 0,
            overlap_fills: 0,
            overlap_wire: 0,
        })
    }

    pub(super) fn drain(&self) -> Result<()> {
        crate::generic_native::checked!(cudaStreamSynchronize, unsafe {
            cudaStreamSynchronize(self.unpack.ptr)
        })
    }
}

impl PacketRuntime {
    pub(super) fn submit_wire(
        &mut self,
        ctx: WireContext,
        packet: Packet,
        outgoing: Option<StagedSend>,
        rx_index: usize,
    ) -> PacketResult<()> {
        let rank = ctx.rank;

        let source = packet.source == rank;
        if packet.depth != ctx.depth
            || packet.descriptors == 0
            || packet.descriptors > ctx.shards
            || packet.descriptors > packet.rows
            || packet.rows > ctx.queue_capacity
            || packet.source >= ctx.world
            || packet.destination >= ctx.world
            || packet.source == packet.destination
            || (source && !outgoing.is_some_and(|staged| staged.packet == packet))
            || (!source && packet.destination != rank)
        {
            return Err("SEGMENT_PACKET_UNEXPECTED_GRANT".into());
        }
        let layout = PacketBytes::new(
            packet.descriptors,
            packet.rows,
            self.wire_record_bytes,
            self.wire_payload_bytes,
        )?;
        let stream = self.stream.ptr;
        let peer = if source {
            packet.destination
        } else {
            packet.source
        };
        let buffer = if source {
            let slot = &self.tx[outgoing.ok_or("SEGMENT_PACKET_MISSING_SEND")?.slot];
            if let Some(packed) = &slot.packed {
                packed.wait_on(&self.stream)?;
            }
            &slot.payload
        } else {
            &self.rx[rx_index].payload
        };
        if layout.total > buffer.bytes {
            return Err("SEGMENT_PACKET_BUFFER_BOUNDS".into());
        }
        crate::native_fault::begin("mgbfs_nccl_segment_packet_v1", file!(), line!());
        let status = unsafe {
            (self.config.wire)(
                ctx.comm,
                buffer.ptr,
                layout.total as u64,
                peer,
                u32::from(source),
                stream,
            )
        };
        if source && status == 0 {
            // Count successful NCCL submission even if the stream is later cancelled.
            self.sent_packets = self
                .sent_packets
                .checked_add(1)
                .ok_or("SEGMENT_PACKET_COUNTER_OVERFLOW")?;
            self.sent_bytes = self
                .sent_bytes
                .checked_add(layout.total as u64)
                .ok_or("SEGMENT_PACKET_COUNTER_OVERFLOW")?;
            self.sent_header_bytes = self
                .sent_header_bytes
                .checked_add(layout.header as u64)
                .ok_or("SEGMENT_PACKET_COUNTER_OVERFLOW")?;
            self.sent_rows = self
                .sent_rows
                .checked_add(u64::from(packet.rows))
                .ok_or("SEGMENT_PACKET_COUNTER_OVERFLOW")?;
        }
        let terminal = self.client()?.terminal_code();
        if terminal != 0 {
            return Err(PacketError::Terminal(terminal));
        }
        crate::native_fault::finish(status)?;
        Ok(())
    }

    pub(super) fn queue_unpack(
        &self,
        ctx: WireContext,
        packet: Packet,
        index: usize,
        stream: &Stream,
    ) -> Result<()> {
        let slot = &self.rx[index];
        let header = PacketBytes::new(
            packet.descriptors,
            packet.rows,
            self.wire_record_bytes,
            self.wire_payload_bytes,
        )?
        .header;
        if let (Some(codec), Some(decoded)) = (self.config.codec, slot.decoded.as_ref()) {
            crate::generic_native::checked!(segmented_packet_decode, unsafe {
                codec.transform(
                    slot.payload.ptr,
                    packet.descriptors,
                    packet.rows,
                    ctx.elements,
                    decoded.at(packet.rows as usize * 32),
                    decoded.ptr,
                    slot.payload.at(header),
                    1,
                    stream.ptr,
                )
            })?;
        }
        if let Some(pinned) = &slot.header {
            crate::generic_native::checked!(cudaMemcpyAsync, unsafe {
                cudaMemcpyAsync(pinned.ptr.cast(), slot.payload.ptr, header, 2, stream.ptr)
            })?;
        }
        Ok(())
    }

    pub(super) fn read_incoming(
        &mut self,
        ctx: WireContext,
        packet: Packet,
        index: usize,
        stats: &mut PacketLayerStats,
    ) -> Result<()> {
        let slot = &self.rx[index];
        let layout = PacketBytes::new(
            packet.descriptors,
            packet.rows,
            self.wire_record_bytes,
            self.wire_payload_bytes,
        )?;
        let header = layout.header;
        let incoming_header = if let Some(pinned) = &slot.header {
            unsafe { std::slice::from_raw_parts(pinned.ptr.cast::<u8>(), header) }
        } else {
            self.header.resize(header, 0);
            crate::generic_native::checked!(cudaMemcpy, unsafe {
                cudaMemcpy(self.header.as_mut_ptr().cast(), slot.payload.ptr, header, 2)
            })?;
            &self.header
        };
        decode_runs(
            incoming_header,
            packet.rows,
            ctx.shards,
            &mut self.incoming_runs,
        )?;
        let rows = packet.rows as usize;
        let direct_pack4 = self.config.pack4_wire_owner.is_some();
        let byte_wire = self.config.byte_wire_owner.is_some();
        let mut offset = 0usize;
        self.incoming_owner.clear();
        for (index, run) in self.incoming_runs.iter().enumerate() {
            let queue_index = packet.source as usize * ctx.shards as usize + run.shard as usize;
            if !stats.received.is_empty() {
                stats.received[queue_index] += u64::from(run.rows);
            }
            stats.max_received_bucket = stats.max_received_bucket.max(run.rows);
            let queue = QueueSide {
                records: match &slot.decoded {
                    Some(decoded) => decoded.at(offset * 32),
                    None if byte_wire => slot
                        .payload
                        .at(header + offset * (self.wire_record_bytes + self.wire_payload_bytes)),
                    None => slot.payload.at(header + offset * self.wire_record_bytes),
                },
                states: match &slot.decoded {
                    Some(decoded) => decoded.at(rows * 32 + offset * ctx.width),
                    None if direct_pack4 || byte_wire => ptr::null_mut(),
                    None => slot.payload.at(layout.states + offset * ctx.width),
                },
                counts: slot
                    .payload
                    .at((index + 1) * SHARD_DESCRIPTOR_BYTES - std::mem::size_of::<u32>()),
            };
            self.incoming_owner.push(PacketOwnerShard {
                shard: run.shard as usize,
                queue,
                rows: run.rows,
                byte_wire,
            });
            offset += run.rows as usize;
        }
        self.received_packets = self
            .received_packets
            .checked_add(1)
            .ok_or("SEGMENT_PACKET_COUNTER_OVERFLOW")?;
        Ok(())
    }

    pub(super) fn remote_owner_config(&self) -> PacketConfig {
        let mut config = self.config;
        if let Some(owner) = config.pack4_wire_owner {
            config.pack4_owner = Some(owner);
        }
        config
    }
}

impl PacketRuntime {
    fn start_receive(&mut self, ctx: WireContext, packet: Packet) -> PacketResult<()> {
        let ring = self.receive.as_ref().ok_or("SEGMENT_PACKET_RX_MISSING")?;
        if ring.wire_slot.is_some() {
            return Err("SEGMENT_PACKET_RX_WIRE_BUSY".into());
        }
        let index = (0..self.rx.len())
            .map(|step| (ring.next_slot + step) % self.rx.len())
            .find(|&index| self.rx[index].packet.is_none())
            .ok_or("SEGMENT_PACKET_RX_CREDIT_EXHAUSTED")?;

        self.submit_wire(ctx, packet, None, index)?;
        let ring = self.receive.as_mut().ok_or("SEGMENT_PACKET_RX_MISSING")?;
        let slot = &mut self.rx[index];
        let (wire, _) = slot.events()?;
        wire.record(&self.stream)?;
        wire.wait_on(&ring.unpack)?;
        slot.packet = Some(packet);
        slot.deadline = Some(Instant::now() + PACKET_TIMEOUT);
        slot.to_unpack = true;
        ring.wire_slot = Some(index);
        ring.next_slot = (index + 1) % self.rx.len();
        ring.live += 1;
        ring.peak_live = ring.peak_live.max(ring.live);
        let ring = self.receive.as_ref().ok_or("SEGMENT_PACKET_RX_MISSING")?;
        self.queue_unpack(ctx, packet, index, &ring.unpack)?;
        self.rx[index].events()?.1.record(&ring.unpack)?;
        Ok(())
    }

    fn advance_receive(
        &mut self,
        ctx: WireContext,
        stats: &mut PacketLayerStats,
        owners: &[(usize, *mut c_void)],
    ) -> PacketResult<bool> {
        let mut progress = false;
        if let Some(index) = self.receive.as_ref().and_then(|ring| ring.wire_slot) {
            if self.rx[index].events()?.0.is_ready()? {
                let packet = self.rx[index].packet.ok_or("SEGMENT_PACKET_RX_STATE")?;
                self.client()?.send(PacketFrame::Received(packet))?;
                let ring = self.receive.as_mut().ok_or("SEGMENT_PACKET_RX_MISSING")?;
                ring.wire_slot = None;
                if owners.iter().any(|&(_, stream)| stream_pending(stream)) {
                    ring.overlap_wire += 1;
                }
                progress = true;
            }
        }
        for index in 0..self.rx.len() {
            if self.rx[index].to_unpack {
                let slot = &self.rx[index];
                if slot.events()?.1.is_ready()? {
                    // Both events can complete after this poll checked the
                    // wire. Publish the owner lease only after Received.
                    if self.receive.as_ref().and_then(|ring| ring.wire_slot) == Some(index) {
                        continue;
                    }
                    let packet = slot.packet.ok_or("SEGMENT_PACKET_RX_STATE")?;
                    self.read_incoming(ctx, packet, index, stats)?;
                    let slot = &mut self.rx[index];
                    slot.remaining = self.incoming_owner.len();
                    slot.parked.extend(self.incoming_owner.drain(..));
                    slot.to_unpack = false;
                    slot.deadline = None;
                    progress = true;
                } else if slot
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return Err("SEGMENT_PACKET_RX_DEVICE_TIMEOUT".into());
                }
            }
            let ring = self.receive.as_mut().ok_or("SEGMENT_PACKET_RX_MISSING")?;
            let parked = &mut self.rx[index].parked;
            for _ in 0..parked.len() {
                let span = parked.pop_front().ok_or("SEGMENT_PACKET_RX_PARKED")?;
                if ring.banks.offer(span, OwnerSource::Remote(index))? {
                    if ring.banks.has_sibling(span.shard) {
                        ring.double_shards += 1;
                        ring.peak_double = ring.peak_double.max(ring.double_shards);
                        if owners
                            .iter()
                            .any(|&(shard, stream)| shard == span.shard && stream_pending(stream))
                        {
                            ring.overlap_fills += 1;
                        }
                    }
                    progress = true;
                } else {
                    parked.push_back(span);
                }
            }
        }
        Ok(progress)
    }

    pub(super) fn progress_carousel(
        &mut self,
        ctx: WireContext,
        stats: &mut PacketLayerStats,
        stop: &impl Fn() -> u32,
        owners: &[(usize, *mut c_void)],
        sealed: bool,
    ) -> PacketResult<bool> {
        check_packet_stop(stop)?;
        let terminal = self.client()?.terminal_code();
        if terminal != 0 {
            return Err(PacketError::Terminal(terminal));
        }
        let progress = self.advance_receive(ctx, stats, owners)?;
        let offered = match self.poll_transmit()? {
            TransmitPoll::Progressed => return Ok(true),
            TransmitPoll::Waiting => return Ok(progress),
            TransmitPoll::PollHub(offered) => offered,
        };

        match self.client()?.next(None) {
            Some(PacketFrame::Grant(packet)) if packet.source == ctx.rank => {
                self.submit_wire(ctx, packet, offered, 0)?;
                self.set_transmit_progress(TransmitProgress::Submitted(
                    Instant::now() + PACKET_TIMEOUT,
                ));
            }
            Some(PacketFrame::Grant(packet)) => self.start_receive(ctx, packet)?,
            Some(PacketFrame::Commit(depth)) if sealed && depth == ctx.depth => {
                let ring = self.receive.as_mut().ok_or("SEGMENT_PACKET_RX_MISSING")?;
                if ring.live != 0
                    || ring.local_live.iter().any(|&count| count != 0)
                    || !ring.banks.is_empty()
                    || offered.is_some()
                {
                    return Err("SEGMENT_PACKET_RX_COMMIT_WITH_LIVE_LEASES".into());
                }
                ring.commit = true;
            }
            Some(PacketFrame::Abort(code)) => return Err(PacketError::Terminal(code)),
            Some(_) => return Err("SEGMENT_PACKET_UNEXPECTED_CAROUSEL_FRAME".into()),
            None => return Ok(progress),
        };
        Ok(true)
    }

    pub(super) fn offer_local_span(&mut self, slot: usize, span: PacketOwnerShard) -> Result<bool> {
        let ring = self.receive.as_mut().ok_or("SEGMENT_PACKET_RX_MISSING")?;
        let local = ring
            .local_live
            .get_mut(slot)
            .ok_or("SEGMENT_PACKET_LOCAL_SLOT")?;
        if !ring.banks.offer(span, OwnerSource::Local(slot))? {
            return Ok(false);
        }
        *local += 1;
        if ring.banks.has_sibling(span.shard) {
            ring.double_shards += 1;
            ring.peak_double = ring.peak_double.max(ring.double_shards);
        }
        Ok(true)
    }

    pub(super) fn has_pending_local(&self, slot: usize) -> Result<bool> {
        let ring = self.receive.as_ref().ok_or("SEGMENT_PACKET_RX_MISSING")?;
        let local = ring
            .local_live
            .get(slot)
            .ok_or("SEGMENT_PACKET_LOCAL_SLOT")?;
        Ok(*local != 0)
    }

    pub(super) fn owner_wave(&mut self, lanes: usize) -> Result<Vec<OwnerLease>> {
        let ring = self.receive.as_mut().ok_or("SEGMENT_PACKET_RX_MISSING")?;
        ring.banks.begin_wave(lanes, &mut ring.wave)?;
        if ring.wave.is_empty() {
            return Ok(Vec::new());
        }
        Ok(std::mem::take(&mut ring.wave))
    }

    pub(super) fn complete_owner_wave(&mut self, wave: Vec<OwnerLease>) -> PacketResult<()> {
        let ring = self.receive.as_mut().ok_or("SEGMENT_PACKET_RX_MISSING")?;
        for lease in &wave {
            if ring.banks.has_sibling(lease.span.shard) {
                ring.double_shards -= 1;
            }
        }
        ring.banks.complete_wave(&wave)?;
        for lease in &wave {
            match lease.source {
                OwnerSource::Local(slot) => {
                    self.receive
                        .as_mut()
                        .ok_or("SEGMENT_PACKET_RX_MISSING")?
                        .local_live[slot] -= 1;
                }
                OwnerSource::Remote(index) => self.retire_receive(index)?,
            }
        }
        self.receive
            .as_mut()
            .ok_or("SEGMENT_PACKET_RX_MISSING")?
            .wave = wave;
        Ok(())
    }

    pub(super) fn owner_source_rank(&self, lease: &OwnerLease, rank: u32) -> Result<u32> {
        match lease.source {
            OwnerSource::Local(_) => Ok(rank),
            OwnerSource::Remote(index) => self.rx[index]
                .packet
                .map(|packet| packet.source)
                .ok_or_else(|| "SEGMENT_PACKET_RX_STATE".into()),
        }
    }

    pub(super) fn take_commit(&mut self) -> Result<bool> {
        let ring = self.receive.as_mut().ok_or("SEGMENT_PACKET_RX_MISSING")?;
        Ok(std::mem::take(&mut ring.commit))
    }

    fn retire_receive(&mut self, index: usize) -> PacketResult<()> {
        let slot = &mut self.rx[index];
        slot.remaining = slot
            .remaining
            .checked_sub(1)
            .ok_or("SEGMENT_PACKET_RX_SHARD_CREDIT")?;
        if slot.remaining == 0 {
            if slot.to_unpack
                || !slot.parked.is_empty()
                || self.receive.as_ref().and_then(|ring| ring.wire_slot) == Some(index)
            {
                return Err("SEGMENT_PACKET_RX_RELEASE_IN_FLIGHT".into());
            }
            let packet = slot.packet.take().ok_or("SEGMENT_PACKET_RX_STATE")?;
            self.client()?.send(PacketFrame::Ack(packet))?;
            self.receive
                .as_mut()
                .ok_or("SEGMENT_PACKET_RX_MISSING")?
                .live -= 1;
        }
        Ok(())
    }
}

fn stream_pending(stream: *mut c_void) -> bool {
    unsafe { cudaStreamQuery(stream) == 600 }
}

impl SegmentedDistributedBfs {
    pub(super) fn packet_wire_context(&self, epoch: u64) -> WireContext {
        WireContext {
            rank: self.rank,
            depth: epoch,
            world: self.plan.world,
            shards: self.plan.shards,
            queue_capacity: self.plan.queue_capacity,
            elements: self.plan.elements,
            width: self.plan.queue_payload_bytes(),
            comm: self.comm.ptr,
        }
    }

    pub(super) fn service_receive_carousel(
        &mut self,
        transport: &mut PacketRuntime,
        future: u32,
        stats: &mut PacketLayerStats,
        stop: &impl Fn() -> u32,
        sealed: bool,
    ) -> PacketResult<bool> {
        let ctx = self.packet_wire_context(transport.epoch);
        // Once sealed, ranks without work must accept Commit and use the layer
        // boundary vote. A live Window lease still prevents the hub from committing.
        let transport_stop = || if sealed { 0 } else { stop() };
        let progress = transport.progress_carousel(ctx, stats, &transport_stop, &[], sealed)?;
        let wave = transport.owner_wave(self.lanes.len())?;
        if wave.is_empty() {
            return Ok(progress);
        }
        let ring = transport
            .receive
            .as_mut()
            .ok_or("SEGMENT_PACKET_RX_MISSING")?;
        let mut spans = std::mem::take(&mut ring.spans);
        spans.clear();
        spans.extend(wave.iter().map(|lease| lease.span));
        let config = match wave[0].source {
            OwnerSource::Local(_) => transport.config,
            OwnerSource::Remote(_) => transport.remote_owner_config(),
        };
        let watchdog = transport.client()?.owner_watchdog();
        let mut pump_error = None;
        let mut pump = |owners: &[(usize, *mut c_void)]| -> Result<()> {
            transport
                .progress_carousel(ctx, stats, &transport_stop, owners, sealed)
                .map(|_| ())
                .map_err(|error| {
                    pump_error = Some(error);
                    "SEGMENT_PACKET_OWNER_PROGRESS".into()
                })
        };
        self.count_packet_owner_input(&spans)?;
        let status = self.accept_input_with_progress(
            OwnerInputs::Compact {
                shards: &spans,
                config,
            },
            future,
            None,
            Some(&mut pump),
            watchdog.as_deref(),
        );
        if let Some(error) = pump_error {
            return Err(error);
        }
        match status? {
            0 => {}
            10 => return Err(PacketError::Terminal(10)),
            other => return Err(format!("SEGMENTED_OWNER_{other}").into()),
        }
        if self.window.is_some() {
            for lease in &wave {
                let source = transport.owner_source_rank(lease, self.rank)?;
                let client = transport.client()?;
                let mut peer_abort = 0;
                self.window_record_edges(lease.span, source, || {
                    peer_abort = client.terminal_code();
                    if peer_abort != 0 {
                        return Err("SEGMENTED_WINDOW_PEER_ABORT".into());
                    }
                    match stop() {
                        1 => Err("CANCELLED".into()),
                        2 => Err("TIME_LIMIT".into()),
                        _ => Ok(()),
                    }
                })
                .map_err(|error| match error.as_str() {
                    "SEGMENTED_WINDOW_LOOKUP_RESOURCE" => PacketError::Terminal(10),
                    "SEGMENTED_WINDOW_PEER_ABORT" => PacketError::Terminal(peer_abort),
                    "CANCELLED" => PacketError::Terminal(PACKET_CANCELLED),
                    "TIME_LIMIT" => PacketError::Terminal(PACKET_DEADLINE),
                    "LINEAGE_LIMIT" => PacketError::Terminal(PACKET_LINEAGE_LIMIT),
                    "LINEAGE_DISK_FULL" => PacketError::Terminal(PACKET_LINEAGE_DISK_FULL),
                    _ => PacketError::Failure(error),
                })?;
            }
        }
        transport.complete_owner_wave(wave)?;
        transport
            .receive
            .as_mut()
            .ok_or("SEGMENT_PACKET_RX_MISSING")?
            .spans = spans;
        Ok(true)
    }

    pub(super) fn accept_local_leases(
        &mut self,
        transport: &mut PacketRuntime,
        slot: usize,
        spans: &[PacketOwnerShard],
        future: u32,
        stats: &mut PacketLayerStats,
        stop: &impl Fn() -> u32,
    ) -> PacketResult<()> {
        for &span in spans {
            loop {
                if transport.offer_local_span(slot, span)? {
                    break;
                }
                if !self.service_receive_carousel(transport, future, stats, stop, false)? {
                    return Err("SEGMENT_PACKET_OWNER_BANK_STALLED".into());
                }
            }
        }
        while transport.has_pending_local(slot)? {
            if !self.service_receive_carousel(transport, future, stats, stop, false)? {
                return Err("SEGMENT_PACKET_OWNER_LOCAL_STALLED".into());
            }
        }
        Ok(())
    }

    pub(super) fn finish_receive_carousel(
        &mut self,
        transport: &mut PacketRuntime,
        future: u32,
        mut stats: PacketLayerStats,
        stop: &impl Fn() -> u32,
    ) -> PacketResult<PacketLayerStats> {
        loop {
            if !self.service_receive_carousel(transport, future, &mut stats, stop, true)? {
                // Healthy peers may still be expanding a large layer. The
                // hub times outstanding credits and detects lost connections.
                std::thread::sleep(Duration::from_micros(50));
            }
            if transport.take_commit()? {
                self.note_sync(|counts| counts.packet_hub_layer_commits += 1);
                return Ok(stats);
            }
        }
    }
}
