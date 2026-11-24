//! Raw records and exact state rows on the compact three-slot exchange.
use super::*;

const SLOTS: u32 = 3;
const RECORD_BYTES: usize = std::mem::size_of::<GenericRouteRecord>();

/// One nonempty shard run. The states use the producer's representation and
/// are contiguous for this run. Pointers are borrowed until their wave retires.
#[derive(Clone, Copy)]
pub(crate) struct RawShardRun {
    pub(crate) shard: usize,
    pub(crate) source_rank: u32,
    pub(crate) rows: u32,
    pub(crate) records: *mut c_void,
    pub(crate) states: *mut c_void,
}

pub(crate) struct RawPacketTransport {
    exchange: PacketRuntime,
    ctx: receive::WireContext,
    sequence: u64,
    sealed: bool,
    owner_runs: Vec<RawShardRun>,
}

impl RawPacketTransport {
    pub(crate) fn device_buffer_requirements(
        shards: u32,
        queue_capacity: u32,
        state_bytes: usize,
    ) -> Result<[usize; 6]> {
        if shards == 0 || queue_capacity == 0 {
            return Err("SEGMENT_PACKET_RAW_GEOMETRY".into());
        }
        let bytes = PacketBytes::new(
            shards.min(queue_capacity),
            queue_capacity,
            RECORD_BYTES,
            state_bytes,
        )?
        .total;
        // Order matches the transport's allocations: TX, then RX.
        Ok([bytes; 6])
    }

    pub(crate) fn new(
        endpoint: PacketEndpoint,
        id: &[u8; 128],
        world: u32,
        rank: u32,
        shards: u32,
        queue_capacity: u32,
        state_bytes: usize,
        device: i32,
    ) -> Result<Self> {
        Self::device_buffer_requirements(shards, queue_capacity, state_bytes)?;
        if world < 2 || rank >= world {
            return Err("SEGMENT_PACKET_RAW_RANK".into());
        }
        let exchange = PacketRuntime::new_exchange(
            PacketConfig::raw(endpoint)?,
            id,
            shards,
            queue_capacity,
            state_bytes,
            [SLOTS, SLOTS],
            device,
        )?;
        Ok(Self {
            exchange,
            ctx: receive::WireContext {
                rank,
                depth: 0,
                world,
                shards,
                queue_capacity,
                elements: 0,
                width: state_bytes,
                comm: ptr::null_mut(),
            },
            sequence: 0,
            sealed: false,
            owner_runs: Vec::new(),
        })
    }

    /// Start this generation after all participants have admitted its buffers.
    ///
    /// Safety: the communicator wrapper must outlive this transport or be
    /// detached first. Abort a failed communicator before draining the streams.
    pub(crate) unsafe fn connect(&mut self, comm: *mut c_void) -> Result<()> {
        if comm.is_null() || self.exchange.client.is_some() {
            return Err("SEGMENT_PACKET_RAW_CONNECT".into());
        }
        let client = self.exchange.config.endpoint.connect(
            self.ctx.rank,
            self.ctx.world,
            self.ctx.shards,
            self.ctx.queue_capacity,
            &self.exchange.identity,
            SLOTS,
        )?;
        crate::generic_native::checked!(mgbfs_nccl_bind_cancel, unsafe {
            mgbfs_nccl_bind_cancel(
                comm,
                Some(packet_cancel),
                client.terminal_ptr().cast_mut().cast(),
            )
        })?;
        self.exchange.client = Some(client);
        self.ctx.comm = comm;
        Ok(())
    }

    /// Queue complete ordered runs from one destination within a ready tile.
    /// False means TX backpressure and leaves the input untouched.
    ///
    /// Safety: the source planes must stay GPU-readable until the last pack
    /// event is consumed. The caller supplies one contiguous records plane
    /// and one contiguous state plane covering the listed runs in order.
    pub(crate) unsafe fn stage(
        &mut self,
        depth: u64,
        destination: u32,
        runs: &[ShardRun],
        records: *mut c_void,
        states: *mut c_void,
    ) -> Result<bool> {
        self.check_layer(depth)?;
        if self.sealed || destination >= self.ctx.world || destination == self.ctx.rank {
            return Err("SEGMENT_PACKET_RAW_DESTINATION".into());
        }
        let tx = self
            .exchange
            .carousel
            .as_ref()
            .ok_or("SEGMENT_PACKET_CAROUSEL_MISSING")?;
        let slots = self.exchange.tx.len();
        if tx.queue.len() == slots {
            return Ok(false);
        }
        let slot = tx.next_slot;
        let mut rows = 0u32;
        let mut previous = None;
        for run in runs {
            if run.rows == 0
                || run.shard >= self.ctx.shards
                || previous.is_some_and(|last| last >= run.shard)
            {
                return Err("SEGMENT_PACKET_SHARD_HEADER".into());
            }
            previous = Some(run.shard);
            rows = rows
                .checked_add(run.rows)
                .ok_or("SEGMENT_PACKET_PRODUCER_SIZE")?;
        }
        if rows == 0
            || rows > self.ctx.queue_capacity
            || records.is_null()
            || (self.ctx.width != 0 && states.is_null())
        {
            return Err("SEGMENT_PACKET_RAW_BOUNDS".into());
        }
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or("SEGMENT_PACKET_SEQUENCE_OVERFLOW")?;
        self.exchange.stage_payload(
            slot,
            runs,
            rows,
            0,
            self.ctx.width,
            QueueSide {
                records,
                states,
                counts: ptr::null_mut(),
            },
        )?;
        self.exchange.enqueue_transmit(StagedSend {
            packet: Packet {
                depth,
                sequence: self.sequence,
                source: self.ctx.rank,
                destination,
                descriptors: runs.len() as u32,
                rows,
            },
            slot,
        })?;
        self.sequence = sequence;
        Ok(true)
    }

    /// Record once after the last successful stage reading this producer slot.
    pub(crate) fn record_source_read(&self, slot: usize) -> Result<()> {
        let (event, pack) = self.exchange.source_read_event(slot)?;
        event.record(pack)
    }

    /// Make the next producer pass wait for every pack reading this slot.
    pub(crate) fn wait_source_read(&self, slot: usize, stream: &Stream) -> Result<()> {
        self.exchange.source_read_event(slot)?.0.wait_on(stream)
    }

    /// Safety: local rows stay live until this slot has no pending owner lease.
    pub(crate) unsafe fn offer_local(&mut self, slot: usize, run: RawShardRun) -> Result<bool> {
        if self.ctx.comm.is_null()
            || self.sealed
            || run.source_rank != self.ctx.rank
            || run.rows > self.ctx.queue_capacity
            || run.records.is_null()
            || (self.ctx.width != 0 && run.states.is_null())
        {
            return Err("SEGMENT_PACKET_RAW_LOCAL".into());
        }
        self.exchange.offer_local_span(
            slot,
            PacketOwnerShard {
                shard: run.shard,
                rows: run.rows,
                queue: QueueSide {
                    records: run.records,
                    states: run.states,
                    counts: ptr::null_mut(),
                },
                byte_wire: false,
            },
        )
    }

    pub(crate) fn has_pending_local(&self, slot: usize) -> Result<bool> {
        self.exchange.has_pending_local(slot)
    }

    /// Run a wave of distinct owner shards, retaining all receive credits.
    /// The consumer may pump transport while GPU work is pending. It must
    /// drain all GPU reads of the provided spans before returning, even on
    /// error. Only a successful wave retires its leases and sends RX ACKs.
    pub(crate) fn progress(
        &mut self,
        depth: u64,
        stop: &impl Fn() -> u32,
        owner_lanes: usize,
        mut consume: impl FnMut(
            &[RawShardRun],
            &mut dyn FnMut(&[(usize, *mut c_void)]) -> Result<()>,
        ) -> Result<()>,
    ) -> PacketResult<bool> {
        self.check_layer(depth)?;
        let no_stop = || 0;
        let stop: &dyn Fn() -> u32 = if self.sealed { &no_stop } else { stop };
        let mut stats = PacketLayerStats::default();
        let exchange = &mut self.exchange;
        let progressed =
            exchange.progress_carousel(self.ctx, &mut stats, &stop, &[], self.sealed)?;
        let wave = exchange.owner_wave(owner_lanes)?;
        if wave.is_empty() {
            return Ok(progressed);
        }
        self.owner_runs.clear();
        for lease in &wave {
            self.owner_runs.push(RawShardRun {
                shard: lease.span.shard,
                source_rank: exchange.owner_source_rank(lease, self.ctx.rank)?,
                rows: lease.span.rows,
                records: lease.span.queue.records,
                states: lease.span.queue.states,
            });
        }
        let mut pump_error = None;
        let status = consume(&self.owner_runs, &mut |owners| {
            if pump_error.is_some() {
                return Err("SEGMENT_PACKET_OWNER_PROGRESS".into());
            }
            exchange
                .progress_carousel(self.ctx, &mut stats, &stop, owners, self.sealed)
                .map(|_| ())
                .map_err(|error| {
                    pump_error = Some(error);
                    "SEGMENT_PACKET_OWNER_PROGRESS".into()
                })
        });
        if let Some(error) = pump_error {
            return Err(error);
        }
        status?;
        exchange.complete_owner_wave(wave)?;
        Ok(true)
    }

    pub(crate) fn seal(&mut self, depth: u64) -> PacketResult<bool> {
        self.check_layer(depth)?;
        if self.sealed {
            return Ok(true);
        }
        let tx = self
            .exchange
            .carousel
            .as_ref()
            .ok_or("SEGMENT_PACKET_CAROUSEL_MISSING")?;
        if !tx.queue.is_empty() || self.has_pending_local(0)? || self.has_pending_local(1)? {
            return Ok(false);
        }
        let client = self.exchange.client()?;
        let terminal = client.terminal_code();
        if terminal != 0 {
            return Err(PacketError::Terminal(terminal));
        }
        client.send(PacketFrame::Seal(depth))?;
        self.sealed = true;
        Ok(true)
    }

    pub(crate) fn committed(&mut self, depth: u64) -> PacketResult<bool> {
        self.check_layer(depth)?;
        if !self.sealed || !self.exchange.take_commit()? {
            return Ok(false);
        }
        self.ctx.depth = depth
            .checked_add(1)
            .ok_or("SEGMENT_PACKET_DEPTH_OVERFLOW")?;
        self.sequence = 0;
        self.sealed = false;
        Ok(true)
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.ctx.depth
    }

    pub(crate) fn notify_failure(&mut self) {
        if let Ok(client) = self.exchange.client() {
            if client.terminal_code() == 0 {
                let _ = client.send(PacketFrame::Fail(REMOTE_FAILURE));
            }
        }
    }

    /// Called after the owning runtime aborts the failed communicator.
    pub(crate) fn drain_after_abort(&mut self) -> Result<()> {
        self.ctx.comm = ptr::null_mut();
        self.exchange.drain_after_abort()
    }

    fn check_layer(&self, depth: u64) -> Result<()> {
        if self.ctx.comm.is_null() || depth != self.ctx.depth {
            return Err("SEGMENT_PACKET_RAW_LAYER".into());
        }
        Ok(())
    }
}

impl Drop for RawPacketTransport {
    fn drop(&mut self) {
        // Keep the owner of the callback's AtomicU32 alive until this binding
        // has been cleared or the communicator has been aborted.
        if !self.ctx.comm.is_null() {
            unsafe { mgbfs_nccl_bind_cancel(self.ctx.comm, None, ptr::null_mut()) };
        }
    }
}
