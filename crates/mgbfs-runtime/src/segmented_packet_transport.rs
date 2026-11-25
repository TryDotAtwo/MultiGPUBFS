//! Sparse peer rendezvous. The hub handles fixed-size metadata only.
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

const FRAME_BYTES: usize = 40;
const HELLO_BYTES: usize = 36;
const DEADLINE: Duration = Duration::from_secs(120);
const PEER_OUTPUT_CAPACITY: usize = 4;
const CAROUSEL_RECEIVE_SLOTS: usize = 3;
#[cfg(all(feature = "cuda", target_os = "linux"))]
const OWNER_STALL_EXIT_STATUS: libc::c_int = 75;
pub(crate) const REMOTE_FAILURE: u32 = u32::MAX;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) const PACKET_CANCELLED: u32 = 101;
#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
pub(crate) const PACKET_DEADLINE: u32 = 102;
/// A Window owner cannot stage the rest of the uncommitted layer.
pub(crate) const PACKET_LINEAGE_LIMIT: u32 = 103;
pub(crate) const PACKET_LINEAGE_DISK_FULL: u32 = 104;
type Result<T> = std::result::Result<T, String>;
pub(crate) type PacketJobToken = [u8; 16];

/// Independent of the CUDA-calling thread, which can stall during a local
/// owner pass even when the hub has no transfer credit to time out.
pub(crate) struct OwnerWaveWatchdog {
    origin: Instant,
    deadline_ms: AtomicU64,
}

impl OwnerWaveWatchdog {
    fn start(socket: TcpStream, rank: u32) -> Result<Arc<Self>> {
        let watchdog = Arc::new(Self {
            origin: Instant::now(),
            deadline_ms: AtomicU64::new(0),
        });
        let observer = Arc::downgrade(&watchdog);
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let managed_worker = std::env::var_os("MGBFS_JOB_PHASE_PATH").is_some();
        thread::Builder::new()
            .name("segment-owner-watchdog".into())
            .stack_size(128 * 1024)
            .spawn(move || loop {
                thread::sleep(Duration::from_millis(100));
                let Some(watchdog) = observer.upgrade() else {
                    break;
                };
                let deadline = watchdog.deadline_ms.load(Ordering::Acquire);
                if deadline == 0
                    || watchdog.elapsed_ms() < deadline
                    || watchdog
                        .deadline_ms
                        .compare_exchange(deadline, 0, Ordering::AcqRel, Ordering::Acquire)
                        .is_err()
                {
                    continue;
                }
                // Wake the hub even if this rank has no active wire transfer.
                let _ = socket.shutdown(Shutdown::Both);
                // The isolated worker still owns live GPU references. Do not
                // block on logging, run its destructors or return it to the session.
                #[cfg(all(feature = "cuda", target_os = "linux"))]
                if managed_worker {
                    unsafe { libc::_exit(OWNER_STALL_EXIT_STATUS) };
                }
                eprintln!("SEGMENT_PACKET_OWNER_STALL rank={rank}");
                break;
            })
            .map_err(|e| e.to_string())?;
        Ok(watchdog)
    }

    fn elapsed_ms(&self) -> u64 {
        self.origin.elapsed().as_millis() as u64
    }

    #[cfg_attr(not(all(feature = "cuda", target_os = "linux")), allow(dead_code))]
    pub(crate) fn lease(&self) -> OwnerWaveLease<'_> {
        self.lease_for(DEADLINE)
    }

    fn lease_for(&self, timeout: Duration) -> OwnerWaveLease<'_> {
        let deadline = self.elapsed_ms() + (timeout.as_millis() as u64).max(1);
        self.deadline_ms.store(deadline, Ordering::Release);
        OwnerWaveLease(self)
    }
}

pub(crate) struct OwnerWaveLease<'a>(&'a OwnerWaveWatchdog);

impl Drop for OwnerWaveLease<'_> {
    fn drop(&mut self) {
        self.0.deadline_ms.store(0, Ordering::Release);
    }
}

/// The same packet rendezvous can carry traversal or path candidate records.
/// It does not choose the encoding or interpret the payload.
#[cfg(all(feature = "cuda", target_os = "linux"))]
#[derive(Clone, Copy)]
pub(crate) struct PacketEndpoint {
    pub(crate) advertised: SocketAddr,
    bind: SocketAddr,
    job_token: PacketJobToken,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
impl PacketEndpoint {
    pub(crate) fn from_env(
        world: u32,
        rank: u32,
        job_token: Option<PacketJobToken>,
    ) -> Result<Option<Self>> {
        if world == 1 {
            return Ok(None);
        }
        let advertised: SocketAddr = match std::env::var("MGBFS_SEGMENT_PACKET_CONTROL_ADDR") {
            Ok(value) => value.parse().map_err(|_| "SEGMENT_PACKET_CONTROL_ADDR")?,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(_) => return Err("SEGMENT_PACKET_CONTROL_ADDR".into()),
        };
        let job_token = job_token.ok_or("SEGMENT_PACKET_JOB_TOKEN_MISSING")?;
        let bind = if rank == 0 {
            match std::env::var("MGBFS_SEGMENT_PACKET_BIND_ADDR") {
                Ok(value) => value.parse().map_err(|_| "SEGMENT_PACKET_BIND_ADDR")?,
                Err(std::env::VarError::NotPresent) => advertised,
                Err(_) => return Err("SEGMENT_PACKET_BIND_ADDR".into()),
            }
        } else {
            advertised
        };
        if advertised.port() == 0 || bind.port() == 0 {
            return Err("SEGMENT_PACKET_CONTROL_PORT".into());
        }
        Ok(Some(Self {
            advertised,
            bind,
            job_token,
        }))
    }

    pub(crate) fn connect(
        self,
        rank: u32,
        world: u32,
        shards: u32,
        max_rows: u32,
        identity: &[u8; 128],
        receive_slots: u32,
    ) -> Result<PacketClient> {
        PacketClient::connect_with_receive_slots(
            self.advertised,
            self.bind,
            rank,
            world,
            shards,
            max_rows,
            identity,
            &self.job_token,
            receive_slots,
        )
    }
}

pub(crate) fn parse_job_token(value: Option<&str>) -> Result<PacketJobToken> {
    let value = value.ok_or("SEGMENT_PACKET_JOB_TOKEN_MISSING")?.as_bytes();
    let mut token = [0; 16];
    if value.len() != token.len() * 2 {
        return Err("SEGMENT_PACKET_JOB_TOKEN_INVALID".into());
    }
    for (pair, byte) in value.chunks_exact(2).zip(&mut token) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or("SEGMENT_PACKET_JOB_TOKEN_INVALID")?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or("SEGMENT_PACKET_JOB_TOKEN_INVALID")?;
        *byte = (high * 16 + low) as u8;
    }
    Ok(token)
}

fn handshake_tag(identity: &[u8; 128], job_token: &PacketJobToken, receive_slots: u32) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mgbfs-segment-packet-v5");
    digest.update(identity);
    digest.update(job_token);
    if receive_slots != 1 {
        digest.update(receive_slots.to_le_bytes());
    }
    digest.finalize().into()
}

fn write_hello(socket: &mut TcpStream, rank: u32, tag: &[u8; 32]) -> std::io::Result<()> {
    let mut hello = [0u8; HELLO_BYTES];
    hello[..4].copy_from_slice(&rank.to_le_bytes());
    hello[4..].copy_from_slice(tag);
    socket.write_all(&hello)
}

fn startup_disconnected(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::UnexpectedEof
            | ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::BrokenPipe
            | ErrorKind::NotConnected
    )
}

fn read_ready(socket: &mut TcpStream, deadline: Instant) -> Result<bool> {
    let mut bytes = [0u8; FRAME_BYTES];
    let mut filled = 0;
    while filled < bytes.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("SEGMENT_PACKET_STARTUP_TIMEOUT".into());
        }
        socket
            .set_read_timeout(Some(remaining))
            .map_err(|e| e.to_string())?;
        match socket.read(&mut bytes[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err("SEGMENT_PACKET_READ truncated startup frame".into()),
            Ok(size) => filled += size,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) if filled == 0 && startup_disconnected(&error) => return Ok(false),
            Err(error) => return Err(format!("SEGMENT_PACKET_READ {error}")),
        }
    }
    if Frame::decode(bytes)? != Frame::Ready {
        return Err("SEGMENT_PACKET_STARTUP_REJECTED".into());
    }
    Ok(true)
}

fn connect_until_ready(
    advertised: SocketAddr,
    rank: u32,
    tag: &[u8; 32],
    deadline: Instant,
) -> Result<TcpStream> {
    let mut last_error = "SEGMENT_PACKET_STARTUP_TIMEOUT".to_owned();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(last_error);
        }
        last_error = match TcpStream::connect_timeout(
            &advertised,
            remaining.min(Duration::from_millis(250)),
        ) {
            Err(error) => format!("SEGMENT_PACKET_CONNECT {error}"),
            Ok(mut socket) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(last_error);
                }
                socket.set_nodelay(true).map_err(|e| e.to_string())?;
                socket
                    .set_write_timeout(Some(remaining))
                    .map_err(|e| e.to_string())?;
                match write_hello(&mut socket, rank, tag) {
                    Ok(()) if read_ready(&mut socket, deadline)? => return Ok(socket),
                    // A relay can accept TCP before the destination is listening.
                    Ok(()) => "SEGMENT_PACKET_READ closed before READY".into(),
                    Err(error) if startup_disconnected(&error) => {
                        format!("SEGMENT_PACKET_HANDSHAKE {error}")
                    }
                    Err(error) => return Err(format!("SEGMENT_PACKET_HANDSHAKE {error}")),
                }
            }
        };
        thread::sleep(
            deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(20)),
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Packet {
    pub depth: u64,
    pub sequence: u64,
    pub source: u32,
    pub destination: u32,
    pub descriptors: u32,
    pub rows: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Frame {
    Ready,
    Offer(Packet),
    Grant(Packet),
    Ack(Packet),
    Received(Packet),
    Seal(u64),
    Commit(u64),
    Fail(u32),
    Abort(u32),
    Close,
}

impl Frame {
    fn encode(self) -> [u8; FRAME_BYTES] {
        let mut bytes = [0; FRAME_BYTES];
        let packet = match self {
            Self::Ready => {
                bytes[0] = 1;
                return bytes;
            }
            Self::Offer(p) => {
                bytes[0] = 2;
                p
            }
            Self::Grant(p) => {
                bytes[0] = 3;
                p
            }
            Self::Ack(p) => {
                bytes[0] = 4;
                p
            }
            Self::Received(p) => {
                bytes[0] = 5;
                p
            }
            Self::Seal(depth) | Self::Commit(depth) => {
                bytes[0] = if matches!(self, Self::Seal(_)) { 6 } else { 7 };
                bytes[8..16].copy_from_slice(&depth.to_le_bytes());
                return bytes;
            }
            Self::Fail(code) | Self::Abort(code) => {
                bytes[0] = if matches!(self, Self::Fail(_)) { 8 } else { 9 };
                bytes[36..40].copy_from_slice(&code.to_le_bytes());
                return bytes;
            }
            Self::Close => {
                bytes[0] = 10;
                return bytes;
            }
        };
        bytes[8..16].copy_from_slice(&packet.depth.to_le_bytes());
        bytes[16..24].copy_from_slice(&packet.sequence.to_le_bytes());
        bytes[24..28].copy_from_slice(&packet.source.to_le_bytes());
        bytes[28..32].copy_from_slice(&packet.destination.to_le_bytes());
        bytes[32..36].copy_from_slice(&packet.descriptors.to_le_bytes());
        bytes[36..40].copy_from_slice(&packet.rows.to_le_bytes());
        bytes
    }

    fn decode(bytes: [u8; FRAME_BYTES]) -> Result<Self> {
        let depth = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let sequence = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let packet = Packet {
            depth,
            sequence,
            source: word(24),
            destination: word(28),
            descriptors: word(32),
            rows: word(36),
        };
        let frame = match bytes[0] {
            1 => Self::Ready,
            2 => Self::Offer(packet),
            3 => Self::Grant(packet),
            4 => Self::Ack(packet),
            5 => Self::Received(packet),
            6 => Self::Seal(depth),
            7 => Self::Commit(depth),
            8 if word(36) != 0 => Self::Fail(word(36)),
            9 if word(36) != 0 => Self::Abort(word(36)),
            10 => Self::Close,
            _ => return Err("SEGMENT_PACKET_FRAME_KIND".into()),
        };
        if frame.encode() != bytes {
            return Err("SEGMENT_PACKET_FRAME_RESERVED".into());
        }
        Ok(frame)
    }

    fn read(socket: &mut TcpStream) -> Result<Self> {
        let mut bytes = [0; FRAME_BYTES];
        socket
            .read_exact(&mut bytes)
            .map_err(|e| format!("SEGMENT_PACKET_READ {e}"))?;
        Self::decode(bytes)
    }

    fn write(self, socket: &mut TcpStream) -> Result<()> {
        socket
            .write_all(&self.encode())
            .map_err(|e| format!("SEGMENT_PACKET_WRITE {e}"))
    }
}

pub(crate) const SHARD_DESCRIPTOR_BYTES: usize = 8;

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) type PacketWireFn = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    u64,
    u32,
    u32,
    *mut std::ffi::c_void,
) -> i32;

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) fn packet_wire_kernel() -> Result<PacketWireFn> {
    let symbol = unsafe {
        libc::dlsym(
            libc::RTLD_DEFAULT,
            b"mgbfs_nccl_segment_packet_v1\0".as_ptr().cast(),
        )
    };
    if symbol.is_null() {
        return Err("SEGMENT_PACKET_CUDA_ABI_UNAVAILABLE".into());
    }
    Ok(unsafe { std::mem::transmute::<*mut std::ffi::c_void, PacketWireFn>(symbol) })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) struct PacketBytes {
    pub(crate) header: usize,
    pub(crate) states: usize,
    pub(crate) total: usize,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
impl PacketBytes {
    pub(crate) fn new(
        descriptors: u32,
        rows: u32,
        record_bytes: usize,
        state_bytes: usize,
    ) -> Result<Self> {
        let rows = rows as usize;
        let header = descriptors as usize * SHARD_DESCRIPTOR_BYTES;
        let states = rows
            .checked_mul(record_bytes)
            .and_then(|bytes| header.checked_add(bytes))
            .ok_or("SEGMENT_PACKET_BUFFER_SIZE")?;
        let total = rows
            .checked_mul(state_bytes)
            .and_then(|bytes| states.checked_add(bytes))
            .ok_or("SEGMENT_PACKET_BUFFER_SIZE")?;
        Ok(Self {
            header,
            states,
            total,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShardRun {
    pub shard: u32,
    pub rows: u32,
}

pub(crate) fn encode_runs(runs: &[ShardRun], buffer: &mut Vec<u8>) {
    buffer.clear();
    for run in runs {
        buffer.extend_from_slice(&run.shard.to_le_bytes());
        buffer.extend_from_slice(&run.rows.to_le_bytes());
    }
}

pub(crate) fn decode_runs(
    bytes: &[u8],
    expected_rows: u32,
    shards: u32,
    runs: &mut Vec<ShardRun>,
) -> Result<()> {
    if bytes.is_empty() || bytes.len() % SHARD_DESCRIPTOR_BYTES != 0 {
        return Err("SEGMENT_PACKET_SHARD_HEADER_SIZE".into());
    }
    runs.clear();
    let mut rows_left = expected_rows;
    for bytes in bytes.chunks_exact(SHARD_DESCRIPTOR_BYTES) {
        let shard = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let rows = u32::from_le_bytes(bytes[4..].try_into().unwrap());
        if rows == 0
            || rows > rows_left
            || shard >= shards
            || runs.last().is_some_and(|previous| previous.shard >= shard)
        {
            return Err("SEGMENT_PACKET_SHARD_HEADER".into());
        }
        rows_left -= rows;
        runs.push(ShardRun { shard, rows });
    }
    if rows_left != 0 {
        return Err("SEGMENT_PACKET_SHARD_HEADER_ROWS".into());
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Active {
    packet: Packet,
    expires_at: Instant,
    source_ack: bool,
    received: bool,
    destination_ack: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct CreditSlot {
    destination: usize,
    slot: usize,
}

struct Scheduler {
    depth: u64,
    shards: u32,
    max_rows: u32,
    credit_timeout: Duration,
    cursor: usize,
    next_sequence: Vec<u64>,
    sealed: Vec<bool>,
    sealed_count: usize,
    closed: Vec<bool>,
    closed_count: usize,
    pending: Vec<Option<Packet>>,
    pending_sources: u128,
    receive_slots: usize,
    active: Vec<[Option<Active>; CAROUSEL_RECEIVE_SLOTS]>,
    active_destinations: u128,
    active_count: u32,
    peak_active: Arc<AtomicU32>,
    last_sent: Vec<Option<Packet>>,
    endpoint: Vec<Option<CreditSlot>>,
}

impl Scheduler {
    #[cfg(test)]
    fn new(world: u32, shards: u32, max_rows: u32) -> Self {
        Self::with_receive_slots(world, shards, max_rows, 1)
    }

    fn with_receive_slots(world: u32, shards: u32, max_rows: u32, receive_slots: u32) -> Self {
        let ranks = world as usize;
        Self {
            depth: 0,
            shards,
            max_rows,
            credit_timeout: DEADLINE,
            cursor: 0,
            next_sequence: vec![0; ranks],
            active_destinations: 0,
            pending_sources: 0,
            closed_count: 0,
            peak_active: Arc::new(AtomicU32::new(0)),
            sealed: vec![false; ranks],
            sealed_count: 0,
            closed: vec![false; ranks],
            pending: vec![None; ranks],
            receive_slots: receive_slots as usize,
            active_count: 0,
            active: vec![[None; CAROUSEL_RECEIVE_SLOTS]; ranks],
            last_sent: vec![None; ranks],
            endpoint: vec![None; ranks],
        }
    }

    fn next_credit_timeout(&self, now: Instant) -> Option<Duration> {
        let mut destinations = self.active_destinations;
        let mut earliest: Option<Instant> = None;
        while destinations != 0 {
            let destination = destinations.trailing_zeros() as usize;
            destinations &= destinations - 1;
            for credit in self.active[destination][..self.receive_slots]
                .iter()
                .flatten()
            {
                earliest =
                    Some(earliest.map_or(credit.expires_at, |time| time.min(credit.expires_at)));
            }
        }
        earliest.map(|expires| expires.saturating_duration_since(now))
    }

    fn sending(&self, rank: usize) -> bool {
        self.endpoint[rank].is_some_and(|endpoint| {
            self.active[endpoint.destination][endpoint.slot]
                .is_some_and(|credit| credit.packet.source as usize == rank)
        })
    }

    fn complete(&mut self, rank: usize, packet: Packet, received: bool) -> Result<bool> {
        let source = packet.source as usize;
        let destination = packet.destination as usize;
        if source >= self.active.len()
            || destination >= self.active.len()
            || source == destination
            || (rank != source && rank != destination)
            || (received && (rank != destination || self.receive_slots == 1))
        {
            return Err("SEGMENT_PACKET_ACK_MISMATCH".into());
        }

        let active = &mut self.active[destination][..self.receive_slots];
        let Some(slot) = active.iter().position(|credit| {
            credit.is_some_and(|credit| {
                let live = credit.packet;
                (live.source, live.depth, live.sequence)
                    == (packet.source, packet.depth, packet.sequence)
            })
        }) else {
            if self.last_sent[source].is_some_and(|last| {
                packet == last || (packet.depth, packet.sequence) < (last.depth, last.sequence)
            }) {
                return Ok(false);
            }
            return Err("SEGMENT_PACKET_ACK_MISMATCH".into());
        };
        let credit = active[slot].as_mut().expect("occupied credit slot");
        if credit.packet != packet {
            return Err("SEGMENT_PACKET_ACK_MISMATCH".into());
        }
        let endpoint = CreditSlot { destination, slot };
        if rank == source {
            if credit.source_ack {
                return Ok(false);
            }
            if self.endpoint[rank] != Some(endpoint) {
                return Err("SEGMENT_PACKET_ACK_WITHOUT_CREDIT".into());
            }
            credit.source_ack = true;
            self.last_sent[source] = Some(packet);
            self.endpoint[source] = None;
        } else if received {
            if credit.received {
                return Ok(false);
            }
            if self.endpoint[rank] != Some(endpoint) {
                return Err("SEGMENT_PACKET_ACK_WITHOUT_CREDIT".into());
            }
            credit.received = true;
            self.endpoint[destination] = None;
        } else {
            if credit.destination_ack {
                return Ok(false);
            }
            if self.receive_slots == 1 {
                if self.endpoint[rank] != Some(endpoint) {
                    return Err("SEGMENT_PACKET_ACK_WITHOUT_CREDIT".into());
                }
            } else if !credit.received {
                return Err("SEGMENT_PACKET_OWNER_ACK_BEFORE_RECEIVE".into());
            }
            credit.destination_ack = true;
        }
        if credit.source_ack && credit.destination_ack {
            if self.receive_slots == 1 {
                self.endpoint[destination] = None;
            }
            active[slot] = None;
            self.active_count -= 1;
            if active.iter().all(Option::is_none) {
                self.active_destinations &= !(1u128 << destination);
            }
        }
        Ok(true)
    }

    #[cfg(test)]
    fn input(&mut self, rank: u32, frame: Frame) -> Result<Vec<(u32, Frame)>> {
        let mut output = Vec::new();
        self.input_into(rank, frame, &mut output)?;
        Ok(output)
    }

    fn input_into(
        &mut self,
        rank: u32,
        frame: Frame,
        output: &mut Vec<(u32, Frame)>,
    ) -> Result<()> {
        let rank = rank as usize;
        let world = self.pending.len();
        output.clear();
        if rank >= world || self.closed[rank] {
            return Err("SEGMENT_PACKET_CLOSED_RANK".into());
        }
        match frame {
            Frame::Offer(packet) => {
                let destination = packet.destination as usize;
                if packet.depth != self.depth
                    || packet.source as usize != rank
                    || destination >= world
                    || destination == rank
                    || self.closed[destination]
                    || packet.descriptors == 0
                    || packet.descriptors > self.shards
                    || packet.descriptors > packet.rows
                    || packet.rows > self.max_rows
                    || packet.sequence != self.next_sequence[rank]
                    || self.sealed[rank]
                    || self.pending[rank].is_some()
                    || self.sending(rank)
                {
                    return Err("SEGMENT_PACKET_INVALID_OFFER".into());
                }
                self.next_sequence[rank] = packet
                    .sequence
                    .checked_add(1)
                    .ok_or("SEGMENT_PACKET_SEQUENCE_OVERFLOW")?;
                self.pending[rank] = Some(packet);
                self.pending_sources |= 1u128 << rank;
            }
            Frame::Ack(packet) | Frame::Received(packet) => {
                if !self.complete(rank, packet, matches!(frame, Frame::Received(_)))? {
                    return Ok(());
                }
            }
            Frame::Seal(depth) => {
                if depth != self.depth
                    || self.sealed[rank]
                    || self.pending[rank].is_some()
                    || self.sending(rank)
                {
                    return Err("SEGMENT_PACKET_INVALID_SEAL".into());
                }
                self.sealed[rank] = true;
                self.sealed_count += 1;
            }
            Frame::Close => {
                if self.endpoint[rank].is_some()
                    || self.pending[rank].is_some()
                    || self.active.iter().flatten().flatten().any(|credit| {
                        credit.packet.source as usize == rank
                            || credit.packet.destination as usize == rank
                    })
                    || self
                        .pending
                        .iter()
                        .flatten()
                        .any(|p| p.destination as usize == rank)
                {
                    return Err("SEGMENT_PACKET_CLOSE_WITH_LIVE_CREDIT".into());
                }
                self.closed[rank] = true;
                self.closed_count += 1;
            }
            _ => return Err("SEGMENT_PACKET_UNEXPECTED_CLIENT_FRAME".into()),
        }
        // Rotate only for traversal so active sources are considered in the
        // same round-robin order without scanning empty ranks on every ACK.
        let begin = self.cursor as u32;
        let mut offers = self.pending_sources.rotate_right(begin);
        while offers != 0 {
            let source = ((begin + offers.trailing_zeros()) % u128::BITS) as usize;
            offers &= offers - 1;
            let packet = self.pending[source].ok_or("SEGMENT_PACKET_MISSING_OFFER")?;
            let destination = packet.destination as usize;
            if self.endpoint[source].is_some() || self.endpoint[destination].is_some() {
                continue;
            }
            let Some(slot) = self.active[destination][..self.receive_slots]
                .iter()
                .position(Option::is_none)
            else {
                continue;
            };
            self.pending[source] = None;
            self.pending_sources &= !(1u128 << source);
            let endpoint = CreditSlot { destination, slot };
            self.endpoint[source] = Some(endpoint);
            self.endpoint[destination] = Some(endpoint);
            self.active[destination][slot] = Some(Active {
                packet,
                expires_at: Instant::now() + self.credit_timeout,
                source_ack: false,
                received: false,
                destination_ack: false,
            });
            self.active_destinations |= 1u128 << destination;
            self.active_count += 1;
            self.peak_active
                .fetch_max(self.active_count, Ordering::Relaxed);
            self.cursor = (source + 1) % world;
            output.push((packet.source, Frame::Grant(packet)));
            output.push((packet.destination, Frame::Grant(packet)));
        }
        if self.sealed_count == world && self.pending_sources == 0 && self.active_destinations == 0
        {
            if self.closed_count != 0 {
                return Err("SEGMENT_PACKET_COMMIT_WITH_CLOSED_RANK".into());
            }
            output.extend((0..world as u32).map(|rank| (rank, Frame::Commit(self.depth))));
            self.depth = self
                .depth
                .checked_add(1)
                .ok_or("SEGMENT_PACKET_DEPTH_OVERFLOW")?;
            self.sealed.fill(false);
            self.sealed_count = 0;
            self.next_sequence.fill(0);
        }
        Ok(())
    }
}

pub(crate) struct PacketClient {
    socket: TcpStream,
    events: mpsc::Receiver<Frame>,
    terminal: Arc<AtomicU32>,
    hub_peak_active: Option<Arc<AtomicU32>>,
    owner_watchdog: Option<Arc<OwnerWaveWatchdog>>,
}

impl PacketClient {
    #[cfg(test)]
    pub fn connect(
        advertised: SocketAddr,
        bind: SocketAddr,
        rank: u32,
        world: u32,
        shards: u32,
        max_rows: u32,
        identity: &[u8; 128],
        job_token: &PacketJobToken,
    ) -> Result<Self> {
        Self::connect_with_receive_slots(
            advertised, bind, rank, world, shards, max_rows, identity, job_token, 1,
        )
    }

    pub fn connect_with_receive_slots(
        advertised: SocketAddr,
        bind: SocketAddr,
        rank: u32,
        world: u32,
        shards: u32,
        max_rows: u32,
        identity: &[u8; 128],
        job_token: &PacketJobToken,
        receive_slots: u32,
    ) -> Result<Self> {
        if !(2..=u128::BITS).contains(&world)
            || rank >= world
            || shards == 0
            || max_rows == 0
            || advertised.port() == 0
            || bind.port() == 0
            || (receive_slots != 1 && receive_slots as usize != CAROUSEL_RECEIVE_SLOTS)
        {
            return Err("SEGMENT_PACKET_CONTROL_CONFIGURATION".into());
        }
        let tag = handshake_tag(identity, job_token, receive_slots);
        let deadline = Instant::now() + DEADLINE;
        let hub_peak_active = if rank == 0 {
            let listener = loop {
                match TcpListener::bind(bind) {
                    Ok(listener) => break listener,
                    Err(error)
                        if error.kind() == ErrorKind::AddrInUse && Instant::now() < deadline =>
                    {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(error) => return Err(format!("SEGMENT_PACKET_BIND {error}")),
                }
            };
            listener.set_nonblocking(true).map_err(|e| e.to_string())?;
            let scheduler = Scheduler::with_receive_slots(world, shards, max_rows, receive_slots);
            let peak = Arc::clone(&scheduler.peak_active);
            thread::Builder::new()
                .name("segment-packet-hub".into())
                .spawn(move || {
                    if let Err(error) = run_hub(listener, scheduler, tag, deadline) {
                        eprintln!("{error}");
                    }
                })
                .map_err(|e| e.to_string())?;
            Some(peak)
        } else {
            None
        };
        let socket = connect_until_ready(advertised, rank, &tag, deadline)?;
        socket
            .set_write_timeout(Some(DEADLINE))
            .map_err(|e| e.to_string())?;
        socket.set_read_timeout(None).map_err(|e| e.to_string())?;
        let owner_watchdog = if receive_slots as usize == CAROUSEL_RECEIVE_SLOTS {
            Some(OwnerWaveWatchdog::start(
                socket.try_clone().map_err(|e| e.to_string())?,
                rank,
            )?)
        } else {
            None
        };
        let mut incoming = socket.try_clone().map_err(|e| e.to_string())?;
        let terminal = Arc::new(AtomicU32::new(0));
        let reader_terminal = Arc::clone(&terminal);
        let (sender, events) = mpsc::sync_channel(4);
        thread::Builder::new()
            .name("segment-packet-events".into())
            .spawn(move || loop {
                match Frame::read(&mut incoming) {
                    Ok(Frame::Abort(code)) => {
                        reader_terminal.store(code, Ordering::Release);
                        let _ = sender.try_send(Frame::Abort(code));
                        break;
                    }
                    Ok(frame) => {
                        if sender.send(frame).is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        reader_terminal.store(REMOTE_FAILURE, Ordering::Release);
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            socket,
            events,
            terminal,
            hub_peak_active,
            owner_watchdog,
        })
    }

    #[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
    pub(crate) fn owner_watchdog(&self) -> Option<Arc<OwnerWaveWatchdog>> {
        self.owner_watchdog.clone()
    }

    pub fn send(&mut self, frame: Frame) -> Result<()> {
        frame.write(&mut self.socket)
    }

    pub fn next(&self, wait: Option<Duration>) -> Option<Frame> {
        let terminal = self.terminal_code();
        if terminal != 0 {
            return Some(Frame::Abort(terminal));
        }
        let frame = if let Some(wait) = wait {
            match self.events.recv_timeout(wait) {
                Ok(frame) => Some(frame),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => Some(Frame::Abort(REMOTE_FAILURE)),
            }
        } else {
            match self.events.try_recv() {
                Ok(frame) => Some(frame),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => Some(Frame::Abort(REMOTE_FAILURE)),
            }
        };
        let terminal = self.terminal_code();
        if terminal != 0 {
            Some(Frame::Abort(terminal))
        } else {
            frame
        }
    }

    pub fn next_after_seal(&self) -> Frame {
        let terminal = self.terminal_code();
        if terminal != 0 {
            return Frame::Abort(terminal);
        }
        // Another rank may still be expanding its local frontier. The hub
        // times out outstanding transfers and the reader reports disconnects.
        let frame = self.events.recv();
        let terminal = self.terminal_code();
        if terminal != 0 {
            return Frame::Abort(terminal);
        }
        frame.unwrap_or(Frame::Abort(REMOTE_FAILURE))
    }

    pub fn terminal_code(&self) -> u32 {
        self.terminal.load(Ordering::Acquire)
    }

    pub fn hub_peak_active_credits(&self) -> Option<u32> {
        self.hub_peak_active
            .as_ref()
            .map(|peak| peak.load(Ordering::Relaxed))
    }

    pub fn terminal_ptr(&self) -> *const AtomicU32 {
        Arc::as_ptr(&self.terminal)
    }
}

impl Drop for PacketClient {
    fn drop(&mut self) {
        let _ = Frame::Close.write(&mut self.socket);
        let _ = self.socket.shutdown(Shutdown::Both);
    }
}

fn run_hub(
    listener: TcpListener,
    scheduler: Scheduler,
    tag: [u8; 32],
    deadline: Instant,
) -> Result<()> {
    struct PendingHello {
        socket: TcpStream,
        bytes: [u8; HELLO_BYTES],
        filled: usize,
    }

    let world = scheduler.closed.len();
    let mut peers = (0..world).map(|_| None).collect::<Vec<Option<TcpStream>>>();
    let pending_capacity = 2 * world;
    let mut pending: VecDeque<PendingHello> = VecDeque::with_capacity(pending_capacity);
    let mut count = 0;
    while count != world {
        if Instant::now() >= deadline {
            return Err("SEGMENT_PACKET_HELLO_TIMEOUT".into());
        }
        for _ in 0..pending.len() {
            let Some(mut hello) = pending.pop_front() else {
                break;
            };
            match hello.socket.read(&mut hello.bytes[hello.filled..]) {
                Ok(0) => continue,
                Ok(bytes) => hello.filled += bytes,
                Err(error)
                    if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) =>
                {
                    pending.push_back(hello);
                    continue;
                }
                Err(_) => continue,
            }
            if hello.filled != HELLO_BYTES {
                pending.push_back(hello);
                continue;
            }
            if hello.bytes[4..] != tag {
                // Reject a previous job before it can occupy a rank slot.
                let _ = Frame::Abort(REMOTE_FAILURE).write(&mut hello.socket);
                continue;
            }
            let rank = u32::from_le_bytes(hello.bytes[..4].try_into().unwrap()) as usize;
            if rank >= peers.len() {
                return Err("SEGMENT_PACKET_INVALID_HELLO".into());
            }
            let previous = &mut peers[rank];
            if let Some(peer) = previous {
                // A retry can replace a disconnected HELLO before Ready, never a live rank.
                if !startup_peer_disconnected(peer) {
                    return Err("SEGMENT_PACKET_INVALID_HELLO".into());
                }
            } else {
                count += 1;
            }
            hello
                .socket
                .set_write_timeout(Some(DEADLINE))
                .map_err(|e| e.to_string())?;
            hello.socket.set_nodelay(true).map_err(|e| e.to_string())?;
            *previous = Some(hello.socket);
            if count == world {
                break;
            }
        }
        if count == world {
            for peer in &mut peers {
                if peer.as_ref().is_some_and(startup_peer_disconnected) {
                    *peer = None;
                    count -= 1;
                }
            }
            if count == world {
                break;
            }
        }

        let mut accepted = false;
        for _ in 0..world {
            match listener.accept() {
                Ok((socket, _)) => {
                    socket.set_nonblocking(true).map_err(|e| e.to_string())?;
                    if pending.len() == pending_capacity {
                        // Every older socket had a chance to finish above.
                        // Unfinished HELLOs must not exhaust the startup slots.
                        pending.pop_front();
                    }
                    pending.push_back(PendingHello {
                        socket,
                        bytes: [0; HELLO_BYTES],
                        filled: 0,
                    });
                    accepted = true;
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(format!("SEGMENT_PACKET_ACCEPT {error}")),
            }
        }
        if !accepted {
            thread::sleep(
                deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(5)),
            );
        }
    }
    if Instant::now() >= deadline {
        return Err("SEGMENT_PACKET_HELLO_TIMEOUT".into());
    }
    drop(pending);
    let peers = peers.into_iter().map(Option::unwrap).collect::<Vec<_>>();
    for peer in &peers {
        peer.set_nonblocking(false).map_err(|e| e.to_string())?;
    }
    hub_loop(&peers, scheduler)
}

fn startup_peer_disconnected(peer: &TcpStream) -> bool {
    match peer.peek(&mut [0u8; 1]) {
        Ok(size) => size == 0,
        Err(error) => startup_disconnected(&error),
    }
}

enum HubInput {
    Frame(u32, Result<Frame>),
    WriteError(usize, String),
}

fn write_hub_frame(peer: &mut TcpStream, frame: Frame) -> Result<()> {
    let deadline = Instant::now() + DEADLINE;
    let encoded = frame.encode();
    let mut bytes = encoded.as_slice();
    // A full write uses the socket's original timeout. Only partial writes
    // or interrupted syscalls need to reduce it to the remaining time.
    let mut adjusted = false;
    loop {
        match peer.write(bytes) {
            Ok(written) if written == bytes.len() => break,
            Ok(0) => return Err("SEGMENT_PACKET_WRITE_ZERO".into()),
            Ok(written) => bytes = &bytes[written..],
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(format!("SEGMENT_PACKET_WRITE {error}")),
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("SEGMENT_PACKET_WRITE_TIMEOUT".into());
        }
        peer.set_write_timeout(Some(remaining))
            .map_err(|e| e.to_string())?;
        adjusted = true;
    }
    if adjusted {
        peer.set_write_timeout(Some(DEADLINE))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn write_peer_output(
    mut write: impl FnMut(Frame) -> Result<()>,
    output: mpsc::Receiver<Frame>,
    terminal: &AtomicU32,
) -> Result<()> {
    for frame in output {
        if terminal.load(Ordering::Acquire) != 0 {
            break;
        }
        write(frame)?;
    }
    let code = terminal.load(Ordering::Acquire);
    if code != 0 {
        write(Frame::Abort(code))?;
    }
    Ok(())
}

fn hub_loop(peers: &[TcpStream], scheduler: Scheduler) -> Result<()> {
    // One pending offer and active endpoint per rank bound metadata storage.
    let (sender, inbox) = mpsc::sync_channel(peers.len() * 3);
    let terminal = Arc::new(AtomicU32::new(0));
    let mut outputs = Vec::with_capacity(peers.len());
    let outcome = (|| {
        for (rank, peer) in peers.iter().enumerate() {
            let mut reader = peer.try_clone().map_err(|e| e.to_string())?;
            let mut writer = peer.try_clone().map_err(|e| e.to_string())?;
            let (output, commands) = mpsc::sync_channel(PEER_OUTPUT_CAPACITY);
            let tx = sender.clone();
            let stop = Arc::clone(&terminal);
            // Each peer owns its socket writes and keeps its own FIFO.
            // Backpressure here cannot pause the coordinator or other peers.
            thread::Builder::new()
                .stack_size(128 * 1024)
                .spawn(move || {
                    let result = write_peer_output(
                        |frame| write_hub_frame(&mut writer, frame),
                        commands,
                        &stop,
                    );
                    if let Err(error) = result {
                        let _ = tx.send(HubInput::WriteError(rank, error));
                    }
                    let _ = writer.shutdown(Shutdown::Both);
                })
                .map_err(|e| e.to_string())?;
            outputs.push(output);
            let tx = sender.clone();
            thread::Builder::new()
                .stack_size(128 * 1024)
                .spawn(move || loop {
                    let frame = Frame::read(&mut reader);
                    let finished =
                        frame.is_err() || matches!(frame, Ok(Frame::Close) | Ok(Frame::Fail(_)));
                    if tx.send(HubInput::Frame(rank as u32, frame)).is_err() || finished {
                        break;
                    }
                })
                .map_err(|e| e.to_string())?;
            enqueue_output(&outputs, rank, Frame::Ready)?;
        }
        drop(sender);
        schedule_hub(&outputs, inbox, scheduler)
    })();
    let code = match &outcome {
        Ok(Some(code)) => *code,
        Err(_) => REMOTE_FAILURE,
        Ok(None) => 0,
    };
    // On failure, each writer discards queued frames and sends Abort when
    // its current write returns. Healthy peers wake even if one is stalled.
    terminal.store(code, Ordering::Release);
    for peer in &peers[outputs.len()..] {
        let _ = peer.shutdown(Shutdown::Both);
    }
    drop(outputs);
    outcome.map(|_| ())
}

fn enqueue_output(outputs: &[mpsc::SyncSender<Frame>], rank: usize, command: Frame) -> Result<()> {
    outputs[rank]
        .try_send(command)
        .map_err(|error| match error {
            mpsc::TrySendError::Full(_) => format!("SEGMENT_PACKET_WRITE_QUEUE_FULL rank={rank}"),
            mpsc::TrySendError::Disconnected(_) => {
                format!("SEGMENT_PACKET_WRITER_DISCONNECTED rank={rank}")
            }
        })
}

fn schedule_hub(
    outputs: &[mpsc::SyncSender<Frame>],
    inbox: mpsc::Receiver<HubInput>,
    mut scheduler: Scheduler,
) -> Result<Option<u32>> {
    let mut commands = Vec::with_capacity(outputs.len());
    while scheduler.closed_count < outputs.len() {
        let event = match scheduler.next_credit_timeout(Instant::now()) {
            Some(remaining) if remaining.is_zero() => {
                return Err("SEGMENT_PACKET_CREDIT_TIMEOUT".into());
            }
            Some(remaining) => inbox.recv_timeout(remaining).map_err(|e| match e {
                mpsc::RecvTimeoutError::Timeout => "SEGMENT_PACKET_CREDIT_TIMEOUT".to_owned(),
                mpsc::RecvTimeoutError::Disconnected => {
                    "SEGMENT_PACKET_HUB_DISCONNECTED".to_owned()
                }
            })?,
            None => inbox
                .recv()
                .map_err(|e| format!("SEGMENT_PACKET_HUB_WAIT {e}"))?,
        };
        let (rank, frame) = match event {
            HubInput::Frame(rank, frame) => (rank, frame?),
            HubInput::WriteError(rank, error) => {
                return Err(format!("SEGMENT_PACKET_PEER_WRITE rank={rank} {error}"))
            }
        };
        if let Frame::Fail(code) = frame {
            return Ok(Some(code));
        }
        scheduler.input_into(rank, frame, &mut commands)?;
        for (dst, command) in commands.drain(..) {
            enqueue_output(outputs, dst as usize, command)?;
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(source: u32, destination: u32, sequence: u64) -> Packet {
        Packet {
            depth: 0,
            sequence,
            source,
            destination,
            descriptors: 1,
            rows: 7,
        }
    }

    #[test]
    fn disjoint_pairs_progress_and_commit_waits_for_publication() {
        let mut hub = Scheduler::new(4, 2, 8);
        let first = packet(0, 1, 0);
        let second = packet(2, 3, 0);
        let reply = hub.input(0, Frame::Offer(first)).unwrap();
        assert_eq!(
            reply,
            vec![(0, Frame::Grant(first)), (1, Frame::Grant(first))]
        );
        let reply = hub.input(2, Frame::Offer(second)).unwrap();
        assert_eq!(
            reply,
            vec![(2, Frame::Grant(second)), (3, Frame::Grant(second))]
        );
        for rank in [1, 3] {
            assert!(hub.input(rank, Frame::Seal(0)).unwrap().is_empty());
        }
        assert!(hub.input(0, Frame::Ack(first)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(first)).unwrap().is_empty());
        assert!(hub.input(0, Frame::Seal(0)).unwrap().is_empty());
        assert!(hub.input(2, Frame::Ack(second)).unwrap().is_empty());
        assert!(hub.input(3, Frame::Ack(second)).unwrap().is_empty());
        let commit = hub.input(2, Frame::Seal(0)).unwrap();
        assert_eq!(
            commit,
            (0..4)
                .map(|rank| (rank, Frame::Commit(0)))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn carousel_drains_wire_under_owner_backpressure_and_commits_after_last_owner() {
        let mut hub = Scheduler::with_receive_slots(4, 2, 8, 3);
        let first = packet(0, 1, 0);
        let second = packet(2, 1, 0);
        let third = packet(3, 1, 0);
        let fourth = packet(0, 1, 1);
        let outgoing = packet(1, 2, 0);
        let grant = |p: Packet| {
            vec![
                (p.source, Frame::Grant(p)),
                (p.destination, Frame::Grant(p)),
            ]
        };

        for p in [first, second, third] {
            assert_eq!(hub.input(p.source, Frame::Offer(p)).unwrap(), grant(p));
            assert!(hub.input(1, Frame::Ack(p)).is_err());
            assert!(hub.input(1, Frame::Received(p)).unwrap().is_empty());
            assert!(hub.input(p.source, Frame::Ack(p)).unwrap().is_empty());
        }
        assert_eq!(hub.active_count, 3);
        assert!(hub.input(0, Frame::Offer(fourth)).unwrap().is_empty());
        assert_eq!(
            hub.input(1, Frame::Offer(outgoing)).unwrap(),
            grant(outgoing)
        );
        assert!(hub.input(2, Frame::Received(outgoing)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(outgoing)).unwrap().is_empty());
        assert!(hub.input(2, Frame::Ack(outgoing)).unwrap().is_empty());
        assert_eq!(hub.peak_active.load(Ordering::Relaxed), 4);
        assert_eq!(hub.input(1, Frame::Ack(first)).unwrap(), grant(fourth));
        assert!(hub.input(1, Frame::Received(first)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(first)).unwrap().is_empty());
        assert!(hub.input(0, Frame::Ack(fourth)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Received(fourth)).unwrap().is_empty());
        assert_eq!(hub.active_count, 3);

        for rank in 0..4 {
            assert!(hub.input(rank, Frame::Seal(0)).unwrap().is_empty());
        }
        assert!(hub.input(1, Frame::Close).is_err());
        assert!(hub.input(1, Frame::Ack(third)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(fourth)).unwrap().is_empty());
        let commit = hub.input(1, Frame::Ack(second)).unwrap();
        assert_eq!(
            commit,
            (0..4)
                .map(|rank| (rank, Frame::Commit(0)))
                .collect::<Vec<_>>()
        );
        assert_eq!(hub.active_count, 0);
    }

    #[test]
    fn pending_rank_order_wraps_across_rank_127() {
        let mut hub = Scheduler::new(128, 2, 8);
        let busy = packet(126, 9, 0);
        hub.input(126, Frame::Offer(busy)).unwrap();
        let low = packet(0, 9, 0);
        let high = packet(127, 9, 0);
        assert!(hub.input(0, Frame::Offer(low)).unwrap().is_empty());
        assert!(hub.input(127, Frame::Offer(high)).unwrap().is_empty());
        hub.input(126, Frame::Ack(busy)).unwrap();
        assert_eq!(
            hub.input(9, Frame::Ack(busy)).unwrap(),
            vec![(127, Frame::Grant(high)), (9, Frame::Grant(high))]
        );
        hub.input(127, Frame::Ack(high)).unwrap();
        assert_eq!(
            hub.input(9, Frame::Ack(high)).unwrap(),
            vec![(0, Frame::Grant(low)), (9, Frame::Grant(low))]
        );
    }

    #[test]
    fn source_can_queue_its_next_packet_before_receiver_publication() {
        let mut hub = Scheduler::new(3, 2, 8);
        let first = packet(0, 1, 0);
        let second = packet(0, 2, 1);
        hub.input(0, Frame::Offer(first)).unwrap();
        assert!(hub.input(0, Frame::Ack(first)).unwrap().is_empty());
        assert_eq!(
            hub.input(0, Frame::Offer(second)).unwrap(),
            vec![(0, Frame::Grant(second)), (2, Frame::Grant(second))]
        );
        assert!(hub.input(0, Frame::Seal(0)).is_err());
        assert!(hub.input(0, Frame::Ack(second)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(first)).unwrap().is_empty());
        for rank in 0..3 {
            assert!(hub.input(rank, Frame::Seal(0)).unwrap().is_empty());
        }
        assert_eq!(hub.input(2, Frame::Ack(second)).unwrap().len(), 3);
        assert_eq!(hub.active_destinations.count_ones(), 0);
        assert_eq!(hub.peak_active.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn sender_can_receive_while_its_previous_destination_publishes() {
        let mut hub = Scheduler::new(3, 2, 8);
        let first = packet(0, 1, 0);
        let incoming = packet(2, 0, 0);
        hub.input(0, Frame::Offer(first)).unwrap();
        assert!(hub.input(2, Frame::Offer(incoming)).unwrap().is_empty());
        assert_eq!(
            hub.input(0, Frame::Ack(first)).unwrap(),
            vec![(2, Frame::Grant(incoming)), (0, Frame::Grant(incoming))]
        );
        assert!(hub.input(0, Frame::Ack(first)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(first)).unwrap().is_empty());
        assert!(hub.input(2, Frame::Ack(incoming)).unwrap().is_empty());
        assert!(hub.input(0, Frame::Ack(incoming)).unwrap().is_empty());
        for rank in 0..2 {
            assert!(hub.input(rank, Frame::Seal(0)).unwrap().is_empty());
        }
        assert_eq!(hub.input(2, Frame::Seal(0)).unwrap().len(), 3);
    }

    #[test]
    fn unrelated_traffic_cannot_extend_a_granted_credit() {
        let mut hub = Scheduler::new(4, 2, 8);
        let stalled = packet(0, 1, 0);
        hub.input(0, Frame::Offer(stalled)).unwrap();
        let expiry = hub.active[1][0].unwrap().expires_at;
        hub.input(0, Frame::Ack(stalled)).unwrap();
        hub.input(0, Frame::Ack(stalled)).unwrap();
        let other = packet(2, 3, 0);
        hub.input(2, Frame::Offer(other)).unwrap();
        hub.input(2, Frame::Ack(other)).unwrap();
        hub.input(3, Frame::Ack(other)).unwrap();
        assert_eq!(hub.next_credit_timeout(expiry), Some(Duration::ZERO));
        hub.input(1, Frame::Ack(stalled)).unwrap();
        assert_eq!(hub.next_credit_timeout(expiry), None);
    }

    #[test]
    fn cyclic_offers_make_progress_with_duplicate_and_delayed_ack() {
        let mut hub = Scheduler::new(3, 2, 8);
        let ab = packet(0, 1, 0);
        let bc = packet(1, 2, 0);
        let ca = packet(2, 0, 0);
        assert_eq!(
            hub.input(0, Frame::Offer(ab)).unwrap()[1],
            (1, Frame::Grant(ab))
        );
        assert!(hub.input(1, Frame::Offer(bc)).unwrap().is_empty());
        assert!(hub.input(2, Frame::Offer(ca)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(ab)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(ab)).unwrap().is_empty());
        assert_eq!(
            hub.input(0, Frame::Ack(ab)).unwrap()[0],
            (1, Frame::Grant(bc))
        );
        assert!(hub.input(1, Frame::Ack(ab)).unwrap().is_empty());
        assert!(hub.input(2, Frame::Ack(ab)).is_err());
        assert!(hub
            .input(
                1,
                Frame::Ack(Packet {
                    rows: ab.rows + 1,
                    ..ab
                })
            )
            .is_err());
        assert!(hub.input(2, Frame::Ack(bc)).unwrap().is_empty());
        assert_eq!(
            hub.input(1, Frame::Ack(bc)).unwrap()[0],
            (2, Frame::Grant(ca))
        );
        assert!(hub.input(0, Frame::Ack(ca)).unwrap().is_empty());
        assert!(hub.input(2, Frame::Ack(ca)).unwrap().is_empty());
        for rank in 0..2 {
            assert!(hub.input(rank, Frame::Seal(0)).unwrap().is_empty());
        }
        assert_eq!(hub.input(2, Frame::Seal(0)).unwrap().len(), 3);
        let next = Packet { depth: 1, ..ca };
        assert_eq!(hub.input(2, Frame::Offer(next)).unwrap().len(), 2);
        assert!(hub.input(0, Frame::Ack(ca)).unwrap().is_empty());
        assert!(hub.input(2, Frame::Ack(next)).unwrap().is_empty());
        assert!(hub.input(0, Frame::Ack(next)).unwrap().is_empty());
    }

    #[test]
    fn old_ack_cannot_consume_a_newer_credit() {
        let mut hub = Scheduler::new(3, 2, 8);
        let first = packet(0, 1, 0);
        let second = packet(0, 2, 1);
        let current = packet(0, 1, 2);
        for released in [first, second] {
            assert_eq!(hub.input(0, Frame::Offer(released)).unwrap().len(), 2);
            assert!(hub.input(0, Frame::Ack(released)).unwrap().is_empty());
            assert!(hub
                .input(released.destination, Frame::Ack(released))
                .unwrap()
                .is_empty());
        }
        assert_eq!(hub.input(0, Frame::Offer(current)).unwrap().len(), 2);
        assert!(hub.input(1, Frame::Ack(first)).unwrap().is_empty());
        assert!(hub.input(0, Frame::Ack(current)).unwrap().is_empty());
        assert!(hub.input(1, Frame::Ack(current)).unwrap().is_empty());
    }

    #[test]
    fn invalid_offers_and_stale_epoch_frames_are_rejected() {
        let mut hub = Scheduler::new(2, 2, 8);
        assert!(hub.input(1, Frame::Offer(packet(0, 1, 0))).is_err());
        assert!(hub.input(0, Frame::Offer(packet(0, 1, 1))).is_err());
        assert!(hub
            .input(
                0,
                Frame::Offer(Packet {
                    rows: 9,
                    ..packet(0, 1, 0)
                })
            )
            .is_err());
        assert!(hub.input(0, Frame::Seal(0)).unwrap().is_empty());
        assert!(hub.input(0, Frame::Offer(packet(0, 1, 0))).is_err());
        assert_eq!(hub.input(1, Frame::Seal(0)).unwrap().len(), 2);
        assert!(hub.input(0, Frame::Offer(packet(0, 1, 0))).is_err());
    }

    #[test]
    fn wire_roundtrip_checks_reserved_bytes() {
        let message = Frame::Grant(Packet {
            depth: u64::MAX,
            ..packet(0, 1, u64::MAX)
        });
        let mut encoded = message.encode();
        assert_eq!(Frame::decode(encoded).unwrap(), message);
        encoded[5] = 1;
        assert!(Frame::decode(encoded).is_err());
        for message in [
            Frame::Ready,
            Frame::Received(packet(0, 1, 0)),
            Frame::Seal(9),
            Frame::Commit(9),
            Frame::Fail(10),
            Frame::Abort(10),
            Frame::Close,
        ] {
            assert_eq!(Frame::decode(message.encode()).unwrap(), message);
        }
    }

    #[test]
    fn packet_header_encodes_multiple_shards_and_verifies_total_rows() {
        let original = [
            ShardRun { shard: 1, rows: 3 },
            ShardRun { shard: 2, rows: 2 },
        ];
        let mut runs = Vec::new();
        let mut header = Vec::new();
        encode_runs(&original, &mut header);
        decode_runs(&header, 5, 6, &mut runs).unwrap();
        assert_eq!(runs, original);
        assert!(decode_runs(&header, 6, 6, &mut runs).is_err());
        header[SHARD_DESCRIPTOR_BYTES] = 1;
        assert!(decode_runs(&header, 5, 6, &mut runs).is_err());
    }

    #[test]
    fn blocked_writer_cannot_pause_disjoint_grants() {
        let (input, inbox) = mpsc::sync_channel(12);
        let (outputs, writes): (Vec<_>, Vec<_>) = (0..4)
            .map(|_| mpsc::sync_channel(PEER_OUTPUT_CAPACITY))
            .unzip();
        let [rank_0, rank_1, rank_2, rank_3] = <[_; 4]>::try_from(writes).unwrap();
        let stalled = packet(0, 1, 0);
        let (entered, blocked) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let terminal = Arc::new(AtomicU32::new(0));
        let writer_terminal = Arc::clone(&terminal);
        let writer = thread::spawn(move || {
            write_peer_output(
                |frame| {
                    assert_eq!(frame, Frame::Grant(stalled));
                    entered.send(()).unwrap();
                    gate.recv_timeout(Duration::from_secs(5))
                        .map_err(|error| error.to_string())
                },
                rank_0,
                &writer_terminal,
            )
        });
        let hub = thread::spawn(move || schedule_hub(&outputs, inbox, Scheduler::new(4, 2, 8)));
        let send = |rank, frame| input.send(HubInput::Frame(rank, Ok(frame))).unwrap();
        let wait = Duration::from_secs(2);
        send(0, Frame::Offer(stalled));
        blocked.recv_timeout(wait).unwrap();
        assert_eq!(rank_1.recv_timeout(wait).unwrap(), Frame::Grant(stalled));
        for healthy in [packet(2, 3, 0), packet(2, 3, 1)] {
            send(2, Frame::Offer(healthy));
            assert_eq!(rank_2.recv_timeout(wait).unwrap(), Frame::Grant(healthy));
            assert_eq!(rank_3.recv_timeout(wait).unwrap(), Frame::Grant(healthy));
            send(2, Frame::Ack(healthy));
            send(3, Frame::Ack(healthy));
        }
        send(3, Frame::Fail(PACKET_DEADLINE));
        assert_eq!(hub.join().unwrap().unwrap(), Some(PACKET_DEADLINE));
        release.send(()).unwrap();
        writer.join().unwrap().unwrap();
    }

    #[test]
    fn tcp_rendezvous_delivers_grant_then_commit_and_propagates_abort() {
        let reserve = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reserve.local_addr().unwrap();
        drop(reserve);
        let starter = thread::spawn(move || {
            PacketClient::connect(address, address, 0, 2, 2, 8, &[5; 128], &[2; 16]).unwrap()
        });
        let mut receiver =
            PacketClient::connect(address, address, 1, 2, 2, 8, &[5; 128], &[2; 16]).unwrap();
        let mut sender = starter.join().unwrap();
        let p = packet(0, 1, 0);
        sender.send(Frame::Offer(p)).unwrap();
        assert_eq!(sender.next(Some(DEADLINE)), Some(Frame::Grant(p)));
        assert_eq!(receiver.next(Some(DEADLINE)), Some(Frame::Grant(p)));
        sender.send(Frame::Ack(p)).unwrap();
        receiver.send(Frame::Seal(0)).unwrap();
        receiver.send(Frame::Ack(p)).unwrap();
        sender.send(Frame::Ack(p)).unwrap();
        sender.send(Frame::Seal(0)).unwrap();
        assert_eq!(sender.next_after_seal(), Frame::Commit(0));
        assert_eq!(receiver.next_after_seal(), Frame::Commit(0));
        sender.send(Frame::Seal(1)).unwrap();
        receiver.send(Frame::Fail(PACKET_DEADLINE)).unwrap();
        assert_eq!(sender.next_after_seal(), Frame::Abort(PACKET_DEADLINE));
        assert_eq!(
            receiver.next(Some(DEADLINE)),
            Some(Frame::Abort(PACKET_DEADLINE))
        );
        assert_eq!(sender.terminal_code(), PACKET_DEADLINE);
    }

    #[test]
    fn tcp_commit_waits_for_older_receive_after_newer_credit_returns() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let tag = handshake_tag(&[5; 128], &[6; 16], 1);
        let hub = thread::spawn(move || {
            run_hub(
                listener,
                Scheduler::new(3, 2, 8),
                tag,
                Instant::now() + Duration::from_secs(3),
            )
        });
        let mut peers = (0..3)
            .map(|rank| {
                let mut socket = TcpStream::connect(address).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                write_hello(&mut socket, rank, &tag).unwrap();
                socket
            })
            .collect::<Vec<_>>();
        for peer in &mut peers {
            assert_eq!(Frame::read(peer).unwrap(), Frame::Ready);
        }
        let older = packet(0, 1, 0);
        let newer = packet(0, 2, 1);
        for transfer in [older, newer] {
            Frame::Offer(transfer).write(&mut peers[0]).unwrap();
            assert_eq!(Frame::read(&mut peers[0]).unwrap(), Frame::Grant(transfer));
            assert_eq!(
                Frame::read(&mut peers[transfer.destination as usize]).unwrap(),
                Frame::Grant(transfer)
            );
            Frame::Ack(transfer).write(&mut peers[0]).unwrap();
        }
        for peer in &mut peers {
            Frame::Seal(0).write(peer).unwrap();
        }
        Frame::Ack(newer).write(&mut peers[2]).unwrap();
        peers[0]
            .set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(Frame::read(&mut peers[0]).is_err());
        peers[0]
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();

        Frame::Ack(older).write(&mut peers[1]).unwrap();
        for peer in &mut peers {
            assert_eq!(Frame::read(peer).unwrap(), Frame::Commit(0));
            Frame::Close.write(peer).unwrap();
        }
        hub.join().unwrap().unwrap();
    }

    #[test]
    fn lost_receiver_aborts_generation_with_credit_in_flight() {
        let reserve = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reserve.local_addr().unwrap();
        drop(reserve);
        let starter = thread::spawn(move || {
            PacketClient::connect(address, address, 0, 2, 2, 8, &[5; 128], &[3; 16]).unwrap()
        });
        let receiver =
            PacketClient::connect(address, address, 1, 2, 2, 8, &[5; 128], &[3; 16]).unwrap();
        let mut sender = starter.join().unwrap();
        let outstanding = packet(0, 1, 0);
        sender.send(Frame::Offer(outstanding)).unwrap();
        assert_eq!(sender.next(Some(DEADLINE)), Some(Frame::Grant(outstanding)));
        assert_eq!(
            receiver.next(Some(DEADLINE)),
            Some(Frame::Grant(outstanding))
        );
        sender.send(Frame::Ack(outstanding)).unwrap();
        receiver.socket.shutdown(Shutdown::Both).unwrap();
        assert_eq!(
            sender.next(Some(DEADLINE)),
            Some(Frame::Abort(REMOTE_FAILURE))
        );
        assert_eq!(sender.terminal_code(), REMOTE_FAILURE);
    }

    #[test]
    fn disconnected_rank_aborts_wait_after_seal_without_an_active_credit() {
        let reserve = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reserve.local_addr().unwrap();
        drop(reserve);
        let starter = thread::spawn(move || {
            PacketClient::connect(address, address, 0, 2, 2, 8, &[5; 128], &[4; 16]).unwrap()
        });
        let other =
            PacketClient::connect(address, address, 1, 2, 2, 8, &[5; 128], &[4; 16]).unwrap();
        let mut sealed = starter.join().unwrap();
        sealed.send(Frame::Seal(0)).unwrap();
        let (result, observed) = mpsc::channel();
        let waiter = thread::spawn(move || result.send(sealed.next_after_seal()).unwrap());
        other.socket.shutdown(Shutdown::Both).unwrap();
        assert_eq!(
            observed.recv_timeout(Duration::from_secs(5)).unwrap(),
            Frame::Abort(REMOTE_FAILURE)
        );
        waiter.join().unwrap();
    }

    #[test]
    fn stalled_owner_wave_aborts_sealed_peer_without_an_active_credit() {
        let reserve = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reserve.local_addr().unwrap();
        drop(reserve);
        let connect = move |rank| {
            PacketClient::connect_with_receive_slots(
                address, address, rank, 2, 2, 8, &[5; 128], &[8; 16], 3,
            )
            .unwrap()
        };
        let starter = thread::spawn(move || connect(0));
        let owner = connect(1);
        let mut sealed = starter.join().unwrap();
        let watchdog = owner.owner_watchdog().unwrap();

        // Completing a pass disarms the lease; an idle peer can seal normally.
        drop(watchdog.lease_for(Duration::from_millis(20)));
        sealed.send(Frame::Seal(0)).unwrap();
        assert_eq!(sealed.next(Some(Duration::from_millis(250))), None);
        assert_eq!(sealed.hub_peak_active_credits(), Some(0));

        // Keep rank 1 connected, never issue an Offer and never drop its owner lease.
        let _stalled_owner = watchdog.lease_for(Duration::from_millis(20));
        assert_eq!(
            sealed.next(Some(Duration::from_secs(3))),
            Some(Frame::Abort(REMOTE_FAILURE))
        );
        assert_eq!(
            owner.next(Some(Duration::from_secs(3))),
            Some(Frame::Abort(REMOTE_FAILURE))
        );
    }

    #[test]
    fn compact_job_tokens_require_exactly_128_hex_bits() {
        assert_eq!(
            parse_job_token(Some("0123456789abcdefFEDCBA9876543210")).unwrap(),
            [
                0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
                0x32, 0x10
            ]
        );
        assert!(parse_job_token(None).is_err());
        assert!(parse_job_token(Some("abc")).is_err());
        assert!(parse_job_token(Some("0123456789abcdef0123456789abcdeg")).is_err());
    }

    #[test]
    fn unreturned_credit_aborts_both_tcp_peers() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let tag = handshake_tag(&[9; 128], &[4; 16], 1);
        let mut scheduler = Scheduler::new(2, 2, 8);
        scheduler.credit_timeout = Duration::from_millis(200);
        let hub = thread::spawn(move || {
            run_hub(
                listener,
                scheduler,
                tag,
                Instant::now() + Duration::from_secs(3),
            )
        });
        let hello = |rank| {
            let mut socket = TcpStream::connect(address).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            write_hello(&mut socket, rank, &tag).unwrap();
            socket
        };
        let mut source = hello(0);
        let mut destination = hello(1);
        assert_eq!(Frame::read(&mut source).unwrap(), Frame::Ready);
        assert_eq!(Frame::read(&mut destination).unwrap(), Frame::Ready);
        let stalled = packet(0, 1, 0);
        Frame::Offer(stalled).write(&mut source).unwrap();
        assert_eq!(Frame::read(&mut source).unwrap(), Frame::Grant(stalled));
        assert_eq!(
            Frame::read(&mut destination).unwrap(),
            Frame::Grant(stalled)
        );
        assert_eq!(
            Frame::read(&mut source).unwrap(),
            Frame::Abort(REMOTE_FAILURE)
        );
        assert_eq!(
            Frame::read(&mut destination).unwrap(),
            Frame::Abort(REMOTE_FAILURE)
        );
        assert_eq!(
            hub.join().unwrap().unwrap_err(),
            "SEGMENT_PACKET_CREDIT_TIMEOUT"
        );
    }

    #[test]
    fn partial_hello_does_not_block_ready_ranks() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let tag = handshake_tag(&[9; 128], &[4; 16], 1);
        // Queue a connection ahead of both ranks and keep it open.
        let mut stalled = TcpStream::connect(address).unwrap();
        stalled.write_all(&[0]).unwrap();
        let hub = thread::spawn(move || {
            run_hub(
                listener,
                Scheduler::new(2, 2, 8),
                tag,
                Instant::now() + Duration::from_secs(5),
            )
        });
        let mut healthy = (0..2)
            .map(|rank| {
                let mut socket = TcpStream::connect(address).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                write_hello(&mut socket, rank, &tag).unwrap();
                socket
            })
            .collect::<Vec<_>>();
        for peer in &mut healthy {
            assert_eq!(Frame::read(peer).unwrap(), Frame::Ready);
            Frame::Close.write(peer).unwrap();
        }
        hub.join().unwrap().unwrap();
        drop(stalled);
    }

    #[test]
    fn tcp_client_retries_when_a_forwarder_accepts_before_the_hub_binds() {
        let relay = TcpListener::bind("127.0.0.1:0").unwrap();
        let advertised = relay.local_addr().unwrap();
        let tag = handshake_tag(&[9; 128], &[4; 16], 1);
        let timeout = Duration::from_secs(3);
        let deadline = Instant::now() + Duration::from_secs(5);
        let (closed_tx, closed_rx) = mpsc::channel();
        let (origin_tx, origin_rx) = mpsc::channel();

        let forwarder = thread::spawn(move || {
            let (mut early, _) = relay.accept().unwrap();
            early.set_read_timeout(Some(timeout)).unwrap();
            let mut hello = [0u8; HELLO_BYTES];
            early.read_exact(&mut hello).unwrap();
            drop(early);
            closed_tx.send(()).unwrap();

            let origin = origin_rx.recv_timeout(timeout).unwrap();
            let (mut client, _) = relay.accept().unwrap();
            client.set_read_timeout(Some(timeout)).unwrap();
            client.read_exact(&mut hello).unwrap();
            let mut backend = TcpStream::connect(origin).unwrap();
            backend.set_read_timeout(Some(timeout)).unwrap();
            backend.write_all(&hello).unwrap();
            let ready = Frame::read(&mut backend).unwrap();
            assert_eq!(ready, Frame::Ready);
            ready.write(&mut client).unwrap();
            let close = Frame::read(&mut client).unwrap();
            assert_eq!(close, Frame::Close);
            close.write(&mut backend).unwrap();
        });
        let remote = thread::spawn(move || {
            let mut socket = connect_until_ready(advertised, 1, &tag, deadline).unwrap();
            Frame::Close.write(&mut socket).unwrap();
        });

        // The first TCP dial worked and its HELLO arrived at the forwarder before any hub existed.
        closed_rx.recv_timeout(timeout).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let hub = thread::spawn(move || run_hub(listener, Scheduler::new(2, 1, 8), tag, deadline));
        origin_tx.send(origin).unwrap();

        let mut root = connect_until_ready(origin, 0, &tag, deadline).unwrap();
        Frame::Close.write(&mut root).unwrap();
        remote.join().unwrap();
        forwarder.join().unwrap();
        hub.join().unwrap().unwrap();
    }

    #[test]
    fn tcp_hub_rejects_a_previous_job_with_the_same_nccl_id() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let nccl_id = [9; 128];
        let previous = handshake_tag(&nccl_id, &[3; 16], 1);
        let current = handshake_tag(&nccl_id, &[4; 16], 1);
        let hub = thread::spawn(move || {
            run_hub(
                listener,
                Scheduler::new(2, 2, 8),
                current,
                Instant::now() + Duration::from_secs(3),
            )
        });
        let hello = |rank, tag| {
            let mut socket = TcpStream::connect(address).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            write_hello(&mut socket, rank, &tag).unwrap();
            socket
        };
        let mut stale = hello(0, previous);
        assert_eq!(
            Frame::read(&mut stale).unwrap(),
            Frame::Abort(REMOTE_FAILURE)
        );
        let mut fresh_0 = hello(0, current);
        let mut fresh_1 = hello(1, current);
        assert_eq!(Frame::read(&mut fresh_0).unwrap(), Frame::Ready);
        assert_eq!(Frame::read(&mut fresh_1).unwrap(), Frame::Ready);
        Frame::Close.write(&mut fresh_0).unwrap();
        Frame::Close.write(&mut fresh_1).unwrap();
        hub.join().unwrap().unwrap();
    }
}
