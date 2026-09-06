//! Fixed-size native control-plane frame codec; not yet connected to the GPU dispatcher.
use mgbfs_core::Result;
use std::io::{Read, Write};
pub const FRAME_BYTES: usize = 64;
pub const NO_SLOT: u64 = u64::MAX;
/// One fixed pending frame. Full capacity is an explicit error, never an
/// allocation or overwrite. Caller supplies a nonblocking stream.
pub struct FrameWriter {
    bytes: [u8; FRAME_BYTES],
    written: usize,
    world: u32,
    pending: bool,
    poisoned: bool,
}
impl FrameWriter {
    pub fn new(world: u32) -> Result<Self> {
        if world == 0 {
            return Err("CONTROL_RANK".into());
        }
        Ok(Self {
            bytes: [0; FRAME_BYTES],
            written: 0,
            world,
            pending: false,
            poisoned: false,
        })
    }
    pub fn enqueue(&mut self, frame: ControlFrame) -> Result<()> {
        if self.poisoned {
            return Err("CONTROL_WRITER_POISONED".into());
        }
        if self.pending {
            return Err("CONTROL_SEND_CAPACITY".into());
        }
        self.bytes = frame.encode(self.world)?;
        self.written = 0;
        self.pending = true;
        Ok(())
    }
    /// True means no bytes remain locally; it is NOT a peer/GPU acknowledgement.
    pub fn poll(&mut self, stream: &mut impl Write) -> Result<bool> {
        if self.poisoned {
            return Err("CONTROL_WRITER_POISONED".into());
        }
        if !self.pending {
            return Ok(true);
        }
        while self.written < FRAME_BYTES {
            match stream.write(&self.bytes[self.written..]) {
                Ok(0) => {
                    self.poisoned = true;
                    return Err("CONTROL_WRITE_ZERO".into());
                }
                Ok(n) => self.written += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(false)
                }
                Err(e) => {
                    self.poisoned = true;
                    return Err(format!("CONTROL_WRITE: {e}"));
                }
            }
        }
        self.pending = false;
        Ok(true)
    }
}
/// One preallocated receive frame per peer. The caller must supply a
/// nonblocking stream; poll consumes at most one frame, never the next one.
pub struct FrameReader {
    bytes: [u8; FRAME_BYTES],
    filled: usize,
    world: u32,
    poisoned: bool,
}
impl FrameReader {
    pub fn new(world: u32) -> Result<Self> {
        if world == 0 {
            return Err("CONTROL_RANK".into());
        }
        Ok(Self {
            bytes: [0; FRAME_BYTES],
            filled: 0,
            world,
            poisoned: false,
        })
    }
    pub fn poll(&mut self, stream: &mut impl Read) -> Result<Option<ControlFrame>> {
        if self.poisoned {
            return Err("CONTROL_READER_POISONED".into());
        }
        while self.filled < FRAME_BYTES {
            match stream.read(&mut self.bytes[self.filled..]) {
                Ok(0) => {
                    self.poisoned = true;
                    return Err("CONTROL_EOF".into());
                }
                Ok(n) => self.filled += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(None)
                }
                Err(e) => {
                    self.poisoned = true;
                    return Err(format!("CONTROL_READ: {e}"));
                }
            }
        }
        match ControlFrame::decode(&self.bytes, self.world) {
            Ok(frame) => {
                self.filled = 0;
                Ok(Some(frame))
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Action {
    Ready = 1,
    Begin = 2,
    Complete = 3,
    SourceClosed = 4,
    Fatal = 5,
    Finalize = 6,
    Consumed = 7,
    Finalized = 8,
    Publish = 9,
    OfferBytes = 10,
    TicketBytes = 11,
    Admitted = 12,
    Launch = 13,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Plane {
    None = 0,
    Candidate = 1,
    Request = 2,
    Response = 3,
    Receipt = 4,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlFrame {
    pub action: Action,
    pub rank: u32,
    pub depth: u64,
    pub epoch: u64,
    pub slot: u64,
    pub plane: Plane,
    pub fatal_code: u32,
    pub source_rank: u32,
    pub destination_rank: u32,
    pub payload_bytes: u64,
}
impl ControlFrame {
    fn validate(&self, world: u32) -> Result<()> {
        if world == 0 || self.rank >= world {
            return Err("CONTROL_RANK".into());
        }
        let admission = matches!(
            self.action,
            Action::OfferBytes | Action::TicketBytes | Action::Admitted | Action::Launch
        );
        let source_bearing = admission || self.action == Action::Begin;
        if (source_bearing && self.source_rank >= world)
            || (!source_bearing && self.source_rank != 0)
        {
            return Err("CONTROL_SOURCE_RANK".into());
        }
        if (!admission && (self.destination_rank != 0 || self.payload_bytes != 0))
            || (admission && self.destination_rank >= world)
        {
            return Err("CONTROL_PAYLOAD_FIELDS".into());
        }
        let valid = match self.action {
            Action::Ready => {
                self.slot != NO_SLOT
                    && self.epoch == 0
                    && self.plane != Plane::None
                    && self.fatal_code == 0
            }
            Action::Begin => self.rank == 0 && self.plane != Plane::None && self.fatal_code == 0,
            Action::Complete | Action::Consumed => {
                self.slot == NO_SLOT && self.plane != Plane::None && self.fatal_code == 0
            }
            Action::SourceClosed => {
                self.slot == NO_SLOT
                    && self.epoch == 0
                    && self.plane == Plane::None
                    && self.fatal_code == 0
            }
            Action::Fatal => {
                self.slot == NO_SLOT && self.plane == Plane::None && self.fatal_code != 0
            }
            Action::Finalize | Action::Publish => {
                self.rank == 0
                    && self.slot == NO_SLOT
                    && self.plane == Plane::None
                    && self.fatal_code == 0
            }
            Action::Finalized => {
                self.slot == NO_SLOT && self.plane == Plane::None && self.fatal_code == 0
            }
            Action::OfferBytes | Action::TicketBytes | Action::Admitted | Action::Launch => {
                self.slot != NO_SLOT
                    && self.plane != Plane::None
                    && self.fatal_code == 0
                    && match self.action {
                        Action::OfferBytes => self.rank == self.source_rank,
                        Action::TicketBytes => self.rank == 0,
                        Action::Admitted => self.rank == self.destination_rank,
                        Action::Launch => {
                            self.rank == 0 && self.destination_rank == 0 && self.payload_bytes == 0
                        }
                        _ => unreachable!(),
                    }
            }
        };
        if !valid {
            return Err("CONTROL_FIELDS".into());
        }
        Ok(())
    }
    pub fn encode(&self, world: u32) -> Result<[u8; FRAME_BYTES]> {
        self.validate(world)?;
        let mut bytes = [0; FRAME_BYTES];
        bytes[..8].copy_from_slice(b"MGBCTRL1");
        bytes[8..10].copy_from_slice(&3u16.to_le_bytes());
        bytes[10..12].copy_from_slice(&(self.action as u16).to_le_bytes());
        bytes[12..16].copy_from_slice(&self.rank.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.depth.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.epoch.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.slot.to_le_bytes());
        bytes[40..44].copy_from_slice(&(self.plane as u32).to_le_bytes());
        bytes[44..48].copy_from_slice(&self.fatal_code.to_le_bytes());
        bytes[48..52].copy_from_slice(&self.source_rank.to_le_bytes());
        bytes[52..56].copy_from_slice(&self.destination_rank.to_le_bytes());
        bytes[56..64].copy_from_slice(&self.payload_bytes.to_le_bytes());
        Ok(bytes)
    }
    pub fn decode(bytes: &[u8], world: u32) -> Result<Self> {
        if bytes.len() != FRAME_BYTES || &bytes[..8] != b"MGBCTRL1" || bytes[8..10] != [3, 0] {
            return Err("CONTROL_HEADER".into());
        }
        let action = match u16::from_le_bytes(bytes[10..12].try_into().unwrap()) {
            1 => Action::Ready,
            2 => Action::Begin,
            3 => Action::Complete,
            4 => Action::SourceClosed,
            5 => Action::Fatal,
            6 => Action::Finalize,
            7 => Action::Consumed,
            8 => Action::Finalized,
            9 => Action::Publish,
            10 => Action::OfferBytes,
            11 => Action::TicketBytes,
            12 => Action::Admitted,
            13 => Action::Launch,
            _ => return Err("CONTROL_ACTION".into()),
        };
        let plane = match u32::from_le_bytes(bytes[40..44].try_into().unwrap()) {
            0 => Plane::None,
            1 => Plane::Candidate,
            2 => Plane::Request,
            3 => Plane::Response,
            4 => Plane::Receipt,
            _ => return Err("CONTROL_PLANE".into()),
        };
        let frame = Self {
            action,
            plane,
            rank: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            depth: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            epoch: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            slot: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            source_rank: u32::from_le_bytes(bytes[48..52].try_into().unwrap()),
            fatal_code: u32::from_le_bytes(bytes[44..48].try_into().unwrap()),
            destination_rank: u32::from_le_bytes(bytes[52..56].try_into().unwrap()),
            payload_bytes: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
        };
        frame.validate(world)?;
        Ok(frame)
    }
    /// The connection owner supplies timeouts and must discard the connection
    /// after any I/O error: a partial frame cannot safely be retried in place.
    pub fn write_to(&self, stream: &mut impl Write, world: u32) -> Result<()> {
        stream
            .write_all(&self.encode(world)?)
            .map_err(|e| format!("CONTROL_WRITE: {e}"))
    }
    pub fn read_from(stream: &mut impl Read, world: u32) -> Result<Self> {
        let mut bytes = [0; FRAME_BYTES];
        stream
            .read_exact(&mut bytes)
            .map_err(|e| format!("CONTROL_READ: {e}"))?;
        Self::decode(&bytes, world)
    }
}
