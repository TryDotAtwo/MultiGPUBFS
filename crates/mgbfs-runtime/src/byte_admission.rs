//! Coordinator-side byte messages; dispatcher integration is separate.
use crate::{
    control_wire::{Action, ControlFrame},
    scatter_admission::{ScatterAdmission, TicketKey},
};
use mgbfs_core::Result;
pub struct ByteAdmission {
    guard: ScatterAdmission,
    key: Option<TicketKey>,
    sizes: Vec<u64>,
    offered: Vec<bool>,
    capacity: u64,
    total: u64,
    published: bool,
    failed: bool,
}
impl ByteAdmission {
    pub fn new(world: u32) -> Result<Self> {
        let guard = ScatterAdmission::new(world)?;
        let mut sizes = Vec::new();
        let mut offered = Vec::new();
        sizes
            .try_reserve_exact(world as usize)
            .map_err(|_| "BYTE_ADMISSION_CAPACITY")?;
        offered
            .try_reserve_exact(world as usize)
            .map_err(|_| "BYTE_ADMISSION_CAPACITY")?;
        sizes.resize(world as usize, 0);
        offered.resize(world as usize, false);
        Ok(Self {
            guard,
            key: None,
            sizes,
            offered,
            capacity: 0,
            total: 0,
            published: false,
            failed: false,
        })
    }
    fn apply<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.failed {
            return Err("BYTE_ADMISSION_FAILED".into());
        }
        let result = f(self);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn frame(key: TicketKey, action: Action, dst: u32, bytes: u64) -> ControlFrame {
        ControlFrame {
            action,
            rank: 0,
            depth: key.depth,
            epoch: key.epoch,
            slot: key.generation,
            plane: key.plane,
            source_rank: key.source,
            destination_rank: dst,
            payload_bytes: bytes,
            fatal_code: 0,
        }
    }
    fn matches(&self, frame: ControlFrame) -> Result<TicketKey> {
        frame.encode(self.sizes.len() as u32)?;
        let key = TicketKey {
            depth: frame.depth,
            epoch: frame.epoch,
            source: frame.source_rank,
            plane: frame.plane,
            generation: frame.slot,
        };
        if self.key != Some(key) {
            return Err("BYTE_ADMISSION_STALE".into());
        }
        Ok(key)
    }
    pub fn begin(&mut self, key: TicketKey, capacity: u64) -> Result<()> {
        self.apply(|s| {
            if s.key.is_some() {
                return Err("BYTE_ADMISSION_BUSY".into());
            }
            Self::frame(key, Action::TicketBytes, 0, 0).encode(s.sizes.len() as u32)?;
            s.key = Some(key);
            s.capacity = capacity;
            s.total = 0;
            s.published = false;
            s.sizes.fill(0);
            s.offered.fill(false);
            Ok(())
        })
    }
    /// Caller-provided output is valid only when true; publish one frame to each
    /// indexed destination. Source buffer capacity is fixed before begin.
    pub fn offer(&mut self, frame: ControlFrame, out: &mut [ControlFrame]) -> Result<bool> {
        self.apply(|s| {
            let key = s.matches(frame)?;
            if frame.action != Action::OfferBytes || s.published || out.len() != s.sizes.len() {
                return Err("BYTE_ADMISSION_OFFER".into());
            }
            let dst = frame.destination_rank as usize;
            if s.offered[dst] {
                return Err("BYTE_ADMISSION_DUPLICATE".into());
            }
            let total = s
                .total
                .checked_add(frame.payload_bytes)
                .ok_or("BYTE_ADMISSION_OVERFLOW")?;
            if total > s.capacity {
                return Err("BYTE_ADMISSION_SEND_CAPACITY".into());
            }
            s.total = total;
            s.sizes[dst] = frame.payload_bytes;
            s.offered[dst] = true;
            if !s.offered.iter().all(|&x| x) {
                return Ok(false);
            }
            s.guard.prepare(key, &s.sizes, 1, s.capacity)?;
            for (dst, output) in out.iter_mut().enumerate() {
                *output = Self::frame(key, Action::TicketBytes, dst as u32, s.sizes[dst]);
            }
            s.published = true;
            Ok(true)
        })
    }
    pub fn ack(&mut self, frame: ControlFrame) -> Result<()> {
        self.apply(|s| {
            let key = s.matches(frame)?;
            if frame.action != Action::Admitted || !s.published {
                return Err("BYTE_ADMISSION_ACK".into());
            }
            s.guard.admit(key, frame.rank, frame.payload_bytes)
        })
    }
    pub fn launch(&mut self, next: &mut u64, out: &mut [ControlFrame]) -> Result<bool> {
        self.apply(|s| {
            if out.len() != s.sizes.len() {
                return Err("BYTE_ADMISSION_OUTPUT".into());
            }
            let key = s.key.ok_or("BYTE_ADMISSION_IDLE")?;
            if !s.published {
                return Ok(false);
            }
            if !s.guard.launch_ordered(key, next)? {
                return Ok(false);
            }
            out.fill(Self::frame(key, Action::Launch, 0, 0));
            Ok(true)
        })
    }
    /// Dispatcher proves all transfer and consumer leases drained first.
    pub fn retire(&mut self, key: TicketKey) -> Result<()> {
        self.apply(|s| {
            s.guard.retire(key)?;
            s.key = None;
            s.published = false;
            Ok(())
        })
    }
}
