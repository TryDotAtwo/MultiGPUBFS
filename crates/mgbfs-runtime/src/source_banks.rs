//! Source storage identity exists before a transport epoch is assigned.
//! Host bookkeeping only; caller owns the flat allocation and CUDA events.
use crate::{control_wire::Plane, scatter_admission::TicketKey};
use mgbfs_core::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceBank {
    index: usize,
    token: u64,
    depth: u64,
}
impl SourceBank {
    pub fn token(self) -> u64 {
        self.token
    }
}
struct Live {
    bank: SourceBank,
    ready: bool,
    ticket: Option<TicketKey>,
}
/// Handles are pool-local. Allocate `bytes()` before use; this structure does
/// not allocate device storage. The admitted control pump must validate global
/// epoch/depth order. Use distinct instances for independent traffic planes.
pub struct SourceBanks {
    rank: u32,
    stride: u64,
    bytes: u64,
    next: u64,
    slots: Vec<Option<Live>>,
    failed: bool,
}
impl SourceBanks {
    pub fn new(rank: u32, slots: usize, capacity: u64) -> Result<Self> {
        if slots == 0 {
            return Err("SOURCE_CAPACITY".into());
        }
        let stride = capacity.max(1).checked_add(255).ok_or("SOURCE_CAPACITY")? & !255;
        let bytes = stride
            .checked_mul(u64::try_from(slots).map_err(|_| "SOURCE_CAPACITY")?)
            .ok_or("SOURCE_CAPACITY")?;
        usize::try_from(bytes).map_err(|_| "SOURCE_CAPACITY")?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(slots)
            .map_err(|_| "SOURCE_CAPACITY")?;
        storage.resize_with(slots, || None);
        Ok(Self {
            rank,
            stride,
            bytes,
            next: 0,
            slots: storage,
            failed: false,
        })
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    fn apply<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.failed {
            return Err("SOURCE_FAILED".into());
        }
        let result = f(self);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn live(&mut self, bank: SourceBank) -> Result<&mut Live> {
        let live = self
            .slots
            .get_mut(bank.index)
            .and_then(Option::as_mut)
            .ok_or("SOURCE_HANDLE")?;
        if live.bank != bank {
            return Err("SOURCE_HANDLE".into());
        }
        Ok(live)
    }
    /// Reserve before launching generation; no epoch is known yet. Full means
    /// no producer admission, not permission to overwrite a live bank.
    pub fn reserve(&mut self, depth: u64) -> Result<Option<SourceBank>> {
        self.apply(|s| {
            let Some(index) = s.slots.iter().position(Option::is_none) else {
                return Ok(None);
            };
            // u64::MAX is the control protocol's NO_SLOT sentinel.
            let next = s.next.checked_add(1).ok_or("SOURCE_SEQUENCE")?;
            let bank = SourceBank {
                index,
                token: s.next,
                depth,
            };
            s.next = next;
            s.slots[index] = Some(Live {
                bank,
                ready: false,
                ticket: None,
            });
            Ok(Some(bank))
        })
    }
    pub fn offset(&mut self, bank: SourceBank) -> Result<u64> {
        self.apply(|s| {
            s.live(bank)?;
            Ok(bank.index as u64 * s.stride)
        })
    }
    /// Caller has observed generation/packing completion before offering READY.
    pub fn ready(&mut self, bank: SourceBank) -> Result<()> {
        self.apply(|s| {
            let live = s.live(bank)?;
            if live.ready {
                return Err("SOURCE_READY".into());
            }
            live.ready = true;
            Ok(())
        })
    }
    /// Resolve a source-local BEGIN token without assuming epoch or ready order
    /// equals physical bank order. The caller must validate the control frame.
    pub fn bind_ticket(&mut self, ticket: TicketKey) -> Result<SourceBank> {
        self.apply(|s| {
            let bank = s
                .slots
                .iter()
                .flatten()
                .find(|x| x.bank.token == ticket.generation)
                .map(|x| x.bank)
                .ok_or("SOURCE_TICKET")?;
            s.bind(bank, ticket)?;
            Ok(bank)
        })
    }
    pub fn bind(&mut self, bank: SourceBank, ticket: TicketKey) -> Result<()> {
        self.apply(|s| {
            if ticket.source != s.rank
                || ticket.generation != bank.token
                || ticket.depth != bank.depth
                || ticket.plane == Plane::None
                || s.slots
                    .iter()
                    .flatten()
                    .any(|x| x.ticket.is_some_and(|t| t.epoch == ticket.epoch))
            {
                return Err("SOURCE_TICKET".into());
            }
            let live = s.live(bank)?;
            if !live.ready || live.ticket.is_some() {
                return Err("SOURCE_TICKET".into());
            }
            live.ticket = Some(ticket);
            Ok(())
        })
    }
    /// Caller proves send completion AND all self-view consumers drained.
    /// Transfer COMPLETE alone is insufficient. Parent leases are separate.
    pub fn retire(&mut self, bank: SourceBank, ticket: TicketKey) -> Result<()> {
        self.apply(|s| {
            if s.live(bank)?.ticket != Some(ticket) {
                return Err("SOURCE_TICKET".into());
            }
            s.slots[bank.index] = None;
            Ok(())
        })
    }
}
