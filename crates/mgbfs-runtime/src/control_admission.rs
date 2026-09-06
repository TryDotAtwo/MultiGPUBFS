//! Fixed host metadata pools for ControlPump's explicitly admitted mode.
//! No GPU storage or completion is inferred from these protocol records.
use crate::{
    byte_admission::{ByteAdmission, RankByteAdmission},
    control_wire::{ControlFrame, Plane, NO_SLOT},
    scatter_admission::TicketKey,
};
use mgbfs_core::Result;

fn plane(p: Plane) -> Result<usize> {
    match p {
        Plane::Candidate => Ok(0),
        Plane::Request => Ok(1),
        Plane::Response => Ok(2),
        Plane::Receipt => Ok(3),
        Plane::None => Err("CONTROL_ADMISSION_PLANE".into()),
    }
}
fn key(f: ControlFrame) -> TicketKey {
    TicketKey {
        depth: f.depth,
        epoch: f.epoch,
        source: f.source_rank,
        plane: f.plane,
        generation: f.slot,
    }
}
struct Local {
    begin: Option<ControlFrame>,
    ticket: Option<ControlFrame>,
    guard: RankByteAdmission,
    described: bool,
    launched: bool,
}
struct Root {
    begin: Option<ControlFrame>,
    guard: ByteAdmission,
    consumed: Vec<bool>,
    launched: bool,
}
pub(crate) struct Admission {
    world: u32,
    rank: u32,
    slots: usize,
    root_slots: usize,
    capacities: [u64; 4],
    local: Vec<Local>,
    root: Vec<Root>,
    next: u64,
}
impl Admission {
    pub fn new(world: u32, rank: u32, slots: usize, capacities: [u64; 4]) -> Result<Self> {
        let count = slots.checked_mul(4).ok_or("CONTROL_ADMISSION_CAPACITY")?;
        // Every globally live ticket retains at least one rank-local credit,
        // not necessarily a credit on rank zero or on the same slow rank.
        let root_slots = slots
            .checked_mul(world as usize)
            .ok_or("CONTROL_ADMISSION_CAPACITY")?;
        let root_count = root_slots
            .checked_mul(4)
            .ok_or("CONTROL_ADMISSION_CAPACITY")?;
        let mut local = Vec::new();
        local
            .try_reserve_exact(count)
            .map_err(|_| "CONTROL_ADMISSION_CAPACITY")?;
        let mut root = Vec::new();
        if rank == 0 {
            root.try_reserve_exact(root_count)
                .map_err(|_| "CONTROL_ADMISSION_CAPACITY")?;
        }
        for _ in 0..count {
            local.push(Local {
                begin: None,
                ticket: None,
                guard: RankByteAdmission::new(world, rank)?,
                described: false,
                launched: false,
            });
        }
        if rank == 0 {
            for _ in 0..root_count {
                let mut consumed = Vec::new();
                consumed
                    .try_reserve_exact(world as usize)
                    .map_err(|_| "CONTROL_ADMISSION_CAPACITY")?;
                consumed.resize(world as usize, false);
                root.push(Root {
                    begin: None,
                    guard: ByteAdmission::new(world)?,
                    consumed,
                    launched: false,
                });
            }
        }
        Ok(Self {
            world,
            rank,
            slots,
            root_slots,
            capacities,
            local,
            root,
            next: 0,
        })
    }
    fn local_index(&self, epoch: u64) -> Result<usize> {
        self.local
            .iter()
            .position(|x| x.begin.is_some_and(|f| f.epoch == epoch))
            .ok_or("CONTROL_ADMISSION_UNKNOWN".into())
    }
    fn root_index(&self, epoch: u64) -> Result<usize> {
        self.root
            .iter()
            .position(|x| x.begin.is_some_and(|f| f.epoch == epoch))
            .ok_or("CONTROL_ADMISSION_UNKNOWN".into())
    }
    pub fn begin(&mut self, f: ControlFrame) -> Result<()> {
        let base = plane(f.plane)? * self.slots;
        let x = self.local[base..base + self.slots]
            .iter_mut()
            .find(|x| x.begin.is_none())
            .ok_or("CONTROL_ADMISSION_CAPACITY")?;
        x.begin = Some(f);
        x.ticket = None;
        x.described = false;
        x.launched = false;
        Ok(())
    }
    pub fn root_begin(&mut self, frames: &[ControlFrame]) -> Result<()> {
        let f = frames[frames[0].source_rank as usize];
        let kind = plane(f.plane)?;
        let base = kind * self.root_slots;
        let x = self.root[base..base + self.root_slots]
            .iter_mut()
            .find(|x| x.begin.is_none())
            .ok_or("CONTROL_ADMISSION_CAPACITY")?;
        x.guard.begin(key(f), self.capacities[kind])?;
        x.begin = Some(f);
        x.consumed.fill(false);
        x.launched = false;
        Ok(())
    }
    pub fn describe(&mut self, begin: ControlFrame, sizes: &[u64]) -> Result<()> {
        let i = self.local_index(begin.epoch)?;
        if self.local[i].begin != Some(begin)
            || begin.source_rank != self.rank
            || self.local[i].described
            || sizes.len() != self.world as usize
        {
            return Err("CONTROL_ADMISSION_DESCRIPTION".into());
        }
        let total = sizes
            .iter()
            .try_fold(0u64, |a, &b| a.checked_add(b))
            .ok_or("CONTROL_ADMISSION_OVERFLOW")?;
        if total > self.capacities[plane(begin.plane)?] {
            return Err("CONTROL_ADMISSION_SEND_CAPACITY".into());
        }
        self.local[i].described = true;
        Ok(())
    }
    pub fn offer(&mut self, f: ControlFrame, out: &mut [ControlFrame]) -> Result<bool> {
        let i = self.root_index(f.epoch)?;
        self.root[i].guard.offer(f, out)
    }
    pub fn ticket(&mut self, f: ControlFrame) -> Result<()> {
        let i = self.local_index(f.epoch)?;
        let x = &mut self.local[i];
        let b = x.begin.unwrap();
        if f.depth != b.depth
            || f.plane != b.plane
            || f.source_rank != b.source_rank
            || f.destination_rank != self.rank
            || x.ticket.is_some()
            || (b.slot != NO_SLOT && b.slot != f.slot)
        {
            return Err("CONTROL_ADMISSION_TICKET".into());
        }
        x.ticket = Some(f);
        Ok(())
    }
    pub fn admit(&mut self, f: ControlFrame, capacity: u64) -> Result<ControlFrame> {
        let i = self.local_index(f.epoch)?;
        if self.local[i].ticket != Some(f) {
            return Err("CONTROL_ADMISSION_TICKET".into());
        }
        self.local[i].guard.accept(f, capacity)
    }
    pub fn ack(&mut self, f: ControlFrame) -> Result<()> {
        let i = self.root_index(f.epoch)?;
        self.root[i].guard.ack(f)
    }
    pub fn issue_launch(&mut self, out: &mut [ControlFrame]) -> Result<bool> {
        let Some(i) = self
            .root
            .iter()
            .position(|x| x.begin.is_some_and(|f| f.epoch == self.next) && !x.launched)
        else {
            return Ok(false);
        };
        if !self.root[i].guard.launch(&mut self.next, out)? {
            return Ok(false);
        }
        self.root[i].launched = true;
        Ok(true)
    }
    pub fn launch(&mut self, f: ControlFrame) -> Result<()> {
        let i = self.local_index(f.epoch)?;
        self.local[i].guard.launch(f)?;
        self.local[i].launched = true;
        Ok(())
    }
    pub fn require_launched(&self, epoch: u64) -> Result<()> {
        if !self.local[self.local_index(epoch)?].launched {
            return Err("CONTROL_ADMISSION_NOT_LAUNCHED".into());
        }
        Ok(())
    }
    pub fn require_root_launched(&self, epoch: u64) -> Result<()> {
        if !self.root[self.root_index(epoch)?].launched {
            return Err("CONTROL_ADMISSION_NOT_LAUNCHED".into());
        }
        Ok(())
    }
    pub fn consume(&mut self, epoch: u64) -> Result<()> {
        let i = self.local_index(epoch)?;
        let x = &mut self.local[i];
        x.guard
            .retire(key(x.ticket.ok_or("CONTROL_ADMISSION_TICKET")?))?;
        x.begin = None;
        x.ticket = None;
        Ok(())
    }
    pub fn root_consume(&mut self, f: ControlFrame) -> Result<()> {
        let i = self.root_index(f.epoch)?;
        let x = &mut self.root[i];
        if x.consumed[f.rank as usize] {
            return Err("CONTROL_ADMISSION_CONSUMED".into());
        }
        x.consumed[f.rank as usize] = true;
        if x.consumed.iter().all(|&v| v) {
            x.guard.retire(key(x.begin.unwrap()))?;
            x.begin = None;
        }
        Ok(())
    }
    pub fn finalize(&mut self, f: ControlFrame) -> Result<()> {
        if self.local.iter().any(|x| x.begin.is_some())
            || self.root.iter().any(|x| x.begin.is_some())
        {
            return Err("CONTROL_ADMISSION_NOT_DRAINED".into());
        }
        if self.rank == 0 {
            if f.epoch != self.next {
                return Err("CONTROL_ADMISSION_SEQUENCE".into());
            }
            self.next = self
                .next
                .checked_add(1)
                .ok_or("CONTROL_ADMISSION_SEQUENCE")?;
        }
        Ok(())
    }
}
