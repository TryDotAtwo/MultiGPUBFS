//! Bounded data-epoch coordinator. Finalization and socket/GPU pumping are
//! deliberately separate; no NCCL calls or payload-count validation occur here.
use crate::control_wire::{Action, ControlFrame, Plane, NO_SLOT};
use crate::rank_epochs::RankEpochs;
use mgbfs_core::Result;
use std::collections::VecDeque;
pub struct EpochCoordinator {
    world: u32,
    ranks: Vec<RankEpochs>,
    pending: Vec<VecDeque<u64>>,
    next: u64,
    failed: bool,
}
fn plane_index(plane: Plane) -> Result<usize> {
    match plane {
        Plane::Candidate => Ok(0),
        Plane::Request => Ok(1),
        Plane::Response => Ok(2),
        Plane::Receipt => Ok(3),
        Plane::None => Err("CONTROL_COORDINATOR_PLANE".into()),
    }
}
impl EpochCoordinator {
    pub fn new(world: u32, slots: usize) -> Result<Self> {
        if world == 0 || slots == 0 {
            return Err("CONTROL_COORDINATOR_CAPACITY".into());
        }
        let count = (world as usize)
            .checked_mul(4)
            .ok_or("CONTROL_COORDINATOR_CAPACITY")?;
        count
            .checked_mul(slots)
            .ok_or("CONTROL_COORDINATOR_CAPACITY")?;
        let mut ranks = Vec::new();
        ranks
            .try_reserve_exact(world as usize)
            .map_err(|_| "CONTROL_COORDINATOR_CAPACITY")?;
        for rank in 0..world {
            ranks.push(RankEpochs::new(world, rank, slots)?);
        }
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(count)
            .map_err(|_| "CONTROL_COORDINATOR_CAPACITY")?;
        for _ in 0..count {
            let mut queue = VecDeque::new();
            queue
                .try_reserve_exact(slots)
                .map_err(|_| "CONTROL_COORDINATOR_CAPACITY")?;
            pending.push(queue);
        }
        Ok(Self {
            world,
            ranks,
            pending,
            next: 0,
            failed: false,
        })
    }
    fn alive(&self) -> Result<()> {
        if self.failed {
            Err("CONTROL_COORDINATOR_FAILED".into())
        } else {
            Ok(())
        }
    }
    pub fn receive(&mut self, frame: ControlFrame) -> Result<()> {
        self.alive()?;
        let result = self.receive_inner(frame);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn receive_inner(&mut self, frame: ControlFrame) -> Result<()> {
        frame.encode(self.world)?;
        let rank = frame.rank as usize;
        let expected = match frame.action {
            Action::Ready => {
                let expected = self.ranks[rank].offer(frame.plane, frame.slot)?;
                if expected != frame {
                    return Err("CONTROL_COORDINATOR_FRAME".into());
                }
                let queue = plane_index(frame.plane)? * self.world as usize + rank;
                self.pending[queue].push_back(frame.slot);
                expected
            }
            Action::Complete => self.ranks[rank].transfer_complete(frame.epoch)?,
            Action::Consumed => self.ranks[rank].consume(frame.epoch)?,
            _ => return Err("CONTROL_COORDINATOR_ACTION".into()),
        };
        if expected != frame {
            return Err("CONTROL_COORDINATOR_FRAME".into());
        }
        Ok(())
    }
    /// Caller reserves all outbound command capacity BEFORE calling. The output
    /// is one BEGIN per recipient, with sender rank 0 and the same global epoch.
    pub fn issue(&mut self, frames: &mut [ControlFrame]) -> Result<bool> {
        self.alive()?;
        let result = self.issue_inner(frames);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn issue_inner(&mut self, frames: &mut [ControlFrame]) -> Result<bool> {
        if frames.len() != self.world as usize {
            return Err("CONTROL_COORDINATOR_OUTPUT".into());
        }
        for plane in [
            Plane::Response,
            Plane::Request,
            Plane::Receipt,
            Plane::Candidate,
        ] {
            let base = plane_index(plane)? * self.world as usize;
            if self.pending[base..base + self.world as usize]
                .iter()
                .all(VecDeque::is_empty)
            {
                continue;
            }
            let mut available = true;
            for rank in &self.ranks {
                available &= rank.receive_credit_available(plane)?;
            }
            if !available {
                continue;
            }
            let next = self
                .next
                .checked_add(1)
                .ok_or("CONTROL_COORDINATOR_SEQUENCE")?;
            for (rank, frame) in frames.iter_mut().enumerate() {
                let slot = self.pending[base + rank]
                    .front()
                    .copied()
                    .unwrap_or(NO_SLOT);
                *frame = ControlFrame {
                    action: Action::Begin,
                    rank: 0,
                    depth: 0,
                    epoch: self.next,
                    slot,
                    plane,
                    fatal_code: 0,
                };
                self.ranks[rank].begin(*frame)?;
                if slot != NO_SLOT {
                    self.pending[base + rank].pop_front();
                }
            }
            self.next = next;
            return Ok(true);
        }
        Ok(false)
    }
}
