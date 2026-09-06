//! Bounded epoch/finalization control state. Socket/GPU pumping is separate;
//! no NCCL calls, CUDA drain proof or payload-count validation occur here.
use crate::control_wire::{Action, ControlFrame, Plane, NO_SLOT};
use crate::rank_epochs::RankEpochs;
use mgbfs_core::Result;
use std::collections::VecDeque;
pub struct EpochCoordinator {
    world: u32,
    ranks: Vec<RankEpochs>,
    pending: Vec<VecDeque<u64>>,
    next: u64,
    source_cursor: [usize; 4],
    source_closed: Vec<bool>,
    finalizing: Option<u64>,
    finalized: Vec<bool>,
    depth: u64,
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
        let mut source_closed = Vec::new();
        source_closed
            .try_reserve_exact(world as usize)
            .map_err(|_| "CONTROL_COORDINATOR_CAPACITY")?;
        source_closed.resize(world as usize, false);
        let mut finalized = Vec::new();
        finalized
            .try_reserve_exact(world as usize)
            .map_err(|_| "CONTROL_COORDINATOR_CAPACITY")?;
        finalized.resize(world as usize, false);
        Ok(Self {
            world,
            ranks,
            pending,
            next: 0,
            source_cursor: [0; 4],
            source_closed,
            finalizing: None,
            finalized,
            depth: 0,
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
        if let Some(epoch) = self.finalizing {
            let rank = frame.rank as usize;
            if frame.action != Action::Finalized
                || frame.epoch != epoch
                || frame.depth != self.depth
                || self.finalized[rank]
            {
                return Err("CONTROL_FINALIZATION_ADMISSION_CLOSED".into());
            }
            let expected = self.ranks[rank].finish_depth(
                ControlFrame {
                    action: Action::Finalize,
                    rank: 0,
                    ..frame
                },
                true,
            )?;
            if expected != frame {
                return Err("CONTROL_FINALIZATION_ACK".into());
            }
            self.finalized[rank] = true;
            return Ok(());
        }
        let rank = frame.rank as usize;
        let expected = match frame.action {
            Action::Ready => {
                if frame.plane == Plane::Candidate && self.source_closed[rank] {
                    return Err("CONTROL_SOURCE_CLOSED".into());
                }
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
            Action::SourceClosed => {
                if frame.depth != self.depth || self.source_closed[rank] {
                    return Err("CONTROL_SOURCE_CLOSE".into());
                }
                self.source_closed[rank] = true;
                frame
            }
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
        if let Some(epoch) = self.finalizing {
            if !self.finalized.iter().all(|x| *x) {
                return Ok(false);
            }
            let depth = self
                .depth
                .checked_add(1)
                .ok_or("CONTROL_COORDINATOR_DEPTH")?;
            let publish = ControlFrame {
                action: Action::Publish,
                rank: 0,
                depth,
                epoch,
                slot: NO_SLOT,
                plane: Plane::None,
                source_rank: 0,
                fatal_code: 0,
            };
            for rank in &mut self.ranks {
                rank.publish(publish)?;
            }
            frames.fill(publish);
            self.depth = depth;
            self.finalizing = None;
            self.finalized.fill(false);
            self.source_closed.fill(false);
            return Ok(true);
        }
        for plane in [
            Plane::Response,
            Plane::Request,
            Plane::Receipt,
            Plane::Candidate,
        ] {
            let kind = plane_index(plane)?;
            let base = kind * self.world as usize;
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
            // One bounded source fragment per ticket avoids summing independent
            // source maxima into the same receive bank.
            let source = (0..self.world as usize)
                .map(|offset| (self.source_cursor[kind] + offset) % self.world as usize)
                .find(|&rank| !self.pending[base + rank].is_empty())
                .unwrap();
            let next = self
                .next
                .checked_add(1)
                .ok_or("CONTROL_COORDINATOR_SEQUENCE")?;
            for (rank, frame) in frames.iter_mut().enumerate() {
                let slot = if rank == source {
                    *self.pending[base + rank].front().unwrap()
                } else {
                    NO_SLOT
                };
                *frame = ControlFrame {
                    action: Action::Begin,
                    rank: 0,
                    depth: self.depth,
                    epoch: self.next,
                    slot,
                    plane,
                    source_rank: source as u32,
                    fatal_code: 0,
                };
                self.ranks[rank].begin(*frame)?;
                if slot != NO_SLOT {
                    self.pending[base + rank].pop_front();
                }
            }
            self.next = next;
            self.source_cursor[kind] = (source + 1) % self.world as usize;
            return Ok(true);
        }
        if self.source_closed.iter().all(|x| *x)
            && self.pending.iter().all(VecDeque::is_empty)
            && self.ranks.iter().all(RankEpochs::drained)
        {
            let next = self
                .next
                .checked_add(1)
                .ok_or("CONTROL_COORDINATOR_SEQUENCE")?;
            frames.fill(ControlFrame {
                action: Action::Finalize,
                rank: 0,
                depth: self.depth,
                epoch: self.next,
                slot: NO_SLOT,
                plane: Plane::None,
                source_rank: 0,
                fatal_code: 0,
            });
            self.finalizing = Some(self.next);
            self.next = next;
            return Ok(true);
        }
        Ok(false)
    }
}
