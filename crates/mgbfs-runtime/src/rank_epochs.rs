//! Bounded rank-local admission bookkeeping. Not yet a TCP/GPU dispatcher.
//! COMPLETE marks transfer completion; CONSUMED releases local receive credit.
//! Every epoch reserves a receive credit even
//! when this rank contributes no source payload.
use crate::control_wire::{Action, ControlFrame, Plane, NO_SLOT};
use mgbfs_core::Result;
#[derive(Clone, Copy)]
struct Offer {
    plane: Plane,
    slot: u64,
}
#[derive(Clone, Copy)]
struct Live {
    offer: Offer,
    epoch: u64,
    transferred: bool,
}
pub struct RankEpochs {
    world: u32,
    rank: u32,
    slots: usize,
    depth: u64,
    next: u64,
    awaiting_publish: Option<u64>,
    pending: Vec<Option<Offer>>,
    live: Vec<Option<Live>>,
    failed: bool,
}
fn storage<T: Clone>(count: usize) -> Result<Vec<Option<T>>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| "CONTROL_EPOCH_CAPACITY")?;
    result.resize(count, None);
    Ok(result)
}
impl RankEpochs {
    pub(crate) fn drained(&self) -> bool {
        !self.failed
            && self.awaiting_publish.is_none()
            && self.pending.iter().all(Option::is_none)
            && self.live.iter().all(Option::is_none)
    }
    pub(crate) fn receive_credit_available(&self, plane: Plane) -> Result<bool> {
        self.alive()?;
        if plane == Plane::None {
            return Err("CONTROL_EPOCH_PROTOCOL".into());
        }
        Ok(self
            .live
            .iter()
            .flatten()
            .filter(|x| x.offer.plane == plane)
            .count()
            < self.slots)
    }
    /// Called only after the externally coordinated FinalizeDepth jobs finish.
    /// Receipt of a FINALIZE frame by itself is not sufficient authorization.
    pub fn finish_depth(
        &mut self,
        frame: ControlFrame,
        local_drained: bool,
    ) -> Result<ControlFrame> {
        self.alive()?;
        if self.awaiting_publish.is_some()
            || frame.encode(self.world).is_err()
            || frame.action != Action::Finalize
            || frame.depth != self.depth
            || frame.epoch != self.next
            || !local_drained
            || self.pending.iter().any(Option::is_some)
            || self.live.iter().any(Option::is_some)
        {
            return self.reject();
        }
        let Some(next) = self.next.checked_add(1) else {
            return self.reject();
        };
        let Some(depth) = self.depth.checked_add(1) else {
            return self.reject();
        };
        self.next = next;
        self.depth = depth;
        self.awaiting_publish = Some(frame.epoch);
        Ok(ControlFrame {
            action: Action::Finalized,
            rank: self.rank,
            ..frame
        })
    }
    pub fn publish(&mut self, frame: ControlFrame) -> Result<()> {
        self.alive()?;
        if frame.encode(self.world).is_err()
            || frame.action != Action::Publish
            || frame.depth != self.depth
            || self.awaiting_publish != Some(frame.epoch)
        {
            return self.reject();
        }
        self.awaiting_publish = None;
        Ok(())
    }
    pub fn new(world: u32, rank: u32, slots: usize) -> Result<Self> {
        if world == 0 || rank >= world || slots == 0 {
            return Err("CONTROL_EPOCH_CAPACITY".into());
        }
        let count = slots.checked_mul(4).ok_or("CONTROL_EPOCH_CAPACITY")?;
        Ok(Self {
            world,
            rank,
            slots,
            depth: 0,
            next: 0,
            awaiting_publish: None,
            pending: storage(count)?,
            live: storage(count)?,
            failed: false,
        })
    }
    fn alive(&self) -> Result<()> {
        if self.failed {
            Err("CONTROL_EPOCH_FAILED".into())
        } else {
            Ok(())
        }
    }
    fn reject<T>(&mut self) -> Result<T> {
        self.failed = true;
        Err("CONTROL_EPOCH_PROTOCOL".into())
    }
    pub fn offer(&mut self, plane: Plane, slot: u64) -> Result<ControlFrame> {
        self.alive()?;
        if self.awaiting_publish.is_some() {
            return self.reject();
        }
        let pending = self.pending.iter().flatten().filter(|x| x.plane == plane);
        let live = self
            .live
            .iter()
            .flatten()
            .filter(|x| x.offer.plane == plane && x.offer.slot != NO_SLOT);
        if plane == Plane::None
            || slot == NO_SLOT
            || pending.clone().any(|x| x.slot == slot)
            || live.clone().any(|x| x.offer.slot == slot)
            || pending.count() + live.count() >= self.slots
        {
            return self.reject();
        }
        let Some(index) = self.pending.iter().position(Option::is_none) else {
            return self.reject();
        };
        self.pending[index] = Some(Offer { plane, slot });
        Ok(ControlFrame {
            action: Action::Ready,
            rank: self.rank,
            depth: self.depth,
            epoch: 0,
            slot,
            plane,
            fatal_code: 0,
        })
    }
    pub fn begin(&mut self, frame: ControlFrame) -> Result<()> {
        self.alive()?;
        if self.awaiting_publish.is_some() {
            return self.reject();
        }
        if frame.encode(self.world).is_err()
            || frame.action != Action::Begin
            || frame.depth != self.depth
            || frame.epoch != self.next
            || self
                .live
                .iter()
                .flatten()
                .filter(|x| x.offer.plane == frame.plane)
                .count()
                >= self.slots
        {
            return self.reject();
        }
        let pending = self
            .pending
            .iter()
            .position(|x| x.is_some_and(|x| x.plane == frame.plane && x.slot == frame.slot));
        if frame.slot != NO_SLOT && pending.is_none() {
            return self.reject();
        }
        let Some(index) = self.live.iter().position(Option::is_none) else {
            return self.reject();
        };
        let Some(next) = self.next.checked_add(1) else {
            return self.reject();
        };
        if let Some(pending) = pending {
            self.pending[pending] = None;
        }
        self.live[index] = Some(Live {
            offer: Offer {
                plane: frame.plane,
                slot: frame.slot,
            },
            epoch: frame.epoch,
            transferred: false,
        });
        self.next = next;
        Ok(())
    }
    /// Caller has observed the ordered NCCL completion event. This does not
    /// release source ownership or receive credit while consumers are active.
    pub fn transfer_complete(&mut self, epoch: u64) -> Result<ControlFrame> {
        self.alive()?;
        let expected = self
            .live
            .iter()
            .flatten()
            .filter(|x| !x.transferred)
            .map(|x| x.epoch)
            .min();
        if expected != Some(epoch) {
            return self.reject();
        }
        let live = self
            .live
            .iter_mut()
            .flatten()
            .find(|x| x.epoch == epoch)
            .unwrap();
        live.transferred = true;
        Ok(ControlFrame {
            action: Action::Complete,
            rank: self.rank,
            depth: self.depth,
            epoch,
            slot: NO_SLOT,
            plane: live.offer.plane,
            fatal_code: 0,
        })
    }
    /// Caller proves all local epoch consumers have completed before retiring.
    /// GPU completion events and global receive-credit coordination are external.
    pub fn consume(&mut self, epoch: u64) -> Result<ControlFrame> {
        self.alive()?;
        let Some(index) = self
            .live
            .iter()
            .position(|x| x.is_some_and(|x| x.epoch == epoch))
        else {
            return self.reject();
        };
        if !self.live[index].unwrap().transferred {
            return self.reject();
        }
        let live = self.live[index].take().unwrap();
        Ok(ControlFrame {
            action: Action::Consumed,
            rank: self.rank,
            depth: self.depth,
            epoch,
            slot: NO_SLOT,
            plane: live.offer.plane,
            fatal_code: 0,
        })
    }
}
