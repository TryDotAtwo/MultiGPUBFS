use mgbfs_core::Result;
use std::collections::VecDeque;
#[derive(Debug, PartialEq, Eq)]
pub struct Epoch {
    pub id: u64,
    pub offers: Vec<Option<u64>>,
}
/// CPU protocol oracle. All ranks acknowledge each issued epoch, including
/// ranks with no payload. This is not the TCP/NCCL implementation.
pub struct Sequencer {
    pending: Vec<VecDeque<u64>>,
    closed: Vec<bool>,
    limit: usize,
    next: u64,
    active: Option<(Epoch, Vec<bool>)>,
}
impl Sequencer {
    pub fn new(ranks: usize, slots_per_rank: usize) -> Result<Self> {
        if ranks == 0 || slots_per_rank == 0 {
            return Err("invalid sequencer capacity".into());
        }
        Ok(Self {
            pending: (0..ranks)
                .map(|_| VecDeque::with_capacity(slots_per_rank))
                .collect(),
            closed: vec![false; ranks],
            limit: slots_per_rank,
            next: 0,
            active: None,
        })
    }
    pub fn ready(&mut self, rank: usize, slot: u64) -> Result<()> {
        let q = self.pending.get_mut(rank).ok_or("invalid rank")?;
        let active_slot = self.active.as_ref().and_then(|(e, _)| e.offers[rank]);
        if self.closed[rank] {
            return Err("rank already closed".into());
        }
        if q.contains(&slot) || active_slot == Some(slot) {
            return Err("duplicate live slot".into());
        }
        if q.len() + usize::from(active_slot.is_some()) >= self.limit {
            return Err("route slot capacity".into());
        }
        q.push_back(slot);
        Ok(())
    }
    pub fn close(&mut self, rank: usize) -> Result<()> {
        let closed = self.closed.get_mut(rank).ok_or("invalid rank")?;
        if *closed {
            return Err("rank already closed".into());
        }
        *closed = true;
        Ok(())
    }
    pub fn begin(&mut self) -> Result<Option<Epoch>> {
        if self.active.is_some() {
            return Err("epoch still in flight".into());
        }
        if self.pending.iter().all(|q| q.is_empty()) {
            return Ok(None);
        }
        let next = self.next.checked_add(1).ok_or("epoch counter overflow")?;
        let offers: Vec<_> = self.pending.iter_mut().map(|q| q.pop_front()).collect();
        let epoch = Epoch {
            id: self.next,
            offers,
        };
        self.next = next;
        self.active = Some((
            Epoch {
                id: epoch.id,
                offers: epoch.offers.clone(),
            },
            vec![false; self.pending.len()],
        ));
        Ok(Some(epoch))
    }
    pub fn complete(&mut self, rank: usize, epoch: u64) -> Result<()> {
        let (active, acked) = self.active.as_mut().ok_or("no active epoch")?;
        if epoch != active.id {
            return Err("wrong epoch".into());
        }
        let ack = acked.get_mut(rank).ok_or("invalid rank")?;
        if *ack {
            return Err("duplicate completion".into());
        }
        *ack = true;
        if acked.iter().all(|v| *v) {
            self.active = None;
        }
        Ok(())
    }
    pub fn drained(&self) -> bool {
        self.active.is_none()
            && self.closed.iter().all(|v| *v)
            && self.pending.iter().all(|q| q.is_empty())
    }
}
