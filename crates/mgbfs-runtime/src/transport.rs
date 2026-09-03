//! CPU typed transport oracle. Not TCP/NCCL execution.
use mgbfs_core::Result;
use std::collections::{BTreeMap, VecDeque};
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Candidate,
    Request,
    Response,
    Receipt,
    Finalize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ticket {
    pub seq: u64,
    pub kind: Kind,
    pub source: usize,
    pub slot: u64,
    pub target_depth: u32,
    pub counts: Vec<u64>,
}
struct Live {
    ticket: Ticket,
    ack: Vec<bool>,
}
pub struct Transport {
    ranks: usize,
    slots: usize,
    records: u64,
    limit: usize,
    pending: Vec<VecDeque<Ticket>>,
    live: BTreeMap<u64, Live>,
    receiving: Vec<[usize; 4]>,
    closed: Vec<bool>,
    work: Vec<u64>,
    cursor: [usize; 4],
    next: u64,
    depth: u32,
    finalizing: bool,
    finished: bool,
}
fn index(kind: Kind) -> Result<usize> {
    match kind {
        Kind::Candidate => Ok(0),
        Kind::Request => Ok(1),
        Kind::Response => Ok(2),
        Kind::Receipt => Ok(3),
        Kind::Finalize => Err("RESERVED_TICKET_KIND".into()),
    }
}
impl Transport {
    /// Called after the semantic finalization represented by all Finalize ACKs.
    /// Transport sequence remains monotonic across depths.
    pub fn advance_depth(&mut self) -> Result<()> {
        if !self.finished {
            return Err("DEPTH_NOT_DRAINED".into());
        }
        self.live.clear();
        self.closed.fill(false);
        self.finalizing = false;
        self.finished = false;
        self.depth = self.depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;
        Ok(())
    }
    pub fn new(ranks: usize, slots: usize, records: u64) -> Result<Self> {
        if ranks == 0 || slots == 0 || records == 0 {
            return Err("TRANSPORT_CAPACITY".into());
        }
        let queues = ranks.checked_mul(4).ok_or("TRANSPORT_CAPACITY")?;
        let limit = queues.checked_mul(slots).ok_or("TRANSPORT_CAPACITY")?;
        Ok(Self {
            ranks,
            slots,
            records,
            limit,
            pending: (0..queues).map(|_| VecDeque::new()).collect(),
            live: BTreeMap::new(),
            receiving: vec![[0; 4]; ranks],
            closed: vec![false; ranks],
            work: vec![0; ranks],
            cursor: [0; 4],
            next: 0,
            depth: 0,
            finalizing: false,
            finished: false,
        })
    }
    pub fn offer(&mut self, kind: Kind, source: usize, slot: u64, counts: Vec<u64>) -> Result<()> {
        let target_depth = self.depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;
        self.offer_at(kind, target_depth, source, slot, counts)
    }
    pub fn offer_at(
        &mut self,
        kind: Kind,
        target_depth: u32,
        source: usize,
        slot: u64,
        counts: Vec<u64>,
    ) -> Result<()> {
        let k = index(kind)?;
        let total = counts.iter().try_fold(0u64, |sum, &n| sum.checked_add(n));
        if self.finalizing
            || target_depth <= self.depth
            || source >= self.ranks
            || counts.len() != self.ranks
            || counts.iter().any(|&n| n > self.records)
            || total.map_or(true, |n| n > self.records)
        {
            return Err("INVALID_TRANSPORT_OFFER".into());
        }
        if kind == Kind::Candidate && self.closed[source] {
            return Err("SOURCE_CLOSED".into());
        }
        let q = &mut self.pending[k * self.ranks + source];
        let active: Vec<_> = self
            .live
            .values()
            .filter(|x| {
                x.ticket.kind == kind && x.ticket.source == source && !x.ack.iter().all(|a| *a)
            })
            .collect();
        if q.iter().any(|x| x.slot == slot)
            || active.iter().any(|x| x.ticket.slot == slot)
            || q.len() + active.len() >= self.slots
        {
            return Err("SOURCE_SLOT_CAPACITY_OR_ALIAS".into());
        }
        q.push_back(Ticket {
            seq: 0,
            kind,
            source,
            slot,
            target_depth,
            counts,
        });
        Ok(())
    }
    pub fn close_source(&mut self, rank: usize) -> Result<()> {
        if self.finalizing || rank >= self.ranks || self.closed[rank] {
            return Err("SOURCE_CLOSE_STATE".into());
        }
        self.closed[rank] = true;
        Ok(())
    }
    /// Outstanding non-transport jobs/leases. Register before retiring their input ticket.
    pub fn work(&mut self, rank: usize, add: bool) -> Result<()> {
        if self.finalizing || rank >= self.ranks {
            return Err("WORK_STATE".into());
        }
        self.work[rank] = if add {
            self.work[rank].checked_add(1)
        } else {
            self.work[rank].checked_sub(1)
        }
        .ok_or("WORK_COUNT")?;
        Ok(())
    }
    pub fn issue(&mut self) -> Result<Option<Ticket>> {
        if self.finalizing || self.live.len() >= self.limit {
            return Ok(None);
        }
        for k in [2, 1, 3, 0] {
            // Metadata credits are partitioned too, including zero-payload tickets.
            if self
                .live
                .values()
                .filter(|x| index(x.ticket.kind) == Ok(k))
                .count()
                >= self.ranks * self.slots
            {
                continue;
            }
            for step in 0..self.ranks {
                let src = (self.cursor[k] + step) % self.ranks;
                let q = k * self.ranks + src;
                let eligible = self.pending[q].front().is_some_and(|t| {
                    t.counts
                        .iter()
                        .enumerate()
                        .all(|(dst, &n)| n == 0 || self.receiving[dst][k] < self.slots)
                });
                if !eligible {
                    continue;
                }
                let next = self
                    .next
                    .checked_add(1)
                    .ok_or("TRANSPORT_SEQUENCE_OVERFLOW")?;
                let mut t = self.pending[q].pop_front().unwrap();
                t.seq = self.next;
                self.next = next;
                for (dst, &n) in t.counts.iter().enumerate() {
                    if n > 0 {
                        self.receiving[dst][k] += 1;
                    }
                }
                self.cursor[k] = (src + 1) % self.ranks;
                self.live.insert(
                    t.seq,
                    Live {
                        ticket: t.clone(),
                        ack: vec![false; self.ranks],
                    },
                );
                return Ok(Some(t));
            }
        }
        if self.closed.iter().all(|x| *x)
            && self.work.iter().all(|&n| n == 0)
            && self.pending.iter().all(|q| q.is_empty())
            && self.live.is_empty()
        {
            let next = self
                .next
                .checked_add(1)
                .ok_or("TRANSPORT_SEQUENCE_OVERFLOW")?;
            let t = Ticket {
                seq: self.next,
                kind: Kind::Finalize,
                source: 0,
                slot: 0,
                target_depth: self.depth,
                counts: vec![0; self.ranks],
            };
            self.next = next;
            self.finalizing = true;
            self.live.insert(
                t.seq,
                Live {
                    ticket: t.clone(),
                    ack: vec![false; self.ranks],
                },
            );
            return Ok(Some(t));
        }
        Ok(None)
    }
    pub fn complete(&mut self, rank: usize, seq: u64) -> Result<()> {
        if rank >= self.ranks {
            return Err("INVALID_RANK".into());
        }
        let expected = self
            .live
            .iter()
            .find(|(_, x)| !x.ack[rank])
            .map(|(&n, _)| n);
        if expected != Some(seq) {
            return Err("TRANSPORT_COMPLETION_ORDER".into());
        }
        let x = self.live.get_mut(&seq).unwrap();
        x.ack[rank] = true;
        if x.ticket.kind == Kind::Finalize && x.ack.iter().all(|a| *a) {
            self.finished = true;
        }
        Ok(())
    }
    pub fn consume(&mut self, seq: u64) -> Result<()> {
        let x = self.live.get(&seq).ok_or("STALE_TICKET")?;
        let k = index(x.ticket.kind)?;
        if !x.ack.iter().all(|a| *a) {
            return Err("TRANSFER_IN_FLIGHT".into());
        }
        let x = self.live.remove(&seq).unwrap();
        for (dst, &n) in x.ticket.counts.iter().enumerate() {
            if n > 0 {
                self.receiving[dst][k] -= 1;
            }
        }
        Ok(())
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
}
