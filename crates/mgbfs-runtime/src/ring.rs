use mgbfs_core::Result;
use std::collections::VecDeque;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub id: u64,
    pub begin: u64,
    pub count: u64,
}
#[derive(PartialEq, Eq)]
enum State {
    Reserved,
    Materialized,
    Current,
    Enumerated,
}
struct Live {
    extent: Extent,
    end: u64,
    state: State,
    archived: bool,
    origin_leases: u64,
}
pub struct StateRing {
    capacity: u64,
    descriptors: usize,
    head: u64,
    tail: u64,
    next_id: u64,
    live: VecDeque<Live>,
}
impl StateRing {
    pub fn new(records: u64, descriptors: usize) -> Result<Self> {
        if records == 0 || descriptors == 0 {
            return Err("STATE_RING_CAPACITY".into());
        }
        Ok(Self {
            capacity: records,
            descriptors,
            head: 0,
            tail: 0,
            next_id: 0,
            live: VecDeque::with_capacity(descriptors),
        })
    }
    pub fn reserve(&mut self, records: u64) -> Result<Extent> {
        if records == 0 || records > self.capacity || self.live.len() == self.descriptors {
            return Err("STATE_RING_CAPACITY".into());
        }
        let mut start = self.tail;
        let remainder = self.capacity - start % self.capacity;
        if records > remainder {
            start = start
                .checked_add(remainder)
                .ok_or("STATE_RING_COUNTER_OVERFLOW")?;
        }
        let end = start
            .checked_add(records)
            .ok_or("STATE_RING_COUNTER_OVERFLOW")?;
        let head = if self.live.is_empty() {
            start
        } else {
            self.head
        };
        if end - head > self.capacity {
            return Err("STATE_RING_CAPACITY".into());
        }
        let next_id = self
            .next_id
            .checked_add(1)
            .ok_or("STATE_RING_ID_OVERFLOW")?;
        let extent = Extent {
            id: self.next_id,
            begin: start % self.capacity,
            count: records,
        };
        self.live.push_back(Live {
            extent,
            end,
            state: State::Reserved,
            archived: false,
            origin_leases: 0,
        });
        self.head = head;
        self.tail = end;
        self.next_id = next_id;
        Ok(extent)
    }
    fn find(&mut self, id: u64) -> Result<&mut Live> {
        self.live
            .iter_mut()
            .find(|x| x.extent.id == id)
            .ok_or_else(|| "STALE_STATE_REF".into())
    }
    fn transition(&mut self, id: u64, from: State, to: State) -> Result<()> {
        let x = self.find(id)?;
        if x.state != from {
            return Err("STATE_RING_TRANSITION".into());
        }
        x.state = to;
        Ok(())
    }
    pub fn materialized(&mut self, id: u64) -> Result<()> {
        self.transition(id, State::Reserved, State::Materialized)
    }
    pub fn archived(&mut self, id: u64) -> Result<()> {
        let x = self.find(id)?;
        if x.state == State::Reserved || x.archived {
            return Err("ARCHIVE_LEASE_STATE".into());
        }
        x.archived = true;
        Ok(())
    }
    pub fn enumerated(&mut self, id: u64) -> Result<()> {
        self.transition(id, State::Current, State::Enumerated)
    }
    pub fn hold_origins(&mut self, id: u64) -> Result<()> {
        let x = self.find(id)?;
        if x.state != State::Current {
            return Err("ORIGIN_LEASE_STATE".into());
        }
        x.origin_leases = x
            .origin_leases
            .checked_add(1)
            .ok_or("ORIGIN_LEASE_OVERFLOW")?;
        Ok(())
    }
    pub fn release_origins(&mut self, id: u64) -> Result<()> {
        let x = self.find(id)?;
        x.origin_leases = x
            .origin_leases
            .checked_sub(1)
            .ok_or("ORIGIN_LEASE_UNDERFLOW")?;
        Ok(())
    }
    pub fn publish(&mut self, id: u64) -> Result<()> {
        self.transition(id, State::Materialized, State::Current)
    }
    pub fn reclaim(&mut self) -> u64 {
        let mut records = 0;
        while self
            .live
            .front()
            .is_some_and(|x| x.state == State::Enumerated && x.archived && x.origin_leases == 0)
        {
            let x = self.live.pop_front().unwrap();
            self.head = x.end;
            records += x.extent.count;
        }
        if self.live.is_empty() {
            self.head = self.tail;
        }
        records
    }
}
