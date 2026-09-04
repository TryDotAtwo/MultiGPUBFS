use mgbfs_core::Result;
use std::collections::VecDeque;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub id: u64,
    pub sequence: u64,
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
    peak: u64,
    live: VecDeque<Live>,
}
impl StateRing {
    /// Physical high-water including wrap padding, in records (not bytes).
    pub fn peak_records(&self) -> u64 {
        self.peak
    }
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
            peak: 0,
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
            sequence: start,
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
        self.peak = self.peak.max(end - head);
        Ok(extent)
    }
    /// Absolute record address, not an extent descriptor ID or physical index.
    pub fn state_ref(&self, id: u64, row: u64) -> Result<u64> {
        let x = self
            .live
            .iter()
            .find(|x| x.extent.id == id)
            .ok_or("STALE_STATE_REF")?;
        if row >= x.extent.count {
            return Err("STATE_REF_ROW".into());
        }
        let reference = x
            .extent
            .sequence
            .checked_add(row)
            .ok_or("STATE_RING_COUNTER_OVERFLOW")?;
        self.resolve(reference)?;
        Ok(reference)
    }
    pub fn resolve(&self, reference: u64) -> Result<u64> {
        // CPU contract model scans bounded extent metadata, never state bytes.
        // Production lookup must operate on the flat extent directory.
        let x = self
            .live
            .iter()
            .find(|x| reference >= x.extent.sequence && reference < x.end)
            .ok_or("STALE_STATE_REF")?;
        if x.state != State::Current && !(x.state == State::Enumerated && x.origin_leases > 0) {
            return Err("STATE_REF_NOT_READABLE".into());
        }
        Ok(reference % self.capacity)
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
    /// Retire an already archived DENSE parent prefix as soon as generation no
    /// longer reads it. HASH_FIRST callers must keep using origin leases and
    /// may not use this shortcut.
    pub fn retire_dense_prefix(&mut self, id: u64, records: u64) -> Result<u64> {
        if records == 0 {
            return Ok(0);
        }
        let front = self.live.front_mut().ok_or("STALE_STATE_REF")?;
        if front.extent.id != id {
            return Err("DENSE_PREFIX_NOT_HEAD".into());
        }
        if front.state != State::Current || records > front.extent.count {
            return Err("DENSE_PREFIX_RANGE".into());
        }
        if !front.archived {
            return Err("DENSE_PREFIX_ARCHIVE_LIVE".into());
        }
        if front.origin_leases != 0 {
            return Err("DENSE_PREFIX_ORIGIN_LIVE".into());
        }
        let new_sequence = front
            .extent
            .sequence
            .checked_add(records)
            .ok_or("STATE_RING_COUNTER_OVERFLOW")?;
        front.extent.sequence = new_sequence;
        front.extent.begin = new_sequence % self.capacity;
        front.extent.count -= records;
        self.head = new_sequence;
        if front.extent.count == 0 {
            self.live.pop_front();
            if self.live.is_empty() {
                self.head = self.tail;
            }
        }
        Ok(records)
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
