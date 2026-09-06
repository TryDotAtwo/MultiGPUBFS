//! Fixed host fanout bookkeeping for one physical payload bank.
//! This does not observe CUDA: retirement additionally requires the dispatcher
//! to prove transfer and consumer event completion. No device data lives here.
use crate::{control_wire::Plane, scatter_admission::TicketKey};
use mgbfs_core::Result;

#[derive(Clone, Copy, Debug)]
pub struct PayloadConsumer {
    key: TicketKey,
    index: usize,
}

pub struct PayloadLease {
    world: u32,
    capacity: u64,
    active: Option<TicketKey>,
    retired: Option<TicketKey>,
    completed: Vec<bool>,
    issued: usize,
    remaining: usize,
    sealed: bool,
    failed: bool,
}
impl PayloadLease {
    pub fn new(world: u32, capacity: u64, jobs: usize) -> Result<Self> {
        if world == 0 || jobs == 0 {
            return Err("PAYLOAD_CAPACITY".into());
        }
        let mut completed = Vec::new();
        completed
            .try_reserve_exact(jobs)
            .map_err(|_| "PAYLOAD_CAPACITY")?;
        completed.resize(jobs, false);
        Ok(Self {
            world,
            capacity,
            active: None,
            retired: None,
            completed,
            issued: 0,
            remaining: 0,
            sealed: false,
            failed: false,
        })
    }
    fn apply<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.failed {
            return Err("PAYLOAD_FAILED".into());
        }
        let result = f(self);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn ticket(&self, key: TicketKey) -> Result<()> {
        if self.active != Some(key) {
            return Err("PAYLOAD_TICKET".into());
        }
        Ok(())
    }
    /// Reserve before byte-admission ACK. Physical capacity is fixed at setup.
    pub fn reserve(&mut self, key: TicketKey, bytes: u64) -> Result<()> {
        self.apply(|s| {
            if s.active.is_some() {
                return Err("PAYLOAD_BUSY".into());
            }
            if key.source >= s.world
                || key.plane == Plane::None
                || s.retired
                    .is_some_and(|old| key.epoch <= old.epoch || key.depth < old.depth)
            {
                return Err("PAYLOAD_TICKET".into());
            }
            if bytes > s.capacity {
                return Err("PAYLOAD_BYTE_CAPACITY".into());
            }
            s.active = Some(key);
            s.issued = 0;
            s.remaining = 0;
            s.sealed = false;
            s.completed.fill(false);
            Ok(())
        })
    }
    /// Issue once per downstream job before submission. IDs are not recycled
    /// within a ticket, so duplicate completion cannot consume another lease.
    pub fn consumer(&mut self, key: TicketKey) -> Result<PayloadConsumer> {
        self.apply(|s| {
            s.ticket(key)?;
            if s.sealed {
                return Err("PAYLOAD_SEALED".into());
            }
            if s.issued == s.completed.len() {
                return Err("PAYLOAD_JOB_CAPACITY".into());
            }
            let index = s.issued;
            s.issued += 1;
            s.remaining += 1;
            Ok(PayloadConsumer { key, index })
        })
    }
    /// Job splitter has enumerated ALL consumers, including zero jobs.
    pub fn seal(&mut self, key: TicketKey) -> Result<()> {
        self.apply(|s| {
            s.ticket(key)?;
            if s.sealed {
                return Err("PAYLOAD_SEALED".into());
            }
            s.sealed = true;
            Ok(())
        })
    }
    /// Caller has observed this specific consumer's completion event.
    pub fn complete(&mut self, consumer: PayloadConsumer) -> Result<()> {
        self.apply(|s| {
            s.ticket(consumer.key)?;
            if consumer.index >= s.issued || s.completed[consumer.index] {
                return Err("PAYLOAD_CONSUMER".into());
            }
            s.completed[consumer.index] = true;
            s.remaining -= 1;
            Ok(())
        })
    }
    pub fn drained(&mut self, key: TicketKey) -> Result<bool> {
        self.apply(|s| {
            s.ticket(key)?;
            Ok(s.sealed && s.remaining == 0)
        })
    }
    /// Additionally requires the ticket's transfer event to be complete.
    /// Dispatcher must not return CONSUMED or reuse the bank before this call.
    pub fn retire(&mut self, key: TicketKey) -> Result<()> {
        self.apply(|s| {
            s.ticket(key)?;
            if !s.sealed || s.remaining != 0 {
                return Err("PAYLOAD_NOT_DRAINED".into());
            }
            s.retired = s.active.take();
            Ok(())
        })
    }
}
