//! Fixed host fanout bookkeeping for one physical payload bank.
//! This does not observe CUDA: retirement additionally requires the dispatcher
//! to prove transfer and consumer event completion. No device data lives here.
use crate::{control_wire::Plane, scatter_admission::TicketKey};
use mgbfs_core::Result;

#[derive(Clone, Copy, Debug)]
pub struct PayloadBank {
    index: usize,
    key: TicketKey,
}
#[derive(Clone, Copy, Debug)]
pub struct BankConsumer {
    bank: PayloadBank,
    consumer: PayloadConsumer,
}

/// Fixed flat physical ranges and their host leases. The allocation owner must
/// allocate exactly `bytes()` device bytes before use and retain them until
/// all native events/consumers drain. Handles are local to this pool, not wire
/// IDs. Use separate pools for independent resource classes/traffic planes.
pub struct PayloadBanks {
    leases: Vec<PayloadLease>,
    stride: u64,
    bytes: u64,
    failed: bool,
}
impl PayloadBanks {
    pub fn new(
        world: u32,
        slots: usize,
        capacity: u64,
        jobs: usize,
        alignment: u64,
    ) -> Result<Self> {
        if slots == 0 || !alignment.is_power_of_two() {
            return Err("PAYLOAD_CAPACITY".into());
        }
        let stride = capacity
            .max(1)
            .checked_add(alignment - 1)
            .ok_or("PAYLOAD_CAPACITY")?
            & !(alignment - 1);
        let bytes = stride
            .checked_mul(u64::try_from(slots).map_err(|_| "PAYLOAD_CAPACITY")?)
            .ok_or("PAYLOAD_CAPACITY")?;
        usize::try_from(bytes).map_err(|_| "PAYLOAD_CAPACITY")?;
        let mut leases = Vec::new();
        leases
            .try_reserve_exact(slots)
            .map_err(|_| "PAYLOAD_CAPACITY")?;
        for _ in 0..slots {
            leases.push(PayloadLease::new(world, capacity, jobs)?);
        }
        Ok(Self {
            leases,
            stride,
            bytes,
            failed: false,
        })
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
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
    fn lease(&mut self, bank: PayloadBank) -> Result<&mut PayloadLease> {
        let lease = self.leases.get_mut(bank.index).ok_or("PAYLOAD_TICKET")?;
        lease.ticket(bank.key)?;
        Ok(lease)
    }
    /// A busy pool is ordinary admission unavailability, not overflow. Do not
    /// acknowledge a ticket until this returns a reserved physical bank.
    pub fn reserve(&mut self, key: TicketKey, bytes: u64) -> Result<Option<PayloadBank>> {
        self.apply(|s| {
            if s.leases
                .iter()
                .any(|l| l.active.is_some_and(|k| k.epoch == key.epoch))
            {
                return Err("PAYLOAD_DUPLICATE_TICKET".into());
            }
            if bytes > s.leases[0].capacity {
                return Err("PAYLOAD_BYTE_CAPACITY".into());
            }
            let Some(index) = s.leases.iter().position(|l| l.active.is_none()) else {
                return Ok(None);
            };
            s.leases[index].reserve(key, bytes)?;
            Ok(Some(PayloadBank { index, key }))
        })
    }
    pub fn offset(&mut self, bank: PayloadBank) -> Result<u64> {
        self.apply(|s| {
            s.lease(bank)?;
            Ok(bank.index as u64 * s.stride)
        })
    }
    pub fn consumer(&mut self, bank: PayloadBank) -> Result<BankConsumer> {
        self.apply(|s| {
            Ok(BankConsumer {
                bank,
                consumer: s.lease(bank)?.consumer(bank.key)?,
            })
        })
    }
    pub fn complete(&mut self, consumer: BankConsumer) -> Result<()> {
        self.apply(|s| s.lease(consumer.bank)?.complete(consumer.consumer))
    }
    pub fn seal(&mut self, bank: PayloadBank) -> Result<()> {
        self.apply(|s| s.lease(bank)?.seal(bank.key))
    }
    pub fn drained(&mut self, bank: PayloadBank) -> Result<bool> {
        self.apply(|s| s.lease(bank)?.drained(bank.key))
    }
    /// Caller additionally proves native transfer completion before release.
    pub fn retire(&mut self, bank: PayloadBank) -> Result<()> {
        self.apply(|s| s.lease(bank)?.retire(bank.key))
    }
}

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
