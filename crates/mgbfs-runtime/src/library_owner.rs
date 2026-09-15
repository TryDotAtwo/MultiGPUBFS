//! Host control for a single serialized library-owner shard, not a CPU dedup backend.
use mgbfs_core::Result;

#[derive(Clone, Copy)]
enum Phase {
    Idle,
    Compared { epoch: u64, survivors: u64 },
    Reserved { epoch: u64, end: u64 },
    Poisoned,
}

/// One gate per exclusively leased shard. Counts are host-visible control data;
/// hash/state arrays remain on the device. The caller must terminate the rank
/// group after any error, including a CUDA failure after `reserve`.
pub struct OwnerCommitGate {
    capacity: u64,
    accepted: u64,
    last_epoch: Option<u64>,
    phase: Phase,
}
impl OwnerCommitGate {
    /// Sticky external CUDA/FFI failure: never publish or reuse this shard.
    pub fn abort(&mut self) {
        self.phase = Phase::Poisoned;
    }

    /// Check before calling a library that can replace its borrowed result.
    pub fn check_compare(&mut self, epoch: u64) -> Result<()> {
        if !matches!(self.phase, Phase::Idle) || self.last_epoch.is_some_and(|last| epoch <= last) {
            return self.fail("LIBRARY_OWNER_ORDER");
        }
        Ok(())
    }

    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            accepted: 0,
            last_epoch: None,
            phase: Phase::Idle,
        }
    }

    /// Called after the library result/count is ready, before persistent writes.
    pub fn compared(&mut self, epoch: u64, candidates: u64, survivors: u64) -> Result<()> {
        self.check_compare(epoch)?;
        if survivors > candidates {
            return self.fail("LIBRARY_OWNER_COUNT");
        }
        if survivors > self.capacity - self.accepted {
            return self.fail("LIBRARY_OWNER_CAPACITY");
        }
        self.phase = Phase::Compared { epoch, survivors };
        Ok(())
    }

    /// `granted` is the actual StateRing/materialization/archive credit, never
    /// merely the size requested. Returned range authorizes the device commit.
    pub fn reserve(&mut self, epoch: u64, granted: u64) -> Result<std::ops::Range<u64>> {
        let Phase::Compared {
            epoch: expected,
            survivors,
        } = self.phase
        else {
            return self.fail("LIBRARY_OWNER_ORDER");
        };
        if epoch != expected {
            return self.fail("LIBRARY_OWNER_ORDER");
        }
        if granted < survivors {
            return self.fail("LIBRARY_OWNER_CREDIT");
        }
        // compared() proved that this addition cannot overflow.
        let end = self.accepted + survivors;
        self.phase = Phase::Reserved { epoch, end };
        Ok(self.accepted..end)
    }

    /// Call only after the corresponding device commit event completes
    /// successfully. A returned range alone is not publication evidence.
    pub fn completed(&mut self, epoch: u64) -> Result<()> {
        let Phase::Reserved {
            epoch: expected,
            end,
        } = self.phase
        else {
            return self.fail("LIBRARY_OWNER_ORDER");
        };
        if epoch != expected {
            return self.fail("LIBRARY_OWNER_ORDER");
        }
        self.accepted = end;
        self.last_epoch = Some(epoch);
        self.phase = Phase::Idle;
        Ok(())
    }

    pub fn accepted(&self) -> u64 {
        self.accepted
    }

    fn fail<T>(&mut self, code: &str) -> Result<T> {
        self.phase = Phase::Poisoned;
        Err(code.into())
    }
}
