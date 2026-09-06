//! Bounded host ticket admission. TCP/GPU dispatcher integration is separate.
use crate::control_wire::Plane;
use mgbfs_core::Result;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TicketKey {
    pub depth: u64,
    pub epoch: u64,
    pub source: u32,
    pub plane: Plane,
    pub generation: u64,
}
pub struct ScatterAdmission {
    ranges: Vec<(u64, u64)>,
    ack: Vec<bool>,
    key: Option<TicketKey>,
    retired: Option<TicketKey>,
    launched: bool,
    failed: bool,
}
impl ScatterAdmission {
    /// Invoke only after transfer/consumer leases have drained. This guard does
    /// not observe CUDA events; the dispatcher owns that proof.
    pub fn retire(&mut self, key: TicketKey) -> Result<()> {
        self.apply(|s| {
            if s.key != Some(key) || !s.launched {
                return Err("ADMISSION_NOT_LAUNCHED".into());
            }
            s.retired = s.key.take();
            s.ack.fill(false);
            s.launched = false;
            Ok(())
        })
    }
    /// Allocate once per physical ticket slot during setup.
    pub fn new(world: u32) -> Result<Self> {
        if world == 0 {
            return Err("ADMISSION_WORLD".into());
        }
        let mut ranges = Vec::new();
        let mut ack = Vec::new();
        ranges
            .try_reserve_exact(world as usize)
            .map_err(|_| "ADMISSION_CAPACITY")?;
        ack.try_reserve_exact(world as usize)
            .map_err(|_| "ADMISSION_CAPACITY")?;
        ranges.resize(world as usize, (0, 0));
        ack.resize(world as usize, false);
        Ok(Self {
            ranges,
            ack,
            key: None,
            retired: None,
            launched: false,
            failed: false,
        })
    }
    fn apply<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.failed {
            return Err("ADMISSION_FAILED".into());
        }
        let result = f(self);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn prepare(
        &mut self,
        key: TicketKey,
        counts: &[u64],
        width: u64,
        capacity: u64,
    ) -> Result<()> {
        self.apply(|s| {
            if s.key.is_some()
                || s.retired.is_some_and(|old| {
                    key.epoch <= old.epoch
                        || key.generation <= old.generation
                        || key.depth < old.depth
                })
                || counts.len() != s.ranges.len()
                || key.source as usize >= counts.len()
                || key.plane == Plane::None
                || width == 0
            {
                return Err("ADMISSION_TICKET".into());
            }
            let mut total = 0u64;
            for (range, &count) in s.ranges.iter_mut().zip(counts) {
                let bytes = count.checked_mul(width).ok_or("ADMISSION_BYTE_OVERFLOW")?;
                *range = (total, bytes);
                total = total.checked_add(bytes).ok_or("ADMISSION_BYTE_OVERFLOW")?;
            }
            if total > capacity {
                return Err("ADMISSION_SEND_CAPACITY".into());
            }
            s.key = Some(key);
            Ok(())
        })
    }
    pub fn range(&self, rank: u32) -> Result<(u64, u64)> {
        if self.failed || self.key.is_none() {
            return Err("ADMISSION_NOT_PREPARED".into());
        }
        self.ranges
            .get(rank as usize)
            .copied()
            .ok_or_else(|| "ADMISSION_RANK".into())
    }
    /// Call only after the identified rank has reserved and validated its slot.
    /// This method does not perform that reservation or authenticate TCP messages.
    pub fn admit(&mut self, key: TicketKey, rank: u32, capacity: u64) -> Result<()> {
        self.apply(|s| {
            if s.key != Some(key) || s.launched {
                return Err("ADMISSION_STALE".into());
            }
            let ack = s.ack.get_mut(rank as usize).ok_or("ADMISSION_RANK")?;
            if *ack {
                return Err("ADMISSION_DUPLICATE".into());
            }
            if rank != key.source && s.ranges[rank as usize].1 > capacity {
                return Err("ADMISSION_RECV_CAPACITY".into());
            }
            *ack = true;
            Ok(())
        })
    }
    /// Caller must additionally enforce global epoch issue order across slots.
    pub fn launch(&mut self, key: TicketKey) -> Result<bool> {
        self.apply(|s| {
            if s.key != Some(key) || s.launched {
                return Err("ADMISSION_STALE".into());
            }
            if !s.ack.iter().all(|&a| a) {
                return Ok(false);
            }
            s.launched = true;
            Ok(true)
        })
    }
}
