use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allocation {
    pub name: String,
    pub offset: u64,
    pub payload_bytes: u64,
    pub reserved_bytes: u64,
}
pub struct AllocationLedger {
    limit: u64,
    total: u64,
    pub allocations: Vec<Allocation>,
}
impl AllocationLedger {
    pub fn new(budget: u64, reserve: u64) -> Result<Self> {
        Ok(Self {
            limit: budget.checked_sub(reserve).ok_or("VRAM_RESERVE")?,
            total: 0,
            allocations: Vec::new(),
        })
    }
    pub fn add(&mut self, name: &str, count: u64, stride: u64, alignment: u64) -> Result<u64> {
        if name.is_empty() || self.allocations.iter().any(|a| a.name == name) {
            return Err("ALLOCATION_NAME".into());
        }
        if !alignment.is_power_of_two() {
            return Err("ALLOCATION_ALIGNMENT".into());
        }
        let align = |n: u64| {
            n.checked_add(alignment - 1)
                .map(|x| x & !(alignment - 1))
                .ok_or("BYTE_OVERFLOW")
        };
        let payload_bytes = count.checked_mul(stride).ok_or("BYTE_OVERFLOW")?;
        let offset = align(self.total)?;
        let reserved_bytes = align(payload_bytes)?;
        let end = offset.checked_add(reserved_bytes).ok_or("BYTE_OVERFLOW")?;
        if end > self.limit {
            return Err("DEVICE_CAPACITY".into());
        }
        self.allocations.push(Allocation {
            name: name.into(),
            offset,
            payload_bytes,
            reserved_bytes,
        });
        self.total = end;
        Ok(offset)
    }
    pub fn total(&self) -> u64 {
        self.total
    }
}
