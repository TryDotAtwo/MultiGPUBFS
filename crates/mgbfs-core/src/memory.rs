use crate::Result;
use serde::{Deserialize, Serialize};

/// Owner-lane scratch only; library scratch is supplied by measured queries.
pub fn bounded_owner_ledger(i: u64, j: u64, k: u64, library: [u64; 3]) -> Result<AllocationLedger> {
    if i == 0 || j == 0 || k == 0 || i > i32::MAX as u64 {
        return Err("OWNER_JOB_SHAPE".into());
    }
    let mut ledger = AllocationLedger::new(u64::MAX, 0)?;
    for bank in 0..2 {
        for (field, bytes) in [("hash", 16), ("payload", 8), ("ordinal", 4)] {
            ledger.add(&format!("merge_{bank}_{field}"), i, bytes, 256)?;
        }
    }
    ledger.add("indices", i, 4, 256)?;
    for category in 0..4 {
        ledger.add(&format!("flags_{category}"), i, 1, 256)?;
    }
    ledger.add(
        "accepted_output",
        j.checked_mul(k).ok_or("BYTE_OVERFLOW")?,
        16,
        256,
    )?;
    ledger.add("bucket_descriptors", j, 64, 256)?;
    ledger.add("bucket_counts_offsets", j, 32, 256)?;
    ledger.add("job_control", 1, 64, 256)?;
    for (name, bytes) in ["merge_query", "select_query", "scan_query"]
        .into_iter()
        .zip(library)
    {
        ledger.add(name, bytes, 1, 256)?;
    }
    Ok(ledger)
}

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
