//! Admission for the explicitly fixed, suballocating library pool.
use crate::Result;
use serde::{Deserialize, Serialize};

/// Fixed candidate conversion buffer: four u32 hash planes and source ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateSoaLayout {
    pub plane_stride_bytes: u64,
    pub allocation_bytes: u64,
}

impl CandidateSoaLayout {
    pub fn plan(capacity: u64) -> Result<Self> {
        if capacity == 0 || capacity > i32::MAX as u64 {
            return Err("LIBRARY_SOA_CAPACITY".into());
        }
        // The signed 32-bit row bound proves both products fit u64.
        let plane_stride_bytes = (capacity * 4 + 255) & !255;
        Ok(Self {
            plane_stride_bytes,
            allocation_bytes: plane_stride_bytes * 5,
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryMemoryBudget {
    /// Entire initial == maximum RMM pool reservation, including unused space.
    pub pool_bytes: u64,
    /// Other allocations still to be made, not memory already excluded from free.
    pub fixed_device_bytes: u64,
    pub untouched_reserve_bytes: u64,
    pub free_after_warmup_bytes: u64,
}

impl LibraryMemoryBudget {
    pub fn validate(self) -> Result<u64> {
        if self.pool_bytes == 0 || self.pool_bytes % 256 != 0 {
            return Err("LIBRARY_POOL_ALIGNMENT".into());
        }
        if self.untouched_reserve_bytes < 1 << 30 {
            return Err("LIBRARY_T4_RESERVE".into());
        }
        let allocated = self
            .pool_bytes
            .checked_add(self.fixed_device_bytes)
            .ok_or("BYTE_OVERFLOW")?;
        let required = allocated
            .checked_add(self.untouched_reserve_bytes)
            .ok_or("BYTE_OVERFLOW")?;
        if required > self.free_after_warmup_bytes {
            return Err(format!(
                "LIBRARY_CAPACITY: required={required} free={}",
                self.free_after_warmup_bytes
            ));
        }
        Ok(allocated)
    }
}
