//! CPU oracle for one source batch's HASH_FIRST terminal obligations.
//! Not the transport implementation or a GPU fallback.
use crate::ring::StateRing;
use mgbfs_core::Result;
use std::collections::BTreeSet;
struct Owner {
    emitted: u64,
    accepted: Option<u64>,
    served: BTreeSet<u64>,
}
pub struct BatchReceipts {
    owners: Vec<Owner>,
    failed: bool,
}
impl BatchReceipts {
    pub fn new(emitted: &[u64]) -> Result<Self> {
        Ok(Self {
            owners: emitted
                .iter()
                .map(|&n| Owner {
                    emitted: n,
                    accepted: if n == 0 { Some(0) } else { None },
                    served: BTreeSet::new(),
                })
                .collect(),
            failed: false,
        })
    }
    pub fn receipt(&mut self, owner: usize, emitted: u64, accepted: u64) -> Result<()> {
        if self.failed {
            return Err("RECEIPT_BATCH_FAILED".into());
        }
        let valid = self.owners.get(owner).is_some_and(|o| {
            o.accepted.is_none()
                && o.emitted == emitted
                && accepted <= emitted
                && o.served.len() as u64 <= accepted
        });
        if !valid {
            self.failed = true;
            return Err("TERMINAL_RECEIPT_MISMATCH".into());
        }
        self.owners[owner].accepted = Some(accepted);
        Ok(())
    }
    /// Invoke at response SEND COMPLETION, not request arrival or enqueue.
    /// Request identity is scoped to this source batch and owner.
    pub fn served(&mut self, owner: usize, request: u64) -> Result<()> {
        if self.failed {
            return Err("RECEIPT_BATCH_FAILED".into());
        }
        let valid = self.owners.get(owner).is_some_and(|o| {
            !o.served.contains(&request)
                && (o.served.len() as u64) < o.accepted.unwrap_or(o.emitted)
        });
        if !valid {
            self.failed = true;
            return Err("RESPONSE_COMPLETION_MISMATCH".into());
        }
        self.owners[owner].served.insert(request);
        Ok(())
    }
    pub fn closed(&self) -> bool {
        !self.failed
            && self
                .owners
                .iter()
                .all(|o| o.accepted == Some(o.served.len() as u64))
    }
}

/// Source-side HASH_FIRST lifetime contract for one emitted parent batch.
/// The parent extent remains readable until every owner has published its
/// terminal receipt and every accepted materialization response has completed.
pub struct HashFirstLease {
    parent_extent: u64,
    receipts: BatchReceipts,
    origin_held: bool,
    closed: bool,
}

impl HashFirstLease {
    pub fn begin(ring: &mut StateRing, parent_extent: u64, emitted: &[u64]) -> Result<Self> {
        let receipts = BatchReceipts::new(emitted)?;
        let origin_held = emitted.iter().any(|&count| count != 0);
        if origin_held {
            ring.hold_origins(parent_extent)?;
        }
        Ok(Self {
            parent_extent,
            receipts,
            origin_held,
            closed: false,
        })
    }

    pub fn receipt(&mut self, owner: usize, emitted: u64, accepted: u64) -> Result<()> {
        self.receipts.receipt(owner, emitted, accepted)
    }

    pub fn served(&mut self, owner: usize, request: u64) -> Result<()> {
        self.receipts.served(owner, request)
    }

    /// Returns true exactly once, when this call closes the lease.
    pub fn try_close(&mut self, ring: &mut StateRing) -> Result<bool> {
        if self.receipts.failed {
            return Err("RECEIPT_BATCH_FAILED".into());
        }
        if self.closed || !self.receipts.closed() {
            return Ok(false);
        }
        if self.origin_held {
            ring.release_origins(self.parent_extent)?;
        }
        self.closed = true;
        Ok(true)
    }
}
