use mgbfs_core::{hash::Hash128, Result};
use std::collections::BTreeSet;
pub struct OwnerModel {
    old: BTreeSet<Hash128>,
    accepted: BTreeSet<Hash128>,
    capacity: usize,
    last_epoch: Option<u64>,
}
impl OwnerModel {
    pub fn new(prev: Vec<Hash128>, curr: Vec<Hash128>, capacity: usize) -> Self {
        Self {
            old: prev.into_iter().chain(curr).collect(),
            accepted: BTreeSet::new(),
            capacity,
            last_epoch: None,
        }
    }
    pub fn commit(&mut self, epoch: u64, incoming: &[Hash128]) -> Result<Vec<Hash128>> {
        if self.last_epoch.is_some_and(|last| epoch <= last) {
            return Err("OWNER_EPOCH_ORDER".into());
        }
        let survivors: BTreeSet<_> = incoming
            .iter()
            .copied()
            .filter(|h| !self.old.contains(h) && !self.accepted.contains(h))
            .collect();
        if survivors.len() > self.capacity - self.accepted.len() {
            return Err("OWNER_BUCKET_CAPACITY".into());
        }
        self.accepted.extend(survivors.iter().copied());
        self.last_epoch = Some(epoch);
        Ok(survivors.into_iter().collect())
    }
    pub fn accepted(&self) -> Vec<Hash128> {
        self.accepted.iter().copied().collect()
    }
}
