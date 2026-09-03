//! Deterministic CPU contract for a macro-lookahead owner.
//!
//! Production CUDA uses sorted runs and scans; this model specifies the same
//! irreversible point: offers are provisional, only `settle` publishes a layer.
use mgbfs_core::{hash::Hash128, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CandidateKey {
    pub source_depth: u32,
    pub weight: u32,
    pub source_rank: u32,
    pub source_batch: u64,
    pub ordinal: u32,
}

impl CandidateKey {
    pub fn new(
        source_depth: u32,
        weight: u32,
        source_rank: u32,
        source_batch: u64,
        ordinal: u32,
    ) -> Self {
        Self {
            source_depth,
            weight,
            source_rank,
            source_batch,
            ordinal,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FutureOffer {
    pub target_depth: u32,
    pub hash: Hash128,
    pub state_ref: u64,
    pub key: CandidateKey,
}

impl FutureOffer {
    pub fn new(target_depth: u32, hash: Hash128, state_ref: u64, key: CandidateKey) -> Self {
        Self {
            target_depth,
            hash,
            state_ref,
            key,
        }
    }
}

pub struct MacroOwner {
    macro_depth: u32,
    future_capacity: usize,
    settled_capacity: usize,
    settled_depth: Option<u32>,
    history: BTreeMap<u32, BTreeSet<Hash128>>,
    future: BTreeMap<u32, Vec<FutureOffer>>,
}

impl MacroOwner {
    pub fn new(
        macro_depth: u32,
        future_capacity_per_depth: usize,
        settled_capacity_per_depth: usize,
    ) -> Result<Self> {
        if macro_depth == 0 || future_capacity_per_depth == 0 || settled_capacity_per_depth == 0 {
            return Err("MACRO_OWNER_CONFIG".into());
        }
        Ok(Self {
            macro_depth,
            future_capacity: future_capacity_per_depth,
            settled_capacity: settled_capacity_per_depth,
            settled_depth: None,
            history: BTreeMap::new(),
            future: BTreeMap::new(),
        })
    }

    pub fn seed<I>(&mut self, depth: u32, hashes: I) -> Result<()>
    where
        I: IntoIterator<Item = Hash128>,
    {
        if self.settled_depth.is_some() || !self.future.is_empty() {
            return Err("MACRO_OWNER_ALREADY_STARTED".into());
        }
        let layer: BTreeSet<_> = hashes.into_iter().collect();
        if layer.len() > self.settled_capacity {
            return Err("MACRO_SETTLED_CAPACITY".into());
        }
        self.history.insert(depth, layer);
        self.settled_depth = Some(depth);
        Ok(())
    }

    pub fn offer(&mut self, offer: FutureOffer) -> Result<()> {
        let current = self.settled_depth.ok_or("MACRO_OWNER_NOT_SEEDED")?;
        let maximum = current
            .checked_add(self.macro_depth)
            .ok_or("MACRO_DEPTH_OVERFLOW")?;
        if offer.target_depth <= current || offer.target_depth > maximum {
            return Err("MACRO_FUTURE_DEPTH".into());
        }
        if offer.key.weight == 0
            || offer.key.weight > self.macro_depth
            || offer.key.source_depth.checked_add(offer.key.weight) != Some(offer.target_depth)
        {
            return Err("MACRO_OFFER_IDENTITY".into());
        }
        let old_len = self.future.get(&offer.target_depth).map_or(0, Vec::len);
        if old_len >= self.future_capacity {
            return Err("MACRO_FUTURE_CAPACITY".into());
        }
        self.future
            .entry(offer.target_depth)
            .or_default()
            .push(offer);
        Ok(())
    }

    pub fn settle(&mut self, depth: u32) -> Result<Vec<(Hash128, u64)>> {
        let current = self.settled_depth.ok_or("MACRO_OWNER_NOT_SEEDED")?;
        if current.checked_add(1) != Some(depth) {
            return Err("MACRO_SETTLE_ORDER".into());
        }
        // Preview from the immutable provisional run. Publication, slot release
        // and history rotation happen only after every capacity check passes.
        let mut candidates = self.future.get(&depth).cloned().unwrap_or_default();
        candidates.sort_by_key(|candidate| (candidate.hash, candidate.key));
        let seen: BTreeSet<Hash128> = self
            .history
            .values()
            .flat_map(|layer| layer.iter().copied())
            .collect();
        let mut accepted = Vec::new();
        let mut last = None;
        for candidate in candidates {
            if last == Some(candidate.hash) || seen.contains(&candidate.hash) {
                continue;
            }
            last = Some(candidate.hash);
            accepted.push((candidate.hash, candidate.state_ref));
        }
        if accepted.len() > self.settled_capacity {
            return Err("MACRO_SETTLED_CAPACITY".into());
        }
        self.future.remove(&depth);
        self.history
            .insert(depth, accepted.iter().map(|entry| entry.0).collect());
        self.settled_depth = Some(depth);
        let retain_from =
            depth.saturating_sub(self.macro_depth.saturating_mul(2).saturating_sub(1));
        self.history
            .retain(|layer_depth, _| *layer_depth >= retain_from);
        Ok(accepted)
    }
}
