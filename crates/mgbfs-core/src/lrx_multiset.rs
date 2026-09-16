use crate::Result;
use std::collections::BTreeSet;

/// L/R cyclic position shifts and X swapping positions 0/1. This is a
/// Schreier graph of words, not a group of invertible state matrices.
pub struct LrxMultiset { start: Vec<u8>, order: u64 }
impl LrxMultiset {
    pub fn from_label(label: &str) -> Result<Self> {
        let (n,r) = label.strip_prefix("lrx").and_then(|s| s.split_once('r'))
            .ok_or("LRX_MULTISET_LABEL")?;
        Self::new(n.parse().map_err(|_| "LRX_MULTISET_LABEL")?,
                  r.parse().map_err(|_| "LRX_MULTISET_LABEL")?)
    }
    pub fn label(&self) -> String {
        let repeated = self.start.iter().rev().take_while(|x| **x == *self.start.last().unwrap()).count();
        format!("lrx{}r{repeated}", self.start.len())
    }
    /// Only the position action is represented by invertible matrices. The
    /// multiset word remains a separate start, never a singular MatrixGroup.
    pub fn position_group(&self) -> Result<crate::matrix::MatrixGroup> {
        crate::matrix::MatrixGroup::symmetric_permutation_matrices(self.start.len())
    }
    pub fn new(n: usize, repeated: usize) -> Result<Self> {
        if !(2..=255).contains(&n) || repeated == 0 || repeated > n {
            return Err("LRX_MULTISET_SHAPE".into());
        }
        let order = ((repeated + 1)..=n).try_fold(1u64, |a,b|
            a.checked_mul(b as u64).ok_or("LRX_MULTISET_ORDER_OVERFLOW"))?;
        let start = (0..n).map(|i| i.min(n-repeated) as u8).collect();
        Ok(Self { start, order })
    }
    pub fn start(&self) -> &[u8] { &self.start }
    pub fn order(&self) -> u64 { self.order }
    pub fn successor(&self, state: &[u8], generator: usize) -> Result<Vec<u8>> {
        let mut canonical = state.to_vec();
        canonical.sort_unstable();
        if canonical != self.start { return Err("LRX_MULTISET_STATE".into()); }
        let mut child = state.to_vec();
        match generator {
            0 => child.rotate_left(1),
            1 => child.rotate_right(1),
            2 => child.swap(0,1),
            _ => return Err("LRX_MULTISET_GENERATOR".into()),
        }
        Ok(child)
    }
    /// Independent bounded CPU oracle using full words, never hashes.
    pub fn exact_layers(&self, capacity: usize) -> Result<Vec<Vec<Vec<u8>>>> {
        if capacity == 0 { return Err("ORACLE_CAPACITY".into()); }
        let mut seen = BTreeSet::from([self.start.clone()]);
        let mut layers = vec![vec![self.start.clone()]];
        loop {
            let mut next = BTreeSet::new();
            for state in layers.last().unwrap() {
                for generator in 0..3 {
                    let child = self.successor(state, generator)?;
                    if !seen.contains(&child) { next.insert(child); }
                    if next.len() > capacity - seen.len() { return Err("ORACLE_CAPACITY".into()); }
                }
            }
            if next.is_empty() { return Ok(layers); }
            seen.extend(next.iter().cloned());
            layers.push(next.into_iter().collect());
        }
    }
}
