use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixGroup {
    pub schema: u32,
    pub rows: usize,
    pub cols: usize,
    pub modulus: u16,
    pub start: Vec<u8>,
    pub generators: Vec<Vec<u8>>,
    pub inverse_map: Vec<usize>,
    pub expected_max_unique_states: u64,
}

impl MatrixGroup {
    pub fn validate(&self) -> Result<()> {
        if self.schema != 1 {
            return Err("MATRIX_MANIFEST_SCHEMA".into());
        }
        let n = self.rows;
        let bytes = n.checked_mul(self.cols).ok_or("STATE_WIDTH")?;
        if n == 0 || n != self.cols || bytes > 33025 || self.start.len() != bytes {
            return Err("STATE_SHAPE".into());
        }
        if !(2..=256).contains(&self.modulus) || self.expected_max_unique_states == 0 {
            return Err("MODULUS_OR_CAPACITY".into());
        }
        if self.start.iter().any(|&x| x as u16 >= self.modulus) {
            return Err("NONCANONICAL_STATE".into());
        }
        if self.generators.is_empty()
            || self.generators.len() > u16::MAX as usize
            || self.inverse_map.len() != self.generators.len()
        {
            return Err("GENERATOR_COUNT".into());
        }
        for g in &self.generators {
            if g.len() != bytes || g.iter().any(|&x| x as u16 >= self.modulus) {
                return Err("GENERATOR_SHAPE_OR_RANGE".into());
            }
        }
        if n as u64 * (self.modulus as u64 - 1).pow(2) > i32::MAX as u64 {
            return Err("MATRIX_ACCUMULATOR_BOUND".into());
        }
        let eye = identity(n);
        for (i, &j) in self.inverse_map.iter().enumerate() {
            if j >= self.generators.len() || self.inverse_map[j] != i {
                return Err("INVERSE_MAP".into());
            }
            if multiply(&self.generators[i], &self.generators[j], n, self.modulus) != eye {
                return Err("NOT_INVERSE_CLOSED".into());
            }
        }
        let mut remaining = self.modulus;
        for p in 2..=self.modulus {
            if remaining % p == 0 {
                if !invertible_mod_prime(&self.start, n, p) {
                    return Err("SINGULAR_START".into());
                }
                while remaining % p == 0 {
                    remaining /= p;
                }
            }
        }
        Ok(())
    }
    pub fn successor(&self, state: &[u8], move_id: usize) -> Result<Vec<u8>> {
        if state.len() != self.start.len() || state.iter().any(|&x| x as u16 >= self.modulus) {
            return Err("STATE_WIDTH_OR_RANGE".into());
        }
        let g = self.generators.get(move_id).ok_or("MOVE_ID")?;
        Ok(multiply(g, state, self.rows, self.modulus))
    }
    pub fn unitriangular(n: usize, modulus: u16) -> Result<Self> {
        if n < 2 || n.checked_mul(n).unwrap_or(usize::MAX) > 33025 || !(2..=256).contains(&modulus)
        {
            return Err("STATE_SHAPE_OR_MODULUS".into());
        }
        let expected_max_unique_states = (modulus as u64)
            .checked_pow((n * (n - 1) / 2) as u32)
            .ok_or("CAPACITY_OVERFLOW")?;
        let mut generators = Vec::new();
        for delta in [1, modulus - 1] {
            for i in 0..n - 1 {
                let mut g = identity(n);
                g[i * n + i + 1] = delta as u8;
                generators.push(g);
            }
        }
        let inverse_map = (0..2 * (n - 1))
            .map(|i| (i + n - 1) % (2 * (n - 1)))
            .collect();
        let group = Self {
            schema: 1,
            rows: n,
            cols: n,
            modulus,
            start: identity(n),
            generators,
            inverse_map,
            expected_max_unique_states,
        };
        group.validate()?;
        Ok(group)
    }
    /// Matrix realization of S_n using the inverse-closed set consisting of
    /// an n-cycle, its inverse, and the transposition (0 1).
    pub fn symmetric_permutation_matrices(n: usize) -> Result<Self> {
        if n < 2 || n.checked_mul(n).unwrap_or(usize::MAX) > 33025 {
            return Err("SYMMETRIC_DEGREE".into());
        }
        let expected_max_unique_states = (2..=n).try_fold(1u64, |value, factor| {
            value
                .checked_mul(factor as u64)
                .ok_or("SYMMETRIC_ORDER_OVERFLOW")
        })?;
        let cycle: Vec<_> = (0..n).map(|i| (i + 1) % n).collect();
        let inverse_cycle: Vec<_> = (0..n).map(|i| (i + n - 1) % n).collect();
        let mut transposition: Vec<_> = (0..n).collect();
        transposition.swap(0, 1);
        let permutation_matrix = |permutation: &[usize]| {
            let mut matrix = vec![0u8; n * n];
            for (row, &column) in permutation.iter().enumerate() {
                matrix[row * n + column] = 1;
            }
            matrix
        };
        let group = Self {
            schema: 1,
            rows: n,
            cols: n,
            modulus: 2,
            start: identity(n),
            generators: vec![
                permutation_matrix(&cycle),
                permutation_matrix(&inverse_cycle),
                permutation_matrix(&transposition),
            ],
            inverse_map: vec![1, 0, 2],
            expected_max_unique_states,
        };
        group.validate()?;
        Ok(group)
    }
    /// Independent exact oracle: full canonical state bytes, never hashes.
    pub fn exact_layers(&self, capacity: usize) -> Result<Vec<Vec<Vec<u8>>>> {
        self.validate()?;
        if capacity == 0 {
            return Err("ORACLE_CAPACITY".into());
        }
        let mut seen = BTreeSet::from([self.start.clone()]);
        let mut layers = vec![vec![self.start.clone()]];
        loop {
            let mut next = BTreeSet::new();
            for parent in layers.last().unwrap() {
                for m in 0..self.generators.len() {
                    let child = self.successor(parent, m)?;
                    if !seen.contains(&child) {
                        next.insert(child);
                    }
                    if seen.len() + next.len() > capacity {
                        return Err("ORACLE_CAPACITY".into());
                    }
                }
            }
            if next.is_empty() {
                return Ok(layers);
            }
            seen.extend(next.iter().cloned());
            layers.push(next.into_iter().collect());
        }
    }

    /// Exact small-graph oracle for the weighted macro-transition runtime.
    /// Future buckets are provisional until their numerical depth is settled.
    pub fn exact_layers_with_macros(
        &self,
        macros: &crate::macro_generators::MacroGeneratorSet,
        capacity: usize,
    ) -> Result<Vec<Vec<Vec<u8>>>> {
        self.validate()?;
        if capacity == 0 || macros.transitions.is_empty() {
            return Err("ORACLE_CAPACITY_OR_MACROS".into());
        }
        let mut settled = BTreeSet::new();
        let mut pending = BTreeMap::<usize, BTreeSet<Vec<u8>>>::new();
        pending.entry(0).or_default().insert(self.start.clone());
        let mut layers = Vec::new();
        while let Some((&depth, _)) = pending.first_key_value() {
            let candidates = pending.remove(&depth).unwrap();
            let layer: Vec<_> = candidates
                .into_iter()
                .filter(|state| !settled.contains(state))
                .collect();
            if layer.is_empty() {
                continue;
            }
            if settled
                .len()
                .checked_add(layer.len())
                .ok_or("ORACLE_CAPACITY")?
                > capacity
            {
                return Err("ORACLE_CAPACITY".into());
            }
            while layers.len() < depth {
                layers.push(Vec::new());
            }
            for state in &layer {
                settled.insert(state.clone());
                for transition in &macros.transitions {
                    let target = depth
                        .checked_add(transition.weight as usize)
                        .ok_or("DEPTH_OVERFLOW")?;
                    let child = multiply(&transition.matrix, state, self.rows, self.modulus);
                    if !settled.contains(&child) {
                        pending.entry(target).or_default().insert(child);
                    }
                }
            }
            layers.push(layer);
        }
        Ok(layers)
    }
}

pub type MatrixGroupManifestV1 = MatrixGroup;

fn identity(n: usize) -> Vec<u8> {
    let mut a = vec![0; n * n];
    for i in 0..n {
        a[i * n + i] = 1;
    }
    a
}

fn multiply(a: &[u8], b: &[u8], n: usize, modulus: u16) -> Vec<u8> {
    let mut c = vec![0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0u64;
            for k in 0..n {
                sum += a[i * n + k] as u64 * b[k * n + j] as u64;
            }
            c[i * n + j] = (sum % modulus as u64) as u8;
        }
    }
    c
}

fn invertible_mod_prime(bytes: &[u8], n: usize, p: u16) -> bool {
    let p = p as u32;
    let mut a: Vec<u32> = bytes.iter().map(|&x| x as u32 % p).collect();
    for col in 0..n {
        let Some(pivot) = (col..n).find(|&r| a[r * n + col] != 0) else {
            return false;
        };
        for j in 0..n {
            a.swap(col * n + j, pivot * n + j);
        }
        let inv = (1..p).find(|&x| x * a[col * n + col] % p == 1).unwrap();
        for j in col..n {
            a[col * n + j] = a[col * n + j] * inv % p;
        }
        for r in col + 1..n {
            let factor = a[r * n + col];
            for j in col..n {
                a[r * n + j] = (a[r * n + j] + p - factor * a[col * n + j] % p) % p;
            }
        }
    }
    true
}
