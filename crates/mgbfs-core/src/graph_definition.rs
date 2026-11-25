//! Lossless action contract for CayleyPy permutations and int64 matrix states.
use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixGeneratorV2 {
    pub matrix: Vec<i64>,
    pub modulo: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GraphAction {
    Permutation {
        degree: u32,
        generators: Vec<Vec<u32>>,
    },
    Matrix {
        rows: u32,
        cols: u32,
        generators: Vec<MatrixGeneratorV2>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphDefinitionV2 {
    pub schema: u32,
    pub name: String,
    pub generator_names: Vec<String>,
    pub action: GraphAction,
    pub start: Vec<i64>,
    pub expected_max_unique_states: Option<u64>,
}
// Gauss-Jordan over Z/(2^k). An invertible matrix always has an odd
// pivot in each remaining column. Wrapping arithmetic is reduction for k=64.
fn reverse_power_of_two(matrix: &[i64], n: usize, modulo: u32) -> Option<Vec<i64>> {
    let mask = if modulo == 0 {
        u64::MAX
    } else {
        u64::from(modulo - 1)
    };
    let width = n.checked_mul(2)?;
    let mut rows = vec![0u64; n.checked_mul(width)?];
    for i in 0..n {
        for j in 0..n {
            rows[i * width + j] = matrix[i * n + j] as u64 & mask;
        }
        rows[i * width + n + i] = 1;
    }
    for col in 0..n {
        let pivot = (col..n).find(|&r| rows[r * width + col] & 1 != 0)?;
        for j in 0..width {
            rows.swap(col * width + j, pivot * width + j);
        }
        let value = rows[col * width + col];
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(value.wrapping_mul(inv)));
        }
        for j in 0..width {
            let v = &mut rows[col * width + j];
            *v = v.wrapping_mul(inv) & mask;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let scale = rows[r * width + col];
            for j in 0..width {
                let term = rows[col * width + j].wrapping_mul(scale);
                let v = &mut rows[r * width + j];
                *v = v.wrapping_sub(term) & mask;
            }
        }
    }
    Some(
        rows.chunks_exact(width)
            .flat_map(|row| row[n..].iter().map(|&v| v as i64))
            .collect(),
    )
}

fn inverse_unit(value: i64, modulus: i64) -> i64 {
    let (mut r, mut next_r) = (modulus, value.rem_euclid(modulus));
    let (mut coefficient, mut next_coefficient) = (0, 1);
    while next_r != 0 {
        let quotient = r / next_r;
        (r, next_r) = (next_r, r - quotient * next_r);
        (coefficient, next_coefficient) =
            (next_coefficient, coefficient - quotient * next_coefficient);
    }
    debug_assert_eq!(r, 1);
    coefficient.rem_euclid(modulus)
}

fn reverse_modular(matrix: &[i64], n: usize, modulus: i64) -> Option<Vec<i64>> {
    let width = n.checked_mul(2)?;
    let mut rows = vec![0; n.checked_mul(width)?];
    let mut inverse = vec![0; matrix.len()];
    let (mut remaining, mut combined, mut prime) = (modulus, 1, 2);
    // Factor into coprime prime powers. Within each component, a
    // pivot is a unit exactly when it is not divisible by the prime.
    while remaining > 1 {
        if prime * prime > remaining {
            prime = remaining;
        }
        if remaining % prime != 0 {
            prime += 1;
            continue;
        }
        let mut power = 1;
        while remaining % prime == 0 {
            remaining /= prime;
            power *= prime;
        }
        for i in 0..n {
            let row = &mut rows[i * width..(i + 1) * width];
            for (j, entry) in row[..n].iter_mut().enumerate() {
                *entry = matrix[i * n + j].rem_euclid(power);
            }
            row[n..].fill(0);
            row[n + i] = 1;
        }
        for col in 0..n {
            let pivot = (col..n).find(|&r| rows[r * width + col] % prime != 0)?;
            for j in 0..width {
                rows.swap(col * width + j, pivot * width + j);
            }
            let scale = inverse_unit(rows[col * width + col], power);
            for j in 0..width {
                let v = &mut rows[col * width + j];
                *v = (*v * scale).rem_euclid(power);
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let scale = rows[r * width + col];
                for j in 0..width {
                    let term = rows[col * width + j] * scale;
                    let v = &mut rows[r * width + j];
                    *v = (*v - term).rem_euclid(power);
                }
            }
        }
        let scale = inverse_unit(combined, power);
        for (row, out) in rows.chunks_exact(width).zip(inverse.chunks_exact_mut(n)) {
            for (&v, x) in row[n..].iter().zip(out) {
                *x += combined * ((v - *x) * scale).rem_euclid(power);
            }
        }
        combined *= power;
    }
    // Center coefficients to minimize the bound on the int64 dot product.
    for v in &mut inverse {
        if *v > modulus / 2 {
            *v -= modulus;
        }
    }
    Some(inverse)
}

impl GraphDefinitionV2 {
    pub fn generator_count(&self) -> usize {
        match &self.action {
            GraphAction::Permutation { generators, .. } => generators.len(),
            GraphAction::Matrix { generators, .. } => generators.len(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.schema != 2
            || self.generator_count() == 0
            || self.generator_names.len() != self.generator_count()
            || self.expected_max_unique_states == Some(0)
        {
            return Err("GRAPH_SCHEMA_OR_GENERATORS".into());
        }
        match &self.action {
            GraphAction::Permutation { degree, generators } => {
                let n = *degree as usize;
                if n == 0 || self.start.len() != n {
                    return Err("GRAPH_PERMUTATION_SHAPE".into());
                }
                for g in generators {
                    if g.len() != n {
                        return Err("GRAPH_PERMUTATION_GENERATOR".into());
                    }
                    let mut seen = vec![false; n];
                    for &i in g {
                        if i >= *degree || seen[i as usize] {
                            return Err("GRAPH_PERMUTATION_GENERATOR".into());
                        }
                        seen[i as usize] = true;
                    }
                }
            }
            GraphAction::Matrix {
                rows,
                cols,
                generators,
            } => {
                let n = *rows as usize;
                let m = *cols as usize;
                let size = n.checked_mul(m).ok_or("GRAPH_MATRIX_SHAPE")?;
                if n == 0 || m == 0 || size > u32::MAX as usize || self.start.len() != size {
                    return Err("GRAPH_MATRIX_SHAPE".into());
                }
                let square = n.checked_mul(n).ok_or("GRAPH_MATRIX_SHAPE")?;
                for g in generators {
                    if g.matrix.len() != square
                        || (g.modulo != 0 && !(2..=1u32 << 31).contains(&g.modulo))
                    {
                        return Err("GRAPH_MATRIX_GENERATOR_OR_MODULUS".into());
                    }
                }
            }
        }
        Ok(())
    }
    pub fn semantic_digest(&self) -> Result<[u8; 32]> {
        self.validate()?;
        // Value maps serialize in sorted key order, matching Python canonical JSON.
        let v = serde_json::json!({"schema":self.schema,"action":self.action,"start":self.start});
        Ok(Sha256::digest(serde_json::to_vec(&v).map_err(|e| e.to_string())?).into())
    }
    fn multiply(mx: &[i64], state: &[i64], n: usize, m: usize, modulo: u32) -> Vec<i64> {
        let mut out = vec![0; n * m];
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0i64;
                for k in 0..n {
                    sum = sum.wrapping_add(mx[i * n + k].wrapping_mul(state[k * m + j]));
                }
                out[i * m + j] = if modulo == 0 {
                    sum
                } else {
                    sum.rem_euclid(i64::from(modulo))
                };
            }
        }
        out
    }
    pub fn successor(&self, state: &[i64], generator: usize) -> Result<Vec<i64>> {
        self.validate()?;
        if state.len() != self.start.len() || generator >= self.generator_count() {
            return Err("GRAPH_SUCCESSOR_ARGUMENT".into());
        }
        Ok(match &self.action {
            GraphAction::Permutation { generators, .. } => generators[generator]
                .iter()
                .map(|&i| state[i as usize])
                .collect(),
            GraphAction::Matrix {
                rows,
                cols,
                generators,
            } => {
                let g = &generators[generator];
                Self::multiply(&g.matrix, state, *rows as usize, *cols as usize, g.modulo)
            }
        })
    }
    pub fn inverse_closed(&self) -> Result<bool> {
        self.validate()?;
        Ok(match &self.action {
            GraphAction::Permutation { degree, generators } => {
                let available: BTreeSet<Vec<u32>> = generators.iter().cloned().collect();
                generators.iter().all(|g| {
                    let mut inverse = vec![0; *degree as usize];
                    for (i, &x) in g.iter().enumerate() {
                        inverse[x as usize] = i as u32;
                    }
                    available.contains(&inverse)
                })
            }
            GraphAction::Matrix {
                rows, generators, ..
            } => {
                let n = *rows as usize;
                let modulus = generators[0].modulo;
                if generators.iter().any(|g| g.modulo != modulus)
                    || (modulus != 0
                        && self.start.iter().any(|&x| x < 0 || x >= i64::from(modulus)))
                {
                    return Ok(false);
                }
                // CUDA uses wrapping int64 arithmetic before modular reduction. For
                // non-power-of-two moduli, wrapping is not a homomorphism modulo m.
                // Admit compact history only when every canonical-state dot product
                // is overflow-free; otherwise conservatively retain all history.
                if modulus != 0 && !modulus.is_power_of_two() {
                    let bound = i128::from(modulus - 1);
                    if generators.iter().any(|g| {
                        g.matrix.chunks(n).any(|row| {
                            row.iter()
                                .map(|&v| i128::from(v).abs() * bound)
                                .sum::<i128>()
                                > i128::from(i64::MAX)
                        })
                    }) {
                        return Ok(false);
                    }
                }
                let eye: Vec<i64> = (0..n)
                    .flat_map(|i| (0..n).map(move |j| i64::from(i == j)))
                    .collect();
                let compose = |a: &[i64], b: &[i64]| -> Vec<i64> {
                    if modulus == 0 {
                        return Self::multiply(a, b, n, n, 0);
                    }
                    // Normalize the right operand, as an actual canonical modular state.
                    let canonical = b
                        .iter()
                        .map(|&x| x.rem_euclid(i64::from(modulus)))
                        .collect::<Vec<_>>();
                    Self::multiply(a, &canonical, n, n, modulus)
                };
                generators.iter().all(|g| {
                    generators.iter().any(|h| {
                        compose(&g.matrix, &h.matrix) == eye && compose(&h.matrix, &g.matrix) == eye
                    })
                })
            }
        })
    }
    /// Exact reverse edges in original generator-ID order.
    pub fn reversed_for_paths(&self, goals: &[Vec<i64>]) -> Result<Self> {
        self.validate()?;
        if goals.is_empty() || goals.iter().any(|goal| goal.len() != self.start.len()) {
            return Err("NATIVE_PATH_TARGETS".into());
        }
        let action = match &self.action {
            GraphAction::Permutation { degree, generators } => {
                let generators = generators
                    .iter()
                    .map(|g| {
                        let mut inverse = vec![0; *degree as usize];
                        for (i, &source) in g.iter().enumerate() {
                            inverse[source as usize] = i as u32;
                        }
                        inverse
                    })
                    .collect();
                GraphAction::Permutation {
                    degree: *degree,
                    generators,
                }
            }
            GraphAction::Matrix {
                rows,
                cols,
                generators,
            } => {
                let modulo = generators[0].modulo;
                if generators.iter().any(|g| g.modulo != modulo)
                    || (modulo != 0
                        && self
                            .start
                            .iter()
                            .chain(goals.iter().flatten())
                            .any(|&v| v < 0 || v >= i64::from(modulo)))
                {
                    return Err("PATH_MATRIX_REVERSE_UNSUPPORTED".into());
                }
                let n = *rows as usize;
                let identity = (0..n)
                    .flat_map(|i| (0..n).map(move |j| i64::from(i == j)))
                    .collect::<Vec<_>>();
                let product = |left: &MatrixGeneratorV2, right: &MatrixGeneratorV2| {
                    if modulo == 0 {
                        return Self::multiply(&left.matrix, &right.matrix, n, n, 0);
                    }
                    let canonical = right
                        .matrix
                        .iter()
                        .map(|&v| v.rem_euclid(i64::from(modulo)))
                        .collect::<Vec<_>>();
                    Self::multiply(&left.matrix, &canonical, n, n, modulo)
                };
                let binary_ring = modulo == 0 || modulo.is_power_of_two();
                let overflow_free = |matrix: &[i64]| {
                    let bound = i128::from(modulo.saturating_sub(1));
                    matrix.chunks_exact(n).all(|row| {
                        row.iter()
                            .map(|&v| i128::from(v).abs() * bound)
                            .sum::<i128>()
                            <= i128::from(i64::MAX)
                    })
                };
                let generators = generators
                    .iter()
                    .map(|g| {
                        if !binary_ring && !overflow_free(&g.matrix) {
                            return Err("PATH_MATRIX_REVERSE_UNSUPPORTED");
                        }
                        let matrix = if binary_ring {
                            reverse_power_of_two(&g.matrix, n, modulo)
                        } else {
                            reverse_modular(&g.matrix, n, i64::from(modulo))
                        }
                        .ok_or("PATH_MATRIX_REVERSE_UNSUPPORTED")?;
                        let inverse = MatrixGeneratorV2 { matrix, modulo };
                        if (!binary_ring && !overflow_free(&inverse.matrix))
                            || product(g, &inverse) != identity
                            || product(&inverse, g) != identity
                        {
                            return Err("PATH_MATRIX_REVERSE_UNSUPPORTED");
                        }
                        Ok(inverse)
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                GraphAction::Matrix {
                    rows: *rows,
                    cols: *cols,
                    generators,
                }
            }
        };
        // Keep the original start as the shared rank-map anchor.
        Ok(Self {
            action,
            ..self.clone()
        })
    }
    /// Bounded full-state CPU oracle, never a production compute fallback.
    pub fn exact_layers(&self, maximum_states: usize) -> Result<Vec<Vec<Vec<i64>>>> {
        self.validate()?;
        if maximum_states == 0 {
            return Err("ORACLE_CAPACITY".into());
        }
        let mut seen = BTreeSet::from([self.start.clone()]);
        let mut frontier = seen.clone();
        let mut layers = Vec::new();
        while !frontier.is_empty() {
            layers.push(frontier.iter().cloned().collect());
            let mut future = BTreeSet::new();
            for x in &frontier {
                for g in 0..self.generator_count() {
                    let y = self.successor(x, g)?;
                    if !seen.contains(&y) {
                        future.insert(y);
                    }
                    if future.len() > maximum_states - seen.len() {
                        return Err("ORACLE_CAPACITY".into());
                    }
                }
            }
            seen.extend(future.iter().cloned());
            frontier = future;
        }
        Ok(layers)
    }
}
