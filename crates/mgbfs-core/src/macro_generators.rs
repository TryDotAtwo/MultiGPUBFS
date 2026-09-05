use crate::{matrix::MatrixGroup, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MacroTransition {
    pub matrix: Vec<u8>,
    pub weight: u32,
    pub word: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MacroGeneratorSet {
    pub requested_depth: u32,
    pub effective_depth: u32,
    pub transitions: Vec<MacroTransition>,
}

impl MacroGeneratorSet {
    pub fn digest_v1(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"MACRO_GENERATORS_V1\0");
        digest.update(self.requested_depth.to_le_bytes());
        digest.update(self.effective_depth.to_le_bytes());
        digest.update((self.transitions.len() as u64).to_le_bytes());
        for transition in &self.transitions {
            digest.update(transition.weight.to_le_bytes());
            digest.update((transition.matrix.len() as u32).to_le_bytes());
            digest.update(&transition.matrix);
            digest.update((transition.word.len() as u32).to_le_bytes());
            for &movement in &transition.word {
                digest.update(movement.to_le_bytes());
            }
        }
        digest.finalize().into()
    }

    /// Enumerate the ball of transition matrices in shortlex order. The first
    /// discovery of a matrix is therefore its deterministic shortest word.
    pub fn compile(graph: &MatrixGroup, requested_depth: u32) -> Result<Self> {
        graph.validate()?;
        if requested_depth == 0 {
            return Err("MACRO_DEPTH_ZERO".into());
        }
        // Compile operators, not states reached from the requested BFS source.
        // Starting at A would produce G*A, then apply (G*A)*state at runtime.
        let mut identity = vec![0; graph.start.len()];
        for row in 0..graph.rows {
            identity[row * graph.cols + row] = 1;
        }
        let mut seen = BTreeMap::from([(identity.clone(), usize::MAX)]);
        let mut transitions = Vec::new();
        let mut frontier = vec![(identity, Vec::<u16>::new())];
        let mut effective_depth = 0;
        for depth in 1..=requested_depth {
            let mut next = Vec::new();
            for (matrix, word) in &frontier {
                for movement in 0..graph.generators.len() {
                    let product = graph.successor(matrix, movement)?;
                    if seen.contains_key(&product) {
                        continue;
                    }
                    if seen.len() as u64 >= graph.expected_max_unique_states {
                        return Err("MACRO_GENERATOR_CAPACITY".into());
                    }
                    if transitions.len() >= u16::MAX as usize {
                        return Err("MACRO_GENERATOR_ABI_CAPACITY".into());
                    }
                    let mut child_word = word.clone();
                    child_word.push(movement as u16);
                    let index = transitions.len();
                    seen.insert(product.clone(), index);
                    transitions.push(MacroTransition {
                        matrix: product.clone(),
                        weight: depth,
                        word: child_word.clone(),
                    });
                    next.push((product, child_word));
                }
            }
            if next.is_empty() {
                break;
            }
            effective_depth = depth;
            frontier = next;
        }
        Ok(Self {
            requested_depth,
            effective_depth,
            transitions,
        })
    }
}
