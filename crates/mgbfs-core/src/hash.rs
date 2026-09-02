use crate::Result;
use serde::{Deserialize, Serialize};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

pub const PRIME: u64 = 4_294_967_291;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Hash128(pub [u32; 4]);

impl Ord for Hash128 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}
impl PartialOrd for Hash128 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Hash128 {
    pub fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0; 16];
        for (dst, word) in out.chunks_exact_mut(4).zip(self.0) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        out
    }
    pub fn prefix(self, bits: u32) -> u64 {
        assert!(bits <= 64);
        if bits == 0 {
            return 0;
        }
        let hi = (self.0[3] as u64) << 32 | self.0[2] as u64;
        hi >> (64 - bits)
    }
}

pub struct GemmHash {
    pub coefficients: Vec<[u32; 4]>,
    pub offsets: [u32; 4],
}

impl GemmHash {
    pub fn from_seed(bytes: usize, seed: [u8; 16]) -> Result<Self> {
        if !(1..=33025).contains(&bytes) {
            return Err("HASH_ACCUMULATOR_BOUND".into());
        }
        let mut xof = Shake256::default();
        xof.update(b"MGBFS/GEMM_U8_P32X4/V1\0");
        xof.update(&(bytes as u32).to_le_bytes());
        xof.update(&seed);
        let mut reader = xof.finalize_xof();
        let mut sample = || loop {
            let mut b = [0; 4];
            reader.read(&mut b);
            let x = u32::from_le_bytes(b);
            if (x as u64) < PRIME {
                break x;
            }
        };
        let coefficients = (0..bytes)
            .map(|_| std::array::from_fn(|_| sample()))
            .collect();
        let offsets = std::array::from_fn(|_| sample());
        Ok(Self {
            coefficients,
            offsets,
        })
    }
    pub fn hash(&self, state: &[u8]) -> Result<Hash128> {
        self.validate()?;
        if state.len() != self.coefficients.len() {
            return Err("STATE_WIDTH".into());
        }
        let mut out = self.offsets.map(u64::from);
        for (&x, a) in state.iter().zip(&self.coefficients) {
            for j in 0..4 {
                out[j] = (out[j] + x as u64 * a[j] as u64) % PRIME;
            }
        }
        Ok(Hash128(out.map(|x| x as u32)))
    }
    pub fn limbs(&self) -> Vec<u8> {
        self.coefficients
            .iter()
            .flat_map(|row| row.iter().flat_map(|a| a.to_le_bytes()))
            .collect()
    }
    pub fn hash_from_partials(&self, partials: &[i32; 16]) -> Result<Hash128> {
        self.validate()?;
        let bound = self.coefficients.len() as i64 * 255 * 255;
        if partials.iter().any(|&s| s < 0 || s as i64 > bound) {
            return Err("HASH_PARTIAL_BOUND".into());
        }
        let mut out = [0; 4];
        for j in 0..4 {
            let sum = (0..4)
                .map(|l| (partials[4 * j + l] as u64) << (8 * l))
                .sum::<u64>();
            out[j] = ((sum + self.offsets[j] as u64) % PRIME) as u32;
        }
        Ok(Hash128(out))
    }
    fn validate(&self) -> Result<()> {
        if !(1..=33025).contains(&self.coefficients.len()) {
            return Err("HASH_ACCUMULATOR_BOUND".into());
        }
        if self
            .coefficients
            .iter()
            .flatten()
            .chain(self.offsets.iter())
            .any(|&x| x as u64 >= PRIME)
        {
            return Err("HASH_COEFFICIENT_RANGE".into());
        }
        Ok(())
    }
}
