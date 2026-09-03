//! Versioned CPU contracts. No CUDA dependency and no implicit GPU fallback.

pub mod config;
pub mod hash;
pub mod macro_generators;
pub mod macro_memory;
pub mod matrix;
pub mod memory;
pub mod owner_job;
pub mod rank_plan;
pub mod wire;

pub type Result<T> = std::result::Result<T, String>;
