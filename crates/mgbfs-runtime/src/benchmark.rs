//! Benchmark orchestration outside the search timer and production hot path.
use mgbfs_core::Result;
use mgbfs_core::config::{FrontierProfile, ReferenceOwner};

pub fn library_backend_label(owner: ReferenceOwner, profile: FrontierProfile) -> Option<&'static str> {
    match (owner, profile) {
        (ReferenceOwner::CudfRelational, FrontierProfile::Dense) => Some("library_nccl_dense_cudf_v1"),
        (ReferenceOwner::CudfRelational, FrontierProfile::HashFirst) => Some("library_nccl_hash_first_cudf_v1"),
        (ReferenceOwner::CucoIndexed, FrontierProfile::Dense) => Some("library_nccl_dense_cuco_v1"),
        (ReferenceOwner::CucoIndexed, FrontierProfile::HashFirst) => Some("library_nccl_hash_first_cuco_v1"),
        (ReferenceOwner::CucoRank, FrontierProfile::Dense) => Some("library_nccl_dense_cuco_rank_v1"),
        (ReferenceOwner::CucoRank, FrontierProfile::HashFirst) => None,
        (ReferenceOwner::Native(_), _) => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Warmup,
    Measure,
}

/// Warmup uses a separate runtime instance in the same process. Never continue
/// into a measured run after failed warmup, including archive finalization.
pub fn run_phases(warmup: bool, mut run: impl FnMut(Phase) -> Result<()>) -> Result<()> {
    if warmup {
        run(Phase::Warmup)?;
    }
    run(Phase::Measure)
}
