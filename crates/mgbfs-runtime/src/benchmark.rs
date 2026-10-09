//! Benchmark orchestration outside the search timer and production hot path.
use mgbfs_core::config::{FrontierProfile, ReferenceOwner};
use mgbfs_core::Result;

pub fn library_backend_label(
    owner: ReferenceOwner,
    profile: FrontierProfile,
) -> Option<&'static str> {
    match (owner, profile) {
        (ReferenceOwner::CudfRelational, FrontierProfile::Dense) => {
            Some("library_nccl_dense_cudf_v1")
        }
        (ReferenceOwner::CudfRelational, FrontierProfile::HashFirst) => {
            Some("library_nccl_hash_first_cudf_v1")
        }
        (ReferenceOwner::CucoIndexed, FrontierProfile::Dense) => Some("library_nccl_dense_cuco_v1"),
        (ReferenceOwner::CucoIndexed, FrontierProfile::HashFirst) => {
            Some("library_nccl_hash_first_cuco_v1")
        }
        (ReferenceOwner::CucoRank, FrontierProfile::Dense) => {
            Some("library_nccl_dense_cuco_rank_v1")
        }
        (ReferenceOwner::CucoRank, FrontierProfile::HashFirst) => None,
        (ReferenceOwner::Native(_), _) => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Warmup,
    Measure,
}

/// Bind launch-phase agreement without changing the persistent config digest.
pub fn phase_digest(config: [u8; 32], warmup: bool, measure: bool) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"mgbfs-launch-phase-v1\0");
    hash.update(config);
    hash.update([u8::from(warmup), u8::from(measure)]);
    hash.finalize().into()
}

/// Warmup uses a separate runtime instance in the same process. Never continue
/// into a measured run after failed warmup, including archive finalization.
pub fn run_phases(warmup: bool, mut run: impl FnMut(Phase) -> Result<()>) -> Result<()> {
    if warmup {
        run(Phase::Warmup)?;
    }
    run(Phase::Measure)
}
