//! Benchmark orchestration outside the search timer and production hot path.
use mgbfs_core::Result;

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
