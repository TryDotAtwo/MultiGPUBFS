//! Route count contract: sorting without compaction preserves cardinality.
use mgbfs_core::Result;

/// `mgbfs_route_run` with pre-dedup OFF only sorts/copies and publishes input
/// count. Same-stream consumers may use that known count without host polling.
/// ON still requires the compacted count until device-count packing is wired.
pub fn routed_count(prededup: bool, input: u32, read: impl FnOnce() -> Result<u32>) -> Result<u32> {
    if !prededup {
        return Ok(input);
    }
    let count = read()?;
    if count > input {
        return Err("ROUTE_COUNT_BOUND".into());
    }
    Ok(count)
}
