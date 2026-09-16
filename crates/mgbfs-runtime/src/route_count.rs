//! Route count contract: sorting without compaction preserves cardinality.
use mgbfs_core::Result;

/// Validate the split published by exchange packing before either owner reads
/// its range. UINT32_MAX in the first word is the pack kernel's fatal marker.
pub fn packed_count(input: u32, owners: impl AsRef<[u32]>) -> Result<u32> {
    let owners = owners.as_ref();
    if !owners.len().is_power_of_two() || owners.len() > 8 {
        return Err("EXCHANGE_OWNER_COUNT".into());
    }
    if owners[0] == u32::MAX {
        return Err("EXCHANGE_SOURCE_REF".into());
    }
    let count = owners.iter().try_fold(0u32, |total, &rows| {
        total.checked_add(rows).ok_or("EXCHANGE_COUNT_BOUND")
    })?;
    if count > input {
        return Err("EXCHANGE_COUNT_BOUND".into());
    }
    Ok(count)
}

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
