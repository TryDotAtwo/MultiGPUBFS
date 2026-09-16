use mgbfs_core::Result;

/// Average local occupancy plus explicit fixed headroom, not a worst-case
/// hash-distribution guarantee. Runtime bucket overflow remains fatal.
pub fn reference_bucket_capacity(records: u32, local_buckets: u32, slack: u32) -> Result<u32> {
    if records == 0 || !local_buckets.is_power_of_two() {
        return Err("REFERENCE_BUCKET_CAPACITY".into());
    }
    records
        .div_ceil(local_buckets)
        .checked_add(slack)
        .ok_or_else(|| "REFERENCE_BUCKET_CAPACITY".into())
}

/// Validate global bucket/shard geometry for power-of-two rank groups up to
/// eight. This query alone does not provide an N-rank data plane. The legacy
/// one-rank map [0,0] assigns both historical hash halves to the same GPU.
pub fn reference_owner_geometry(
    world: u32,
    rank: u32,
    map: impl AsRef<[u32]>,
    buckets: u32,
    shards: u32,
) -> Result<(u32, u32)> {
    let map = map.as_ref();
    if !world.is_power_of_two()
        || world > 8
        || rank >= world
        || !buckets.is_power_of_two()
        || !shards.is_power_of_two()
        || shards < world
        || shards > buckets
        || (world == 1 && map != [0, 0])
    {
        return Err("REFERENCE_TOPOLOGY".into());
    }
    if world > 1 {
        if map.len() != world as usize {
            return Err("REFERENCE_TOPOLOGY".into());
        }
        let mut seen = 0u32;
        for &owner_rank in map {
            if owner_rank >= world || seen & (1u32 << owner_rank) != 0 {
                return Err("REFERENCE_TOPOLOGY".into());
            }
            seen |= 1u32 << owner_rank;
        }
    }
    Ok((buckets / world, shards / world))
}
