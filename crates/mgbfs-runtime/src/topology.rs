use mgbfs_core::Result;

/// Reference runtime supports one or two ranks. This is not an N-rank router.
pub fn reference_owner_geometry(
    world: u32,
    rank: u32,
    map: [u32; 2],
    buckets: u32,
    shards: u32,
) -> Result<(u32, u32)> {
    if !(1..=2).contains(&world)
        || rank >= world
        || !buckets.is_power_of_two()
        || !shards.is_power_of_two()
        || shards < world
        || shards > buckets
        || (world == 1 && map != [0, 0])
        || (world == 2 && map != [0, 1] && map != [1, 0])
    {
        return Err("REFERENCE_TOPOLOGY".into());
    }
    Ok((buckets / world, shards / world))
}
