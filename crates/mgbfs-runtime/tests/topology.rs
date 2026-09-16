use mgbfs_runtime::topology::reference_owner_geometry;
#[test]
fn eight_ranks_partition_global_buckets_and_shards_with_permuted_owners() {
    let map = [7, 3, 0, 6, 1, 5, 2, 4];
    for rank in 0..8 {
        assert_eq!(
            reference_owner_geometry(8, rank, map, 256, 64).unwrap(),
            (32, 8)
        );
    }
    assert_eq!(
        reference_owner_geometry(4, 3, [3, 0, 2, 1], 256, 16).unwrap(),
        (64, 4)
    );
    assert!(reference_owner_geometry(8, 0, [0, 1, 2, 3, 4, 5, 6, 6], 256, 64).is_err());
    assert!(reference_owner_geometry(8, 0, [0, 1], 256, 64).is_err());
    assert!(reference_owner_geometry(8, 0, map, 256, 4).is_err());
    assert!(reference_owner_geometry(8, 8, map, 256, 64).is_err());
}
#[test]
fn bucket_budget_uses_actual_local_geometry_and_checked_headroom() {
    use mgbfs_runtime::topology::reference_bucket_capacity;
    // Same global capacity split over the same 256 global buckets.
    assert_eq!(
        reference_bucket_capacity(39_916_800, 256, 4096).unwrap(),
        160021
    );
    assert_eq!(
        reference_bucket_capacity(19_958_400, 128, 4096).unwrap(),
        160021
    );
    assert_eq!(reference_bucket_capacity(257, 256, 0).unwrap(), 2);
    for (records, buckets, slack) in [(0, 256, 0), (1, 0, 0), (1, 3, 0), (u32::MAX, 1, 1)] {
        assert!(reference_bucket_capacity(records, buckets, slack).is_err());
    }
}
#[test]
fn one_rank_owns_both_hash_halves_without_halving_storage() {
    assert_eq!(
        reference_owner_geometry(1, 0, [0, 0], 256, 64).unwrap(),
        (256, 64)
    );
    assert_eq!(
        reference_owner_geometry(2, 0, [0, 1], 256, 64).unwrap(),
        (128, 32)
    );
    assert_eq!(
        reference_owner_geometry(2, 1, [1, 0], 256, 64).unwrap(),
        (128, 32)
    );
    for (world, rank, map) in [
        (1, 0, [0, 1]),
        (1, 1, [0, 0]),
        (2, 0, [0, 0]),
        (2, 2, [0, 1]),
        (3, 0, [0, 1]),
    ] {
        assert!(reference_owner_geometry(world, rank, map, 256, 64).is_err());
    }
    assert!(reference_owner_geometry(2, 0, [0, 1], 256, 1).is_err());
}
