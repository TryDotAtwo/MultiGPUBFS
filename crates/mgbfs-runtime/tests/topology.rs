use mgbfs_runtime::topology::reference_owner_geometry;
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
