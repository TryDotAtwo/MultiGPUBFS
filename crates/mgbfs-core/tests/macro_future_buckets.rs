use mgbfs_core::macro_memory::FutureBucketLayout;

#[test]
fn sparse_fixed_bucket_extents_are_disjoint_and_depth_slots_rotate() {
    // Three target depths, four buckets each. Physical storage is the sum of
    // configured capacities, not macro_depth * buckets * bucket_limit.
    let caps = [2, 0, 1, 3, 0, 4, 0, 0, 1, 1, 0, 2];
    let layout = FutureBucketLayout::derive(3, 4, 4, &caps, 14).unwrap();
    assert_eq!(layout.hash_records, 14);
    assert_eq!(layout.hash_bytes, 14 * 16);
    assert_eq!(layout.bucket_range(0, 0).unwrap(), (0, 2));
    assert_eq!(layout.bucket_range(1, 1).unwrap(), (6, 4));
    assert_eq!(layout.bucket_range(2, 3).unwrap(), (12, 2));
    assert_eq!(layout.slot_directory(1).unwrap(), (&[6, 6, 10, 10, 10][..], &[0, 4, 0, 0][..]));
    assert_eq!(layout.target_bucket(0, 1, 1).unwrap(), (6, 4));
    assert_eq!(layout.target_bucket(0, 3, 0).unwrap(), (0, 2));
    assert_eq!(layout.target_bucket(1, 4, 0).unwrap(), (6, 0));
    assert!(layout.target_bucket(0, 0, 0).is_err());
    assert!(layout.target_bucket(0, 4, 0).is_err());
}

#[test]
fn preflight_rejects_overcommit_bad_shape_and_integer_overflow() {
    assert!(FutureBucketLayout::derive(0, 1, 1, &[], 0).is_err());
    assert!(FutureBucketLayout::derive(2, 2, 4, &[1, 2, 3], 6).is_err());
    assert!(FutureBucketLayout::derive(1, 1, 4, &[5], 5).is_err());
    assert!(FutureBucketLayout::derive(1, 2, 4, &[4, 4], 7).is_err());
    assert!(FutureBucketLayout::derive(1, 1, 1, &[0], 1).is_err());
    assert!(FutureBucketLayout::derive(u32::MAX, u32::MAX, 1, &[], 0).is_err());
}
