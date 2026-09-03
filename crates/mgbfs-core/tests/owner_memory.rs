use mgbfs_core::memory::bounded_owner_ledger;

#[test]
fn owner_scratch_is_job_bounded_with_individually_aligned_planes() {
    // I=16,J=1,K=8: 15 owned allocations of 256 bytes.
    // Queries add 256+512+0.
    let plan = bounded_owner_ledger(16, 1, 8, [1, 257, 0]).unwrap();
    assert_eq!(plan.total(), 4608);
    assert!(plan.allocations.iter().all(|a| a.offset % 256 == 0));
    assert_eq!(
        plan.allocations
            .iter()
            .find(|a| a.name == "accepted_output")
            .unwrap()
            .payload_bytes,
        128
    );
    // No layer capacity argument exists: enlarging a layer cannot enlarge a lane.
}

#[test]
fn invalid_shapes_and_arithmetic_overflow_are_rejected() {
    for (i, j, k) in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (u64::MAX, 1, 1),
        (1, u64::MAX, 2),
        (1, 1, u64::MAX),
    ] {
        assert!(bounded_owner_ledger(i, j, k, [0; 3]).is_err());
    }
    assert!(bounded_owner_ledger(1, 1, 1, [u64::MAX, 0, 0]).is_err());
}
