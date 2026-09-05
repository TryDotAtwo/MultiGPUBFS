use mgbfs_core::memory::hash_first_reference_ledger;

#[test]
fn all_source_banks_and_response_buffers_are_counted() {
    let plan = hash_first_reference_ledger(9, 4, 64, 16, [512, 512, 256, 256, 257]).unwrap();
    assert_eq!(plan.allocations.len(), 21);
    assert_eq!(plan.total(), 11520);
    assert_eq!(
        plan.allocations
            .iter()
            .map(|a| a.payload_bytes)
            .sum::<u64>(),
        9685
    );
    assert!(plan.allocations.iter().all(|a| a.offset % 256 == 0));
}

#[test]
fn invalid_shapes_or_unrepresentable_scratch_fail() {
    for (width, moves, capacity, stride) in [
        (0, 4, 64, 16),
        (9, 0, 64, 16),
        (9, 4, 0, 16),
        (17, 4, 64, 16),
        (9, 4, 64, 17),
        (9, 65537, 64, 16),
        (9, 4, u64::MAX, 16),
    ] {
        assert!(hash_first_reference_ledger(width, moves, capacity, stride, [0; 5]).is_err());
    }
    assert!(hash_first_reference_ledger(9, 4, 64, 16, [u64::MAX; 5]).is_err());
}
