use mgbfs_core::memory::AllocationLedger;

#[test]
fn ledger_accounts_padding_and_preserves_reserve_before_allocation() {
    let mut l = AllocationLedger::new(1024, 256).unwrap();
    assert_eq!(l.add("states", 3, 17, 256).unwrap(), 0);
    assert_eq!(l.add("hashes", 10, 16, 256).unwrap(), 256);
    assert_eq!(l.total(), 512);
    assert!(l.add("scratch", 257, 1, 256).is_err());
    assert_eq!(l.total(), 512);
}

#[test]
fn ledger_rejects_overflow_duplicate_names_and_invalid_alignment() {
    let mut l = AllocationLedger::new(u64::MAX, 0).unwrap();
    assert!(l.add("overflow", u64::MAX, 16, 256).is_err());
    assert!(l.add("unaligned", 1, 1, 3).is_err());
    l.add("states", 1, 16, 16).unwrap();
    assert!(l.add("states", 1, 16, 16).is_err());
    assert!(AllocationLedger::new(1, 2).is_err());
}
