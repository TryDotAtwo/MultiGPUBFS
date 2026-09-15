use mgbfs_runtime::library_owner::OwnerCommitGate;

#[test]
fn ffi_failure_cannot_later_publish_reserved_rows() {
    let mut gate = OwnerCommitGate::new(8);
    gate.compared(1, 4, 2).unwrap();
    gate.reserve(1, 2).unwrap();
    gate.abort();
    assert!(gate.completed(1).is_err());
    assert_eq!(gate.accepted(), 0);
    assert!(gate.check_compare(2).is_err());
}

#[test]
fn next_compare_is_rejected_before_reusing_uncompleted_device_result() {
    let mut gate = OwnerCommitGate::new(8);
    gate.check_compare(1).unwrap();
    gate.compared(1, 4, 2).unwrap();
    gate.reserve(1, 2).unwrap();
    assert!(gate.check_compare(2).is_err());
    assert_eq!(gate.accepted(), 0);
}

#[test]
fn publication_requires_reservation_and_device_completion() {
    let mut gate = OwnerCommitGate::new(5);
    gate.compared(0, 8, 3).unwrap();
    assert_eq!(gate.accepted(), 0);
    assert_eq!(gate.reserve(0, 3).unwrap(), 0..3);
    assert_eq!(gate.accepted(), 0);
    gate.completed(0).unwrap();
    assert_eq!(gate.accepted(), 3);
    gate.compared(1, 9, 2).unwrap();
    assert_eq!(gate.reserve(1, 2).unwrap(), 3..5);
    gate.completed(1).unwrap();
    assert_eq!(gate.accepted(), 5);
}

#[test]
fn insufficient_credit_is_fatal_without_publishing_or_retry() {
    let mut gate = OwnerCommitGate::new(5);
    gate.compared(0, 3, 3).unwrap();
    assert!(gate.reserve(0, 2).is_err());
    assert_eq!(gate.accepted(), 0);
    assert!(gate.reserve(0, 3).is_err());
    assert!(gate.compared(1, 0, 0).is_err());
}

#[test]
fn capacity_failure_preserves_previous_commit() {
    let mut gate = OwnerCommitGate::new(3);
    gate.compared(0, 2, 2).unwrap();
    gate.reserve(0, 2).unwrap();
    gate.completed(0).unwrap();
    assert!(gate.compared(1, 2, 2).is_err());
    assert_eq!(gate.accepted(), 2);
}

#[test]
fn empty_epochs_are_real_transactions_and_cannot_be_replayed() {
    let mut gate = OwnerCommitGate::new(0);
    gate.compared(4, 0, 0).unwrap();
    assert_eq!(gate.reserve(4, 0).unwrap(), 0..0);
    gate.completed(4).unwrap();
    assert!(gate.compared(4, 0, 0).is_err());
}

#[test]
fn stale_device_completion_cannot_publish_another_batch() {
    let mut gate = OwnerCommitGate::new(8);
    gate.compared(2, 3, 3).unwrap();
    gate.reserve(2, 3).unwrap();
    assert!(gate.completed(1).is_err());
    assert_eq!(gate.accepted(), 0);
    assert!(gate.completed(2).is_err());
}

#[test]
fn second_compare_cannot_overwrite_pending_result() {
    let mut gate = OwnerCommitGate::new(8);
    gate.compared(0, 3, 3).unwrap();
    assert!(gate.compared(1, 3, 3).is_err());
    assert_eq!(gate.accepted(), 0);
}

#[test]
fn corrupted_survivor_count_poisoned_before_reservation() {
    let mut gate = OwnerCommitGate::new(8);
    assert_eq!(gate.compared(0, 1, 2).unwrap_err(), "LIBRARY_OWNER_COUNT");
    assert!(gate.compared(1, 0, 0).is_err());
}

#[test]
fn completion_without_credit_never_publishes() {
    let mut gate = OwnerCommitGate::new(8);
    gate.compared(0, 3, 3).unwrap();
    assert!(gate.completed(0).is_err());
    assert_eq!(gate.accepted(), 0);
}

#[test]
fn near_u64_capacity_cannot_wrap_on_next_batch() {
    let mut gate = OwnerCommitGate::new(u64::MAX);
    gate.compared(0, u64::MAX, u64::MAX - 1).unwrap();
    gate.reserve(0, u64::MAX - 1).unwrap();
    gate.completed(0).unwrap();
    assert!(gate.compared(1, 2, 2).is_err());
    assert_eq!(gate.accepted(), u64::MAX - 1);
}
