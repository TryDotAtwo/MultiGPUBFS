use mgbfs_runtime::{receipts::HashFirstLease, ring::StateRing};

fn current_parent() -> (StateRing, u64) {
    let mut ring = StateRing::new(16, 4).unwrap();
    let extent = ring.reserve(8).unwrap();
    ring.materialized(extent.id).unwrap();
    ring.publish(extent.id).unwrap();
    (ring, extent.id)
}

#[test]
fn parent_lease_closes_only_after_every_owner_receipt_and_response_completion() {
    let (mut ring, parent) = current_parent();
    let mut lease = HashFirstLease::begin(&mut ring, parent, &[3, 2, 0]).unwrap();
    ring.enumerated(parent).unwrap();
    ring.archived(parent).unwrap();

    lease.receipt(0, 3, 2).unwrap();
    lease.served(0, 41).unwrap();
    lease.receipt(1, 2, 1).unwrap();
    assert!(!lease.try_close(&mut ring).unwrap());
    assert_eq!(ring.reclaim(), 0);

    lease.served(1, 77).unwrap();
    assert!(!lease.try_close(&mut ring).unwrap());
    lease.served(0, 42).unwrap();
    assert!(lease.try_close(&mut ring).unwrap());
    assert_eq!(ring.reclaim(), 8);
    assert!(!lease.try_close(&mut ring).unwrap());
}

#[test]
fn zero_payload_batch_does_not_take_an_origin_lease() {
    let (mut ring, parent) = current_parent();
    let mut lease = HashFirstLease::begin(&mut ring, parent, &[0, 0]).unwrap();
    assert!(lease.try_close(&mut ring).unwrap());
    ring.enumerated(parent).unwrap();
    ring.archived(parent).unwrap();
    assert_eq!(ring.reclaim(), 8);
}

#[test]
fn poisoned_receipts_never_release_the_parent() {
    let (mut ring, parent) = current_parent();
    let mut lease = HashFirstLease::begin(&mut ring, parent, &[1]).unwrap();
    ring.enumerated(parent).unwrap();
    ring.archived(parent).unwrap();
    assert!(lease.receipt(0, 2, 1).is_err());
    assert!(lease.try_close(&mut ring).is_err());
    assert_eq!(ring.reclaim(), 0);
}
