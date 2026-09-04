use mgbfs_core::hash::Hash128;
use mgbfs_runtime::{owner::OwnerModel, ring::StateRing};
fn h(x: u32) -> Hash128 {
    Hash128([x, 0, 0, 0])
}
#[test]
fn forced_hash_collision_exposes_probabilistic_not_exact_identity() {
    let state_a = [1u8, 0, 0, 1];
    let state_b = [1u8, 1, 0, 1];
    assert_ne!(state_a, state_b);
    // Explicit collision fixture: both distinct states have the same test hash.
    let mut owner = OwnerModel::new(vec![], vec![], 16);
    assert_eq!(owner.commit(0, &[h(42), h(42)]).unwrap().len(), 1);
    assert!(owner.commit(1, &[h(42)]).unwrap().is_empty());
    // Therefore hash-only BFS cannot claim exact full-state equivalence for all
    // inputs. Independent seeded reruns mitigate, not eliminate, this boundary.
}

#[test]
fn empty_ring_can_reserve_full_capacity_after_nonzero_tail() {
    let mut r = StateRing::new(10, 4).unwrap();
    let a = r.reserve(3).unwrap();
    r.materialized(a.id).unwrap();
    r.publish(a.id).unwrap();
    r.enumerated(a.id).unwrap();
    r.archived(a.id).unwrap();
    assert_eq!(r.reclaim(), 3);
    let b = r.reserve(10).unwrap();
    assert_eq!(b.begin, 0);
    assert!(r.publish(a.id).is_err());
}

#[test]
fn owner_rejects_all_old_and_cross_epoch_duplicates_irrevocably() {
    let mut o = OwnerModel::new(vec![h(1)], vec![h(2)], 4);
    assert_eq!(
        o.commit(0, &[h(4), h(3), h(3), h(2), h(1)]).unwrap(),
        vec![h(3), h(4)]
    );
    assert_eq!(o.commit(1, &[h(4), h(5)]).unwrap(), vec![h(5)]);
    assert_eq!(o.accepted(), vec![h(3), h(4), h(5)]);
    assert!(o.commit(1, &[h(6)]).is_err());
    assert!(o.commit(2, &[h(6), h(7)]).is_err());
    assert_eq!(o.accepted(), vec![h(3), h(4), h(5)]);
}

#[test]
fn ring_never_reuses_live_archive_leases_and_wraps_without_overlap() {
    let mut r = StateRing::new(10, 8).unwrap();
    let a = r.reserve(6).unwrap();
    r.materialized(a.id).unwrap();
    r.publish(a.id).unwrap();
    r.enumerated(a.id).unwrap();
    assert_eq!(r.reclaim(), 0);
    let b = r.reserve(3).unwrap();
    assert!(r.reserve(2).is_err());
    r.archived(a.id).unwrap();
    assert_eq!(r.reclaim(), 6);
    let c = r.reserve(4).unwrap();
    assert_eq!((b.begin, c.begin), (6, 0));
    assert!(r.reserve(3).is_err());
    assert!(r.enumerated(c.id).is_err());
}

#[test]
fn ring_rejects_unmaterialized_publish_and_descriptor_exhaustion() {
    let mut r = StateRing::new(100, 1).unwrap();
    let a = r.reserve(1).unwrap();
    assert!(r.publish(a.id).is_err());
    assert!(r.reserve(1).is_err());
    assert!(r.archived(a.id).is_err());
}

#[test]
fn ring_peak_counts_wrap_padding_and_does_not_change_after_failed_reservation() {
    let mut r = StateRing::new(10, 8).unwrap();
    let a = r.reserve(6).unwrap();
    r.materialized(a.id).unwrap();
    r.publish(a.id).unwrap();
    r.enumerated(a.id).unwrap();
    r.archived(a.id).unwrap();
    r.reclaim();
    let b = r.reserve(3).unwrap();
    let c = r.reserve(4).unwrap(); // physical 6..9 and 0..4: 8 including wrap gap
    assert_eq!(r.peak_records(), 8);
    assert!(r.reserve(3).is_err());
    assert_eq!(r.peak_records(), 8);
    for x in [b, c] {
        r.materialized(x.id).unwrap();
        r.publish(x.id).unwrap();
        r.enumerated(x.id).unwrap();
        r.archived(x.id).unwrap();
    }
    r.reclaim();
    r.reserve(10).unwrap();
    assert_eq!(r.peak_records(), 10);
}

#[test]
fn dense_parent_prefix_is_reused_before_the_rest_of_the_frontier_is_enumerated() {
    let mut ring = StateRing::new(8, 4).unwrap();
    let current = ring.reserve(8).unwrap();
    ring.materialized(current.id).unwrap();
    ring.publish(current.id).unwrap();
    ring.archived(current.id).unwrap();

    assert_eq!(ring.retire_dense_prefix(current.id, 3).unwrap(), 3);
    let next = ring.reserve(3).unwrap();
    assert_eq!((next.begin, next.count), (0, 3));
    assert_eq!(
        ring.resolve(current.sequence + 2).unwrap_err(),
        "STALE_STATE_REF"
    );
    assert_eq!(ring.resolve(current.sequence + 3).unwrap(), 3);
}

#[test]
fn dense_parent_prefix_cannot_retire_archive_or_origin_live_bytes() {
    let mut ring = StateRing::new(8, 4).unwrap();
    let current = ring.reserve(8).unwrap();
    ring.materialized(current.id).unwrap();
    ring.publish(current.id).unwrap();
    assert_eq!(
        ring.retire_dense_prefix(current.id, 1).unwrap_err(),
        "DENSE_PREFIX_ARCHIVE_LIVE"
    );
    ring.archived(current.id).unwrap();
    ring.hold_origins(current.id).unwrap();
    assert_eq!(
        ring.retire_dense_prefix(current.id, 1).unwrap_err(),
        "DENSE_PREFIX_ORIGIN_LIVE"
    );
}
