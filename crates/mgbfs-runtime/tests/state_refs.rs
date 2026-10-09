use mgbfs_runtime::ring::StateRing;

#[test]
fn absolute_refs_reject_wrap_padding_and_recycled_physical_addresses() {
    let mut r = StateRing::new(10, 8).unwrap();
    let a = r.reserve(6).unwrap();
    assert!(r.state_ref(a.id, 0).is_err());
    r.materialized(a.id).unwrap();
    r.publish(a.id).unwrap();
    let old = r.state_ref(a.id, 0).unwrap();
    assert_eq!(old, 0);
    assert_eq!(r.state_ref(a.id, 5).unwrap(), 5);
    assert!(r.state_ref(a.id, 6).is_err());
    r.enumerated(a.id).unwrap();
    r.archived(a.id).unwrap();
    r.reclaim();
    let b = r.reserve(3).unwrap();
    r.materialized(b.id).unwrap();
    r.publish(b.id).unwrap();
    let c = r.reserve(4).unwrap(); // seq10..14, physical0..4, padding at seq9
    r.materialized(c.id).unwrap();
    r.publish(c.id).unwrap();
    assert_eq!((c.sequence, c.begin), (10, 0));
    assert_eq!(r.state_ref(c.id, 0).unwrap(), 10);
    assert_eq!(r.resolve(10).unwrap(), 0);
    assert_eq!(r.resolve(13).unwrap(), 3);
    assert_eq!(r.resolve(8).unwrap(), 8);
    assert!(r.resolve(old).is_err());
    assert!(r.resolve(9).is_err());
    assert!(r.resolve(14).is_err());
}

#[test]
fn enumerated_parent_is_readable_only_while_origin_lease_is_live() {
    let mut r = StateRing::new(4, 2).unwrap();
    let a = r.reserve(4).unwrap();
    r.materialized(a.id).unwrap();
    r.publish(a.id).unwrap();
    let origin = r.state_ref(a.id, 2).unwrap();
    r.hold_origins(a.id).unwrap();
    r.enumerated(a.id).unwrap();
    r.archived(a.id).unwrap();
    assert_eq!(r.resolve(origin).unwrap(), 2);
    assert_eq!(r.reclaim(), 0);
    r.release_origins(a.id).unwrap();
    assert!(r.resolve(origin).is_err());
    assert_eq!(r.reclaim(), 4);
    assert!(r.state_ref(a.id, 2).is_err());
}

#[test]
fn weighted_owner_uses_provisional_refs_without_publishing_a_future_frontier() {
    use mgbfs_core::hash::Hash128;
    use mgbfs_runtime::macro_owner::{CandidateKey, FutureOffer, MacroOwner};

    let mut ring = StateRing::new(8, 4).unwrap();
    let far = ring.reserve(3).unwrap();
    let near = ring.reserve(1).unwrap();
    assert!(ring.state_ref(far.id, 0).is_err());
    ring.materialized(far.id).unwrap();
    ring.materialized(near.id).unwrap();
    // A handle identifies materialized bytes; it is not generation permission.
    let far_ref = ring.state_ref(far.id, 0).unwrap();
    let near_ref = ring.state_ref(near.id, 0).unwrap();
    assert_eq!((far_ref, near_ref), (0, 3));
    assert_eq!(ring.resolve(far_ref).unwrap_err(), "STATE_REF_NOT_READABLE");
    assert_eq!(
        ring.resolve(near_ref).unwrap_err(),
        "STATE_REF_NOT_READABLE"
    );
    assert!(ring.state_ref(far.id, 3).is_err());

    let mut owner = MacroOwner::new(3, 8, 8).unwrap();
    owner.seed(0, [Hash128([0; 4])]).unwrap();
    let duplicate = Hash128([7; 4]);
    owner
        .offer(FutureOffer::new(
            3,
            duplicate,
            far_ref,
            CandidateKey::new(0, 3, 0, 0, 0),
        ))
        .unwrap();
    owner
        .offer(FutureOffer::new(
            1,
            duplicate,
            near_ref,
            CandidateKey::new(0, 1, 1, 0, 0),
        ))
        .unwrap();
    assert_eq!(owner.settle(1).unwrap(), vec![(duplicate, near_ref)]);
    ring.publish(near.id).unwrap();
    assert_eq!(ring.resolve(near_ref).unwrap(), 3);
    assert_eq!(ring.resolve(far_ref).unwrap_err(), "STATE_REF_NOT_READABLE");

    ring.archived(near.id).unwrap();
    ring.enumerated(near.id).unwrap();
    // Later-allocated near extent cannot reclaim an earlier live future extent.
    assert_eq!(ring.reclaim(), 0);
    assert_eq!(ring.reserve(5).unwrap_err(), "STATE_RING_CAPACITY");
    assert!(owner.settle(2).unwrap().is_empty());
    assert!(owner.settle(3).unwrap().is_empty());
    assert_eq!(ring.resolve(far_ref).unwrap_err(), "STATE_REF_NOT_READABLE");
}
