use mgbfs_runtime::ring::StateRing;

#[test]
fn parent_cannot_be_reclaimed_while_origin_obligation_is_live() {
    let mut ring = StateRing::new(8, 4).unwrap();
    let p = ring.reserve(8).unwrap();
    ring.materialized(p.id).unwrap();
    ring.publish(p.id).unwrap();
    ring.hold_origins(p.id).unwrap();
    ring.hold_origins(p.id).unwrap();
    ring.archived(p.id).unwrap();
    ring.enumerated(p.id).unwrap();
    assert_eq!(ring.reclaim(), 0);
    assert!(ring.reserve(1).is_err());
    ring.release_origins(p.id).unwrap();
    assert_eq!(ring.reclaim(), 0);
    ring.release_origins(p.id).unwrap();
    assert_eq!(ring.reclaim(), 8);
    assert!(ring.release_origins(p.id).is_err());
}

#[test]
fn lease_underflow_and_late_registration_are_errors() {
    let mut ring = StateRing::new(8, 4).unwrap();
    let p = ring.reserve(2).unwrap();
    assert!(ring.hold_origins(p.id).is_err());
    ring.materialized(p.id).unwrap();
    ring.publish(p.id).unwrap();
    assert!(ring.release_origins(p.id).is_err());
    ring.enumerated(p.id).unwrap();
    assert!(ring.hold_origins(p.id).is_err());
}
