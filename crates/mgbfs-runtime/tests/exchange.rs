use mgbfs_runtime::exchange::{Epoch, Sequencer};
#[test]
fn empty_rank_participates_and_drain_waits_for_every_completion() {
    let mut s = Sequencer::new(2, 3).unwrap();
    s.ready(1, 7).unwrap();
    s.close(0).unwrap();
    s.close(1).unwrap();
    assert!(!s.drained());
    assert_eq!(
        s.begin().unwrap(),
        Some(Epoch {
            id: 0,
            offers: vec![None, Some(7)]
        })
    );
    assert!(s.begin().is_err());
    s.complete(1, 0).unwrap();
    assert!(!s.drained());
    assert!(s.complete(1, 0).is_err());
    s.complete(0, 0).unwrap();
    assert!(s.drained());
    assert_eq!(s.begin().unwrap(), None);
}
#[test]
fn ready_order_does_not_change_peer_order_and_slots_are_bounded() {
    let mut s = Sequencer::new(2, 1).unwrap();
    s.ready(1, 20).unwrap();
    s.ready(0, 10).unwrap();
    assert!(s.ready(1, 21).is_err());
    assert_eq!(
        s.begin().unwrap(),
        Some(Epoch {
            id: 0,
            offers: vec![Some(10), Some(20)]
        })
    );
    assert!(s.complete(0, 1).is_err());
    s.complete(0, 0).unwrap();
    s.complete(1, 0).unwrap();
    s.ready(0, 11).unwrap();
    s.close(0).unwrap();
    assert!(s.ready(0, 12).is_err());
    assert_eq!(s.begin().unwrap().unwrap().id, 1);
}
