use mgbfs_core::hash::Hash128;
use mgbfs_runtime::macro_owner::{CandidateKey, FutureOffer, MacroOwner};

fn hash(value: u32) -> Hash128 {
    Hash128([value, value ^ 0xaaaa_aaaa, !value, value.rotate_left(7)])
}

#[test]
fn later_shorter_arrival_wins_without_mutating_the_longer_slot() {
    let mut owner = MacroOwner::new(3, 8, 16).unwrap();
    owner.seed(0, [hash(0)]).unwrap();
    owner
        .offer(FutureOffer::new(
            2,
            hash(7),
            70,
            CandidateKey::new(0, 2, 0, 0, 0),
        ))
        .unwrap();
    owner
        .offer(FutureOffer::new(
            1,
            hash(7),
            71,
            CandidateKey::new(0, 1, 0, 1, 0),
        ))
        .unwrap();
    assert_eq!(owner.settle(1).unwrap(), vec![(hash(7), 71)]);
    assert!(owner.settle(2).unwrap().is_empty());
}

#[test]
fn settlement_deduplicates_future_runs_and_last_two_k_layers() {
    let mut owner = MacroOwner::new(2, 3, 16).unwrap();
    owner.seed(0, [hash(0)]).unwrap();
    owner
        .offer(FutureOffer::new(
            1,
            hash(1),
            10,
            CandidateKey::new(0, 1, 1, 0, 4),
        ))
        .unwrap();
    owner
        .offer(FutureOffer::new(
            1,
            hash(1),
            11,
            CandidateKey::new(0, 1, 0, 0, 2),
        ))
        .unwrap();
    owner
        .offer(FutureOffer::new(
            1,
            hash(0),
            12,
            CandidateKey::new(0, 1, 0, 0, 1),
        ))
        .unwrap();
    assert_eq!(owner.settle(1).unwrap(), vec![(hash(1), 11)]);
    owner
        .offer(FutureOffer::new(
            2,
            hash(1),
            20,
            CandidateKey::new(1, 1, 0, 0, 0),
        ))
        .unwrap();
    assert!(owner.settle(2).unwrap().is_empty());
}

#[test]
fn depth_order_and_all_capacities_fail_before_mutation() {
    assert!(MacroOwner::new(0, 1, 1).is_err());
    let mut owner = MacroOwner::new(2, 1, 1).unwrap();
    owner.seed(0, [hash(0)]).unwrap();
    assert!(owner
        .offer(FutureOffer::new(
            3,
            hash(3),
            3,
            CandidateKey::new(0, 3, 0, 0, 0)
        ))
        .is_err());
    owner
        .offer(FutureOffer::new(
            1,
            hash(1),
            1,
            CandidateKey::new(0, 1, 0, 0, 0),
        ))
        .unwrap();
    assert!(owner
        .offer(FutureOffer::new(
            1,
            hash(2),
            2,
            CandidateKey::new(0, 1, 0, 0, 1)
        ))
        .is_err());
    assert_eq!(owner.settle(1).unwrap(), vec![(hash(1), 1)]);
    assert!(owner.settle(3).is_err());

    let mut owner = MacroOwner::new(1, 2, 1).unwrap();
    owner.seed(0, [hash(0)]).unwrap();
    owner
        .offer(FutureOffer::new(
            1,
            hash(1),
            1,
            CandidateKey::new(0, 1, 0, 0, 0),
        ))
        .unwrap();
    owner
        .offer(FutureOffer::new(
            1,
            hash(2),
            2,
            CandidateKey::new(0, 1, 0, 0, 1),
        ))
        .unwrap();
    assert_eq!(owner.settle(1).unwrap_err(), "MACRO_SETTLED_CAPACITY");
    assert_eq!(owner.settle(1).unwrap_err(), "MACRO_SETTLED_CAPACITY");
}
