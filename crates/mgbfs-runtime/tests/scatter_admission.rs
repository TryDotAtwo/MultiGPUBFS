use mgbfs_runtime::control_wire::Plane;
use mgbfs_runtime::scatter_admission::{ScatterAdmission, TicketKey};
fn key() -> TicketKey {
    TicketKey {
        depth: 7,
        epoch: 19,
        source: 2,
        plane: Plane::Candidate,
        generation: 8,
    }
}
#[test]
fn switching_source_may_use_a_lower_source_local_slot_token() {
    let mut a = ScatterAdmission::new(3).unwrap();
    a.prepare(key(), &[0, 0, 0], 16, 0).unwrap();
    for rank in 0..3 {
        a.admit(key(), rank, 0).unwrap();
    }
    assert!(a.launch(key()).unwrap());
    a.retire(key()).unwrap();
    let next = TicketKey {
        epoch: 20,
        source: 1,
        generation: 0,
        ..key()
    };
    a.prepare(next, &[0, 0, 0], 16, 0).unwrap();
    for rank in 0..3 {
        a.admit(next, rank, 0).unwrap();
    }
    assert!(a.launch(next).unwrap());
}
#[test]
fn ready_later_ticket_cannot_overtake_global_issue_order() {
    let mut earlier = ScatterAdmission::new(3).unwrap();
    let mut later = ScatterAdmission::new(3).unwrap();
    let later_key = TicketKey { epoch: 20, ..key() };
    earlier.prepare(key(), &[0, 0, 0], 16, 0).unwrap();
    later.prepare(later_key, &[0, 0, 0], 16, 0).unwrap();
    for rank in [2, 0, 1] {
        later.admit(later_key, rank, 0).unwrap();
    }
    let mut next = 19;
    assert!(!later.launch_ordered(later_key, &mut next).unwrap());
    assert_eq!(next, 19);
    assert!(!earlier.launch_ordered(key(), &mut next).unwrap());
    for rank in [1, 2, 0] {
        earlier.admit(key(), rank, 0).unwrap();
    }
    assert!(earlier.launch_ordered(key(), &mut next).unwrap());
    assert_eq!(next, 20);
    assert!(later.launch_ordered(later_key, &mut next).unwrap());
    assert_eq!(next, 21);
}
#[test]
fn reuse_resets_acknowledgments_and_rejects_old_generation() {
    let mut a = ScatterAdmission::new(3).unwrap();
    a.prepare(key(), &[0, 0, 0], 16, 0).unwrap();
    for rank in 0..3 {
        a.admit(key(), rank, 0).unwrap();
    }
    assert!(a.launch(key()).unwrap());
    a.retire(key()).unwrap();
    let next = TicketKey {
        epoch: 20,
        generation: 9,
        ..key()
    };
    a.prepare(next, &[1, 0, 0], 16, 16).unwrap();
    assert!(!a.launch(next).unwrap());
    assert!(a.admit(key(), 0, 16).is_err());
    let mut a = ScatterAdmission::new(3).unwrap();
    a.prepare(key(), &[0, 0, 0], 16, 0).unwrap();
    assert!(a.retire(key()).is_err());
    let mut a = ScatterAdmission::new(3).unwrap();
    a.prepare(key(), &[0, 0, 0], 16, 0).unwrap();
    for rank in 0..3 {
        a.admit(key(), rank, 0).unwrap();
    }
    assert!(a.launch(key()).unwrap());
    a.retire(key()).unwrap();
    assert!(a.prepare(key(), &[0, 0, 0], 16, 0).is_err());
}
#[test]
fn asymmetric_sizes_require_every_rank_including_empty_and_source() {
    let mut a = ScatterAdmission::new(3).unwrap();
    a.prepare(key(), &[2, 0, 3], 16, 80).unwrap();
    assert_eq!(a.range(0).unwrap(), (0, 32));
    assert_eq!(a.range(1).unwrap(), (32, 0));
    assert_eq!(a.range(2).unwrap(), (32, 48));
    a.admit(key(), 2, 0).unwrap(); // self is a send view, no receive allocation
    a.admit(key(), 0, 32).unwrap();
    assert!(!a.launch(key()).unwrap());
    a.admit(key(), 1, 0).unwrap();
    assert!(a.launch(key()).unwrap());
    assert!(a.launch(key()).is_err());
}
#[test]
fn overflow_or_stale_ack_poison_admission() {
    let mut a = ScatterAdmission::new(3).unwrap();
    a.prepare(key(), &[2, 0, 3], 16, 80).unwrap();
    assert!(a.admit(key(), 0, 31).is_err());
    assert!(a.admit(key(), 0, 32).is_err());
    assert!(a.launch(key()).is_err());
    let mut a = ScatterAdmission::new(3).unwrap();
    a.prepare(key(), &[2, 0, 3], 16, 80).unwrap();
    assert!(a
        .admit(
            TicketKey {
                generation: 7,
                ..key()
            },
            0,
            32
        )
        .is_err());
    assert!(a.launch(key()).is_err());
}
#[test]
fn byte_arithmetic_and_duplicate_ack_fail_before_launch() {
    for (counts, width, capacity) in [
        ([u64::MAX, 0, 0], 2, u64::MAX),
        ([u64::MAX, 1, 0], 1, u64::MAX),
        ([2, 0, 3], 16, 79),
    ] {
        let mut a = ScatterAdmission::new(3).unwrap();
        assert!(a.prepare(key(), &counts, width, capacity).is_err());
        assert!(a.launch(key()).is_err());
    }
    let mut a = ScatterAdmission::new(3).unwrap();
    a.prepare(key(), &[0, 0, 0], 16, 0).unwrap();
    a.admit(key(), 0, 0).unwrap();
    assert!(a.admit(key(), 0, 0).is_err());
    assert!(a.launch(key()).is_err());
}
