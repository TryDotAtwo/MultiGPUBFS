use mgbfs_runtime::{
    control_wire::Plane, payload_lease::PayloadLease, scatter_admission::TicketKey,
};

fn key(epoch: u64, source: u32, generation: u64) -> TicketKey {
    TicketKey {
        depth: 0,
        epoch,
        source,
        plane: Plane::Candidate,
        generation,
    }
}

#[test]
fn bank_retains_every_consumer_until_fanout_is_sealed() {
    let mut bank = PayloadLease::new(2, 64, 3).unwrap();
    let ticket = key(0, 0, 100);
    bank.reserve(ticket, 64).unwrap();
    let first = bank.consumer(ticket).unwrap();
    let second = bank.consumer(ticket).unwrap();
    bank.complete(second).unwrap();
    assert!(!bank.drained(ticket).unwrap());
    bank.complete(first).unwrap();
    assert!(!bank.drained(ticket).unwrap()); // splitter may still discover jobs
    bank.seal(ticket).unwrap();
    assert!(bank.drained(ticket).unwrap());
    bank.retire(ticket).unwrap(); // caller must also prove transfer completion
    let next = key(1, 1, 0); // source-local token is not globally monotonic
    bank.reserve(next, 0).unwrap();
    bank.seal(next).unwrap();
    assert!(bank.drained(next).unwrap());
    bank.retire(next).unwrap();
}

#[test]
fn duplicate_consumer_completion_is_terminal() {
    let mut bank = PayloadLease::new(2, 8, 2).unwrap();
    let ticket = key(0, 0, 0);
    bank.reserve(ticket, 8).unwrap();
    let a = bank.consumer(ticket).unwrap();
    let _b = bank.consumer(ticket).unwrap();
    bank.seal(ticket).unwrap();
    bank.complete(a).unwrap();
    assert_eq!(bank.complete(a).unwrap_err(), "PAYLOAD_CONSUMER");
    assert_eq!(bank.drained(ticket).unwrap_err(), "PAYLOAD_FAILED");
}

#[test]
fn old_consumer_cannot_retire_reused_bank() {
    let mut bank = PayloadLease::new(2, 8, 1).unwrap();
    let old = key(0, 0, 10);
    bank.reserve(old, 8).unwrap();
    let consumer = bank.consumer(old).unwrap();
    bank.seal(old).unwrap();
    bank.complete(consumer).unwrap();
    bank.retire(old).unwrap();
    bank.reserve(key(1, 1, 0), 8).unwrap();
    assert_eq!(bank.complete(consumer).unwrap_err(), "PAYLOAD_TICKET");
}

#[test]
fn busy_capacity_and_unsealed_retirement_fail_before_reuse() {
    for case in 0..5 {
        let mut bank = PayloadLease::new(2, 8, 1).unwrap();
        let ticket = key(0, 0, 10);
        if case == 0 {
            assert_eq!(
                bank.reserve(ticket, 9).unwrap_err(),
                "PAYLOAD_BYTE_CAPACITY"
            );
        } else {
            bank.reserve(ticket, 8).unwrap();
            match case {
                1 => assert_eq!(bank.reserve(key(1, 1, 0), 0).unwrap_err(), "PAYLOAD_BUSY"),
                2 => assert_eq!(bank.retire(ticket).unwrap_err(), "PAYLOAD_NOT_DRAINED"),
                3 => {
                    bank.consumer(ticket).unwrap();
                    assert_eq!(bank.consumer(ticket).unwrap_err(), "PAYLOAD_JOB_CAPACITY");
                }
                4 => {
                    bank.seal(ticket).unwrap();
                    assert_eq!(bank.consumer(ticket).unwrap_err(), "PAYLOAD_SEALED");
                }
                _ => unreachable!(),
            }
        }
        assert_eq!(bank.reserve(key(2, 1, 1), 0).unwrap_err(), "PAYLOAD_FAILED");
    }
}
