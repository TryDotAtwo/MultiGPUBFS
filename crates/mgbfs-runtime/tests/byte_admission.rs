use mgbfs_runtime::byte_admission::ByteAdmission;
use mgbfs_runtime::control_wire::{Action, ControlFrame, Plane};
use mgbfs_runtime::scatter_admission::TicketKey;
fn key() -> TicketKey {
    TicketKey {
        depth: 4,
        epoch: 9,
        source: 2,
        plane: Plane::Candidate,
        generation: 7,
    }
}
fn offer(dst: u32, bytes: u64) -> ControlFrame {
    ControlFrame {
        action: Action::OfferBytes,
        rank: 2,
        source_rank: 2,
        destination_rank: dst,
        payload_bytes: bytes,
        depth: 4,
        epoch: 9,
        slot: 7,
        plane: Plane::Candidate,
        fatal_code: 0,
    }
}
#[test]
fn complete_source_description_and_all_acks_precede_launch() {
    let mut a = ByteAdmission::new(3).unwrap();
    a.begin(key(), 80).unwrap();
    let mut out = [offer(0, 0); 3];
    assert!(!a.offer(offer(2, 48), &mut out).unwrap());
    assert!(!a.offer(offer(0, 32), &mut out).unwrap());
    assert!(a.offer(offer(1, 0), &mut out).unwrap());
    assert_eq!(
        out.map(|f| (f.action, f.rank, f.destination_rank, f.payload_bytes)),
        [
            (Action::TicketBytes, 0, 0, 32),
            (Action::TicketBytes, 0, 1, 0),
            (Action::TicketBytes, 0, 2, 48)
        ]
    );
    let mut next = 9;
    for rank in [2usize, 0] {
        a.ack(ControlFrame {
            action: Action::Admitted,
            rank: rank as u32,
            payload_bytes: if rank == 0 { 32 } else { 0 },
            ..out[rank]
        })
        .unwrap();
    }
    assert!(!a.launch(&mut next, &mut out).unwrap());
    a.ack(ControlFrame {
        action: Action::Admitted,
        rank: 1,
        ..out[1]
    })
    .unwrap();
    assert!(a.launch(&mut next, &mut out).unwrap());
    assert_eq!(next, 10);
    assert!(out
        .iter()
        .all(|f| f.action == Action::Launch && f.source_rank == 2));
    a.retire(key()).unwrap();
    a.begin(
        TicketKey {
            epoch: 10,
            generation: 8,
            ..key()
        },
        80,
    )
    .unwrap();
    assert!(a.offer(offer(0, 32), &mut out).is_err());
}
#[test]
fn malformed_or_duplicate_description_is_terminal() {
    for bad in [
        ControlFrame {
            rank: 1,
            ..offer(0, 32)
        },
        offer(0, 81),
    ] {
        let mut a = ByteAdmission::new(3).unwrap();
        a.begin(key(), 80).unwrap();
        let mut out = [offer(0, 0); 3];
        assert!(a.offer(bad, &mut out).is_err());
        assert!(a.offer(offer(1, 0), &mut out).is_err());
    }
    let mut a = ByteAdmission::new(3).unwrap();
    a.begin(key(), 80).unwrap();
    let mut out = [offer(0, 0); 3];
    a.offer(offer(0, 32), &mut out).unwrap();
    assert!(a.offer(offer(0, 32), &mut out).is_err());
}
