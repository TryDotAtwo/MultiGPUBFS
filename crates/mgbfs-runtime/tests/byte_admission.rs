use mgbfs_runtime::byte_admission::{ByteAdmission, RankByteAdmission};
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
#[test]
fn receiver_reuse_accepts_another_sources_lower_slot_token() {
    let mut a = RankByteAdmission::new(3, 1).unwrap();
    let first = ControlFrame {
        action: Action::TicketBytes,
        rank: 0,
        ..offer(1, 32)
    };
    a.accept(first, 32).unwrap();
    a.launch(ControlFrame {
        action: Action::Launch,
        destination_rank: 0,
        payload_bytes: 0,
        ..first
    })
    .unwrap();
    a.retire(key()).unwrap();
    let second = ControlFrame {
        epoch: 10,
        source_rank: 0,
        slot: 0,
        ..first
    };
    a.accept(second, 32).unwrap();
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
fn receiver_reserves_own_ticket_before_accepting_launch() {
    let mut rank = RankByteAdmission::new(3, 1).unwrap();
    let ticket = ControlFrame {
        action: Action::TicketBytes,
        rank: 0,
        ..offer(1, 32)
    };
    let ack = rank.accept(ticket, 32).unwrap();
    assert_eq!(
        (
            ack.action,
            ack.rank,
            ack.destination_rank,
            ack.payload_bytes
        ),
        (Action::Admitted, 1, 1, 32)
    );
    let launch = ControlFrame {
        action: Action::Launch,
        destination_rank: 0,
        payload_bytes: 0,
        ..ticket
    };
    assert_eq!(rank.launch(launch).unwrap(), 32);
    assert!(rank.launch(launch).is_err());
    for ticket in [
        ControlFrame {
            destination_rank: 0,
            ..ticket
        },
        ControlFrame {
            payload_bytes: 33,
            ..ticket
        },
    ] {
        let mut rank = RankByteAdmission::new(3, 1).unwrap();
        assert!(rank.accept(ticket, 32).is_err());
        assert!(rank.launch(launch).is_err());
    }
}
#[test]
fn source_view_and_reuse_do_not_accept_stale_launch() {
    let mut rank = RankByteAdmission::new(3, 2).unwrap();
    let ticket = ControlFrame {
        action: Action::TicketBytes,
        rank: 0,
        ..offer(2, 48)
    };
    rank.accept(ticket, 0).unwrap();
    let launch = ControlFrame {
        action: Action::Launch,
        destination_rank: 0,
        payload_bytes: 0,
        ..ticket
    };
    assert_eq!(rank.launch(launch).unwrap(), 48);
    rank.retire(key()).unwrap();
    rank.accept(
        ControlFrame {
            epoch: 10,
            slot: 8,
            ..ticket
        },
        0,
    )
    .unwrap();
    assert!(rank.launch(launch).is_err());
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
