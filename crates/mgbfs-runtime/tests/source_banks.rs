use mgbfs_runtime::{control_wire::Plane, scatter_admission::TicketKey, source_banks::SourceBanks};

#[test]
fn producer_banks_bind_in_ready_order_not_allocation_order() {
    let mut banks = SourceBanks::new(0, 2, 257).unwrap();
    let a = banks.reserve(0).unwrap().unwrap();
    let b = banks.reserve(0).unwrap().unwrap();
    assert_eq!(banks.offset(a).unwrap(), 0);
    assert_eq!(banks.offset(b).unwrap(), 512);
    assert_eq!(banks.bytes(), 1024);
    assert!(banks.reserve(0).unwrap().is_none());
    // Batch b finishes generation first. Epoch zero must bind b, not a.
    banks.ready(b).unwrap();
    let kb = TicketKey {
        depth: 0,
        epoch: 0,
        source: 0,
        plane: Plane::Candidate,
        generation: b.token(),
    };
    banks.bind(b, kb).unwrap();
    banks.ready(a).unwrap();
    let ka = TicketKey {
        epoch: 1,
        generation: a.token(),
        ..kb
    };
    banks.bind(a, ka).unwrap();
    banks.retire(b, kb).unwrap();
    let c = banks.reserve(0).unwrap().unwrap();
    assert_eq!(banks.offset(c).unwrap(), 512);
    assert_ne!(c.token(), b.token());
    assert_eq!(banks.offset(a).unwrap(), 0);
    banks.retire(a, ka).unwrap();
}

#[test]
fn stale_source_handle_cannot_address_reused_bank() {
    let mut banks = SourceBanks::new(0, 1, 16).unwrap();
    let a = banks.reserve(0).unwrap().unwrap();
    banks.ready(a).unwrap();
    let key = TicketKey {
        depth: 0,
        epoch: 0,
        source: 0,
        plane: Plane::Candidate,
        generation: a.token(),
    };
    banks.bind(a, key).unwrap();
    banks.retire(a, key).unwrap();
    let _b = banks.reserve(0).unwrap().unwrap();
    assert!(banks.offset(a).is_err());
    assert!(banks.reserve(0).is_err());
}

#[test]
fn source_binding_rejects_unready_wrong_identity_and_duplicate_epoch() {
    for bad in 0..5 {
        let mut banks = SourceBanks::new(0, 2, 16).unwrap();
        let a = banks.reserve(3).unwrap().unwrap();
        let mut key = TicketKey {
            depth: 3,
            epoch: 0,
            source: 0,
            plane: Plane::Candidate,
            generation: a.token(),
        };
        if bad != 0 {
            banks.ready(a).unwrap();
        }
        match bad {
            1 => key.source = 1,
            2 => key.depth = 2,
            3 => key.generation = 99,
            4 => key.plane = Plane::None,
            _ => (),
        }
        assert!(banks.bind(a, key).is_err());
        assert!(banks.reserve(3).is_err());
    }
    let mut banks = SourceBanks::new(0, 2, 16).unwrap();
    let a = banks.reserve(0).unwrap().unwrap();
    let b = banks.reserve(0).unwrap().unwrap();
    banks.ready(a).unwrap();
    banks.ready(b).unwrap();
    let key = TicketKey {
        depth: 0,
        epoch: 0,
        source: 0,
        plane: Plane::Candidate,
        generation: a.token(),
    };
    banks.bind(a, key).unwrap();
    assert!(banks
        .bind(
            b,
            TicketKey {
                generation: b.token(),
                ..key
            }
        )
        .is_err());
}

#[test]
fn source_capacity_and_unbound_retirement_are_rejected() {
    assert!(SourceBanks::new(0, 0, 16).is_err());
    assert!(SourceBanks::new(0, 1, u64::MAX).is_err());
    assert!(SourceBanks::new(0, usize::MAX, 256).is_err());
    let mut banks = SourceBanks::new(0, 1, 0).unwrap();
    assert_eq!(banks.bytes(), 256);
    let a = banks.reserve(0).unwrap().unwrap();
    let key = TicketKey {
        depth: 0,
        epoch: 0,
        source: 0,
        plane: Plane::Candidate,
        generation: a.token(),
    };
    assert!(banks.retire(a, key).is_err());
}
