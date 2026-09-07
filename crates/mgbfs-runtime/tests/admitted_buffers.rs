use mgbfs_runtime::{
    admitted_buffers::{AdmittedBuffers, BufferEvent},
    control_wire::Plane,
};

#[test]
fn submission_sequence_survives_finalize_and_rejects_reordered_launches() {
    let mut d = AdmittedBuffers::new(1, 0, 2, vec![None], [16; 4], 1).unwrap();
    for depth in 0..3 {
        let h = d.reserve(Plane::Candidate, depth).unwrap().unwrap();
        d.ready(h, &[16]).unwrap();
        let l = (0..64)
            .find_map(|_| match d.poll().unwrap() {
                Some(BufferEvent::Launch(l)) => Some(l),
                _ => None,
            })
            .unwrap();
        let view = d.payload_view(l).unwrap();
        assert!(view.source_pool);
        assert_eq!((view.offset, view.bytes), (l.source_offset.unwrap(), 16));
        d.submit(l, || Ok(())).unwrap();
        d.seal(l).unwrap();
        d.transfer_complete(l).unwrap();
        d.consume(l).unwrap();
        d.close_source().unwrap();
        let mut published = false;
        for _ in 0..64 {
            match d.poll().unwrap() {
                Some(BufferEvent::Finalize(_)) => d.finalized(true).unwrap(),
                Some(BufferEvent::Publish(f)) => {
                    assert_eq!(f.depth, depth + 1);
                    published = true;
                    break;
                }
                Some(BufferEvent::Launch(_)) => panic!("unexpected launch"),
                None => {}
            }
        }
        assert!(published);
    }
    for _ in 0..2 {
        let h = d.reserve(Plane::Candidate, 3).unwrap().unwrap();
        d.ready(h, &[16]).unwrap();
    }
    let mut launches = Vec::new();
    for _ in 0..128 {
        if let Some(BufferEvent::Launch(l)) = d.poll().unwrap() {
            launches.push(l);
        }
        if launches.len() == 2 {
            break;
        }
    }
    assert_eq!(launches.len(), 2);
    assert!(d
        .submit(launches[1], || panic!("out-of-order enqueue"))
        .is_err());
}

#[test]
fn native_submission_failure_and_duplicate_never_allow_reuse() {
    for mode in 0..3 {
        let mut d = AdmittedBuffers::new(1, 0, 1, vec![None], [16; 4], 1).unwrap();
        let h = d.reserve(Plane::Candidate, 0).unwrap().unwrap();
        d.ready(h, &[16]).unwrap();
        let l = (0..64)
            .find_map(|_| match d.poll().unwrap() {
                Some(BufferEvent::Launch(l)) => Some(l),
                _ => None,
            })
            .unwrap();
        match mode {
            0 => assert!(d.transfer_complete(l).is_err()),
            1 => {
                assert_eq!(
                    d.submit(l, || Err("NATIVE_INJECTED".into())).unwrap_err(),
                    "NATIVE_INJECTED"
                );
            }
            _ => {
                d.submit(l, || Ok(())).unwrap();
                assert!(d.submit(l, || panic!("duplicate enqueue")).is_err());
            }
        }
        assert!(d.reserve(Plane::Candidate, 0).is_err());
        assert!(d.poll().is_err());
    }
}

#[test]
fn source_and_receive_reservations_follow_actual_pump_commands() {
    let mut d = AdmittedBuffers::new(1, 0, 2, vec![None], [257; 4], 2).unwrap();
    let slow = d.reserve(Plane::Candidate, 0).unwrap().unwrap();
    let fast = d.reserve(Plane::Candidate, 0).unwrap().unwrap();
    assert_eq!(d.source_offset(slow).unwrap(), 0);
    assert_eq!(d.source_offset(fast).unwrap(), 512);
    d.ready(fast, &[16]).unwrap();
    let launch = (0..64)
        .find_map(|_| match d.poll().unwrap() {
            Some(BufferEvent::Launch(l)) => Some(l),
            None => None,
            _ => panic!("unexpected finalization"),
        })
        .expect("launch");
    assert_eq!(launch.source_offset, Some(512));
    assert_eq!(launch.receive_offset, 0);
    assert_eq!(launch.bytes, 16);
    let reader = d.consumer(launch).unwrap();
    d.seal(launch).unwrap();
    d.submit(launch, || Ok(())).unwrap();
    d.transfer_complete(launch).unwrap();
    assert!(!d.drained(launch).unwrap());
    assert!(d.reserve(Plane::Candidate, 0).unwrap().is_none());
    d.complete(reader).unwrap();
    assert!(d.drained(launch).unwrap());
    d.consume(launch).unwrap();
    let reused = d.reserve(Plane::Candidate, 0).unwrap().unwrap();
    assert_eq!(d.source_offset(reused).unwrap(), 512);
    assert_eq!(d.source_offset(slow).unwrap(), 0);
}

#[test]
fn unoffered_generation_prevents_source_close() {
    let mut d = AdmittedBuffers::new(1, 0, 1, vec![None], [16; 4], 1).unwrap();
    let _pending = d.reserve(Plane::Candidate, 0).unwrap().unwrap();
    assert!(d.close_source().is_err());
    assert!(d.poll().is_err());
}

#[test]
fn transport_allocation_ledger_matches_all_independent_physical_pools() {
    let d = AdmittedBuffers::new(1, 0, 2, vec![None], [0, 1, 257, 512], 1).unwrap();
    let ledger = d.device_ledger().unwrap();
    assert_eq!(ledger.total(), 6144);
    let got: Vec<_> = ledger
        .allocations
        .iter()
        .map(|a| (a.name.as_str(), a.reserved_bytes))
        .collect();
    assert_eq!(
        got,
        vec![
            ("candidate.source", 512),
            ("candidate.receive", 512),
            ("request.source", 512),
            ("request.receive", 512),
            ("response.source", 1024),
            ("response.receive", 1024),
            ("receipt.source", 1024),
            ("receipt.receive", 1024),
        ]
    );
}

#[test]
fn closed_or_wrong_depth_cannot_reserve_new_candidate_storage() {
    let mut d = AdmittedBuffers::new(1, 0, 1, vec![None], [16; 4], 1).unwrap();
    d.close_source().unwrap();
    assert!(d.reserve(Plane::Candidate, 0).is_err());
    let mut d = AdmittedBuffers::new(1, 0, 1, vec![None], [16; 4], 1).unwrap();
    assert!(d.reserve(Plane::Candidate, 1).is_err());
}

#[test]
fn finalization_cannot_ignore_reserved_unoffered_response() {
    let mut d = AdmittedBuffers::new(1, 0, 1, vec![None], [16; 4], 1).unwrap();
    let _response = d.reserve(Plane::Response, 0).unwrap().unwrap();
    d.close_source().unwrap();
    let finalizing = (0..64).any(|_| matches!(d.poll().unwrap(), Some(BufferEvent::Finalize(_))));
    assert!(finalizing);
    assert!(d.finalized(true).is_err());
}

#[test]
fn ticket_reservation_does_not_authorize_consumer_before_launch() {
    use mgbfs_runtime::{admitted_buffers::BufferLaunch, scatter_admission::TicketKey};
    let mut d = AdmittedBuffers::new(1, 0, 1, vec![None], [16; 4], 1).unwrap();
    let h = d.reserve(Plane::Candidate, 0).unwrap().unwrap();
    d.ready(h, &[16]).unwrap();
    assert!(d.poll().unwrap().is_none()); // BEGIN
    assert!(d.poll().unwrap().is_none()); // TicketBytes, not LAUNCH
    let forged = BufferLaunch {
        key: TicketKey {
            depth: 0,
            epoch: 0,
            source: 0,
            plane: Plane::Candidate,
            generation: 0,
        },
        source_offset: Some(0),
        receive_offset: 0,
        bytes: 16,
    };
    assert!(d.consumer(forged).is_err());
}

#[test]
fn two_tcp_ranks_hold_empty_receiver_ticket_until_consumer_drain() {
    use mgbfs_runtime::control_connection::ControlConnection;
    use std::{
        net::{TcpListener, TcpStream},
        time::{Duration, Instant},
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    let mut ranks = [
        AdmittedBuffers::new(
            2,
            0,
            2,
            vec![None, Some(ControlConnection::new(server, 2, 0, 1).unwrap())],
            [32; 4],
            1,
        )
        .unwrap(),
        AdmittedBuffers::new(
            2,
            1,
            2,
            vec![Some(ControlConnection::new(client, 2, 1, 0).unwrap()), None],
            [32; 4],
            1,
        )
        .unwrap(),
    ];
    let source = ranks[1].reserve(Plane::Candidate, 0).unwrap().unwrap();
    ranks[1].ready(source, &[0, 32]).unwrap();
    for d in &mut ranks {
        d.close_source().unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut launches = [None, None];
    while launches.iter().any(Option::is_none) {
        for (i, d) in ranks.iter_mut().enumerate() {
            if let Some(e) = d.poll().unwrap() {
                match e {
                    BufferEvent::Launch(l) => launches[i] = Some(l),
                    _ => panic!("early finalize"),
                }
            }
        }
        assert!(Instant::now() < deadline);
    }
    let a = launches[0].unwrap();
    let b = launches[1].unwrap();
    assert_eq!(a.bytes, 0);
    assert_eq!(a.source_offset, None);
    assert_eq!(b.bytes, 32);
    assert_eq!(b.source_offset, Some(0));
    let mut sizes = [99; 2];
    assert!(!ranks[0].source_sizes(a, &mut sizes).unwrap());
    assert_eq!(sizes, [0, 0]);
    assert!(ranks[1].source_sizes(b, &mut sizes).unwrap());
    assert_eq!(sizes, [0, 32]);
    ranks[0].seal(a).unwrap();
    let reader = ranks[1].consumer(b).unwrap();
    ranks[1].seal(b).unwrap();
    ranks[0].submit(a, || Ok(())).unwrap();
    ranks[1].submit(b, || Ok(())).unwrap();
    ranks[0].transfer_complete(a).unwrap();
    ranks[1].transfer_complete(b).unwrap();
    ranks[0].consume(a).unwrap();
    for _ in 0..16 {
        for d in &mut ranks {
            assert!(d.poll().unwrap().is_none());
        }
    }
    ranks[1].complete(reader).unwrap();
    ranks[1].consume(b).unwrap();
    let mut published = [false; 2];
    while !published.iter().all(|x| *x) {
        for (i, d) in ranks.iter_mut().enumerate() {
            match d.poll().unwrap() {
                Some(BufferEvent::Finalize(_)) => d.finalized(true).unwrap(),
                Some(BufferEvent::Publish(f)) => {
                    assert_eq!(f.depth, 1);
                    published[i] = true;
                }
                None => (),
                _ => panic!("unexpected launch"),
            }
        }
        assert!(Instant::now() < deadline);
    }
}
