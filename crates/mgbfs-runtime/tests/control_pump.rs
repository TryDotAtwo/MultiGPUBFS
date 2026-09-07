use mgbfs_runtime::{control_connection::ControlConnection, control_wire::ControlFrame};
use mgbfs_runtime::{
    control_pump::ControlPump,
    control_wire::{Action, Plane},
};
use std::{
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};
fn pair() -> [ControlPump; 2] {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    [
        ControlPump::new(
            2,
            0,
            2,
            vec![None, Some(ControlConnection::new(server, 2, 0, 1).unwrap())],
        )
        .unwrap(),
        ControlPump::new(
            2,
            1,
            2,
            vec![Some(ControlConnection::new(client, 2, 1, 0).unwrap()), None],
        )
        .unwrap(),
    ]
}
fn commands(pumps: &mut [ControlPump; 2]) -> [ControlFrame; 2] {
    let mut out = [None; 2];
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        for (rank, pump) in pumps.iter_mut().enumerate() {
            pump.poll().unwrap();
            if out[rank].is_none() {
                out[rank] = pump.command().unwrap();
            }
        }
        if let [Some(a), Some(b)] = out {
            return [a, b];
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}
#[test]
fn tcp_pump_drives_two_depths_with_alternating_source_and_empty_peer() {
    let mut pumps = pair();
    for depth in 0..2 {
        pumps[(1 - depth) as usize]
            .offer(Plane::Candidate, 7)
            .unwrap();
        for pump in &mut pumps {
            pump.close_source().unwrap();
        }
        let frames = commands(&mut pumps);
        assert!(frames
            .iter()
            .all(|x| x.action == Action::Begin && x.depth == depth && x.epoch == depth * 2));
        for pump in &mut pumps {
            pump.transfer_complete(depth * 2).unwrap();
            pump.consumed(depth * 2).unwrap();
        }
        assert!(commands(&mut pumps)
            .iter()
            .all(|x| x.action == Action::Finalize));
        for pump in &mut pumps {
            pump.finalized(true).unwrap();
        }
        assert!(commands(&mut pumps)
            .iter()
            .all(|x| x.action == Action::Publish && x.depth == depth + 1));
    }
}
#[test]
fn single_rank_pump_preserves_full_control_lifecycle_without_sockets() {
    let mut pump = ControlPump::new(1, 0, 2, vec![None]).unwrap();
    for depth in 0..2 {
        pump.offer(Plane::Candidate, 7).unwrap();
        pump.poll().unwrap();
        let begin = pump.command().unwrap().unwrap();
        assert_eq!(
            (begin.action, begin.depth, begin.epoch),
            (Action::Begin, depth, depth * 2)
        );
        pump.transfer_complete(begin.epoch).unwrap();
        pump.consumed(begin.epoch).unwrap();
        pump.close_source().unwrap();
        pump.poll().unwrap();
        assert_eq!(pump.command().unwrap().unwrap().action, Action::Finalize);
        pump.finalized(true).unwrap();
        pump.poll().unwrap();
        assert_eq!(pump.command().unwrap().unwrap().action, Action::Publish);
        assert!(pump.command().unwrap().is_none());
    }
}

#[test]
fn tcp_descendant_chain_survives_source_close_and_prevents_early_finalize() {
    let mut pumps = pair();
    pumps[1].offer(Plane::Candidate, 10).unwrap();
    for pump in &mut pumps {
        pump.close_source().unwrap();
    }
    let stages = [
        (Plane::Candidate, 1, 10),
        (Plane::Request, 0, 20),
        (Plane::Response, 1, 30),
        (Plane::Receipt, 0, 40),
    ];
    for (epoch, &(plane, source, slot)) in stages.iter().enumerate() {
        let frames = commands(&mut pumps);
        for (rank, frame) in frames.iter().enumerate() {
            assert_eq!(
                (frame.action, frame.plane, frame.epoch),
                (Action::Begin, plane, epoch as u64)
            );
            assert_eq!(frame.slot, if rank == source { slot } else { u64::MAX });
        }
        for pump in &mut pumps {
            pump.transfer_complete(epoch as u64).unwrap();
        }
        // No next offer exists yet; COMPLETE must not release consumer credit
        // or permit Finalize even though both source streams are already closed.
        for _ in 0..32 {
            for pump in &mut pumps {
                pump.poll().unwrap();
                assert!(pump.command().unwrap().is_none());
            }
        }
        if let Some(&(next_plane, next_source, next_slot)) = stages.get(epoch + 1) {
            pumps[next_source].offer(next_plane, next_slot).unwrap();
        }
        for pump in &mut pumps {
            pump.consumed(epoch as u64).unwrap();
        }
    }
    assert!(commands(&mut pumps)
        .iter()
        .all(|f| f.action == Action::Finalize && f.epoch == 4 && f.depth == 0));
    for pump in &mut pumps {
        pump.finalized(true).unwrap();
    }
    assert!(commands(&mut pumps)
        .iter()
        .all(|f| f.action == Action::Publish && f.epoch == 4 && f.depth == 1));
}

#[test]
fn local_protocol_failure_closes_tcp_peer_and_poisoning_is_permanent() {
    let mut pumps = pair();
    pumps[1].offer(Plane::Candidate, 10).unwrap();
    assert!(pumps[1].offer(Plane::Candidate, 10).is_err());
    assert!(pumps[1].poll().is_err());
    assert!(pumps[1].command().is_err());
    let deadline = Instant::now() + Duration::from_secs(5);
    while pumps[0].poll().is_ok() {
        assert!(
            Instant::now() < deadline,
            "peer did not observe terminal close"
        );
        std::thread::yield_now();
    }
    assert!(pumps[0].offer(Plane::Request, 20).is_err());
    assert!(pumps[0].command().is_err());
}

#[test]
fn expired_poll_deadline_closes_peer_and_cannot_be_extended_after_failure() {
    let mut pumps = pair();
    let expired = Instant::now();
    assert!(pumps[1].poll_before(expired).is_err());
    assert!(pumps[1]
        .poll_before(Instant::now() + Duration::from_secs(60))
        .is_err());
    let deadline = Instant::now() + Duration::from_secs(5);
    while pumps[0].poll().is_ok() {
        assert!(
            Instant::now() < deadline,
            "deadline failure did not close peer"
        );
        std::thread::yield_now();
    }
    assert!(pumps[0].command().is_err());
}

#[test]
fn live_poll_deadline_allows_commands_without_waiting_for_expiry() {
    let mut pump = ControlPump::new(1, 0, 1, vec![None]).unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    pump.offer(Plane::Candidate, 99).unwrap();
    pump.poll_before(deadline).unwrap();
    let frame = pump.command().unwrap().unwrap();
    assert_eq!(
        (frame.action, frame.slot, frame.epoch),
        (Action::Begin, 99, 0)
    );
    pump.poll_before(deadline).unwrap();
    assert!(pump.command().unwrap().is_none());
}
#[test]
fn three_tcp_ranks_receive_identical_source_identity() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut root_peers = vec![None];
    let mut clients = Vec::new();
    for rank in 1..3 {
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        root_peers.push(Some(ControlConnection::new(server, 3, 0, rank).unwrap()));
        let mut peers: Vec<_> = (0..3).map(|_| None).collect();
        peers[0] = Some(ControlConnection::new(client, 3, rank, 0).unwrap());
        clients.push(ControlPump::new(3, rank, 2, peers).unwrap());
    }
    let mut pumps = vec![ControlPump::new(3, 0, 2, root_peers).unwrap()];
    pumps.extend(clients);
    pumps[2].offer(Plane::Candidate, 71).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut seen = [false; 3];
    while seen.iter().any(|x| !x) {
        for (rank, pump) in pumps.iter_mut().enumerate() {
            pump.poll_before(deadline).unwrap();
            if let Some(frame) = pump.command().unwrap() {
                assert!(!seen[rank]);
                assert_eq!(
                    (frame.action, frame.source_rank, frame.epoch),
                    (Action::Begin, 2, 0)
                );
                assert_eq!(frame.slot, if rank == 2 { 71 } else { u64::MAX });
                seen[rank] = true;
            }
        }
        std::thread::yield_now();
    }
}

#[test]
fn admitted_commands_fit_all_begin_and_ticket_frames_before_consumer_poll() {
    let mut pumps = admitted_pair();
    for plane in [
        Plane::Candidate,
        Plane::Request,
        Plane::Response,
        Plane::Receipt,
    ] {
        for slot in 0..2 {
            pumps[1].offer(plane, slot).unwrap();
        }
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut described = 0;
    let mut tickets = 0;
    while tickets < 8 {
        pumps[0].poll().unwrap();
        pumps[1].poll().unwrap();
        if let Some(f) = pumps[1].command().unwrap() {
            match f.action {
                Action::Begin => {
                    pumps[1].describe_bytes(f, &[0, 0]).unwrap();
                    described += 1;
                }
                Action::TicketBytes => {
                    tickets += 1;
                }
                _ => panic!("unexpected command"),
            }
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(described, 8);
    // Rank zero has not dispatched any consumer commands yet. All eight
    // BEGIN plus eight TicketBytes entries must fit in preallocated storage.
    let mut received = 0;
    while received < 16 {
        pumps[0].poll().unwrap();
        if pumps[0].command().unwrap().is_some() {
            received += 1;
        }
        assert!(Instant::now() < deadline);
    }
}

fn admitted_pair() -> [ControlPump; 2] {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    [
        ControlPump::new_admitted(
            2,
            0,
            2,
            vec![None, Some(ControlConnection::new(server, 2, 0, 1).unwrap())],
            [64; 4],
        )
        .unwrap(),
        ControlPump::new_admitted(
            2,
            1,
            2,
            vec![Some(ControlConnection::new(client, 2, 1, 0).unwrap()), None],
            [64; 4],
        )
        .unwrap(),
    ]
}

#[test]
fn admitted_tcp_pump_requires_empty_receiver_ack_and_advances_past_finalize() {
    let mut pumps = admitted_pair();
    for depth in 0..2 {
        let source = depth as usize;
        pumps[source].offer(Plane::Candidate, 100 - depth).unwrap();
        let begin = commands(&mut pumps);
        assert!(begin
            .iter()
            .all(|f| f.action == Action::Begin && f.epoch == depth * 2));
        pumps[source]
            .describe_bytes(begin[source], &[32, 0])
            .unwrap();
        let tickets = commands(&mut pumps);
        for (rank, ticket) in tickets.iter().enumerate() {
            assert_eq!(ticket.action, Action::TicketBytes);
            assert_eq!(ticket.destination_rank, rank as u32);
            assert_eq!(ticket.payload_bytes, if rank == 0 { 32 } else { 0 });
        }
        pumps[0].admit_bytes(tickets[0], 32).unwrap();
        // Removing the all-rank ACK gate would expose Launch here.
        for _ in 0..32 {
            for pump in &mut pumps {
                pump.poll().unwrap();
                assert!(pump.command().unwrap().is_none());
            }
        }
        pumps[1].admit_bytes(tickets[1], 0).unwrap();
        let launch = commands(&mut pumps);
        assert!(launch
            .iter()
            .all(|f| f.action == Action::Launch && f.epoch == depth * 2));
        for pump in &mut pumps {
            pump.transfer_complete(depth * 2).unwrap();
            pump.consumed(depth * 2).unwrap();
            pump.close_source().unwrap();
        }
        assert!(commands(&mut pumps)
            .iter()
            .all(|f| f.action == Action::Finalize));
        for pump in &mut pumps {
            pump.finalized(true).unwrap();
        }
        assert!(commands(&mut pumps)
            .iter()
            .all(|f| f.action == Action::Publish));
    }
}

#[test]
fn admitted_pump_rejects_completion_before_launch() {
    let mut pumps = admitted_pair();
    pumps[0].offer(Plane::Candidate, 1).unwrap();
    let _begin = commands(&mut pumps);
    assert!(pumps[0].transfer_complete(0).is_err());
    assert!(pumps[0].poll().is_err());
}

#[test]
fn later_admitted_ticket_waits_for_earlier_epoch_without_releasing_consumers() {
    let mut pumps = admitted_pair();
    pumps[0].offer(Plane::Candidate, 100).unwrap();
    let first = commands(&mut pumps);
    pumps[1].offer(Plane::Candidate, 1).unwrap();
    let second = commands(&mut pumps);
    assert_eq!((first[0].epoch, second[0].epoch), (0, 1));
    pumps[1].describe_bytes(second[1], &[0, 16]).unwrap();
    let tickets = commands(&mut pumps);
    for rank in 0..2 {
        pumps[rank].admit_bytes(tickets[rank], 16).unwrap();
    }
    for _ in 0..32 {
        for pump in &mut pumps {
            pump.poll().unwrap();
            assert!(pump.command().unwrap().is_none());
        }
    }
    pumps[0].describe_bytes(first[0], &[16, 0]).unwrap();
    let tickets = commands(&mut pumps);
    for rank in 0..2 {
        pumps[rank].admit_bytes(tickets[rank], 16).unwrap();
    }
    assert!(commands(&mut pumps)
        .iter()
        .all(|f| f.action == Action::Launch && f.epoch == 0));
    assert!(commands(&mut pumps)
        .iter()
        .all(|f| f.action == Action::Launch && f.epoch == 1));
    for pump in &mut pumps {
        pump.transfer_complete(0).unwrap();
        pump.transfer_complete(1).unwrap();
        // A later consumer may finish first; finalization still needs both.
        pump.consumed(1).unwrap();
        pump.close_source().unwrap();
    }
    for _ in 0..32 {
        for pump in &mut pumps {
            pump.poll().unwrap();
            assert!(pump.command().unwrap().is_none());
        }
    }
    for pump in &mut pumps {
        pump.consumed(0).unwrap();
    }
    assert!(commands(&mut pumps)
        .iter()
        .all(|f| f.action == Action::Finalize && f.epoch == 2));
}

#[test]
fn receiver_capacity_rejection_poisoning_reaches_source_before_launch() {
    let mut pumps = admitted_pair();
    pumps[0].offer(Plane::Candidate, 1).unwrap();
    let begin = commands(&mut pumps);
    pumps[0].describe_bytes(begin[0], &[0, 33]).unwrap();
    let tickets = commands(&mut pumps);
    pumps[0].admit_bytes(tickets[0], 0).unwrap();
    assert!(pumps[1].admit_bytes(tickets[1], 32).is_err());
    assert!(pumps[1].command().is_err());
    let deadline = Instant::now() + Duration::from_secs(5);
    while pumps[0].poll().is_ok() {
        assert!(pumps[0].command().unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(pumps[0].command().is_err());
}

#[test]
fn disjoint_slow_consumers_do_not_exhaust_global_ticket_metadata() {
    let mut pumps = admitted_pair();
    for epoch in 0..2 {
        pumps[0].offer(Plane::Candidate, epoch).unwrap();
        let begin = commands(&mut pumps);
        pumps[0].describe_bytes(begin[0], &[1, 1]).unwrap();
        let tickets = commands(&mut pumps);
        for rank in 0..2 {
            pumps[rank].admit_bytes(tickets[rank], 1).unwrap();
        }
        assert!(commands(&mut pumps)
            .iter()
            .all(|f| f.action == Action::Launch));
        for pump in &mut pumps {
            pump.transfer_complete(epoch).unwrap();
        }
    }
    pumps[0].consumed(0).unwrap();
    pumps[1].consumed(1).unwrap();
    // Each rank has a free bank, although neither global ticket is drained.
    pumps[0].offer(Plane::Candidate, 2).unwrap();
    assert!(commands(&mut pumps)
        .iter()
        .all(|f| f.action == Action::Begin && f.epoch == 2));
}

#[test]
fn root_launch_is_not_exposed_before_peer_command_has_left_outbox() {
    let mut pumps = admitted_pair();
    pumps[0].offer(Plane::Candidate, 1).unwrap();
    let begin = commands(&mut pumps);
    pumps[0].describe_bytes(begin[0], &[1, 1]).unwrap();
    let tickets = commands(&mut pumps);
    for rank in 0..2 {
        pumps[rank].admit_bytes(tickets[rank], 1).unwrap();
    }
    pumps[1].poll().unwrap(); // submit remote ACK
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        pumps[0].poll().unwrap();
        if let Some(f) = pumps[0].command().unwrap() {
            assert_eq!(f.action, Action::Launch);
            break;
        }
        assert!(Instant::now() < deadline);
    }
    // The coordinator may now enter a host-blocking NCCL API. It must not
    // require another root poll to deliver the peer's matching LAUNCH.
    loop {
        pumps[1].poll().unwrap();
        if let Some(f) = pumps[1].command().unwrap() {
            assert_eq!(f.action, Action::Launch);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "root exposed LAUNCH while peer command was still queued"
        );
        std::thread::yield_now();
    }
}
