use mgbfs_runtime::control_connection::ControlConnection;
use mgbfs_runtime::control_wire::{Action, ControlFrame, Plane, NO_SLOT};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    (server, client)
}
fn ready() -> ControlFrame {
    ControlFrame {
        action: Action::Ready,
        rank: 1,
        depth: 0,
        epoch: 0,
        slot: 3,
        plane: Plane::Candidate,
        source_rank: 0,
        fatal_code: 0,
        destination_rank: 0,
        payload_bytes: 0,
    }
}
fn receive(connection: &mut ControlConnection) -> Result<ControlFrame, String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(frame) = connection.poll_receive()? {
            return Ok(frame);
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}
fn send(connection: &mut ControlConnection, frame: ControlFrame) {
    connection.enqueue(frame).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !connection.poll_send().unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}
#[test]
fn three_rank_tcp_byte_admission_waits_for_empty_receiver_ack() {
    use mgbfs_runtime::{byte_admission::ByteAdmission, scatter_admission::TicketKey};
    let (server1, client1) = pair();
    let (server2, client2) = pair();
    let mut root1 = ControlConnection::new(server1, 3, 0, 1).unwrap();
    let mut peer1 = ControlConnection::new(client1, 3, 1, 0).unwrap();
    let mut root2 = ControlConnection::new(server2, 3, 0, 2).unwrap();
    let mut peer2 = ControlConnection::new(client2, 3, 2, 0).unwrap();
    let key = TicketKey {
        depth: 0,
        epoch: 0,
        source: 2,
        plane: Plane::Candidate,
        generation: 3,
    };
    let mut admission = ByteAdmission::new(3).unwrap();
    admission.begin(key, 80).unwrap();
    let mut out = [ready(); 3];
    for (dst, bytes, complete) in [(2, 48, false), (0, 32, false), (1, 0, true)] {
        let frame = ControlFrame {
            action: Action::OfferBytes,
            rank: 2,
            source_rank: 2,
            destination_rank: dst,
            payload_bytes: bytes,
            ..ready()
        };
        send(&mut peer2, frame);
        assert_eq!(
            admission
                .offer(receive(&mut root2).unwrap(), &mut out)
                .unwrap(),
            complete
        );
    }
    send(&mut root1, out[1]);
    send(&mut root2, out[2]);
    let ticket1 = receive(&mut peer1).unwrap();
    let ticket2 = receive(&mut peer2).unwrap();
    assert_eq!(ticket1.payload_bytes, 0);
    assert_eq!(ticket2.payload_bytes, 48);
    admission
        .ack(ControlFrame {
            action: Action::Admitted,
            rank: 0,
            ..out[0]
        })
        .unwrap();
    send(
        &mut peer2,
        ControlFrame {
            action: Action::Admitted,
            rank: 2,
            payload_bytes: 0,
            ..ticket2
        },
    );
    admission.ack(receive(&mut root2).unwrap()).unwrap();
    let mut next = 0;
    assert!(!admission.launch(&mut next, &mut out).unwrap());
    assert!(peer1.poll_receive().unwrap().is_none());
    assert!(peer2.poll_receive().unwrap().is_none());
    send(
        &mut peer1,
        ControlFrame {
            action: Action::Admitted,
            rank: 1,
            ..ticket1
        },
    );
    admission.ack(receive(&mut root1).unwrap()).unwrap();
    assert!(admission.launch(&mut next, &mut out).unwrap());
    send(&mut root1, out[1]);
    send(&mut root2, out[2]);
    for command in [receive(&mut peer1).unwrap(), receive(&mut peer2).unwrap()] {
        assert_eq!(command.action, Action::Launch);
        assert_eq!(
            (command.epoch, command.source_rank, command.slot),
            (0, 2, 3)
        );
    }
    assert_eq!(next, 1);
}
#[test]
fn tcp_admission_metadata_roundtrip_preserves_sizes_and_source() {
    let (server, client) = pair();
    let mut root = ControlConnection::new(server, 2, 0, 1).unwrap();
    let mut peer = ControlConnection::new(client, 2, 1, 0).unwrap();
    let offer = ControlFrame {
        action: Action::OfferBytes,
        source_rank: 1,
        destination_rank: 1,
        payload_bytes: 1u64 << 34,
        epoch: 19,
        ..ready()
    };
    send(&mut peer, offer);
    assert_eq!(receive(&mut root).unwrap(), offer);
    let ticket = ControlFrame {
        action: Action::TicketBytes,
        rank: 0,
        ..offer
    };
    send(&mut root, ticket);
    assert_eq!(receive(&mut peer).unwrap(), ticket);
    let ack = ControlFrame {
        action: Action::Admitted,
        rank: 1,
        ..ticket
    };
    send(&mut peer, ack);
    assert_eq!(receive(&mut root).unwrap(), ack);
    let launch = ControlFrame {
        action: Action::Launch,
        rank: 0,
        destination_rank: 0,
        payload_bytes: 0,
        ..ticket
    };
    send(&mut root, launch);
    assert_eq!(receive(&mut peer).unwrap(), launch);
    assert!(peer.enqueue(ControlFrame { rank: 1, ..launch }).is_err());
}

#[test]
fn bounded_outbox_preserves_order_and_capacity_including_pending_frame() {
    let (server, client) = pair();
    let mut root = ControlConnection::new(server, 2, 0, 1).unwrap();
    let mut peer = ControlConnection::with_send_capacity(client, 2, 1, 0, 2).unwrap();
    peer.enqueue(ready()).unwrap();
    peer.enqueue(ControlFrame { slot: 7, ..ready() }).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !peer.poll_send().unwrap() {
        assert!(Instant::now() < deadline);
    }
    assert_eq!(receive(&mut root).unwrap().slot, 3);
    assert_eq!(receive(&mut root).unwrap().slot, 7);
    // Exercise wrapping physical queue indices after retirement.
    send(&mut peer, ControlFrame { slot: 9, ..ready() });
    assert_eq!(receive(&mut root).unwrap().slot, 9);
    peer.enqueue(ready()).unwrap();
    peer.enqueue(ControlFrame { slot: 7, ..ready() }).unwrap();
    assert!(peer.enqueue(ControlFrame { slot: 9, ..ready() }).is_err());
    assert!(peer.poll_send().is_err());
}

#[test]
fn rank_bound_nonblocking_ready_begin_complete_roundtrip() {
    let (server, client) = pair();
    let mut root = ControlConnection::new(server, 2, 0, 1).unwrap();
    let mut peer = ControlConnection::new(client, 2, 1, 0).unwrap();
    assert_eq!(root.poll_receive().unwrap(), None);
    send(&mut peer, ready());
    assert_eq!(receive(&mut root).unwrap(), ready());
    let begin = ControlFrame {
        action: Action::Begin,
        rank: 0,
        epoch: 42,
        ..ready()
    };
    send(&mut root, begin);
    assert_eq!(receive(&mut peer).unwrap(), begin);
    let ack = ControlFrame {
        action: Action::Complete,
        rank: 1,
        slot: NO_SLOT,
        ..begin
    };
    send(&mut peer, ack);
    assert_eq!(receive(&mut root).unwrap(), ack);
}

#[test]
fn tcp_begin_uses_registered_epoch_slot_while_a_later_ready_is_pending() {
    use mgbfs_runtime::rank_epochs::RankEpochs;
    let (server, client) = pair();
    let mut root = ControlConnection::new(server, 2, 0, 1).unwrap();
    let mut peer = ControlConnection::new(client, 2, 1, 0).unwrap();
    let mut epochs = RankEpochs::new(2, 1, 2).unwrap();
    send(&mut peer, epochs.offer(Plane::Candidate, 3).unwrap());
    let first = receive(&mut root).unwrap();
    // The later offer is registered locally before the root's first BEGIN.
    send(&mut peer, epochs.offer(Plane::Candidate, 7).unwrap());
    send(
        &mut root,
        ControlFrame {
            action: Action::Begin,
            rank: 0,
            source_rank: 1,
            ..first
        },
    );
    epochs.begin(receive(&mut peer).unwrap()).unwrap();
    send(&mut peer, epochs.transfer_complete(0).unwrap());
    send(&mut peer, epochs.consume(0).unwrap());
    let later = receive(&mut root).unwrap();
    assert_eq!((later.action, later.slot), (Action::Ready, 7));
    let complete = receive(&mut root).unwrap();
    assert_eq!((complete.action, complete.epoch), (Action::Complete, 0));
    assert_eq!(receive(&mut root).unwrap().action, Action::Consumed);
    send(
        &mut root,
        ControlFrame {
            action: Action::Begin,
            rank: 0,
            epoch: 1,
            source_rank: 1,
            ..later
        },
    );
    epochs.begin(receive(&mut peer).unwrap()).unwrap();
    // Retired slot 3 is reusable even while slot 7's consumers still own it.
    send(&mut peer, epochs.offer(Plane::Candidate, 3).unwrap());
    assert_eq!(receive(&mut root).unwrap().slot, 3);
    send(&mut peer, epochs.transfer_complete(1).unwrap());
    assert_eq!(receive(&mut root).unwrap().epoch, 1);
    send(&mut peer, epochs.consume(1).unwrap());
    assert_eq!(receive(&mut root).unwrap().action, Action::Consumed);
}

#[test]
fn tcp_coordinator_publishes_two_depths_only_after_both_ranks_finalize() {
    use mgbfs_runtime::{epoch_coordinator::EpochCoordinator, rank_epochs::RankEpochs};
    let (server, client) = pair();
    let mut root = ControlConnection::new(server, 2, 0, 1).unwrap();
    let mut peer = ControlConnection::new(client, 2, 1, 0).unwrap();
    let mut coordinator = EpochCoordinator::new(2, 2).unwrap();
    let mut local = RankEpochs::new(2, 0, 2).unwrap();
    let mut remote = RankEpochs::new(2, 1, 2).unwrap();
    let mut frames = [ready(); 2];
    for depth in 0..2 {
        if depth == 0 {
            coordinator
                .receive(local.offer(Plane::Candidate, 7).unwrap())
                .unwrap();
        } else {
            send(&mut peer, remote.offer(Plane::Candidate, 9).unwrap());
            coordinator.receive(receive(&mut root).unwrap()).unwrap();
        }
        assert!(coordinator.issue(&mut frames).unwrap());
        assert_eq!((frames[0].depth, frames[0].epoch), (depth, depth * 2));
        local.begin(frames[0]).unwrap();
        send(&mut root, frames[1]);
        remote.begin(receive(&mut peer).unwrap()).unwrap();
        coordinator
            .receive(local.transfer_complete(depth * 2).unwrap())
            .unwrap();
        send(&mut peer, remote.transfer_complete(depth * 2).unwrap());
        coordinator.receive(receive(&mut root).unwrap()).unwrap();
        coordinator
            .receive(local.consume(depth * 2).unwrap())
            .unwrap();
        send(&mut peer, remote.consume(depth * 2).unwrap());
        coordinator.receive(receive(&mut root).unwrap()).unwrap();
        let close = ControlFrame {
            action: Action::SourceClosed,
            rank: 0,
            depth,
            epoch: 0,
            slot: NO_SLOT,
            plane: Plane::None,
            source_rank: 0,
            fatal_code: 0,
            destination_rank: 0,
            payload_bytes: 0,
        };
        coordinator.receive(close).unwrap();
        send(&mut peer, ControlFrame { rank: 1, ..close });
        coordinator.receive(receive(&mut root).unwrap()).unwrap();
        assert!(coordinator.issue(&mut frames).unwrap());
        assert_eq!(frames[0].action, Action::Finalize);
        coordinator
            .receive(local.finish_depth(frames[0], true).unwrap())
            .unwrap();
        send(&mut root, frames[1]);
        let finalize = receive(&mut peer).unwrap();
        assert!(!coordinator.issue(&mut frames).unwrap());
        send(&mut peer, remote.finish_depth(finalize, true).unwrap());
        coordinator.receive(receive(&mut root).unwrap()).unwrap();
        assert!(coordinator.issue(&mut frames).unwrap());
        assert_eq!(
            (frames[0].action, frames[0].depth, frames[0].epoch),
            (Action::Publish, depth + 1, depth * 2 + 1)
        );
        local.publish(frames[0]).unwrap();
        send(&mut root, frames[1]);
        remote.publish(receive(&mut peer).unwrap()).unwrap();
    }
}

#[test]
fn forged_sender_poisoning_is_terminal() {
    let (server, mut client) = pair();
    let mut root = ControlConnection::new(server, 2, 0, 1).unwrap();
    ControlFrame { rank: 0, ..ready() }
        .write_to(&mut client, 2)
        .unwrap();
    assert!(receive(&mut root)
        .unwrap_err()
        .contains("CONTROL_PEER_RANK"));
    assert!(root.poll_receive().is_err());
    assert!(root.poll_send().is_err());
}

#[test]
fn client_cannot_issue_begin_or_replace_a_pending_message() {
    let (server, client) = pair();
    let _root = ControlConnection::new(server, 2, 0, 1).unwrap();
    let mut peer = ControlConnection::new(client, 2, 1, 0).unwrap();
    let spoofed = ControlFrame {
        action: Action::Begin,
        rank: 0,
        ..ready()
    };
    assert!(peer.enqueue(spoofed).is_err());
    assert!(peer.enqueue(ready()).is_err());
    let (server, client) = pair();
    let _root = ControlConnection::new(server, 2, 0, 1).unwrap();
    let mut peer = ControlConnection::new(client, 2, 1, 0).unwrap();
    peer.enqueue(ready()).unwrap();
    assert!(peer.enqueue(ready()).is_err());
    assert!(peer.poll_send().is_err());
}

#[test]
fn rejects_wrong_command_direction_even_with_matching_peer_rank() {
    let (mut server, client) = pair();
    let mut peer = ControlConnection::new(client, 2, 1, 0).unwrap();
    ControlFrame { rank: 0, ..ready() }
        .write_to(&mut server, 2)
        .unwrap();
    assert!(receive(&mut peer)
        .unwrap_err()
        .contains("CONTROL_DIRECTION"));
    assert!(peer.poll_send().is_err());
}

#[test]
fn only_valid_star_connections_are_admitted() {
    for (world, local, peer) in [(0, 0, 1), (2, 2, 0), (2, 0, 2), (2, 1, 1), (3, 1, 2)] {
        let (server, _client) = pair();
        assert!(ControlConnection::new(server, world, local, peer).is_err());
    }
}
