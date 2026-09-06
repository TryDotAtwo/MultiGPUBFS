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
        fatal_code: 0,
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
