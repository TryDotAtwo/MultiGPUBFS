use mgbfs_runtime::control_connection::ControlConnection;
use mgbfs_runtime::control_handshake::RunIdentity;
use mgbfs_runtime::control_wire::{Action, ControlFrame, Plane};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

fn identity() -> RunIdentity {
    RunIdentity {
        config_digest: [7; 32],
        run_id: [9; 16],
    }
}

#[test]
fn silent_peer_has_stable_timeout_error() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let _peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (stream, _) = listener.accept().unwrap();
    let started = Instant::now();
    let result = ControlConnection::accept_peer(stream, 2, identity(), Duration::from_millis(50));
    assert_eq!(result.err().unwrap(), "CONTROL_HANDSHAKE_TIMEOUT");
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[test]
fn client_rejects_wrong_coordinator_identity_after_sending_hello() {
    use std::io::{Read, Write};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut hello = [0; 80];
        stream.read_exact(&mut hello).unwrap();
        assert_eq!(&hello[..8], b"MGBHEL01");
        assert_eq!(hello[16], 1);
        hello[16] = 0;
        hello[56] ^= 1;
        stream.write_all(&hello).unwrap();
    });
    let stream = TcpStream::connect(address).unwrap();
    assert_eq!(
        ControlConnection::connect_peer(stream, 2, 1, identity(), Duration::from_secs(3))
            .err()
            .unwrap(),
        "CONTROL_HANDSHAKE_IDENTITY"
    );
    server.join().unwrap();
}

#[test]
fn fragmented_hello_is_accepted_without_consuming_following_ready() {
    use std::io::{Read, Write};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let (_, mut connection) =
            ControlConnection::accept_peer(stream, 2, identity(), Duration::from_secs(3)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(frame) = connection.poll_receive().unwrap() {
                assert_eq!(frame.slot, 7);
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    });
    let mut peer = TcpStream::connect(address).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    let mut hello = [0u8; 80];
    hello[..8].copy_from_slice(b"MGBHEL01");
    hello[8] = 1;
    hello[12] = 2;
    hello[16] = 1;
    hello[24..56].fill(7);
    hello[56..72].fill(9);
    for part in hello.chunks(3) {
        peer.write_all(part).unwrap();
    }
    let frame = ControlFrame {
        action: Action::Ready,
        rank: 1,
        depth: 0,
        epoch: 0,
        slot: 7,
        plane: Plane::Candidate,
        fatal_code: 0,
    };
    peer.write_all(&frame.encode(2).unwrap()).unwrap();
    let mut ack = [0; 80];
    peer.read_exact(&mut ack).unwrap();
    hello[16] = 0;
    assert_eq!(ack, hello);
    server.join().unwrap();
}

#[test]
fn reserved_bytes_are_rejected_before_acknowledgement() {
    use std::io::{Read, Write};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(
            ControlConnection::accept_peer(stream, 2, identity(), Duration::from_secs(3))
                .err()
                .unwrap(),
            "CONTROL_HANDSHAKE_IDENTITY"
        );
    });
    let mut peer = TcpStream::connect(address).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    let mut hello = [0u8; 80];
    hello[..8].copy_from_slice(b"MGBHEL01");
    hello[8] = 1;
    hello[12] = 2;
    hello[16] = 1;
    hello[24..56].fill(7);
    hello[56..72].fill(9);
    hello[79] = 1;
    peer.write_all(&hello).unwrap();
    let mut ack = [0; 80];
    assert_eq!(peer.read(&mut ack).unwrap(), 0);
    server.join().unwrap();
}

#[test]
fn matching_handshake_returns_rank_bound_nonblocking_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let (rank, mut connection) =
            ControlConnection::accept_peer(stream, 2, identity(), Duration::from_secs(3)).unwrap();
        assert_eq!(rank, 1);
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(frame) = connection.poll_receive().unwrap() {
                assert_eq!(frame.slot, 3);
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    });
    let stream = TcpStream::connect(address).unwrap();
    let mut connection =
        ControlConnection::connect_peer(stream, 2, 1, identity(), Duration::from_secs(3)).unwrap();
    connection
        .enqueue(ControlFrame {
            action: Action::Ready,
            rank: 1,
            depth: 0,
            epoch: 0,
            slot: 3,
            plane: Plane::Candidate,
            fatal_code: 0,
        })
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while !connection.poll_send().unwrap() {
        assert!(Instant::now() < deadline);
    }
    server.join().unwrap();
}

#[test]
fn mismatched_run_identity_or_world_rejects_both_sides() {
    for variant in 0..3 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            assert!(
                ControlConnection::accept_peer(stream, 2, identity(), Duration::from_secs(3))
                    .is_err()
            );
        });
        let mut wrong = identity();
        if variant == 0 {
            wrong.config_digest[0] ^= 1;
        }
        if variant == 1 {
            wrong.run_id[0] ^= 1;
        }
        let world = if variant == 2 { 3 } else { 2 };
        let stream = TcpStream::connect(address).unwrap();
        assert!(
            ControlConnection::connect_peer(stream, world, 1, wrong, Duration::from_secs(3))
                .is_err()
        );
        server.join().unwrap();
    }
}
