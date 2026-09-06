use mgbfs_runtime::control_wire::{Action, ControlFrame, Plane, FRAME_BYTES, NO_SLOT};
use std::io::{self, Cursor, Read, Write};

fn ready() -> ControlFrame {
    ControlFrame {
        action: Action::Ready,
        rank: 1,
        depth: 7,
        epoch: 0,
        slot: 3,
        plane: Plane::Candidate,
        fatal_code: 0,
    }
}

#[test]
fn frozen_ready_layout_and_roundtrip() {
    let frame = ready();
    let expected: [u8; 64] = [
        77, 71, 66, 67, 84, 82, 76, 49, 1, 0, 1, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(FRAME_BYTES, expected.len());
    assert_eq!(frame.encode(2).unwrap(), expected);
    assert_eq!(ControlFrame::decode(&expected, 2).unwrap(), frame);
}

#[test]
fn rejects_corruption_and_cross_world_rank() {
    let bytes = ready().encode(2).unwrap();
    for offset in [0, 8, 10, 40, 48, 63] {
        let mut bad = bytes;
        bad[offset] = 255;
        assert!(ControlFrame::decode(&bad, 2).is_err(), "offset {offset}");
    }
    assert!(ControlFrame::decode(&bytes, 1).is_err());
    assert!(ControlFrame::decode(&bytes, 0).is_err());
    assert!(ControlFrame::decode(&bytes[..63], 2).is_err());
}

#[test]
fn rejects_impossible_action_fields() {
    for frame in [
        ControlFrame {
            slot: NO_SLOT,
            ..ready()
        },
        ControlFrame {
            epoch: 1,
            ..ready()
        },
        ControlFrame {
            fatal_code: 1,
            ..ready()
        },
        ControlFrame {
            plane: Plane::None,
            ..ready()
        },
        ControlFrame {
            action: Action::Begin,
            slot: NO_SLOT,
            ..ready()
        },
        ControlFrame {
            action: Action::SourceClosed,
            slot: NO_SLOT,
            ..ready()
        },
    ] {
        assert!(frame.encode(2).is_err(), "{frame:?}");
    }
}

struct Fragmented(Cursor<Vec<u8>>);
impl Read for Fragmented {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let n = out.len().min(3);
        self.0.read(&mut out[..n])
    }
}
impl Write for Fragmented {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.0.write(&data[..data.len().min(5)])
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn fragmented_stream_preserves_frame_boundaries_and_rejects_eof() {
    let first = ready();
    let second = ControlFrame {
        action: Action::Complete,
        epoch: 42,
        slot: NO_SLOT,
        ..first
    };
    let mut stream = Fragmented(Cursor::new(Vec::new()));
    first.write_to(&mut stream, 2).unwrap();
    second.write_to(&mut stream, 2).unwrap();
    assert_eq!(stream.0.get_ref().len(), 128);
    stream.0.set_position(0);
    assert_eq!(ControlFrame::read_from(&mut stream, 2).unwrap(), first);
    assert_eq!(ControlFrame::read_from(&mut stream, 2).unwrap(), second);
    assert!(ControlFrame::read_from(&mut stream, 2).is_err());
    for length in [1, 31, 63] {
        assert!(ControlFrame::read_from(&mut Cursor::new(vec![0; length]), 2).is_err());
    }
}

#[test]
fn actual_tcp_ready_begin_complete_exchange() {
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let begin = ControlFrame {
        action: Action::Begin,
        rank: 0,
        epoch: 42,
        slot: NO_SLOT,
        ..ready()
    };
    let complete = ControlFrame {
        action: Action::Complete,
        rank: 1,
        ..begin
    };
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        assert_eq!(ControlFrame::read_from(&mut stream, 2).unwrap(), ready());
        // Network fragmentation must not be interpreted as message framing.
        for chunk in begin.encode(2).unwrap().chunks(3) {
            stream.write_all(chunk).unwrap();
        }
        assert_eq!(ControlFrame::read_from(&mut stream, 2).unwrap(), complete);
    });
    let mut client = TcpStream::connect_timeout(&address, Duration::from_secs(5)).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    ready().write_to(&mut client, 2).unwrap();
    assert_eq!(ControlFrame::read_from(&mut client, 2).unwrap(), begin);
    complete.write_to(&mut client, 2).unwrap();
    server.join().unwrap();
}

#[test]
fn control_only_messages_have_no_payload_plane_or_slot() {
    for action in [Action::SourceClosed, Action::Fatal, Action::Finalize] {
        let frame = ControlFrame {
            action,
            rank: 0,
            depth: 7,
            epoch: 0,
            slot: NO_SLOT,
            plane: Plane::None,
            fatal_code: if action == Action::Fatal { 17 } else { 0 },
        };
        assert_eq!(
            ControlFrame::decode(&frame.encode(2).unwrap(), 2).unwrap(),
            frame
        );
        assert!(ControlFrame { slot: 0, ..frame }.encode(2).is_err());
        assert!(ControlFrame {
            plane: Plane::Candidate,
            ..frame
        }
        .encode(2)
        .is_err());
    }
    let mut output = Vec::new();
    assert!(ControlFrame {
        slot: NO_SLOT,
        ..ready()
    }
    .write_to(&mut output, 2)
    .is_err());
    assert!(output.is_empty());
}
