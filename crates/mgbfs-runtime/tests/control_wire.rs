use mgbfs_runtime::control_wire::{Action, ControlFrame, Plane, FRAME_BYTES, NO_SLOT};
use mgbfs_runtime::control_wire::{FrameReader, FrameWriter};
use std::io::{self, Cursor, Read, Write};

fn ready() -> ControlFrame {
    ControlFrame {
        action: Action::Ready,
        rank: 1,
        depth: 7,
        epoch: 0,
        slot: 3,
        plane: Plane::Candidate,
        source_rank: 0,
        fatal_code: 0,
        destination_rank: 0,
        payload_bytes: 0,
    }
}
#[test]
fn admission_frames_preserve_large_byte_counts_and_destination() {
    for (action, rank) in [
        (Action::OfferBytes, 2),
        (Action::TicketBytes, 0),
        (Action::Admitted, 1),
    ] {
        let frame = ControlFrame {
            action,
            rank,
            source_rank: 2,
            destination_rank: 1,
            payload_bytes: 0x0102030405060708,
            epoch: 19,
            slot: 8,
            ..ready()
        };
        let bytes = frame.encode(3).unwrap();
        assert_eq!(&bytes[52..56], &[1, 0, 0, 0]);
        assert_eq!(&bytes[56..64], &[8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(ControlFrame::decode(&bytes, 3).unwrap(), frame);
        assert!(ControlFrame {
            destination_rank: 3,
            ..frame
        }
        .encode(3)
        .is_err());
    }
    let launch = ControlFrame {
        action: Action::Launch,
        rank: 0,
        source_rank: 2,
        epoch: 19,
        slot: 8,
        ..ready()
    };
    assert_eq!(
        ControlFrame::decode(&launch.encode(3).unwrap(), 3).unwrap(),
        launch
    );
    assert!(ControlFrame {
        payload_bytes: 1,
        ..launch
    }
    .encode(3)
    .is_err());
    assert!(ControlFrame {
        payload_bytes: 1,
        ..ready()
    }
    .encode(3)
    .is_err());
}

#[test]
fn version_three_begin_preserves_explicit_source_rank() {
    let frame = ControlFrame {
        action: Action::Begin,
        rank: 0,
        slot: NO_SLOT,
        ..ready()
    };
    let mut bytes = frame.encode(3).unwrap();
    bytes[8..10].copy_from_slice(&3u16.to_le_bytes());
    bytes[48..52].copy_from_slice(&2u32.to_le_bytes());
    let decoded = ControlFrame::decode(&bytes, 3).expect("V3 source-bearing command");
    assert_eq!(decoded.encode(3).unwrap(), bytes);
    bytes[48..52].copy_from_slice(&3u32.to_le_bytes());
    assert!(ControlFrame::decode(&bytes, 3).is_err());
    bytes[48..52].fill(0);
    bytes[8..10].copy_from_slice(&2u16.to_le_bytes());
    assert!(ControlFrame::decode(&bytes, 3).is_err());
}

struct PausingRead {
    bytes: Cursor<Vec<u8>>,
    pause: bool,
}

struct PausingWrite {
    bytes: Vec<u8>,
    pause: bool,
}
impl Write for PausingWrite {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.pause = !self.pause;
        if self.pause {
            return Err(io::ErrorKind::WouldBlock.into());
        }
        let n = data.len().min(7);
        self.bytes.extend_from_slice(&data[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn pending_writer_cannot_be_overwritten_and_resumes_exact_byte_offset() {
    let mut writer = FrameWriter::new(2).unwrap();
    let mut stream = PausingWrite {
        bytes: Vec::new(),
        pause: false,
    };
    writer.enqueue(ready()).unwrap();
    assert!(!writer.poll(&mut stream).unwrap());
    assert!(writer.enqueue(ControlFrame { slot: 9, ..ready() }).is_err());
    let mut done = false;
    for _ in 0..20 {
        if writer.poll(&mut stream).unwrap() {
            done = true;
            break;
        }
    }
    assert!(done);
    assert_eq!(stream.bytes, ready().encode(2).unwrap());
    assert!(writer.poll(&mut stream).unwrap());
    assert_eq!(stream.bytes.len(), 64);
    writer.enqueue(ControlFrame { slot: 9, ..ready() }).unwrap();
    for _ in 0..20 {
        if writer.poll(&mut stream).unwrap() {
            break;
        }
    }
    assert_eq!(stream.bytes.len(), 128);
    assert_eq!(
        ControlFrame::decode(&stream.bytes[64..], 2).unwrap().slot,
        9
    );
}

#[test]
fn zero_write_permanently_poisons_pending_connection() {
    let mut writer = FrameWriter::new(2).unwrap();
    writer.enqueue(ready()).unwrap();
    let mut full = Cursor::new([0u8; 0]);
    assert!(writer.poll(&mut full).is_err());
    let mut later = Vec::new();
    assert!(writer.poll(&mut later).is_err());
    assert!(writer.enqueue(ready()).is_err());
    assert!(later.is_empty());
    assert!(FrameWriter::new(0).is_err());
}
impl Read for PausingRead {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.pause = !self.pause;
        if self.pause {
            return Err(io::ErrorKind::WouldBlock.into());
        }
        let n = out.len().min(7);
        self.bytes.read(&mut out[..n])
    }
}

#[test]
fn nonblocking_reader_retains_partial_frames_without_stalling_other_peers() {
    let mut slow = PausingRead {
        bytes: Cursor::new(ready().encode(2).unwrap().to_vec()),
        pause: false,
    };
    let mut reader = FrameReader::new(2).unwrap();
    assert_eq!(reader.poll(&mut slow).unwrap(), None);
    let mut fast_reader = FrameReader::new(2).unwrap();
    let mut fast = Cursor::new(ready().encode(2).unwrap());
    assert_eq!(fast_reader.poll(&mut fast).unwrap(), Some(ready()));
    let mut received = None;
    for _ in 0..20 {
        if let Some(frame) = reader.poll(&mut slow).unwrap() {
            received = Some(frame);
            break;
        }
    }
    assert_eq!(received, Some(ready()));
}

#[test]
fn framing_error_poisoning_prevents_resynchronizing_onto_later_valid_data() {
    let mut invalid = ready().encode(2).unwrap();
    invalid[0] = 0;
    let mut bytes = invalid.to_vec();
    bytes.extend_from_slice(&ready().encode(2).unwrap());
    let mut stream = Cursor::new(bytes);
    let mut reader = FrameReader::new(2).unwrap();
    assert!(reader.poll(&mut stream).is_err());
    let position = stream.position();
    assert!(reader.poll(&mut stream).is_err());
    assert_eq!(stream.position(), position);
    assert!(FrameReader::new(0).is_err());
}

#[test]
fn nonblocking_tcp_fragment_arrival_and_frame_reuse() {
    use std::net::{TcpListener, TcpStream};
    use std::time::{Duration, Instant};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    server.set_nonblocking(true).unwrap();
    client
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut reader = FrameReader::new(2).unwrap();
    assert_eq!(reader.poll(&mut server).unwrap(), None);
    let bytes = ready().encode(2).unwrap();
    client.write_all(&bytes[..13]).unwrap();
    assert_eq!(reader.poll(&mut server).unwrap(), None);
    client.write_all(&bytes[13..]).unwrap();
    client.set_nonblocking(true).unwrap();
    let mut writer = FrameWriter::new(2).unwrap();
    writer.enqueue(ready()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !writer.poll(&mut client).unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let mut received = 0;
    while received < 2 {
        if let Some(frame) = reader.poll(&mut server).unwrap() {
            assert_eq!(frame, ready());
            received += 1;
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(reader.poll(&mut server).unwrap(), None);
}

#[test]
fn partial_eof_is_terminal_for_incremental_reader() {
    let mut reader = FrameReader::new(2).unwrap();
    let mut partial = Cursor::new(ready().encode(2).unwrap()[..13].to_vec());
    assert!(reader.poll(&mut partial).is_err());
    let mut next = Cursor::new(ready().encode(2).unwrap());
    assert!(reader.poll(&mut next).is_err());
    assert_eq!(next.position(), 0);
}

#[test]
fn frozen_ready_layout_and_roundtrip() {
    let frame = ready();
    let expected: [u8; 64] = [
        77, 71, 66, 67, 84, 82, 76, 49, 3, 0, 1, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(FRAME_BYTES, expected.len());
    assert_eq!(frame.encode(2).unwrap(), expected);
    assert_eq!(ControlFrame::decode(&expected, 2).unwrap(), frame);
}

#[test]
fn begin_pins_the_receiver_offer_instead_of_resampling_ready_slots() {
    for slot in [3, NO_SLOT] {
        let begin = ControlFrame {
            action: Action::Begin,
            rank: 0,
            epoch: 42,
            slot,
            ..ready()
        };
        let bytes = begin.encode(2).unwrap();
        assert_eq!(ControlFrame::decode(&bytes, 2).unwrap(), begin);
    }
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
            source_rank: 0,
            fatal_code: 1,
            destination_rank: 0,
            payload_bytes: 0,
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
            source_rank: 0,
            fatal_code: if action == Action::Fatal { 17 } else { 0 },
            destination_rank: 0,
            payload_bytes: 0,
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
