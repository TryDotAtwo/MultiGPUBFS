#![cfg(feature = "cuda")]
use mgbfs_cuda::ffi::*;
use mgbfs_runtime::{
    byte_admission::{ByteAdmission, RankByteAdmission},
    control_connection::ControlConnection,
    control_wire::{Action, ControlFrame, Plane},
    event_generation::NativeEvent,
    scatter_admission::TicketKey,
};
use std::{
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};

// Test driver only: serialized epochs intentionally isolate admission + NCCL.
// This is not the production event-driven BFS dispatcher or an overlap test.
struct AdmissionDriver {
    rank: u32,
    connection: ControlConnection,
    root: Option<ByteAdmission>,
    local: RankByteAdmission,
    next: u64,
}
impl AdmissionDriver {
    fn send(&mut self, frame: ControlFrame) {
        self.connection.enqueue(frame).unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while !self.connection.poll_send().unwrap() {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    }
    fn receive(&mut self) -> ControlFrame {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(f) = self.connection.poll_receive().unwrap() {
                return f;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    }
    fn admit(&mut self, source: u32, epoch: u64, sizes: [u64; 2]) -> (TicketKey, u64) {
        let key = TicketKey {
            depth: 0,
            epoch,
            source,
            plane: Plane::Candidate,
            generation: epoch,
        };
        let blank = ControlFrame {
            action: Action::OfferBytes,
            rank: source,
            source_rank: source,
            destination_rank: 0,
            payload_bytes: 0,
            depth: 0,
            epoch,
            slot: epoch,
            plane: Plane::Candidate,
            fatal_code: 0,
        };
        let mut out = [blank; 2];
        if let Some(root) = &mut self.root {
            root.begin(key, 8).unwrap();
        }
        if self.rank == source {
            for dst in 0..2 {
                let offer = ControlFrame {
                    destination_rank: dst,
                    payload_bytes: sizes[dst as usize],
                    ..blank
                };
                if let Some(root) = &mut self.root {
                    assert_eq!(root.offer(offer, &mut out).unwrap(), dst == 1);
                } else {
                    self.send(offer);
                }
            }
        }
        let ticket = if self.rank == 0 {
            if source != 0 {
                for index in 0..2 {
                    let f = self.receive();
                    assert_eq!(
                        self.root.as_mut().unwrap().offer(f, &mut out).unwrap(),
                        index == 1
                    );
                }
            }
            self.send(out[1]);
            out[0]
        } else {
            self.receive()
        };
        let ack = self.local.accept(ticket, 4).unwrap();
        let launch = if self.rank == 0 {
            self.root.as_mut().unwrap().ack(ack).unwrap();
            let remote = self.receive();
            let root = self.root.as_mut().unwrap();
            root.ack(remote).unwrap();
            assert!(root.launch(&mut self.next, &mut out).unwrap());
            self.send(out[1]);
            out[0]
        } else {
            self.send(ack);
            self.receive()
        };
        (key, self.local.launch(launch).unwrap())
    }
    fn retire(&mut self, key: TicketKey) {
        self.local.retire(key).unwrap();
        if let Some(root) = &mut self.root {
            root.retire(key).unwrap();
        }
    }
}

#[test]
fn two_devices_scatter_exact_bytes_from_each_source_and_drain_empty_epochs() {
    let mut id = [0u8; 128];
    assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    let workers: Vec<_> = [server, client]
        .into_iter()
        .enumerate()
        .map(|(rank, connection)| {
            let rank = rank as u32;
            std::thread::spawn(move || unsafe {
                let mut admission = AdmissionDriver {
                    rank,
                    connection: ControlConnection::new(connection, 2, rank, rank ^ 1).unwrap(),
                    root: if rank == 0 {
                        Some(ByteAdmission::new(2).unwrap())
                    } else {
                        None
                    },
                    local: RankByteAdmission::new(2, rank).unwrap(),
                    next: 0,
                };
                let mut comm = std::ptr::null_mut();
                let mut error = [0i8; 512];
                assert_eq!(
                    mgbfs_nccl_create(
                        rank,
                        2,
                        rank,
                        id.as_ptr().cast(),
                        &mut comm,
                        error.as_mut_ptr(),
                        error.len()
                    ),
                    0
                );
                let mut stream = std::ptr::null_mut();
                assert_eq!(cudaStreamCreateWithFlags(&mut stream, 1), 0);
                let mut send = std::ptr::null_mut();
                let mut recv = std::ptr::null_mut();
                assert_eq!(cudaMalloc(&mut send, 8), 0);
                assert_eq!(cudaMalloc(&mut recv, 4), 0);
                let mut completion = NativeEvent::new().unwrap();
                for source in 0..2u32 {
                    let payload = [11u8, 12, 13, 14, 21, 22, 23, 24];
                    assert_eq!(cudaMemcpy(send, payload.as_ptr().cast(), 8, 1), 0);
                    let sizes = [4u64, 4];
                    let (key, received_bytes) =
                        admission.admit(source, u64::from(source) * 2, sizes);
                    assert_eq!(received_bytes, 4);
                    // A rejected local capacity check must not enqueue an unmatched
                    // receive. The following valid exchange must still match.
                    if rank != source {
                        assert_ne!(
                            mgbfs_nccl_scatter(
                                comm,
                                source,
                                send,
                                8,
                                sizes.as_ptr(),
                                recv,
                                5,
                                4,
                                stream
                            ),
                            0
                        );
                    }
                    assert_eq!(
                        mgbfs_nccl_scatter(
                            comm,
                            source,
                            send,
                            8,
                            sizes.as_ptr(),
                            recv,
                            received_bytes,
                            4,
                            stream
                        ),
                        0
                    );
                    completion.record(key.generation, stream).unwrap();
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while !completion.poll(key.generation).unwrap() {
                        assert_eq!(mgbfs_nccl_poll(comm), 0);
                        assert!(Instant::now() < deadline);
                        std::thread::yield_now();
                    }
                    assert_eq!(mgbfs_nccl_poll(comm), 0);
                    let mut actual = [0u8; 4];
                    let selected = if rank == source {
                        send.cast::<u8>().add(rank as usize * 4).cast()
                    } else {
                        recv
                    };
                    assert_eq!(cudaMemcpy(actual.as_mut_ptr().cast(), selected, 4, 2), 0);
                    assert_eq!(
                        actual,
                        if rank == 0 {
                            [11, 12, 13, 14]
                        } else {
                            [21, 22, 23, 24]
                        }
                    );
                    completion.retire(key.generation).unwrap();
                    admission.retire(key);
                    let zero = [0u64; 2];
                    let (key, received_bytes) =
                        admission.admit(source, u64::from(source) * 2 + 1, zero);
                    assert_eq!(received_bytes, 0);
                    assert_eq!(
                        mgbfs_nccl_scatter(
                            comm,
                            source,
                            send,
                            8,
                            zero.as_ptr(),
                            recv,
                            received_bytes,
                            4,
                            stream
                        ),
                        0
                    );
                    completion.record(key.generation, stream).unwrap();
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while !completion.poll(key.generation).unwrap() {
                        assert_eq!(mgbfs_nccl_poll(comm), 0);
                        assert!(Instant::now() < deadline);
                        std::thread::yield_now();
                    }
                    completion.retire(key.generation).unwrap();
                    admission.retire(key);
                }
                assert_eq!(mgbfs_nccl_abort(comm), 0);
                assert_eq!(mgbfs_nccl_abort(comm), 0);
                mgbfs_nccl_destroy(comm);
                assert_eq!(cudaFree(send), 0);
                assert_eq!(cudaFree(recv), 0);
                assert_eq!(cudaStreamDestroy(stream), 0);
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
