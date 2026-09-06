#![cfg(feature = "cuda")]
use mgbfs_cuda::ffi::*;
use mgbfs_runtime::{
    control_connection::ControlConnection,
    control_pump::ControlPump,
    control_wire::{Action, ControlFrame, Plane},
    event_generation::NativeEvent,
    payload_lease::PayloadLease,
    scatter_admission::TicketKey,
};
use std::{
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};

// Test driver only: two live data epochs followed by an empty epoch per source.
// This proves leased bank correctness, not production BFS/kernel overlap.
struct AdmissionDriver {
    rank: u32,
    pump: ControlPump,
}
impl AdmissionDriver {
    fn receive(&mut self) -> ControlFrame {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            self.pump.poll_before(deadline).unwrap();
            if let Some(f) = self.pump.command().unwrap() {
                return f;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    }
    fn admit(
        &mut self,
        source: u32,
        epoch: u64,
        sizes: [u64; 2],
        bank: &mut PayloadLease,
    ) -> (TicketKey, u64) {
        let key = TicketKey {
            depth: 0,
            epoch,
            source,
            plane: Plane::Candidate,
            generation: if source == 0 { 100 + epoch } else { epoch - 2 },
        };
        if self.rank == source {
            self.pump.offer(Plane::Candidate, key.generation).unwrap();
        }
        let begin = self.receive();
        assert_eq!(
            (begin.action, begin.epoch, begin.source_rank),
            (Action::Begin, epoch, source)
        );
        if self.rank == source {
            self.pump.describe_bytes(begin, &sizes).unwrap();
        }
        let ticket = self.receive();
        assert_eq!(
            (
                ticket.action,
                ticket.epoch,
                ticket.slot,
                ticket.destination_rank
            ),
            (Action::TicketBytes, epoch, key.generation, self.rank)
        );
        bank.reserve(key, ticket.payload_bytes).unwrap();
        self.pump.admit_bytes(ticket, 4).unwrap();
        let launch = self.receive();
        assert_eq!(
            (launch.action, launch.epoch, launch.slot),
            (Action::Launch, epoch, key.generation)
        );
        (key, ticket.payload_bytes)
    }
    fn retire(&mut self, key: TicketKey) {
        self.pump.transfer_complete(key.epoch).unwrap();
        self.pump.consumed(key.epoch).unwrap();
    }
    fn finalize(&mut self) {
        self.pump.close_source().unwrap();
        assert_eq!(self.receive().action, Action::Finalize);
        self.pump.finalized(true).unwrap();
        assert_eq!(self.receive().action, Action::Publish);
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
                let mut peers: Vec<_> = (0..2).map(|_| None).collect();
                peers[(rank ^ 1) as usize] =
                    Some(ControlConnection::new(connection, 2, rank, rank ^ 1).unwrap());
                let mut admission = AdmissionDriver {
                    rank,
                    pump: ControlPump::new_admitted(2, rank, 2, peers, [8; 4]).unwrap(),
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
                assert_eq!(cudaMalloc(&mut send, 16), 0);
                assert_eq!(cudaMalloc(&mut recv, 8), 0);
                let mut events = [NativeEvent::new().unwrap(), NativeEvent::new().unwrap()];
                let mut leases = [
                    PayloadLease::new(2, 4, 2).unwrap(),
                    PayloadLease::new(2, 4, 2).unwrap(),
                ];
                for source in 0..2u32 {
                    let payload = [
                        11u8, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44,
                    ];
                    assert_eq!(cudaMemcpy(send, payload.as_ptr().cast(), 16, 1), 0);
                    let sizes = [4u64, 4];
                    let mut pending = [None; 2];
                    let mut consumers = [None; 2];
                    for lane in 0..2usize {
                        let send_bank = send.cast::<u8>().add(lane * 8).cast();
                        let recv_bank = recv.cast::<u8>().add(lane * 4).cast();
                        let (key, received_bytes) = admission.admit(
                            source,
                            u64::from(source) * 3 + lane as u64,
                            sizes,
                            &mut leases[lane],
                        );
                        consumers[lane] = Some([
                            leases[lane].consumer(key).unwrap(),
                            leases[lane].consumer(key).unwrap(),
                        ]);
                        leases[lane].seal(key).unwrap();
                        assert_eq!(received_bytes, 4);
                        // A rejected local capacity check must not enqueue an unmatched
                        // receive. The following valid exchange must still match.
                        if rank != source {
                            assert_ne!(
                                mgbfs_nccl_scatter(
                                    comm,
                                    source,
                                    send_bank,
                                    8,
                                    sizes.as_ptr(),
                                    recv_bank,
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
                                send_bank,
                                8,
                                sizes.as_ptr(),
                                recv_bank,
                                received_bytes,
                                4,
                                stream
                            ),
                            0
                        );
                        events[lane].record(key.epoch, stream).unwrap();
                        pending[lane] = Some(key);
                    }
                    // Both payload calls and both event records have been
                    // submitted before either bank can be consumed or reused.
                    for lane in 0..2usize {
                        let key = pending[lane].unwrap();
                        let completion = &mut events[lane];
                        let deadline = Instant::now() + Duration::from_secs(30);
                        while !completion.poll(key.epoch).unwrap() {
                            assert_eq!(mgbfs_nccl_poll(comm), 0);
                            assert!(Instant::now() < deadline);
                            std::thread::yield_now();
                        }
                        assert_eq!(mgbfs_nccl_poll(comm), 0);
                        let mut actual = [0u8; 4];
                        let selected: *mut std::ffi::c_void = if rank == source {
                            send.cast::<u8>().add(lane * 8 + rank as usize * 4).cast()
                        } else {
                            recv.cast::<u8>().add(lane * 4).cast()
                        };
                        // Two actual downstream readers share one payload bank.
                        // The first completion must not release the second reader.
                        for (part, consumer) in consumers[lane].unwrap().into_iter().enumerate() {
                            assert_eq!(
                                cudaMemcpy(
                                    actual.as_mut_ptr().add(part * 2).cast(),
                                    selected.cast::<u8>().add(part * 2).cast(),
                                    2,
                                    2
                                ),
                                0
                            );
                            leases[lane].complete(consumer).unwrap();
                            assert_eq!(leases[lane].drained(key).unwrap(), part == 1);
                        }
                        assert_eq!(
                            actual,
                            match (lane, rank) {
                                (0, 0) => [11, 12, 13, 14],
                                (0, 1) => [21, 22, 23, 24],
                                (1, 0) => [31, 32, 33, 34],
                                (1, 1) => [41, 42, 43, 44],
                                _ => unreachable!(),
                            }
                        );
                        completion.retire(key.epoch).unwrap();
                        leases[lane].retire(key).unwrap();
                        admission.retire(key);
                    }
                    let completion = &mut events[0];
                    let zero = [0u64; 2];
                    let (key, received_bytes) =
                        admission.admit(source, u64::from(source) * 3 + 2, zero, &mut leases[0]);
                    leases[0].seal(key).unwrap();
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
                    completion.record(key.epoch, stream).unwrap();
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while !completion.poll(key.epoch).unwrap() {
                        assert_eq!(mgbfs_nccl_poll(comm), 0);
                        assert!(Instant::now() < deadline);
                        std::thread::yield_now();
                    }
                    completion.retire(key.epoch).unwrap();
                    leases[0].retire(key).unwrap();
                    admission.retire(key);
                }
                admission.finalize();
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
