#![cfg(feature = "cuda")]
use mgbfs_cuda::ffi::*;
use mgbfs_cuda::native_owner::{cudaMemcpyAsync, cudaSetDevice};
use mgbfs_runtime::{
    admitted_buffers::{AdmittedBuffers, BufferEvent},
    control_connection::ControlConnection,
    control_pump::ControlPump,
    control_wire::{Action, ControlFrame, Plane},
    event_generation::NativeEvent,
    payload_lease::{PayloadBank, PayloadBanks},
    scatter_admission::TicketKey,
};

// Exercises the production buffer adapter, including the source's zero-copy
// self view and empty epochs. This is a transport gate, not a BFS benchmark.
#[test]
fn admitted_adapter_native_scatter_and_depth_rollover() {
    let mut id = [0u8; 128];
    assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    let workers: Vec<_> = [server, client]
        .into_iter()
        .enumerate()
        .map(|(rank, socket)| {
            std::thread::spawn(move || unsafe {
                let rank = rank as u32;
                let mut peers = vec![None, None];
                peers[(rank ^ 1) as usize] =
                    Some(ControlConnection::new(socket, 2, rank, rank ^ 1).unwrap());
                let mut buffers = AdmittedBuffers::new(2, rank, 2, peers, [2048; 4], 1).unwrap();
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
                let (send_bytes, recv_bytes) = buffers.allocation_bytes(Plane::Candidate).unwrap();
                assert_eq!(cudaMalloc(&mut send, send_bytes as usize), 0);
                assert_eq!(cudaMalloc(&mut recv, recv_bytes as usize), 0);
                let mut done = NativeEvent::new().unwrap();
                let mut generate_stream = std::ptr::null_mut();
                assert_eq!(cudaStreamCreateWithFlags(&mut generate_stream, 1), 0);
                let mut generated = NativeEvent::new().unwrap();
                let mut generation = 0u64;
                let mut inputs = [std::ptr::null_mut(); 4];
                for (ptr, bytes) in inputs.iter_mut().zip([32, 32, 16, 4]) {
                    assert_eq!(cudaMalloc(ptr, bytes), 0);
                }
                let [states, hashes, refs, fatal] = inputs;
                let state_words = [11u32, 12, 13, 14, 21, 22, 23, 24].map(|n| n + rank * 100);
                let hash_words = [31u32, 32, 33, 34, 41, 42, 43, 44];
                let ref_words = [0u64, 1];
                assert_eq!(cudaMemcpy(states, state_words.as_ptr().cast(), 32, 1), 0);
                assert_eq!(cudaMemcpy(hashes, hash_words.as_ptr().cast(), 32, 1), 0);
                assert_eq!(cudaMemcpy(refs, ref_words.as_ptr().cast(), 16, 1), 0);
                assert_eq!(cudaMemsetAsync(fatal, 0, 4, generate_stream), 0);
                let mut frames =
                    mgbfs_runtime::dense_frames::DenseFrames::new(&[0, 1], 16, 2, 2048).unwrap();
                // Isolate materialization from transport. This fixture supplies
                // an already committed one-row owner result, not an owner BFS.
                let mut owner_stream = std::ptr::null_mut();
                assert_eq!(cudaStreamCreateWithFlags(&mut owner_stream, 1), 0);
                let mut materialized = NativeEvent::new().unwrap();
                let mut owner_storage = [std::ptr::null_mut(); 5];
                for (ptr, bytes) in owner_storage.iter_mut().zip([64, 64, 64, 4, 16]) {
                    assert_eq!(cudaMalloc(ptr, bytes), 0);
                }
                let [ring_ptr, control_ptr, extent_ptr, selected_ptr, final_states] = owner_storage;
                use mgbfs_cuda::native_owner::{Control, Extent, Ring};
                let ring = Ring {
                    tail: 1,
                    descriptor_tail: 1,
                    capacity: 1,
                    descriptor_capacity: 1,
                    ..Ring::default()
                };
                let control = Control {
                    stage: 2,
                    survivors: 1,
                    ..Control::default()
                };
                let extent = Extent {
                    count: 1,
                    granted_rows: 1,
                    ..Extent::default()
                };
                assert_eq!(cudaMemsetAsync(selected_ptr, 0, 4, owner_stream), 0);
                for depth in 0..2 {
                    for source in 0..2 {
                        for empty in [false, true] {
                            if rank == source {
                                let h = buffers.reserve(Plane::Candidate, depth).unwrap().unwrap();
                                let offset = buffers.source_offset(h).unwrap();
                                frames
                                    .prepare(if empty { &[0, 0] } else { &[1, 1] })
                                    .unwrap();
                                frames
                                    .enqueue_native(
                                        states.cast(),
                                        2,
                                        hashes,
                                        refs.cast(),
                                        send.cast::<u8>().add(offset as usize),
                                        fatal.cast(),
                                        generate_stream,
                                    )
                                    .unwrap();
                                generation += 1;
                                generated.record(generation, generate_stream).unwrap();
                                let pack_deadline = Instant::now() + Duration::from_secs(30);
                                while !generated.poll(generation).unwrap() {
                                    assert!(Instant::now() < pack_deadline);
                                    std::thread::yield_now();
                                }
                                let mut code = 99u32;
                                assert_eq!(
                                    cudaMemcpy((&mut code as *mut u32).cast(), fatal, 4, 2),
                                    0
                                );
                                assert_eq!(code, 0);
                                generated.retire(generation).unwrap();
                                buffers.ready(h, frames.sizes().unwrap()).unwrap();
                            }
                            let deadline = Instant::now() + Duration::from_secs(30);
                            let launch = loop {
                                if let Some(BufferEvent::Launch(l)) = buffers.poll().unwrap() {
                                    break l;
                                }
                                assert!(Instant::now() < deadline);
                                std::thread::yield_now();
                            };
                            let view = buffers.payload_view(launch).unwrap();
                            assert_eq!(view.source_pool, rank == source);
                            assert_eq!(view.bytes, if empty { 0 } else { 1024 });
                            let mut sizes = [0u64; 2];
                            buffers
                                .submit_native_with_prefix(
                                    launch,
                                    comm,
                                    send.cast(),
                                    recv.cast(),
                                    stream,
                                    &mut done,
                                    &mut sizes,
                                    || {
                                        if rank == source {
                                            frames.enqueue_headers_native(
                                                launch.key,
                                                7,
                                                send.cast::<u8>()
                                                    .add(launch.source_offset.unwrap() as usize),
                                                stream,
                                            )
                                        } else {
                                            Ok(())
                                        }
                                    },
                                )
                                .unwrap();
                            while !done.poll(launch.key.epoch).unwrap() {
                                assert_eq!(mgbfs_nccl_poll(comm), 0);
                                assert!(Instant::now() < deadline);
                                std::thread::yield_now();
                            }
                            buffers.transfer_complete(launch).unwrap();
                            if !empty {
                                let base = if view.source_pool { send } else { recv };
                                let mut prefix = [0u8; 256];
                                assert_eq!(
                                    cudaMemcpy(
                                        prefix.as_mut_ptr().cast(),
                                        base.cast::<u8>().add(view.offset as usize).cast(),
                                        256,
                                        2
                                    ),
                                    0
                                );
                                let (reader, input) = buffers
                                    .dense_consumer(launch, &prefix, 7, 16, 2)
                                    .unwrap()
                                    .unwrap();
                                assert_eq!(input.rows, 1);
                                assert_eq!(input.source_pool, view.source_pool);
                                buffers.seal(launch).unwrap();
                                assert!(!buffers.drained(launch).unwrap());
                                assert_eq!(
                                    cudaMemcpy(ring_ptr, (&ring as *const Ring).cast(), 64, 1),
                                    0
                                );
                                assert_eq!(
                                    cudaMemcpy(
                                        control_ptr,
                                        (&control as *const Control).cast(),
                                        std::mem::size_of::<Control>(),
                                        1
                                    ),
                                    0
                                );
                                assert_eq!(
                                    cudaMemcpy(
                                        extent_ptr,
                                        (&extent as *const Extent).cast(),
                                        64,
                                        1
                                    ),
                                    0
                                );
                                assert_eq!(
                                    mgbfs_cuda::native_owner::mgbfs_state_materialize_packed(
                                        base.cast::<u8>().add(input.state_offset as usize),
                                        input.rows,
                                        selected_ptr.cast(),
                                        1,
                                        16,
                                        final_states.cast(),
                                        ring_ptr.cast(),
                                        control_ptr.cast(),
                                        extent_ptr.cast(),
                                        owner_stream,
                                    ),
                                    0
                                );
                                materialized.record(launch.key.epoch, owner_stream).unwrap();
                                while !materialized.poll(launch.key.epoch).unwrap() {
                                    assert!(!buffers.drained(launch).unwrap());
                                    assert!(Instant::now() < deadline);
                                    std::thread::yield_now();
                                }
                                let mut actual = [0u32; 4];
                                assert_eq!(
                                    cudaMemcpy(actual.as_mut_ptr().cast(), final_states, 16, 2),
                                    0
                                );
                                assert_eq!(
                                    actual,
                                    (if rank == 0 {
                                        [11, 12, 13, 14]
                                    } else {
                                        [21, 22, 23, 24]
                                    })
                                    .map(|n| n + source * 100)
                                );
                                let mut published = Extent::default();
                                assert_eq!(
                                    cudaMemcpy(
                                        (&mut published as *mut Extent).cast(),
                                        extent_ptr,
                                        64,
                                        2
                                    ),
                                    0
                                );
                                assert_eq!(published.ready, 1);
                                materialized.retire(launch.key.epoch).unwrap();
                                buffers.complete(reader).unwrap();
                            } else {
                                assert!(buffers
                                    .dense_consumer(launch, &[], 7, 16, 2)
                                    .unwrap()
                                    .is_none());
                                buffers.seal(launch).unwrap();
                            }
                            done.retire(launch.key.epoch).unwrap();
                            buffers.consume(launch).unwrap();
                        }
                    }
                    buffers.close_source().unwrap();
                    let deadline = Instant::now() + Duration::from_secs(30);
                    loop {
                        match buffers.poll().unwrap() {
                            Some(BufferEvent::Finalize(_)) => buffers.finalized(true).unwrap(),
                            Some(BufferEvent::Publish(f)) => {
                                assert_eq!(f.depth, depth + 1);
                                break;
                            }
                            Some(BufferEvent::Launch(_)) => panic!("unexpected payload"),
                            None => {}
                        }
                        assert!(Instant::now() < deadline);
                        std::thread::yield_now();
                    }
                }
                assert_eq!(mgbfs_nccl_abort(comm), 0);
                mgbfs_nccl_destroy(comm);
                assert_eq!(cudaFree(send), 0);
                assert_eq!(cudaFree(recv), 0);
                for ptr in inputs {
                    assert_eq!(cudaFree(ptr), 0);
                }
                for ptr in owner_storage {
                    assert_eq!(cudaFree(ptr), 0);
                }
                assert_eq!(cudaStreamDestroy(owner_stream), 0);
                assert_eq!(cudaStreamDestroy(generate_stream), 0);
                assert_eq!(cudaStreamDestroy(stream), 0);
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
use std::{
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};

#[test]
fn dense_frame_gpu_gather_matches_schema2_without_intermediate_states() {
    use mgbfs_core::wire::{payload_layout, validate_payload, FrameKind};
    unsafe {
        assert_eq!(cudaSetDevice(0), 0);
        let mut stream = std::ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 1), 0);
        let mut storage = [std::ptr::null_mut(); 5];
        for (ptr, bytes) in storage.iter_mut().zip([64, 32, 64, 2048, 4]) {
            assert_eq!(cudaMalloc(ptr, bytes), 0);
        }
        let [hashes, refs, states, output, fatal] = storage;
        let hash_words = [99u32, 99, 99, 99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let reference_words = [99u64, 3, 0, 2];
        let state_words = [
            10u32, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43,
        ];
        assert_eq!(cudaMemcpy(hashes, hash_words.as_ptr().cast(), 64, 1), 0);
        assert_eq!(cudaMemcpy(refs, reference_words.as_ptr().cast(), 32, 1), 0);
        assert_eq!(cudaMemcpy(states, state_words.as_ptr().cast(), 64, 1), 0);
        assert_eq!(cudaMemsetAsync(fatal, 0, 4, stream), 0);
        assert_eq!(cudaMemsetAsync(output, 0xff, 768, stream), 0);
        assert_eq!(
            mgbfs_exchange_pack_frame(
                16,
                states.cast(),
                4,
                hashes,
                refs.cast(),
                4,
                1,
                3,
                output.cast(),
                768,
                fatal.cast(),
                stream
            ),
            0
        );
        assert_eq!(cudaStreamSynchronize(stream), 0);
        let mut actual = [0u8; 768];
        assert_eq!(cudaMemcpy(actual.as_mut_ptr().cast(), output, 768, 2), 0);
        let mut expected = [0u8; 768];
        for (i, word) in (1u32..=12).enumerate() {
            expected[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        for (i, word) in [3u32, 0, 2].iter().enumerate() {
            expected[256 + i * 4..260 + i * 4].copy_from_slice(&word.to_le_bytes());
        }
        for (i, word) in [40u32, 41, 42, 43, 10, 11, 12, 13, 30, 31, 32, 33]
            .iter()
            .enumerate()
        {
            expected[512 + i * 4..516 + i * 4].copy_from_slice(&word.to_le_bytes());
        }
        assert_eq!(actual, expected);
        validate_payload(&actual, &payload_layout(FrameKind::Dense, 3, 16).unwrap()).unwrap();
        let mut code = 99u32;
        assert_eq!(cudaMemcpy((&mut code as *mut u32).cast(), fatal, 4, 2), 0);
        assert_eq!(code, 0);
        // Actual fixed descriptor plan -> native frame enqueue, with a swapped
        // owner map. The leading invalid input ref is outside this sorted view.
        let mut frames =
            mgbfs_runtime::dense_frames::DenseFrames::new(&[1, 0], 16, 4, 2048).unwrap();
        frames.prepare(&[1, 2]).unwrap();
        frames
            .enqueue_native(
                states.cast(),
                4,
                hashes.cast::<u8>().add(16).cast(),
                refs.cast::<u64>().add(1),
                output.cast(),
                fatal.cast(),
                stream,
            )
            .unwrap();
        let key = TicketKey {
            depth: 9,
            epoch: 99,
            source: 0,
            plane: Plane::Candidate,
            generation: 123,
        };
        frames
            .enqueue_headers_native(key, 7, output.cast(), stream)
            .unwrap();
        assert_eq!(cudaStreamSynchronize(stream), 0);
        let mut routed = [0u8; 2048];
        assert_eq!(cudaMemcpy(routed.as_mut_ptr().cast(), output, 2048, 2), 0);
        let mut want = [0u8; 2048];
        for rank in 0..2 {
            let header = mgbfs_core::wire::FrameHeader {
                kind: FrameKind::Dense,
                run_tag: 7,
                sequence: 99,
                batch: 123,
                depth: 9,
                source: 0,
                destination: rank,
                count: if rank == 0 { 2 } else { 1 },
            }
            .encode(16)
            .unwrap();
            want[rank as usize * 1024..rank as usize * 1024 + 64].copy_from_slice(&header);
        }
        for (offset, words) in [
            (256, &[5u32, 6, 7, 8, 9, 10, 11, 12][..]),
            (512, &[0u32, 2][..]),
            (768, &[10u32, 11, 12, 13, 30, 31, 32, 33][..]),
            (1280, &[1u32, 2, 3, 4][..]),
            (1536, &[3u32][..]),
            (1792, &[40u32, 41, 42, 43][..]),
        ] {
            for (i, word) in words.iter().enumerate() {
                want[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        assert_eq!(routed, want);
        assert_ne!(
            mgbfs_exchange_pack_frame(
                16,
                states.cast(),
                4,
                hashes,
                refs.cast(),
                4,
                1,
                3,
                output.cast(),
                767,
                fatal.cast(),
                stream
            ),
            0
        );
        assert_ne!(
            mgbfs_exchange_pack_frame(
                16,
                states.cast(),
                4,
                hashes,
                refs.cast(),
                4,
                3,
                3,
                output.cast(),
                768,
                fatal.cast(),
                stream
            ),
            0
        );
        // Valid enqueue with an invalid reference must report a device fatal,
        // not read the out-of-range source or silently accept its frame.
        assert_eq!(
            mgbfs_exchange_pack_frame(
                16,
                states.cast(),
                4,
                hashes,
                refs.cast(),
                4,
                0,
                3,
                output.cast(),
                768,
                fatal.cast(),
                stream
            ),
            0
        );
        assert_eq!(cudaStreamSynchronize(stream), 0);
        assert_eq!(cudaMemcpy((&mut code as *mut u32).cast(), fatal, 4, 2), 0);
        assert_eq!(code, 1);
        assert_eq!(
            mgbfs_exchange_pack_frame(
                16,
                states.cast(),
                4,
                hashes,
                refs.cast(),
                4,
                4,
                0,
                output.cast(),
                0,
                fatal.cast(),
                stream
            ),
            0
        );
        for ptr in storage {
            assert_eq!(cudaFree(ptr), 0);
        }
        assert_eq!(cudaStreamDestroy(stream), 0);
    }
}

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
        banks: &mut PayloadBanks,
    ) -> (TicketKey, u64, PayloadBank) {
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
        let bank = banks.reserve(key, ticket.payload_bytes).unwrap().unwrap();
        self.pump.admit_bytes(ticket, 4).unwrap();
        let launch = self.receive();
        assert_eq!(
            (launch.action, launch.epoch, launch.slot),
            (Action::Launch, epoch, key.generation)
        );
        (key, ticket.payload_bytes, bank)
    }
    fn retire(&mut self, key: TicketKey) {
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
                let mut readers = [std::ptr::null_mut(); 2];
                for reader in &mut readers {
                    assert_eq!(cudaStreamCreateWithFlags(reader, 1), 0);
                }
                let mut send = std::ptr::null_mut();
                let mut recv = std::ptr::null_mut();
                let mut copied = std::ptr::null_mut();
                let mut banks = PayloadBanks::new(2, 2, 4, 2, 256).unwrap();
                assert_eq!(cudaMalloc(&mut send, 16), 0);
                assert_eq!(cudaMalloc(&mut recv, banks.bytes() as usize), 0);
                assert_eq!(cudaMalloc(&mut copied, 8), 0);
                let mut events = [NativeEvent::new().unwrap(), NativeEvent::new().unwrap()];
                let mut reader_events = [
                    [NativeEvent::new().unwrap(), NativeEvent::new().unwrap()],
                    [NativeEvent::new().unwrap(), NativeEvent::new().unwrap()],
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
                        let (key, received_bytes, bank) = admission.admit(
                            source,
                            u64::from(source) * 3 + lane as u64,
                            sizes,
                            &mut banks,
                        );
                        let offset = banks.offset(bank).unwrap() as usize;
                        let physical = offset / 256;
                        let recv_bank = recv.cast::<u8>().add(offset).cast();
                        consumers[lane] =
                            Some([banks.consumer(bank).unwrap(), banks.consumer(bank).unwrap()]);
                        banks.seal(bank).unwrap();
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
                        events[physical].record(key.epoch, stream).unwrap();
                        let selected: *mut std::ffi::c_void = if rank == source {
                            send_bank.cast::<u8>().add(rank as usize * 4).cast()
                        } else {
                            recv_bank
                        };
                        // Queue actual readers on distinct streams BEFORE any
                        // host query of the transfer event. No host sync supplies
                        // this dependency: each stream waits on its generation.
                        for part in 0..2 {
                            events[physical].wait(key.epoch, readers[part]).unwrap();
                            assert_eq!(
                                cudaMemcpyAsync(
                                    copied.cast::<u8>().add(lane * 4 + part * 2).cast(),
                                    selected.cast::<u8>().add(part * 2).cast(),
                                    2,
                                    3,
                                    readers[part]
                                ),
                                0
                            );
                            reader_events[physical][part]
                                .record(key.epoch, readers[part])
                                .unwrap();
                        }
                        pending[lane] = Some((key, bank, physical));
                    }
                    // Both payload calls and both event records have been
                    // submitted before either bank can be consumed or reused.
                    for lane in 0..2usize {
                        let (key, _, physical) = pending[lane].unwrap();
                        let completion = &mut events[physical];
                        let deadline = Instant::now() + Duration::from_secs(30);
                        while !completion.poll(key.epoch).unwrap() {
                            assert_eq!(mgbfs_nccl_poll(comm), 0);
                            assert!(Instant::now() < deadline);
                            std::thread::yield_now();
                        }
                        assert_eq!(mgbfs_nccl_poll(comm), 0);
                        admission.pump.transfer_complete(key.epoch).unwrap();
                    }
                    // Transport COMPLETE stays ordered, but consumer retirement
                    // and physical bank release are deliberately reversed.
                    for lane in [1usize, 0] {
                        let (key, bank, physical) = pending[lane].unwrap();
                        let completion = &mut events[physical];
                        let mut actual = [0u8; 4];
                        let selected = copied.cast::<u8>().add(lane * 4);
                        // Two actual downstream readers share one payload bank.
                        // The first completion must not release the second reader.
                        for (part, consumer) in consumers[lane].unwrap().into_iter().enumerate() {
                            let done = &mut reader_events[physical][part];
                            let deadline = Instant::now() + Duration::from_secs(30);
                            while !done.poll(key.epoch).unwrap() {
                                assert_eq!(mgbfs_nccl_poll(comm), 0);
                                assert!(Instant::now() < deadline);
                                std::thread::yield_now();
                            }
                            assert_eq!(
                                cudaMemcpy(
                                    actual.as_mut_ptr().add(part * 2).cast(),
                                    selected.cast::<u8>().add(part * 2).cast(),
                                    2,
                                    2
                                ),
                                0
                            );
                            done.retire(key.epoch).unwrap();
                            banks.complete(consumer).unwrap();
                            assert_eq!(banks.drained(bank).unwrap(), part == 1);
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
                        banks.retire(bank).unwrap();
                        admission.retire(key);
                    }
                    let zero = [0u64; 2];
                    let (key, received_bytes, bank) =
                        admission.admit(source, u64::from(source) * 3 + 2, zero, &mut banks);
                    let offset = banks.offset(bank).unwrap() as usize;
                    let completion = &mut events[offset / 256];
                    banks.seal(bank).unwrap();
                    assert_eq!(received_bytes, 0);
                    assert_eq!(
                        mgbfs_nccl_scatter(
                            comm,
                            source,
                            send,
                            8,
                            zero.as_ptr(),
                            recv.cast::<u8>().add(offset).cast(),
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
                    banks.retire(bank).unwrap();
                    admission.pump.transfer_complete(key.epoch).unwrap();
                    admission.retire(key);
                }
                admission.finalize();
                assert_eq!(mgbfs_nccl_abort(comm), 0);
                assert_eq!(mgbfs_nccl_abort(comm), 0);
                mgbfs_nccl_destroy(comm);
                assert_eq!(cudaFree(send), 0);
                assert_eq!(cudaFree(recv), 0);
                assert_eq!(cudaFree(copied), 0);
                for reader in readers {
                    assert_eq!(cudaStreamDestroy(reader), 0);
                }
                assert_eq!(cudaStreamDestroy(stream), 0);
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
