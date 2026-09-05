#![cfg(feature = "cuda")]
//! Requires two real CUDA devices; run this test alone, with --test-threads=1.
use mgbfs_core::{
    hash::GemmHash,
    matrix::{encode_permutation_matrix, MatrixGroup},
};
use mgbfs_runtime::{
    archive::{verify, Extent},
    distributed_native::{DistributedConfig, DistributedNativeBfs},
    pinned_archive::PinnedArchive,
};
use std::sync::{Arc, Mutex};

struct Disk(Arc<Mutex<Vec<u8>>>);

#[test]
fn native_request_response_epochs_include_empty_ranks_and_group_fatal() {
    use mgbfs_core::wire::OriginRef;
    use mgbfs_cuda::ffi::*;
    use mgbfs_runtime::hash_first_exchange::{enqueue_round_trip, ExchangeBuffers, MatrixSource};
    use std::ffi::c_void;
    let mut id = [0u8; 128];
    assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
    let workers: Vec<_> = (0..2u32)
        .map(|rank| {
            std::thread::spawn(move || unsafe {
                let peer = rank ^ 1;
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
                let mut allocations = Vec::<*mut c_void>::new();
                let mut upload = |bytes: &[u8]| {
                    let mut p = std::ptr::null_mut();
                    assert_eq!(cudaMalloc(&mut p, bytes.len()), 0);
                    allocations.push(p);
                    assert_eq!(cudaMemcpy(p, bytes.as_ptr().cast(), bytes.len(), 1), 0);
                    p
                };
                let mut parents = [0u8; 32];
                parents[..4].copy_from_slice(&[1, 0, 0, 1]);
                parents[16..20].copy_from_slice(&[1, rank as u8 + 1, 0, 1]);
                let p = upload(&parents);
                let g = upload(&[1, 1, 0, 1, 1, 0, 1, 1]);
                let send_requests = upload(&[0u8; 48]);
                let recv_requests = upload(&[0u8; 48]);
                let send_states = upload(&[0u8; 48]);
                let recv_states = upload(&[0u8; 48]);
                let count = upload(&0u32.to_le_bytes());
                let fatal = upload(&0u32.to_le_bytes());
                let group_fatal = upload(&0u32.to_le_bytes());
                let source = MatrixSource {
                    n: 2,
                    moves: 2,
                    modulus: 5,
                    stride: 16,
                    rank,
                    parent_begin: 100 + u64::from(rank) * 10,
                    parent_count: 2,
                    parents: p.cast(),
                    generators: g.cast(),
                };
                for (epoch, counts) in [[3u32, 2], [0, 2], [0, 0], [1, 1]].into_iter().enumerate() {
                    let outgoing = counts[rank as usize];
                    let incoming = counts[peer as usize];
                    let begin = 100 + u64::from(peer) * 10;
                    let mut origins = [
                        OriginRef {
                            source: peer,
                            movement: 1,
                            parent: begin + 1,
                        },
                        OriginRef {
                            source: peer,
                            movement: 0,
                            parent: begin,
                        },
                        OriginRef {
                            source: peer,
                            movement: 0,
                            parent: begin + 1,
                        },
                    ];
                    if epoch == 3 && rank == 0 {
                        origins[0].source = rank;
                    }
                    let bytes: Vec<u8> = origins.into_iter().flat_map(|x| x.encode()).collect();
                    assert_eq!(cudaMemcpy(send_requests, bytes.as_ptr().cast(), 48, 1), 0);
                    assert_eq!(cudaMemcpy(count, (&incoming as *const u32).cast(), 4, 1), 0);
                    let buffers = ExchangeBuffers {
                        capacity: 3,
                        outgoing_count: outgoing,
                        incoming_count: incoming,
                        incoming_count_device: count.cast(),
                        outgoing_requests: send_requests.cast(),
                        incoming_requests: recv_requests.cast(),
                        outgoing_responses: send_states.cast(),
                        incoming_responses: recv_states.cast(),
                        local_fatal: fatal.cast(),
                        group_fatal: group_fatal.cast(),
                    };
                    enqueue_round_trip(comm, peer, &source, &buffers, stream).unwrap();
                    assert_eq!(cudaStreamSynchronize(stream), 0);
                    let mut status = 99u32;
                    assert_eq!(
                        cudaMemcpy((&mut status as *mut u32).cast(), group_fatal, 4, 2),
                        0
                    );
                    assert_eq!(status, if epoch == 3 { 2 } else { 0 });
                    if epoch != 3 {
                        let mut actual = [0u8; 48];
                        assert_eq!(
                            cudaMemcpy(actual.as_mut_ptr().cast(), recv_states, 48, 2),
                            0
                        );
                        let a = peer as u8 + 1;
                        let matrices = [[1, a, 1, a + 1], [1, 1, 0, 1], [1, a + 1, 0, 1]];
                        for i in 0..outgoing as usize {
                            assert_eq!(&actual[i * 16..i * 16 + 4], &matrices[i]);
                            assert!(actual[i * 16 + 4..(i + 1) * 16].iter().all(|&x| x == 0));
                        }
                    }
                }
                assert_eq!(cudaStreamDestroy(stream), 0);
                for allocation in allocations {
                    assert_eq!(cudaFree(allocation), 0);
                }
                mgbfs_nccl_destroy(comm);
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn hash_only_origins_regenerate_cpu_successors_without_host_count_readback() {
    use mgbfs_cuda::ffi::*;
    use mgbfs_cuda::native_owner::cudaSetDevice;
    use std::ffi::c_void;
    unsafe {
        for rank in 0..2 {
            assert_eq!(cudaSetDevice(rank), 0);
            for (n, modulus) in [(7, 2), (7, 5), (3, 256)] {
                let group = MatrixGroup::unitriangular(n, modulus).unwrap();
                let width = n * n;
                let stride = (width + 15) & !15;
                let moves = group.generators.len();
                let mut states = vec![group.start.clone()];
                for i in 0..8 {
                    states.push(group.successor(states.last().unwrap(), i % moves).unwrap());
                }
                let rows = states.len() * moves;
                let mut packed = vec![0u8; states.len() * stride];
                for (dst, state) in packed.chunks_exact_mut(stride).zip(&states) {
                    dst[..width].copy_from_slice(state);
                }
                let children: Vec<Vec<u8>> = states
                    .iter()
                    .flat_map(|state| {
                        (0..moves)
                            .map(|m| group.successor(state, m).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect();
                for seed in [0u128, 1, 20260828] {
                    let hash = GemmHash::from_seed(width, seed.to_le_bytes()).unwrap();
                    let mut allocations = Vec::<*mut c_void>::new();
                    let mut upload = |bytes: &[u8]| {
                        let mut p = std::ptr::null_mut();
                        assert_eq!(cudaMalloc(&mut p, bytes.len()), 0);
                        allocations.push(p);
                        assert_eq!(cudaMemcpy(p, bytes.as_ptr().cast(), bytes.len(), 1), 0);
                        p
                    };
                    let parents = upload(&packed);
                    let generators = upload(&group.generators.concat());
                    let weights = upload(&hash.limbs());
                    let offsets = upload(
                        &hash
                            .offsets
                            .iter()
                            .flat_map(|x| x.to_le_bytes())
                            .collect::<Vec<_>>(),
                    );
                    let count = upload(&(states.len() as u32).to_le_bytes());
                    let output_count = upload(&0u32.to_le_bytes());
                    let fatal = upload(&0u32.to_le_bytes());
                    let hashes = upload(&vec![0xcc; rows * 16]);
                    let origins = upload(&vec![0xcc; rows * 16]);
                    let responses = upload(&vec![0xcc; rows * stride]);
                    let mut stream = std::ptr::null_mut();
                    assert_eq!(cudaStreamCreateWithFlags(&mut stream, 1), 0);
                    let begin = 0x100000001u64;
                    assert_eq!(
                        mgbfs_generate_hash_only(
                            n as u32,
                            moves as u32,
                            modulus as u32,
                            stride as u32,
                            states.len() as u32,
                            rows as u32,
                            rank as u32,
                            begin,
                            parents.cast(),
                            generators.cast(),
                            weights.cast(),
                            offsets.cast(),
                            count.cast(),
                            hashes.cast(),
                            origins.cast(),
                            output_count.cast(),
                            fatal.cast(),
                            stream
                        ),
                        0
                    );
                    // Device-generated origins/count feed regeneration on the same stream.
                    assert_eq!(
                        mgbfs_regenerate_selected(
                            n as u32,
                            moves as u32,
                            modulus as u32,
                            stride as u32,
                            rows as u32,
                            rank as u32,
                            begin,
                            states.len() as u32,
                            parents.cast(),
                            generators.cast(),
                            origins.cast(),
                            output_count.cast(),
                            responses.cast(),
                            fatal.cast(),
                            stream
                        ),
                        0
                    );
                    assert_eq!(cudaStreamSynchronize(stream), 0);
                    let mut actual_hashes = vec![0u8; rows * 16];
                    let mut actual_states = vec![0u8; rows * stride];
                    assert_eq!(
                        cudaMemcpy(actual_hashes.as_mut_ptr().cast(), hashes, rows * 16, 2),
                        0
                    );
                    assert_eq!(
                        cudaMemcpy(
                            actual_states.as_mut_ptr().cast(),
                            responses,
                            rows * stride,
                            2
                        ),
                        0
                    );
                    for (i, child) in children.iter().enumerate() {
                        assert_eq!(
                            &actual_hashes[i * 16..(i + 1) * 16],
                            &hash.hash(child).unwrap().to_le_bytes()
                        );
                        assert_eq!(&actual_states[i * stride..i * stride + width], child);
                        assert!(actual_states[i * stride + width..(i + 1) * stride]
                            .iter()
                            .all(|&x| x == 0));
                    }
                    let mut value = 99u32;
                    assert_eq!(cudaMemcpy((&mut value as *mut u32).cast(), fatal, 4, 2), 0);
                    assert_eq!(value, 0);
                    assert_eq!(
                        cudaMemcpy((&mut value as *mut u32).cast(), output_count, 4, 2),
                        0
                    );
                    assert_eq!(value, rows as u32);
                    assert_eq!(cudaStreamDestroy(stream), 0);
                    for allocation in allocations {
                        assert_eq!(cudaFree(allocation), 0);
                    }
                }
            }
        }
    }
}

// Catches ABI width/order mismatches between wire OriginRef and CUDA regeneration.
#[test]
fn wire_origins_regenerate_selected_states_on_each_device() {
    use mgbfs_core::wire::OriginRef;
    use mgbfs_cuda::ffi::*;
    use mgbfs_cuda::native_owner::cudaSetDevice;
    use std::ffi::c_void;
    unsafe {
        for rank in 0..2 {
            assert_eq!(cudaSetDevice(rank), 0);
            let mut allocations = Vec::<*mut c_void>::new();
            let mut upload = |bytes: &[u8]| {
                let mut p = std::ptr::null_mut();
                assert_eq!(cudaMalloc(&mut p, bytes.len()), 0);
                allocations.push(p);
                assert_eq!(cudaMemcpy(p, bytes.as_ptr().cast(), bytes.len(), 1), 0);
                p
            };
            let mut parents = [0u8; 32];
            parents[..4].copy_from_slice(&[1, 0, 0, 1]);
            parents[16..20].copy_from_slice(&[1, 2, 0, 1]);
            let p = upload(&parents);
            let g = upload(&[1, 1, 0, 1, 1, 0, 1, 1]);
            let requests: Vec<u8> = [(101, 1), (100, 0), (101, 0)]
                .into_iter()
                .flat_map(|(parent, movement)| {
                    OriginRef {
                        source: rank as u32,
                        movement,
                        parent,
                    }
                    .encode()
                })
                .collect();
            let r = upload(&requests);
            let count = upload(&3u32.to_le_bytes());
            let fatal = upload(&0u32.to_le_bytes());
            let output = upload(&[0xcc; 48]);
            assert_eq!(
                mgbfs_regenerate_selected(
                    2,
                    2,
                    5,
                    16,
                    3,
                    rank as u32,
                    100,
                    2,
                    p.cast(),
                    g.cast(),
                    r.cast(),
                    count.cast(),
                    output.cast(),
                    fatal.cast(),
                    std::ptr::null_mut()
                ),
                0
            );
            assert_eq!(cudaDeviceSynchronize(), 0);
            let mut actual = [0u8; 48];
            assert_eq!(cudaMemcpy(actual.as_mut_ptr().cast(), output, 48, 2), 0);
            let mut expected = [0u8; 48];
            expected[..4].copy_from_slice(&[1, 2, 1, 3]);
            expected[16..20].copy_from_slice(&[1, 1, 0, 1]);
            expected[32..36].copy_from_slice(&[1, 3, 0, 1]);
            assert_eq!(actual, expected);
            let mut error = 99u32;
            assert_eq!(cudaMemcpy((&mut error as *mut u32).cast(), fatal, 4, 2), 0);
            assert_eq!(error, 0);
            for allocation in allocations {
                assert_eq!(cudaFree(allocation), 0);
            }
        }
    }
}
impl Extent for Disk {
    fn reserve(&mut self, n: u64) -> std::io::Result<()> {
        self.0.lock().unwrap().resize(n as usize, 0);
        Ok(())
    }
    fn write_at(&mut self, at: u64, x: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap()[at as usize..at as usize + x.len()].copy_from_slice(x);
        Ok(x.len())
    }
    fn sync(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn parent_batch_archive_preserves_full_layers_and_hashes_on_two_devices() {
    archive_fixture(false);
}

#[test]
fn compact_permutation_archive_preserves_full_layers_and_hashes_on_two_devices() {
    archive_fixture(true);
}

fn archive_fixture(compact: bool) {
    let width = if compact { 4 } else { 9 };
    let capacity = if compact { 24 } else { 27 };
    let mut id = [0u8; 128];
    assert_eq!(
        unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) },
        0
    );
    let workers: Vec<_> = (0..2)
        .map(|rank| {
            std::thread::spawn(move || {
                let g = if compact {
                    MatrixGroup::symmetric_permutation_matrices(4).unwrap()
                } else {
                    MatrixGroup::unitriangular(3, 3).unwrap()
                };
                let cfg = DistributedConfig {
                    rank,
                    world: 2,
                    logical_owner_to_rank: [1, 0],
                    batch: 7,
                    layer_capacity: capacity,
                    state_ring_capacity: capacity,
                    buckets: 8,
                    shards: 2,
                    job_buckets: 2,
                    bucket_capacity: capacity,
                    prededup: true,
                    generation_variant: if compact { 5 } else { 1 },
                };
                let mut bfs = DistributedNativeBfs::new(&g, [0; 16], id, cfg).unwrap();
                let data = Arc::new(Mutex::new(Vec::new()));
                // Archive rows deliberately smaller than a compute batch.
                let mut archive =
                    PinnedArchive::new(Disk(data.clone()), 100_000, width, [0; 32], 3, 64).unwrap();
                while bfs.advance_archived(&mut archive).unwrap() {}
                archive.finish().unwrap();
                let bytes = data.lock().unwrap().clone();
                verify(&bytes).unwrap();
                bytes
            })
        })
        .collect();
    let g = if compact {
        MatrixGroup::symmetric_permutation_matrices(4).unwrap()
    } else {
        MatrixGroup::unitriangular(3, 3).unwrap()
    };
    let mut expected = g.exact_layers(capacity as usize).unwrap();
    if compact {
        for layer in &mut expected {
            for state in layer.iter_mut() {
                *state = encode_permutation_matrix(state, 4).unwrap();
            }
            layer.sort();
        }
    }
    let mut actual = vec![Vec::new(); expected.len()];
    let hash = GemmHash::from_seed(width, [0; 16]).unwrap();
    for worker in workers {
        let data = worker.join().unwrap();
        let mut at = 48;
        loop {
            let word =
                |o| u64::from_le_bytes(data[at + o..at + o + 8].try_into().unwrap()) as usize;
            let (kind, depth, count, size) = (word(8), word(16), word(24), word(32));
            if kind == 3 {
                break;
            }
            if kind == 1 {
                assert!(count <= 3);
                let payload = &data[at + 80..at + 80 + size];
                for row in 0..count {
                    let state = &payload[row * width..(row + 1) * width];
                    assert_eq!(
                        hash.hash(state).unwrap().to_le_bytes(),
                        payload[count * width + row * 16..count * width + (row + 1) * 16]
                    );
                    actual[depth].push(state.to_vec());
                }
            }
            at += 112 + size;
        }
    }
    for layer in &mut actual {
        layer.sort();
    }
    assert_eq!(actual, expected);
}
