#![cfg(all(feature = "cuda", feature = "library-owner"))]
//! Requires two physical GPUs visible in one process. Not a torchrun benchmark.
use mgbfs_core::{hash::GemmHash, matrix::MatrixGroup};
use mgbfs_runtime::{
    archive::{verify, Extent},
    distributed_native::{DistributedConfig, DistributedNativeBfs},
    pinned_archive::PinnedArchive,
};
use std::sync::{Arc, Mutex};

struct TestDisk(Arc<Mutex<Vec<u8>>>);

struct SlowDisk(TestDisk, bool);

impl Extent for SlowDisk {
    fn reserve(&mut self, bytes: u64) -> std::io::Result<()> {
        self.0.reserve(bytes)
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> std::io::Result<usize> {
        if self.1 {
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
        self.0.write_at(offset, bytes)
    }
    fn sync(&mut self) -> std::io::Result<()> {
        self.0.sync()
    }
}

#[test]
#[ignore = "requires two physical GPUs; injects one-rank archive slot exhaustion"]
fn archive_slot_failure_votes_group_fatal_before_exchange() {
    archive_slot_failure_fixture(mgbfs_core::config::ReferenceTransport::HostSizedNccl);
}

#[test]
#[ignore = "requires two physical P2P GPUs and NCCL LSA; injects one-rank archive slot exhaustion"]
fn archive_slot_failure_votes_group_fatal_before_lsa_exchange() {
    archive_slot_failure_fixture(mgbfs_core::config::ReferenceTransport::Lsa);
}

fn archive_slot_failure_fixture(transport: mgbfs_core::config::ReferenceTransport) {
    let graph = MatrixGroup::symmetric_permutation_matrices(4).unwrap();
    let mut id = [0u8; 128];
    assert_eq!(
        unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) },
        0
    );
    let workers: Vec<_> = (0..2u32)
        .map(|rank| {
            let graph = graph.clone();
            std::thread::spawn(move || {
                let cfg = DistributedConfig {
                    rank,
                    world: 2,
                    logical_owner_to_rank: vec![0, 1],
                    batch: 8,
                    layer_capacity: 64,
                    state_ring_capacity: 64,
                    buckets: 8,
                    shards: 4,
                    job_buckets: 2,
                    bucket_capacity: 32,
                    prededup: false,
                    transport,
                    generation_variant: 1,
                    untouched_vram_reserve: 1 << 30,
                };
                let mut bfs = if transport == mgbfs_core::config::ReferenceTransport::Lsa {
                    DistributedNativeBfs::new_library_reference_with_owner(
                        &graph, [7; 16], id, cfg, None, 64 << 20, false,
                        mgbfs_core::config::ReferenceOwner::CucoRank,
                    )
                } else {
                    DistributedNativeBfs::new_reference_with_owner(
                        &graph, [7; 16], id, cfg, None,
                        mgbfs_core::config::OwnerBackend::CubSortMerge, 256,
                    )
                }
                .unwrap();
                let disk = Arc::new(Mutex::new(Vec::new()));
                let mut archive = PinnedArchive::new(
                    SlowDisk(TestDisk(disk), rank == 0),
                    100_000,
                    graph.start.len(),
                    [0; 32],
                    1,
                    if rank == 0 { 2 } else { 128 },
                )
                .unwrap();
                loop {
                    match bfs.advance_archived(&mut archive) {
                        Ok(true) => continue,
                        Ok(false) => return Err("ARCHIVE_FAULT_NOT_TRIGGERED".to_string()),
                        Err(error) => return Ok(error),
                    }
                }
            })
        })
        .collect();
    let errors: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap().unwrap())
        .collect();
    assert!(
        errors[0].contains("ARCHIVE_PIN_RING_FATAL"),
        "{}",
        errors[0]
    );
    assert_eq!(errors[1], "REMOTE_ARCHIVE_FATAL", "{transport:?}");
}

#[test]
fn retirement_fifo_fault_votes_group_fatal_on_two_devices() {
    use mgbfs_cuda::{ffi::*, native_owner::*};
    use std::ffi::c_void;

    let mut id = [0u8; 128];
    assert_eq!(unsafe { mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
    let workers: Vec<_> = (0..2u32)
        .map(|rank| {
            std::thread::spawn(move || unsafe {
                assert_eq!(cudaSetDevice(rank as i32), 0);
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
                        error.len(),
                    ),
                    0
                );
                let mut stream = std::ptr::null_mut();
                assert_eq!(cudaStreamCreateWithFlags(&mut stream, 1), 0);
                let mut ring_gpu: *mut c_void = std::ptr::null_mut();
                let mut send: *mut c_void = std::ptr::null_mut();
                let mut receive: *mut c_void = std::ptr::null_mut();
                assert_eq!(cudaMalloc(&mut ring_gpu, std::mem::size_of::<Ring>()), 0);
                assert_eq!(cudaMalloc(&mut send, 4), 0);
                assert_eq!(cudaMalloc(&mut receive, 4), 0);
                let ring = Ring {
                    head: 6,
                    tail: 14,
                    descriptor_head: 0,
                    descriptor_tail: 2,
                    capacity: 10,
                    descriptor_capacity: 4,
                    ..Ring::default()
                };
                assert_eq!(
                    cudaMemcpy(ring_gpu, (&ring as *const Ring).cast(), 64, 1),
                    0
                );
                let extent = Extent {
                    sequence: 6,
                    begin: 6,
                    count: 3,
                    descriptor: if rank == 0 { 1 } else { 0 },
                    granted_rows: 3,
                    ready: 1,
                    ..Extent::default()
                };
                assert_eq!(
                    mgbfs_state_retire_dense_prefix_value(ring_gpu.cast(), extent, 3, stream),
                    0
                );
                assert_eq!(
                    mgbfs_state_ring_fatal_vote_word(ring_gpu.cast(), send.cast(), stream),
                    0
                );
                assert_eq!(
                    mgbfs_nccl_all_reduce_max_u32(comm, send.cast(), receive.cast(), stream),
                    0
                );
                assert_eq!(cudaStreamSynchronize(stream), 0);
                let mut group_fatal = 0u32;
                let mut local = Ring::default();
                assert_eq!(
                    cudaMemcpy((&mut group_fatal as *mut u32).cast(), receive, 4, 2),
                    0
                );
                assert_eq!(
                    cudaMemcpy((&mut local as *mut Ring).cast(), ring_gpu, 64, 2),
                    0
                );
                assert_eq!(cudaFree(receive), 0);
                assert_eq!(cudaFree(send), 0);
                assert_eq!(cudaFree(ring_gpu), 0);
                assert_eq!(cudaStreamDestroy(stream), 0);
                mgbfs_nccl_destroy(comm);
                (group_fatal, local.fatal, local.head)
            })
        })
        .collect();
    let results: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    assert_eq!(results, [(1, 17, 6), (1, 0, 9)]);
}
impl Extent for TestDisk {
    fn reserve(&mut self, bytes: u64) -> std::io::Result<()> {
        self.0.lock().unwrap().resize(bytes as usize, 0);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> std::io::Result<usize> {
        let mut data = self.0.lock().unwrap();
        let end = (offset as usize).checked_add(bytes.len()).unwrap();
        if end > data.len() {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        data[offset as usize..end].copy_from_slice(bytes);
        Ok(bytes.len())
    }
    fn sync(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn library_two_rank_layers_and_archives_match_oracle() {
    for library_owner in [
        mgbfs_core::config::ReferenceOwner::Native(mgbfs_core::config::OwnerBackend::CubSortMerge),
        mgbfs_core::config::ReferenceOwner::CudfRelational,
        mgbfs_core::config::ReferenceOwner::CucoIndexed,
    ] {
        for symmetric in [false, true] {
            for hash_first in [false, true] {
                for owners in [[0, 1], [1, 0]] {
                    fixture(
                        symmetric,
                        hash_first,
                        &owners,
                        library_owner,
                        true,
                        mgbfs_core::config::ReferenceTransport::HostSizedNccl,
                    );
                }
            }
        }
    }
}

#[test]
fn cuco_rank_two_gpu_dense_layers_and_archives_match_oracle() {
    for symmetric in [false, true] {
        for owners in [[0, 1], [1, 0]] {
            for prededup in [false, true] {
                fixture(
                    symmetric,
                    false,
                    &owners,
                    mgbfs_core::config::ReferenceOwner::CucoRank,
                    prededup,
                    mgbfs_core::config::ReferenceTransport::HostSizedNccl,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a two-GPU NCCL 2.29+ LSA-capable P2P host"]
fn cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle() {
    for symmetric in [false, true] {
        for owners in [[0, 1], [1, 0]] {
            for prededup in [false, true] {
                fixture(
                    symmetric,
                    false,
                    &owners,
                    mgbfs_core::config::ReferenceOwner::CucoRank,
                    prededup,
                    mgbfs_core::config::ReferenceTransport::Lsa,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a two-GPU NCCL 2.29+ LSA-capable P2P host"]
fn cuco_rank_lsa_single_fixture_for_sanitizer() {
    fixture(
        false,
        false,
        &[0, 1],
        mgbfs_core::config::ReferenceOwner::CucoRank,
        false,
        mgbfs_core::config::ReferenceTransport::Lsa,
    );
}

#[test]
#[ignore = "requires eight physical CUDA devices; run explicitly on 8-GPU host"]
fn library_eight_rank_layers_and_archives_match_oracle() {
    for backend in [
        mgbfs_core::config::ReferenceOwner::CucoIndexed,
        mgbfs_core::config::ReferenceOwner::Native(mgbfs_core::config::OwnerBackend::CubSortMerge),
    ] {
        for prededup in [false, true] {
            for hash_first in [false, true] {
                for owners in [[0, 1, 2, 3, 4, 5, 6, 7], [7, 3, 0, 6, 1, 5, 2, 4]] {
                    for symmetric in [false, true] {
                        fixture(
                            symmetric,
                            hash_first,
                            &owners,
                            backend,
                            prededup,
                            mgbfs_core::config::ReferenceTransport::HostSizedNccl,
                        );
                    }
                }
            }
        }
    }
}

fn fixture(
    symmetric: bool,
    hash_first: bool,
    owners: &[u32],
    library_owner: mgbfs_core::config::ReferenceOwner,
    prededup: bool,
    transport: mgbfs_core::config::ReferenceTransport,
) {
    let graph = if symmetric {
        MatrixGroup::symmetric_permutation_matrices(4).unwrap()
    } else {
        MatrixGroup::unitriangular(3, 3).unwrap()
    };
    let expected = graph.exact_layers(64).unwrap();
    let width = graph.start.len();
    let seed = [7; 16];
    let mut id = [0u8; 128];
    assert_eq!(
        unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) },
        0
    );
    let world = owners.len() as u32;
    let workers: Vec<_> = (0..world)
        .map(|rank| {
            let graph = graph.clone();
            let owners = owners.to_vec();
            std::thread::spawn(move || {
                // A failed rank must not leave its peer blocked in a collective.
                // The test executable is an isolated rank group owned by the runner.
                std::panic::catch_unwind(|| {
                    if transport == mgbfs_core::config::ReferenceTransport::Lsa {
                        eprintln!("MGBFS_LSA_GATE rank={rank} phase=before_constructor");
                    }
                    let cfg = DistributedConfig {
                        rank,
                        world,
                        logical_owner_to_rank: owners.to_vec(),
                        batch: 1,
                        layer_capacity: 64,
                        state_ring_capacity: 64,
                        buckets: world * 4,
                        shards: world * 2,
                        job_buckets: 2,
                        bucket_capacity: 32,
                        prededup,
                        transport,
                        generation_variant: 1,
                        untouched_vram_reserve: 1 << 30,
                    };
                    let mut bfs =
                        if let mgbfs_core::config::ReferenceOwner::Native(owner) = library_owner {
                            DistributedNativeBfs::new_reference_with_owner(
                                &graph,
                                seed,
                                id,
                                cfg,
                                hash_first.then_some(128),
                                owner,
                                256,
                            )
                        } else {
                            DistributedNativeBfs::new_library_reference_with_owner(
                                &graph,
                                seed,
                                id,
                                cfg,
                                hash_first.then_some(128),
                                64 << 20,
                                false,
                                library_owner,
                            )
                        }
                        .unwrap();
                    if transport == mgbfs_core::config::ReferenceTransport::Lsa {
                        eprintln!("MGBFS_LSA_GATE rank={rank} phase=after_constructor");
                    }
                    let bytes = Arc::new(Mutex::new(Vec::new()));
                    let mut archive = PinnedArchive::new(
                        TestDisk(bytes.clone()),
                        100_000,
                        width,
                        [0; 32],
                        3,
                        128,
                    )
                    .unwrap();
                    let mut layers = Vec::new();
                    loop {
                        layers.push(bfs.snapshot().unwrap());
                        if transport == mgbfs_core::config::ReferenceTransport::Lsa {
                            eprintln!(
                                "MGBFS_LSA_GATE rank={rank} phase=advance depth={}",
                                bfs.depth()
                            );
                        }
                        if !bfs.advance_archived(&mut archive).unwrap() {
                            break;
                        }
                    }
                    archive.finish().unwrap();
                    let data = bytes.lock().unwrap().clone();
                    verify(&data).unwrap();
                    (layers, data)
                })
                .unwrap_or_else(|_| std::process::abort())
            })
        })
        .collect();
    let hash = GemmHash::from_seed(width, seed).unwrap();
    let mut snapshots = vec![Vec::new(); expected.len()];
    let mut archived = vec![Vec::new(); expected.len()];
    for (rank, worker) in workers.into_iter().enumerate() {
        let (layers, bytes) = worker.join().unwrap();
        assert_eq!(layers.len(), expected.len());
        let mut commits = vec![false; expected.len()];
        let mut offset = 48;
        loop {
            let word = |field| {
                u64::from_le_bytes(
                    bytes[offset + field..offset + field + 8]
                        .try_into()
                        .unwrap(),
                ) as usize
            };
            let (kind, depth, count, size) = (word(8), word(16), word(24), word(32));
            if kind == 3 {
                break;
            }
            assert!(depth < expected.len());
            if kind == 1 {
                let payload = &bytes[offset + 80..offset + 80 + size];
                assert!(count <= 3);
                for row in 0..count {
                    let state = &payload[row * width..(row + 1) * width];
                    let key = hash.hash(state).unwrap();
                    assert_eq!(
                        owners[((u64::from(key.0[3]) * u64::from(world)) >> 32) as usize],
                        rank as u32
                    );
                    assert_eq!(
                        key.to_le_bytes(),
                        payload[count * width + row * 16..count * width + (row + 1) * 16]
                    );
                    archived[depth].push(state.to_vec());
                }
            } else if kind == 2 {
                assert!(!commits[depth]);
                commits[depth] = true;
                assert_eq!(count, layers[depth].len());
            } else {
                panic!("Unexpected archive frame {kind}");
            }
            offset += 112 + size;
        }
        assert!(commits.into_iter().all(|present| present));
        for (depth, layer) in layers.into_iter().enumerate() {
            snapshots[depth].extend(layer);
        }
    }
    for layers in [&mut snapshots, &mut archived] {
        for layer in layers.iter_mut() {
            layer.sort();
        }
        assert_eq!(
            layers, &expected,
            "symmetric={symmetric} hash_first={hash_first} owners={owners:?}"
        );
    }
}
