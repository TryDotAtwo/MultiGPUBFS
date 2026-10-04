#![cfg(all(feature = "cuda", feature = "library-owner"))]
//! Full generation/owner/materialization/depth-rotation gate, not a benchmark.
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::distributed_native::{DistributedConfig, DistributedNativeBfs};
#[path = "support/route_archive.rs"]
mod route_archive;

// The library backend owns a process-global RMM resource. Independent Rust
// test threads must not concurrently replace it. Real rank concurrency is
// tested in separate processes, not by sharing that resource in this binary.
static GPU_FIXTURE: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn gpu_fixture() -> std::sync::MutexGuard<'static, ()> {
    GPU_FIXTURE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn cuco_lsa_route_bank_capacity_failure_latches_after_reuse() {
    let _fixture = gpu_fixture();
    // A missing device capacity decision or failure latch must fail this test.
    // U4/F2 has layers 1,3,5,8,11,...; an eight-state next frontier cannot
    // accept layer four. Batch one exercises bank reuse before that failure.
    let graph = MatrixGroup::unitriangular(4, 2).unwrap();
    let expected = graph.exact_layers(64).unwrap();
    for banks in [2, 3, 4] {
        for hash_first in [false, true] {
            eprintln!("ROUTE_BANK_CAPACITY banks={banks} hash_first={hash_first} phase=create");
            let mut id = [0u8; 128];
            assert_eq!(unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
            let cfg = DistributedConfig {
                route_banks: banks, epoch_window: 3,
                rank: 0, world: 1, logical_owner_to_rank: vec![0, 0],
                transport: mgbfs_core::config::ReferenceTransport::Lsa,
                batch: 1, layer_capacity: 8, state_ring_capacity: 256,
                state_descriptor_capacity: 256, buckets: 8, shards: 2,
                job_buckets: 2, bucket_capacity: 32, prededup: true,
                generation_variant: 1, untouched_vram_reserve: 1 << 30,
            };
            let mut bfs = DistributedNativeBfs::new_library_reference_with_owner(
                &graph, [42; 16], id, cfg,
                hash_first.then_some(graph.generators.len() as u32),
                64 << 20, false, mgbfs_core::config::ReferenceOwner::CucoRank,
            ).unwrap();
            for depth in 0..3 {
                let mut actual = bfs.snapshot().unwrap();
                actual.sort();
                assert_eq!(actual, expected[depth]);
                assert!(bfs.advance().unwrap());
            }
            assert_eq!(bfs.snapshot().unwrap().len(), 8);
            let error = bfs.advance().expect_err("overfull next frontier was admitted");
            assert!(error.starts_with("GROUP_OWNER_OR_PRE_OWNER_FATAL"),
                "unexpected failure instead of device capacity fatal: {error}");
            assert!(bfs.route_bank_reuses() > 0, "failure did not exercise source-bank reuse");
            assert_eq!(bfs.advance().unwrap_err(), "DISTRIBUTED_FAILED");
            eprintln!("ROUTE_BANK_CAPACITY banks={banks} hash_first={hash_first} phase=failed error={error}");
        }
    }
}

#[test]
fn cuco_lsa_route_banks_preserve_full_states_and_reuse_across_depths() {
    let _fixture = gpu_fixture();
    let graph = MatrixGroup::unitriangular(4, 2).unwrap();
    let expected = graph.exact_layers(64).unwrap();
    for banks in [2, 3, 4] {
        for hash_first in [false, true] {
            eprintln!("ROUTE_BANK_GATE banks={banks} hash_first={hash_first} phase=create_begin");
            let mut id = [0u8; 128];
            assert_eq!(unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
            let cfg = DistributedConfig {
                route_banks: banks, epoch_window: 3,
                rank: 0, world: 1, logical_owner_to_rank: vec![0, 0],
                transport: mgbfs_core::config::ReferenceTransport::Lsa,
                batch: 1, layer_capacity: 64, state_ring_capacity: 256,
                state_descriptor_capacity: 256, buckets: 8, shards: 2,
                job_buckets: 2, bucket_capacity: 32, prededup: true,
                generation_variant: 1, untouched_vram_reserve: 1 << 30,
            };
            let materialization = hash_first.then_some(graph.generators.len() as u32);
            let mut bfs = DistributedNativeBfs::new_library_reference_with_owner(
                &graph, [42; 16], id, cfg, materialization, 64 << 20, false,
                mgbfs_core::config::ReferenceOwner::CucoRank,
            ).unwrap();
            eprintln!("ROUTE_BANK_GATE banks={banks} hash_first={hash_first} phase=create_done");
            assert_eq!(bfs.route_bank_count(), banks);
            assert_eq!(bfs.epoch_window(), 3);
            assert_eq!(bfs.route_bank_reuses(), 0);
            let mut expected_reuses = 0u64;
            let archive_bytes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut archive = mgbfs_runtime::pinned_archive::PinnedArchive::new(
                route_archive::MemoryExtent(archive_bytes.clone()),
                100_000, 16, [0; 32], 3, 128).unwrap();
            for (depth, wanted) in expected.iter().enumerate() {
                let mut actual = bfs.snapshot().unwrap();
                actual.sort();
                assert_eq!(&actual, wanted, "banks={banks} hash_first={hash_first} depth={depth}");
                assert_eq!(bfs.advance_archived(&mut archive).unwrap(), depth + 1 < expected.len());
                // Batch=1: the first N parents use distinct banks; every
                // remaining parent must reuse one bank at this depth.
                expected_reuses += wanted.len().saturating_sub(banks) as u64;
                assert_eq!(bfs.route_bank_reuses(), expected_reuses);
                eprintln!("ROUTE_BANK_GATE banks={banks} hash_first={hash_first} phase=depth_done depth={depth}");
            }
            archive.finish().unwrap();
            route_archive::assert_layers(&archive_bytes.lock().unwrap(), &expected, [42; 16]);
            assert!(bfs.dense_lookahead_batches() > banks as u64);
            assert!(bfs.route_bank_reuses() > 0,
                "fixture did not reuse a physical bank within a depth");
        }
    }
}

#[test]
fn cuco_rank_dense_layers_match_full_state_oracle() {
    let _fixture = gpu_fixture();
    let graph = MatrixGroup::unitriangular(4, 2).unwrap();
    let expected = graph.exact_layers(64).unwrap();
    let mut id = [0u8; 128];
    assert_eq!(unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
    let cfg = DistributedConfig {
        route_banks: 2,
        epoch_window: 2,
        rank: 0,
        world: 1,
        logical_owner_to_rank: vec![0, 0],
        transport: mgbfs_core::config::ReferenceTransport::HostSizedNccl,
        batch: 7,
        layer_capacity: 64,
        state_ring_capacity: 128, state_descriptor_capacity: 128,
        buckets: 8,
        shards: 2,
        job_buckets: 2,
        bucket_capacity: 32,
        prededup: true,
        generation_variant: 1,
        untouched_vram_reserve: 1 << 30,
    };
    let mut bfs = DistributedNativeBfs::new_library_reference_with_owner(
        &graph, [0; 16], id, cfg, None, 64 << 20, false,
        mgbfs_core::config::ReferenceOwner::CucoRank,
    ).unwrap();
    for (depth, wanted) in expected.iter().enumerate() {
        let mut actual = bfs.snapshot().unwrap();
        actual.sort();
        assert_eq!(&actual, wanted, "depth={depth}");
        assert_eq!(bfs.advance().unwrap(), depth + 1 < expected.len());
    }
}

#[test]
fn cuco_rank_capacity_failure_releases_pool_after_gpu_work() {
    let _fixture = gpu_fixture();
    let graph = MatrixGroup::unitriangular(4, 2).unwrap();
    let mut id = [0u8; 128];
    assert_eq!(unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
    let mut cfg = DistributedConfig {
        route_banks: 2,
        epoch_window: 2,
        rank: 0,
        world: 1,
        logical_owner_to_rank: vec![0, 0],
        transport: mgbfs_core::config::ReferenceTransport::HostSizedNccl,
        batch: 7,
        layer_capacity: 2,
        state_ring_capacity: 128, state_descriptor_capacity: 128,
        buckets: 8,
        shards: 2,
        job_buckets: 2,
        bucket_capacity: 32,
        prededup: true,
        generation_variant: 1,
        untouched_vram_reserve: 1 << 30,
    };
    let mut failed = DistributedNativeBfs::new_library_reference_with_owner(
        &graph, [0; 16], id, cfg.clone(), None, 64 << 20, false,
        mgbfs_core::config::ReferenceOwner::CucoRank,
    ).unwrap();
    let error = failed.advance().unwrap_err();
    assert!(error.contains("LIBRARY_RANK_DEPTH_FATAL_"), "{error}");
    drop(failed);
    cfg.layer_capacity = 64;
    assert_eq!(unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
    let mut recovered = DistributedNativeBfs::new_library_reference_with_owner(
        &graph, [0; 16], id, cfg, None, 64 << 20, false,
        mgbfs_core::config::ReferenceOwner::CucoRank,
    ).unwrap();
    assert!(recovered.advance().unwrap());
}

#[test]
fn library_bfs_layers_match_full_state_oracle_in_both_profiles() {
    let _fixture = gpu_fixture();
    let graph = MatrixGroup::unitriangular(4, 2).unwrap();
    let expected = graph.exact_layers(64).unwrap();
    for library_owner in [
        mgbfs_core::config::ReferenceOwner::CudfRelational,
        mgbfs_core::config::ReferenceOwner::CucoIndexed,
    ] {
        for (materialization_capacity, tensor_generation) in
            [(None, false), (Some(128), false), (Some(128), true)]
        {
            for prededup in [false, true] {
                let mut id = [0u8; 128];
                assert_eq!(
                    unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) },
                    0
                );
                let cfg = DistributedConfig {
                    route_banks: 2,
                    epoch_window: 2,
                    rank: 0,
                    world: 1,
                    logical_owner_to_rank: vec![0, 0],
                    transport: mgbfs_core::config::ReferenceTransport::HostSizedNccl,
                    batch: 7,
                    layer_capacity: 64,
                    state_ring_capacity: 128, state_descriptor_capacity: 128,
                    buckets: 8,
                    shards: 2,
                    job_buckets: 2,
                    bucket_capacity: 32,
                    prededup,
                    generation_variant: 1,
                    untouched_vram_reserve: 1 << 30,
                };
                // Guard the unchanged default path after separating its storage.
                // This is correctness evidence only, not a timed A/B run.
                let mut baseline = if tensor_generation {
                    DistributedNativeBfs::new_hash_first_tc_with_owner(
                        &graph,
                        [0; 16],
                        id,
                        cfg.clone(),
                        128,
                        mgbfs_core::config::OwnerBackend::CubSortMerge,
                        256,
                    )
                } else if let Some(capacity) = materialization_capacity {
                    DistributedNativeBfs::new_hash_first_reference(
                        &graph,
                        [0; 16],
                        id,
                        cfg.clone(),
                        capacity,
                    )
                } else {
                    DistributedNativeBfs::new(&graph, [0; 16], id, cfg.clone())
                }
                .unwrap();
                for (depth, wanted) in expected.iter().enumerate() {
                    let mut actual = baseline.snapshot().unwrap();
                    actual.sort();
                    assert_eq!(&actual, wanted);
                    assert_eq!(baseline.advance().unwrap(), depth + 1 < expected.len());
                }
                drop(baseline);
                assert_eq!(
                    unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) },
                    0
                );
                let mut bfs = DistributedNativeBfs::new_library_reference_with_owner(
                    &graph,
                    [0; 16],
                    id,
                    cfg,
                    materialization_capacity,
                    64 << 20,
                    tensor_generation,
                    library_owner,
                )
                .unwrap();
                assert!(!bfs
                    .shared_memory()
                    .allocations
                    .iter()
                    .any(|a| a.name == "accepted"));
                let pool = bfs
                    .owned_memory()
                    .allocations
                    .iter()
                    .find(|a| a.name == "library.fixed_pool")
                    .unwrap();
                assert_eq!(pool.payload_bytes, 64 << 20);
                assert!(!bfs
                    .owned_memory()
                    .allocations
                    .iter()
                    .any(|a| a.name.starts_with("owner.")));
                for (depth, wanted) in expected.iter().enumerate() {
                    let mut actual = bfs.snapshot().unwrap();
                    actual.sort();
                    assert_eq!(
                    &actual, wanted,
                    "depth={depth} prededup={prededup} hash_first={materialization_capacity:?} tensor={tensor_generation}"
                );
                    assert_eq!(bfs.advance().unwrap(), depth + 1 < expected.len());
                }
            }
        }
    }
}
