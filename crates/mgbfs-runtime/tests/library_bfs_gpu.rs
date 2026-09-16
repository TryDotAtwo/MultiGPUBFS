#![cfg(all(feature = "cuda", feature = "library-owner"))]
//! Full generation/owner/materialization/depth-rotation gate, not a benchmark.
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::distributed_native::{DistributedConfig, DistributedNativeBfs};

#[test]
fn library_bfs_layers_match_full_state_oracle_in_both_profiles() {
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
                    rank: 0,
                    world: 1,
                    logical_owner_to_rank: [0, 0],
                    batch: 7,
                    layer_capacity: 64,
                    state_ring_capacity: 128,
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
                        cfg,
                        128,
                        mgbfs_core::config::OwnerBackend::CubSortMerge,
                        256,
                    )
                } else if let Some(capacity) = materialization_capacity {
                    DistributedNativeBfs::new_hash_first_reference(
                        &graph, [0; 16], id, cfg, capacity,
                    )
                } else {
                    DistributedNativeBfs::new(&graph, [0; 16], id, cfg)
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
