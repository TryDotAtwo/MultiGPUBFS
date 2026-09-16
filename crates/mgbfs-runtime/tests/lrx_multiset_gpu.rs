#![cfg(all(feature = "cuda", feature = "library-owner"))]
use mgbfs_core::{config::{OwnerBackend, ReferenceOwner}, lrx_multiset::LrxMultiset};
use mgbfs_runtime::distributed_native::{DistributedConfig, DistributedNativeBfs};

#[test]
#[ignore = "requires eight physical GPUs; compares full words, not only counts"]
fn lrx_multiset_one_two_eight_rank_full_state_oracle() {
    for world in [1u32, 2, 8] {
        for n in [5, 7] {
            for owner in [ReferenceOwner::Native(OwnerBackend::CubSortMerge), ReferenceOwner::CucoIndexed] {
                for prededup in [false, true] {
                    let mut expected = LrxMultiset::new(n,4).unwrap().exact_layers(256).unwrap();
                    let mut id = [0u8;128];
                    assert_eq!(unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) },0);
                    let workers: Vec<_> = (0..world).map(|rank| std::thread::spawn(move || {
                        std::panic::catch_unwind(|| {
                            let graph = LrxMultiset::new(n,4).unwrap();
                            let cfg = DistributedConfig {
                                rank, world, logical_owner_to_rank: (0..world).rev().collect(),
                                batch: 2, layer_capacity: 256, state_ring_capacity: 512,
                                buckets: world*4, shards: world*2, job_buckets: 2,
                                bucket_capacity: 256, prededup, generation_variant: 5,
                                untouched_vram_reserve: 1<<30,
                            };
                            let pool = (owner == ReferenceOwner::CucoIndexed).then_some(64<<20);
                            let mut bfs = DistributedNativeBfs::new_lrx_multiset_reference(
                                &graph,[7;16],id,cfg,owner,pool).unwrap();
                            let mut layers = Vec::new();
                            loop {
                                layers.push(bfs.snapshot().unwrap());
                                if !bfs.advance().unwrap() { break; }
                            }
                            layers
                        }).unwrap_or_else(|_| std::process::abort())
                    })).collect();
                    let mut actual = vec![Vec::new(); expected.len()];
                    for worker in workers {
                        let local = worker.join().unwrap();
                        assert_eq!(local.len(), actual.len());
                        for (dst, src) in actual.iter_mut().zip(local) { dst.extend(src); }
                    }
                    for layer in &mut actual { layer.sort(); }
                    for layer in &mut expected { layer.sort(); }
                    assert_eq!(actual, expected, "world={world} n={n} owner={owner:?} prededup={prededup}");
                    eprintln!("LRX_MULTISET_ORACLE_PASS world={world} n={n} owner={owner:?} prededup={prededup}");
                }
            }
        }
    }
}
