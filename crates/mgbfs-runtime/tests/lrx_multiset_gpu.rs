#![cfg(all(feature = "cuda", feature = "library-owner"))]
use mgbfs_core::{config::{OwnerBackend, ReferenceOwner}, lrx_multiset::LrxMultiset};
use mgbfs_runtime::distributed_native::{DistributedConfig, DistributedNativeBfs};

#[test]
#[ignore = "requires eight physical GPUs; compares full words, not only counts"]
fn lrx_multiset_one_two_eight_rank_full_state_oracle() {
    multiset_oracle(&[1, 2, 8], false);
}

#[test]
#[ignore = "requires two physical GPUs; compares full words for both cuco backends"]
fn lrx_multiset_two_rank_cuco_full_state_oracle() {
    multiset_oracle(&[2], true);
}

#[test]
#[ignore = "requires two P2P/LSA GPUs and MGBFS_CUDA_GRAPH_BATCHES=32; full windows and all states"]
fn lrx_multiset_two_rank_graph_windows_full_state_oracle() {
    assert_eq!(std::env::var("MGBFS_CUDA_GRAPH_BATCHES").as_deref(), Ok("32"));
    for (n, r) in [(7, 1), (8, 4)] {
        for prededup in [false, true] {
            for reversed in [false, true] {
                let mut expected = LrxMultiset::new(n, r).unwrap().exact_layers(8192).unwrap();
                let mut id = [0u8;128];
                assert_eq!(unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) }, 0);
                let workers: Vec<_> = (0..2).map(|rank| std::thread::spawn(move || {
                    std::panic::catch_unwind(|| {
                        let cfg = DistributedConfig {
                            rank, world: 2,
                            logical_owner_to_rank: if reversed { vec![1, 0] } else { vec![0, 1] },
                            transport: mgbfs_core::config::ReferenceTransport::Lsa,
                            batch: 2, layer_capacity: 2048, state_ring_capacity: 4096,
                            buckets: 8, shards: 4, job_buckets: 2, bucket_capacity: 2048,
                            prededup, generation_variant: 5, untouched_vram_reserve: 1<<30,
                        };
                        let graph = LrxMultiset::new(n,r).unwrap();
                        let mut bfs = DistributedNativeBfs::new_lrx_multiset_reference(
                            &graph,[7;16],id,cfg,ReferenceOwner::CucoRank,Some(64<<20)).unwrap();
                        let mut layers = Vec::new();
                        let trace_depths = std::env::var_os("MGBFS_TEST_GRAPH_DEPTH_TRACE").is_some();
                        loop {
                            layers.push(bfs.snapshot().unwrap());
                            if trace_depths {
                                eprintln!("GRAPH_DEPTH_BEGIN rank={rank} n={n} r={r} depth={}", layers.len()-1);
                            }
                            if !bfs.advance().unwrap() { break; }
                            if trace_depths {
                                eprintln!("GRAPH_DEPTH_END rank={rank} n={n} r={r} depth={}", layers.len()-1);
                            }
                        }
                        let stats = bfs.batch_graph_stats().unwrap().unwrap();
                        assert!(stats["full_windows"].as_u64().unwrap() > 0);
                        eprintln!("GRAPH_WINDOW_ORACLE rank={rank} n={n} r={r} stats={stats}");
                        layers
                    }).unwrap_or_else(|_| std::process::abort())
                })).collect();
                let mut actual = vec![Vec::new();expected.len()];
                for worker in workers {
                    let local = worker.join().unwrap();
                    assert_eq!(local.len(),actual.len());
                    for (dst,src) in actual.iter_mut().zip(local) { dst.extend(src); }
                }
                for layer in &mut actual { layer.sort(); }
                for layer in &mut expected { layer.sort(); }
                assert_eq!(actual,expected,"graph window n={n} r={r} prededup={prededup} reversed={reversed}");
            }
        }
    }
}

fn multiset_oracle(worlds: &[u32], include_rank: bool) {
    for &world in worlds {
        for n in [5, 7] {
            let mut owners = vec![ReferenceOwner::Native(OwnerBackend::CubSortMerge), ReferenceOwner::CucoIndexed];
            if include_rank { owners.push(ReferenceOwner::CucoRank); }
            for owner in owners {
                for prededup in [false, true] {
                    let mut expected = LrxMultiset::new(n,4).unwrap().exact_layers(256).unwrap();
                    let mut id = [0u8;128];
                    assert_eq!(unsafe { mgbfs_cuda::ffi::mgbfs_nccl_unique_id(id.as_mut_ptr().cast()) },0);
                    let workers: Vec<_> = (0..world).map(|rank| std::thread::spawn(move || {
                        std::panic::catch_unwind(|| {
                            let graph = LrxMultiset::new(n,4).unwrap();
                            let cfg = DistributedConfig {
                                rank, world, logical_owner_to_rank: if world == 1 { vec![0, 0] } else { (0..world).rev().collect() },
                                transport: mgbfs_core::config::ReferenceTransport::HostSizedNccl,
                                batch: 2, layer_capacity: 256, state_ring_capacity: 512,
                                buckets: world*4, shards: world*2, job_buckets: 2,
                                bucket_capacity: 256, prededup, generation_variant: 5,
                                untouched_vram_reserve: 1<<30,
                            };
                            let pool = matches!(owner, ReferenceOwner::CucoIndexed | ReferenceOwner::CucoRank).then_some(64<<20);
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
