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
        mgbfs_core::config::ReferenceOwner::CudfRelational,
        mgbfs_core::config::ReferenceOwner::CucoIndexed,
    ] {
        for symmetric in [false, true] {
            for hash_first in [false, true] {
                for owners in [[0, 1], [1, 0]] {
                    fixture(symmetric, hash_first, owners, library_owner);
                }
            }
        }
    }
}

fn fixture(
    symmetric: bool,
    hash_first: bool,
    owners: [u32; 2],
    library_owner: mgbfs_core::config::ReferenceOwner,
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
    let workers: Vec<_> = (0..2u32)
        .map(|rank| {
            let graph = graph.clone();
            std::thread::spawn(move || {
                // A failed rank must not leave its peer blocked in a collective.
                // The test executable is an isolated rank group owned by the runner.
                std::panic::catch_unwind(|| {
                    let cfg = DistributedConfig {
                        rank,
                        world: 2,
                        logical_owner_to_rank: owners.to_vec(),
                        batch: 1,
                        layer_capacity: 64,
                        state_ring_capacity: 64,
                        buckets: 8,
                        shards: 4,
                        job_buckets: 2,
                        bucket_capacity: 32,
                        prededup: true,
                        generation_variant: 1,
                        untouched_vram_reserve: 1 << 30,
                    };
                    let mut bfs = DistributedNativeBfs::new_library_reference_with_owner(
                        &graph,
                        seed,
                        id,
                        cfg,
                        hash_first.then_some(128),
                        64 << 20,
                        false,
                        library_owner,
                    )
                    .unwrap();
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
                    assert_eq!(owners[(key.0[3] >> 31) as usize], rank as u32);
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
