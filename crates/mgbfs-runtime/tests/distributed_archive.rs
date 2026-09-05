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
