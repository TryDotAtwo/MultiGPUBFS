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
