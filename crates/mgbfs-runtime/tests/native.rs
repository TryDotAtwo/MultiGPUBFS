#![cfg(feature = "cuda")]
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::native::{NativeBfs, NativeConfig};
#[test]
fn native_archive_roundtrip() {
    use mgbfs_runtime::{
        archive::{verify, Extent},
        pinned_archive::PinnedArchive,
    };
    use std::sync::{Arc, Mutex};
    struct Disk(Arc<Mutex<Vec<u8>>>);
    impl Extent for Disk {
        fn reserve(&mut self, n: u64) -> std::io::Result<()> {
            self.0.lock().unwrap().resize(n as usize, 0);
            Ok(())
        }
        fn write_at(&mut self, at: u64, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap()[at as usize..at as usize + bytes.len()].copy_from_slice(bytes);
            Ok(bytes.len())
        }
        fn sync(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let g = MatrixGroup::unitriangular(4, 3).unwrap();
    let cfg = NativeConfig {
        batch: 31,
        layer_capacity: 729,
        buckets: 8,
        shards: 2,
        job_buckets: 2,
        bucket_capacity: 729,
        prededup: true,
    };
    let mut archive =
        PinnedArchive::new(Disk(bytes.clone()), 100_000, 16, [0; 32], 31, 32).unwrap();
    let mut bfs = NativeBfs::new(&g, [0; 16], cfg).unwrap();
    loop {
        bfs.archive_current(&mut archive).unwrap();
        if !bfs.advance().unwrap() {
            break;
        }
    }
    archive.finish().unwrap();
    verify(&bytes.lock().unwrap()).unwrap();
}
#[test]
fn native_feedback_full_layers() {
    for m in 2..=4 {
        let g = MatrixGroup::unitriangular(4, m).unwrap();
        let oracle = g.exact_layers((m as usize).pow(6)).unwrap();
        for pre in [false, true] {
            let cfg = NativeConfig {
                batch: 31,
                layer_capacity: (m as u32).pow(6),
                buckets: 8,
                shards: 2,
                job_buckets: 2,
                bucket_capacity: (m as u32).pow(6),
                prededup: pre,
            };
            let mut bfs = NativeBfs::new(&g, [0; 16], cfg).unwrap();
            for (depth, expected) in oracle.iter().enumerate() {
                assert!(
                    bfs.frontier_extents() <= 2,
                    "contiguous frontier fragmented into tiny GEMMs"
                );
                let mut actual = bfs.snapshot().unwrap();
                actual.sort();
                assert_eq!(&actual, expected, "m={m} depth={depth} pre={pre}");
                assert_eq!(bfs.advance().unwrap(), depth + 1 < oracle.len());
            }
            assert!(bfs.snapshot().unwrap().is_empty());
        }
    }
}

#[test]
fn layer_capacity_failure_is_terminal() {
    let g = MatrixGroup::unitriangular(4, 3).unwrap();
    let cfg = NativeConfig {
        batch: 31,
        layer_capacity: 1,
        buckets: 8,
        shards: 2,
        job_buckets: 2,
        bucket_capacity: 729,
        prededup: false,
    };
    let mut bfs = NativeBfs::new(&g, [0; 16], cfg).unwrap();
    assert_eq!(bfs.advance().unwrap_err(), "NATIVE_OWNER_FATAL_16");
    assert_eq!(bfs.advance().unwrap_err(), "NATIVE_FAILED");
    assert_eq!(bfs.snapshot().unwrap_err(), "NATIVE_FAILED");
}

#[test]
#[ignore = "larger full-state oracle gate"]
fn native_large_full_layers() {
    for m in 5..=8 {
        let g = MatrixGroup::unitriangular(4, m).unwrap();
        let oracle = g.exact_layers((m as usize).pow(6)).unwrap();
        for pre in [false, true] {
            let cfg = NativeConfig {
                batch: 4096,
                layer_capacity: (m as u32).pow(6),
                buckets: 64,
                shards: 4,
                job_buckets: 16,
                bucket_capacity: (m as u32).pow(6) / 8,
                prededup: pre,
            };
            let mut bfs = NativeBfs::new(&g, 20260828u128.to_le_bytes(), cfg).unwrap();
            for (depth, expected) in oracle.iter().enumerate() {
                let mut actual = bfs.snapshot().unwrap();
                actual.sort();
                assert_eq!(&actual, expected, "m={m} depth={depth} pre={pre}");
                assert_eq!(bfs.advance().unwrap(), depth + 1 < oracle.len());
            }
            eprintln!("FULL_STATE_PASS m={m} pre={pre}");
        }
    }
}
