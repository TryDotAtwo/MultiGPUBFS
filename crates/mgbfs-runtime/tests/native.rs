#![cfg(feature = "cuda")]
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::native::{NativeBfs, NativeConfig};
#[test]
fn impossible_reserve_fails_preflight() {
    let g = MatrixGroup::unitriangular(4, 2).unwrap();
    let cfg = NativeConfig {
        batch: 31,
        layer_capacity: 64,
        buckets: 8,
        shards: 2,
        job_buckets: 2,
        bucket_capacity: 64,
        prededup: true,
    };
    let error = NativeBfs::new_with_reserve(&g, [0; 16], cfg, u64::MAX)
        .err()
        .expect("must reject impossible reserve");
    assert!(error.starts_with("VRAM_PREFLIGHT"), "{error}");
}

#[test]
fn generation_variants_match_assembled_feedback() {
    let g = MatrixGroup::unitriangular(4, 3).unwrap();
    let oracle = g.exact_layers(729).unwrap();
    let cfg = NativeConfig {
        batch: 31,
        layer_capacity: 729,
        buckets: 8,
        shards: 2,
        job_buckets: 2,
        bucket_capacity: 729,
        prededup: true,
    };
    for variant in 1..=4 {
        let mut bfs = NativeBfs::new_with_generation(&g, [0; 16], cfg, variant).unwrap();
        for (depth, expected) in oracle.iter().enumerate() {
            let mut actual = bfs.snapshot().unwrap();
            actual.sort();
            assert_eq!(&actual, expected, "generation={variant} depth={depth}");
            assert_eq!(bfs.advance().unwrap(), depth + 1 < oracle.len());
        }
    }
    assert!(NativeBfs::new_with_generation(&g, [0; 16], cfg, 99).is_err());
}
#[test]
fn native_archive_roundtrip() {
    archive_roundtrip(4, 3, 729, false);
}
#[test]
fn archive_overlap_survives_blocked_disk_and_ring_wrap() {
    archive_roundtrip(4, 4, 1536, true);
    archive_roundtrip(3, 3, 16, true);
}
#[test]
fn asynchronous_archive_disk_error_is_not_complete() {
    use mgbfs_runtime::{archive::Extent, pinned_archive::PinnedArchive};
    struct BrokenDisk;
    impl Extent for BrokenDisk {
        fn reserve(&mut self, _: u64) -> std::io::Result<()> {
            Ok(())
        }
        fn write_at(&mut self, at: u64, bytes: &[u8]) -> std::io::Result<usize> {
            if at >= 48 {
                return Err(std::io::Error::from_raw_os_error(28));
            }
            Ok(bytes.len())
        }
        fn sync(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let g = MatrixGroup::unitriangular(4, 2).unwrap();
    let cfg = NativeConfig {
        batch: 31,
        layer_capacity: 64,
        buckets: 8,
        shards: 2,
        job_buckets: 2,
        bucket_capacity: 64,
        prededup: true,
    };
    let mut archive = PinnedArchive::new(BrokenDisk, 100_000, 16, [0; 32], 31, 32).unwrap();
    let mut bfs = NativeBfs::new(&g, [0; 16], cfg).unwrap();
    // Either the worker fails before Layer submission or finish observes it.
    let _ = bfs.archive_current(&mut archive);
    drop(bfs);
    assert!(archive.finish().is_err());
}
fn archive_roundtrip(n: usize, modulus: u16, capacity: u32, block_disk: bool) {
    use mgbfs_runtime::{
        archive::{verify, Extent},
        pinned_archive::PinnedArchive,
    };
    use std::sync::{Arc, Condvar, Mutex};
    struct Release(Arc<(Mutex<bool>, Condvar)>);
    impl Drop for Release {
        fn drop(&mut self) {
            *self.0 .0.lock().unwrap() = true;
            self.0 .1.notify_all();
        }
    }
    struct Disk(Arc<Mutex<Vec<u8>>>, Arc<(Mutex<bool>, Condvar)>);
    impl Extent for Disk {
        fn reserve(&mut self, n: u64) -> std::io::Result<()> {
            self.0.lock().unwrap().resize(n as usize, 0);
            Ok(())
        }
        fn write_at(&mut self, at: u64, bytes: &[u8]) -> std::io::Result<usize> {
            if at >= 48 {
                let (lock, wake) = &*self.1;
                let mut allowed = lock.lock().unwrap();
                while !*allowed {
                    allowed = wake.wait(allowed).unwrap();
                }
            }
            self.0.lock().unwrap()[at as usize..at as usize + bytes.len()].copy_from_slice(bytes);
            Ok(bytes.len())
        }
        fn sync(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let g = MatrixGroup::unitriangular(n, modulus).unwrap();
    let width = g.start.len();
    let cfg = NativeConfig {
        batch: 31,
        layer_capacity: capacity,
        buckets: 8,
        shards: 2,
        job_buckets: 2,
        bucket_capacity: capacity,
        prededup: true,
    };
    let gate = Arc::new((Mutex::new(!block_disk), Condvar::new()));
    let mut archive = PinnedArchive::new(
        Disk(bytes.clone(), gate.clone()),
        500_000,
        width,
        [0; 32],
        31,
        512,
    )
    .unwrap();
    // Declared after archive so unwinding releases the worker before archive Drop joins.
    let release = Release(gate);
    let mut bfs = NativeBfs::new(&g, [0; 16], cfg).unwrap();
    loop {
        bfs.archive_current(&mut archive).unwrap();
        if !bfs.advance().unwrap() {
            break;
        }
    }
    if block_disk {
        assert!(
            bytes.lock().unwrap()[48..].iter().all(|&b| b == 0),
            "search completed while all record writes were blocked"
        );
    }
    drop(release);
    archive.finish().unwrap();
    let data = bytes.lock().unwrap();
    verify(&data).unwrap();
    let hash = mgbfs_core::hash::GemmHash::from_seed(width, [0; 16]).unwrap();
    let oracle = g
        .exact_layers(g.expected_max_unique_states as usize)
        .unwrap();
    let mut layers = vec![Vec::new(); oracle.len()];
    let mut at = 48;
    loop {
        let word = |offset| {
            u64::from_le_bytes(data[at + offset..at + offset + 8].try_into().unwrap()) as usize
        };
        let (kind, depth, count, size) = (word(8), word(16), word(24), word(32));
        if kind == 3 {
            break;
        }
        if kind == 1 {
            let payload = &data[at + 80..at + 80 + size];
            for row in 0..count {
                let state = &payload[row * width..(row + 1) * width];
                assert_eq!(
                    hash.hash(state).unwrap().to_le_bytes(),
                    payload[count * width + row * 16..count * width + (row + 1) * 16]
                );
                layers[depth].push(state.to_vec());
            }
        }
        at += 112 + size;
    }
    for layer in &mut layers {
        layer.sort();
    }
    assert_eq!(layers, oracle);
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
fn padded_states_and_ping_pong_slot_reuse() {
    for (n, m) in [(2, 7), (3, 3), (4, 2)] {
        let mut g = MatrixGroup::unitriangular(n, m).unwrap();
        g.start = g.successor(&g.start, 0).unwrap();
        let capacity = g.expected_max_unique_states as u32;
        let oracle = g.exact_layers(capacity as usize).unwrap();
        for (batch, seed) in [(1, 0u128), (2, 1), (7, 20260828)] {
            let cfg = NativeConfig {
                batch,
                layer_capacity: capacity,
                buckets: 8,
                shards: 2,
                job_buckets: 2,
                bucket_capacity: capacity,
                prededup: true,
            };
            let mut bfs = NativeBfs::new(&g, seed.to_le_bytes(), cfg).unwrap();
            for (depth, expected) in oracle.iter().enumerate() {
                let mut actual = bfs.snapshot().unwrap();
                actual.sort();
                assert_eq!(&actual, expected, "n={n} m={m} batch={batch} depth={depth}");
                assert_eq!(bfs.advance().unwrap(), depth + 1 < oracle.len());
            }
        }
    }
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

#[test]
#[ignore = "local optimization probe, not archived A/B"]
fn native_timing_probe() {
    let m = 16u16;
    let g = MatrixGroup::unitriangular(4, m).unwrap();
    let variant = std::env::var("MGBFS_PROBE_GENERATION")
        .map(|v| v.parse().unwrap())
        .unwrap_or(0);
    let cfg = NativeConfig {
        batch: std::env::var("MGBFS_PROBE_BATCH")
            .map(|v| v.parse().unwrap())
            .unwrap_or(65536),
        layer_capacity: (m as u32).pow(6),
        buckets: 256,
        shards: 16,
        job_buckets: 16,
        bucket_capacity: (m as u32).pow(6) / 128 + 256,
        prededup: true,
    };
    for iteration in 0..2 {
        let mut bfs =
            NativeBfs::new_with_generation(&g, 20260828u128.to_le_bytes(), cfg, variant).unwrap();
        let start = std::time::Instant::now();
        let mut total = 1u64;
        while bfs.advance().unwrap() {
            total += u64::from(bfs.frontier_len());
        }
        assert_eq!(total, g.expected_max_unique_states);
        eprintln!(
            "LOCAL_UNARCHIVED_PROBE iteration={iteration} generation={variant} batch={} seconds={} requested_bytes={}",
            cfg.batch,
            start.elapsed().as_secs_f64(),
            bfs.requested_device_bytes()
        );
    }
}
