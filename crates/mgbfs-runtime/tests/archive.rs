use mgbfs_runtime::archive::{verify, Archive, Extent, StreamExtent};

#[test]
fn archive_ring_plan_checks_all_storage_before_allocating_slots() {
    use mgbfs_runtime::archive::ArchiveRingPlan;
    let p = ArchiveRingPlan::new(11, 16384, 4).unwrap();
    assert_eq!(p.slot_bytes, 442368);
    assert_eq!(p.pinned_bytes, 1769472);
    assert_eq!(p.descriptor_capacity, 10);
    for (width, rows, slots) in [
        (0, 1, 2),
        (11, 0, 2),
        (11, 1, 1),
        (usize::MAX, 1, 2),
        (11, 1, usize::MAX),
    ] {
        assert!(ArchiveRingPlan::new(width, rows, slots).is_err());
    }
}
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};

#[derive(Clone, Default)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);
impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn sequential_extent_streams_exact_frames_without_materializing_capacity() {
    let writer = SharedWriter::default();
    let readable = writer.clone();
    let mut archive = Archive::new(StreamExtent::new(writer), 4096, 4, [9; 32]).unwrap();
    archive.records(0, &[1, 0, 0, 1], &[[1, 2, 3, 4]]).unwrap();
    archive.layer_commit(0, 1).unwrap();
    archive.run_commit().unwrap();
    let bytes = readable.0.lock().unwrap();
    verify(&bytes).unwrap();
    assert!(bytes.len() < 4096);
}
#[test]
fn prepacked_records_match_reference_codec() {
    let mut reference = Archive::new(Disk::default(), 4096, 4, [7; 32]).unwrap();
    let mut packed = Archive::new(Disk::default(), 4096, 4, [7; 32]).unwrap();
    let states = [1, 0, 0, 1, 1, 1, 0, 1];
    let hashes = [[1u32, 2, 3, 4], [5, 6, 7, 8]];
    reference.records(0, &states, &hashes).unwrap();
    let mut wire = states.to_vec();
    for hash in hashes {
        for word in hash {
            wire.extend_from_slice(&word.to_le_bytes());
        }
    }
    assert!(packed.records_wire(0, 3, &wire).is_err());
    packed.records_wire(0, 2, &wire).unwrap();
    for a in [&mut reference, &mut packed] {
        a.layer_commit(0, 2).unwrap();
        a.run_commit().unwrap();
    }
    assert_eq!(reference.extent.bytes, packed.extent.bytes);
    verify(&packed.extent.bytes).unwrap();
}
#[derive(Default)]
struct Disk {
    bytes: Vec<u8>,
    writes: usize,
    short_at: Option<usize>,
    full: bool,
    sync_fail: bool,
    syncs: usize,
}

#[test]
fn stream_verifier_handles_short_reads_with_bounded_memory_and_rejects_corruption() {
    use std::io::{Cursor, Read};
    struct ShortReader(Cursor<Vec<u8>>);
    impl Read for ShortReader {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            assert!(
                output.len() <= 65536,
                "verifier requested an unbounded read"
            );
            let n = output.len().min(17);
            self.0.read(&mut output[..n])
        }
    }
    let mut a = Archive::new(Disk::default(), 300_000, 4, [0; 32]).unwrap();
    a.records(0, &vec![1; 40_000], &vec![[1; 4]; 10_000])
        .unwrap();
    a.layer_commit(0, 10_000).unwrap();
    a.run_commit().unwrap();
    mgbfs_runtime::archive::verify_reader(&mut ShortReader(Cursor::new(a.extent.bytes.clone())))
        .unwrap();
    for cut in [0, 47, 48, 128, 200_383] {
        assert!(
            mgbfs_runtime::archive::verify_reader(&mut Cursor::new(&a.extent.bytes[..cut]))
                .is_err()
        );
    }
    a.extent.bytes[150] ^= 1;
    assert_eq!(
        mgbfs_runtime::archive::verify_reader(&mut Cursor::new(&a.extent.bytes)).unwrap_err(),
        "ARCHIVE_CHECKSUM"
    );
}
#[test]
fn run_durable_syncs_only_at_completion_and_propagates_failure() {
    for fail in [false, true] {
        let mut a = Archive::new_run_durable(Disk::default(), 4096, 4, [0; 32]).unwrap();
        a.records(0, &[1, 0, 0, 1], &[[1; 4]]).unwrap();
        a.layer_commit(0, 1).unwrap();
        a.layer_commit(1, 0).unwrap();
        assert_eq!(a.extent.syncs, 0);
        assert!(!a.is_complete());
        a.extent.sync_fail = fail;
        assert_eq!(a.run_commit().is_err(), fail);
        assert_eq!(a.extent.syncs, 1);
        assert_eq!(a.timings.sync_calls, 1);
        assert_eq!(a.is_complete(), !fail);
        if !fail {
            verify(&a.extent.bytes).unwrap();
        } else {
            assert!(a.run_commit().is_err());
        }
    }
}
#[test]
fn rank_with_no_local_states_still_commits_empty_layers() {
    let mut a = Archive::new(Disk::default(), 4096, 4, [0; 32]).unwrap();
    a.layer_commit(0, 0).unwrap();
    a.layer_commit(1, 0).unwrap();
    a.run_commit().unwrap();
    verify(&a.extent.bytes).unwrap();
}
impl Extent for Disk {
    fn reserve(&mut self, n: u64) -> io::Result<()> {
        if self.full {
            return Err(io::Error::from_raw_os_error(28));
        }
        self.bytes.resize(n as usize, 0);
        Ok(())
    }
    fn write_at(&mut self, o: u64, b: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        let n = if self.short_at == Some(self.writes) {
            b.len() / 2
        } else {
            b.len()
        };
        self.bytes[o as usize..o as usize + n].copy_from_slice(&b[..n]);
        Ok(n)
    }
    fn sync(&mut self) -> io::Result<()> {
        self.syncs += 1;
        if self.sync_fail {
            Err(io::Error::from_raw_os_error(5))
        } else {
            Ok(())
        }
    }
}
#[test]
fn durable_chain_roundtrips_and_detects_corruption_and_truncation() {
    let mut a = Archive::new(Disk::default(), 4096, 4, [7; 32]).unwrap();
    a.records(0, &[1, 0, 0, 1], &[[1, 2, 3, 4]]).unwrap();
    assert!(a.layer_commit(0, 2).is_err());
    a.layer_commit(0, 1).unwrap();
    a.records(1, &[1, 1, 0, 1, 1, 2, 0, 1], &[[5; 4], [6; 4]])
        .unwrap();
    a.layer_commit(1, 2).unwrap();
    assert!(!a.is_complete());
    a.run_commit().unwrap();
    assert!(a.is_complete());
    assert!(a.records(2, &[1, 0, 0, 1], &[[9; 4]]).is_err());
    verify(&a.extent.bytes).unwrap();
    assert!(a.extent.syncs >= 1);
    let mut corrupt = a.extent.bytes.clone();
    corrupt[40] ^= 1;
    assert!(verify(&corrupt).is_err());
    assert!(verify(&a.extent.bytes[..100]).is_err());
}
#[test]
fn full_disk_short_write_and_sync_failure_never_commit() {
    assert!(Archive::new(
        Disk {
            full: true,
            ..Disk::default()
        },
        4096,
        4,
        [0; 32]
    )
    .is_err());
    let mut a = Archive::new(Disk::default(), 4096, 4, [0; 32]).unwrap();
    a.extent.short_at = Some(a.extent.writes + 1);
    assert!(a.records(0, &[1; 4], &[[1; 4]]).is_err());
    assert!(a.layer_commit(0, 0).is_err());
    assert!(a.run_commit().is_err());
    assert!(!a.is_complete());
    let mut a = Archive::new(Disk::default(), 4096, 4, [0; 32]).unwrap();
    a.records(0, &[1; 4], &[[1; 4]]).unwrap();
    a.layer_commit(0, 1).unwrap();
    a.extent.sync_fail = true;
    assert!(a.run_commit().is_err());
    assert!(!a.is_complete());
}
#[test]
fn capacity_and_record_shape_are_checked_before_writing() {
    let mut a = Archive::new(Disk::default(), 256, 4, [0; 32]).unwrap();
    let before = a.extent.writes;
    assert!(a.records(0, &[1; 3], &[[1; 4]]).is_err());
    assert_eq!(a.extent.writes, before);
    assert!(a.records(0, &[1; 128], &[[1; 4]; 32]).is_err());
    assert_eq!(a.extent.writes, before);
    assert!(!a.is_complete());
}
