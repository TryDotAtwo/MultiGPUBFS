use mgbfs_runtime::archive::{verify, Archive, Extent, StreamExtent};

#[test]
fn stream_writer_fails_when_reader_stops_draining() {
    struct Stalled;
    impl std::io::Write for Stalled {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::WouldBlock.into())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let mut extent = StreamExtent::with_stall_timeout(
        Stalled, std::time::Duration::from_millis(20));
    extent.reserve(1).unwrap();
    let started = std::time::Instant::now();
    assert_eq!(extent.write_at(0, b"x").unwrap_err().kind(),
        std::io::ErrorKind::TimedOut);
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
}

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
#[test]
fn archive_extent_plan_charges_rank_records_and_every_frame() {
    use mgbfs_runtime::archive::ArchiveRingPlan;
    // 48 header + three 20-byte state/hash rows + three record frames,
    // four global layer frames (including empty local layers), one commit.
    assert_eq!(ArchiveRingPlan::extent_bytes(4, 3, 3, 4).unwrap(), 1004);
    assert!(ArchiveRingPlan::extent_bytes(4, 3, 0, 4).is_err());
    assert!(ArchiveRingPlan::extent_bytes(4, 3, 3, 0).is_err());
    assert!(ArchiveRingPlan::extent_bytes(4, u64::MAX, 1, 1).is_err());
}
#[test]
fn reference_archive_budget_covers_more_total_states_than_one_layer() {
    use mgbfs_runtime::archive::ArchiveRingPlan;
    // A 24-state graph may have a largest rank-local layer of only four.
    // Its run archive still needs room for up to 24 accepted records.
    assert_eq!(ArchiveRingPlan::reference_extent_bytes(4, 24, 4).unwrap(), 6016);
}
#[cfg(target_os = "linux")]
#[test]
fn fifo_open_reports_missing_consumer_within_its_deadline() {
    use mgbfs_runtime::archive::create_archive_extent_with_timeout;
    use std::{ffi::CString, os::unix::ffi::OsStrExt, time::{Duration, Instant}};
    extern "C" { fn mkfifo(path: *const std::ffi::c_char, mode: u32) -> i32; }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("mgbfs-archive-fifo-{}-{nonce}", std::process::id()));
    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { mkfifo(name.as_ptr(), 0o600) }, 0);
    let started = Instant::now();
    let result = create_archive_extent_with_timeout(&path, true, Duration::from_millis(50));
    assert_eq!(result.err().unwrap().kind(), std::io::ErrorKind::TimedOut);
    assert!(started.elapsed() < Duration::from_secs(1));
    std::fs::remove_file(path).unwrap();
}
#[cfg(target_os = "linux")]
#[test]
fn fifo_open_streams_to_the_archive_consumer() {
    use mgbfs_runtime::archive::create_archive_extent_with_timeout;
    use std::{ffi::CString, io::Read, os::unix::ffi::OsStrExt, time::Duration};
    extern "C" { fn mkfifo(path: *const std::ffi::c_char, mode: u32) -> i32; }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("mgbfs-archive-reader-{}-{nonce}", std::process::id()));
    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { mkfifo(name.as_ptr(), 0o600) }, 0);
    let reader_path = path.clone();
    let reader = std::thread::spawn(move || {
        let mut file = std::fs::File::open(reader_path).unwrap();
        let mut bytes = [0; 3];
        file.read_exact(&mut bytes).unwrap();
        bytes
    });
    let mut writer = create_archive_extent_with_timeout(&path, true, Duration::from_secs(2)).unwrap();
    writer.reserve(3).unwrap();
    assert_eq!(writer.write_at(0, b"abc").unwrap(), 3);
    writer.sync().unwrap();
    drop(writer);
    assert_eq!(reader.join().unwrap(), *b"abc");
    std::fs::remove_file(path).unwrap();
}
#[cfg(target_os = "linux")]
#[test]
fn connected_fifo_reader_that_never_drains_times_out() {
    use mgbfs_runtime::archive::create_archive_extent_with_timeout;
    use std::{ffi::CString, os::unix::ffi::OsStrExt, sync::mpsc,
              time::{Duration, Instant}};
    extern "C" { fn mkfifo(path: *const std::ffi::c_char, mode: u32) -> i32; }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!(
        "mgbfs-archive-stalled-{}-{nonce}", std::process::id()));
    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { mkfifo(name.as_ptr(), 0o600) }, 0);
    let reader_path = path.clone();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let _file = std::fs::File::open(reader_path).unwrap();
        ready_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    let mut writer = create_archive_extent_with_timeout(&path, true,
        Duration::from_millis(50)).unwrap();
    ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let payload = vec![1u8; 8 * 1024 * 1024];
    writer.reserve(payload.len() as u64).unwrap();
    let started = Instant::now();
    let error = writer.write_at(0, &payload).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    assert!(started.elapsed() < Duration::from_secs(1));
    drop(writer);
    release_tx.send(()).unwrap();
    reader.join().unwrap();
    std::fs::remove_file(path).unwrap();
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
