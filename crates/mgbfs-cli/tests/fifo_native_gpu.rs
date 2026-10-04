#![cfg(all(feature = "library-owner", target_os = "linux"))]
//! Real CLI/FIFO/archive gate. Small U4/F2 output is bounded by the oracle.
use std::{
    fs,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
#[path = "../../mgbfs-runtime/tests/support/route_archive.rs"]
mod route_archive;

#[test]
fn native_cli_fifo_preserves_full_layers_and_wire_budget() {
    let expected = mgbfs_core::matrix::MatrixGroup::unitriangular(4, 2)
        .unwrap()
        .exact_layers(64)
        .unwrap();
    for profile in ["DENSE", "HASH_FIRST"] {
        for fault in [
            None,
            Some("MGBFS_TEST_ARCHIVE_ADMISSION_FAULT_RANK"),
            Some("MGBFS_TEST_OWNER_HOST_FAULT_RANK"),
            Some("MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK"),
        ] {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root =
                std::env::temp_dir().join(format!("mgbfs-cli-fifo-{}-{nonce}", std::process::id()));
            fs::create_dir(&root).unwrap();
            let fifo = root.join("archive-rank-0.mgbfsar1");
            assert!(Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success());
            // Open before launch without blocking: even an admission failure must
            // allow this test to terminate. Read errors are never converted to EOF.
            let mut input = fs::OpenOptions::new()
                .read(true)
                .custom_flags(0x800)
                .open(&fifo)
                .unwrap();
            let done = Arc::new(AtomicBool::new(false));
            let stop = done.clone();
            let reader = std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    match input.read(&mut buf) {
                        Ok(0) if stop.load(Ordering::Acquire) => return bytes,
                        Ok(0) => std::thread::sleep(Duration::from_millis(1)),
                        Ok(n) => {
                            bytes.extend_from_slice(&buf[..n]);
                            assert!(bytes.len() < 1_000_000);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(e) => panic!("FIFO read: {e}"),
                    }
                }
            });
            let output = root.join("results");
            let log = root.join("cli.log");
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_mgbfs"));
            for (key, _) in std::env::vars().filter(|(key, _)| key.starts_with("MGBFS_")) {
                cmd.env_remove(key);
            }
            cmd.args(["bench", "--reference", "u4m2", "1"])
                .arg(root.join("bootstrap"))
                .arg(root.join("archive"))
                .arg(&output)
                .env("RANK", "0")
                .env("LOCAL_RANK", "0")
                .env("WORLD_SIZE", "1")
                .env("TORCHELASTIC_RUN_ID", format!("fifo-{nonce}"))
                .env("MGBFS_BENCH_WORLD_SIZE", "1")
                .env("MGBFS_BENCH_WARMUP", "0")
                .env("MGBFS_ARCHIVE_STREAM", "1")
                .env("MGBFS_PROFILE", profile)
                .env("MGBFS_OWNER_BACKEND", "CUCO_RANK")
                .env("MGBFS_TRANSPORT_BACKEND", "NCCL_LSA")
                .env("MGBFS_BENCH_CAPACITY", "64")
                .env("MGBFS_FUTURE_CAPACITY", "256")
                .env("MGBFS_SHARDS", "2")
                .env("MGBFS_BUCKETS", "8")
                .env("MGBFS_JOB_BUCKETS", "2")
                .env("MGBFS_BUCKET_CAPACITY", "32")
                .env("MGBFS_LIBRARY_POOL_BYTES", "67108864")
                .env("MGBFS_ARCHIVE_ROWS", "3")
                .env("MGBFS_ARCHIVE_SLOTS", "128")
                .env("MGBFS_ROUTE_BANKS", "3")
                .env("MGBFS_EPOCH_WINDOW", "3")
                .stdout(Stdio::from(fs::File::create(&log).unwrap()))
                .stderr(Stdio::from(
                    fs::OpenOptions::new().append(true).open(&log).unwrap(),
                ));
            if let Some(key) = fault {
                cmd.env(key, "0");
            }
            let mut child = cmd.spawn().unwrap();
            let deadline = Instant::now() + Duration::from_secs(60);
            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break status;
                }
                if Instant::now() > deadline {
                    child.kill().unwrap();
                    break child.wait().unwrap();
                }
                std::thread::sleep(Duration::from_millis(10));
            };
            done.store(true, Ordering::Release);
            let bytes = reader.join().unwrap();
            if let Some(key) = fault {
                assert!(!status.success(), "fault was ignored: {key}");
                assert!(
                    Instant::now() < deadline,
                    "fault termination timed out: {key}"
                );
                assert!(
                    !output.join("group-complete.json").exists(),
                    "false group COMPLETE: {key}"
                );
                assert!(
                    !output.join("rank-0.json").exists(),
                    "false rank COMPLETE: {key}"
                );
                let log_text = fs::read_to_string(&log).unwrap();
                assert!(
                    log_text.contains("TEST_INJECTED_"),
                    "wrong failure for {key}: {log_text}"
                );
                fs::remove_dir_all(&root).unwrap();
                continue;
            }
            assert!(status.success(), "{}", fs::read_to_string(&log).unwrap());
            route_archive::assert_layers(&bytes, &expected, 20260828u128.to_le_bytes());
            let record: serde_json::Value =
                serde_json::from_slice(&fs::read(output.join("rank-0.json")).unwrap()).unwrap();
            assert_eq!(record["disk_reserved_bytes"], 0);
            assert_eq!(record["archive_wire_limit_bytes"].as_u64(), Some(u64::MAX));
            let commit: serde_json::Value =
                serde_json::from_slice(&fs::read(output.join("group-complete.json")).unwrap())
                    .unwrap();
            assert_eq!(commit["status"], "COMPLETE");
            assert_eq!(commit["archive_commit_scope"], "fifo_flush");
            fs::remove_dir_all(&root).unwrap();
        }
    }
}
