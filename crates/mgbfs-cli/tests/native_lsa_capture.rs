#![cfg(all(feature = "cuda", target_os = "linux", debug_assertions))]

use std::{path::PathBuf, process::Command, time::{SystemTime, UNIX_EPOCH}};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

#[test]
fn native_only_cli_executes_requested_owner_capture() {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let fixture = Fixture(std::env::temp_dir().join(format!(
        "mgbfs-native-lsa-capture-{}-{nonce}", std::process::id())));
    std::fs::create_dir_all(&fixture.0).unwrap();
    let mut command = Command::new("timeout");
    command.args(["120", env!("CARGO_BIN_EXE_mgbfs"), "bench", "--reference", "s4", "7"])
        .arg(fixture.0.join("bootstrap"))
        .arg(fixture.0.join("archive"))
        .arg(fixture.0.join("result"));
    for (key, value) in [
        ("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
        ("TORCHELASTIC_RUN_ID", "native-lsa-capture"),
        ("MGBFS_OWNER_BACKEND", "CUB_SORT_MERGE"), ("MGBFS_PROFILE", "DENSE"),
        ("MGBFS_PRE_DEDUP", "ON"), ("MGBFS_BENCH_CAPACITY", "64"),
        ("MGBFS_FUTURE_CAPACITY", "128"), ("MGBFS_BUCKETS", "8"),
        ("MGBFS_SHARDS", "4"), ("MGBFS_JOB_BUCKETS", "2"),
        ("MGBFS_BUCKET_CAPACITY", "32"), ("MGBFS_STATE_CODEC", "matrix_u8"),
        ("MGBFS_ARCHIVE_CODEC", "matrix_u8"), ("MGBFS_ARCHIVE_ROWS", "3"),
        ("MGBFS_ARCHIVE_SLOTS", "128"), ("MGBFS_BENCH_WARMUP", "0"),
        ("MGBFS_BENCH_SKIP_ARCHIVE", "0"), ("MGBFS_ARCHIVE_STREAM", "0"),
        ("MGBFS_CAPACITY_MODE", "max_per_rank"), ("MGBFS_RANK_MAP", "0"),
        ("MGBFS_TRANSPORT_BACKEND", "NCCL_LSA"),
        ("MGBFS_TEST_OWNER_DAG_CAPTURE", "1"), ("MGBFS_MACRO_DEPTH", "1"),
    ] { command.env(key, value); }
    command.env_remove("MGBFS_TRACE_ROUTE");
    let output = command.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "real native LSA run failed: {stderr}");
    assert!(stderr.contains("MGBFS_OWNER_DAG_CAPTURE launched"),
        "requested native owner capture was silently ignored: {stderr}");
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture.0.join("result/rank-0.json")).unwrap()).unwrap();
    assert_eq!(record["status"], "COMPLETE");
    assert_eq!(record["local_layer_sizes"], serde_json::json!([1, 3, 5, 6, 5, 3, 1]));
    assert!(fixture.0.join("result/group-complete.json").exists());
}

// One physical GPU is deliberately not described as a two-rank gate. This
// exercises the actual BFS/archive failure boundary; the independent-rank
// supervisor supplies asymmetric peer cancellation coverage on remote GPUs.
#[cfg(feature = "library-owner")]
#[test]
fn archive_worker_io_failures_stop_real_bfs_without_group_complete() {
    for profile in ["DENSE", "HASH_FIRST"] {
        for (fault, marker) in [
            ("MGBFS_TEST_ARCHIVE_WORKER_WRITE_FAULT_RANK", "TEST_INJECTED_ARCHIVE_WORKER_WRITE_ERROR"),
            ("MGBFS_TEST_ARCHIVE_WORKER_SYNC_FAULT_RANK", "TEST_INJECTED_ARCHIVE_WORKER_SYNC_ERROR"),
        ] {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
            let fixture = Fixture(std::env::temp_dir().join(format!(
                "mgbfs-worker-io-{}-{nonce}", std::process::id())));
            std::fs::create_dir_all(&fixture.0).unwrap();
            let mut command = Command::new("timeout");
            command.args(["30", env!("CARGO_BIN_EXE_mgbfs"), "bench", "--reference", "s4", "7"])
                .arg(fixture.0.join("bootstrap"))
                .arg(fixture.0.join("archive"))
                .arg(fixture.0.join("result"));
            for (key, value) in [
                ("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                ("TORCHELASTIC_RUN_ID", "archive-worker-io"),
                ("MGBFS_OWNER_BACKEND", "CUCO_RANK"), ("MGBFS_PROFILE", profile),
                ("MGBFS_LIBRARY_POOL_BYTES", "67108864"), ("MGBFS_PRE_DEDUP", "ON"),
                ("MGBFS_BENCH_CAPACITY", "64"), ("MGBFS_FUTURE_CAPACITY", "128"),
                ("MGBFS_BUCKETS", "8"), ("MGBFS_SHARDS", "4"), ("MGBFS_JOB_BUCKETS", "2"),
                ("MGBFS_BUCKET_CAPACITY", "32"), ("MGBFS_STATE_CODEC", "matrix_u8"),
                ("MGBFS_ARCHIVE_CODEC", "matrix_u8"), ("MGBFS_ARCHIVE_ROWS", "3"),
                ("MGBFS_ARCHIVE_SLOTS", "128"), ("MGBFS_BENCH_WARMUP", "0"),
                ("MGBFS_BENCH_SKIP_ARCHIVE", "0"), ("MGBFS_ARCHIVE_STREAM", "0"),
                ("MGBFS_CAPACITY_MODE", "max_per_rank"), ("MGBFS_RANK_MAP", "0"),
                ("MGBFS_TRANSPORT_BACKEND", "NCCL_LSA"), ("MGBFS_MACRO_DEPTH", "1"),
            ] { command.env(key, value); }
            command.env_remove("MGBFS_TRACE_ROUTE").env_remove("MGBFS_TEST_OWNER_DAG_CAPTURE");
            for other in ["MGBFS_TEST_ARCHIVE_WORKER_WRITE_FAULT_RANK", "MGBFS_TEST_ARCHIVE_WORKER_SYNC_FAULT_RANK"] {
                command.env_remove(other);
            }
            command.env(fault, "0");
            let output = command.output().unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(output.status.code(), Some(1), "{profile}/{fault}: {stderr}");
            assert!(stderr.contains(marker), "injection was not reached: {stderr}");
            assert!(stderr.contains("MGBFS_ARCHIVE_WORKER_FATAL"), "not a worker I/O failure: {stderr}");
            assert!(!fixture.0.join("result/group-complete.json").exists());
            assert!(!fixture.0.join("result/rank-0.json").exists());
        }
    }
}
