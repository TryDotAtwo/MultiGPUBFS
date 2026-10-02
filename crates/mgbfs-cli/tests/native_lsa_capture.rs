#![cfg(all(feature = "cuda", target_os = "linux", debug_assertions))]

use std::{path::PathBuf, process::Command, time::{SystemTime, UNIX_EPOCH}};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

// Independent full-state oracle: checksummed counts alone cannot detect a
// generation/layout bug which substitutes a different state at the same depth.
fn assert_s4_archive_full_states(path: &std::path::Path) {
    let oracle = mgbfs_core::matrix::MatrixGroup::symmetric_permutation_matrices(4)
        .unwrap().exact_layers(24).unwrap();
    assert_archive_full_states(path, 16, oracle);
}

fn assert_archive_full_states(path: &std::path::Path, expected_width: usize,
                              oracle: Vec<Vec<Vec<u8>>>) {
    let bytes = std::fs::read(path).unwrap();
    mgbfs_runtime::archive::verify(&bytes).unwrap();
    let word = |offset: usize| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    let width = word(8);
    assert_eq!(width, expected_width);
    let hash = mgbfs_core::hash::GemmHash::from_seed(width, 20260828u128.to_le_bytes()).unwrap();
    let mut actual = vec![std::collections::BTreeSet::new(); oracle.len()];
    let mut cursor = 48;
    loop {
        let kind = word(cursor + 8);
        let depth = word(cursor + 16);
        let rows = word(cursor + 24);
        let size = word(cursor + 32);
        if kind == 1 {
            assert!(depth < actual.len(), "unexpected archive depth");
            for row in 0..rows {
                let start = cursor + 80 + row * width;
                let hash_start = cursor + 80 + rows * width + row * 16;
                assert_eq!(&bytes[hash_start..hash_start + 16],
                    &hash.hash(&bytes[start..start + width]).unwrap().to_le_bytes(),
                    "state/hash mismatch at depth {depth}, row {row}");
                assert!(actual[depth].insert(bytes[start..start + width].to_vec()),
                    "duplicate archived state at depth {depth}");
            }
        }
        if kind == 3 { break; }
        cursor += 112 + size;
    }
    for (depth, expected) in oracle.into_iter().enumerate() {
        assert_eq!(actual[depth], expected.into_iter().collect(),
            "full-state mismatch at depth {depth}");
    }
}

#[test]
#[ignore = "requires actual CUDA/NCCL LSA hardware"]
fn manifest_nonidentity_start_runs_both_profiles_with_full_state_archive() {
    for profile in ["DENSE", "HASH_FIRST"] {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let fixture = Fixture(std::env::temp_dir().join(format!("mgbfs-manifest-gpu-{profile}-{nonce}")));
        std::fs::create_dir_all(&fixture.0).unwrap();
        let manifest = fixture.0.join("graph.json");
        std::fs::write(&manifest, r#"{"schema":1,"rows":2,"cols":2,"modulus":3,
            "start":[1,1,0,1],"generators":[[1,1,0,1],[1,2,0,1]],
            "inverse_map":[1,0],"expected_max_unique_states":3}"#).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
            .args(["bench", "--manifest"]).arg(&manifest).arg("1")
            .arg(fixture.0.join("bootstrap")).arg(fixture.0.join("archive"))
            .arg(fixture.0.join("result"))
            .env("RANK", "0").env("LOCAL_RANK", "0").env("WORLD_SIZE", "1")
            .env("TORCHELASTIC_RUN_ID", format!("manifest-{nonce}"))
            .env("MGBFS_HASH_SEED_HEX", "000000000000000000000000013527dc")
            .env("MGBFS_OWNER_BACKEND", "CUB_SORT_MERGE").env("MGBFS_PROFILE", profile)
            .env("MGBFS_HASH_FIRST_GENERATION", "SCALAR")
            .env("MGBFS_TRANSPORT_BACKEND", "NCCL_LSA").env("MGBFS_RANK_MAP", "0")
            .env("MGBFS_STATE_CODEC", "matrix_u8").env("MGBFS_ARCHIVE_CODEC", "matrix_u8")
            .env("MGBFS_BUCKETS", "8").env("MGBFS_SHARDS", "4")
            .env("MGBFS_JOB_BUCKETS", "2").env("MGBFS_BUCKET_CAPACITY", "32")
            .env("MGBFS_BENCH_CAPACITY", "64").env("MGBFS_FUTURE_CAPACITY", "128")
            .env("MGBFS_BENCH_WARMUP", "0").env("MGBFS_BENCH_SKIP_ARCHIVE", "0")
            .env("MGBFS_ARCHIVE_STREAM", "0").env("MGBFS_MACRO_DEPTH", "1")
            .env("MGBFS_ARCHIVE_ROWS", "8").env("MGBFS_ARCHIVE_SLOTS", "64")
            .env("MGBFS_PRE_DEDUP", "ON").env("NCCL_CUMEM_ENABLE", "1")
            .env_remove("MGBFS_LIBRARY_POOL_BYTES").env_remove("MGBFS_TEST_OWNER_DAG_CAPTURE")
            .output().unwrap();
        assert!(output.status.success(), "{profile}: {}", String::from_utf8_lossy(&output.stderr));
        assert!(fixture.0.join("result/group-complete.json").exists());
        assert_archive_full_states(&fixture.0.join("archive-rank-0.mgbfsar1"), 4,
            vec![vec![vec![1,1,0,1]], vec![vec![1,0,0,1], vec![1,2,0,1]]]);
    }
}

#[test]
fn full_state_gate_rejects_checksummed_wrong_states_with_correct_counts() {
    use mgbfs_runtime::archive::{Archive, FileExtent};
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let fixture = Fixture(std::env::temp_dir().join(format!("mgbfs-oracle-mutation-{nonce}")));
    std::fs::create_dir_all(&fixture.0).unwrap();
    let path = fixture.0.join("wrong.mgbfsar1");
    let graph = mgbfs_core::matrix::MatrixGroup::symmetric_permutation_matrices(4).unwrap();
    let mut layers = graph.exact_layers(24).unwrap();
    // Swap two distinct states across depths: preserve every layer count and
    // the complete unique set, but falsify BFS distance. Recompute checksums
    // through the real archive writer so structural verification still passes.
    let old_start = layers[0][0].clone();
    layers[0][0] = layers[1][0].clone();
    layers[1][0] = old_start;
    let mut archive = Archive::new(FileExtent::create_new(&path).unwrap(), 16384, 16, [0; 32]).unwrap();
    let hash = mgbfs_core::hash::GemmHash::from_seed(16, 20260828u128.to_le_bytes()).unwrap();
    for (depth, states) in layers.iter().enumerate() {
        let flat: Vec<u8> = states.iter().flatten().copied().collect();
        let hashes: Vec<_> = states.iter().map(|state| hash.hash(state).unwrap().0).collect();
        archive.records(depth as u64, &flat, &hashes).unwrap();
        archive.layer_commit(depth as u64, states.len() as u64).unwrap();
    }
    archive.run_commit().unwrap();
    drop(archive);
    mgbfs_runtime::archive::verify(&std::fs::read(&path).unwrap()).unwrap();
    assert!(std::panic::catch_unwind(|| assert_s4_archive_full_states(&path)).is_err(),
        "full-state gate accepted a checksummed wrong depth assignment");
}

#[test]
#[ignore = "requires CUDA build; exercises two-process pre-communicator control only"]
fn peer_malformed_manifest_cancels_valid_rank_before_archive_admission() {
    use std::{process::Stdio, time::{Duration, Instant}};
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let fixture = Fixture(std::env::temp_dir().join(format!("mgbfs-manifest-peer-failure-{nonce}")));
    std::fs::create_dir_all(&fixture.0).unwrap();
    let good = fixture.0.join("good.json");
    let bad = fixture.0.join("bad.json");
    std::fs::write(&good, r#"{"schema":1,"rows":2,"cols":2,"modulus":3,
        "start":[1,1,0,1],"generators":[[1,1,0,1],[1,2,0,1]],
        "inverse_map":[1,0],"expected_max_unique_states":3}"#).unwrap();
    std::fs::write(&bad, "{truncated").unwrap();
    let mut children = Vec::new();
    for rank in 0..2 {
        let log = std::fs::File::create(fixture.0.join(format!("rank-{rank}.log"))).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_mgbfs"));
        command.args(["bench", "--manifest"]).arg(if rank == 0 { &good } else { &bad })
            .arg("1").arg(fixture.0.join("bootstrap")).arg(fixture.0.join("archive"))
            .arg(fixture.0.join("result"))
            .env("RANK", rank.to_string()).env("LOCAL_RANK", rank.to_string()).env("WORLD_SIZE", "2")
            .env("TORCHELASTIC_RUN_ID", format!("manifest-peer-{nonce}"))
            .env("MGBFS_PROFILE", "DENSE").env("MGBFS_OWNER_BACKEND", "CUB_SORT_MERGE")
            .env("MGBFS_TRANSPORT_BACKEND", "NCCL_LSA").env("MGBFS_RANK_MAP", "0,1")
            .env("MGBFS_HASH_FIRST_GENERATION", "SCALAR")
            .env("MGBFS_STATE_CODEC", "matrix_u8").env("MGBFS_ARCHIVE_CODEC", "matrix_u8")
            .env("MGBFS_BUCKETS", "8").env("MGBFS_SHARDS", "4").env("MGBFS_JOB_BUCKETS", "2")
            .env("MGBFS_BUCKET_CAPACITY", "32").env("MGBFS_BENCH_CAPACITY", "64")
            .env("MGBFS_FUTURE_CAPACITY", "128").env("MGBFS_BENCH_WARMUP", "0")
            .env("MGBFS_BENCH_SKIP_ARCHIVE", "0").env("MGBFS_ARCHIVE_STREAM", "0")
            .env("MGBFS_MACRO_DEPTH", "1").env("MGBFS_PRE_DEDUP", "ON")
            .env_remove("MGBFS_LIBRARY_POOL_BYTES").env_remove("MGBFS_TEST_OWNER_DAG_CAPTURE")
            .stdout(Stdio::null()).stderr(Stdio::from(log));
        match command.spawn() {
            Ok(child) => children.push(child),
            Err(error) => {
                for child in &mut children { let _ = child.kill(); let _ = child.wait(); }
                panic!("rank spawn failed: {error}");
            }
        }
    }
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut completed = false;
    while Instant::now() < deadline {
        if children.iter_mut().all(|child| child.try_wait().unwrap().is_some()) {
            completed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if !completed {
        for child in &mut children { let _ = child.kill(); let _ = child.wait(); }
        panic!("manifest admission did not terminate both independent processes within 15s");
    }
    for child in &mut children { assert!(!child.wait().unwrap().success()); }
    let valid_log = std::fs::read_to_string(fixture.0.join("rank-0.log")).unwrap();
    let failed_log = std::fs::read_to_string(fixture.0.join("rank-1.log")).unwrap();
    assert!(valid_log.contains("REMOTE_CONFIGURATION_FATAL"), "{valid_log}");
    assert!(failed_log.contains("MATRIX_MANIFEST_PARSE"), "{failed_log}");
    assert!(!fixture.0.join("result/group-complete.json").exists());
    for rank in 0..2 {
        assert!(!fixture.0.join(format!("archive-rank-{rank}.mgbfsar1")).exists());
        assert!(!fixture.0.join(format!("result/rank-{rank}.json")).exists());
    }
}

#[test]
fn tensor_generation_hardware_admission_and_layer_counts() {
    let hardware = Command::new("nvidia-smi")
        .args(["--id=0", "--query-gpu=compute_cap", "--format=csv,noheader"])
        .output().unwrap();
    assert!(hardware.status.success());
    let supported = String::from_utf8_lossy(&hardware.stdout).trim() == "7.5";
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let fixture = Fixture(std::env::temp_dir().join(format!("mgbfs-tc-admission-{nonce}")));
    std::fs::create_dir_all(&fixture.0).unwrap();
    let output = Command::new("timeout")
        .args(["30", env!("CARGO_BIN_EXE_mgbfs"), "bench", "--reference", "s4", "7"])
        .arg(fixture.0.join("bootstrap")).arg(fixture.0.join("archive"))
        .arg(fixture.0.join("result"))
        .env("RANK", "0").env("LOCAL_RANK", "0").env("WORLD_SIZE", "1")
        .env("TORCHELASTIC_RUN_ID", "tc-admission")
        .env("MGBFS_HASH_SEED_HEX", "000000000000000000000000013527dc")
        .env("MGBFS_OWNER_BACKEND", "CUB_SORT_MERGE")
        .env("MGBFS_PROFILE", "HASH_FIRST")
        .env("MGBFS_HASH_FIRST_GENERATION", "INT_MMA_SM75")
        .env("MGBFS_TRANSPORT_BACKEND", "NCCL_LSA").env("MGBFS_RANK_MAP", "0")
        .env("MGBFS_STATE_CODEC", "matrix_u8").env("MGBFS_ARCHIVE_CODEC", "matrix_u8")
        .env("MGBFS_BUCKETS", "8").env("MGBFS_SHARDS", "4")
        .env("MGBFS_JOB_BUCKETS", "2").env("MGBFS_BUCKET_CAPACITY", "32")
        .env("MGBFS_BENCH_CAPACITY", "64").env("MGBFS_FUTURE_CAPACITY", "128")
        .env("MGBFS_BENCH_WARMUP", "0").env("MGBFS_BENCH_SKIP_ARCHIVE", "0")
        .env("MGBFS_ARCHIVE_STREAM", "0").env("MGBFS_MACRO_DEPTH", "1")
        .env_remove("MGBFS_LIBRARY_POOL_BYTES").env_remove("MGBFS_TEST_OWNER_DAG_CAPTURE")
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if supported {
        assert!(output.status.success(), "SM75 Tensor BFS failed: {stderr}");
        let record: serde_json::Value = serde_json::from_slice(
            &std::fs::read(fixture.0.join("result/rank-0.json")).unwrap()).unwrap();
        assert_eq!(record["status"], "COMPLETE");
        assert_eq!(record["hash_first_generation"], "INT_MMA_SM75");
        assert_eq!(record["local_layer_sizes"], serde_json::json!([1, 3, 5, 6, 5, 3, 1]));
        assert!(fixture.0.join("result/group-complete.json").exists());
        assert!(Command::new(env!("CARGO_BIN_EXE_mgbfs"))
            .arg("verify").arg(fixture.0.join("archive-rank-0.mgbfsar1"))
            .status().unwrap().success());
        assert_s4_archive_full_states(&fixture.0.join("archive-rank-0.mgbfsar1"));
        return;
    }
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("HASH_FIRST_TC_DEVICE_UNSUPPORTED"), "{stderr}");
    assert!(!fixture.0.join("archive-rank-0.mgbfsar1").exists(),
        "unsupported backend reserved an archive before hardware admission");
    assert!(!fixture.0.join("result/group-complete.json").exists());
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
        ("MGBFS_HASH_SEED_HEX", "000000000000000000000000013527dc"),
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
    assert_s4_archive_full_states(&fixture.0.join("archive-rank-0.mgbfsar1"));
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
