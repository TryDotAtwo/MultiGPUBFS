#![cfg(all(feature = "cuda", target_os = "linux"))]

use mgbfs_runtime::reference_bench;
use std::{ffi::OsString, path::PathBuf};

static ENVIRONMENT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Environment(Vec<(&'static str, Option<OsString>)>);
impl Environment {
    fn set(values: &[(&'static str, &str)]) -> Self {
        let previous = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect();
        for (key, value) in values {
            std::env::set_var(key, value);
        }
        Self(previous)
    }
}
impl Drop for Environment {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "mgbfs-macro-publication-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn run(&self) -> mgbfs_core::Result<()> {
        reference_bench::run(vec![
            "mgbfs-test".into(),
            "u3m2".into(),
            "3".into(),
            self.0.join("bootstrap").display().to_string(),
            self.0.join("archive").display().to_string(),
            self.0.join("result").display().to_string(),
        ])
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn macro_environment() -> Environment {
    Environment::set(&[
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("MGBFS_MACRO_DEPTH", "2"),
        ("MGBFS_PROFILE", "DENSE"),
        ("MGBFS_OWNER_BACKEND", "CUB_SORT_MERGE"),
        ("MGBFS_PRE_DEDUP", "ON"),
        ("MGBFS_STATE_CODEC", "matrix_u8"),
        ("MGBFS_ARCHIVE_CODEC", "matrix_u8"),
        ("MGBFS_BENCH_CAPACITY", "8"),
        ("MGBFS_FUTURE_CAPACITY", "512"),
        ("MGBFS_ARCHIVE_ROWS", "3"),
        ("MGBFS_ARCHIVE_SLOTS", "64"),
        ("MGBFS_BENCH_WARMUP", "0"),
        ("MGBFS_ARCHIVE_STREAM", "0"),
        ("MGBFS_BENCH_SKIP_ARCHIVE", "0"),
        ("MGBFS_HASH_SEED_HEX", "00000000000000000000000000000001"),
    ])
}

#[test]
fn macro_runtime_publishes_group_commit_without_overwriting_prior_results() {
    let _lock = ENVIRONMENT_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _environment = macro_environment();
    let fixture = Fixture::new();
    fixture.run().unwrap();
    let output = fixture.0.join("result");
    let marker_path = output.join("group-complete.json");
    assert!(
        marker_path.is_file(),
        "successful macro runtime must publish group RunCommit"
    );
    let rank_bytes = std::fs::read(output.join("rank-0.json")).unwrap();
    let marker_bytes = std::fs::read(&marker_path).unwrap();
    let rank: serde_json::Value = serde_json::from_slice(&rank_bytes).unwrap();
    let marker: serde_json::Value = serde_json::from_slice(&marker_bytes).unwrap();
    assert_eq!(rank["archive_commit_scope"], "file_fsync");
    assert_eq!(marker["archive_commit_scope"], "file_fsync");
    assert_eq!(marker["bootstrap_digest"], rank["bootstrap_digest"]);
    assert_eq!(
        rank["local_layer_sizes"],
        serde_json::json!([1, 2, 2, 2, 1])
    );
    std::env::set_var("MGBFS_BENCH_SKIP_ARCHIVE", "1");
    assert!(
        fixture.run().is_err(),
        "a repeated target must not overwrite its rank result"
    );
    assert_eq!(
        std::fs::read(output.join("rank-0.json")).unwrap(),
        rank_bytes
    );
    assert_eq!(std::fs::read(marker_path).unwrap(), marker_bytes);
}

#[test]
fn macro_archive_finalize_failure_never_publishes_complete() {
    let _lock = ENVIRONMENT_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _environment = macro_environment();
    let _fault = Environment::set(&[("MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK", "0")]);
    let fixture = Fixture::new();
    assert!(
        fixture.run().is_err(),
        "macro finalize fault must propagate before publication"
    );
    assert!(!fixture.0.join("result/group-complete.json").exists());
    assert!(!fixture.0.join("result/rank-0.json").exists());
}

#[test]
fn macro_warmup_releases_ephemeral_archive_without_durable_claim() {
    let _lock = ENVIRONMENT_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _environment = macro_environment();
    let _warmup = Environment::set(&[("MGBFS_BENCH_WARMUP", "1")]);
    let fixture = Fixture::new();
    fixture.run().unwrap();
    assert!(fixture.0.join("result/group-complete.json").is_file());
    assert!(!fixture.0.join("result.warmup/group-complete.json").exists());
    assert!(
        !fixture.0.join("archive.warmup-rank-0.mgbfsar1").exists(),
        "warmup archive must not consume capacity during measured search"
    );
    let rank: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture.0.join("result.warmup/rank-0.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(rank["archive_commit_scope"], "warmup_ephemeral");
    assert_eq!(rank["output_contract"], "warmup_layer_counts");
    assert!(rank["durable_run_commit_seconds"].is_null());
}

#[test]
fn macro_invalid_warmup_is_rejected_before_output() {
    let _lock = ENVIRONMENT_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _environment = macro_environment();
    let _invalid = Environment::set(&[("MGBFS_BENCH_WARMUP", "invalid")]);
    let fixture = Fixture::new();
    assert!(
        fixture.run().is_err(),
        "macro dispatch must not silently default an invalid config"
    );
    assert!(!fixture.0.join("result").exists());
    assert!(!fixture.0.join("archive-rank-0.mgbfsar1").exists());
}
