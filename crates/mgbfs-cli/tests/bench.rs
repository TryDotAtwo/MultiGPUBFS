use std::process::Command;

fn invoke(args: &[&str]) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
        .args(args)
        .env_remove("RANK")
        .env_remove("LOCAL_RANK")
        .env_remove("WORLD_SIZE")
        .env_remove("MGBFS_BENCH_WARMUP")
        .env_remove("MGBFS_ARCHIVE_STREAM")
        .env_remove("MGBFS_BENCH_SKIP_ARCHIVE")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    serde_json::from_slice(&output.stderr).unwrap()
}

#[test]
fn bench_requires_explicit_reference_contract_and_all_paths() {
    for args in [
        vec!["bench"],
        vec!["bench", "s3", "16", "bootstrap", "archive", "results"],
        vec!["bench", "--reference", "s3", "16"],
    ] {
        let result = invoke(&args);
        assert!(result["error"].as_str().unwrap().starts_with("CLI_USAGE:"));
    }
}

#[test]
fn bench_reaches_native_launcher_or_reports_missing_build_without_fallback() {
    let result = invoke(&[
        "bench",
        "--reference",
        "s3",
        "16",
        "bootstrap",
        "archive",
        "results",
    ]);
    // Linux CUDA reaches the actual launcher's required topology check before
    // touching a device or file. Other builds must not substitute a CPU BFS.
    let expected = if cfg!(all(feature = "cuda", target_os = "linux")) {
        "ENV_RANK"
    } else {
        "CLI_BENCH_REQUIRES_LINUX_CUDA"
    };
    assert_eq!(result["error"], expected);
}

#[test]
fn manifest_bench_reaches_same_launcher_without_cpu_fallback() {
    let result = invoke(&["bench", "--manifest", "graph.json", "16",
                          "bootstrap", "archive", "results"]);
    let expected = if cfg!(all(feature = "cuda", target_os = "linux")) {
        "ENV_RANK"
    } else {
        "CLI_BENCH_REQUIRES_LINUX_CUDA"
    };
    assert_eq!(result["error"], expected);
}

#[test]
fn public_bench_cannot_disable_the_archive_output_contract() {
    for value in ["1", "invalid"] {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "mgbfs-cli-archive-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_mgbfs"));
        command.args([
                "bench",
                "--reference",
                "s3",
                "16",
                "bootstrap",
                "archive",
                "results",
            ])
            .current_dir(&directory)
            .env_remove("MGBFS_MACRO_DEPTH")
            .env("MGBFS_BENCH_WARMUP", "0")
            .env("MGBFS_BENCH_SKIP_ARCHIVE", value);
        let native = cfg!(all(feature = "cuda", target_os = "linux"));
        if native {
            // Native admission must happen collectively, not as a local CLI
            // rejection before a peer enters bootstrap. A single-rank group
            // tests that admission without requiring a second GPU.
            command.env("RANK", "0").env("LOCAL_RANK", "0").env("WORLD_SIZE", "1")
                .env("TORCHELASTIC_RUN_ID", format!("cli-archive-{nonce}"));
        }
        let output = command.output().unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        assert_eq!(output.status.code(), Some(if native { 1 } else { 2 }));
        let result: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(result["error"], "CLI_BENCH_ARCHIVE_REQUIRED");
    }
}

#[test]
fn explicit_search_only_reaches_launcher_without_weakening_default() {
    let result = invoke(&[
        "bench",
        "--reference",
        "s3",
        "16",
        "bootstrap",
        "unused-archive",
        "results",
        "--search-only",
    ]);
    let expected = if cfg!(all(feature = "cuda", target_os = "linux")) {
        "ENV_RANK"
    } else {
        "CLI_BENCH_REQUIRES_LINUX_CUDA"
    };
    assert_eq!(result["error"], expected);
}

#[test]
fn macro_depth_must_not_be_silently_ignored_by_reference_bench() {
    let native = cfg!(all(feature = "cuda", target_os = "linux"));
    // Positive depths are supported by the native single-rank macro runtime;
    // unavailable-build validation must not be imposed on that runtime.
    let values: &[&str] = if native { &["0", "invalid"] } else { &["0", "2", "10", "invalid"] };
    for value in values {
        let mut command = Command::new(env!("CARGO_BIN_EXE_mgbfs"));
        command
            .args(["bench", "--reference", "s3", "16", "bootstrap", "archive", "results"])
            .env("MGBFS_MACRO_DEPTH", value);
        if native {
            command.env("RANK", "0").env("LOCAL_RANK", "0").env("WORLD_SIZE", "1");
        }
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), Some(if native { 1 } else { 2 }));
        let result: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(result["error"], if native { "ENV_MGBFS_MACRO_DEPTH" }
            else { "CLI_BENCH_MACRO_DEPTH_UNAVAILABLE" });
    }
}

#[test]
fn single_rank_macro_reference_reaches_native_launcher() {
    let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
        .args(["bench", "--reference", "s3", "16", "bootstrap", "archive", "results"])
        .env("MGBFS_MACRO_DEPTH", "2")
        .env("WORLD_SIZE", "1")
        .output()
        .unwrap();
    let result: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    let expected = if cfg!(all(feature = "cuda", target_os = "linux")) {
        "ENV_RANK"
    } else {
        "CLI_BENCH_REQUIRES_LINUX_CUDA"
    };
    assert_eq!(result["error"], expected);
}
