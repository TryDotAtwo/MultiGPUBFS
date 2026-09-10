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
fn public_bench_cannot_disable_the_archive_output_contract() {
    for value in ["1", "invalid"] {
        let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
            .args([
                "bench",
                "--reference",
                "s3",
                "16",
                "bootstrap",
                "archive",
                "results",
            ])
            .env("MGBFS_BENCH_SKIP_ARCHIVE", value)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        let result: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(result["error"], "CLI_BENCH_ARCHIVE_REQUIRED");
    }
}
