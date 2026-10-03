use std::process::Command;

#[test]
fn run_requires_config_and_all_output_paths() {
    for args in [vec!["run"], vec!["run", "config.json", "bootstrap", "archive"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_mgbfs")).args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
        assert!(error["error"].as_str().unwrap().starts_with("CLI_USAGE: mgbfs run"));
    }
}

#[test]
fn run_never_uses_cpu_or_benchmark_fallback() {
    let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
        .args(["run", "missing-config.json", "bootstrap", "archive", "output"])
        .env_remove("RANK").env_remove("LOCAL_RANK").env_remove("WORLD_SIZE")
        .output().unwrap();
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    assert_eq!(error["error"], "ENV_RANK");
    #[cfg(not(all(feature = "cuda", target_os = "linux")))]
    assert_eq!(error["error"], "CLI_RUN_REQUIRES_LINUX_CUDA");
    assert!(!output.status.success());
}
