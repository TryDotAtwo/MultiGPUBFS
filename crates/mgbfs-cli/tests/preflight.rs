use mgbfs_core::config::RunConfigV1;
use std::{fs, process::Command};

#[test]
fn offline_preflight_validates_config_without_claiming_hardware_readiness() {
    let path = std::env::temp_dir().join(format!("mgbfs-preflight-{}.json", std::process::id()));
    let mut config = RunConfigV1::fixture(3).unwrap();
    fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_mgbfs"))
            .args(["preflight", "--offline"])
            .arg(&path)
            .output()
            .unwrap()
    };
    let output = run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["status"], "CONFIG_VALIDATED");
    assert_eq!(result["hardware_ready"], false);
    assert_eq!(result["scope"], "offline_config_only");
    assert_eq!(result["config_digest"].as_str().unwrap().len(), 64);
    config.capacities.route_slot_records = 1;
    fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let output = run();
    assert_eq!(output.status.code(), Some(1));
    let result: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(result["error"], "ROUTE_SLOT_CAPACITY");
    fs::write(&path, b"{malformed").unwrap();
    let output = run();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("CONFIG_PARSE"));
    fs::remove_file(path).unwrap();
}

#[test]
fn offline_mode_must_be_explicit() {
    let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
        .args(["preflight", "config.json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(!output.stdout.windows(16).any(|x| x == b"CONFIG_VALIDATED"));
}
