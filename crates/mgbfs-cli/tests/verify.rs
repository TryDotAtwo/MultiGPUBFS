use mgbfs_runtime::archive::{Archive, StreamExtent};
use std::{fs::File, process::Command};

#[test]
fn cli_verifies_committed_archive_and_rejects_truncation() {
    let path = std::env::temp_dir().join(format!("mgbfs-cli-verify-{}.bin", std::process::id()));
    let mut archive = Archive::new(
        StreamExtent::new(File::create(&path).unwrap()),
        4096,
        4,
        [0; 32],
    )
    .unwrap();
    archive.records(0, &[1, 0, 0, 1], &[[1; 4]]).unwrap();
    archive.layer_commit(0, 1).unwrap();
    archive.run_commit().unwrap();
    drop(archive);
    let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
        .arg("verify")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["status"], "VERIFIED");
    let file = File::options().write(true).open(&path).unwrap();
    file.set_len(50).unwrap();
    drop(file);
    let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
        .arg("verify")
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(output.status.code(), Some(1));
    let result: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(result["status"], "ERROR");
    assert_eq!(result["error"], "ARCHIVE_TRUNCATED");
}

#[test]
fn unknown_commands_do_not_silently_run_another_algorithm() {
    let output = Command::new(env!("CARGO_BIN_EXE_mgbfs"))
        .arg("not-a-command")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let result: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(result["error"], "CLI_COMMAND_UNAVAILABLE");
}
