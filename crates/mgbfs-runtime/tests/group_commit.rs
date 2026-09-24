use mgbfs_runtime::group_commit::{write_group_commit, write_rank_result};

#[test]
fn rank_result_write_is_durable_and_never_overwrites_existing_run() {
    let root = std::env::temp_dir().join(format!(
        "mgbfs-rank-result-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    write_rank_result(&root, 0, b"first").unwrap();
    assert!(write_rank_result(&root, 0, b"second").is_err());
    assert_eq!(std::fs::read(root.join("rank-0.json")).unwrap(), b"first");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn group_commit_requires_every_matching_rank_result() {
    let root = std::env::temp_dir().join(format!(
        "mgbfs-group-commit-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let config = [7u8; 32];
    let rank = |id| format!(
        "{{\"status\":\"COMPLETE\",\"rank\":{id},\"world_size\":2,\"archive_commit_scope\":\"file_fsync\",\"bootstrap_digest\":{}}}",
        serde_json::to_string(&config).unwrap()
    );
    std::fs::write(root.join("rank-0.json"), rank(0)).unwrap();
    assert!(write_group_commit(&root, 2, config).is_err());
    assert!(!root.join("group-complete.json").exists());
    std::fs::write(root.join("rank-1.json"), rank(0)).unwrap();
    assert!(write_group_commit(&root, 2, config).is_err());
    std::fs::write(root.join("rank-1.json"), rank(1)).unwrap();
    write_group_commit(&root, 2, config).unwrap();
    let marker: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("group-complete.json")).unwrap()).unwrap();
    assert_eq!(marker["status"], "COMPLETE");
    assert_eq!(marker["world_size"], 2);
    assert_eq!(marker["archive_commit_scope"], "file_fsync");
    assert_eq!(marker["rank_sha256"].as_array().unwrap().len(), 2);
    assert!(write_group_commit(&root, 2, config).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
