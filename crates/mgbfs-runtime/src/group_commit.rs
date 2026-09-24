//! Host-side group completion marker; never part of the GPU data plane.
use mgbfs_core::Result;
use sha2::{Digest, Sha256};
use std::{io::{Read, Write}, path::Path};

pub fn write_rank_result(dir: &Path, rank: u32, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true).create_new(true)
        .open(dir.join(format!("rank-{rank}.json")))
        .map_err(|e| format!("RANK_RESULT_CREATE: {e}"))?;
    file.write_all(bytes).map_err(|e| format!("RANK_RESULT_WRITE: {e}"))?;
    file.sync_all().map_err(|e| format!("RANK_RESULT_SYNC: {e}"))?;
    Ok(())
}

pub fn write_group_commit(dir: &Path, world: u32, bootstrap_digest: [u8; 32]) -> Result<()> {
    if world == 0 || world > 8 || dir.join("group-complete.json").exists() {
        return Err("GROUP_COMMIT_CONFIG_OR_EXISTS".into());
    }
    let mut hashes = Vec::with_capacity(world as usize);
    let mut commit_scope: Option<String> = None;
    for rank in 0..world {
        let mut file = std::fs::OpenOptions::new().read(true).write(true)
            .open(dir.join(format!("rank-{rank}.json")))
            .map_err(|e| format!("GROUP_RANK_RESULT_{rank}: {e}"))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| format!("GROUP_RANK_READ_{rank}: {e}"))?;
        file.sync_all().map_err(|e| format!("GROUP_RANK_SYNC_{rank}: {e}"))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("GROUP_RANK_JSON_{rank}: {e}"))?;
        if value["status"] != "COMPLETE"
            || value["rank"] != rank
            || value["world_size"] != world
            || value["bootstrap_digest"] != serde_json::json!(bootstrap_digest)
        {
            return Err(format!("GROUP_RANK_MISMATCH_{rank}"));
        }
        let scope = value["archive_commit_scope"]
            .as_str().ok_or("GROUP_COMMIT_SCOPE")?;
        if !matches!(scope, "file_fsync" | "fifo_flush" | "search_only")
            || commit_scope.as_deref().is_some_and(|previous| previous != scope)
        {
            return Err("GROUP_COMMIT_SCOPE".into());
        }
        commit_scope = Some(scope.to_owned());
        hashes.push(Sha256::digest(&bytes).to_vec());
    }
    let marker = serde_json::to_vec(&serde_json::json!({
        "schema": "mgbfs-group-run-commit-v1",
        "status": "COMPLETE",
        "world_size": world,
        "bootstrap_digest": bootstrap_digest,
        "archive_commit_scope": commit_scope,
        "rank_sha256": hashes,
    }))
    .map_err(|e| format!("GROUP_COMMIT_JSON: {e}"))?;
    let temporary = dir.join("group-complete.json.tmp");
    let final_path = dir.join("group-complete.json");
    let mut file = std::fs::OpenOptions::new()
        .write(true).create_new(true).open(&temporary)
        .map_err(|e| format!("GROUP_COMMIT_CREATE: {e}"))?;
    file.write_all(&marker).map_err(|e| format!("GROUP_COMMIT_WRITE: {e}"))?;
    file.sync_all().map_err(|e| format!("GROUP_COMMIT_SYNC: {e}"))?;
    drop(file);
    std::fs::rename(&temporary, &final_path)
        .map_err(|e| format!("GROUP_COMMIT_RENAME: {e}"))?;
    #[cfg(target_os = "linux")]
    std::fs::File::open(dir)
        .and_then(|directory| directory.sync_all())
        .map_err(|e| format!("GROUP_COMMIT_DIR_SYNC: {e}"))?;
    Ok(())
}
