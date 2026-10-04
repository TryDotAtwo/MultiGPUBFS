//! Rank process stays alive across jobs; failed jobs retain honest exit codes.
use mgbfs_core::Result;
use std::{path::Path, io::Write, time::Duration};

pub fn run(root: &Path) -> Result<()> {
    let rank: u32 = std::env::var("RANK").map_err(|e| e.to_string())?.parse().map_err(|_| "SESSION_RANK")?;
    crate::distributed_native::enable_session_cache();
    println!("MGBFS_SESSION_READY rank={rank} pid={}", std::process::id());
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    let mut previous = std::collections::BTreeSet::new();
    for sequence in 0u64.. {
        let path = root.join(format!("job-{sequence:08}.json"));
        while !path.exists() {
            if root.join("shutdown").exists() { return Ok(()); }
            std::thread::sleep(Duration::from_millis(1));
        }
        let job = crate::session_protocol::Job::parse(&std::fs::read(path).map_err(|e| e.to_string())?, sequence)?;
        for key in &previous { std::env::remove_var(key); }
        previous.clear();
        for (key, value) in job.env { std::env::set_var(&key, value); previous.insert(key); }
        println!("MGBFS_SESSION_BEGIN sequence={sequence} rank={rank}");
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        let outcome = crate::reference_bench::run(job.args);
        let code = if outcome.is_ok() { 0 } else { 1 };
        if let Err(error) = &outcome { println!("{}", serde_json::json!({"status":"ERROR", "rank":rank, "error":error})); }
        // Query-only construction intentionally returns MEMORY_QUERY_DONE.
        // All other failures terminate the rank group: never reuse an aborted
        // communicator or partially initialized GPU state after a failed job.
        let query = outcome.as_ref().err().is_some_and(|e| e == "MEMORY_QUERY_DONE");
        println!("MGBFS_SESSION_CACHE sequence={sequence} rank={rank} {}", crate::session_cache::stats());
        println!("MGBFS_SESSION_DONE sequence={sequence} rank={rank} code={code}");
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        if code != 0 && !query { return outcome; }
    }
    Ok(())
}
