//! Versioned, sequential jobs for one resident rank group.
use mgbfs_core::Result;
use std::collections::BTreeMap;

pub struct Job {
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}
impl Job {
    pub fn parse(bytes: &[u8], sequence: u64) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if v["schema"] != 1 || v["sequence"].as_u64() != Some(sequence) {
            return Err("SESSION_JOB_SEQUENCE".into());
        }
        let args: Vec<String> = serde_json::from_value(v["args"].clone()).map_err(|e| e.to_string())?;
        if args.len() != 6 || !args[1].starts_with("lrx") {
            return Err("SESSION_JOB_ARGS".into());
        }
        let env: BTreeMap<String, String> = serde_json::from_value(v["env"].clone()).map_err(|e| e.to_string())?;
        if env.keys().any(|k| !k.starts_with("MGBFS_") && !k.starts_with("NCCL_")) {
            return Err("SESSION_JOB_ENV".into());
        }
        Ok(Self { args, env })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sequence_and_environment_are_not_inherited_from_unrelated_jobs() {
        let valid = br#"{"schema":1,"sequence":3,"args":["bench","lrx17r1","2","b","a","o"],"env":{"MGBFS_MEMORY_QUERY":"1"}}"#;
        assert!(Job::parse(valid, 3).is_ok());
        assert!(Job::parse(valid, 4).is_err());
        let invalid = String::from_utf8(valid.to_vec()).unwrap().replace("MGBFS_MEMORY_QUERY", "RANK");
        assert!(Job::parse(invalid.as_bytes(), 3).is_err());
    }
}
