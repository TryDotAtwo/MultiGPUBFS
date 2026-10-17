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

/// Whether a rejected read-only plan leaves a healthy resident rank reusable.
pub fn expected_plan_rejection(_command: Option<&str>, _error: &str, _native_failed: bool) -> bool {
    false
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
    #[test]
    fn expected_resource_plan_refusals_keep_a_healthy_session() {
        for command in ["graph-plan", "graph-local-plan"] {
            for error in ["REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION", "REQUESTED_CAPACITY_EXCEEDS_ADMISSION", "GEMM_BATCH_EXCEEDS_ADMISSION"] {
                assert!(expected_plan_rejection(Some(command), error, false), "{command}: {error}");
            }
        }
    }
    #[test]
    fn capacity_error_from_bfs_or_nonplan_command_is_fatal() {
        for command in [None, Some("graph-rank"), Some("bench"), Some("inventory"), Some("release-memory"), Some("graph-plan-extra")] {
            assert!(!expected_plan_rejection(command, "REQUESTED_CAPACITY_EXCEEDS_ADMISSION", false));
        }
    }
    #[test]
    fn real_native_fault_always_invalidates_session_even_for_a_plan() {
        assert!(!expected_plan_rejection(Some("graph-plan"), "REQUESTED_CAPACITY_EXCEEDS_ADMISSION", true));
        assert!(!expected_plan_rejection(Some("graph-local-plan"), "GENERIC_CUDA_999", false));
    }
    #[test]
    fn unknown_or_decorated_plan_errors_are_not_recoverable() {
        for error in ["", "unknown", "GENERIC_CUDA_700", "GENERIC_CUDA_999", "REQUESTED_CAPACITY_EXCEEDS_ADMISSION later CUDA failure", "prefix REQUESTED_CAPACITY_EXCEEDS_ADMISSION", "SESSION_JOB_SEQUENCE"] {
            assert!(!expected_plan_rejection(Some("graph-plan"), error, false));
        }
    }

}
