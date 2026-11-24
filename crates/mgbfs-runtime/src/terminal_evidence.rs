//! Small exact witnesses for the two final layers of a completed traversal.
use mgbfs_core::Result;
use serde::Serialize;

// Global across all ranks. Leaves room for both arms and the ordinary samples
// in the profile registry's 4 MiB serialized evidence object.
const MAX_BYTES: u64 = 768 * 1024;
const ENVELOPE_BYTES: u64 = 512;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) const FILE: &str = "terminal_states.json";
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) const TEMP_FILE: &str = "terminal_states.json.tmp";

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) fn requested_depth() -> Result<Option<u32>> {
    match std::env::var("MGBFS_PROFILE_TERMINAL_EVIDENCE_DEPTH") {
        Ok(depth) => depth
            .parse::<u32>()
            .map(Some)
            .map_err(|_| "PROFILE_TERMINAL_EVIDENCE_DEPTH".to_owned()),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("PROFILE_TERMINAL_EVIDENCE_DEPTH {error}")),
    }
}

pub(crate) fn fits(width: u32, current_rows: u64, previous_rows: u64) -> bool {
    // Signed i64 takes at most 20 characters, plus a comma. Allow brackets
    // and the row separator independently of the exact coordinate values.
    let row_bytes = 21 * u64::from(width) + 3;
    current_rows
        .checked_add(previous_rows)
        .and_then(|rows| rows.checked_mul(row_bytes))
        .is_some_and(|bytes| bytes <= MAX_BYTES - ENVELOPE_BYTES)
}

#[derive(Serialize)]
pub(crate) struct TerminalEvidence<'a> {
    pub schema: u32,
    pub graph_digest: &'a str,
    pub depth: u32,
    pub current: Vec<Vec<i64>>,
    pub previous_small: Vec<Vec<i64>>,
}

impl TerminalEvidence<'_> {
    pub(crate) fn encode(self) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec(&self).map_err(|error| error.to_string())?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err("PROFILE_TERMINAL_EVIDENCE_SIZE".to_owned());
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permits_both_full_layers_over_the_sample_limit_under_one_global_cap() {
        assert!(fits(2, 1024, 1024));
        let layer = vec![vec![i64::MIN, i64::MAX]; 1024];
        let evidence = TerminalEvidence {
            schema: 1,
            graph_digest: &"f".repeat(64),
            depth: 12,
            current: layer.clone(),
            previous_small: layer,
        };
        let bytes = evidence.encode().unwrap();
        let object: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(object["current"].as_array().unwrap().len(), 1024);
        assert_eq!(object["previous_small"][1023][0].as_i64(), Some(i64::MIN));
        assert!(!fits(512, 1024, 0));
        assert!(!fits(2, u64::MAX, 1));
    }
}
