//! Nearest-path witnesses over committed owner rows.

use mgbfs_core::Result;
use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct Receipt {
    status: &'static str,
    stop_reason: Option<&'static str>,
    pub(super) d_min: Option<u64>,
    target_index: Option<u32>,
    generator_ids: Vec<u32>,
}

impl Receipt {
    pub(crate) fn complete(distance: u64, target: u32, labels: Vec<u32>) -> Self {
        Self {
            status: "COMPLETE",
            stop_reason: None,
            d_min: Some(distance),
            target_index: Some(target),
            generator_ids: labels,
        }
    }

    pub(crate) fn unreachable() -> Self {
        Self {
            status: "UNREACHABLE",
            stop_reason: None,
            d_min: None,
            target_index: None,
            generator_ids: vec![],
        }
    }

    pub(crate) fn incomplete(reason: &'static str, d_min: Option<u64>) -> Self {
        Self {
            status: "INCOMPLETE",
            stop_reason: Some(reason),
            d_min,
            target_index: None,
            generator_ids: vec![],
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct OwnerRow {
    pub(super) owner: usize,
    pub(super) row: u64,
}

impl OwnerRow {
    pub(super) fn reply_words(self) -> [i64; 2] {
        [self.owner as i64, self.row as i64]
    }

    pub(super) fn from_reply_words([owner, row]: [i64; 2], world: usize) -> Result<Self> {
        let owner = usize::try_from(owner).map_err(|_| "NATIVE_PATH_PARENT_LINK")?;
        if owner >= world {
            return Err("NATIVE_PATH_PARENT_LINK".into());
        }
        Ok(Self { owner, row: row as u64 })
    }
}

pub(super) enum Trace<T> {
    Found(T),
    Stopped(&'static str),
}

/// Follow one parent per committed non-root layer, from the child to its
/// root. Callers can retain a previously found final edge in `labels`.
pub(super) fn walk_parents(
    layers: &[(u32, u32)],
    mut child: OwnerRow,
    labels: &mut Vec<u32>,
    mut lookup: impl FnMut((u32, u32), OwnerRow) -> Result<Trace<(OwnerRow, u32)>>,
) -> Result<Trace<OwnerRow>> {
    for &range in layers.iter().rev() {
        match lookup(range, child)? {
            Trace::Found((parent, label)) => {
                labels.push(label);
                child = parent;
            }
            Trace::Stopped(reason) => return Ok(Trace::Stopped(reason)),
        }
    }
    Ok(Trace::Found(child))
}
