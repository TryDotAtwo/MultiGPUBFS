//! Rank-local architecture-v2 allocation contract. No implicit scratch estimates.
use crate::{
    config::FrontierProfile,
    memory::{bounded_owner_ledger, Allocation, AllocationLedger},
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankShape {
    pub n: u64,
    pub moves: u64,
    pub modulus: u64,
    pub parents: u64,
    pub state_records: u64,
    pub extent_descriptors: u64,
    pub layer_records: u64,
    pub shards: u64,
    pub buckets: u64,
    pub bucket_records: u64,
    pub incoming: u64,
    pub touched_buckets: u64,
    pub materialize_records: u64,
    pub generation_lanes: u64,
    pub route_lanes: u64,
    pub owner_lanes: u64,
    pub materialize_lanes: u64,
    pub archive_slots: u64,
    pub archive_slot_bytes: u64,
    pub profile: FrontierProfile,
    /// Identifies compiled generation/hash/owner/route policy, not a display label.
    pub policy_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QueryKind {
    Generation,
    Hash,
    Route,
    OwnerMerge,
    OwnerSelect,
    OwnerScan,
    Materialize,
    Transport,
    Obligations,
    FixedDevice,
    ArchiveDevice,
    ControlPinned,
}
pub const REQUIRED_QUERIES: [QueryKind; 12] = [
    QueryKind::Generation,
    QueryKind::Hash,
    QueryKind::Route,
    QueryKind::OwnerMerge,
    QueryKind::OwnerSelect,
    QueryKind::OwnerScan,
    QueryKind::Materialize,
    QueryKind::Transport,
    QueryKind::Obligations,
    QueryKind::FixedDevice,
    QueryKind::ArchiveDevice,
    QueryKind::ControlPinned,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryAllocation {
    pub name: String,
    pub bytes: u64,
    pub alignment: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryResult {
    /// Required provenance: implementation/function/version; empty vector means
    /// explicitly queried zero, not unavailable.
    pub source: String,
    pub allocations: Vec<QueryAllocation>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankQueries {
    pub shape: RankShape,
    pub device_uuid: String,
    pub build_digest: [u8; 32],
    pub results: BTreeMap<QueryKind, QueryResult>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankPlan {
    pub device: Vec<Allocation>,
    pub pinned: Vec<Allocation>,
    pub device_bytes: u64,
    pub pinned_bytes: u64,
    pub untouched_reserve: u64,
    pub free_after_warmup: u64,
    pub queries: RankQueries,
}
pub fn rank_plan(
    shape: &RankShape,
    queries: &RankQueries,
    device_uuid: &str,
    build_digest: [u8; 32],
    free_after_warmup: u64,
    untouched_reserve: u64,
    pinned_budget: u64,
) -> Result<RankPlan> {
    let s = shape;
    if s != &queries.shape {
        return Err("QUERY_SHAPE_MISMATCH".into());
    }
    if build_digest != queries.build_digest {
        return Err("QUERY_BUILD_MISMATCH".into());
    }
    if device_uuid.is_empty() || device_uuid != queries.device_uuid {
        return Err("QUERY_DEVICE_MISMATCH".into());
    }
    for k in REQUIRED_QUERIES {
        let q = queries
            .results
            .get(&k)
            .ok_or_else(|| format!("MISSING_QUERY:{k:?}"))?;
        if q.source.trim().is_empty() {
            return Err("QUERY_PROVENANCE".into());
        }
    }
    if [
        s.n,
        s.moves,
        s.parents,
        s.state_records,
        s.extent_descriptors,
        s.layer_records,
        s.bucket_records,
        s.incoming,
        s.touched_buckets,
        s.materialize_records,
        s.generation_lanes,
        s.route_lanes,
        s.owner_lanes,
        s.materialize_lanes,
        s.archive_slots,
        s.archive_slot_bytes,
    ]
    .contains(&0)
        || !s.shards.is_power_of_two()
        || !s.buckets.is_power_of_two()
        || s.shards > s.buckets
        || s.touched_buckets > s.buckets
        || !(2..=256).contains(&s.modulus)
        || s.moves > 65535
    {
        return Err("RANK_SHAPE".into());
    }
    let d = mul(s.n, s.n)?;
    let stride = align(d, 16)?;
    let c = mul(s.parents, s.moves)?;
    let bucket_total = mul(s.buckets, s.bucket_records)?;
    if s.layer_records > bucket_total
        || d > 33025
        || mul(mul(s.n, s.modulus - 1)?, s.modulus - 1)? > i32::MAX as u64
        || [
            c,
            s.incoming,
            s.bucket_records,
            s.touched_buckets,
            s.materialize_records,
        ]
        .iter()
        .any(|&x| x > i32::MAX as u64)
    {
        return Err("RANK_SHAPE_BOUND".into());
    }
    // No overlays: every simultaneous lifetime is charged. This builds a layout
    // only; it never allocates GPU/pinned storage or launches CUDA.
    let mut device = AllocationLedger::new(u64::MAX, 0)?;
    let mut pinned = AllocationLedger::new(u64::MAX, 0)?;
    plane(
        &mut pinned,
        "archive_slots",
        s.archive_slots,
        s.archive_slot_bytes,
        4096,
    )?;
    for (name, count, size) in [
        ("state_ring", s.state_records, stride),
        ("hash_arena_prev", s.layer_records, 16),
        ("hash_arena_curr", s.layer_records, 16),
        ("accepted_hashes", bucket_total, 16),
        ("bucket_counts", s.buckets, 4),
        (
            "hash_directory_prev",
            s.buckets.checked_add(1).ok_or("BYTE_OVERFLOW")?,
            8,
        ),
        (
            "hash_directory_curr",
            s.buckets.checked_add(1).ok_or("BYTE_OVERFLOW")?,
            8,
        ),
        ("extent_descriptors", s.extent_descriptors, 64),
    ] {
        device.add(name, count, size, 256)?;
    }
    for (name, lanes, count, size) in [
        ("parent_banks", s.generation_lanes, s.parents, stride),
        ("generation_states", s.generation_lanes, c, stride),
        ("generation_hashes", s.generation_lanes, c, 16),
        ("route_hashes_0", s.route_lanes, c, 16),
        ("route_hashes_1", s.route_lanes, c, 16),
        ("route_ordinals_0", s.route_lanes, c, 4),
        ("route_ordinals_1", s.route_lanes, c, 4),
    ] {
        plane(&mut device, name, lanes, mul(count, size)?, 256)?;
    }
    // The owner ledger knows only I,J,K, never L or the whole accepted arena.
    let owner = bounded_owner_ledger(s.incoming, s.touched_buckets, s.bucket_records, [0; 3])?;
    for a in owner
        .allocations
        .into_iter()
        .filter(|a| a.reserved_bytes != 0)
    {
        plane(
            &mut device,
            &format!("owner/{}", a.name),
            s.owner_lanes,
            a.payload_bytes,
            256,
        )?;
    }
    // Materialize query owns ALL Qmat planes, including request order, expected
    // hashes, response states and reconstruction scratch. Transport query owns
    // ALL typed send/receive banks, framing, directories and ticket metadata.
    // Generation/hash queries own their internal packed inputs, tables and
    // int32 intermediates; the external state/hash banks above are excluded.
    for k in REQUIRED_QUERIES {
        let (prefix, lanes) = match k {
            QueryKind::Generation | QueryKind::Hash => ("generation", s.generation_lanes),
            QueryKind::Route => ("route", s.route_lanes),
            QueryKind::OwnerMerge | QueryKind::OwnerSelect | QueryKind::OwnerScan => {
                ("owner", s.owner_lanes)
            }
            QueryKind::Materialize => ("materialize", s.materialize_lanes),
            _ => ("rank", 1),
        };
        let ledger = if k == QueryKind::ControlPinned {
            &mut pinned
        } else {
            &mut device
        };
        for a in &queries.results[&k].allocations {
            if a.name.is_empty()
                || !a
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            {
                return Err("QUERY_ALLOCATION_NAME".into());
            }
            plane(
                ledger,
                &format!("{prefix}/{k:?}/{}", a.name),
                lanes,
                a.bytes,
                a.alignment,
            )?;
        }
    }
    let available = free_after_warmup
        .checked_sub(untouched_reserve)
        .ok_or("VRAM_RESERVE")?;
    if device.total() > available {
        return Err(format!(
            "DEVICE_CAPACITY:required={},available={available}",
            device.total()
        ));
    }
    if pinned.total() > pinned_budget {
        return Err(format!(
            "PINNED_CAPACITY:required={},available={pinned_budget}",
            pinned.total()
        ));
    }
    Ok(RankPlan {
        device_bytes: device.total(),
        pinned_bytes: pinned.total(),
        device: device.allocations,
        pinned: pinned.allocations,
        untouched_reserve,
        free_after_warmup,
        queries: queries.clone(),
    })
}

fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| "BYTE_OVERFLOW".into())
}
fn align(n: u64, a: u64) -> Result<u64> {
    if !a.is_power_of_two() {
        return Err("ALLOCATION_ALIGNMENT".into());
    }
    n.checked_add(a - 1)
        .map(|x| x & !(a - 1))
        .ok_or_else(|| "BYTE_OVERFLOW".into())
}
fn plane(
    l: &mut AllocationLedger,
    name: &str,
    lanes: u64,
    bytes: u64,
    alignment: u64,
) -> Result<()> {
    if alignment < 256 {
        return Err("ALLOCATION_ALIGNMENT".into());
    }
    l.add(name, lanes, align(bytes, alignment)?, alignment)?;
    Ok(())
}
