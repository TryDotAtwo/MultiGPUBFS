//! Host query reports. Converting a report never launches or emulates CUDA.
use mgbfs_core::{
    rank_plan::{QueryAllocation, QueryResult},
    Result,
};
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct GenerateBytes {
    pub generators: u64,
    pub packed_parents: u64,
    pub products_s32: u64,
    pub workspace: u64,
    pub k: u32,
    pub stride: u32,
    pub rows: u32,
    pub columns: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct HashBytes {
    pub weights: u64,
    pub offsets: u64,
    pub partials_s32: u64,
    pub workspace: u64,
    pub stride: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct MaterializeBytes {
    pub keys: u64,
    pub sorted: u64,
    pub indices: u64,
    pub order: u64,
    pub scratch: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct FutureMergeBytes {
    pub merged: u64,
    pub unique: u64,
    pub tags: u64,
    pub unique_tags: u64,
    pub indices: u64,
    pub selected: u64,
    pub selected_count: u64,
    pub flags: u64,
    pub states: u64,
    pub state: u64,
    pub scratch: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct RouteBytes {
    pub sorted: u64,
    pub refs: u64,
    pub indices: u64,
    pub selected: u64,
    pub flags: u64,
    pub scratch: u64,
    pub sort_scratch: u64,
    pub select_scratch: u64,
}
impl RouteBytes {
    pub fn report(self, capacity: u32) -> Result<QueryResult> {
        let c = u64::from(capacity);
        if capacity == 0
            || capacity > i32::MAX as u32
            || (
                self.sorted,
                self.refs,
                self.indices,
                self.selected,
                self.flags,
            ) != (c * 16, c * 8, c * 4, c * 4, c)
            || self.sort_scratch == 0
            || self.select_scratch == 0
            || self.scratch != self.sort_scratch.max(self.select_scratch)
        {
            return Err("INVALID_ROUTE_QUERY".into());
        }
        Ok(report(
            format!(
                "mgbfs_route_query/v1;capacity={capacity};sort_scratch={};select_scratch={}",
                self.sort_scratch, self.select_scratch
            ),
            [
                ("sorted", self.sorted),
                ("refs", self.refs),
                ("indices", self.indices),
                ("selected", self.selected),
                ("flags", self.flags),
                ("scratch", self.scratch),
            ],
        ))
    }
}
impl GenerateBytes {
    pub fn report(self, variant: u32) -> Result<QueryResult> {
        if variant > 4
            || [self.generators, self.packed_parents, self.products_s32].contains(&0)
            || [self.k, self.stride, self.rows, self.columns].contains(&0)
        {
            return Err("INVALID_GENERATION_QUERY".into());
        }
        Ok(report(
            format!(
                "mgbfs_generate_query/v1;variant={variant};k={};stride={};rows={};columns={}",
                self.k, self.stride, self.rows, self.columns
            ),
            [
                ("generators", self.generators),
                ("packed_parents", self.packed_parents),
                ("products_s32", self.products_s32),
                ("workspace", self.workspace),
            ],
        ))
    }
}
impl HashBytes {
    pub fn report(self) -> Result<QueryResult> {
        if self.reserved != 0
            || self.stride == 0
            || [self.weights, self.offsets, self.partials_s32].contains(&0)
        {
            return Err("INVALID_HASH_QUERY".into());
        }
        Ok(report(
            format!("mgbfs_hash_query/v1;stride={}", self.stride),
            [
                ("weights", self.weights),
                ("offsets", self.offsets),
                ("partials_s32", self.partials_s32),
                ("workspace", self.workspace),
            ],
        ))
    }
}
impl MaterializeBytes {
    pub fn report(self, capacity: u32) -> Result<QueryResult> {
        let count = u64::from(capacity);
        if capacity == 0
            || (self.keys, self.sorted, self.indices, self.order)
                != (count * 8, count * 8, count * 4, count * 4)
            || self.scratch == 0
        {
            return Err("INVALID_MATERIALIZE_QUERY".into());
        }
        Ok(report(
            format!("mgbfs_materialize_query/v1;capacity={capacity}"),
            [
                ("keys", self.keys),
                ("sorted", self.sorted),
                ("indices", self.indices),
                ("order", self.order),
                ("scratch", self.scratch),
            ],
        ))
    }
}
impl FutureMergeBytes {
    pub fn report(self, stride: u32, future: u32, incoming: u32) -> Result<QueryResult> {
        let total = u64::from(future)
            .checked_add(u64::from(incoming))
            .ok_or("INVALID_FUTURE_MERGE_QUERY")?;
        if stride == 0
            || stride % 16 != 0
            || future == 0
            || incoming == 0
            || total > i32::MAX as u64
            || (
                self.merged,
                self.unique,
                self.tags,
                self.unique_tags,
                self.indices,
                self.selected,
                self.selected_count,
                self.flags,
                self.states,
                self.state,
            ) != (
                total * 16,
                u64::from(future) * 16,
                total * 8,
                u64::from(future) * 8,
                total * 4,
                total * 4,
                4,
                total,
                u64::from(future) * u64::from(stride),
                8,
            )
            || self.scratch == 0
        {
            return Err("INVALID_FUTURE_MERGE_QUERY".into());
        }
        Ok(report(
            format!(
                "mgbfs_future_merge_query/v1;stride={stride};future={future};incoming={incoming}"
            ),
            [
                ("merged", self.merged),
                ("unique", self.unique),
                ("tags", self.tags),
                ("unique_tags", self.unique_tags),
                ("indices", self.indices),
                ("selected", self.selected),
                ("selected_count", self.selected_count),
                ("flags", self.flags),
                ("states", self.states),
                ("state", self.state),
                ("scratch", self.scratch),
            ],
        ))
    }
}
fn report<const N: usize>(source: String, planes: [(&str, u64); N]) -> QueryResult {
    QueryResult {
        source,
        allocations: planes
            .into_iter()
            .map(|(name, bytes)| QueryAllocation {
                name: name.into(),
                bytes,
                alignment: 256,
            })
            .collect(),
    }
}
/// Returns only the Generation query group; never fabricates other groups.
#[cfg(feature = "cuda")]
pub fn query_generation(
    n: u32,
    moves: u32,
    modulus: u32,
    parents: u32,
    variant: u32,
) -> Result<QueryResult> {
    let mut q = GenerateBytes::default();
    let status =
        unsafe { crate::ffi::mgbfs_generate_query(n, moves, modulus, parents, variant, &mut q) };
    if status != 0 {
        return Err(format!("GENERATION_QUERY:{status}"));
    }
    q.report(variant)
}
#[cfg(feature = "cuda")]
pub fn query_hash(bytes: u32, candidates: u32) -> Result<QueryResult> {
    let mut q = HashBytes::default();
    let status = unsafe { crate::ffi::mgbfs_hash_query(bytes, candidates, &mut q) };
    if status != 0 {
        return Err(format!("HASH_QUERY:{status}"));
    }
    q.report()
}
#[cfg(feature = "cuda")]
pub fn query_route(capacity: u32) -> Result<QueryResult> {
    let mut q = RouteBytes::default();
    let status = unsafe { crate::ffi::mgbfs_route_query(capacity, &mut q) };
    if status != 0 {
        return Err(format!("ROUTE_QUERY:{status}"));
    }
    q.report(capacity)
}
