//! Checked address and sizing contracts for an independently segmented history.
//! This module alone is not a GPU BFS backend.
use std::ops::Range;

/// Number of distinct values representable by a 24-bit local row.
pub const LOCAL_ROWS: u32 = 1 << 24;
const ROW_MASK: u32 = LOCAL_ROWS - 1;
const BANK_SHIFT: u32 = 24;
const VALID_PAYLOAD_MASK: u32 = (1 << 26) - 1;

/// A segment number is separate from the 24-bit row in that segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateAddress { segment: u32, bank: u8, row: u32 }

impl StateAddress {
    pub fn new(segment: u32, bank: u8, row: u32) -> Result<Self, &'static str> {
        if bank >= 3 || row >= LOCAL_ROWS { return Err("SEGMENTED_ADDRESS_RANGE"); }
        Ok(Self { segment, bank, row })
    }
    pub fn local_payload(self) -> u32 { (u32::from(self.bank) << BANK_SHIFT) | self.row }
    pub fn encode(self) -> u64 { (u64::from(self.segment) << 32) | u64::from(self.local_payload()) }
    pub fn decode(value: u64) -> Result<Self, &'static str> {
        let local = value as u32;
        if local & !VALID_PAYLOAD_MASK != 0 { return Err("SEGMENTED_ADDRESS_RESERVED_BITS"); }
        Self::new((value >> 32) as u32, (local >> BANK_SHIFT) as u8, local & ROW_MASK)
    }
    pub fn segment(self) -> u32 { self.segment }
    pub fn bank(self) -> u8 { self.bank }
    pub fn row(self) -> u32 { self.row }
}

/// Uniform upper bounds; actual allocations and NCCL reserves are admitted separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentGeometry { segments: u32, rows_per_bank: u32, banks: u32 }
impl SegmentGeometry {
    pub fn new(segments: u32, rows_per_bank: u32, banks: u32) -> Result<Self, &'static str> {
        if segments == 0 || rows_per_bank == 0 || rows_per_bank > LOCAL_ROWS || !matches!(banks, 1 | 3) {
            return Err("SEGMENTED_GEOMETRY_RANGE");
        }
        Ok(Self { segments, rows_per_bank, banks })
    }
    pub fn rows(self) -> u64 { u64::from(self.segments) * u64::from(self.rows_per_bank) * u64::from(self.banks) }
    pub fn arena_bytes(self, state_bytes: u32) -> Result<u64, &'static str> {
        if state_bytes == 0 { return Err("SEGMENTED_ZERO_STATE_BYTES"); }
        self.rows().checked_mul(u64::from(state_bytes)).ok_or("SEGMENTED_ARENA_BYTES_OVERFLOW")
    }
}

pub fn sum_rank_counts(counts: &[u64]) -> Result<u64, &'static str> {
    counts.iter().try_fold(0u64, |sum, &count| sum.checked_add(count).ok_or("SEGMENTED_RANK_COUNT_OVERFLOW"))
}

/// Split a 64-bit cursor into bounded per-launch ranges without narrowing it.
#[derive(Clone, Debug)]
pub struct ChunkRanges { next: u64, total: u64, chunk: u32 }
impl ChunkRanges {
    pub fn new(total: u64, chunk: u32) -> Result<Self, &'static str> {
        if chunk == 0 { return Err("SEGMENTED_ZERO_CHUNK"); }
        Ok(Self { next: 0, total, chunk })
    }
}
impl Iterator for ChunkRanges {
    type Item = Range<u64>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.total { return None; }
        let start = self.next;
        let end = start.saturating_add(u64::from(self.chunk)).min(self.total);
        self.next = end;
        Some(start..end)
    }
}

/// Device-side storage of one segment; owner queues and NCCL are admitted as
/// fixed allocations, rather than hidden in a percentage of free memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentLayout {
    pub rows_per_bank: u32,
    pub banks: u32,
    pub slots: u64,
    pub state_bytes: u64,
    pub position_bytes: u64,
    pub device_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentAdmission {
    pub segments: u32,
    pub rows_per_bank: u64,
    /// Segment allocations plus explicitly supplied fixed buffers; excludes reserve.
    pub device_bytes: u64,
}
impl SegmentLayout {
    pub fn new(elements: u32, state_bytes: u32, rows_per_bank: u32, banks: u32) -> Result<Self, &'static str> {
        if elements == 0 || !matches!(state_bytes, 1 | 8) || rows_per_bank == 0 || rows_per_bank > LOCAL_ROWS || !matches!(banks, 1 | 3) {
            return Err("SEGMENTED_LAYOUT_RANGE");
        }
        let rows = u64::from(rows_per_bank).checked_mul(u64::from(banks)).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let slots = rows.checked_mul(2).and_then(u64::checked_next_power_of_two).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let states = rows.checked_mul(u64::from(elements)).and_then(|v| v.checked_mul(u64::from(state_bytes))).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let positions = if banks == 3 { rows.checked_mul(8).ok_or("SEGMENTED_LAYOUT_OVERFLOW")? } else { 0 };
        let bytes = slots.checked_mul(8).and_then(|v| v.checked_add(states)).and_then(|v| v.checked_add(positions)).and_then(|v| v.checked_add(16)).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        Ok(Self { rows_per_bank, banks, slots, state_bytes: states, position_bytes: positions, device_bytes: bytes })
    }
    /// State-only storage for a bitmap backend; visited bits are allocated
    /// independently for the full rank-owned domain and must be admitted first.
    pub fn bitmap(elements:u32,state_bytes:u32,rows_per_bank:u32)->Result<Self,&'static str>{
        if elements==0||!matches!(state_bytes,1|8)||rows_per_bank==0||rows_per_bank>LOCAL_ROWS{return Err("BITMAP_LAYOUT_RANGE")}
        let rows=u64::from(rows_per_bank)*3;
        let states=rows.checked_mul(u64::from(elements)).and_then(|v|v.checked_mul(u64::from(state_bytes))).ok_or("BITMAP_LAYOUT_OVERFLOW")?;
        Ok(Self{rows_per_bank,banks:3,slots:0,state_bytes:states,position_bytes:0,device_bytes:states.checked_add(16).ok_or("BITMAP_LAYOUT_OVERFLOW")?})
    }
    /// `reserve` covers driver/NCCL headroom; `fixed` includes all other
    /// allocations, e.g. descriptors, communication pages and owner lanes.
    pub fn admit(self, free: u64, reserve: u64, fixed: u64, rank_upper_states: Option<u64>) -> Result<SegmentAdmission, &'static str> {
        if rank_upper_states == Some(0) { return Err("SEGMENTED_EMPTY_DOMAIN"); }
        let remaining = free.checked_sub(reserve).and_then(|v| v.checked_sub(fixed)).ok_or("SEGMENTED_ADMISSION_HEADROOM")?;
        let by_budget = remaining / self.device_bytes;
        // A shard can have a partially filled segment. Without per-shard
        // occupancy, ceil(states/rows) undercounts fragmentation. One
        // nonempty segment per state is a conservative global upper bound.
        let by_domain = rank_upper_states.unwrap_or(u64::MAX);
        let segments = by_budget.min(by_domain).min(u64::from(u32::MAX));
        if segments == 0 { return Err("SEGMENTED_NO_CAPACITY"); }
        let rows = segments.checked_mul(u64::from(self.rows_per_bank)).ok_or("SEGMENTED_ADMISSION_OVERFLOW")?;
        let used = segments.checked_mul(self.device_bytes).and_then(|v| v.checked_add(fixed)).ok_or("SEGMENTED_ADMISSION_OVERFLOW")?;
        Ok(SegmentAdmission { segments: segments as u32, rows_per_bank: rows, device_bytes: used })
    }
}

/// Classify source queue skew separately from regeneration validation errors.
/// Values above one dominate a retry in a max-reduction across ranks.
pub fn classify_route_and_regeneration(route: u32, regen: u32) -> u32 {
    if route & !1 != 0 || regen & !4 != 0 || (regen & 4 != 0 && route & 1 == 0) {
        0x1_0000 | ((route & 0xff) << 8) | (regen & 0xff)
    } else if route & 1 != 0 { 1 } else { 0 }
}


/// Ordered so a max collective cannot hide a fatal source/owner error behind
/// a capacity stop or a recoverable queue retry from the same or another rank.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum SegmentVote { Continue, Retry, Resource, OwnerFatal(u32), SourceFatal(u32) }
pub fn route_owner_vote(source:u32,owner:u32)->u32 {
    const PAYLOAD:u32=(1<<30)-1;
    if source>1 { 0xc000_0000 | source.min(PAYLOAD) }
    else if owner!=0 && owner!=10 { 0x8000_0000 | owner.min(PAYLOAD) }
    else if owner==10 { 0x4000_0000 }
    else if source==1 { 1 }
    else { 0 }
}
pub fn decode_route_owner_vote(value:u32)->SegmentVote {
    match value>>30 {
        3=>SegmentVote::SourceFatal(value&0x3fff_ffff),
        2=>SegmentVote::OwnerFatal(value&0x3fff_ffff),
        1=>SegmentVote::Resource,
        _=>if value==0 {SegmentVote::Continue} else {SegmentVote::Retry},
    }
}
