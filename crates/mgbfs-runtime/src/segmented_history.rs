//! Checked address and sizing contracts for an independently segmented history.
//! This module alone is not a GPU BFS backend.
use std::ops::Range;

/// A local row is 32 bits; its cardinality needs 64 bits to represent 2^32.
pub const LOCAL_ROWS: u64 = 1u64 << 32;
const BANK_SHIFT: u32 = 32;
const LOCAL_PAYLOAD_BITS: u32 = 34;
const LOCAL_PAYLOAD_MASK: u64 = (1u64 << LOCAL_PAYLOAD_BITS) - 1;

/// A segment number is separate from the local 32-bit row and two bank bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateAddress {
    segment: u32,
    bank: u8,
    row: u32,
}

impl StateAddress {
    pub fn new(segment: u32, bank: u8, row: u64) -> Result<Self, &'static str> {
        if bank >= 3 || row >= LOCAL_ROWS {
            return Err("SEGMENTED_ADDRESS_RANGE");
        }
        Ok(Self {
            segment,
            bank,
            row: row as u32,
        })
    }
    pub fn local_payload(self) -> u64 {
        (u64::from(self.bank) << BANK_SHIFT) | u64::from(self.row)
    }
    // Segment + bank + row need 66 bits; never truncate a global address to u64.
    pub fn encode(self) -> u128 {
        (u128::from(self.segment) << LOCAL_PAYLOAD_BITS) | u128::from(self.local_payload())
    }
    pub fn decode(value: u128) -> Result<Self, &'static str> {
        if value >> (LOCAL_PAYLOAD_BITS + 32) != 0 {
            return Err("SEGMENTED_ADDRESS_RESERVED_BITS");
        }
        let local = (value as u64) & LOCAL_PAYLOAD_MASK;
        Self::new(
            (value >> LOCAL_PAYLOAD_BITS) as u32,
            (local >> BANK_SHIFT) as u8,
            local & (LOCAL_ROWS - 1),
        )
    }
    pub fn segment(self) -> u32 {
        self.segment
    }
    pub fn bank(self) -> u8 {
        self.bank
    }
    pub fn row(self) -> u32 {
        self.row
    }
}

/// Uniform upper bounds; actual allocations and NCCL reserves are admitted separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentGeometry {
    segments: u32,
    rows_per_bank: u64,
    banks: u32,
}
impl SegmentGeometry {
    pub fn new(segments: u32, rows_per_bank: u64, banks: u32) -> Result<Self, &'static str> {
        if segments == 0
            || rows_per_bank == 0
            || rows_per_bank > LOCAL_ROWS
            || !matches!(banks, 1 | 3)
        {
            return Err("SEGMENTED_GEOMETRY_RANGE");
        }
        Ok(Self {
            segments,
            rows_per_bank,
            banks,
        })
    }
    pub fn rows(self) -> u128 {
        u128::from(self.segments) * u128::from(self.rows_per_bank) * u128::from(self.banks)
    }
    pub fn arena_bytes(self, state_bytes: u32) -> Result<u64, &'static str> {
        if state_bytes == 0 {
            return Err("SEGMENTED_ZERO_STATE_BYTES");
        }
        let bytes = self
            .rows()
            .checked_mul(u128::from(state_bytes))
            .ok_or("SEGMENTED_ARENA_BYTES_OVERFLOW")?;
        u64::try_from(bytes).map_err(|_| "SEGMENTED_ARENA_BYTES_OVERFLOW")
    }
}

pub fn sum_rank_counts(counts: &[u64]) -> Result<u64, &'static str> {
    counts.iter().try_fold(0u64, |sum, &count| {
        sum.checked_add(count)
            .ok_or("SEGMENTED_RANK_COUNT_OVERFLOW")
    })
}

/// Assign the 32-bit owner hash space in proportion to admitted rows.
pub fn weighted_owner_cuts(rows: &[u64]) -> Result<Vec<u64>, &'static str> {
    if rows.is_empty() || rows.len() > 128 || rows.contains(&0) {
        return Err("SEGMENTED_OWNER_ADMISSION_ROWS");
    }
    let total: u128 = rows.iter().map(|&value| u128::from(value)).sum();
    let mut cuts = vec![0];
    let mut prefix = 0u128;
    for (index, &value) in rows.iter().enumerate() {
        prefix += u128::from(value);
        let nominal = ((prefix * u128::from(LOCAL_ROWS) + total - 1) / total) as u64;
        let remaining = (rows.len() - index - 1) as u64;
        cuts.push(nominal.clamp(cuts[index] + 1, LOCAL_ROWS - remaining));
    }
    Ok(cuts)
}

/// Split a 64-bit cursor into bounded per-launch ranges without narrowing it.
#[derive(Clone, Debug)]
pub struct ChunkRanges {
    next: u64,
    total: u64,
    chunk: u32,
}
impl ChunkRanges {
    pub fn new(total: u64, chunk: u32) -> Result<Self, &'static str> {
        if chunk == 0 {
            return Err("SEGMENTED_ZERO_CHUNK");
        }
        Ok(Self {
            next: 0,
            total,
            chunk,
        })
    }
}
impl Iterator for ChunkRanges {
    type Item = Range<u64>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.total {
            return None;
        }
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
    pub rows_per_bank: u64,
    pub banks: u32,
    pub slots: u64,
    pub state_bytes: u64,
    pub position_bytes: u64,
    pub position_item_bytes: u32,
    pub device_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentAdmission {
    pub segments: u32,
    pub rows_per_bank: u64,
    /// Segment allocations plus explicitly supplied fixed buffers; excludes reserve.
    pub device_bytes: u64,
}
/// ABI and immutable symbol dictionary for the experimental packed bitmap
/// resident frontier; active only for the explicit experimental bitmap mode.
pub const BITMAP_PACK4_ABI_VERSION: u32 = 2;
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitmapPack4Dictionary {
    pub elements: u32,
    pub distinct: u32,
    pub symbols: [i64; 16],
}
impl BitmapPack4Dictionary {
    pub fn width(elements: u32, distinct: u32) -> u32 {
        if elements <= 16 {
            4
        } else {
            (u32::BITS - distinct.saturating_sub(1).leading_zeros()).max(1)
        }
    }

    /// Canonical signed symbol ordering is independent of root position.
    /// The complete graph fingerprint, including multiplicities, remains a
    /// separate rank agreement; this dictionary only identifies the alphabet.
    pub fn from_root(root: &[i64]) -> Result<Self, &'static str> {
        if root.is_empty() || root.len() > 64 {
            return Err("BITMAP_PACK4_ELEMENTS_RANGE");
        }
        let mut distinct = root.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() > 16
            || root.len() as u32 * Self::width(root.len() as u32, distinct.len() as u32) > 64
        {
            return Err("BITMAP_PACK4_ELEMENTS_RANGE");
        }
        let mut symbols = [0i64; 16];
        symbols[..distinct.len()].copy_from_slice(&distinct);
        Ok(Self {
            elements: root.len() as u32,
            distinct: distinct.len() as u32,
            symbols,
        })
    }
}

impl SegmentLayout {
    pub fn new(
        elements: u32,
        state_bytes: u32,
        rows_per_bank: u64,
        banks: u32,
    ) -> Result<Self, &'static str> {
        if elements == 0
            || !matches!(state_bytes, 1 | 8)
            || rows_per_bank == 0
            || rows_per_bank > LOCAL_ROWS
            || !matches!(banks, 1 | 3)
        {
            return Err("SEGMENTED_LAYOUT_RANGE");
        }
        let rows = rows_per_bank
            .checked_mul(u64::from(banks))
            .ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let slots = rows
            .checked_mul(2)
            .and_then(u64::checked_next_power_of_two)
            .ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let states = rows
            .checked_mul(u64::from(elements))
            .and_then(|v| v.checked_mul(u64::from(state_bytes)))
            .ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let positions = if banks == 3 {
            rows.checked_mul(8).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?
        } else {
            0
        };
        let bytes = slots
            .checked_mul(8)
            .and_then(|v| v.checked_add(states))
            .and_then(|v| v.checked_add(positions))
            .and_then(|v| v.checked_add(24))
            .ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        Ok(Self {
            rows_per_bank,
            banks,
            slots,
            state_bytes: states,
            position_bytes: positions,
            position_item_bytes: if banks == 3 { 8 } else { 0 },
            device_bytes: bytes,
        })
    }
    /// Use exact 32-bit positions only when every hash slot index fits u32.
    /// There is no sentinel in the positions array: it is read only for
    /// initialized live rows, so slot index u32::MAX remains representable.
    pub fn new_compact(
        elements: u32,
        state_bytes: u32,
        rows_per_bank: u64,
        banks: u32,
    ) -> Result<Self, &'static str> {
        let mut layout = Self::new(elements, state_bytes, rows_per_bank, banks)?;
        if banks == 3 && layout.slots <= (1u64 << 32) {
            let saved = layout.position_bytes / 2;
            layout.position_bytes -= saved;
            layout.device_bytes -= saved;
            layout.position_item_bytes = 4;
        }
        Ok(layout)
    }
    /// One full state plus flat u64 key and u32 local row indexes per live
    /// row. Runs describe disjoint extents; states are never duplicated.
    pub fn sorted(
        elements: u32,
        state_bytes: u32,
        rows_per_bank: u64,
        banks: u32,
    ) -> Result<Self, &'static str> {
        Self::sorted_with_row_metadata(elements, state_bytes, rows_per_bank, banks, 0)
    }
    pub(crate) fn sorted_with_row_metadata(
        elements: u32,
        state_bytes: u32,
        rows_per_bank: u64,
        banks: u32,
        metadata_bytes: u64,
    ) -> Result<Self, &'static str> {
        if elements == 0
            || !matches!(state_bytes, 1 | 8)
            || !matches!(banks, 1 | 3)
            || rows_per_bank == 0
            || rows_per_bank > LOCAL_ROWS
        {
            return Err("SEGMENTED_SORTED_LAYOUT_RANGE");
        }
        let rows = rows_per_bank
            .checked_mul(u64::from(banks))
            .ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let states = rows
            .checked_mul(u64::from(elements))
            .and_then(|v| v.checked_mul(u64::from(state_bytes)))
            .ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let keys = rows.checked_mul(8).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let positions = rows.checked_mul(4).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let bytes = states
            .checked_add(keys)
            .and_then(|v| v.checked_add(positions))
            .and_then(|v| v.checked_add(rows.checked_mul(metadata_bytes)?))
            .and_then(|v| v.checked_add(24))
            .ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        Ok(Self {
            rows_per_bank,
            banks,
            slots: rows,
            state_bytes: states,
            position_bytes: positions,
            position_item_bytes: 4,
            device_bytes: bytes,
        })
    }
    /// State-only storage for a bitmap backend; visited bits are allocated
    /// independently for the full rank-owned domain and must be admitted first.
    pub fn bitmap(
        elements: u32,
        state_bytes: u32,
        rows_per_bank: u64,
    ) -> Result<Self, &'static str> {
        Self::bitmap_banks(elements, state_bytes, rows_per_bank, 3)
    }
    pub fn bitmap_banks(
        elements: u32,
        state_bytes: u32,
        rows_per_bank: u64,
        banks: u32,
    ) -> Result<Self, &'static str> {
        if elements == 0
            || !matches!(state_bytes, 1 | 8)
            || !matches!(banks, 2 | 3)
            || rows_per_bank == 0
            || rows_per_bank > LOCAL_ROWS
        {
            return Err("BITMAP_LAYOUT_RANGE");
        }
        let rows = rows_per_bank * u64::from(banks);
        let states = rows
            .checked_mul(u64::from(elements))
            .and_then(|v| v.checked_mul(u64::from(state_bytes)))
            .ok_or("BITMAP_LAYOUT_OVERFLOW")?;
        Ok(Self {
            rows_per_bank,
            banks,
            slots: 0,
            state_bytes: states,
            position_bytes: 0,
            position_item_bytes: 0,
            device_bytes: states.checked_add(24).ok_or("BITMAP_LAYOUT_OVERFLOW")?,
        })
    }
    /// Experimental packed bitmap rows: one u64 per complete state.
    /// Selection requires the experimental CUDA capability and rank agreement;
    /// ordinary bitmap admission keeps its established resident layout.
    pub fn bitmap_banks_pack4(
        elements: u32,
        rows_per_bank: u64,
        banks: u32,
    ) -> Result<Self, &'static str> {
        if elements == 0
            || elements > 64
            || !matches!(banks, 2 | 3)
            || rows_per_bank == 0
            || rows_per_bank > LOCAL_ROWS
        {
            return Err("BITMAP_PACK4_LAYOUT_RANGE");
        }
        let rows = rows_per_bank
            .checked_mul(u64::from(banks))
            .ok_or("BITMAP_PACK4_LAYOUT_OVERFLOW")?;
        let states = rows.checked_mul(8).ok_or("BITMAP_PACK4_LAYOUT_OVERFLOW")?;
        let device_bytes = states
            .checked_add(24)
            .ok_or("BITMAP_PACK4_LAYOUT_OVERFLOW")?;
        Ok(Self {
            rows_per_bank,
            banks,
            slots: 0,
            state_bytes: states,
            position_bytes: 0,
            position_item_bytes: 0,
            device_bytes,
        })
    }
    /// `reserve` covers driver/NCCL headroom; `fixed` includes all other
    /// allocations, e.g. descriptors, communication pages and owner lanes.
    pub fn admit(
        self,
        free: u64,
        reserve: u64,
        fixed: u64,
        rank_upper_states: Option<u64>,
    ) -> Result<SegmentAdmission, &'static str> {
        if rank_upper_states == Some(0) {
            return Err("SEGMENTED_EMPTY_DOMAIN");
        }
        let remaining = free
            .checked_sub(reserve)
            .and_then(|v| v.checked_sub(fixed))
            .ok_or("SEGMENTED_ADMISSION_HEADROOM")?;
        let by_budget = remaining / self.device_bytes;
        // A shard can have a partially filled segment. Without per-shard
        // occupancy, ceil(states/rows) undercounts fragmentation. One
        // nonempty segment per state is a conservative global upper bound.
        let by_domain = rank_upper_states.unwrap_or(u64::MAX);
        let segments = by_budget.min(by_domain).min(u64::from(u32::MAX));
        if segments == 0 {
            return Err("SEGMENTED_NO_CAPACITY");
        }
        let rows = segments
            .checked_mul(self.rows_per_bank)
            .ok_or("SEGMENTED_ADMISSION_OVERFLOW")?;
        let used = segments
            .checked_mul(self.device_bytes)
            .and_then(|v| v.checked_add(fixed))
            .ok_or("SEGMENTED_ADMISSION_OVERFLOW")?;
        Ok(SegmentAdmission {
            segments: segments as u32,
            rows_per_bank: rows,
            device_bytes: used,
        })
    }
}

/// Largest complete segment layout that fits an actual byte budget.
/// The cap describes one bank of one segment, never a rank-wide state limit.
/// Hash table slot rounding and bitmap state storage use the same constructors
/// as device allocations, so admission cannot underestimate these bytes.
pub fn fit_segment_layout(
    bitmap: bool,
    elements: u32,
    state_bytes: u32,
    banks: u32,
    max_rows: u64,
    budget: u64,
) -> Result<Option<SegmentLayout>, &'static str> {
    fit_segment_layout_compact(
        bitmap,
        elements,
        state_bytes,
        banks,
        max_rows,
        budget,
        false,
    )
}

pub fn fit_segment_layout_compact(
    bitmap: bool,
    elements: u32,
    state_bytes: u32,
    banks: u32,
    max_rows: u64,
    budget: u64,
    compact: bool,
) -> Result<Option<SegmentLayout>, &'static str> {
    if max_rows == 0 || max_rows > LOCAL_ROWS {
        return Err("SEGMENTED_LAYOUT_RANGE");
    }
    if bitmap {
        return fit_linear_segment(max_rows, budget, |rows| {
            SegmentLayout::bitmap_banks(elements, state_bytes, rows, banks)
        });
    }
    let make = |rows| {
        if compact {
            SegmentLayout::new_compact(elements, state_bytes, rows, banks)
        } else {
            SegmentLayout::new(elements, state_bytes, rows, banks)
        }
    };
    make(1)?;
    let mut lo = 0u64;
    let mut hi = max_rows;
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if make(mid).map_or(false, |v| v.device_bytes <= budget) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo == 0 {
        Ok(None)
    } else {
        Ok(Some(make(lo)?))
    }
}

// Bitmap and sorted hash extents vary linearly with rows. All three
// layouts prepend exactly three u64 counters, independent of the row count.
fn fit_linear_segment(
    max_rows: u64,
    budget: u64,
    make: impl Fn(u64) -> Result<SegmentLayout, &'static str>,
) -> Result<Option<SegmentLayout>, &'static str> {
    const COUNTER_BYTES: u64 = 3 * 8;
    let bytes_per_capacity_row = make(1)?.device_bytes - COUNTER_BYTES;
    let rows = (budget.saturating_sub(COUNTER_BYTES) / bytes_per_capacity_row).min(max_rows);
    if rows == 0 {
        Ok(None)
    } else {
        make(rows).map(Some)
    }
}

/// Experimental packed bitmap layout for an explicit budget. The maximum
/// applies to one bank of one segment, not to the sum of rows on a rank.
pub fn fit_packed_bitmap_segment_layout(
    elements: u32,
    banks: u32,
    max_rows: u64,
    budget: u64,
) -> Result<Option<SegmentLayout>, &'static str> {
    if max_rows == 0 || max_rows > LOCAL_ROWS {
        return Err("BITMAP_PACK4_LAYOUT_RANGE");
    }
    fit_linear_segment(max_rows, budget, |rows| {
        SegmentLayout::bitmap_banks_pack4(elements, rows, banks)
    })
}

/// Largest sorted-index segment that fits a checked byte budget.
pub fn fit_sorted_segment_layout(
    elements: u32,
    state_bytes: u32,
    banks: u32,
    max_rows: u64,
    budget: u64,
) -> Result<Option<SegmentLayout>, &'static str> {
    fit_sorted_segment_layout_with_row_metadata(elements, state_bytes, banks, max_rows, budget, 0)
}

pub(crate) fn fit_sorted_segment_layout_with_row_metadata(
    elements: u32,
    state_bytes: u32,
    banks: u32,
    max_rows: u64,
    budget: u64,
    metadata_bytes: u64,
) -> Result<Option<SegmentLayout>, &'static str> {
    if max_rows == 0 || max_rows > LOCAL_ROWS {
        return Err("SEGMENTED_SORTED_LAYOUT_RANGE");
    }
    fit_linear_segment(max_rows, budget, |rows| {
        SegmentLayout::sorted_with_row_metadata(elements, state_bytes, rows, banks, metadata_bytes)
    })
}
/// Classify source queue skew separately from regeneration validation errors.
/// Values above one dominate a retry in a max-reduction across ranks.
pub fn classify_route_and_regeneration(route: u32, regen: u32) -> u32 {
    if route & !1 != 0 || regen & !4 != 0 || (regen & 4 != 0 && route & 1 == 0) {
        0x1_0000 | ((route & 0xff) << 8) | (regen & 0xff)
    } else if route & 1 != 0 {
        1
    } else {
        0
    }
}

/// Ordered so a max collective cannot hide a fatal source/owner error behind
/// a capacity stop or a recoverable queue retry from the same or another rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentVote {
    Continue,
    Retry,
    Stop(u32),
    Resource,
    OwnerFatal(u32),
    SourceFatal(u32),
}
pub fn route_owner_vote(source: u32, owner: u32) -> u32 {
    const PAYLOAD: u32 = (1 << 30) - 1;
    if source > 1 {
        0xc000_0000 | source.min(PAYLOAD)
    } else if owner != 0 && owner != 10 {
        0x8000_0000 | owner.min(PAYLOAD)
    } else if owner == 10 {
        0x4000_0000
    } else if source == 1 {
        1
    } else {
        0
    }
}
/// A requested stop outranks a retry, while resource and fatal errors win.
pub fn route_owner_stop_vote(reason: u32) -> u32 {
    match reason {
        1 => 2,
        2 => 3,
        _ => 0,
    }
}
pub fn decode_route_owner_vote(value: u32) -> SegmentVote {
    match value >> 30 {
        3 => SegmentVote::SourceFatal(value & 0x3fff_ffff),
        2 => SegmentVote::OwnerFatal(value & 0x3fff_ffff),
        1 => SegmentVote::Resource,
        _ => match value {
            0 => SegmentVote::Continue,
            2 | 3 => SegmentVote::Stop(value - 1),
            _ => SegmentVote::Retry,
        },
    }
}

#[cfg(test)]
mod batch_layout_tests {
    #[test]
    fn sorted_index_checked_capacity_and_bytes() {
        use super::*;
        let x = SegmentLayout::sorted(14, 1, 100, 3).unwrap();
        assert_eq!(x.state_bytes, 4200);
        assert_eq!(x.slots, 300);
        assert_eq!(x.position_bytes, 1200);
        assert_eq!(x.device_bytes, 3 * 100 * (14 + 8 + 4) + 24);
        assert_eq!(
            fit_sorted_segment_layout(14, 1, 3, 100, x.device_bytes)
                .unwrap()
                .unwrap()
                .rows_per_bank,
            100
        );
        assert_eq!(
            fit_sorted_segment_layout(14, 1, 3, 100, x.device_bytes - 1)
                .unwrap()
                .unwrap()
                .rows_per_bank,
            99
        );
        let last = SegmentLayout::sorted(1, 1, LOCAL_ROWS, 1).unwrap();
        assert_eq!(last.rows_per_bank, LOCAL_ROWS);
        assert_eq!(last.device_bytes, 13 * LOCAL_ROWS + 24);
        assert!(SegmentLayout::sorted(u32::MAX, 8, LOCAL_ROWS, 3).is_err());
    }
}

#[cfg(test)]
mod packed_bitmap_contract_tests {
    use super::*;

    #[test]
    fn descriptor_has_c_abi_sorted_symbols_and_checked_domain() {
        use std::mem::{align_of, size_of};
        let d = BitmapPack4Dictionary::from_root(&[17, -4, 17, 9, -4]).unwrap();
        assert_eq!(d.elements, 5);
        assert_eq!(d.distinct, 3);
        assert_eq!(&d.symbols[..3], &[-4, 9, 17]);
        assert_eq!(
            d,
            BitmapPack4Dictionary::from_root(&[9, -4, 17, 17, -4]).unwrap()
        );
        assert_eq!(size_of::<BitmapPack4Dictionary>(), 136);
        assert_eq!(align_of::<BitmapPack4Dictionary>(), 8);
        assert_eq!(BITMAP_PACK4_ABI_VERSION, 2);
        assert!(BitmapPack4Dictionary::from_root(&[]).is_err());
        assert!(BitmapPack4Dictionary::from_root(&[0; 65]).is_err());
        let mut sixteen = [0i64; 16];
        for (i, v) in sixteen.iter_mut().enumerate() {
            *v = i as i64 - 8;
        }
        let full = BitmapPack4Dictionary::from_root(&sixteen).unwrap();
        assert_eq!(full.elements, 16);
        assert_eq!(full.distinct, 16);
        assert_eq!(full.symbols[15], 7);
    }

    #[test]
    fn packed_bitmap_layout_checked_with_per_segment_rows() {
        let x = SegmentLayout::bitmap_banks_pack4(14, 100, 2).unwrap();
        assert_eq!(x.state_bytes, 1600);
        assert_eq!(x.device_bytes, 1624);
        assert_eq!(x.position_bytes, 0);
        assert_eq!(x.slots, 0);
        assert_eq!(
            fit_packed_bitmap_segment_layout(14, 2, 100, 1624)
                .unwrap()
                .unwrap()
                .rows_per_bank,
            100
        );
        assert_eq!(
            fit_packed_bitmap_segment_layout(14, 2, 100, 1623)
                .unwrap()
                .unwrap()
                .rows_per_bank,
            99
        );
        assert_eq!(
            fit_packed_bitmap_segment_layout(14, 2, 100, 39).unwrap(),
            None
        );
        let last = SegmentLayout::bitmap_banks_pack4(16, LOCAL_ROWS, 2).unwrap();
        assert_eq!(last.rows_per_bank, 1u64 << 32);
        assert_eq!(last.state_bytes, (1u64 << 32) * 16);
        let admitted = last.admit(last.device_bytes * 2, 0, 0, None).unwrap();
        assert_eq!(admitted.segments, 2);
        assert_eq!(admitted.rows_per_bank, 1u64 << 33);
        assert!(SegmentLayout::bitmap_banks_pack4(16, LOCAL_ROWS + 1, 2).is_err());
        assert!(SegmentLayout::bitmap_banks_pack4(65, 100, 2).is_err());
        assert!(SegmentLayout::bitmap_banks_pack4(0, 100, 2).is_err());
        assert!(SegmentLayout::bitmap_banks_pack4(8, 0, 2).is_err());
        assert!(SegmentLayout::bitmap_banks_pack4(8, 100, 1).is_err());
    }
}
/// Preserve physical order while recycling empty segment slots. Run and
/// frontier descriptors use the same old-to-new mapping after a drain.
pub(crate) fn segment_remap(banks: &[[u64; 3]], preserve_indices: bool) -> Vec<u32> {
    let retained = if preserve_indices {
        banks
            .iter()
            .rposition(|counts| *counts != [0; 3])
            .map_or(0, |i| i + 1)
    } else {
        0
    };
    let mut next = 0;
    banks
        .iter()
        .enumerate()
        .map(|(i, counts)| {
            if i >= retained && *counts == [0; 3] {
                return u32::MAX;
            }
            let id = next;
            next += 1;
            id
        })
        .collect()
}

#[cfg(test)]
mod segment_remap_tests {
    use super::segment_remap;

    #[test]
    fn empty_segments_reclaim_without_renumbering_live_order() {
        let banks = [[0, 0, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]];
        assert_eq!(
            segment_remap(&banks, false),
            [u32::MAX, 0, u32::MAX, 1, u32::MAX]
        );
        assert_eq!(segment_remap(&banks, true), [0, 1, 2, 3, u32::MAX]);
        assert_eq!(segment_remap(&banks[..1], true), [u32::MAX]);
    }
}

/// One sample per timed owner lookup, using the immutable run directory
/// supplied to that launch. The count bounds entries that lookup may visit.
#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
#[derive(Default)]
pub(crate) struct HashLookupDirectorySeries {
    samples: u64,
    frequencies: std::collections::BTreeMap<u32, u64>,
}

#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
#[derive(Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct HashLookupDirectorySummary {
    pub samples: u64,
    pub min: Option<u32>,
    pub median: Option<u32>,
    pub p95: Option<u32>,
    pub p99: Option<u32>,
    pub max: Option<u32>,
}

#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
impl HashLookupDirectorySeries {
    pub(crate) fn observe(&mut self, runs: u32) -> Result<(), &'static str> {
        self.samples = self
            .samples
            .checked_add(1)
            .ok_or("SEGMENTED_HASH_TIMING_OVERFLOW")?;
        *self.frequencies.entry(runs).or_default() += 1;
        Ok(())
    }

    pub(crate) fn summary(&self) -> HashLookupDirectorySummary {
        HashLookupDirectorySummary {
            samples: self.samples,
            min: self.frequencies.keys().next().copied(),
            median: self.percentile(50),
            p95: self.percentile(95),
            p99: self.percentile(99),
            max: self.frequencies.keys().next_back().copied(),
        }
    }

    fn percentile(&self, percent: u128) -> Option<u32> {
        if self.samples == 0 {
            return None;
        }
        let rank = ((u128::from(self.samples) * percent + 99) / 100) as u64;
        let mut seen = 0;
        for (&runs, &count) in &self.frequencies {
            seen += count;
            if seen >= rank {
                return Some(runs);
            }
        }
        None
    }
}

#[cfg(test)]
mod hash_lookup_directory_tests {
    use super::{HashLookupDirectorySeries, HashLookupDirectorySummary};

    #[test]
    fn exact_nearest_ranks_include_repeated_and_zero_run_launches() {
        let mut series = HashLookupDirectorySeries::default();
        assert_eq!(
            series.summary(),
            HashLookupDirectorySummary {
                samples: 0,
                min: None,
                median: None,
                p95: None,
                p99: None,
                max: None,
            }
        );
        for runs in [0; 50]
            .into_iter()
            .chain([8; 45])
            .chain([20; 4])
            .chain([u32::MAX])
        {
            series.observe(runs).unwrap();
        }
        assert_eq!(
            series.summary(),
            HashLookupDirectorySummary {
                samples: 100,
                min: Some(0),
                median: Some(0),
                p95: Some(8),
                p99: Some(20),
                max: Some(u32::MAX),
            }
        );
    }
}

/// Decide whether construction fits with or without reclaimable index bytes.
/// The caller decides how to reclaim them without losing a usable lookup.
pub(crate) fn snapshot_rebuild_admission(
    required: u64,
    index_bytes: u64,
    budget_left: u64,
    index_headroom: u64,
    physical_free: u64,
    reserve: u64,
    releasable: u64,
) -> Option<bool> {
    let fits = |released: u64| {
        let available = budget_left.saturating_add(released);
        let physical = physical_free
            .saturating_add(released)
            .saturating_sub(reserve);
        required <= available.min(physical)
            && index_bytes <= index_headroom.saturating_add(released)
    };
    if fits(0) {
        Some(false)
    } else {
        fits(releasable).then_some(true)
    }
}

/// Two amortized rebuild sizes for the oldest uncovered append-only runs.
/// Only spend a directory slot once the live lookup exceeds its bound.
pub(crate) fn append_tail_snapshot_candidates(
    indexed: u32,
    total: u32,
    snapshots: u32,
    limit: u32,
) -> [Option<u32>; 2] {
    const SNAPSHOT_SLOTS: u32 = 3;
    let tail = total - indexed;
    let searched = tail + snapshots;
    if searched <= limit || snapshots >= SNAPSHOT_SLOTS {
        return [None, None];
    }
    let minimum = (searched - limit + 1).max(2);
    // With no existing snapshot, the first attempt already tried the entire
    // directory. Try a smaller prefix rather than rebuilding it again.
    let largest = if indexed == 0 { tail / 2 } else { tail };
    if largest < minimum {
        return [None, None];
    }
    let smaller = (largest / 2).max(minimum);
    [Some(largest), (smaller < largest).then_some(smaller)]
}

#[cfg(test)]
#[test]
fn hash_snapshot_can_reuse_old_storage_only_if_both_limits_allow() {
    assert_eq!(
        snapshot_rebuild_admission(80, 40, 100, 40, 110, 20, 60),
        Some(false)
    );
    assert_eq!(
        snapshot_rebuild_admission(80, 40, 100, 39, 110, 20, 0),
        None
    );
    assert_eq!(
        snapshot_rebuild_admission(120, 40, 100, 20, 110, 20, 40),
        Some(true)
    );
    assert_eq!(
        snapshot_rebuild_admission(140, 40, 100, 40, 110, 20, 40),
        None
    );
    assert_eq!(
        snapshot_rebuild_admission(101, 40, 100, 40, 200, 20, 0),
        None
    );
    assert_eq!(snapshot_rebuild_admission(1, 1, 100, 1, 5, 20, 10), None);
}

#[cfg(test)]
#[test]
fn append_tail_snapshot_keeps_the_existing_prefix_and_reduces_the_lookup_to_its_cap() {
    assert_eq!(
        append_tail_snapshot_candidates(0, 9, 0, 8),
        [Some(4), Some(2)]
    );
    assert_eq!(
        append_tail_snapshot_candidates(5, 14, 1, 8),
        [Some(9), Some(4)]
    );
    assert_eq!(
        append_tail_snapshot_candidates(8, 15, 2, 8),
        [Some(7), Some(3)]
    );
    assert_eq!(append_tail_snapshot_candidates(8, 14, 2, 8), [None, None]);
    assert_eq!(append_tail_snapshot_candidates(8, 15, 3, 8), [None, None]);
}
