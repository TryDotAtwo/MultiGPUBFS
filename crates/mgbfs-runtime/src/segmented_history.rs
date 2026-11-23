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
pub struct StateAddress { segment: u32, bank: u8, row: u32 }

impl StateAddress {
    pub fn new(segment: u32, bank: u8, row: u64) -> Result<Self, &'static str> {
        if bank >= 3 || row >= LOCAL_ROWS { return Err("SEGMENTED_ADDRESS_RANGE"); }
        Ok(Self { segment, bank, row: row as u32 })
    }
    pub fn local_payload(self) -> u64 { (u64::from(self.bank) << BANK_SHIFT) | u64::from(self.row) }
    // Segment + bank + row need 66 bits; never truncate a global address to u64.
    pub fn encode(self) -> u128 { (u128::from(self.segment) << LOCAL_PAYLOAD_BITS) | u128::from(self.local_payload()) }
    pub fn decode(value: u128) -> Result<Self, &'static str> {
        if value >> (LOCAL_PAYLOAD_BITS + 32) != 0 { return Err("SEGMENTED_ADDRESS_RESERVED_BITS"); }
        let local = (value as u64) & LOCAL_PAYLOAD_MASK;
        Self::new((value >> LOCAL_PAYLOAD_BITS) as u32, (local >> BANK_SHIFT) as u8, local & (LOCAL_ROWS - 1))
    }
    pub fn segment(self) -> u32 { self.segment }
    pub fn bank(self) -> u8 { self.bank }
    pub fn row(self) -> u32 { self.row }
}

/// Uniform upper bounds; actual allocations and NCCL reserves are admitted separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentGeometry { segments: u32, rows_per_bank: u64, banks: u32 }
impl SegmentGeometry {
    pub fn new(segments: u32, rows_per_bank: u64, banks: u32) -> Result<Self, &'static str> {
        if segments == 0 || rows_per_bank == 0 || rows_per_bank > LOCAL_ROWS || !matches!(banks, 1 | 3) {
            return Err("SEGMENTED_GEOMETRY_RANGE");
        }
        Ok(Self { segments, rows_per_bank, banks })
    }
    pub fn rows(self) -> u128 { u128::from(self.segments) * u128::from(self.rows_per_bank) * u128::from(self.banks) }
    pub fn arena_bytes(self, state_bytes: u32) -> Result<u64, &'static str> {
        if state_bytes == 0 { return Err("SEGMENTED_ZERO_STATE_BYTES"); }
        let bytes = self.rows().checked_mul(u128::from(state_bytes)).ok_or("SEGMENTED_ARENA_BYTES_OVERFLOW")?;
        u64::try_from(bytes).map_err(|_| "SEGMENTED_ARENA_BYTES_OVERFLOW")
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
pub const BITMAP_PACK4_ABI_VERSION: u32 = 1;
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitmapPack4Dictionary {
    pub elements: u32,
    pub distinct: u32,
    pub symbols: [i64; 16],
}
impl BitmapPack4Dictionary {
    /// Canonical signed symbol ordering is independent of root position.
    /// The complete graph fingerprint, including multiplicities, remains a
    /// separate rank agreement; this dictionary only identifies the alphabet.
    pub fn from_root(root: &[i64]) -> Result<Self, &'static str> {
        if root.is_empty() || root.len() > 16 {
            return Err("BITMAP_PACK4_ELEMENTS_RANGE");
        }
        let mut distinct = root.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let mut symbols = [0i64; 16];
        symbols[..distinct.len()].copy_from_slice(&distinct);
        Ok(Self { elements: root.len() as u32, distinct: distinct.len() as u32, symbols })
    }
}

impl SegmentLayout {
    pub fn new(elements: u32, state_bytes: u32, rows_per_bank: u64, banks: u32) -> Result<Self, &'static str> {
        if elements == 0 || !matches!(state_bytes, 1 | 8) || rows_per_bank == 0 || rows_per_bank > LOCAL_ROWS || !matches!(banks, 1 | 3) {
            return Err("SEGMENTED_LAYOUT_RANGE");
        }
        let rows = rows_per_bank.checked_mul(u64::from(banks)).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let slots = rows.checked_mul(2).and_then(u64::checked_next_power_of_two).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let states = rows.checked_mul(u64::from(elements)).and_then(|v| v.checked_mul(u64::from(state_bytes))).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let positions = if banks == 3 { rows.checked_mul(8).ok_or("SEGMENTED_LAYOUT_OVERFLOW")? } else { 0 };
        let bytes = slots.checked_mul(8).and_then(|v| v.checked_add(states)).and_then(|v| v.checked_add(positions)).and_then(|v| v.checked_add(24)).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        Ok(Self { rows_per_bank, banks, slots, state_bytes: states, position_bytes: positions, position_item_bytes: if banks==3 {8} else {0}, device_bytes: bytes })
    }
    /// Use exact 32-bit positions only when every hash slot index fits u32.
    /// There is no sentinel in the positions array: it is read only for
    /// initialized live rows, so slot index u32::MAX remains representable.
    pub fn new_compact(elements:u32,state_bytes:u32,rows_per_bank:u64,banks:u32)->Result<Self,&'static str>{
        let mut layout=Self::new(elements,state_bytes,rows_per_bank,banks)?;
        if banks==3 && layout.slots <= (1u64<<32) {
            let saved=layout.position_bytes/2;
            layout.position_bytes-=saved;layout.device_bytes-=saved;layout.position_item_bytes=4;
        }
        Ok(layout)
    }
    /// Resident permutation hash: one exact dictionary-coded u64 per row,
    /// with the ordinary hash table and rolling-bank position metadata.
    pub fn hash_banks_pack4(elements:u32,state_bytes:u32,rows_per_bank:u64,banks:u32,compact:bool)->Result<Self,&'static str>{
        if elements==0||elements>16{return Err("HASH_PACK4_LAYOUT_RANGE")}
        let mut layout=if compact{Self::new_compact(elements,state_bytes,rows_per_bank,banks)?}
                       else{Self::new(elements,state_bytes,rows_per_bank,banks)?};
        let states=rows_per_bank.checked_mul(u64::from(banks))
            .and_then(|v|v.checked_mul(8)).ok_or("HASH_PACK4_LAYOUT_OVERFLOW")?;
        layout.device_bytes=layout.device_bytes.checked_sub(layout.state_bytes)
            .and_then(|v|v.checked_add(states)).ok_or("HASH_PACK4_LAYOUT_OVERFLOW")?;
        layout.state_bytes=states;
        Ok(layout)
    }
    /// One full state plus flat u64 key and u32 local row indexes per live
    /// row. Runs describe disjoint extents; states are never duplicated.
    pub fn sorted(elements:u32,state_bytes:u32,rows_per_bank:u64,banks:u32)->Result<Self,&'static str>{
        if elements==0||!matches!(state_bytes,1|8)||!matches!(banks,1|3)||rows_per_bank==0||rows_per_bank>LOCAL_ROWS{return Err("SEGMENTED_SORTED_LAYOUT_RANGE")}
        let rows=rows_per_bank.checked_mul(u64::from(banks)).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let states=rows.checked_mul(u64::from(elements)).and_then(|v|v.checked_mul(u64::from(state_bytes))).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let keys=rows.checked_mul(8).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let positions=rows.checked_mul(4).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        let bytes=states.checked_add(keys).and_then(|v|v.checked_add(positions)).and_then(|v|v.checked_add(24)).ok_or("SEGMENTED_LAYOUT_OVERFLOW")?;
        Ok(Self{rows_per_bank,banks,slots:rows,state_bytes:states,position_bytes:positions,position_item_bytes:4,device_bytes:bytes})
    }
    /// State-only storage for a bitmap backend; visited bits are allocated
    /// independently for the full rank-owned domain and must be admitted first.
    pub fn bitmap(elements:u32,state_bytes:u32,rows_per_bank:u64)->Result<Self,&'static str>{
        Self::bitmap_banks(elements,state_bytes,rows_per_bank,3)
    }
    pub fn bitmap_banks(elements:u32,state_bytes:u32,rows_per_bank:u64,banks:u32)->Result<Self,&'static str>{
        if elements==0||!matches!(state_bytes,1|8)||!matches!(banks,2|3)||rows_per_bank==0||rows_per_bank>LOCAL_ROWS{return Err("BITMAP_LAYOUT_RANGE")}
        let rows=rows_per_bank*u64::from(banks);
        let states=rows.checked_mul(u64::from(elements)).and_then(|v|v.checked_mul(u64::from(state_bytes))).ok_or("BITMAP_LAYOUT_OVERFLOW")?;
        Ok(Self{rows_per_bank,banks,slots:0,state_bytes:states,position_bytes:0,position_item_bytes:0,device_bytes:states.checked_add(24).ok_or("BITMAP_LAYOUT_OVERFLOW")?})
    }
    /// Experimental packed bitmap rows: one u64 per complete state.
    /// Selection requires the experimental CUDA capability and rank agreement;
    /// ordinary bitmap admission keeps its established resident layout.
    pub fn bitmap_banks_pack4(elements:u32,rows_per_bank:u64,banks:u32)->Result<Self,&'static str>{
        if elements==0||elements>16||!matches!(banks,2|3)||rows_per_bank==0||rows_per_bank>LOCAL_ROWS {
            return Err("BITMAP_PACK4_LAYOUT_RANGE");
        }
        let rows=rows_per_bank.checked_mul(u64::from(banks)).ok_or("BITMAP_PACK4_LAYOUT_OVERFLOW")?;
        let states=rows.checked_mul(8).ok_or("BITMAP_PACK4_LAYOUT_OVERFLOW")?;
        let device_bytes=states.checked_add(24).ok_or("BITMAP_PACK4_LAYOUT_OVERFLOW")?;
        Ok(Self{rows_per_bank,banks,slots:0,state_bytes:states,position_bytes:0,position_item_bytes:0,device_bytes})
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
        let rows = segments.checked_mul(self.rows_per_bank).ok_or("SEGMENTED_ADMISSION_OVERFLOW")?;
        let used = segments.checked_mul(self.device_bytes).and_then(|v| v.checked_add(fixed)).ok_or("SEGMENTED_ADMISSION_OVERFLOW")?;
        Ok(SegmentAdmission { segments: segments as u32, rows_per_bank: rows, device_bytes: used })
    }
}

/// Largest complete segment layout that fits an actual byte budget.
/// The cap describes one bank of one segment, never a rank-wide state limit.
/// Hash table slot rounding and bitmap state storage use the same constructors
/// as device allocations, so admission cannot underestimate these bytes.
pub fn fit_segment_layout(bitmap: bool, elements: u32, state_bytes: u32, banks: u32, max_rows: u64, budget: u64) -> Result<Option<SegmentLayout>, &'static str> {
    fit_segment_layout_compact(bitmap,elements,state_bytes,banks,max_rows,budget,false)
}

pub fn fit_segment_layout_compact(bitmap: bool, elements: u32, state_bytes: u32, banks: u32, max_rows: u64, budget: u64, compact:bool) -> Result<Option<SegmentLayout>, &'static str> {
    if max_rows == 0 || max_rows > LOCAL_ROWS { return Err("SEGMENTED_LAYOUT_RANGE"); }
    if bitmap && !matches!(banks,2|3) { return Err("BITMAP_LAYOUT_RANGE"); }
    let make = |rows| if bitmap { SegmentLayout::bitmap_banks(elements, state_bytes, rows, banks) } else if compact { SegmentLayout::new_compact(elements, state_bytes, rows, banks) } else { SegmentLayout::new(elements, state_bytes, rows, banks) };
    // Validate graph and storage shape even if budget is zero.
    make(1)?;
    let mut lo=0u64; let mut hi=max_rows;
    while lo<hi {
        let mid=lo+(hi-lo+1)/2;
        if make(mid).map_or(false, |v| v.device_bytes<=budget) { lo=mid; } else { hi=mid-1; }
    }
    if lo==0 { Ok(None) } else { Ok(Some(make(lo)?)) }
}

/// Experimental packed bitmap layout for an explicit budget. The maximum
/// applies to one bank of one segment, not to the sum of rows on a rank.
pub fn fit_packed_bitmap_segment_layout(elements:u32,banks:u32,max_rows:u64,budget:u64)->Result<Option<SegmentLayout>,&'static str>{
    if max_rows==0||max_rows>LOCAL_ROWS{return Err("BITMAP_PACK4_LAYOUT_RANGE")}
    let make=|rows|SegmentLayout::bitmap_banks_pack4(elements,rows,banks);
    make(1)?;
    let(mut lo,mut hi)=(0u64,max_rows);
    while lo<hi {
        let mid=lo+(hi-lo+1)/2;
        if make(mid).map_or(false,|layout|layout.device_bytes<=budget){lo=mid}else{hi=mid-1}
    }
    if lo==0{Ok(None)}else{Ok(Some(make(lo)?))}
}

/// Checked admission for packed resident hash segments.
pub fn fit_packed_hash_segment_layout(elements:u32,state_bytes:u32,banks:u32,max_rows:u64,budget:u64,compact:bool)->Result<Option<SegmentLayout>,&'static str>{
    if max_rows==0||max_rows>LOCAL_ROWS{return Err("HASH_PACK4_LAYOUT_RANGE")}
    let make=|rows|SegmentLayout::hash_banks_pack4(elements,state_bytes,rows,banks,compact);
    make(1)?;
    let(mut lo,mut hi)=(0u64,max_rows);
    while lo<hi{
        let mid=lo+(hi-lo+1)/2;
        if make(mid).map_or(false,|v|v.device_bytes<=budget){lo=mid}else{hi=mid-1}
    }
    if lo==0{Ok(None)}else{Ok(Some(make(lo)?))}
}

/// Largest sorted-index segment that fits a checked byte budget.
pub fn fit_sorted_segment_layout(elements:u32,state_bytes:u32,banks:u32,max_rows:u64,budget:u64)->Result<Option<SegmentLayout>,&'static str>{
    if max_rows==0||max_rows>LOCAL_ROWS{return Err("SEGMENTED_SORTED_LAYOUT_RANGE")}
    SegmentLayout::sorted(elements,state_bytes,1,banks)?;
    let(mut lo,mut hi)=(0u64,max_rows);
    while lo<hi{
        let mid=lo+(hi-lo+1)/2;
        if SegmentLayout::sorted(elements,state_bytes,mid,banks).map_or(false,|v|v.device_bytes<=budget){lo=mid}else{hi=mid-1}
    }
    if lo==0{Ok(None)}else{Ok(Some(SegmentLayout::sorted(elements,state_bytes,lo,banks)?))}
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

#[cfg(test)]
mod batch_layout_tests {
    #[test] fn sorted_index_checked_capacity_and_bytes(){
        use super::*;
        let x=SegmentLayout::sorted(14,1,100,3).unwrap();
        assert_eq!(x.state_bytes,4200);assert_eq!(x.slots,300);assert_eq!(x.position_bytes,1200);
        assert_eq!(x.device_bytes,3*100*(14+8+4)+24);
        assert_eq!(fit_sorted_segment_layout(14,1,3,100,x.device_bytes).unwrap().unwrap().rows_per_bank,100);
        assert_eq!(fit_sorted_segment_layout(14,1,3,100,x.device_bytes-1).unwrap().unwrap().rows_per_bank,99);
        let last=SegmentLayout::sorted(1,1,LOCAL_ROWS,1).unwrap();
        assert_eq!(last.rows_per_bank,LOCAL_ROWS);assert_eq!(last.device_bytes,13*LOCAL_ROWS+24);
        assert!(SegmentLayout::sorted(u32::MAX,8,LOCAL_ROWS,3).is_err());
    }
}

#[cfg(test)]
mod packed_bitmap_contract_tests {
    use super::*;

    #[test]
    fn descriptor_has_c_abi_sorted_symbols_and_checked_domain() {
        use std::mem::{align_of,size_of};
        let d=BitmapPack4Dictionary::from_root(&[17,-4,17,9,-4]).unwrap();
        assert_eq!(d.elements,5);
        assert_eq!(d.distinct,3);
        assert_eq!(&d.symbols[..3],&[-4,9,17]);
        assert_eq!(d,BitmapPack4Dictionary::from_root(&[9,-4,17,17,-4]).unwrap());
        assert_eq!(size_of::<BitmapPack4Dictionary>(),136);
        assert_eq!(align_of::<BitmapPack4Dictionary>(),8);
        assert_eq!(BITMAP_PACK4_ABI_VERSION,1);
        assert!(BitmapPack4Dictionary::from_root(&[]).is_err());
        assert!(BitmapPack4Dictionary::from_root(&[0;17]).is_err());
        let mut sixteen=[0i64;16];
        for (i,v) in sixteen.iter_mut().enumerate(){*v=i as i64-8;}
        let full=BitmapPack4Dictionary::from_root(&sixteen).unwrap();
        assert_eq!(full.elements,16);assert_eq!(full.distinct,16);
        assert_eq!(full.symbols[15],7);
    }

    #[test]
    fn packed_hash_layout_uses_exact_rows_and_checked_per_segment_capacity() {
        let ordinary=SegmentLayout::new_compact(14,1,100,3).unwrap();
        let packed=SegmentLayout::hash_banks_pack4(14,1,100,3,true).unwrap();
        assert_eq!(packed.state_bytes,100*3*8);
        assert_eq!(packed.slots,ordinary.slots);
        assert_eq!(packed.position_bytes,ordinary.position_bytes);
        assert_eq!(packed.device_bytes,ordinary.device_bytes-100*3*14+100*3*8);
        assert_eq!(fit_packed_hash_segment_layout(14,1,3,100,packed.device_bytes,true).unwrap().unwrap().rows_per_bank,100);
        assert_eq!(fit_packed_hash_segment_layout(14,1,3,100,packed.device_bytes-1,true).unwrap().unwrap().rows_per_bank,99);
        assert_eq!(fit_packed_hash_segment_layout(14,1,3,100,0,true).unwrap(),None);
        let boundary=SegmentLayout::hash_banks_pack4(16,8,LOCAL_ROWS,3,false).unwrap();
        assert_eq!(boundary.rows_per_bank,1u64<<32);
        assert_eq!(boundary.state_bytes,3*(1u64<<32)*8);
        assert_eq!(StateAddress::new(1,2,LOCAL_ROWS-1).unwrap().row(),u32::MAX);
        assert!(StateAddress::new(1,2,LOCAL_ROWS).is_err());
        assert!(SegmentLayout::hash_banks_pack4(16,8,LOCAL_ROWS+1,3,false).is_err());
        assert!(SegmentLayout::hash_banks_pack4(17,8,100,3,false).is_err());
    }

    #[test]
    fn packed_bitmap_layout_checked_with_per_segment_rows() {
        let x=SegmentLayout::bitmap_banks_pack4(14,100,2).unwrap();
        assert_eq!(x.state_bytes,1600); assert_eq!(x.device_bytes,1624);
        assert_eq!(x.position_bytes,0); assert_eq!(x.slots,0);
        assert_eq!(fit_packed_bitmap_segment_layout(14,2,100,1624).unwrap().unwrap().rows_per_bank,100);
        assert_eq!(fit_packed_bitmap_segment_layout(14,2,100,1623).unwrap().unwrap().rows_per_bank,99);
        assert_eq!(fit_packed_bitmap_segment_layout(14,2,100,39).unwrap(),None);
        let last=SegmentLayout::bitmap_banks_pack4(16,LOCAL_ROWS,2).unwrap();
        assert_eq!(last.rows_per_bank,1u64<<32);
        assert_eq!(last.state_bytes,(1u64<<32)*16);
        let admitted=last.admit(last.device_bytes*2,0,0,None).unwrap();
        assert_eq!(admitted.segments,2);
        assert_eq!(admitted.rows_per_bank,1u64<<33);
        assert!(SegmentLayout::bitmap_banks_pack4(16,LOCAL_ROWS+1,2).is_err());
        assert!(SegmentLayout::bitmap_banks_pack4(17,100,2).is_err());
        assert!(SegmentLayout::bitmap_banks_pack4(0,100,2).is_err());
        assert!(SegmentLayout::bitmap_banks_pack4(8,0,2).is_err());
        assert!(SegmentLayout::bitmap_banks_pack4(8,100,1).is_err());
    }
}
