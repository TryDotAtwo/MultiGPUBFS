use crate::Result;

#[derive(Debug, Clone, Copy)]
pub struct MacroMemoryInput {
    pub state_stride: u64,
    pub parent_batch: u64,
    pub macro_count: u64,
    pub effective_depth: u32,
    pub layer_capacity: u64,
    pub future_capacity_per_depth: u64,
    pub route_slot_records: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacroMemoryShape {
    pub candidate_records: u64,
    pub history_layer_slots: u32,
    pub history_hash_records: u64,
    pub future_depth_slots: u32,
    pub future_records: u64,
    pub producer_state_bytes: u64,
    pub producer_hash_bytes: u64,
    pub future_state_bytes: u64,
    pub future_hash_ref_bytes: u64,
    pub history_hash_bytes: u64,
}

impl MacroMemoryShape {
    pub fn derive(input: MacroMemoryInput) -> Result<Self> {
        if input.state_stride == 0
            || input.parent_batch == 0
            || input.macro_count == 0
            || input.effective_depth == 0
            || input.layer_capacity == 0
            || input.future_capacity_per_depth == 0
        {
            return Err("MACRO_MEMORY_SHAPE".into());
        }
        let candidate_records = input
            .parent_batch
            .checked_mul(input.macro_count)
            .ok_or("MACRO_CANDIDATE_OVERFLOW")?;
        if candidate_records > input.route_slot_records {
            return Err("MACRO_ROUTE_CAPACITY".into());
        }
        let history_layer_slots = input
            .effective_depth
            .checked_mul(2)
            .ok_or("MACRO_HISTORY_SLOT_OVERFLOW")?;
        let history_hash_records = u64::from(history_layer_slots)
            .checked_mul(input.layer_capacity)
            .ok_or("MACRO_HISTORY_OVERFLOW")?;
        let future_depth_slots = input.effective_depth;
        let future_records = u64::from(future_depth_slots)
            .checked_mul(input.future_capacity_per_depth)
            .ok_or("MACRO_FUTURE_OVERFLOW")?;
        let bytes =
            |count: u64, stride: u64| count.checked_mul(stride).ok_or("MACRO_BYTE_OVERFLOW");
        Ok(Self {
            candidate_records,
            history_layer_slots,
            history_hash_records,
            future_depth_slots,
            future_records,
            producer_state_bytes: bytes(candidate_records, input.state_stride)?
                .checked_mul(2)
                .ok_or("MACRO_BYTE_OVERFLOW")?,
            producer_hash_bytes: bytes(candidate_records, 16)?
                .checked_mul(2)
                .ok_or("MACRO_BYTE_OVERFLOW")?,
            future_state_bytes: bytes(future_records, input.state_stride)?,
            future_hash_ref_bytes: bytes(future_records, 24)?,
            history_hash_bytes: bytes(history_hash_records, 16)?,
        })
    }
}
