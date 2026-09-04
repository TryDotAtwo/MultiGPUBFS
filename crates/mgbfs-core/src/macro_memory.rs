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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MacroLibraryBytes {
    pub generation: u64,
    pub candidate_hash: u64,
    pub archive_hash: u64,
    pub route: u64,
    pub materialize: u64,
    pub future_merge: u64,
    pub settle: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacroMemoryPlan {
    pub shape: MacroMemoryShape,
    pub external_bytes: u64,
    pub library_bytes: u64,
    pub requested_device_bytes: u64,
}

impl MacroMemoryPlan {
    /// Byte-exact payload contract for allocations owned by the macro runtime.
    /// CUDA allocator metadata/context overhead is deliberately outside this
    /// number and is guarded by the post-allocation reserve check.
    pub fn derive(input: MacroMemoryInput, library: MacroLibraryBytes) -> Result<Self> {
        let shape = MacroMemoryShape::derive(input)?;
        let library_bytes = [
            library.generation,
            library.candidate_hash,
            library.archive_hash,
            library.route,
            library.materialize,
            library.future_merge,
            library.settle,
        ]
        .into_iter()
        .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or("MACRO_LIBRARY_BYTE_OVERFLOW")?;
        let checked_mul = |a: u64, b: u64| a.checked_mul(b).ok_or("MACRO_BYTE_OVERFLOW");
        let checked_sum = |parts: &[u64]| {
            parts
                .iter()
                .try_fold(0u64, |sum, bytes| sum.checked_add(*bytes))
                .ok_or("MACRO_BYTE_OVERFLOW")
        };
        let future_slots = u64::from(shape.future_depth_slots);
        let external_bytes = checked_sum(&[
            checked_mul(input.layer_capacity, input.state_stride)?
                .checked_mul(2)
                .ok_or("MACRO_BYTE_OVERFLOW")?,
            checked_mul(input.layer_capacity, 16)?,
            8,
            shape.producer_state_bytes,
            shape.producer_hash_bytes,
            checked_mul(input.parent_batch, 16)?,
            checked_mul(input.route_slot_records, 32)?,
            4,
            checked_mul(input.future_capacity_per_depth, 24)?,
            4,
            16,
            shape.history_hash_bytes,
            checked_mul(u64::from(shape.history_layer_slots), 4)?,
            shape.future_state_bytes,
            checked_mul(shape.future_records, 16)?,
            checked_mul(future_slots, 8)?,
        ])?;
        let requested_device_bytes = external_bytes
            .checked_add(library_bytes)
            .ok_or("MACRO_TOTAL_BYTE_OVERFLOW")?;
        Ok(Self {
            shape,
            external_bytes,
            library_bytes,
            requested_device_bytes,
        })
    }
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
