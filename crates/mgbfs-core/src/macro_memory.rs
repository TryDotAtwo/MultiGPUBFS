use crate::Result;

/// Preflight layout for provisional macro-depth owner buckets. Each
/// `(target_depth mod macro_depth, bucket)` gets one contiguous fixed extent;
/// the configured capacities sum to the physically reserved future arena.
/// This is a storage contract, not a CPU-side runtime allocator or owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutureBucketLayout {
    macro_depth: u32,
    buckets: u32,
    offsets: Vec<u64>,
    capacities: Vec<u32>,
    pub hash_records: u64,
    pub hash_bytes: u64,
    pub metadata_bytes: u64,
}

impl FutureBucketLayout {
    pub fn derive(
        macro_depth: u32,
        buckets: u32,
        bucket_limit: u32,
        capacities: &[u32],
        future_records: u64,
    ) -> Result<Self> {
        if macro_depth == 0 || buckets == 0 || bucket_limit == 0 {
            return Err("MACRO_FUTURE_BUCKET_SHAPE".into());
        }
        let len = u64::from(macro_depth)
            .checked_mul(u64::from(buckets))
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("MACRO_FUTURE_BUCKET_OVERFLOW")?;
        if capacities.len() != len || capacities.iter().any(|&v| v > bucket_limit) {
            return Err("MACRO_FUTURE_BUCKET_CAPACITY".into());
        }
        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(len.checked_add(1).ok_or("MACRO_FUTURE_BUCKET_OVERFLOW")?)
            .map_err(|_| "MACRO_FUTURE_BUCKET_ALLOCATION")?;
        offsets.push(0u64);
        for &cap in capacities {
            offsets.push(
                offsets
                    .last()
                    .copied()
                    .unwrap()
                    .checked_add(u64::from(cap))
                    .ok_or("MACRO_FUTURE_BUCKET_OVERFLOW")?,
            );
        }
        if future_records == 0 || offsets[len] != future_records {
            return Err("MACRO_FUTURE_RECORD_BUDGET".into());
        }
        let hash_bytes = future_records
            .checked_mul(16)
            .ok_or("MACRO_FUTURE_BUCKET_OVERFLOW")?;
        let metadata_bytes = u64::try_from(len)
            .ok()
            .and_then(|v| v.checked_mul(4))
            .and_then(|count_bytes| {
                u64::try_from(len + 1)
                    .ok()
                    .and_then(|v| v.checked_mul(8))
                    .and_then(|offset_bytes| count_bytes.checked_add(offset_bytes))
            })
            .ok_or("MACRO_FUTURE_BUCKET_OVERFLOW")?;
        Ok(Self {
            macro_depth,
            buckets,
            offsets,
            capacities: capacities.to_vec(),
            hash_records: future_records,
            hash_bytes,
            metadata_bytes,
        })
    }

    /// Upload these immutable slot-local directories once before depth zero.
    /// The terminal offset is included for device-side prefix validation.
    pub fn slot_directory(&self, slot: u32) -> Result<(&[u64], &[u32])> {
        if slot >= self.macro_depth {
            return Err("MACRO_FUTURE_BUCKET_INDEX".into());
        }
        let begin = slot as usize * self.buckets as usize;
        let end = begin + self.buckets as usize;
        Ok((&self.offsets[begin..=end], &self.capacities[begin..end]))
    }

    pub fn bucket_range(&self, slot: u32, bucket: u32) -> Result<(u64, u32)> {
        if slot >= self.macro_depth || bucket >= self.buckets {
            return Err("MACRO_FUTURE_BUCKET_INDEX".into());
        }
        let index = (slot as usize) * (self.buckets as usize) + bucket as usize;
        Ok((
            self.offsets[index],
            u32::try_from(self.offsets[index + 1] - self.offsets[index])
                .map_err(|_| "MACRO_FUTURE_BUCKET_OVERFLOW")?,
        ))
    }

    /// One active window uses each physical slot exactly once. Reuse of a
    /// slot after advancing current depth requires an external drain/reset.
    pub fn target_bucket(
        &self,
        current_depth: u32,
        target_depth: u32,
        bucket: u32,
    ) -> Result<(u64, u32)> {
        let delta = target_depth
            .checked_sub(current_depth)
            .ok_or("MACRO_FUTURE_DEPTH")?;
        if delta == 0 || delta > self.macro_depth {
            return Err("MACRO_FUTURE_DEPTH".into());
        }
        self.bucket_range(target_depth % self.macro_depth, bucket)
    }
}

/// Additional fixed GPU allocations required by each rank's DENSE macro
/// owner exchange. CUDA/NCCL context-internal memory is guarded separately by
/// the untouched VRAM reserve; this counts every explicit cudaMalloc request.
#[derive(Debug, Clone, Copy)]
pub struct MacroExchangeInput {
    pub world: u32,
    pub candidate_capacity: u32,
    pub state_stride: u64,
    pub route_slots: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacroExchangeMemoryPlan {
    pub send_frame_bytes_per_slot: u64,
    pub receive_frame_bytes_per_slot: u64,
    pub owner_count_bytes_per_slot: u64,
    pub count_exchange_bytes_per_slot: u64,
    pub slot_bytes: u64,
    pub failure_collective_bytes: u64,
    pub requested_device_bytes: u64,
}

impl MacroExchangeMemoryPlan {
    pub fn derive(input: MacroExchangeInput) -> Result<Self> {
        if !matches!(input.world, 1 | 2 | 4 | 8)
            || input.candidate_capacity == 0
            || input.candidate_capacity > i32::MAX as u32
            || input.state_stride == 0
            || input.state_stride % 16 != 0
            || input.route_slots == 0
        {
            return Err("MACRO_EXCHANGE_SHAPE".into());
        }
        let rows = u64::from(input.candidate_capacity);
        let record_bytes = input.state_stride.checked_add(32).ok_or("MACRO_EXCHANGE_BYTES")?;
        let payload = rows.checked_mul(record_bytes).ok_or("MACRO_EXCHANGE_BYTES")?;
        // Each of three 16-byte-aligned planes pads by at most 240 bytes;
        // the versioned frame prefix adds 256 more. Empty peers send no frame.
        const FRAME_OVERHEAD: u64 = 256 + 3 * 240;
        let send_frame = payload
            .checked_add(u64::from(input.world).checked_mul(FRAME_OVERHEAD).ok_or("MACRO_EXCHANGE_BYTES")?)
            .ok_or("MACRO_EXCHANGE_BYTES")?;
        let receive_frame = payload.checked_add(FRAME_OVERHEAD).ok_or("MACRO_EXCHANGE_BYTES")?;
        let owners = u64::from(input.world).checked_mul(4).ok_or("MACRO_EXCHANGE_BYTES")?;
        let count_exchange = 8u64; // one u32 sent, one u32 received
        let slot_bytes = send_frame.checked_add(receive_frame)
            .and_then(|sum| sum.checked_add(owners))
            .and_then(|sum| sum.checked_add(count_exchange))
            .ok_or("MACRO_EXCHANGE_BYTES")?;
        let failure_collective_bytes = 8u64; // one u32 send and receive
        let requested_device_bytes = slot_bytes
            .checked_mul(u64::from(input.route_slots))
            .and_then(|sum| sum.checked_add(failure_collective_bytes))
            .ok_or("MACRO_EXCHANGE_BYTES")?;
        Ok(Self {
            send_frame_bytes_per_slot: send_frame,
            receive_frame_bytes_per_slot: receive_frame,
            owner_count_bytes_per_slot: owners,
            count_exchange_bytes_per_slot: count_exchange,
            slot_bytes,
            failure_collective_bytes,
            requested_device_bytes,
        })
    }
}

/// Storage contract shared by the weighted runtime and its archive producer.
/// Compact generation applies permutation matrices directly to these vectors.
pub struct MacroStateLayout {
    pub start: Vec<u8>,
    pub width: usize,
    pub stride: usize,
}
impl MacroStateLayout {
    pub fn derive(graph: &crate::matrix::MatrixGroup, generation: u32) -> Result<Self> {
        graph.validate()?;
        if generation > 5 {
            return Err("MACRO_GENERATION_BACKEND".into());
        }
        let start = if generation == 5 {
            for generator in &graph.generators {
                crate::matrix::encode_permutation_matrix(generator, graph.rows)?;
            }
            crate::matrix::encode_permutation_matrix(&graph.start, graph.rows)?
        } else {
            graph.start.clone()
        };
        let width = start.len();
        Ok(Self {
            start,
            width,
            stride: (width + 15) & !15,
        })
    }
}

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
