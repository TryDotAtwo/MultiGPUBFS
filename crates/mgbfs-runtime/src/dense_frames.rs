//! Fixed host descriptors for rank-contiguous DENSE payloads. Contains no
//! states; successful preparation performs no allocations or device reads.
use mgbfs_core::Result;
/// Transport envelope: 64-byte schema2 header followed by zero padding, then
/// the existing aligned payload planes. Empty destinations remain zero bytes.
pub const DENSE_FRAME_PREFIX_BYTES: u64 = 256;

#[derive(Clone, Copy, Default)]
pub struct DenseFrame {
    pub begin: u32,
    pub count: u32,
    pub offset: u64,
    pub bytes: u64,
}
pub struct DenseFrames {
    owner_to_rank: Vec<u32>,
    frames: Vec<DenseFrame>,
    sizes: Vec<u64>,
    stride: u32,
    max_records: u32,
    capacity: u64,
    prepared: bool,
    failed: bool,
}
impl DenseFrames {
    pub fn new(map: &[u32], stride: u32, max_records: u32, capacity: u64) -> Result<Self> {
        if map.is_empty() || map.len() > 128 || stride == 0 || stride % 16 != 0 {
            return Err("DENSE_FRAME_CONFIG".into());
        }
        for (i, &rank) in map.iter().enumerate() {
            if rank as usize >= map.len() || map[..i].contains(&rank) {
                return Err("DENSE_FRAME_RANK_MAP".into());
            }
        }
        let mut owner_to_rank = Vec::new();
        let mut frames = Vec::new();
        let mut sizes = Vec::new();
        owner_to_rank
            .try_reserve_exact(map.len())
            .map_err(|_| "DENSE_FRAME_ALLOC")?;
        frames
            .try_reserve_exact(map.len())
            .map_err(|_| "DENSE_FRAME_ALLOC")?;
        sizes
            .try_reserve_exact(map.len())
            .map_err(|_| "DENSE_FRAME_ALLOC")?;
        owner_to_rank.extend_from_slice(map);
        frames.resize(map.len(), DenseFrame::default());
        sizes.resize(map.len(), 0);
        Ok(Self {
            owner_to_rank,
            frames,
            sizes,
            stride,
            max_records,
            capacity,
            prepared: false,
            failed: false,
        })
    }
    /// Counts describe logical-owner segments in the already sorted input.
    /// Result frames and byte counts are in physical rank order for NCCL.
    pub fn prepare(&mut self, counts: &[u32]) -> Result<()> {
        if self.failed {
            return Err("DENSE_FRAME_FAILED".into());
        }
        self.prepared = false;
        let result = self.prepare_inner(counts);
        self.failed = result.is_err();
        self.prepared = result.is_ok();
        result
    }
    fn prepare_inner(&mut self, counts: &[u32]) -> Result<()> {
        if counts.len() != self.frames.len() {
            return Err("DENSE_FRAME_COUNTS".into());
        }
        let mut begin = 0u32;
        for (owner, &count) in counts.iter().enumerate() {
            let end = begin
                .checked_add(count)
                .ok_or("DENSE_FRAME_RECORD_OVERFLOW")?;
            if end > self.max_records {
                return Err("DENSE_FRAME_RECORD_CAPACITY".into());
            }
            let payload = mgbfs_core::wire::payload_bytes(
                mgbfs_core::wire::FrameKind::Dense,
                count,
                u64::from(self.stride),
            )?;
            let bytes = if count == 0 {
                0
            } else {
                payload
                    .checked_add(DENSE_FRAME_PREFIX_BYTES)
                    .ok_or("DENSE_FRAME_BYTE_OVERFLOW")?
            };
            let rank = self.owner_to_rank[owner] as usize;
            self.frames[rank] = DenseFrame {
                begin,
                count,
                offset: 0,
                bytes,
            };
            self.sizes[rank] = bytes;
            begin = end;
        }
        let mut offset = 0u64;
        for frame in &mut self.frames {
            frame.offset = offset;
            offset = offset
                .checked_add(frame.bytes)
                .ok_or("DENSE_FRAME_BYTE_OVERFLOW")?;
        }
        if offset > self.capacity {
            return Err("DENSE_FRAME_BYTE_CAPACITY".into());
        }
        Ok(())
    }
    pub fn frames(&self) -> Result<&[DenseFrame]> {
        if !self.prepared || self.failed {
            return Err("DENSE_FRAME_NOT_PREPARED".into());
        }
        Ok(&self.frames)
    }
    pub fn sizes(&self) -> Result<&[u64]> {
        self.frames()?;
        Ok(&self.sizes)
    }
    /// Fill caller-preallocated headers after BEGIN assigns the ticket epoch.
    /// These describe the prefix separate from the GPU payload planes. The caller must bind the
    /// matching source bank, deliver/validate headers before consuming payload,
    /// and retain any pinned header storage through its transfer completion.
    /// This method encodes metadata; it does not send it or establish readiness.
    pub fn encode_headers(
        &self,
        key: crate::scatter_admission::TicketKey,
        run_tag: u64,
        out: &mut [[u8; 64]],
    ) -> Result<()> {
        use mgbfs_core::wire::{FrameHeader, FrameKind};
        self.frames()?;
        if key.plane != crate::control_wire::Plane::Candidate
            || key.source as usize >= self.frames.len()
            || out.len() != self.frames.len()
            || key.generation == crate::control_wire::NO_SLOT
        {
            return Err("DENSE_FRAME_HEADER_TICKET".into());
        }
        let depth = u32::try_from(key.depth).map_err(|_| "DENSE_FRAME_HEADER_DEPTH")?;
        for (rank, (frame, bytes)) in self.frames.iter().zip(out).enumerate() {
            *bytes = FrameHeader {
                kind: FrameKind::Dense,
                run_tag,
                sequence: key.epoch,
                batch: key.generation,
                depth,
                source: key.source,
                destination: rank as u32,
                count: frame.count,
            }
            .encode(u64::from(self.stride))?;
        }
        Ok(())
    }
    /// Bind headers to the assigned ticket, then enqueue their prefix writes.
    /// Kernel arguments copy each 64-byte host header before the call returns.
    ///
    /// # Safety
    /// Output is the same live, aligned source slot passed to enqueue_native(),
    /// with at least configured capacity bytes. The key must belong to this
    /// source bank. Enqueue on the communication stream before its NCCL scatter,
    /// after payload packing readiness has been established. No readers may
    /// access these prefixes yet. Retain storage through all native consumers.
    #[cfg(feature = "cuda")]
    pub unsafe fn enqueue_headers_native(
        &mut self,
        key: crate::scatter_admission::TicketKey,
        run_tag: u64,
        output: *mut u8,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        let result = (|| {
            let mut headers = [[0u8; 64]; 128]; // Bounded stack storage, not a heap allocation.
            self.encode_headers(key, run_tag, &mut headers[..self.frames.len()])?;
            if output.is_null() && self.frames.iter().any(|f| f.count != 0) {
                return Err("DENSE_FRAME_NATIVE_POINTER".into());
            }
            for (frame, header) in self.frames.iter().zip(&headers) {
                if frame.count == 0 {
                    continue;
                }
                let status = mgbfs_cuda::ffi::mgbfs_frame_write_header(
                    header.as_ptr(),
                    output.add(frame.offset as usize),
                    stream,
                );
                if status != 0 {
                    return Err(format!("DENSE_FRAME_HEADER_NATIVE_{status}"));
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
            self.prepared = false;
        }
        result
    }
    /// Enqueue all destination frames directly into one admitted source slot.
    /// No allocation, host synchronization, or publication of READY occurs.
    /// Prefixes are deliberately left untouched until enqueue_headers_native()
    /// binds the later assigned ticket. Never scatter an unbound frame.
    ///
    /// # Safety
    /// Input arrays must be live device storage for source_count children and
    /// the prepared sorted record count, respectively. Output must be a disjoint
    /// 256-byte-aligned allocation of at least the configured capacity bytes.
    /// Keep all storage live until the stream completes. The caller must check
    /// the sticky device fatal flag after packing before advertising sizes().
    /// On an enqueue error stop the rank group and drain/abort before freeing.
    #[cfg(feature = "cuda")]
    pub unsafe fn enqueue_native(
        &mut self,
        states: *const u8,
        source_count: u32,
        hashes: *const std::ffi::c_void,
        refs: *const u64,
        output: *mut u8,
        fatal: *mut u32,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        let result = (|| {
            self.frames()?;
            let sorted_count: u32 = self.frames.iter().map(|f| f.count).sum();
            if source_count > self.max_records
                || sorted_count > source_count
                || (sorted_count != 0 && output.is_null())
            {
                return Err("DENSE_FRAME_NATIVE_CAPACITY".into());
            }
            for frame in &self.frames {
                if frame.count == 0 {
                    continue;
                }
                let status = mgbfs_cuda::ffi::mgbfs_exchange_pack_frame(
                    self.stride,
                    states,
                    source_count,
                    hashes,
                    refs,
                    sorted_count,
                    frame.begin,
                    frame.count,
                    output.add((frame.offset + DENSE_FRAME_PREFIX_BYTES) as usize),
                    frame.bytes - DENSE_FRAME_PREFIX_BYTES,
                    fatal,
                    stream,
                );
                if status != 0 {
                    return Err(format!("DENSE_FRAME_NATIVE_{status}"));
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
            self.prepared = false;
        }
        result
    }
}
