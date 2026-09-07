//! Fixed host descriptors for rank-contiguous DENSE payloads. Contains no
//! states; successful preparation performs no allocations or device reads.
use mgbfs_core::Result;

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
            let mut bytes = 0u64;
            for width in [16, 4, u64::from(self.stride)] {
                let plane = u64::from(count)
                    .checked_mul(width)
                    .and_then(|n| n.checked_add(255))
                    .ok_or("DENSE_FRAME_BYTE_OVERFLOW")?
                    & !255;
                bytes = bytes
                    .checked_add(plane)
                    .ok_or("DENSE_FRAME_BYTE_OVERFLOW")?;
            }
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
    /// Enqueue all destination frames directly into one admitted source slot.
    /// No allocation, host synchronization, or publication of READY occurs.
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
                    output.add(frame.offset as usize),
                    frame.bytes,
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
