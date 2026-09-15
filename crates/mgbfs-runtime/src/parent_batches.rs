//! Allocation-free traversal of immutable physical frontier extents. A batch
//! never crosses a physical wrap. Peeking is not retirement or a read lease:
//! the caller must retain the corresponding StateRing range until GPU reads end.
use mgbfs_core::Result;
use mgbfs_cuda::native_owner::Extent;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParentBatch {
    pub extent: usize,
    pub offset: u64,
    pub begin: u64,
    pub sequence: u64,
    pub count: u32,
}

#[derive(Clone, Copy, Default)]
pub struct ParentCursor {
    extent: usize,
    offset: u64,
}
impl ParentCursor {
    pub fn peek(&self, extents: &[Extent], batch: u32) -> Result<Option<ParentBatch>> {
        if batch == 0 {
            return Err("PARENT_BATCH_ZERO".into());
        }
        let Some(e) = extents.get(self.extent) else {
            return Ok(None);
        };
        if self.offset >= e.count
            || e.begin.checked_add(e.count).is_none()
            || e.sequence.checked_add(e.count).is_none()
        {
            return Err("PARENT_EXTENT_RANGE".into());
        }
        Ok(Some(ParentBatch {
            extent: self.extent,
            offset: self.offset,
            begin: e.begin + self.offset,
            sequence: e.sequence + self.offset,
            count: u64::from(batch).min(e.count - self.offset) as u32,
        }))
    }
    pub fn take(&mut self, extents: &[Extent], batch: u32) -> Result<Option<ParentBatch>> {
        let next = self.peek(extents, batch)?;
        if let Some(b) = next {
            self.offset += u64::from(b.count);
            if self.offset == extents[self.extent].count {
                self.extent += 1;
                self.offset = 0;
            }
        }
        Ok(next)
    }
}
