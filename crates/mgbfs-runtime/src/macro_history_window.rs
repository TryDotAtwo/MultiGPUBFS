//! Host control metadata for fixed `2*Km` GPU hash-history slots.
//! This does not copy hashes or prove that a CUDA event completed: callers
//! signal `settled`, reader release, and archive D2H completion only after
//! their corresponding device events and global FinalizeDepth drain.
use mgbfs_core::Result;

#[derive(Clone, Copy, Default)]
struct Slot {
    depth: Option<u32>,
    readers: u64,
    archive_pending: bool,
}

pub struct MacroHistoryWindow {
    slots: Vec<Slot>,
    next_depth: u32,
    settled: bool,
}

impl MacroHistoryWindow {
    pub fn new(max_weight: u32) -> Result<Self> {
        let count = max_weight
            .checked_mul(2)
            .filter(|&v| v != 0)
            .ok_or("MACRO_HISTORY_WINDOW_SHAPE")?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(count as usize)
            .map_err(|_| "MACRO_HISTORY_WINDOW_CAPACITY")?;
        slots.resize(count as usize, Slot::default());
        Ok(Self {
            slots,
            next_depth: 0,
            settled: false,
        })
    }

    fn index(&self, depth: u32) -> usize {
        depth as usize % self.slots.len()
    }

    fn live(&mut self, depth: u32) -> Result<&mut Slot> {
        let index = self.index(depth);
        let slot = &mut self.slots[index];
        if slot.depth != Some(depth) {
            return Err("MACRO_HISTORY_STALE_DEPTH".into());
        }
        Ok(slot)
    }

    pub fn slot_depth(&self, slot: usize) -> Result<Option<u32>> {
        self.slots
            .get(slot)
            .map(|entry| entry.depth)
            .ok_or_else(|| "MACRO_HISTORY_SLOT_INDEX".into())
    }

    /// The caller has observed all owner/transport work for this exact depth
    /// complete. The old slot remains resident until `publish` succeeds.
    pub fn settled(&mut self, depth: u32) -> Result<()> {
        if depth != self.next_depth || self.settled {
            return Err("MACRO_HISTORY_DEPTH_ORDER".into());
        }
        self.settled = true;
        Ok(())
    }

    /// Publish the final hash layer after settlement. The slot it replaces
    /// held depth `depth - 2*Km`, which was still required during settlement.
    /// A live owner reader or archive D2H lease makes reuse illegal.
    pub fn publish(&mut self, depth: u32) -> Result<usize> {
        if depth != self.next_depth || !self.settled {
            return Err("MACRO_HISTORY_NOT_SETTLED".into());
        }
        let next = depth.checked_add(1).ok_or("MACRO_HISTORY_DEPTH_OVERFLOW")?;
        let index = self.index(depth);
        let old = self.slots[index];
        let expected = depth.checked_sub(self.slots.len() as u32);
        if old.depth != expected {
            return Err("MACRO_HISTORY_SLOT_GENERATION".into());
        }
        if old.readers != 0 || old.archive_pending {
            return Err("MACRO_HISTORY_SLOT_BUSY".into());
        }
        self.slots[index] = Slot {
            depth: Some(depth),
            readers: 0,
            archive_pending: true,
        };
        self.next_depth = next;
        self.settled = false;
        Ok(index)
    }

    /// Register a GPU owner read before enqueue; release only after its event.
    pub fn hold_reader(&mut self, depth: u32) -> Result<usize> {
        let index = self.index(depth);
        let slot = self.live(depth)?;
        slot.readers = slot
            .readers
            .checked_add(1)
            .ok_or("MACRO_HISTORY_READER_OVERFLOW")?;
        Ok(index)
    }

    pub fn release_reader(&mut self, depth: u32) -> Result<()> {
        let slot = self.live(depth)?;
        slot.readers = slot
            .readers
            .checked_sub(1)
            .ok_or("MACRO_HISTORY_READER_UNDERFLOW")?;
        Ok(())
    }

    /// D2H completion releases GPU storage; durable disk commit is separate.
    pub fn archive_copied(&mut self, depth: u32) -> Result<()> {
        let slot = self.live(depth)?;
        if !slot.archive_pending {
            return Err("MACRO_HISTORY_ARCHIVE_UNDERFLOW".into());
        }
        slot.archive_pending = false;
        Ok(())
    }
}
