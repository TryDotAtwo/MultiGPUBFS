//! Fixed-capacity FIFO, including the frame currently being written.
use crate::control_wire::{ControlFrame, FrameWriter};
use mgbfs_core::Result;
use std::io::Write;

pub struct ControlOutbox {
    frames: Vec<Option<ControlFrame>>,
    head: usize,
    len: usize,
    writing: bool,
    failed: bool,
    world: u32,
    writer: FrameWriter,
}
impl ControlOutbox {
    pub(crate) fn available(&self) -> usize {
        self.frames.len() - self.len
    }
    pub fn new(world: u32, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err("CONTROL_SEND_CAPACITY".into());
        }
        let writer = FrameWriter::new(world)?;
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(capacity)
            .map_err(|_| "CONTROL_SEND_CAPACITY")?;
        frames.resize(capacity, None);
        Ok(Self {
            frames,
            head: 0,
            len: 0,
            writing: false,
            failed: false,
            world,
            writer,
        })
    }
    pub fn enqueue(&mut self, frame: ControlFrame) -> Result<()> {
        if self.failed {
            return Err("CONTROL_OUTBOX_FAILED".into());
        }
        if self.len == self.frames.len() {
            return Err("CONTROL_SEND_CAPACITY".into());
        }
        frame.encode(self.world)?;
        let remaining = self.frames.len() - self.head;
        let tail = if self.len >= remaining {
            self.len - remaining
        } else {
            self.head + self.len
        };
        self.frames[tail] = Some(frame);
        self.len += 1;
        Ok(())
    }
    /// Makes progress on at most one frame per call, preserving peer fairness.
    /// True means the local FIFO is empty, not that peer consumers have finished.
    pub fn poll(&mut self, stream: &mut impl Write) -> Result<bool> {
        if self.failed {
            return Err("CONTROL_OUTBOX_FAILED".into());
        }
        let result = self.poll_inner(stream);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn poll_inner(&mut self, stream: &mut impl Write) -> Result<bool> {
        if self.len == 0 {
            return Ok(true);
        }
        if !self.writing {
            self.writer.enqueue(self.frames[self.head].unwrap())?;
            self.writing = true;
        }
        if !self.writer.poll(stream)? {
            return Ok(false);
        }
        self.frames[self.head] = None;
        self.head += 1;
        if self.head == self.frames.len() {
            self.head = 0;
        }
        self.len -= 1;
        self.writing = false;
        Ok(self.len == 0)
    }
}
