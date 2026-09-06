//! Native nonblocking control connection. Bootstrap and dispatcher integration pending.
use crate::control_outbox::ControlOutbox;
use crate::control_wire::{Action, ControlFrame, FrameReader};
use mgbfs_core::Result;
use std::net::{Shutdown, TcpStream};
pub struct ControlConnection {
    stream: TcpStream,
    world: u32,
    started: bool,
    local: u32,
    peer: u32,
    reader: FrameReader,
    writer: ControlOutbox,
    failed: bool,
}
impl ControlConnection {
    pub(crate) fn identity(&self) -> (u32, u32, u32) {
        (self.world, self.local, self.peer)
    }
    pub(crate) fn prepare_send_capacity(&mut self, capacity: usize) -> Result<()> {
        self.alive()?;
        if self.started {
            return Err("CONTROL_ALREADY_STARTED".into());
        }
        self.writer = ControlOutbox::new(self.world, capacity)?;
        Ok(())
    }
    pub(crate) fn send_available(&self) -> Result<usize> {
        self.alive()?;
        Ok(self.writer.available())
    }
    pub(crate) fn abort(&mut self) {
        self.failed = true;
        let _ = self.stream.shutdown(Shutdown::Both);
    }
    /// Wrap only a stream assigned by the bootstrap handshake. Rank checking
    /// detects protocol mismatch, not cryptographic peer authentication.
    pub fn new(stream: TcpStream, world: u32, local: u32, peer: u32) -> Result<Self> {
        Self::with_send_capacity(stream, world, local, peer, 1)
    }
    /// Allocate the complete bounded outbound FIFO during connection setup.
    pub fn with_send_capacity(
        stream: TcpStream,
        world: u32,
        local: u32,
        peer: u32,
        capacity: usize,
    ) -> Result<Self> {
        if world == 0
            || local >= world
            || peer >= world
            || local == peer
            || (local != 0 && peer != 0)
        {
            return Err("CONTROL_TOPOLOGY".into());
        }
        stream
            .set_nonblocking(true)
            .map_err(|e| format!("CONTROL_NONBLOCKING: {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("CONTROL_NODELAY: {e}"))?;
        Ok(Self {
            stream,
            world,
            started: false,
            local,
            peer,
            reader: FrameReader::new(world)?,
            writer: ControlOutbox::new(world, capacity)?,
            failed: false,
        })
    }
    fn alive(&self) -> Result<()> {
        if self.failed {
            Err("CONTROL_CONNECTION_FAILED".into())
        } else {
            Ok(())
        }
    }
    fn finish<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() {
            self.failed = true;
            let _ = self.stream.shutdown(Shutdown::Both);
        }
        result
    }
    fn direction(frame: ControlFrame) -> Result<()> {
        let allowed = if frame.rank == 0 {
            matches!(
                frame.action,
                Action::Begin | Action::Finalize | Action::Publish | Action::Fatal
            )
        } else {
            matches!(
                frame.action,
                Action::Ready
                    | Action::Complete
                    | Action::Consumed
                    | Action::Finalized
                    | Action::SourceClosed
                    | Action::Fatal
            )
        };
        if allowed {
            Ok(())
        } else {
            Err("CONTROL_DIRECTION".into())
        }
    }
    pub fn enqueue(&mut self, frame: ControlFrame) -> Result<()> {
        self.alive()?;
        self.started = true;
        if frame.rank != self.local {
            return self.finish(Err("CONTROL_LOCAL_RANK".into()));
        }
        self.finish(Self::direction(frame))?;
        let result = self.writer.enqueue(frame);
        self.finish(result)
    }
    pub fn poll_send(&mut self) -> Result<bool> {
        self.alive()?;
        self.started = true;
        let result = self.writer.poll(&mut self.stream);
        self.finish(result)
    }
    pub fn poll_receive(&mut self) -> Result<Option<ControlFrame>> {
        self.alive()?;
        self.started = true;
        let result = self.reader.poll(&mut self.stream);
        let frame = self.finish(result)?;
        if let Some(frame) = frame {
            if frame.rank != self.peer {
                return self.finish(Err("CONTROL_PEER_RANK".into()));
            }
            self.finish(Self::direction(frame))?;
        }
        Ok(frame)
    }
}
