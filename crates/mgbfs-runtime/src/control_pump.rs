//! Nonblocking host control loop. GPU commands are returned to the caller;
//! this module does not observe CUDA events or execute NCCL itself.
use crate::{
    control_connection::ControlConnection,
    control_wire::{Action, ControlFrame, Plane, NO_SLOT},
    epoch_coordinator::EpochCoordinator,
    rank_epochs::RankEpochs,
};
use mgbfs_core::Result;
use std::collections::VecDeque;
pub struct ControlPump {
    rank: u32,
    depth: u64,
    local: RankEpochs,
    coordinator: Option<EpochCoordinator>,
    peers: Vec<Option<ControlConnection>>,
    outgoing: Vec<ControlFrame>,
    commands: VecDeque<ControlFrame>,
    command_capacity: usize,
    finalization: Option<ControlFrame>,
    source_closed: bool,
    failed: bool,
}
impl ControlPump {
    pub fn new(
        world: u32,
        rank: u32,
        slots: usize,
        mut peers: Vec<Option<ControlConnection>>,
    ) -> Result<Self> {
        if world == 0 || rank >= world || peers.len() != world as usize {
            return Err("CONTROL_PUMP_TOPOLOGY".into());
        }
        let command_capacity = slots
            .checked_mul(4)
            .and_then(|x| x.checked_add(1))
            .ok_or("CONTROL_PUMP_CAPACITY")?;
        let send_capacity = slots
            .checked_mul(12)
            .and_then(|x| x.checked_add(2))
            .ok_or("CONTROL_PUMP_CAPACITY")?;
        for (index, peer) in peers.iter_mut().enumerate() {
            let required = if rank == 0 { index != 0 } else { index == 0 };
            if peer.is_some() != required {
                return Err("CONTROL_PUMP_TOPOLOGY".into());
            }
            if let Some(peer) = peer {
                if peer.identity() != (world, rank, index as u32) {
                    return Err("CONTROL_PUMP_TOPOLOGY".into());
                }
                peer.prepare_send_capacity(send_capacity)?;
            }
        }
        let local = RankEpochs::new(world, rank, slots)?;
        let coordinator = if rank == 0 {
            Some(EpochCoordinator::new(world, slots)?)
        } else {
            None
        };
        let mut commands = VecDeque::new();
        commands
            .try_reserve_exact(command_capacity)
            .map_err(|_| "CONTROL_PUMP_CAPACITY")?;
        let mut outgoing = Vec::new();
        outgoing
            .try_reserve_exact(world as usize)
            .map_err(|_| "CONTROL_PUMP_CAPACITY")?;
        outgoing.resize(
            world as usize,
            ControlFrame {
                action: Action::Finalize,
                rank: 0,
                depth: 0,
                epoch: 0,
                slot: NO_SLOT,
                plane: Plane::None,
                source_rank: 0,
                fatal_code: 0,
                destination_rank: 0,
                payload_bytes: 0,
            },
        );
        Ok(Self {
            rank,
            depth: 0,
            local,
            coordinator,
            peers,
            outgoing,
            commands,
            command_capacity,
            finalization: None,
            source_closed: false,
            failed: false,
        })
    }
    fn apply<T>(&mut self, action: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.failed {
            return Err("CONTROL_PUMP_FAILED".into());
        }
        let result = action(self);
        if result.is_err() {
            self.failed = true;
            for peer in self.peers.iter_mut().flatten() {
                peer.abort();
            }
        }
        result
    }
    fn emit(&mut self, frame: ControlFrame) -> Result<()> {
        if let Some(coordinator) = &mut self.coordinator {
            coordinator.receive(frame)
        } else {
            self.peers[0].as_mut().unwrap().enqueue(frame)
        }
    }
    pub fn offer(&mut self, plane: Plane, slot: u64) -> Result<()> {
        self.apply(|s| {
            if s.source_closed && plane == Plane::Candidate {
                return Err("CONTROL_SOURCE_CLOSED".into());
            }
            let frame = s.local.offer(plane, slot)?;
            s.emit(frame)
        })
    }
    pub fn transfer_complete(&mut self, epoch: u64) -> Result<()> {
        self.apply(|s| {
            let frame = s.local.transfer_complete(epoch)?;
            s.emit(frame)
        })
    }
    /// Descendant offers must be enqueued before retiring their input epoch.
    pub fn consumed(&mut self, epoch: u64) -> Result<()> {
        self.apply(|s| {
            let frame = s.local.consume(epoch)?;
            s.emit(frame)
        })
    }
    pub fn close_source(&mut self) -> Result<()> {
        self.apply(|s| {
            if s.source_closed {
                return Err("CONTROL_SOURCE_CLOSE".into());
            }
            s.source_closed = true;
            s.emit(ControlFrame {
                action: Action::SourceClosed,
                rank: s.rank,
                depth: s.depth,
                epoch: 0,
                slot: NO_SLOT,
                plane: Plane::None,
                source_rank: 0,
                fatal_code: 0,
                destination_rank: 0,
                payload_bytes: 0,
            })
        })
    }
    /// Called after the caller has finished and verified local FinalizeDepth jobs.
    pub fn finalized(&mut self, drained: bool) -> Result<()> {
        self.apply(|s| {
            let request = s.finalization.ok_or("CONTROL_FINALIZATION_MISSING")?;
            let frame = s.local.finish_depth(request, drained)?;
            s.emit(frame)
        })
    }
    pub fn command(&mut self) -> Result<Option<ControlFrame>> {
        self.apply(|s| Ok(s.commands.pop_front()))
    }
    fn dispatch(&mut self, frame: ControlFrame) -> Result<()> {
        if self.commands.len() == self.command_capacity {
            return Err("CONTROL_COMMAND_CAPACITY".into());
        }
        match frame.action {
            Action::Begin => {
                if self.finalization.is_some() {
                    return Err("CONTROL_FINALIZATION_ACTIVE".into());
                }
                self.local.begin(frame)?;
            }
            Action::Finalize => {
                if self.finalization.is_some() || !self.local.drained() {
                    return Err("CONTROL_FINALIZATION_NOT_DRAINED".into());
                }
                self.finalization = Some(frame);
            }
            Action::Publish => {
                self.local.publish(frame)?;
                self.depth = frame.depth;
                self.finalization = None;
                self.source_closed = false;
            }
            Action::Fatal => {
                return Err(format!(
                    "CONTROL_REMOTE_FATAL rank={} code={}",
                    frame.rank, frame.fatal_code
                ))
            }
            _ => return Err("CONTROL_PUMP_COMMAND".into()),
        }
        self.commands.push_back(frame);
        Ok(())
    }
    /// At most one incoming frame per peer and one globally issued command group.
    pub fn poll(&mut self) -> Result<()> {
        self.apply(Self::poll_inner)
    }
    /// Poll within a caller-owned deadline shared across repeated calls.
    pub fn poll_before(&mut self, deadline: std::time::Instant) -> Result<()> {
        self.apply(|s| {
            if std::time::Instant::now() >= deadline {
                return Err("CONTROL_PROGRESS_TIMEOUT".into());
            }
            s.poll_inner()?;
            if std::time::Instant::now() >= deadline {
                return Err("CONTROL_PROGRESS_TIMEOUT".into());
            }
            Ok(())
        })
    }
    fn poll_inner(&mut self) -> Result<()> {
        for index in 0..self.peers.len() {
            let frame = if let Some(peer) = &mut self.peers[index] {
                peer.poll_send()?;
                peer.poll_receive()?
            } else {
                None
            };
            if let Some(frame) = frame {
                if frame.action == Action::Fatal {
                    return Err(format!(
                        "CONTROL_REMOTE_FATAL rank={} code={}",
                        frame.rank, frame.fatal_code
                    ));
                }
                if let Some(coordinator) = &mut self.coordinator {
                    coordinator.receive(frame)?;
                } else {
                    self.dispatch(frame)?;
                }
            }
        }
        if self.coordinator.is_some() && self.commands.len() < self.command_capacity {
            for peer in self.peers.iter().flatten() {
                if peer.send_available()? == 0 {
                    return Ok(());
                }
            }
            if self
                .coordinator
                .as_mut()
                .unwrap()
                .issue(&mut self.outgoing)?
            {
                self.dispatch(self.outgoing[0])?;
                for index in 1..self.peers.len() {
                    self.peers[index]
                        .as_mut()
                        .unwrap()
                        .enqueue(self.outgoing[index])?;
                }
            }
        }
        Ok(())
    }
}
