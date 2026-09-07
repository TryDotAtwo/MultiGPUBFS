//! Host adapter binding the real admitted pump to independent flat buffer pools.
//! No CUDA calls: the caller allocates reported storage before use and proves
//! native transfer/consumer completion before the corresponding callbacks.
use crate::{
    control_connection::ControlConnection,
    control_pump::ControlPump,
    control_wire::{Action, ControlFrame, Plane},
    payload_lease::{BankConsumer, PayloadBank, PayloadBanks},
    scatter_admission::TicketKey,
    source_banks::{SourceBank, SourceBanks},
};
use mgbfs_core::Result;

fn kind(p: Plane) -> Result<usize> {
    match p {
        Plane::Candidate => Ok(0),
        Plane::Request => Ok(1),
        Plane::Response => Ok(2),
        Plane::Receipt => Ok(3),
        Plane::None => Err("BUFFER_PLANE".into()),
    }
}
fn key(f: ControlFrame) -> TicketKey {
    TicketKey {
        depth: f.depth,
        epoch: f.epoch,
        source: f.source_rank,
        plane: f.plane,
        generation: f.slot,
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceHandle {
    plane: Plane,
    bank: SourceBank,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferLaunch {
    pub key: TicketKey,
    pub source_offset: Option<u64>,
    pub receive_offset: u64,
    pub bytes: u64,
}
pub enum BufferEvent {
    Launch(BufferLaunch),
    Finalize(ControlFrame),
    Publish(ControlFrame),
}
pub struct BufferConsumer {
    plane: Plane,
    token: BankConsumer,
}
struct Source {
    offered: bool,
    handle: Option<SourceHandle>,
    sizes: Vec<u64>,
    ticket: Option<TicketKey>,
}
struct Active {
    launched: bool,
    launch: BufferLaunch,
    bank: PayloadBank,
    source: Option<SourceHandle>,
    transferred: bool,
}
struct Pool {
    source: SourceBanks,
    receive: PayloadBanks,
    descriptions: Vec<Source>,
    live: Vec<Option<Active>>,
    capacity: u64,
}
pub struct AdmittedBuffers {
    depth: u64,
    source_closed: bool,
    finalizing: bool,
    pump: Option<ControlPump>,
    rank: u32,
    world: usize,
    pools: Vec<Pool>,
}
impl AdmittedBuffers {
    pub fn new(
        world: u32,
        rank: u32,
        slots: usize,
        peers: Vec<Option<ControlConnection>>,
        capacities: [u64; 4],
        consumers: usize,
    ) -> Result<Self> {
        let pump = ControlPump::new_admitted(world, rank, slots, peers, capacities)?;
        let mut pools = Vec::new();
        pools.try_reserve_exact(4).map_err(|_| "BUFFER_CAPACITY")?;
        for capacity in capacities {
            let source = SourceBanks::new(rank, slots, capacity)?;
            let receive = PayloadBanks::new(world, slots, capacity, consumers, 256)?;
            let mut descriptions = Vec::new();
            descriptions
                .try_reserve_exact(slots)
                .map_err(|_| "BUFFER_CAPACITY")?;
            let mut live = Vec::new();
            live.try_reserve_exact(slots)
                .map_err(|_| "BUFFER_CAPACITY")?;
            live.resize_with(slots, || None);
            for _ in 0..slots {
                let mut sizes = Vec::new();
                sizes
                    .try_reserve_exact(world as usize)
                    .map_err(|_| "BUFFER_CAPACITY")?;
                sizes.resize(world as usize, 0);
                descriptions.push(Source {
                    offered: false,
                    handle: None,
                    sizes,
                    ticket: None,
                });
            }
            pools.push(Pool {
                source,
                receive,
                descriptions,
                live,
                capacity,
            });
        }
        Ok(Self {
            depth: 0,
            source_closed: false,
            finalizing: false,
            pump: Some(pump),
            rank,
            world: world as usize,
            pools,
        })
    }
    fn apply<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.pump.is_none() {
            return Err("BUFFER_FAILED".into());
        }
        let result = f(self);
        // Close TCP on local failure. Caller must also abort its NCCL group.
        if result.is_err() {
            self.pump.take();
        }
        result
    }
    pub fn allocation_bytes(&self, plane: Plane) -> Result<(u64, u64)> {
        let p = &self.pools[kind(plane)?];
        Ok((p.source.bytes(), p.receive.bytes()))
    }
    pub fn reserve(&mut self, plane: Plane, depth: u64) -> Result<Option<SourceHandle>> {
        self.apply(|s| {
            if depth != s.depth || s.finalizing || (plane == Plane::Candidate && s.source_closed) {
                return Err("BUFFER_RESERVATION_PHASE".into());
            }
            let p = &mut s.pools[kind(plane)?];
            let Some(bank) = p.source.reserve(depth)? else {
                return Ok(None);
            };
            let handle = SourceHandle { plane, bank };
            let d = p
                .descriptions
                .iter_mut()
                .find(|d| d.handle.is_none())
                .ok_or("BUFFER_CAPACITY")?;
            d.handle = Some(handle);
            d.offered = false;
            d.ticket = None;
            d.sizes.fill(0);
            Ok(Some(handle))
        })
    }
    pub fn source_offset(&mut self, h: SourceHandle) -> Result<u64> {
        self.apply(|s| s.pools[kind(h.plane)?].source.offset(h.bank))
    }
    /// Invoke after the source's generation/pack completion event. Counts are
    /// bytes per destination in the preallocated source allocation.
    pub fn ready(&mut self, h: SourceHandle, sizes: &[u64]) -> Result<()> {
        self.apply(|s| {
            let p = &mut s.pools[kind(h.plane)?];
            if sizes.len() != s.world
                || sizes
                    .iter()
                    .try_fold(0u64, |a, &b| a.checked_add(b))
                    .map_or(true, |n| n > p.capacity)
            {
                return Err("BUFFER_SEND_CAPACITY".into());
            }
            let d = p
                .descriptions
                .iter_mut()
                .find(|d| d.handle == Some(h))
                .ok_or("BUFFER_SOURCE")?;
            p.source.ready(h.bank)?;
            d.sizes.copy_from_slice(sizes);
            d.offered = true;
            s.pump.as_mut().unwrap().offer(h.plane, h.bank.token())
        })
    }
    pub fn poll(&mut self) -> Result<Option<BufferEvent>> {
        self.apply(|s| {
            let pump = s.pump.as_mut().unwrap();
            pump.poll()?;
            let Some(f) = pump.command()? else {
                return Ok(None);
            };
            match f.action {
                Action::Begin => {
                    if f.source_rank == s.rank {
                        let p = &mut s.pools[kind(f.plane)?];
                        let bank = p.source.bind_ticket(key(f))?;
                        let h = SourceHandle {
                            plane: f.plane,
                            bank,
                        };
                        let d = p
                            .descriptions
                            .iter_mut()
                            .find(|d| d.handle == Some(h))
                            .ok_or("BUFFER_SOURCE")?;
                        d.ticket = Some(key(f));
                        pump.describe_bytes(f, &d.sizes)?;
                    }
                    Ok(None)
                }
                Action::TicketBytes => {
                    let p = &mut s.pools[kind(f.plane)?];
                    let bank = p
                        .receive
                        .reserve(key(f), f.payload_bytes)?
                        .ok_or("BUFFER_RECEIVE_CAPACITY")?;
                    let source = if f.source_rank == s.rank {
                        Some(
                            p.descriptions
                                .iter()
                                .find(|d| d.ticket == Some(key(f)))
                                .and_then(|d| d.handle)
                                .ok_or("BUFFER_SOURCE")?,
                        )
                    } else {
                        None
                    };
                    let source_offset = source.map(|h| p.source.offset(h.bank)).transpose()?;
                    let launch = BufferLaunch {
                        key: key(f),
                        source_offset,
                        receive_offset: p.receive.offset(bank)?,
                        bytes: f.payload_bytes,
                    };
                    let slot = p
                        .live
                        .iter_mut()
                        .find(|x| x.is_none())
                        .ok_or("BUFFER_CAPACITY")?;
                    *slot = Some(Active {
                        launched: false,
                        launch,
                        bank,
                        source,
                        transferred: false,
                    });
                    pump.admit_bytes(f, p.capacity)?;
                    Ok(None)
                }
                Action::Launch => {
                    let p = &mut s.pools[kind(f.plane)?];
                    let a = p
                        .live
                        .iter_mut()
                        .flatten()
                        .find(|a| a.launch.key == key(f))
                        .ok_or("BUFFER_TICKET")?;
                    if a.launched {
                        return Err("BUFFER_DUPLICATE_LAUNCH".into());
                    }
                    a.launched = true;
                    Ok(Some(BufferEvent::Launch(a.launch)))
                }
                Action::Finalize => {
                    s.finalizing = true;
                    Ok(Some(BufferEvent::Finalize(f)))
                }
                Action::Publish => {
                    s.depth = f.depth;
                    s.finalizing = false;
                    s.source_closed = false;
                    Ok(Some(BufferEvent::Publish(f)))
                }
                _ => Err("BUFFER_COMMAND".into()),
            }
        })
    }
    fn active(&mut self, l: BufferLaunch) -> Result<(&mut Pool, usize)> {
        let p = &mut self.pools[kind(l.key.plane)?];
        let i = p
            .live
            .iter()
            .position(|a| a.as_ref().is_some_and(|a| a.launched && a.launch == l))
            .ok_or("BUFFER_TICKET")?;
        Ok((p, i))
    }
    pub fn consumer(&mut self, l: BufferLaunch) -> Result<BufferConsumer> {
        self.apply(|s| {
            let (p, i) = s.active(l)?;
            Ok(BufferConsumer {
                plane: l.key.plane,
                token: p.receive.consumer(p.live[i].as_ref().unwrap().bank)?,
            })
        })
    }
    /// Copy the immutable per-destination byte counts for native scatter into
    /// caller-preallocated host storage. False means this rank is not source;
    /// its output is zeroed. source_offset is the whole send-bank base, not
    /// the source rank's self-view offset (which adds preceding rank counts).
    pub fn source_sizes(&mut self, l: BufferLaunch, out: &mut [u64]) -> Result<bool> {
        self.apply(|s| {
            if out.len() != s.world {
                return Err("BUFFER_COUNTS_SHAPE".into());
            }
            let (p, i) = s.active(l)?;
            let Some(h) = p.live[i].as_ref().unwrap().source else {
                out.fill(0);
                return Ok(false);
            };
            let d = p
                .descriptions
                .iter()
                .find(|d| d.handle == Some(h) && d.ticket == Some(l.key))
                .ok_or("BUFFER_SOURCE")?;
            out.copy_from_slice(&d.sizes);
            Ok(true)
        })
    }
    pub fn seal(&mut self, l: BufferLaunch) -> Result<()> {
        self.apply(|s| {
            let (p, i) = s.active(l)?;
            p.receive.seal(p.live[i].as_ref().unwrap().bank)
        })
    }
    pub fn complete(&mut self, c: BufferConsumer) -> Result<()> {
        self.apply(|s| s.pools[kind(c.plane)?].receive.complete(c.token))
    }
    pub fn drained(&mut self, l: BufferLaunch) -> Result<bool> {
        self.apply(|s| {
            let (p, i) = s.active(l)?;
            p.receive.drained(p.live[i].as_ref().unwrap().bank)
        })
    }
    pub fn transfer_complete(&mut self, l: BufferLaunch) -> Result<()> {
        self.apply(|s| {
            let (p, i) = s.active(l)?;
            p.live[i].as_mut().unwrap().transferred = true;
            s.pump.as_mut().unwrap().transfer_complete(l.key.epoch)
        })
    }
    pub fn consume(&mut self, l: BufferLaunch) -> Result<()> {
        self.apply(|s| {
            let (p, i) = s.active(l)?;
            let a = p.live[i].as_ref().unwrap();
            if !a.transferred {
                return Err("BUFFER_TRANSFER_PENDING".into());
            }
            p.receive.retire(a.bank)?;
            if let Some(h) = a.source {
                p.source.retire(h.bank, l.key)?;
                let d = p
                    .descriptions
                    .iter_mut()
                    .find(|d| d.handle == Some(h))
                    .ok_or("BUFFER_SOURCE")?;
                d.handle = None;
                d.ticket = None;
            }
            p.live[i] = None;
            s.pump.as_mut().unwrap().consumed(l.key.epoch)
        })
    }
    pub fn close_source(&mut self) -> Result<()> {
        self.apply(|s| {
            if s.pools[0]
                .descriptions
                .iter()
                .any(|d| d.handle.is_some() && !d.offered)
            {
                return Err("BUFFER_GENERATION_PENDING".into());
            }
            s.pump.as_mut().unwrap().close_source()?;
            s.source_closed = true;
            Ok(())
        })
    }
    pub fn finalized(&mut self, drained: bool) -> Result<()> {
        self.apply(|s| {
            let buffers_drained = s.pools.iter().all(|p| {
                p.live.iter().all(Option::is_none)
                    && p.descriptions.iter().all(|d| d.handle.is_none())
            });
            s.pump
                .as_mut()
                .unwrap()
                .finalized(drained && buffers_drained)
        })
    }
}
