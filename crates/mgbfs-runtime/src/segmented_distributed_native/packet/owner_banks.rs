//! Two input leases per logical shard. Payload stays in the source slots.

use super::PacketOwnerShard;
use mgbfs_core::Result;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OwnerSource {
    Local(usize),
    Remote(usize),
}

impl OwnerSource {
    fn ready_queue(self) -> usize {
        usize::from(matches!(self, Self::Remote(_)))
    }
}

#[derive(Clone, Copy)]
pub(super) struct OwnerLease {
    pub(super) span: PacketOwnerShard,
    pub(super) source: OwnerSource,
    generation: u64,
}

#[derive(Default)]
struct ShardBanks {
    leases: [Option<OwnerLease>; 2],
    first: usize,
    wave_position: Option<usize>,
}

pub(super) struct OwnerBanks {
    shards: Vec<ShardBanks>,
    ready: [VecDeque<usize>; 2],
    next_generation: u64,
    running: usize,
    live: usize,
}

impl OwnerBanks {
    pub(super) fn new(shards: usize) -> Self {
        Self {
            shards: (0..shards).map(|_| ShardBanks::default()).collect(),
            ready: [VecDeque::new(), VecDeque::new()],
            next_generation: 0,
            running: 0,
            live: 0,
        }
    }

    /// Returns false without modifying either bank when the source must wait.
    pub(super) fn offer(&mut self, span: PacketOwnerShard, source: OwnerSource) -> Result<bool> {
        if span.rows == 0 {
            return Err("SEGMENT_PACKET_OWNER_EMPTY_SPAN".into());
        }
        let shard = self
            .shards
            .get_mut(span.shard)
            .ok_or("SEGMENT_PACKET_OWNER_SHARD")?;
        let first_empty = shard.leases[shard.first].is_none();
        let slot = if first_empty {
            shard.first
        } else {
            shard.first ^ 1
        };
        if shard.leases[slot].is_some() {
            return Ok(false);
        }
        let generation = self.next_generation;
        self.next_generation = generation
            .checked_add(1)
            .ok_or("SEGMENT_PACKET_OWNER_SEQUENCE")?;
        shard.leases[slot] = Some(OwnerLease {
            span,
            source,
            generation,
        });
        if first_empty {
            self.ready[source.ready_queue()].push_back(span.shard);
        }
        self.live += 1;
        Ok(true)
    }

    /// Each wave has distinct shards and one input format class. Local Pack4
    /// rows and direct-wire Pack4 packets use different owner CUDA entry points.
    pub(super) fn begin_wave(&mut self, lanes: usize, out: &mut Vec<OwnerLease>) -> Result<()> {
        if self.running != 0 || lanes == 0 {
            return Err("SEGMENT_PACKET_OWNER_WAVE_STATE".into());
        }
        out.clear();
        let oldest = |queue: &VecDeque<usize>| {
            queue.front().and_then(|&index| {
                let shard = &self.shards[index];
                shard.leases[shard.first].map(|lease| lease.generation)
            })
        };
        let class = match (oldest(&self.ready[0]), oldest(&self.ready[1])) {
            (None, None) => return Ok(()),
            (Some(local), Some(remote)) => usize::from(remote < local),
            (Some(_), None) => 0,
            (None, Some(_)) => 1,
        };
        while out.len() < lanes {
            let Some(index) = self.ready[class].pop_front() else {
                break;
            };
            let shard = &mut self.shards[index];
            let lease = shard.leases[shard.first].ok_or("SEGMENT_PACKET_OWNER_EMPTY_BANK")?;
            if shard.wave_position.is_some() || lease.source.ready_queue() != class {
                return Err("SEGMENT_PACKET_OWNER_WAVE_STATE".into());
            }
            shard.wave_position = Some(out.len());
            out.push(lease);
        }
        self.running = out.len();
        Ok(())
    }

    /// Call only after the last GPU read by every owner lane in the wave.
    pub(super) fn complete_wave(&mut self, wave: &[OwnerLease]) -> Result<()> {
        if wave.is_empty() || wave.len() != self.running {
            return Err("SEGMENT_PACKET_OWNER_WAVE_SIZE".into());
        }
        for (position, lease) in wave.iter().enumerate() {
            let shard = self
                .shards
                .get(lease.span.shard)
                .ok_or("SEGMENT_PACKET_OWNER_SHARD")?;
            if shard.wave_position != Some(position)
                || !shard.leases[shard.first]
                    .is_some_and(|running| running.generation == lease.generation)
            {
                return Err("SEGMENT_PACKET_OWNER_STALE_LEASE".into());
            }
        }
        for lease in wave {
            let shard = &mut self.shards[lease.span.shard];
            shard.leases[shard.first] = None;
            shard.first ^= 1;
            shard.wave_position = None;
            if let Some(ready) = shard.leases[shard.first] {
                self.ready[ready.source.ready_queue()].push_back(lease.span.shard);
            }
        }
        self.running = 0;
        self.live -= wave.len();
        Ok(())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.live == 0
    }

    #[cfg(test)]
    pub(super) fn is_consuming(&self, shard: usize) -> bool {
        self.shards[shard].wave_position.is_some()
    }

    pub(super) fn has_sibling(&self, shard: usize) -> bool {
        self.shards[shard].leases.iter().all(Option::is_some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmented_distributed_native::QueueSide;
    use std::ptr::null_mut;

    fn span(shard: usize) -> PacketOwnerShard {
        PacketOwnerShard {
            shard,
            rows: 1,
            queue: QueueSide {
                records: null_mut(),
                states: null_mut(),
                counts: null_mut(),
            },
            byte_wire: false,
        }
    }

    #[test]
    fn fills_the_sibling_during_owner_work_and_retires_each_source_once() {
        let mut banks = OwnerBanks::new(3);
        let mut wave = Vec::new();
        assert!(banks.offer(span(1), OwnerSource::Local(0)).unwrap());
        assert!(banks.offer(span(0), OwnerSource::Local(0)).unwrap());
        assert!(banks.offer(span(2), OwnerSource::Remote(0)).unwrap());
        banks.begin_wave(3, &mut wave).unwrap();
        assert_eq!(
            wave.iter()
                .map(|lease| lease.span.shard)
                .collect::<Vec<_>>(),
            [1, 0]
        );
        let old_wave = wave.clone();

        assert!(banks.is_consuming(1));
        assert!(banks.offer(span(1), OwnerSource::Remote(0)).unwrap());
        assert!(!banks.offer(span(1), OwnerSource::Remote(1)).unwrap());
        assert!(!banks.is_empty());
        banks.complete_wave(&wave).unwrap();
        assert!(!banks.is_consuming(1));
        assert!(banks.offer(span(1), OwnerSource::Remote(1)).unwrap());

        banks.begin_wave(3, &mut wave).unwrap();
        assert_eq!(
            wave.iter()
                .map(|lease| lease.span.shard)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        assert!(wave
            .iter()
            .all(|lease| lease.source == OwnerSource::Remote(0)));
        assert!(banks.complete_wave(&old_wave).is_err());
        banks.complete_wave(&wave).unwrap();

        banks.begin_wave(3, &mut wave).unwrap();
        assert_eq!(wave.len(), 1);
        assert_eq!(wave[0].source, OwnerSource::Remote(1));
        banks.complete_wave(&wave).unwrap();
        assert!(banks.is_empty());
    }
}
