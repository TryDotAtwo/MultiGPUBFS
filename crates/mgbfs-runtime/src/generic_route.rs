use mgbfs_core::Result;
/// Logical geometry only. This does not assert transport/hardware acceptance.
#[derive(Debug, Clone)]
pub struct GenericRouteGeometry {
    pub world: u32,
    pub local_shards: u32,
    pub owner_to_rank: Vec<u32>,
    pub owner_cuts: Vec<u64>,
    pub queue_capacity: u32,
    pub bank_bytes: u64,
}
impl GenericRouteGeometry {
    pub fn new(
        weights: &[u64],
        map: Vec<u32>,
        local_shards: u32,
        queue_capacity: u32,
    ) -> Result<Self> {
        let world = weights.len() as u32;
        if world == 0
            || world > 128
            || local_shards == 0
            || local_shards > 4096
            || queue_capacity == 0
            || map.len() != weights.len()
            || weights.iter().any(|&v| v == 0)
        {
            return Err("GENERIC_ROUTE_SHAPE".into());
        }
        let mut seen = vec![false; weights.len()];
        for &rank in &map {
            let at = rank as usize;
            if at >= seen.len() || seen[at] {
                return Err("GENERIC_ROUTE_MAP".into());
            }
            seen[at] = true;
        }
        let total = weights.iter().map(|&v| u128::from(v)).sum::<u128>();
        let mut sum = 0u128;
        let mut cuts = vec![0u64];
        for &weight in weights {
            sum += u128::from(weight);
            cuts.push(((sum * (1u128 << 32)) / total) as u64);
        }
        if cuts.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("GENERIC_ROUTE_WEIGHT_RESOLUTION".into());
        }
        let bytes = u64::from(world) * u64::from(local_shards) * u64::from(queue_capacity) * 32;
        Ok(Self {
            world,
            local_shards,
            owner_to_rank: map,
            owner_cuts: cuts,
            queue_capacity,
            bank_bytes: bytes,
        })
    }
    pub fn destination(&self, hash: u64) -> (u32, u32) {
        let high = hash >> 32;
        let at = self.owner_cuts.partition_point(|&cut| cut <= high) - 1;
        let shard = ((hash & 0xffff_ffff) * u64::from(self.local_shards) >> 32) as u32;
        (self.owner_to_rank[at], shard)
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoundVote {
    Continue,
    Retry,
    Capacity(u32),
    Stop,
    Fatal(u32),
}

#[cfg(any(feature = "cuda", test))]
impl RoundVote {
    // Keep tags in sync with classify_route_retry_v3 in cuda/generic_route.cu.
    // The owner payload includes bit 8, used by the window edge resolver.
    pub(crate) fn current(value: u32) -> Result<Self> {
        let flags = value & 0x0fff_ffff;
        match value {
            0 => Ok(Self::Continue),
            1 => Ok(Self::Retry),
            0x2000_0000 => Ok(Self::Stop),
            _ => match value >> 28 {
                1 if (1..=7).contains(&flags) => Ok(Self::Capacity(flags)),
                3 if flags & !7 != 0 => Ok(Self::Fatal(flags)),
                4 if flags != 0 => Ok(Self::Fatal(0x1_0000 | flags)),
                _ => Err("GENERIC_DISTRIBUTED_ROUTE_VOTE".into()),
            },
        }
    }

    pub(crate) fn owner(flags: u32) -> Self {
        match flags {
            0 => Self::Continue,
            1..=7 => Self::Capacity(flags),
            _ => Self::Fatal(flags),
        }
    }

    // The legacy first vote enters this path only after observing a stop
    // or error. Its owner marker overlaps a raw error flag. The source-only
    // legacy vote and the separate raw owner max are lossless in this case.
    pub(crate) fn legacy_error(source: u32, owner: u32, stop: bool) -> Self {
        if source > 1 {
            return Self::Fatal(source);
        }
        let owner = Self::owner(owner);
        match owner {
            Self::Fatal(_) => owner,
            _ if stop => Self::Stop,
            Self::Continue if source == 1 => Self::Retry,
            _ => owner,
        }
    }
}

#[cfg(test)]
#[test]
fn distributed_round_vote_distinguishes_capacity_from_fatal_and_keeps_collective_order() {
    assert_eq!(
        RoundVote::current(0x1000_0002).unwrap(),
        RoundVote::owner(2)
    );
    assert_eq!(
        RoundVote::current(0x3000_0102).unwrap(),
        RoundVote::owner(258)
    );
    assert_eq!(
        RoundVote::current([1, 0x1000_0002, 0x2000_0000].into_iter().max().unwrap()).unwrap(),
        RoundVote::Stop
    );
    assert_eq!(
        RoundVote::current(
            [0x1000_0002, 0x2000_0000, 0x3000_0100]
                .into_iter()
                .max()
                .unwrap()
        )
        .unwrap(),
        RoundVote::Fatal(256)
    );
    assert_eq!(RoundVote::legacy_error(1, 2, false), RoundVote::Capacity(2));
    assert_eq!(RoundVote::legacy_error(0, 258, true), RoundVote::Fatal(258));
    assert_eq!(
        RoundVote::legacy_error(0x1_0003, 2, true),
        RoundVote::Fatal(0x1_0003)
    );
    assert_eq!(RoundVote::legacy_error(0, 2, true), RoundVote::Stop);
}
