//! Full-state CPU model of the weighted owner schedule; never a production fallback.
use crate::macro_owner::{CandidateKey, FutureOffer, MacroOwner};
use mgbfs_core::{
    hash::{GemmHash, Hash128},
    macro_generators::MacroGeneratorSet,
    matrix::MatrixGroup,
    Result,
};
use std::collections::{BTreeMap, BTreeSet};

pub struct MacroSimulationConfig {
    pub rank_map: Vec<usize>,
    pub buckets: usize,
    pub future_capacity_per_bucket: usize,
    pub settled_capacity_per_bucket: usize,
    pub seed: [u8; 16],
    pub schedule: u64,
    pub pre_dedup: bool,
}

pub struct MacroSimulation {
    pub layers: Vec<Vec<Vec<u8>>>,
    pub generated: u64,
    pub offered: u64,
    pub committed: u64,
}

fn random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 32) as usize
}

pub fn run_macro(
    graph: &MatrixGroup,
    macros: &MacroGeneratorSet,
    config: &MacroSimulationConfig,
) -> Result<MacroSimulation> {
    graph.validate()?;
    if macros.transitions.is_empty()
        || macros.effective_depth == 0
        || config.rank_map.is_empty()
        || !config.rank_map.len().is_power_of_two()
        || !config.buckets.is_power_of_two()
        || config.rank_map.iter().copied().collect::<BTreeSet<_>>()
            != (0..config.rank_map.len()).collect()
    {
        return Err("MACRO_SIMULATION_CONFIG".into());
    }
    let hash = GemmHash::from_seed(graph.start.len(), config.seed)?;
    let prefix_bits = config.rank_map.len().ilog2() + config.buckets.ilog2();
    let locate = |value: Hash128| {
        let prefix = value.prefix(prefix_bits) as usize;
        (
            config.rank_map[prefix / config.buckets],
            prefix % config.buckets,
        )
    };
    let mut owners = BTreeMap::new();
    let start_hash = hash.hash(&graph.start)?;
    let start_location = locate(start_hash);
    for rank in 0..config.rank_map.len() {
        for bucket in 0..config.buckets {
            let mut owner = MacroOwner::new(
                macros.effective_depth,
                config.future_capacity_per_bucket,
                config.settled_capacity_per_bucket,
            )?;
            owner.seed(
                0,
                (rank == start_location.0 && bucket == start_location.1).then_some(start_hash),
            )?;
            owners.insert((rank, bucket), owner);
        }
    }
    let mut current = vec![graph.start.clone()];
    let mut output = MacroSimulation {
        layers: vec![current.clone()],
        generated: 0,
        offered: 0,
        committed: 1,
    };
    let mut references = BTreeMap::<u64, (u32, Vec<u8>)>::new();
    let mut next_ref = 0u64;
    let mut depth = 0u32;
    let mut rng = config.schedule;
    loop {
        let mut offers = Vec::new();
        let mut local = BTreeMap::new();
        for (parent_index, parent) in current.iter().enumerate() {
            for (ordinal, transition) in macros.transitions.iter().enumerate() {
                let target_depth = depth
                    .checked_add(transition.weight)
                    .ok_or("DEPTH_OVERFLOW")?;
                let state = graph.apply_left(&transition.matrix, parent)?;
                let state_hash = hash.hash(&state)?;
                let reference = next_ref;
                next_ref = next_ref.checked_add(1).ok_or("STATE_REF_OVERFLOW")?;
                let offer = FutureOffer::new(
                    target_depth,
                    state_hash,
                    reference,
                    CandidateKey::new(
                        depth,
                        transition.weight,
                        0,
                        parent_index as u64,
                        ordinal as u32,
                    ),
                );
                output.generated = output.generated.checked_add(1).ok_or("COUNT_OVERFLOW")?;
                if config.pre_dedup {
                    // A producer batch spans several weights. A later parent
                    // can emit the same hash at a smaller target depth, so
                    // "first row wins" would silently corrupt BFS distance.
                    match local.entry(state_hash) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert((offer, state));
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry)
                            if offer.key < entry.get().0.key =>
                        {
                            entry.insert((offer, state));
                        }
                        _ => {}
                    }
                } else {
                    offers.push((offer, state));
                }
            }
        }
        if config.pre_dedup {
            offers.extend(local.into_values());
        }
        for i in (1..offers.len()).rev() {
            let other = random(&mut rng) % (i + 1);
            offers.swap(i, other);
        }
        for (offer, state) in offers {
            let location = locate(offer.hash);
            owners.get_mut(&location).unwrap().offer(offer)?;
            references.insert(offer.state_ref, (offer.target_depth, state));
            output.offered = output.offered.checked_add(1).ok_or("COUNT_OVERFLOW")?;
        }
        let settling = depth.checked_add(1).ok_or("DEPTH_OVERFLOW")?;
        let mut next = Vec::new();
        for owner in owners.values_mut() {
            for (_, reference) in owner.settle(settling)? {
                let (target, state) = references.get(&reference).ok_or("MISSING_STATE_REF")?;
                if *target != settling {
                    return Err("STATE_REF_DEPTH".into());
                }
                next.push(state.clone());
            }
        }
        references.retain(|_, (target, _)| *target != settling);
        next.sort();
        if !next.is_empty() {
            output.committed = output
                .committed
                .checked_add(next.len() as u64)
                .ok_or("COUNT_OVERFLOW")?;
            output.layers.push(next.clone());
        }
        depth = settling;
        current = next;
        if current.is_empty() && owners.values().all(|owner| owner.pending_records() == 0) {
            if !references.is_empty() {
                return Err("ORPHAN_STATE_REFS".into());
            }
            return Ok(output);
        }
    }
}
