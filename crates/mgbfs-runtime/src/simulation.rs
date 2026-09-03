//! Small-graph CPU integration oracle, never an execution backend.
use crate::{
    owner::OwnerModel,
    receipts::BatchReceipts,
    ring::StateRing,
    transport::{Kind, Transport},
};
use mgbfs_core::hash::{GemmHash, Hash128};
use mgbfs_core::{matrix::MatrixGroup, Result};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Dense,
    HashFirst,
}
pub struct Config {
    pub profile: Profile,
    pub prededup: bool,
    pub rank_map: Vec<usize>,
    pub buckets: usize,
    pub bucket_capacity: usize,
    pub ring_records: u64,
    pub seed: [u8; 16],
    pub schedule: u64,
    pub delayed_archive: bool,
}
pub struct Simulation {
    pub layers: Vec<Vec<Vec<u8>>>,
    pub generated: u64,
    pub committed: u64,
    pub requests: u64,
    pub responses: u64,
    pub tickets: u64,
}
#[derive(Clone)]
struct Child {
    hash: Hash128,
    state: Vec<u8>,
    mv: usize,
}
struct Chunk {
    rank: usize,
    id: u64,
    states: Vec<Vec<u8>>,
}
fn random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 32) as usize
}
fn send(
    t: &mut Transport,
    kind: Kind,
    src: usize,
    dst: usize,
    n: usize,
    w: usize,
    slot: &mut u64,
    rng: &mut u64,
) -> Result<u64> {
    let mut counts = vec![0; w];
    counts[dst] = n as u64;
    t.offer(kind, src, *slot, counts)?;
    *slot = slot.checked_add(1).ok_or("SLOT_OVERFLOW")?;
    let ticket = t.issue()?.ok_or("MODEL_TRANSPORT_STALLED")?;
    if ticket.kind != kind {
        return Err("MODEL_TICKET_MISMATCH".into());
    }
    let first = random(rng) % w;
    for r in 0..w {
        t.complete((first + r) % w, ticket.seq)?;
    }
    Ok(ticket.seq)
}
fn archive(
    rank: usize,
    id: u64,
    delay: bool,
    rings: &mut [StateRing],
    t: &mut Transport,
    pending: &mut Vec<(usize, u64)>,
) -> Result<()> {
    if delay {
        t.work(rank, true)?;
        pending.push((rank, id));
    } else {
        rings[rank].archived(id)?;
    }
    Ok(())
}
/// Bounded parent batch = one parent in this oracle. The transport/owner lifecycle
/// is integrated, but different parent batches are intentionally serialized.
/// Receipt/request/response readiness and archive delay vary within that boundary.
pub fn run(graph: &MatrixGroup, c: &Config) -> Result<Simulation> {
    graph.validate()?;
    let w = c.rank_map.len();
    if !w.is_power_of_two()
        || !c.buckets.is_power_of_two()
        || c.bucket_capacity == 0
        || c.rank_map.iter().copied().collect::<BTreeSet<_>>() != (0..w).collect()
    {
        return Err("SIMULATION_CONFIG".into());
    }
    let bits = w.trailing_zeros() + c.buckets.trailing_zeros();
    if bits > 64 {
        return Err("SIMULATION_PREFIX".into());
    }
    let hash = GemmHash::from_seed(graph.start.len(), c.seed)?;
    let location = |h: Hash128| {
        let p = h.prefix(bits) as usize;
        (c.rank_map[p / c.buckets], p % c.buckets)
    };
    let mut rings = (0..w)
        .map(|_| StateRing::new(c.ring_records, 65536))
        .collect::<Result<Vec<_>>>()?;
    let mut t = Transport::new(w, 4, graph.generators.len().max(1) as u64)?;
    let mut slot = 0;
    let mut rng = c.schedule;
    let mut pending_archive = vec![];
    let start_rank = location(hash.hash(&graph.start)?).0;
    let start = rings[start_rank].reserve(1)?;
    rings[start_rank].materialized(start.id)?;
    rings[start_rank].publish(start.id)?;
    archive(
        start_rank,
        start.id,
        c.delayed_archive,
        &mut rings,
        &mut t,
        &mut pending_archive,
    )?;
    let mut current = vec![Chunk {
        rank: start_rank,
        id: start.id,
        states: vec![graph.start.clone()],
    }];
    let mut previous: BTreeMap<(usize, usize), Vec<Hash128>> = BTreeMap::new();
    let mut out = Simulation {
        layers: vec![],
        generated: 0,
        committed: 0,
        requests: 0,
        responses: 0,
        tickets: 0,
    };
    loop {
        let mut layer: Vec<_> = current.iter().flat_map(|x| x.states.clone()).collect();
        layer.sort();
        if layer.is_empty() {
            break;
        }
        let mut current_hashes: BTreeMap<(usize, usize), Vec<Hash128>> = BTreeMap::new();
        for state in &layer {
            let h = hash.hash(state)?;
            current_hashes.entry(location(h)).or_default().push(h);
        }
        out.layers.push(layer);
        let mut owners = BTreeMap::new();
        for rank in 0..w {
            for bucket in 0..c.buckets {
                let key = (rank, bucket);
                owners.insert(
                    key,
                    OwnerModel::new(
                        previous.get(&key).cloned().unwrap_or_default(),
                        current_hashes.get(&key).cloned().unwrap_or_default(),
                        c.bucket_capacity,
                    ),
                );
            }
        }
        let mut next = vec![];
        for parent_chunk in &current {
            // The CPU chunk is the independent packed-parent copy. DENSE no
            // longer needs the original StateRing range after that copy exists.
            if c.profile == Profile::Dense {
                rings[parent_chunk.rank].enumerated(parent_chunk.id)?;
                rings[parent_chunk.rank].reclaim();
            }
            for parent in &parent_chunk.states {
                let source = parent_chunk.rank;
                t.work(source, true)?;
                if c.profile == Profile::HashFirst {
                    rings[source].hold_origins(parent_chunk.id)?;
                }
                let mut groups: BTreeMap<(usize, usize), Vec<Child>> = BTreeMap::new();
                let mut local = BTreeSet::new();
                let mut emitted = vec![0u64; w];
                for mv in 0..graph.generators.len() {
                    let state = graph.successor(parent, mv)?;
                    let h = hash.hash(&state)?;
                    out.generated += 1;
                    if c.prededup && !local.insert(h) {
                        continue;
                    }
                    let loc = location(h);
                    emitted[loc.0] += 1;
                    groups
                        .entry(loc)
                        .or_default()
                        .push(Child { hash: h, state, mv });
                }
                let mut receipts = BatchReceipts::new(&emitted)?;
                let mut accepted = vec![0u64; w];
                let mut commits: Vec<(Chunk, Vec<Child>)> = vec![];
                for (key, children) in groups {
                    let owner = key.0;
                    let seq = send(
                        &mut t,
                        Kind::Candidate,
                        source,
                        owner,
                        children.len(),
                        w,
                        &mut slot,
                        &mut rng,
                    )?;
                    let incoming: Vec<_> = children.iter().map(|x| x.hash).collect();
                    // Preview is a CPU-only transaction oracle, not a second production merge.
                    let survivors = owners[&key].clone().commit(seq, &incoming)?;
                    let mut winners: Vec<_> = survivors
                        .iter()
                        .map(|h| children.iter().find(|x| x.hash == *h).unwrap().clone())
                        .collect();
                    winners.sort_by_key(|x| x.mv); // request order before target reservation
                    let target = if winners.is_empty() {
                        None
                    } else {
                        Some(rings[owner].reserve(winners.len() as u64)?)
                    };
                    owners.get_mut(&key).unwrap().commit(seq, &incoming)?;
                    accepted[owner] += winners.len() as u64;
                    out.committed += winners.len() as u64;
                    if let Some(extent) = target {
                        let chunk = Chunk {
                            rank: owner,
                            id: extent.id,
                            states: winners.iter().map(|x| x.state.clone()).collect(),
                        };
                        if c.profile == Profile::Dense {
                            rings[owner].materialized(extent.id)?;
                            archive(
                                owner,
                                extent.id,
                                c.delayed_archive,
                                &mut rings,
                                &mut t,
                                &mut pending_archive,
                            )?;
                            next.push(chunk);
                        } else {
                            commits.push((chunk, winners));
                        }
                    }
                    t.consume(seq)?;
                }
                if c.profile == Profile::HashFirst {
                    // Event 0=request,1=response,2=terminal receipt. A response is
                    // ready only after its request; receipt ordering is independent.
                    let mut events: Vec<(u8, usize)> =
                        (0..commits.len()).flat_map(|i| [(0, i), (1, i)]).collect();
                    for (owner, &n) in emitted.iter().enumerate() {
                        if n > 0 {
                            events.push((2, owner));
                        }
                    }
                    let mut requested = vec![false; commits.len()];
                    while !events.is_empty() {
                        let ready: Vec<_> = events
                            .iter()
                            .enumerate()
                            .filter(|(_, e)| e.0 != 1 || requested[e.1])
                            .map(|(i, _)| i)
                            .collect();
                        let at = ready[random(&mut rng) % ready.len()];
                        let (kind, i) = events.remove(at);
                        if kind == 2 {
                            let seq =
                                send(&mut t, Kind::Receipt, i, source, 1, w, &mut slot, &mut rng)?;
                            receipts.receipt(i, emitted[i], accepted[i])?;
                            t.consume(seq)?;
                        } else {
                            let (chunk, winners) = &commits[i];
                            let owner = chunk.rank;
                            if kind == 0 {
                                let seq = send(
                                    &mut t,
                                    Kind::Request,
                                    owner,
                                    source,
                                    winners.len(),
                                    w,
                                    &mut slot,
                                    &mut rng,
                                )?;
                                requested[i] = true;
                                out.requests += winners.len() as u64;
                                t.consume(seq)?;
                            } else {
                                let seq = send(
                                    &mut t,
                                    Kind::Response,
                                    source,
                                    owner,
                                    winners.len(),
                                    w,
                                    &mut slot,
                                    &mut rng,
                                )?;
                                for child in winners {
                                    let regenerated = graph.successor(parent, child.mv)?;
                                    if regenerated != child.state
                                        || hash.hash(&regenerated)? != child.hash
                                    {
                                        return Err("MATERIALIZE_IDENTITY".into());
                                    }
                                    receipts.served(owner, child.mv as u64)?;
                                }
                                rings[owner].materialized(chunk.id)?;
                                archive(
                                    owner,
                                    chunk.id,
                                    c.delayed_archive,
                                    &mut rings,
                                    &mut t,
                                    &mut pending_archive,
                                )?;
                                out.responses += winners.len() as u64;
                                t.consume(seq)?;
                            }
                        }
                    }
                    if !receipts.closed() {
                        return Err("UNCLOSED_ORIGINS".into());
                    }
                    rings[source].release_origins(parent_chunk.id)?;
                    next.extend(commits.into_iter().map(|x| x.0));
                }
                t.work(source, false)?;
            }
            if c.profile == Profile::HashFirst {
                rings[parent_chunk.rank].enumerated(parent_chunk.id)?;
            }
            rings[parent_chunk.rank].reclaim();
        }
        for (rank, id) in pending_archive.drain(..) {
            rings[rank].archived(id)?;
            t.work(rank, false)?;
        }
        for (rank, ring) in rings.iter_mut().enumerate() {
            ring.reclaim();
            t.close_source(rank)?;
        }
        let f = t.issue()?.ok_or("FINALIZE_NOT_READY")?;
        if f.kind != Kind::Finalize {
            return Err("FINALIZE_KIND".into());
        }
        for r in 0..w {
            t.complete(r, f.seq)?;
        }
        if !t.finished() {
            return Err("FINALIZE_NOT_COMPLETE".into());
        }
        out.tickets = f.seq + 1;
        let expected: usize = owners.values().map(|o| o.accepted().len()).sum();
        if expected != next.iter().map(|x| x.states.len()).sum::<usize>() {
            return Err("COMMIT_COUNT".into());
        }
        for chunk in &next {
            rings[chunk.rank].publish(chunk.id)?;
        }
        previous = current_hashes;
        current = next;
        if current.is_empty() {
            break;
        }
        t.advance_depth()?;
    }
    Ok(out)
}
