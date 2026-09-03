use super::*;
pub struct ConcurrentSimulation {
    pub result: Simulation,
    pub peak_batches: usize,
    pub peak_tickets: usize,
    pub steps: usize,
    pub state_peak_records: Vec<u64>,
}
struct Batch {
    source: usize,
    extent: u64,
    parent: Vec<u8>,
    remaining: Vec<usize>,
    accepted: Vec<u64>,
    emitted: Vec<u64>,
    receipts: BatchReceipts,
}
enum Message {
    Candidate {
        batch: usize,
        key: (usize, usize),
        children: Vec<Child>,
    },
    Request {
        batch: usize,
        job: u64,
    },
    Response {
        batch: usize,
        job: u64,
    },
    Receipt {
        batch: usize,
        owner: usize,
        accepted: u64,
    },
}
impl Message {
    fn bucket(&self) -> Option<(usize, usize)> {
        match self {
            Self::Candidate { key, .. } => Some(*key),
            _ => None,
        }
    }
    fn batch(&self) -> usize {
        match self {
            Self::Candidate { batch, .. }
            | Self::Request { batch, .. }
            | Self::Response { batch, .. }
            | Self::Receipt { batch, .. } => *batch,
        }
    }
}
enum Action {
    Admit,
    Issue,
    Ack(usize, u64),
    Consume(u64),
    Archive(usize),
}
fn enqueue(
    t: &mut Transport,
    messages: &mut BTreeMap<u64, Message>,
    slot: &mut u64,
    kind: Kind,
    src: usize,
    dst: usize,
    count: usize,
    w: usize,
    msg: Message,
) -> Result<()> {
    let mut counts = vec![0; w];
    counts[dst] = count as u64;
    t.offer(kind, src, *slot, counts)?;
    messages.insert(*slot, msg);
    *slot += 1;
    Ok(())
}
/// Event-driven CPU schedule with bounded admitted parent batches. CUDA kernels
/// are atomic semantic jobs here; rank ACK, owner, response, D2H and admission
/// are independent events, not wall-clock GPU timing.
pub fn run_concurrent(g: &MatrixGroup, c: &Config, window: usize) -> Result<ConcurrentSimulation> {
    g.validate()?;
    let w = c.rank_map.len();
    if window == 0
        || !w.is_power_of_two()
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
    let hasher = GemmHash::from_seed(g.start.len(), c.seed)?;
    let location = |h: Hash128| {
        let p = h.prefix(bits) as usize;
        (c.rank_map[p / c.buckets], p % c.buckets)
    };
    let mut rings = (0..w)
        .map(|_| StateRing::new(c.ring_records, 65536))
        .collect::<Result<Vec<_>>>()?;
    let slots = window
        .checked_mul(g.generators.len())
        .ok_or("MODEL_SLOT_BOUND")?;
    let mut t = Transport::new(w, slots, g.generators.len() as u64)?;
    let mut rng = c.schedule;
    let mut slot = 0;
    let mut archives = vec![];
    let rank = location(hasher.hash(&g.start)?).0;
    let e = rings[rank].reserve(1)?;
    rings[rank].materialized(e.id)?;
    rings[rank].publish(e.id)?;
    archive(
        rank,
        e.id,
        c.delayed_archive,
        &mut rings,
        &mut t,
        &mut archives,
    )?;
    let mut current = vec![Chunk {
        rank,
        id: e.id,
        states: vec![g.start.clone()],
    }];
    let mut previous: BTreeMap<(usize, usize), Vec<Hash128>> = BTreeMap::new();
    let mut out = ConcurrentSimulation {
        result: Simulation {
            layers: vec![],
            generated: 0,
            committed: 0,
            requests: 0,
            responses: 0,
            tickets: 0,
        },
        peak_batches: 0,
        peak_tickets: 0,
        steps: 0,
        state_peak_records: vec![],
    };
    loop {
        let mut layer: Vec<_> = current.iter().flat_map(|x| x.states.clone()).collect();
        layer.sort();
        out.result.layers.push(layer.clone());
        let mut hashes: BTreeMap<(usize, usize), Vec<Hash128>> = BTreeMap::new();
        for s in layer {
            let h = hasher.hash(&s)?;
            hashes.entry(location(h)).or_default().push(h);
        }
        let mut owners = BTreeMap::new();
        for rank in 0..w {
            for bucket in 0..c.buckets {
                let key = (rank, bucket);
                owners.insert(
                    key,
                    OwnerModel::new(
                        previous.get(&key).cloned().unwrap_or_default(),
                        hashes.get(&key).cloned().unwrap_or_default(),
                        c.bucket_capacity,
                    ),
                );
            }
        }
        let parents: Vec<_> = current
            .iter()
            .flat_map(|x| {
                x.states
                    .iter()
                    .enumerate()
                    .map(move |(i, s)| (x.rank, x.id, s.clone(), i + 1 == x.states.len()))
            })
            .collect();
        let mut admitted = 0;
        let mut batches: BTreeMap<usize, Batch> = BTreeMap::new();
        let mut messages: BTreeMap<u64, Message> = BTreeMap::new();
        let mut live: BTreeMap<u64, (Option<u64>, Vec<bool>)> = BTreeMap::new();
        let mut jobs: BTreeMap<u64, (Chunk, Vec<Child>)> = BTreeMap::new();
        let mut job_id = 0;
        let mut next = vec![];
        while !t.finished() {
            out.steps += 1;
            if out.steps > 5_000_000 {
                return Err("MODEL_PROGRESS_LIMIT".into());
            }
            let mut actions = vec![Action::Issue];
            if admitted < parents.len() && batches.len() < window {
                actions.push(Action::Admit);
            }
            for r in 0..w {
                if let Some((&seq, _)) = live.iter().find(|(_, x)| !x.1[r]) {
                    actions.push(Action::Ack(r, seq));
                }
            }
            for (&seq, (msg, acks)) in &live {
                if let Some(id) = msg {
                    if acks.iter().all(|x| *x) {
                        let bucket = messages[id].bucket();
                        let blocked = bucket.is_some()
                            && live
                                .range(..seq)
                                .any(|(_, x)| x.0.is_some_and(|p| messages[&p].bucket() == bucket));
                        if !blocked {
                            actions.push(Action::Consume(seq));
                        }
                    }
                }
            }
            for i in 0..archives.len() {
                actions.push(Action::Archive(i));
            }
            let action = actions.swap_remove(random(&mut rng) % actions.len());
            match action {
                Action::Admit => {
                    let bid = admitted;
                    let (source, extent, parent, last) = parents[bid].clone();
                    admitted += 1;
                    t.work(source, true)?;
                    if c.profile == Profile::HashFirst {
                        rings[source].hold_origins(extent)?;
                    }
                    let mut groups: BTreeMap<(usize, usize), Vec<Child>> = BTreeMap::new();
                    let mut unique = BTreeSet::new();
                    let mut emitted = vec![0u64; w];
                    for mv in 0..g.generators.len() {
                        let state = g.successor(&parent, mv)?;
                        let h = hasher.hash(&state)?;
                        out.result.generated += 1;
                        if c.prededup && !unique.insert(h) {
                            continue;
                        }
                        let key = location(h);
                        emitted[key.0] += 1;
                        groups
                            .entry(key)
                            .or_default()
                            .push(Child { hash: h, state, mv });
                    }
                    let mut remaining = vec![0; w];
                    for (key, children) in groups {
                        remaining[key.0] += 1;
                        enqueue(
                            &mut t,
                            &mut messages,
                            &mut slot,
                            Kind::Candidate,
                            source,
                            key.0,
                            children.len(),
                            w,
                            Message::Candidate {
                                batch: bid,
                                key,
                                children,
                            },
                        )?;
                    }
                    let receipts = BatchReceipts::new(&emitted)?;
                    batches.insert(
                        bid,
                        Batch {
                            source,
                            extent,
                            parent,
                            remaining,
                            accepted: vec![0; w],
                            emitted,
                            receipts,
                        },
                    );
                    if last {
                        rings[source].enumerated(extent)?;
                        rings[source].reclaim();
                    }
                    if admitted == parents.len() {
                        for r in 0..w {
                            t.close_source(r)?;
                        }
                    }
                    out.peak_batches = out.peak_batches.max(batches.len());
                }
                Action::Issue => {
                    if let Some(ticket) = t.issue()? {
                        out.result.tickets = ticket.seq + 1;
                        let msg = if ticket.kind == Kind::Finalize {
                            None
                        } else {
                            Some(ticket.slot)
                        };
                        live.insert(ticket.seq, (msg, vec![false; w]));
                        out.peak_tickets = out.peak_tickets.max(live.len());
                    }
                }
                Action::Ack(rank, seq) => {
                    t.complete(rank, seq)?;
                    live.get_mut(&seq).unwrap().1[rank] = true;
                }
                Action::Archive(i) => {
                    let (rank, id) = archives.remove(i);
                    rings[rank].archived(id)?;
                    rings[rank].reclaim();
                    t.work(rank, false)?;
                }
                Action::Consume(seq) => {
                    let id = live.remove(&seq).unwrap().0.unwrap();
                    let msg = messages.remove(&id).ok_or("MODEL_MESSAGE")?;
                    let bid = msg.batch();
                    let batch = batches.get_mut(&bid).ok_or("MODEL_BATCH")?;
                    match msg {
                        Message::Candidate { key, children, .. } => {
                            let incoming: Vec<_> = children.iter().map(|x| x.hash).collect();
                            let survivors = owners[&key].clone().commit(seq, &incoming)?;
                            let mut winners: Vec<_> = survivors
                                .iter()
                                .map(|h| children.iter().find(|x| x.hash == *h).unwrap().clone())
                                .collect();
                            winners.sort_by_key(|x| x.mv);
                            let target = if winners.is_empty() {
                                None
                            } else {
                                Some(rings[key.0].reserve(winners.len() as u64)?)
                            };
                            owners.get_mut(&key).unwrap().commit(seq, &incoming)?;
                            batch.accepted[key.0] += winners.len() as u64;
                            out.result.committed += winners.len() as u64;
                            if let Some(e) = target {
                                let chunk = Chunk {
                                    rank: key.0,
                                    id: e.id,
                                    states: winners.iter().map(|x| x.state.clone()).collect(),
                                };
                                if c.profile == Profile::Dense {
                                    rings[key.0].materialized(e.id)?;
                                    archive(
                                        key.0,
                                        e.id,
                                        c.delayed_archive,
                                        &mut rings,
                                        &mut t,
                                        &mut archives,
                                    )?;
                                    next.push(chunk);
                                } else {
                                    let job = job_id;
                                    job_id += 1;
                                    enqueue(
                                        &mut t,
                                        &mut messages,
                                        &mut slot,
                                        Kind::Request,
                                        key.0,
                                        batch.source,
                                        winners.len(),
                                        w,
                                        Message::Request { batch: bid, job },
                                    )?;
                                    jobs.insert(job, (chunk, winners));
                                }
                            }
                            batch.remaining[key.0] -= 1;
                            if c.profile == Profile::HashFirst && batch.remaining[key.0] == 0 {
                                enqueue(
                                    &mut t,
                                    &mut messages,
                                    &mut slot,
                                    Kind::Receipt,
                                    key.0,
                                    batch.source,
                                    1,
                                    w,
                                    Message::Receipt {
                                        batch: bid,
                                        owner: key.0,
                                        accepted: batch.accepted[key.0],
                                    },
                                )?;
                            }
                        }
                        Message::Request { job, .. } => {
                            let (chunk, winners) = jobs.get(&job).ok_or("MODEL_REQUEST")?;
                            out.result.requests += winners.len() as u64;
                            enqueue(
                                &mut t,
                                &mut messages,
                                &mut slot,
                                Kind::Response,
                                batch.source,
                                chunk.rank,
                                winners.len(),
                                w,
                                Message::Response { batch: bid, job },
                            )?;
                        }
                        Message::Response { job, .. } => {
                            let (chunk, winners) = jobs.remove(&job).ok_or("MODEL_RESPONSE")?;
                            for child in &winners {
                                let s = g.successor(&batch.parent, child.mv)?;
                                if s != child.state || hasher.hash(&s)? != child.hash {
                                    return Err("MATERIALIZE_IDENTITY".into());
                                }
                                batch.receipts.served(chunk.rank, child.mv as u64)?;
                            }
                            rings[chunk.rank].materialized(chunk.id)?;
                            archive(
                                chunk.rank,
                                chunk.id,
                                c.delayed_archive,
                                &mut rings,
                                &mut t,
                                &mut archives,
                            )?;
                            out.result.responses += winners.len() as u64;
                            next.push(chunk);
                        }
                        Message::Receipt {
                            owner, accepted, ..
                        } => batch
                            .receipts
                            .receipt(owner, batch.emitted[owner], accepted)?,
                    }
                    let done = batch.remaining.iter().all(|&n| n == 0)
                        && (c.profile == Profile::Dense || batch.receipts.closed());
                    if done {
                        if c.profile == Profile::HashFirst {
                            rings[batch.source].release_origins(batch.extent)?;
                        }
                        rings[batch.source].reclaim();
                        t.work(batch.source, false)?;
                        batches.remove(&bid);
                    }
                    t.consume(seq)?;
                }
            }
        }
        if !batches.is_empty() || !messages.is_empty() || !jobs.is_empty() || !archives.is_empty() {
            return Err("PREMATURE_FINALIZE".into());
        }
        let committed: usize = owners.values().map(|o| o.accepted().len()).sum();
        if committed != next.iter().map(|x| x.states.len()).sum::<usize>() {
            return Err("COMMIT_COUNT".into());
        }
        for chunk in &next {
            rings[chunk.rank].publish(chunk.id)?;
        }
        if next.is_empty() {
            break;
        }
        previous = hashes;
        current = next;
        t.advance_depth()?;
    }
    out.state_peak_records = rings.iter().map(StateRing::peak_records).collect();
    Ok(out)
}
