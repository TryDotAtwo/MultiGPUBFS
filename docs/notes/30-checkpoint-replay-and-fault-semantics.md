# Exact BFS checkpoint/restart: visited is not enough

Fault tolerance is not obtained by periodically writing the visited bitmap.
Recovery must restore a globally consistent search state in which every reached
but unexpanded vertex still has durable work, every parent/output record agrees
with its visited claim, and messages in flight are either recorded or safely
recreated.

This note studies semantics only.  It does not design a checkpoint subsystem.

## First declare the fault model

Different failures require different claims:

- **fail-stop worker/process:** computation halts and volatile state disappears;
- **lost or duplicated message:** transport/retry semantics affect work delivery;
- **torn durable write:** only part of a checkpoint becomes persistent;
- **silent corruption:** stored bits/records can change without an explicit
  failure signal;
- **graph or generator mutation:** restart input is not the same graph;
- **ownership/topology change:** surviving workers use a different partition.

A checkpoint protocol that handles a stopped GPU process need not detect silent
state corruption.  "Fault tolerant" must name the failures and the promised
output.

## The clean level-boundary checkpoint

The simplest exact checkpoint occurs at a completed logical BFS boundary.  One
valid form records

```text
B_d     = every state at distance at most d
F_d     = every state at distance exactly d, not yet expanded
distance/parent metadata through B_d
no candidate/message from an earlier level in flight.
```

Restart expands `F_d` to construct `F_(d+1)`.  An equally valid convention can
checkpoint after that construction, recording `B_(d+1)` and the unexpanded
`F_(d+1)`.  The manifest must say which side of the expansion was committed.

The boundary is valuable because the BFS induction already supplies a compact
recovery invariant:

- accumulated visited equals a complete ball;
- the pending frontier is its newest shell;
- every earlier shell is fully expanded;
- no smaller-depth contribution remains hidden in a channel.

This does not require globally simultaneous disk writes.  It requires an
atomic/consistent committed view, often represented by immutable pieces plus a
manifest installed only after every piece is durable.

## The orphaned-visited counterexample

Consider

```text
s -> a -> b.
```

While expanding `a`, the system performs:

1. durable `visited[b]=true`;
2. crash before the frontier record for `b` becomes durable.

After restart, replaying `a` generates `b`, but a boolean visited test rejects
it.  No worker ever expands `b`.  The stored reached set may even contain `b`,
yet every successor beyond `b` is lost.

Therefore this implication is false:

```text
durable visited membership  ->  durable future expansion.
```

An exact recovery scheme needs one of these semantic patterns:

- checkpoint only before the partial next level and discard/recompute all
  uncommitted claims;
- commit visited membership and pending frontier work atomically;
- store a richer lifecycle such as `UNSEEN`, `PENDING`, `EXPANDED` and recover
  every durable `PENDING` state;
- maintain a durable accepted-delta log from which the frontier is rebuilt.

The same write-order problem applies to parent metadata: a durable child with no
durable parent cannot satisfy a replayable-path contract.

## A partial level is a distributed state, not a set of files

During expansion of `F_d`, workers may hold:

- completed and uncompleted parent ranges;
- local candidate buffers;
- accepted owner-side child records;
- messages sent but not received;
- messages received but not applied;
- durable claims whose frontier/output side effects are pending.

Independently copying each worker's memory can produce an impossible cut.  For
example, a snapshot may record a send after it happened but record the receiver
before the receive, while omitting the channel message itself.  Recovery then
loses work.

A consistent distributed snapshot includes process states and the appropriate
channel states, or the protocol must recreate channel effects from a clean
ancestor.  The Chandy--Lamport model makes this distinction explicit: global
state contains both local process state and communication-channel state.

## Roll back versus roll forward

Two broad recovery semantics are useful.

### Roll back to a complete boundary

Discard every partial effect after committed `B_d,F_d` and replay the whole
level.  This can be conceptually simple because the checkpoint matches the BFS
induction.  It requires the old graph/version and deterministic availability of
all `F_d` successor work, but not identical scheduling.

### Preserve a partial cut

Keep durable accepted states and continue only missing work.  This can avoid
recomputation, but it must capture:

- which source work units are complete;
- which child states are pending versus already expanded;
- channel messages or replayable send logs;
- duplicate/contribution identifiers where updates are non-idempotent;
- consistent parent, label, count, and ownership metadata.

Preserving more bytes does not automatically make recovery more correct.  The
snapshot must form a legal prefix/cut of the declared distributed computation.

## At-least-once delivery and idempotence

Exactly-once physical execution is not necessary for every BFS output.  For
reached-set accumulation,

```text
B := B union {x}
```

is idempotent.  Receiving the same state candidate several times can remain
correct if:

- no delivery is permanently lost;
- exact identity merges the copies;
- at least one accepted copy creates durable pending expansion;
- capacity overflow is explicit;
- termination waits for all retry/in-flight obligations.

This is commonly described as at-least-once delivery plus idempotent handling.
But the entire state transition must be idempotent, not only the visited bit.
Repeated enqueue, parent append, message acknowledgement, and output writes need
their own recovery semantics.

At-most-once delivery is unsafe if a lost message may be the only route to a
new state.  A protocol marketed as exactly-once must still explain how it
couples message acknowledgement with durable visited/frontier effects; naming
the delivery mode does not prove the application transaction.

## Output contracts have different replay algebra

| Output/update | Natural merge under retry | Qualification |
|---|---|---|
| reached-state membership | set union | idempotent with exact identity |
| minimum distance | `min` | improvement must trigger required propagation |
| one arbitrary shortest parent | choose one valid equal-depth claim | may change across retries |
| deterministic canonical parent | total-order `min` over all equal-depth claims | equality boundary must complete |
| all shortest parents | set union of parent identities | append-only duplicates must be removed |
| shortest-path count | addition of predecessor contributions | not idempotent without contribution IDs/dedup |
| nearest-source label | lexicographic/min reduction | later equal-depth improvements may propagate |

Path counting gives a minimal non-idempotence witness.  If predecessor `p`
contributes `count[p]` twice after retry, naive addition doubles its paths.
Exactly the same state set and distances can survive while the requested count
is wrong.

One solution class associates a stable contribution identity such as
`(level, parent, child, edge occurrence, graph version)` and applies each once.
Another rolls the whole level back and recomputes counts from the previous
boundary.  The note does not choose between them; it records the proof burden.

## Parent durability and replay

For one path per vertex, acceptance should not become durably visible without
enough information to reconstruct a valid parent edge under the same graph
version.  Depending on the representation, that includes:

- parent state/rank and owner;
- original move/generator label;
- child and parent depths;
- left/right action and inverse convention;
- symmetry/product-state frame;
- graph/generator/legality version.

If parent records are deferred to save wire bytes, the checkpoint must retain a
durable way to recover them.  "We can ask the failed sender later" is not a
recovery proof.

For arbitrary-parent output, retries may select a different shortest parent and
still be correct.  Deterministic parent/shortlex output requires replay to
retain or recompute every contender used by the declared reduction.

## Distance levels and asynchronous recovery

At a clean level boundary, boolean first discovery remains final.  In a partial
asynchronous snapshot, messages may carry different tentative distances.  A
longer proposal restored before a shorter in-flight proposal cannot be made
irrevocable unless the protocol preserves the missing proposal or uses minimum
relaxation and reactivation.

Thus a recovered `visited=true` bit without its finalized depth/status is
insufficient outside the strict level invariant.  Note 18's fairness,
relaxation, and quiescence obligations continue across failures.

## Bidirectional and multi-source checkpoints

A bidirectional snapshot must retain both reached balls/frontiers, their exact
completed depth bounds, the best meeting upper bound, and every in-flight
contribution capable of lowering it.  Restoring only the best meeting path can
lose the lower-bound proof that made stopping safe.

A multi-source snapshot must retain source labels and equal-depth tie state if
those labels are semantic.  Distance-only recovery may be complete while
canonical Voronoi labels remain unfinished.

The checkpoint schema follows the output contract, not merely the algorithm
name.

## Ownership and topology changes

A restart may use a different number of GPUs.  Semantic states can be
repartitioned safely only if the recovery process establishes one new
authoritative owner for every exact key and migrates:

- visited/distance state;
- pending frontier work;
- parent/count/label metadata;
- logged or in-flight contributions;
- completion/acknowledgement state.

Changing `owner=hash(key) mod P` without migration can leave old and new owners
both authoritative or leave a state with neither.  The checkpoint therefore
binds an owner epoch, even if recovery deliberately creates a new epoch.

## Version manifest

An exact checkpoint should bind at least:

- source set and target/output contract;
- graph snapshot or implicit successor version;
- generator order, direction, action, legality, and inverse conventions;
- state schema, canonicalization, rank/hash version, and byte order;
- BFS level/phase convention and completed work ranges;
- owner count/function/epoch and communication protocol version;
- visited, frontier, distance, parent, label, and count artifact identifiers;
- channel/message-log state or clean-boundary proof;
- capacity limits and any overflow flags;
- checksums plus a stronger corruption policy where required;
- commit manifest/version installed last.

Restarting against a different generator set is a new search, not recovery.
Silently accepting a checkpoint with changed canonicalization can turn equal
states into duplicates or distinct states into false visited hits.

## Termination is a durable stable-property claim

A local empty queue is not termination before failure, and persisting it does
not improve the proof.  A durable `TERMINATED` record is sound only when its
underlying consistent state proves:

- all workers passive;
- no work/messages in flight;
- every accepted state either expanded or intentionally outside the output
  contract;
- the target/component stopping condition satisfied;
- no uncommitted overflow or failed owner hidden.

Dijkstra--Scholten-style termination detection tracks diffusing obligations;
Chandy--Lamport-style snapshots capture consistent global states/stable
properties.  They solve related but distinct proof problems.  A checkpoint can
record a termination detector's state, but cannot replace its accounting with
a timestamp.

## Validation after restart

Useful recovery checks include:

1. every frontier state belongs to visited at its declared depth;
2. every `PENDING` state appears in some durable work source;
3. every `EXPANDED` state has its successor work committed or replayable;
4. every non-source parent has depth one less and replays under the manifest;
5. no channel/log record belongs to an incompatible owner or graph epoch;
6. counts/parent sets have no duplicate contribution IDs;
7. the first post-restart level matches a fresh bounded exact oracle;
8. repeated crash/restart at each transaction boundary yields the same declared
   output, allowing only explicitly arbitrary parent variation;
9. a torn/incomplete manifest is rejected rather than partially loaded;
10. termination survives a full pending-work/channel audit.

Equal final visited counts are insufficient: one lost state and one duplicate
can cancel.  A replayable returned target path is also insufficient to certify
that other branches were not lost during recovery.

## Sources

- Chandy and Lamport,
  [Distributed Snapshots: Determining Global States of Distributed Systems](https://www.microsoft.com/en-us/research/publication/distributed-snapshots-determining-global-states-distributed-system/),
  formalizes consistent global snapshots as process state plus channel state
  and their use for detecting stable properties.
- Dijkstra and Scholten,
  [Termination detection for diffusing computations](https://www.cs.utexas.edu/~EWD/ewd06xx/EWD687a.PDF),
  provides obligation accounting for distributed work that can create further
  work.
- Note 15 supplies the durable level-transaction/external-memory foundation;
  notes 11, 13, 18, 22, and 28 supply parent/count, label, asynchronous,
  version, and exact-identity contracts.

## Current conclusion

The recoverable unit of exact BFS is not a visited bit.  It is a consistent
state transition connecting reached membership, pending expansion, metadata,
and distributed messages.  Clean completed-level checkpoints inherit the BFS
induction directly.  Mid-level checkpoints are possible, but only by recording
or recreating every causal obligation and by matching the retry algebra to the
requested output.
