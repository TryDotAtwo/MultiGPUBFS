# Authoritative visited, stale replicas, and advisory filters

Distributed BFS often considers replicating visited information to reject
duplicates before network routing. Whether that is exact depends on the
direction of possible replica error and on the requested output.

This note separates authoritative state from monotone stale exact replicas,
probabilistic filters, and local duplicate caches. It does not propose a cache
protocol or implementation.

## One authority linearizes discovery

Within one fixed graph/identity/ownership epoch, let

```text
V(t) = set of semantic states authoritatively accepted by time t.
```

For ordinary BFS visited is monotone:

```text
t1 <= t2  implies  V(t1) subseteq V(t2).
```

An owner-computes design assigns every state `x` to one authority. Concurrent
candidate records for `x` converge there, and one exact decision linearizes

```text
x notin V -> accept and insert
x in V     -> reject as already visited.
```

Replicas can answer some membership questions sooner. They do not automatically
linearize first discovery: two devices consulting separate stale copies can
both believe a new state is absent.

## A stale exact replica is a sound subset

Suppose replica `R_r(t)` contains only exact entries previously accepted by the
authority and updates may be delayed but never invented. Under the same epoch,

```text
R_r(t) subseteq V(t).
```

Then its answers are asymmetric:

```text
x in R_r      -> definitely already accepted
x notin R_r   -> unknown globally; update may be missing
```

Staleness creates false negatives relative to current authoritative visited,
not false positives. A positive remains true forever because visited is
monotone within the epoch.

This property fails across reset, graph/identity version change, owner migration,
or corrupted/invented entries. Epoch identity is part of every positive proof.

## When a sound positive may drop a candidate

For the narrow output of exact distances and one arbitrary shortest parent, a
candidate whose semantic state is already authoritatively accepted can be
dropped before routing only when that acceptance also proves a recoverable
frontier/expansion duty and the required parent payload is published or
helpably publishable. It cannot create a new vertex, but Boolean novelty alone
does not guarantee that the winning record will ever be expanded or replayed.

The same early drop may be wrong for richer outputs:

- all shortest parents need every previous-layer edge into the child;
- shortest-path counts need equal-depth predecessor contributions;
- labeled-word enumeration may distinguish duplicate state arrivals;
- deterministic parent selection may require a global tie rule;
- history-sensitive futures mean the visible key is not the whole state.

Therefore even a logically true `already visited` answer is safe only relative
to the state/output contract. Note 11 supplies the predecessor-output boundary
and note 20 the product-state boundary.

### Diamond trace: true `seen` can still carry new output

Use the directed diamond

```text
s -> a -> t
s -> b -> t.
```

Both `a` and `b` lie in `F_1`. Suppose the owner first accepts proposal
`(a,t)`, so a sound exact replica now answers `t is already visited` when
`(b,t)` is generated.

- For reached membership, distance, and one arbitrary shortest path, dropping
  `(b,t)` preserves the requested result only if `(a,t)` has a recoverably
  published expansion/path duty.
- For all-parent output, it loses predecessor `b`.
- For shortest-path count, it changes the answer from two to one.
- For a canonical parent order preferring `b` over `a`, it preserves a valid
  path but returns the wrong canonical path.

The positive fact certifies only that `t` needs no second vertex insertion. It
does not say that the arriving record has no new parent, label, count, or order
contribution. Early filtering must therefore project records onto the exact
output algebra before deciding which information is idempotent.

There is also a publication failure independent of rich output. If `(a,t)` won
the visited claim but its frontier/parent payload was lost before publication,
then dropping `(b,t)` can leave `t` marked visited with no live expansion duty
or replayable path. A replica positive is safe for early rejection only when it
names or implies the stronger state

```text
accepted and published/expanded,
or accepted with a live helpable publication obligation.
```

Note 178 gives the corresponding `ABSENT -> CLAIMED -> PUBLISHED -> EXPANDED`
state machine.

## A stale negative cannot claim novelty

If `x notin R_r`, another rank may already have sent `x` to the owner, or the
owner may have accepted it before the replica update arrived. The generating
rank must still route/consult the authority unless it has another exact local
linearization mechanism.

Using replica absence to accept `x` locally creates split authority:

```text
rank A: x absent locally -> accepts x
rank B: x absent locally -> accepts x
```

Both decisions can be individually consistent with their snapshots and jointly
violate global unique visited.

Replica negative is therefore useful as a prediction that an owner check may
succeed, not as the owner check itself.

## Dense exact bitmaps

If every state has a proved dense rank, a replicated bitmap can represent a
sound subset of visited:

```text
bit[i]=1 only after exact authority accepts rank i.
```

Monotone delayed OR/update propagation preserves the one-sided property:

- stale zero causes extra routing/work;
- propagated one can safely identify a known state for compatible outputs;
- no replica may independently turn zero into a globally final claim without
  the authority protocol.

The bit index must be exact. A many-to-one fingerprint bitmap can set a bit for
one state and falsely classify another as seen, destroying the subset property.

Replication costs approximately `N` bits per GPU for an `N`-state universe,
plus transport/update metadata. It may be feasible for dense explicit graphs
and impossible for wide implicit spaces without a practical rank.

## Bloom filters have a different error direction

An ideal Bloom filter over inserted set `K` has

```text
negative -> definitely not in K
positive -> maybe in K, maybe false positive.
```

If the filter is also a stale replica of only some authoritative visited
entries, its negative means merely

```text
not in the replicated inserted subset,
```

not `not in current global visited`. Its positive is still probabilistic.

Thus a stale distributed Bloom filter generally supplies neither final decision:

- positive cannot safely discard an exact-BFS candidate;
- negative cannot safely accept it without authority;
- both can influence batching, prefetch, routing priority, or expected traffic;
- an exact authority lookup remains decisive.

An up-to-date Bloom negative can prove absence from the synchronized snapshot,
but another concurrent claimant may still win before insertion. It avoids an
exact **read** only if a separate atomic/owner claim still resolves the race.

### `Z_4` trace: the two wrong final decisions fail differently

At root zero with generators `{1,3}`, exact BFS must construct

```text
F_1={1,3}.
```

Suppose an approximate bit/filter position set while processing state `1` also
matches unequal state `3`. A false positive interpreted as final `seen` drops
`3`, removing a real depth-one vertex. If the same many-to-one bit remains set,
later paths cannot repair the omission; the filter has changed exact BFS into
lossy reachability.

Now consider the opposite error direction: an exact but stale replica has not
yet received the authority's accepted entry for state `1`, so it answers
negative to a duplicate occurrence. Routing that occurrence to the authority
causes only extra work and the owner rejects it exactly. Accepting it locally as
globally new instead creates a second claimant/frontier record.

The safe directions are therefore not symmetric:

```text
approximate positive -> may be false -> cannot final-drop
stale exact negative -> may be outdated -> cannot final-accept
```

Only a sound exact positive may support early rejection, and only an exact
linearized claim may establish novelty, each still relative to the output
contract.

## A decision table

| Local information | What it proves | Safe semantic action |
|---|---|---|
| authoritative exact positive | state already accepted | drop state record if output permits |
| authoritative exact negative followed by atomic claim | this claimant won/lost novelty | commit according to claim result |
| stale exact-replica positive | already accepted in same monotone epoch | early drop if output permits |
| stale exact-replica negative | absent from this replica only | route/check authority |
| Bloom positive | maybe represented | never final-drop for exact BFS |
| stale Bloom negative | absent from replicated subset | route/check authority |
| fingerprint match | possible same key/state | perform exact comparison/authority lookup |

This table assumes no deletion. Dynamic/restarted search needs versioned
semantics before any cached answer is reused.

## Local pre-dedup is another limited authority scope

A source rank can exactly deduplicate equal candidate records within its local
batch before routing. That decision proves only that several local occurrences
represent one candidate state. It does not prove that the state is globally new
or old.

Local pre-dedup is safe for state-frontier membership because at least one record
still reaches the owner. It may be unsafe for all-parent/path-count outputs if
the merged occurrences carry distinct required predecessor contributions.

Across source ranks, equal candidates remain separate until they meet at their
common owner. Note 51 and REF-010 show how increasing rank count moves more of
this convergence to the owner side.

## Best-effort filters and exact fallback

A best-effort replica can be exact overall when it only removes work in a
one-sided-safe direction and every ambiguous case reaches exact fallback.

Examples:

```text
sound exact positive -> optional early reject
anything else        -> authoritative exact check.
```

or

```text
Bloom negative -> skip expensive local exact cache probe, still owner-claim
Bloom positive -> exact cache/owner comparison.
```

By contrast,

```text
Bloom positive -> drop before authority
```

is approximate BFS. The false-positive probability is a search-completeness
risk, not merely a cache-miss-rate statistic.

The term "best effort" should name performance coverage, never weaken the
semantic fallback silently.

## Bidirectional meeting caches

In owner-computes bidirectional BFS, using the same owner map for both directions
makes intersection authoritative after routing: the owner can compare exact
forward and reverse visited records locally.

A stale exact replica of the opposite side can behave as a hint:

- sound positive can identify a genuine previously accepted opposite state and
  form a candidate meeting, subject to exact parent/version data;
- stale negative can miss/delay a meeting but must not certify disjointness;
- Bloom positive can be a false meeting and needs exact confirmation;
- no cache-local miss participates in the global stopping lower bound.

Delayed meeting detection can add work without losing optimality if every
intersection is eventually checked authoritatively before termination. Dropped
checks can miss the best path.

## Replica state does not prove termination

Global exhaustion requires all authoritative owners to complete the level and
all in-flight candidates to be delivered/claimed. Replicas may lag arbitrarily:

- stale zeros do not imply unfinished graph work;
- local all-zero next-frontier replicas do not prove global empty;
- replica convergence after authority completion is not necessarily required
  for search correctness;
- pending replica broadcasts are relevant to termination only if future logic
  depends on receiving them.

The termination protocol must count the messages that can create authoritative
work. Advisory cache updates are a separate channel unless their absence can
change semantic decisions—in which case they are no longer merely advisory.

## Ownership and cache epochs

A cached positive is sound only under matching:

```text
graph/move version
semantic identity/canonicalization
rank/hash encoding
source/target and search direction where relevant
ownership epoch/world size
visited generation/reset identifier
output contract.
```

After restart or world-size change, an old positive can refer to a state accepted
in a different search or under a different graph. Monotonicity does not cross
epochs.

Safe reuse needs explicit invalidation, namespace/version tags, or reconstruction
from the restored authoritative checkpoint. Clearing only authorities while
leaving old replica bits creates false positives immediately.

## Communication trade-off vocabulary

For a replica/filter experiment, record:

```text
authoritative candidates
sound replica-positive early rejects
replica-negative candidates routed
Bloom positives confirmed true/false
extra messages from stale negatives
required parent/count records suppressed or retained
replica update bytes and latency
replica memory per GPU
fallback exact checks
epoch/version mismatches
final frontier/state parity with an exact no-filter oracle.
```

Saved candidate bytes must be compared with replica-update bytes and memory.
Even then the result is workload/topology-specific; the semantic decision table
remains invariant.

## Counterexamples and rejected shortcuts

### A replicated visited bitmap eliminates the owner

Separate replicas cannot atomically linearize one global first discovery without
another coherence/authority mechanism.

### Cache negative means globally unseen

A delayed authoritative update can be absent locally while the state is already
visited elsewhere.

### Bloom positive is safe because false positives are rare

Any false positive can delete the only entrance to a reachable region; rarity
does not make the BFS exact.

### Exact positive can always discard the record

All-parent, path-count, labeled-word, or deterministic-parent outputs may need
the duplicate arrival metadata.

### Replica convergence proves level completion

Termination is about authoritative work and in-flight candidate claims, not
about every advisory copy becoming current.

### Old visited bits remain sound after restart

Monotonicity is scoped to one graph/identity/search epoch; stale positives from
another epoch can be false in the current traversal.

## Sources and evidence

- Burton Bloom,
  [Space/Time Trade-offs in Hash Coding with Allowable Errors](https://doi.org/10.1145/362686.362692),
  defines the false-positive membership trade-off.
- Duane Merrill, Michael Garland, and Andrew Grimshaw,
  [Scalable GPU Graph Traversal](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf),
  includes best-effort bitmap filtering in explicit multi-GPU traversal.
- Friedemann Mattern,
  [Asynchronous Distributed Termination](https://doi.org/10.1007/BF01840392),
  supplies the distinction between local state and globally quiescent work.
- Notes 7, 11, 18, 20, 28, 30, 44, 51 and REF-010 provide the owner authority,
  output, termination, product-state, filter, epoch/checkpoint, source-paper,
  ownership, and bidirectional-routing context used here.

## Current conclusions

1. A monotone stale exact replica is a sound subset of authoritative visited:
   positive is true, negative is globally unknown.
2. A Bloom positive is never a final exact equality decision, and a stale Bloom
   negative describes only the replicated subset.
3. Replicas can avoid work only in a one-sided-safe direction with exact
   authority fallback for every ambiguous case.
4. Even a true already-visited answer may not permit dropping predecessor/path
   metadata required by richer outputs.
5. Replica caches do not replace global first-discovery linearization,
   bidirectional meeting confirmation, or termination detection.
6. Cache soundness is scoped to an immutable graph/identity/ownership/search
   epoch; restart or repartition requires invalidation or reconstruction.
