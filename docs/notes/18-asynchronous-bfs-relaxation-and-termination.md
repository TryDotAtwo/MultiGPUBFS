# Barrier-free BFS: first claim, relaxation, and global quiescence

Removing a level barrier does not produce one uniquely defined algorithm. There
are two fundamentally different disciplines:

```text
irrevocable first discovery  + arbitrary schedule  -> can be wrong
distance relaxation          + correction/fairness -> can converge to BFS distances
```

The second computes the same unit-edge shortest-path labels, but its proof and
intermediate state are no longer the ordinary metric-ball induction.

## The asynchronous relaxation rule

Maintain a tentative label `D[v]`, initially

```text
D[source] = 0
D[v] = infinity otherwise.
```

Processing an edge `u -> v` proposes

```text
p = D[u] + 1
D[v] = min(D[v], p).
```

Every successful strict decrease must eventually make `v` active again so its
better label can propagate through outgoing edges. In a parallel implementation
the minimum decision must be logically atomic or owner-serialized. Physical
messages and work items may be delayed or reordered.

This is a unit-weight label-correcting shortest-path computation. Calling it
"asynchronous BFS" describes its final metric, not ordinary first-discovery
frontier semantics.

## The two-sided correctness proof

Assume every finite label is accompanied by a real source-to-vertex witness
path of that length.

### Labels never undershoot

The source label is exact. If `D[u]` witnesses a real path, then proposal
`D[u]+1` witnesses that path followed by edge `u->v`. Therefore

```text
D[v] >= delta(source,v)
```

for every finite tentative label. Asynchrony may make labels too large, never
artificially smaller than graph distance.

### Quiescent fair relaxation cannot remain too large

At true termination, every improvement has propagated and no edge can still
relax its destination:

```text
D[v] <= D[u] + 1 for every reachable edge u->v.
```

Apply these inequalities along a shortest path of length `k` from the source:

```text
D[v] <= k = delta(source,v).
```

Together with the witness lower bound, `D[v]=delta(source,v)`.

The proof uses global quiescence and fairness, not FIFO order or completed
frontiers.

## Required assumptions

1. Every proposal corresponds to a real edge and witness label.
2. Distance updates implement exact minimum; worse arrivals cannot overwrite a
   better label.
3. Every successful decrease is eventually processed or transmitted.
4. Messages/work are not permanently lost.
5. A stale work item cannot mark a vertex permanently settled.
6. Termination includes passive workers **and** empty/in-accounted channels.
7. The state space and execution satisfy a finiteness/convergence condition.

Bounded message delay is sufficient but stronger than necessary; eventual
delivery and fair activation are the semantic core. Performance may depend
strongly on delay even when correctness does not.

## Why first claim fails without ordered layers

Use

```text
s -> a -> x
s -> b -> c -> x.
```

If the long branch runs first, `x` is claimed at depth three. An irrevocable
visited bit then suppresses the later depth-two route. Atomicity ensures only
one winner; it does not ensure the winner is shortest.

Level-synchronous BFS makes first claim final because all depth-`d` proposals
arrive before depth-`d+1` work. An arbitrary asynchronous schedule removes that
premise.

### Correct multi-depth seeds do not restore the FIFO invariant

Correct stored labels alone are insufficient for first-claim recovery. Use the
undirected edges

```text
s--p--a--v
s--b--v
```

whose true labels include `D(s)=0`, `D(a)=2`, and `D(v)=2`. Suppose recovery
seeds one FIFO with the correctly labeled records `[s:0,a:2]` and initially
marks both states visited. Even though the seed records are sorted, popping
`s:0` appends `p:1` and `b:1` behind the already queued `a:2`:

```text
after expanding s:0:  [a:2,p:1,b:1].
```

Expanding `a` next can irrevocably claim `v` at three. The later proposal
`b:1 -> v:2` is then rejected by first-claim visited, leaving a wrong label.

The usual FIFO proof assumes its queue was produced continuously from a
nondecreasing-distance history. It does not apply to an arbitrary mixture of
old depth buckets. Safe continuation must restore a global nondecreasing-key
discipline or use relaxations in which a later decrease reactivates the
vertex. This is a correctness distinction, not a recommendation of one
recovery implementation.

### Exact offset seeds have a positive recovery theorem

Let `A` contain the original source `s`, and initialize every `a in A` with its
exact old label `delta(s,a)`. If all seeded work is expanded under a global
nondecreasing-key discipline, the induced distance field is

```text
H(v) = min_(a in A) [delta(s,a) + dist(a,v)].
```

For every `a`, concatenating a shortest `s`-to-`a` path with an `a`-to-`v`
path gives

```text
delta(s,v) <= delta(s,a) + dist(a,v),
```

so `delta(s,v) <= H(v)`. Because `s in A`, choosing `a=s` gives
`H(v) <= delta(s,v)`. Hence `H(v)=delta(s,v)` wherever the original distance
is finite.

This is an offset multi-source shortest-path argument, not the ordinary FIFO
continuation proof. It requires the original source (or another seed set proved
to preserve the same lower envelope), exact initial labels, complete pending
expansion, and nondecreasing settlement or corrective relaxation. Omitting `s`
can change the field: exact labels on a sparse seed subset need not reproduce
distances to vertices whose shortest routes do not pass through that subset.

### A completed frontier is an exact offset cut for the exterior

The ordinary completed frontier is a special seed subset that need not contain
`s`. Fix a completed layer `F_d`. For any vertex `v` with
`delta(s,v)=k>=d`, every shortest `s`-to-`v` path has a prefix of length `d`;
let its endpoint be `a`. Then `a in F_d` and the remaining suffix has length
`k-d`. Consequently

```text
delta(s,v) = d + min_(a in F_d) dist(a,v).
```

The `>=` direction is the triangle inequality for every `a`; the `<=`
direction is witnessed by the depth-`d` prefix of a shortest path. Thus
records `a:d` for `a in F_d` are sufficient additive seeds for every vertex
outside the completed inner ball.

They are not a replacement for visited history. If `B_d` is forgotten,
outgoing edges may return to earlier layers and the restarted propagation may
relabel or re-expand the interior. Exact continuation therefore uses `F_d` as
the pending offset cut and `B_d` as the closed exclusion set (or retains an
equivalent certificate). The frontier explains where outward proof continues;
the ball explains what must not become new again.

## `atomicMin` without reactivation is also insufficient

Extend the graph with `x -> y`. Suppose `x` is first labeled `3`, expanded, and
labels `y=4`. Later `atomicMin` corrects `x` to `2`, but `x` is not re-enqueued.
Then `y` remains `4` although its true distance is `3`.

Updating stored labels is not enough. Every decrease that can improve
descendants must propagate. Equivalent designs may coalesce several decreases,
but they need a proof that the best pending value is eventually expanded.

## Stale work is not necessarily incorrect

A queued record `(u, old_label)` may run after `D[u]` has improved.

- Using `old_label+1` emits a valid but possibly weak witness proposal.
- Reading current `D[u]+1` emits a stronger proposal.
- Discarding the record is safe only if work for the newer label is guaranteed.

Stale work changes work volume and timing. It becomes a correctness problem when
its execution overwrites a better value, its discard loses the only propagation
of an improvement, or it writes inconsistent parent metadata.

## Parent records need versioned consistency

Distance `D[v]=k` and parent `p[v]` should satisfy

```text
p[v] -> v
D[p[v]] = k-1
```

for the final labels. If distance and parent are updated by unrelated races, a
better distance can coexist with the parent of an older longer proposal.

Possible semantic contracts are:

- update `(distance,parent,move)` as one winning version;
- store proposal/version metadata and validate after convergence;
- compute distances first and reconstruct a valid predecessor in a second
  pass.

All-shortest-parent DAGs require collecting every edge satisfying the final
distance equation, which is naturally a post-convergence or version-aware task.

## Message coalescing and duplicate suppression

In level BFS, two occurrences of one state in the same next frontier can be
deduplicated by identity. In asynchronous relaxation, two messages for the same
state may carry different labels.

Correct coalescing keeps the minimum proposal:

```text
(v,7), (v,4) -> (v,4).
```

Suppressing the second message merely because `v` was seen before freezes a
nonminimal distance. A visited bitmap is therefore replaced by tentative
distance state plus an activation/version policy.

The physical owner may reduce many messages before enqueueing one item, but a
later smaller proposal must still be able to reactivate the vertex.

## What termination means

For a level-synchronous traversal, a global empty next frontier after a completed
round proves exhaustion. In an asynchronous system, all local queues can be
momentarily empty while a relaxation message is in flight.

Full termination requires a consistent global predicate such as

```text
all workers passive
and no unaccounted work/message can make a worker active.
```

This is the distributed termination-detection problem. Credit/message-counting,
diffusing-computation trees, or consistent snapshots are possible mechanisms;
the BFS proof only needs their predicate to be sound.

Removing per-level barriers can therefore replace many simple synchronizations
with a harder final/global quiescence protocol. "No barrier" does not mean "no
global coordination."

## Target stopping is harder than target discovery

The first label assigned to target `t` is only an upper bound `mu`. Even after a
later improvement, another shorter causal chain may remain active or in flight.

Safe early stopping needs proof that no outstanding work can create a target
label below `mu`. For unit edges, one possible certificate is that the minimum
label of all active and in-flight work is at least `mu` and all smaller labels
have propagated. Computing that minimum consistently may itself act like a
global epoch/bucket boundary.

Global quiescence proves final distance to every reachable vertex. It is safe
but may do much more work than a target-specific lower-bound certificate.

## Finite versus infinite graphs

On a finite graph with nonnegative integer labels and fair strict-decrease
propagation, each vertex can improve only finitely often after receiving a
finite label, and quiescence yields the fixed point.

On an infinite locally finite graph, a finite-depth target may still be found,
but global quiescence need never occur because new vertices remain reachable.
The same solution-completeness versus decision-termination distinction from
note 09 applies. Fairness must also prevent an infinite stream of unrelated
work from starving the finite causal chain to the target.

## Single-GPU conceptual consequences

A persistent work queue without epochs needs more than a visited CAS:

- tentative distances rather than irrevocable seen bits;
- reactivation after successful decreases;
- capacity accounting for repeated work;
- a quiescence protocol that includes producers not yet visible to consumers;
- version-safe parent handling;
- a target lower-bound certificate if stopping before full convergence.

These obligations may outweigh a saved kernel-launch barrier. That is a
measurement question, not a reason to assume either schedule is universally
better.

## Multi-GPU conceptual consequences

With authoritative vertex ownership, each owner can apply exact minimum to all
received proposals. Correctness additionally needs:

- eventual routing of every improvement to the owner;
- re-sending outgoing consequences of an owner-side decrease;
- handling out-of-order duplicate labels by minimum, not first arrival;
- termination accounting for device queues, host staging, collectives, and
  network messages in flight;
- failure/retry semantics that do not lose the minimum proposal;
- a globally valid target bound rather than a rank-local hit.

A rank reporting "idle" describes its current local state, not termination of
the distributed relaxation.

## Work and performance semantics

Barrier-free convergence can perform much more than `|E|` useful relaxation
work because vertices may be improved and expanded repeatedly. Relevant counts
are:

```text
attempted proposals
successful strict decreases
reactivations
stale work items
messages coalesced
edges re-expanded after improvements
maximum and total label overestimation
time/bytes after first target upper bound
termination-detection traffic.
```

Final distance equality alone does not show work efficiency. Conversely, extra
relaxations are not a correctness failure when all final labels and declared
outputs are exact.

## Counterexamples and rejected shortcuts

- **Atomic first claim makes arbitrary scheduling safe.** Atomicity resolves a
  race, not shortest-order finalization.
- **`atomicMin` alone is enough.** Descendants remain stale if a decrease does
  not reactivate propagation.
- **State-only message dedup is safe.** It can discard a later smaller label.
- **All queues empty means termination.** Messages or device-produced work may
  still be in flight.
- **First target discovery is shortest.** It is only an upper bound without a
  global lower-bound certificate.
- **Correct final distances imply valid parents.** Parent/version races can
  leave an edge inconsistent with the final label.
- **Barrier-free means synchronization-free.** Global quiescence and stopping
  still require coordination.

## Audit checklist

1. Are discoveries irrevocable, or can labels decrease?
2. Does every successful decrease reactivate its outgoing propagation?
3. Can a worse stale message overwrite a better label?
4. Does coalescing retain the minimum proposal per state?
5. What fairness and eventual-delivery assumptions are explicit?
6. Are parent/move records tied to the winning distance version?
7. Which objects count as in-flight work for termination?
8. What certificate makes a target label final?
9. Can capacity overflow lose a reactivation rather than merely delay it?
10. Are extra relaxations and post-hit work measured separately from unique
    reached states?

## Sources

- Nancy Lynch, MIT 6.852J *Distributed Algorithms*,
  [Lecture 12](https://ocw.mit.edu/courses/6-852j-distributed-algorithms-fall-2009/d72e861cb207400c3496d81bdabd3f0e_MIT6_852JF09_lec12.pdf),
  for asynchronous BFS with corrections and its relation to diffusing
  computation termination.
- Friedemann Mattern, *Asynchronous distributed termination—parallel and
  symmetric solutions with echo algorithms*, Algorithmica 5 (1990),
  [DOI](https://doi.org/10.1007/BF01840392), for termination over asynchronous,
  potentially non-FIFO communication.
- Guy Blelloch et al., *A Work-Efficient Parallel Breadth-First Search
  Algorithm*, SPAA 2010,
  [DOI](https://doi.org/10.1145/1810479.1810534), for the distinction between
  benign schedule nondeterminism and correctness in a structured parallel BFS.
- Notes 03, 08, 09, and 12 supply the first-claim counterexample,
  target-stopping bounds, infinite-graph termination distinction, and general
  relaxation vocabulary used here.

## Current synthesis

Ordinary parallel BFS makes first discovery final by preserving layer order.
Barrier-free asynchronous search can reach the same distances only by changing
the contract: labels are tentative, smaller proposals win, improvements
reactivate propagation, and termination is global quiescence including in-flight
work. This is not a free removal of barriers; it is a move from metric-ball
induction to a fair relaxation fixed-point proof.
