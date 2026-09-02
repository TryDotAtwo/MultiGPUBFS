# BFS, A*, and bound-certified heuristic pruning

Heuristic scores can interact with a breadth-first traversal in three
fundamentally different ways:

1. order work without dropping it;
2. prove that some work cannot improve a known solution;
3. discard work by rank, width, or an unproved score.

Only the second is exact pruning, and it requires both a lower-bound proof and a
replayable incumbent upper bound. This note draws the boundary among BFS, A*,
branch-and-bound, and beam search. It proposes no implementation.

## Ordering a complete BFS layer

In a unit graph, exact BFS reaches every state in `S_d` with the same path cost

```text
g=d.
```

Sorting all members of `S_d` by a heuristic `h`, a learned score, state rank, or
memory-locality key does not change the mathematical frontier if every eligible
state is eventually processed and no capacity/timeout loss occurs.

This may change:

- which equal-depth parent wins;
- the first target candidate encountered inside the layer;
- cancellation latency;
- memory locality and duplicate convergence;
- deterministic output order.

It does not reduce the logical complete-layer work. A run that claims the full
ball `B_d` must still retain every exact layer member regardless of score.

## The lower-bound meaning of `g+h`

Consider a search record reaching concrete state `x` by a real prefix of cost
`g(x)`. Let `h(x)` be an admissible lower bound on the remaining target cost:

```text
h(x) <= dist(x,T).
```

Every solution extending that record has cost at least

```text
f(x) = g(x)+h(x).
```

The proof is direct: any such solution consists of the stored prefix plus a
suffix from `x` to a goal, whose cost is at least `h(x)`.

If `g(x)` is only a tentative overestimate rather than the cost of the actual
record being extended, `f` remains a bound for that particular prefix but may
not dominate all ways of reaching the same state. State-level pruning then also
needs the usual dominance/reopening argument.

## Incumbent upper bounds

Let `U` be the cost of a concrete replay-valid solution already found. Then

```text
optimal_cost <= U.
```

Combining bounds gives the exact branch-and-bound rule:

```text
if g(x)+h(x) >= U,
no extension of this record can produce a solution strictly cheaper than U.
```

This is proof-driven pruning. Its ingredients are asymmetric:

- `h` must never overestimate;
- `U` must come from a genuine concrete solution, not a model prediction;
- `g` must describe the concrete prefix being pruned;
- graph, target, costs, and versions must match.

A stale **larger** incumbent merely prunes less. A falsely small or unreplayable
incumbent can delete the true optimum.

## Equality depends on the requested output

With one valid solution of cost `U`, discarding records with

```text
g+h >= U
```

is safe when the only remaining objective is to find a **strictly cheaper**
solution. Once every open record has lower bound at least `U`, the incumbent is
optimal.

But equality may still contain requested information:

- another optimal path of the same cost;
- all shortest parents or path count contributions;
- a lexicographically preferred optimal word;
- a solution with equal length but better secondary cost.

For these outputs, records with `g+h=U` cannot be discarded solely by the
scalar bound. One may prune only `g+h>U`, or add a proved lexicographic/vector
bound matching the full objective.

Thus even a mathematically correct numeric bound has an output contract.

## Global optimality certificate

For one minimum-cost solution, the generic certificate is

```text
incumbent U is a replay-valid solution
and
minimum lower bound over every unprocessed feasible record >= U.
```

Then no unseen extension can beat `U`.

This is the heuristic-search analogue of a completed BFS boundary. BFS with
unit edges uses the smallest unsettled depth as its lower bound. A* uses the
smallest open `g+h` value. Bidirectional search uses a lower bound derived from
two completed regions. Different schedules are exact because they close the
same upper-versus-lower-bound gap by different certificates.

An empty open set is a special case whose lower bound is infinity; with no
incumbent it proves failure only if exploration was complete and lossless.

## When the algorithm is still BFS

Heuristic ordering remains a BFS implementation detail when:

- states are finalized by nondecreasing hop distance;
- every exact next-frontier member is retained;
- no heuristic threshold removes a state needed for the declared ball/output;
- termination uses the BFS layer bound.

If the algorithm uses `g+h` across different `g` levels to choose the next
record, it is best-first/A*, not ordinary BFS. It may remain exact under its own
proof, but its settled order and intermediate frontiers are not BFS spheres.

On unit edges with `h=0`, uniform-cost/A* priority reduces to nondecreasing `g`
and is BFS-compatible modulo ties and data structure. A nonzero heuristic can
skip ahead in hop depth because a deeper state may have smaller `g+h` than a
shallower state.

## First generated goal versus first selected goal

Generating a goal supplies an upper bound. It does not by itself prove that no
open record has a smaller lower bound.

In standard A* conditions, selecting/popping a goal whose key is no larger than
every open key closes that gap because `h(goal)=0` and its `g` is the incumbent.
Stopping when the goal is merely generated can return too early.

Minimal abstract example:

```text
s -> a -> goal            cost 2
s -> b -> c -> goal       cost 3
```

If an arbitrary schedule expands the longer branch first, generating the
length-three goal says only `U=3`. The still-open record `a` has a lower bound at
most two and must not be ignored.

Ordinary FIFO BFS avoids this error through its completed shallower-layer
invariant rather than an `f`-priority queue.

## Consistency and reopening

Admissibility bounds total remaining cost. Consistency additionally requires

```text
h(x) <= c(x,y)+h(y)
```

for every edge. Then `f=g+h` is nondecreasing along a path when `g` accumulates
the same edge costs. This supports the familiar A* property that a state removed
at its best key need not later receive a cheaper path under the standard graph
search assumptions.

With an admissible but inconsistent heuristic, optimal search is still possible,
but a state may need to be reopened when a better `g` arrives. A permanent
first-seen visited bit can freeze a more expensive prefix, just as arbitrary
asynchronous BFS first claim can freeze a longer hop label.

Therefore exact graph-search A* must state:

- whether `g` improvements are allowed after discovery;
- whether closed states reopen;
- how stale queue records are recognized;
- when a goal's cost is final;
- how parents track the winning `g` version.

"Admissible heuristic" alone does not specify these mechanics.

## Duplicate-state dominance

Suppose two records reach the same semantic state `x` with costs

```text
g_1 < g_2.
```

If future legal moves and costs depend only on `x`, the larger-cost record is
dominated: any suffix available from it is also available after the cheaper
prefix, giving a no-worse complete solution.

This justifies keeping the best `g` per state. It fails when the visible state
omits history that changes future legality, resources, automaton phase, or the
requested labeled-path identity. Then the semantic search state must include
that context, as note 20 established.

For all-path enumeration, a higher-`g` record is irrelevant to minimum cost, but
equal-`g` records may represent distinct shortest parents/words and cannot all
be merged if those outputs are requested.

## PDB and K1 roles

An exact PDB supplies a proved `h` lower bound. It can participate in `g+h`
pruning because its abstraction proof guarantees no overestimate.

A concrete K1 hit supplies more:

```text
exact residual distance r
and a replayable suffix of length r.
```

For a retained prefix of cost `g`, this gives a concrete incumbent

```text
U = g+r.
```

The same residual is also an exact lower bound for continuations through that
state. A completed K1 miss gives only a radius lower bound such as `R+1`, under
the exact-identity assumptions from notes 40 and 42.

Neither structure makes a learned beam score admissible, and neither makes
top-k deletion a bound comparison. To prune an outer record exactly, the system
would need a certified lower bound for that record and a valid global incumbent,
not merely a relative ranking.

## Beam selection is not branch-and-bound

Beam search keeps the best `k` records even when the `(k+1)`-st record has

```text
g+h < U
```

or when no incumbent exists. That discarded record may lead to the only or the
best solution.

Conversely, branch-and-bound may retain more or fewer than `k` states depending
on their certified bounds. Its memory is not generally fixed. The defining
difference is:

```text
beam: discard because rank exceeds capacity
exact bound pruning: discard because no requested improvement is mathematically possible.
```

Calling both operations "heuristic pruning" hides the guarantee boundary.

## Multi-GPU bound coordination

In a distributed exact best-first/bound search, each rank may hold open records
with different lower bounds and discover different incumbents.

A safe global optimality decision needs:

```text
U_global = minimum replay-valid incumbent over all ranks
L_global = minimum lower bound over all active and in-flight records
terminate when L_global >= U_global
```

under the one-solution objective. The reduction must include records in device
queues, host staging, communication buffers, and owner-side pending claims.

Using an old larger `U` is safe but less effective. Pruning from an unvalidated
rank-local candidate or omitting another rank's smaller `L` is unsafe. A global
min reduction performed before all earlier messages are accounted for is not a
termination certificate.

This coordination can resemble a barrier or distributed termination protocol;
removing level synchronization does not remove the need to close the global
bound gap.

## GPU interpretation without an optimizer

Heuristic search can expose irregular buckets by `f`, repeated `g` improvements,
and remote/reopened records rather than complete BFS layers. Useful conceptual
counters include:

```text
records ordered only
records pruned by certified bound
records removed by capacity/top-k
incumbent improvements and replay status
minimum open lower bound
reopens and stale records
post-incumbent work
global-bound reduction/termination time.
```

These distinguish exact search progress from score throughput. No particular
queue, bucket, kernel, or distributed design follows from the semantics alone.

## Counterexamples and rejected shortcuts

### A good heuristic can safely prune low-ranked states

Ranking quality is empirical. Exact pruning requires a lower bound compared
with a valid upper bound.

### Admissibility alone makes first discovery final

With inconsistent `h` or arbitrary duplicate handling, a cheaper `g` may arrive
later and require reopening.

### First generated goal is optimal in A*

It is only an incumbent while any open/in-flight record has a smaller lower
bound.

### `g+h=U` can always be discarded

It can contain alternative optimal paths, parent counts, or better secondary
ties required by the output.

### K1/PDB makes the surrounding beam exact

The tables certify residual bounds for represented states; they do not cover
prefixes discarded solely by beam width.

### Local bounds are enough for multi-GPU termination

An unseen remote or in-flight record can have a smaller global lower bound.

## Sources

- Peter Hart, Nils Nilsson, and Bertram Raphael,
  [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf),
  gives the classical A* lower-bound ordering and optimality framework.
- Rina Dechter and Judea Pearl,
  [Generalized Best-First Search Strategies and the Optimality of A*](https://doi.org/10.1145/3828.3830),
  clarifies heuristic conditions and A* optimal-efficiency boundaries.
- Richard Korf,
  [Depth-First Iterative-Deepening: An Optimal Admissible Tree Search](https://www.cse.sc.edu/~mgv/csce580f09/gradPres/korf_IDAStar_1985.pdf),
  provides a contrasting thresholded use of admissible `f` bounds.
- Notes 18, 20, 24, 40, 42, and 49 provide the reopening analogy,
  history-state dominance boundary, beam distinction, K1 suffix semantics,
  bounded misses, and PDB proof used here.

## Current conclusions

1. Heuristic ordering inside a complete layer can remain BFS; cross-layer
   `g+h` scheduling is a different exact-search family with its own proof.
2. Exact pruning requires `g+h` as a lower bound and a concrete replay-valid
   incumbent `U` as an upper bound.
3. The rule at equality depends on whether the output asks for one optimum,
   every optimum, path counts, or secondary tie objectives.
4. Admissibility supports lower bounds; consistency supports monotone keys and
   simpler closed-state finalization. Inconsistent heuristics may require reopen.
5. A PDB supplies lower bounds and a K1 hit can supply an incumbent suffix, but
   neither justifies fixed-width beam deletion.
6. Distributed optimal termination requires a global minimum open/in-flight
   lower bound compared with the best validated global incumbent.
