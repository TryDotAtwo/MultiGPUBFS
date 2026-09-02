# BFS level graphs, blocking flows, and matching phases

BFS can be valuable even when shortest-path discovery is not the final problem.
In Dinitz's maximum-flow algorithm and Hopcroft-Karp bipartite matching, BFS
builds a temporary metric skeleton of the current residual problem. A second
procedure then exhausts many shortest improvements before BFS is run again.

This note studies that phase logic. It does not implement either algorithm.

## 1. The graph changes between phases

Let `G_f` be the residual graph of a current feasible flow `f`: an arc is
present exactly when it has positive residual capacity. BFS from `s` computes

```text
ell(v) = dist_(G_f)(s,v).
```

The level graph retains residual arcs satisfying

```text
ell(v) = ell(u)+1.
```

These labels are exact only for this residual snapshot. Sending flow can remove
forward residual capacity and create reverse residual arcs. Consequently the
next phase must not treat the old labels or old `visited` set as permanent
facts about the updated graph.

This differs from ordinary static BFS:

```text
static BFS:      visited grows monotonically in one fixed graph;
phased residual: distances are rebuilt after the graph changes.
```

## 2. What the level graph preserves

Let `L=ell(t)` while the sink is reachable.

Every `s-t` path in the level graph has exactly `L` arcs, because its level
starts at zero and rises by one on every retained arc. Conversely, every
shortest residual `s-t` path has levels `0,1,...,L` along its vertices, so all
its arcs belong to the level graph.

Thus the level graph contains exactly the residual shortest-path corridor in
the following sense:

- all shortest `s-t` paths are retained;
- every retained `s-t` path is shortest;
- vertices or edges not lying on any `s-t` path may still be retained;
- reachability from `s` alone does not prove usefulness for reaching `t`.

A reverse BFS from `t` inside the retained graph can identify the intersection
of forward-reachable and sink-co-reachable vertices, but Dinitz correctness
does not require calling every forward level vertex useful.

## 3. Blocking means every shortest route is hit

A blocking flow in the level graph leaves no residual `s-t` path using only
level-increasing arcs. Equivalently, at least one arc on every level-graph
`s-t` path becomes saturated by the phase's augmentation.

This is stronger than finding one shortest augmenting path. One path may share
little capacity with other shortest paths, leaving the old shortest distance
unchanged after its augmentation. A blocking flow exhausts the entire current
shortest-path layer as a family.

It need not be a maximum flow in the level graph. The semantic requirement is
that no complete `s-t` path remains there.

## 4. Why the next shortest distance increases

Before the blocking phase, every residual arc `u->v` obeys

```text
ell(v) <= ell(u)+1,
```

by the BFS triangle inequality. New reverse arcs created by sending flow along
an admissible arc go from level `i+1` to level `i`.

Suppose the updated residual graph had an `s-t` path of at most `L` arcs. To
rise from level zero to the old sink level `L` in at most `L` steps, every step
would have to increase the old level by exactly one. It could use neither a
same/decreasing-level arc nor a newly created reverse arc. Therefore it would
be a surviving path in the old level graph, contradicting blocking.

Hence, if `t` remains reachable in the next residual graph,

```text
new dist(s,t) > L.
```

The proof uses both ingredients: exact BFS levels and exhaustion of all paths
consistent with those levels.

## 5. Empty BFS is a different certificate here

If residual BFS cannot reach `t`, there is no augmenting `s-t` path. For a
feasible flow this is the max-flow stopping condition; the reached residual set
also supplies the source side of a minimum cut under the usual max-flow/min-cut
theorem.

The certificate is not "the original graph has no path." It is:

```text
the current residual graph has no positive-residual s-t path.
```

Residual capacities, direction, and the current flow are part of the semantic
input to that BFS.

## 6. Hopcroft-Karp uses the same shape with different state

For a bipartite graph and current matching `M`, orient or traverse edges
according to alternating-path legality:

- from unmatched left vertices, start simultaneously;
- unmatched edges go from the left side toward the right side;
- matched edges return from the right side toward the left side;
- free right vertices are targets.

This is a multi-source, multi-target BFS in a graph defined by the current
matching. It finds the minimum length of an augmenting path. A depth-first
phase then constructs a maximal collection of vertex-disjoint shortest
augmenting paths and augments along all of them.

After that phase, the shortest augmenting-path length strictly increases. When
no augmenting path exists, Berge's lemma says the matching is maximum.

The parallel with blocking flow is structural, not identity:

```text
Dinitz:         capacities and a blocking flow in a residual level graph;
Hopcroft-Karp:  matching alternation and disjoint shortest augmenting paths.
```

## 7. Two tempting but false shortcuts

### One BFS can serve all later augmentations

False. An augmentation changes residual directions/capacities or matching
status. Old distances can cease to describe legal paths. Reusing them without
a preservation proof mixes different graph snapshots.

### One shortest path exhausts the shortest-distance phase

False. Consider two internally disjoint shortest `s-t` paths with independent
capacity. Saturating one leaves the other, so the residual `s-t` distance need
not increase. The phase-progress proof needs blocking, or in matching a
maximal family of disjoint shortest augmenting paths, not merely one witness.

## 8. Frontier and stopping subtleties

The first observation of `t` proves its current distance, but a phase that
wants all shortest improvement routes must retain all relevant vertices and
arcs through the preceding levels. Stopping in the middle of a frontier merely
because one target occurrence was generated can omit tied shortest routes.

A safe bounded BFS may avoid exploring vertices beyond level `L` once `L` is
known, while still completing all work needed below `L` and collecting the
declared level-graph edges. "Found the sink" and "finished constructing the
shortest-path phase" are different stopping contracts.

Within a phase, a search procedure can mark a level-graph branch dead after it
proves that branch cannot reach `t` under the remaining capacities. Across
phases, those marks are not permanent state reachability facts.

## 9. Work interpretation

The reason for BFS batching is amortization: one `O(E)`-scale layer construction
can support many improvements of the same minimum length. The classical bounds
come from proving that only a limited number of distinct shortest lengths can
appear, not from claiming that BFS itself computes the flow or matching.

Counts must stay separate:

- vertices/arcs scanned by phase BFS;
- level-graph arcs retained;
- shortest routes represented;
- augmentations actually selected;
- flow value or matching cardinality gained;
- number of later phases avoided.

One large frontier can represent many useful disjoint improvements or enormous
overlap around one bottleneck. Frontier width alone predicts neither gain.

## 10. GPU and multi-GPU reading

The phase structure exposes parallel BFS work, but also global dependencies:

- residual/matching state must be a coherent snapshot for level construction;
- all owners need consistent levels at the depth boundary;
- tied shortest routes can span partitions;
- the blocking or disjointness phase mutates shared resources;
- termination requires global absence of unfinished admissible paths;
- the next BFS cannot safely begin from a mixture of old and new residual state.

Therefore fast level generation alone is not end-to-end evidence. Measurements
would have to separate BFS, route construction, mutation/conflict handling,
phase synchronization, and achieved progress. No GPU design is selected here.

## 11. Evidence checklist

1. Exact residual or alternating graph snapshot used by BFS.
2. Multi-source and target-set semantics where applicable.
3. Whether all shortest paths are represented, not only the first target hit.
4. Exact definition of blocking or maximal disjoint augmentation.
5. Proof that the old shortest length is exhausted.
6. Rebuild boundary after residual or matching mutation.
7. Final no-augmenting-path certificate and its governing theorem.
8. Separate BFS work, augmentation work, and objective gain.

## Sources

- E. A. Dinitz, [*Algorithm for solution of a problem of maximum flow in
  networks with power
  estimation*](https://www.mathnet.ru/eng/dan35701), Soviet Mathematics
  Doklady 11 (1970), 1277-1280; Russian original in Doklady AN SSSR 194(4),
  754-757. Layered residual networks and blocking-flow phases.
- J. E. Hopcroft and R. M. Karp, [*An `n^(5/2)` algorithm for maximum matchings
  in bipartite graphs*](https://doi.org/10.1137/0202019), SIAM Journal on
  Computing 2(4) (1973), 225-231. Shortest augmenting-path phases and the
  `O((m+n)sqrt(n))` bound stated in the paper.
- Notes 03, 04, 08, 11, 13, 18, 22, 25, 40, 48, 56, 57, 74, 75, 89, and 91
  provide level, frontier, stopping, DAG, multi-source, dynamic, fixed-point,
  reverse, separator, distributed termination, orientation, and gateway
  context.

## Takeaway

In phased flow and matching algorithms, BFS does not solve the optimization
problem. It freezes the current shortest-improvement geometry. Exhausting every
improvement compatible with that geometry makes the next shortest length grow;
then mutation invalidates the old layers and BFS starts again. This is a clean
example of BFS as a temporary proof-producing subroutine rather than a
once-for-all traversal.
