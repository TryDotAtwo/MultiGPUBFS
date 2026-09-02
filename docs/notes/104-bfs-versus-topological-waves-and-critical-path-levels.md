# BFS versus topological waves and critical-path levels

Processing a directed acyclic graph in waves of currently ready vertices looks
like frontier BFS. The two procedures nevertheless compute different
quantities. BFS asks when at least one path first reaches a vertex; dependency
levelization asks when all predecessors can have completed.

This note isolates the `min` versus `max` semantics. It adds no scheduler or
implementation.

## 1. Two frontiers on the same DAG

Let `G=(V,E)` be a finite DAG and let `S` be all indegree-zero vertices.

Multi-source BFS from `S` assigns

```text
delta(v) = min length of a directed path from any source in S to v.
```

Kahn's topological procedure repeatedly removes all currently indegree-zero
vertices. If removals are grouped into synchronous waves, assign

```text
lambda(v)=0                                      if v is a source,
lambda(v)=1+max_(u->v) lambda(u)                 otherwise.
```

The vertex appears in Kahn wave `lambda(v)`. This is the maximum number of
edges on a source-to-`v` directed path, not the minimum.

## 2. Why the Kahn recurrence is a maximum

A non-source vertex becomes ready only after every predecessor has been
removed. If predecessor `u` is removed in wave `lambda(u)`, then `v` cannot be
ready before one wave after the latest predecessor. Conversely, after that
latest wave, all predecessors are gone, so `v` is ready.

Therefore

```text
K_r = {v : lambda(v)=r}.
```

Induction also shows that `lambda(v)` is the length of a longest directed path
from any source to `v`. The number of nonempty waves is one plus the longest
path length when every vertex takes one synchronous unit.

## 3. BFS is a minimum recurrence

For the same source set,

```text
delta(v)=0                                      if v is a source,
delta(v)=1+min_(u->v) delta(u)                  otherwise.
```

One reached predecessor supplies a witness path. Other predecessors do not
delay discovery. Thus the two semantics are:

```text
BFS discovery:       exists a reached predecessor;
dependency readiness: every predecessor has completed.
```

Since every source-to-`v` path counted by the minimum is also among those
counted by the maximum,

```text
delta(v) <= lambda(v).
```

The gap measures variation in source-to-vertex path lengths, not an error in
either layering.

## 4. Minimal counterexample

Take the DAG

```text
s -> v
s -> u -> v.
```

BFS discovers `v` at distance one through the direct arc:

```text
delta(v)=1.
```

But `v` cannot become dependency-ready until `u` has completed, so

```text
lambda(v)=2.
```

Calling both numbers "the level of `v`" without the recurrence silently
changes the meaning of a level.

## 5. When the two layerings coincide

For one vertex `v`, equality `delta(v)=lambda(v)` holds exactly when all
directed paths from any source to `v` have the same length.

They coincide for every vertex when the DAG is graded from its sources: there
is a rank `r` with every arc satisfying

```text
r(v)=r(u)+1.
```

Then every source-to-vertex path has length `r(v)`, and both BFS and Kahn waves
recover that rank. A shortest-path DAG created from exact BFS levels has this
property by construction because it retains only arcs from depth `i` to
`i+1`.

An arbitrary DAG need not be graded. Kahn ranks increase strictly along every
arc, but an arc can skip several Kahn waves when another, longer predecessor
chain controls its endpoint.

## 6. Neither wave sequence is a unique topological order

A topological order is a linear ordering in which every arc points forward.
Kahn waves define a coarser partition: vertices in one wave can be ordered
arbitrarily relative to one another. Different tie-breaking choices produce
different valid linear orders while preserving the synchronous wave ranks.

BFS depths are not generally a topological rank at all. For an arc `u->v`, BFS
guarantees only

```text
delta(v) <= delta(u)+1.
```

An alternative short path can put `v` in an earlier BFS layer than `u`, even
though the arc direction remains `u->v`. Sorting arbitrary DAG vertices by BFS
distance can therefore violate topological order.

## 7. Cycle certificates differ

If Kahn removal stops while vertices remain, the residual subgraph has no
indegree-zero vertex and contains a directed cycle. This is a global DAG test
when initialized over every vertex.

If a BFS frontier becomes empty, BFS has merely exhausted the vertices
reachable from its source set. Empty BFS says neither that the whole directed
graph is acyclic nor that every vertex was considered.

Conversely, visiting a back or same-depth arc under BFS is not by itself a
directed-cycle certificate. The parent/ancestor and orientation conditions
from ordinary cycle-detection arguments still matter.

## 8. Weighted task durations

Topological wave number models unit-duration barrier rounds. If vertex `v` has
duration `p(v)`, earliest completion follows a max-plus recurrence such as

```text
C(v) = p(v) + max_(u->v) C(u),
```

with an appropriate base value for sources. The maximum completion time is the
critical-path lower bound on any precedence-respecting schedule.

This is not weighted shortest path. Replacing `max` with `min`, or using BFS
edge count as makespan, changes a universal dependency constraint into one
chosen route.

Resource limits can make actual makespan exceed the critical path: ready tasks
may contend for processors, memory, or communication. The recurrence is a
dependency lower bound, not a complete hardware schedule.

## 9. Frontier accounting is different

Both algorithms may maintain arrays and waves, but their state transitions
differ:

- BFS `visited` prevents rediscovery after one accepted witness;
- Kahn maintains a remaining-predecessor count;
- a BFS candidate may be accepted on the first valid incoming occurrence;
- a Kahn task becomes ready only when its last unfinished predecessor reports;
- duplicate BFS parents concern state identity and tied shortest paths;
- repeated Kahn predecessor notifications concern exact dependency accounting.

Using a single generic "frontier processed" counter hides which invariant made
the next wave legal.

## 10. Cayley and search interpretation

Ordinary Cayley and puzzle state graphs contain inverse moves and cycles, so
Kahn levelization does not apply to them directly. A DAG may arise after a
semantic transformation:

- a BFS shortest-path DAG;
- a selected parent tree;
- a dependency graph for evaluating states or tables;
- a condensation graph of strongly connected components.

The levels then belong to that derived graph. They must not be reported as
distances in the original Cayley graph unless the transformation preserves the
required shortest-path semantics.

## 11. GPU and multi-GPU interpretation

Both frontiers can expose parallel work, but their synchronization traffic is
different:

- BFS routes candidate state identities and resolves first discovery;
- Kahn waves propagate completed dependency counts;
- Kahn readiness may be delayed by the last predecessor/owner;
- BFS depth is delayed by completion of the current metric frontier;
- weighted tasks and resource contention can dominate nominal wave count;
- stale or duplicate completion messages can release a dependency too early if
  counts are not exact.

Throughput comparisons need separate denominators: expanded transitions,
completed tasks, dependency arcs retired, and critical-path progress are not
interchangeable. No execution policy is selected here.

## 12. Evidence checklist

1. Original graph or a derived DAG.
2. Source set and whether all vertices are included.
3. `min` distance recurrence or `max` dependency recurrence.
4. One predecessor witness versus all predecessors complete.
5. Wave partition versus a chosen linear topological order.
6. Unit or weighted task durations.
7. Cycle, exhaustion, or reachability stopping certificate.
8. Logical critical path versus resource-constrained runtime.

## Sources

- A. B. Kahn, [*Topological sorting of large
  networks*](https://doi.org/10.1145/368996.369025), Communications of the ACM
  5(11) (1962), 558-562. Indegree-removal formulation of topological sorting.
- Notes 03, 04, 11, 13, 18, 21, 25, 41, 47, 56, 74, 84, and 91 provide BFS
  levels, frontier identity, shortest-path DAGs, multi-source semantics,
  asynchronous schedules, certificates, fixed points, distance labels,
  work/span, termination, discovery, condensation, and reverse context.

## Takeaway

BFS and Kahn waves share a frontier-shaped schedule but solve opposite path
aggregations. BFS takes the earliest witness through `min`; dependency
levelization waits for the latest predecessor through `max`. They coincide on
graded DAGs, including a properly constructed BFS shortest-path DAG, but not on
arbitrary dependency graphs. Naming the recurrence is more informative than
calling both procedures level-order traversal.
