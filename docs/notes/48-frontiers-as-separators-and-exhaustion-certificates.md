# BFS frontiers as separators and exhaustion certificates

An exact BFS frontier is more than a collection of active work. A completed
distance sphere separates the source from every vertex farther away, and an
empty next boundary certifies that the reached set is successor-closed.

These statements explain why level boundaries are useful checkpoints and why a
beam or partial frontier cannot inherit the same negative guarantees.

## Every farther path crosses every intermediate sphere

Let

```text
B_d = {v | dist(s,v) <= d}
S_d = {v | dist(s,v) = d}.
```

Consider any directed path from `s` to a vertex `t` with `dist(s,t)>d`. Walk
along that path and choose its first vertex `x` outside `B_(d-1)`. Its
predecessor `p` lies in `B_(d-1)`, so

```text
dist(s,x) <= dist(s,p)+1 <= d.
```

Since `x` is outside `B_(d-1)`, its distance is also at least `d`. Therefore

```text
dist(s,x)=d,
```

and the path intersects `S_d`.

Thus `S_d` is an `s`-separator from all vertices beyond depth `d`: removing the
whole sphere destroys every directed source path to those farther vertices.
The proof uses only unit arcs and exact source distances; symmetry is not
required.

The same path may visit a sphere more than once, move backward in distance, or
use same-level arcs. The theorem promises at least one crossing, not a monotone
distance sequence for every arbitrary path.

## The frontier is not necessarily a minimum cut

A separator need not be smallest. Use

```text
s -> a
a -> b_1, ..., b_k
b_i -> t_i.
```

The depth-two sphere is

```text
S_2 = {b_1,...,b_k},
```

but removing the single vertex `a` already separates `s` from every `t_i`.
Hence a BFS sphere can be much larger than a minimum source-to-region vertex
cut.

Conversely, a small sphere is a genuine bottleneck even if later layers are
huge. Menger-type theorems relate minimum separators to the number of disjoint
paths, but BFS does not compute a minimum separator merely by exposing a layer.

The distinction matters for memory: frontier width is the size of a particular
metric separator, not graph connectivity or minimum-cut value.

## A sphere is not generally an antichain

Vertices in one BFS layer have equal minimum source distance. They need not be
incomparable under reachability.

For example,

```text
s -> u
s -> v
u -> v.
```

Both `u` and `v` lie in `S_1`, yet `u` reaches `v`. In an undirected graph, a
same-level edge similarly makes the two vertices mutually reachable.

Therefore these are distinct statements:

- `S_d` is an equal-distance set;
- `S_d` separates `s` from deeper vertices;
- `S_d` is an antichain under some chosen partial order.

The third does not follow from BFS and may not even be meaningful when graph
reachability contains cycles.

## The completed ball is the object behind the separator

For directed BFS, the external out-boundary is

```text
partial_out(B_d) = {v notin B_d | exists u in B_d: u -> v}.
```

Unit-edge distance gives the exact identity

```text
partial_out(B_d) = S_(d+1).
```

So the next frontier is determined by the entire completed ball, even when an
implementation expands only `S_d`: edges from earlier layers cannot lead to a
new outside vertex after those layers were completed.

This is why exact visited history matters. The physical current frontier says
which transitions remain to be generated; the accumulated ball says which
candidate identities are already inside the proved region.

## Empty next frontier proves successor closure

After completely expanding every state in `S_d`, suppose exact identity yields

```text
S_(d+1) = empty.
```

Then

```text
partial_out(B_d) = empty.
```

No directed edge leaves `B_d`. Since `s` is in `B_d`, any source path must stay
inside `B_d`. Therefore

```text
B_d = Reachable(s).
```

This is the structural content of `EXHAUSTED`: the accumulated reached set is
successor-closed, not merely that a queue happened to be empty at one instant.

For a target absent from `B_d`, closure proves unreachability from `s`. On an
infinite locally finite component, no finite `B_d` can become closed, so this
certificate never occurs even though each individual level finishes.

## Why partial emptiness proves nothing global

The closure proof requires complete generation and exact identity. It fails if
any of the following remains possible:

- an unexpanded state still exists in the current sphere;
- a generated candidate is buffered or in flight;
- a state was dropped by capacity, timeout, or cancellation;
- a false-positive visited decision suppressed a new state;
- one owner is locally idle while another has work;
- graph/version changes can add a successor after the boundary was checked.

A momentarily empty queue is an execution state. A completed empty boundary is
a graph certificate.

## Why a beam is not a separator

Let

```text
s -> a -> dead
s -> b -> t.
```

The exact depth-one sphere is `{a,b}`. A width-one beam that retains only `a`
is a subset of the separator, but removing/expanding `{a}` does not intercept
the path through `b`. When that retained branch ends, its empty next beam says
only that the selected branch produced no survivor.

In general, `K subset S_d` is a separator only if an independent proof shows
that every relevant source-to-goal path intersects `K`. Score ranking, width,
or membership in the correct depth does not provide that proof.

This gives a geometric restatement of note 24:

```text
complete sphere retention -> original-graph separator guarantee
beam subset               -> heuristic path cover unless separately certified.
```

Exact K1/K2 suffix lookup cannot repair the missing prefix separator. It can
prove properties of retained states, not that discarded sphere states contain
no solution branch.

## Frontier representations versus frontier semantics

The same mathematical separator `S_d` may be represented as:

- a queue/list of exact state records;
- a bitmap over a proved dense rank;
- sorted unique keys plus exact state payloads;
- owner-sharded sets across devices;
- an external-memory run.

Representation order, duplicates before commitment, and physical sharding do
not change the separator if the union is exactly `S_d`. Conversely, a compact
representation with one false positive or dropped shard may cease to be the
separator even when its cardinality looks plausible.

Validation therefore needs set equality or coverage evidence, not only a
frontier count.

## Level-boundary checkpoints

A clean BFS checkpoint after completing level `d` conceptually records:

```text
graph/state-identity version
completed ball B_d or exact visited representation
complete frontier S_d (or S_(d+1), depending on phase)
distance/parent/output metadata required by the contract
proof that no candidate or owner message from the preceding phase is missing.
```

This boundary is attractive because all undiscovered source paths must cross
the frontier before entering the remaining graph. Restoring the exact visited
and frontier sets reconstructs the same logical cut.

Saving only frontier states without visited permits rediscovery through cycles.
Saving only visited after claims but before durable frontier publication can
create unexpanded orphan states, as note 30 shows. A valid checkpoint must
capture both sides of the commit boundary consistently.

## Multi-GPU union and global closure

Suppose owner `r` stores a shard `S_d^(r)`. The separator is the global union

```text
S_d = union_r S_d^(r),
```

not any one shard. Global exhaustion requires:

```text
all frontier shards expanded
all send/receive and spill buffers drained
all owner-side exact claims committed
union_r S_(d+1)^(r) = empty
all loss/overflow/version flags clear.
```

An all-reduce of local frontier counts is meaningful only after the phase
protocol ensures that no message can subsequently create another accepted
state. Otherwise zero is a transient distributed snapshot, not closure.

Owner partitions themselves are not BFS separators. They are storage/routing
assignments; a source path may cross owner boundaries many times within one or
several distance layers.

## Bidirectional interpretation

A forward sphere separates `s` from deeper forward states. A reverse sphere in
the transposed graph separates states farther from `t` from the target. Their
completed balls and the absence/presence of connecting arcs support the lower
and upper bounds in bidirectional stopping proofs.

A local meeting is an upper-bound witness. Proving it optimal still needs a
global statement that every shorter source-target path would have crossed the
already completed forward/reverse separators. Note 8 supplies the exact bound;
the separator view explains its geometry.

## Counterexamples and rejected shortcuts

### Every BFS frontier is a minimum cut

A narrow earlier bottleneck can separate the source from a much wider later
sphere.

### Same-distance vertices form an antichain

Same-level directed or undirected edges can connect them.

### An empty local queue proves exhaustion

Other workers, buffers, messages, or unfinished current-layer states may still
cross the logical boundary.

### Any subset of the frontier remains a separator

Removing one branch of a two-branch sphere leaves paths through the discarded
state. Beam membership has no automatic cut guarantee.

### Frontier count equality validates the separator

One missing and one spurious state can preserve cardinality while opening a
real source path not represented by the stored set.

### Frontier alone is a complete checkpoint

Without exact visited history, cycles and alternative prefixes can regenerate
old states; without atomic frontier publication, visited can contain orphaned
unexpanded states.

## Sources

- Reinhard Diestel,
  [Graph Theory](https://diestel-graph-theory.com/),
  provides separator, connectivity, and Menger-theorem background.
- L. R. Ford and D. R. Fulkerson,
  [Maximal Flow Through a Network](https://doi.org/10.4153/CJM-1956-045-5),
  supplies the flow/cut perspective; BFS metric spheres are particular cuts,
  not automatically minimum cuts.
- Notes 8, 9, 10, 24, 30, 42, and 46 provide the local bidirectional bound,
  exhaustion semantics, boundary identity, beam counterexample, checkpoint
  atomicity, negative-result contract, and expansion context used here.

## Current conclusions

1. Every source path to a vertex deeper than `d` crosses the exact BFS sphere
   `S_d`, even in a directed unit graph.
2. A BFS sphere is a metric separator but need not be a minimum cut or a
   reachability antichain.
3. A completely generated empty next frontier proves that the reached ball is
   successor-closed and therefore equals the reachable component.
4. Partial queues, incomplete layers, beam subsets, and rank-local emptiness do
   not provide that closure certificate.
5. A level-boundary checkpoint is sound only when visited, frontier publication,
   graph version, and in-flight work form one consistent cut.
6. Distributed exhaustion belongs to the global frontier union after all
   messages and authoritative claims complete.
