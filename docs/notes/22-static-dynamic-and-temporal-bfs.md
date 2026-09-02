# Static, dynamic, and temporal BFS: which graph version is searched?

The ordinary BFS proof assumes one fixed transition relation `E`. If adjacency,
generator sets, or legality predicates change during traversal, the result needs
a new semantic contract. Three common contracts are different:

1. BFS of one immutable **snapshot**;
2. maintenance of distances under **dynamic updates**;
3. reachability by chronologically valid paths in a **temporal graph**.

A concurrent traversal that merely sees "whatever is current" may implement
none of them.

## Snapshot BFS

Choose a graph version

```text
G_k = (V, E_k)
```

and require every expansion and visited decision to refer to that same version.
Then all ordinary metric-ball and parent proofs apply to `G_k`.

Snapshot consistency may be provided by immutability, copy-on-write adjacency,
versioned edge reads, or an update barrier. The mechanism is secondary; the
proof needs one well-defined edge relation.

"The graph changed only a little" is not snapshot semantics. A single changed
edge can alter reachability or many downstream distances.

## A mixed-version path may belong to no snapshot

Initially let edge `s -> a` exist and `a -> t` be absent. BFS expands `s` and
discovers `a`. Then an update deletes `s -> a`, followed by another update that
inserts `a -> t`. BFS later expands its retained `a` record and discovers `t`.

It returns

```text
s -> a -> t,
```

but:

- before the updates, `a -> t` was absent;
- after the updates, `s -> a` was absent;
- no intermediate snapshot contained both edges.

Every observed edge was real at observation time, yet the combined path is not
a path in any static graph version. Per-edge validity does not imply snapshot-
path validity.

If timestamps increase along the two traversals, the same sequence **can** be a
valid temporal journey. That is a different theorem and output meaning.

## Dynamic BFS maintenance

Dynamic single-source BFS asks to maintain exact distances for a sequence

```text
G_0, G_1, G_2, ...
```

of graph versions. After update `k`, labels should equal distances in `G_k`.
One can always recompute a fresh BFS; dynamic algorithms try to repair only the
affected region while preserving the same exact result.

Terminology:

- **incremental:** insertions only;
- **decremental:** deletions only;
- **fully dynamic:** both.

These directions have different monotonicity and invalidation behavior.

## Edge insertion: labels can only improve

For `E_k subset E_(k+1)`, every old path remains valid, so

```text
dist_(k+1)(s,v) <= dist_k(s,v).
```

Old finite labels are valid path lengths and upper bounds on new shortest
distances. Previously unreachable vertices may become reachable.

For inserted edge `u -> v`, if

```text
D[u] + 1 < D[v],
```

`v` improves and the decrease may propagate through descendants/outgoing
edges. This resembles note 18's relaxation, but it starts from a previously
exact fixed point and repairs the consequences of a graph update.

An insertion that does not improve its destination cannot improve a downstream
path through that edge under ordinary unit distances. This local test depends on
the pre-update distances being exact.

### Cayley hand trace: boolean visited survives while depth becomes stale

Run BFS from zero in `Z_4` with generators `{1,3}`. After expanding the root,
the state is

```text
visited = {0,1,3},  frontier = {1,3},
D(0)=0, D(1)=D(3)=1.
```

Now add generator `2` before expanding the frontier, but do not revisit the
already expanded root. Continuing the old queue can generate state `2` from
`1+1` or `3+3` and assign

```text
D(2)=2.
```

That label is correct for the old graph but false for the new graph, where the
new root edge `0+2=2` gives `D'(2)=1`. Exact state equality and a perfectly
implemented boolean visited set do not prevent the error: the missed event is
an improved proposal from an already retired expansion under a new edge
relation.

If the old BFS had already exhausted all four states, boolean visited would
remain the complete reachable set after the insertion while every affected
distance/frontier fact could still be stale. Thus reachability monotonicity
does not provide distance finality. A correct contract must either keep one
generator snapshot for the traversal or perform dynamic distance repair under
a new graph epoch.

## Edge deletion: witnesses can become invalid

For `E_(k+1) subset E_k`, distances can only increase or become infinite:

```text
dist_(k+1)(s,v) >= dist_k(s,v).
```

Old numeric labels are lower bounds, but an old parent path may no longer exist.
This is not a decreasing relaxation problem.

If a deleted edge was one parent witness for `v`, ask whether another incoming
edge still satisfies

```text
D[u] + 1 = D[v].
```

If so, distance can remain while parent metadata changes. If not, `v` must find
a longer predecessor or become unreachable, and the increase/invalidation can
cascade to descendants.

An equality using old labels is only a provisional alternative: the candidate
predecessor's own distance may later increase in the same deletion cascade. The
surviving witness is final only under the converged post-update labels.

Keeping one BFS parent is enough to output one old path, but not enough to know
immediately whether deletion destroys all shortest witnesses. The shortest-path
DAG from note 11 exposes alternative parents, though maintaining it has its own
cost.

### Reverse `Z_4` trace: invalidation is not repair

Start instead from the completed BFS of `Z_4` with generators `{1,2,3}`. Its
Cayley graph is `K_4`, so

```text
D(1)=D(2)=D(3)=1.
```

Delete generator `2` globally. The remaining generators `{1,3}` form the old
four-cycle, and the new exact label is

```text
D'(2)=2,
```

with replacement paths `0--1--2` and `0--3--2`. The old parent edge `0--2`
is gone, so the old label is invalid. But the old shortest-path DAG cannot
produce either replacement: edges `1--2` and `3--2` joined two old depth-one
vertices and were therefore same-layer edges, not old shortest-predecessor
arcs.

This makes the two decremental questions visible on four states:

```text
did an old shortest witness survive?       no
what is the new shortest distance?         requires the full surviving graph
```

A monotone decrease operation such as `min(old, proposal)` cannot repair the
label because the correct value must increase. Boolean visited is even less
informative: all four states remain reachable before and after the deletion.

### Edge roles are source- and epoch-relative

For a surviving directed arc `u -> v`, define its old radial difference

```text
r(u,v) = D(v) - D(u).
```

It is an old shortest-predecessor arc exactly when `r=1`; in an undirected
graph `r=0` means same-layer and `r=-1` means inward relative to the chosen
orientation. After a deletion, write finite label increases as

```text
D'(x) = D(x) + Delta(x).
```

Then the same physical arc has

```text
r'(u,v) = r(u,v) + Delta(v) - Delta(u).
```

Consequently:

- an old predecessor (`r=1`) remains a new predecessor exactly when its two
  endpoints increase equally;
- an old same-layer arc (`r=0`) becomes a new predecessor when
  `Delta(v)=Delta(u)+1`;
- no static label such as “lateral edge” or “tree-direction edge” belongs to
  the edge independently of source distances and graph epoch.

In the reverse `Z_4` trace, `Delta(1)=0` and `Delta(2)=1`, so surviving edge
`1 -> 2` changes from old same-layer (`r=0`) to new predecessor (`r'=1`). The
formula classifies the role change but does not compute the deltas; those are
the dynamic BFS repair problem.

## Batch and fully dynamic updates

Insertions and deletions in one batch can interact. Applying a deletion repair
against distances that have not yet incorporated an earlier insertion may
produce a transient state belonging to neither the pre- nor post-batch graph.

A correctness contract should name:

- update order or atomic batch version;
- when queries may observe intermediate states;
- whether distances are exact after each update or only after the batch;
- how parent/path outputs are versioned;
- how update processing termination is detected.

"Eventually consistent distance" is weaker than exact BFS for a named version
unless query staleness is explicit.

## Generator changes are global graph updates

In a Cayley graph, changing generator collection `S` changes an edge at every
group element:

```text
g -> g*s for every g.
```

Adding one generator is therefore a structured global insertion, not one local
edge insertion. Distances cannot increase, but a new generator can shorten a
large fraction of the group and change diameter/frontier geometry.

Special cases require proof:

- adding identity adds loops but preserves distances;
- adding a duplicate transformation preserves unlabeled distances but changes
  labeled multiplicity;
- adding a genuinely new generator can change the word metric;
- removing an inverse can make the graph directed and distances asymmetric;
- deleting a generator can invalidate every parent edge carrying that label;
- reordering unchanged generators preserves distance but can change shortlex
  parents from note 19.

An old visited ball cannot simply continue as exact visited under new `S`.
After insertion it may omit states newly reachable within the same radius; after
deletion it may contain states whose old depth/path is no longer valid.

## Changing legality in an implicit graph

For implicit BFS, `successors(state)` is the graph. Updating an obstacle set,
move rule, collision predicate, or external resource changes adjacency even if
the state encoding is unchanged.

If legality depends on a fixed version, attach or capture that version in the
oracle. If legality depends on time/history by design, the correct model may be
the product/temporal state from note 20 rather than a mutable static oracle.

Caching successors across versions without a version key can reintroduce
deleted edges or hide inserted ones.

## Temporal graphs and journeys

A temporal edge is available at a time or interval. A valid journey uses edges
in nondecreasing time order, possibly with waiting. Different objectives include:

- **foremost:** earliest arrival time;
- **fastest:** minimum duration between departure and arrival;
- **shortest temporal:** minimum number/cost of traversed temporal edges;
- **latest departure:** depart as late as possible while meeting a deadline.

These need not choose the same journey.

A time-expanded graph can use vertices `(v,t)` and edges for travel and waiting.
Ordinary unit BFS on that expansion minimizes the number of expansion steps,
which equals elapsed discrete time only under the declared edge timing model.
If waiting should cost zero hops while travel costs one, this becomes 0-1 or
weighted shortest path rather than ordinary BFS.

Taking the static union of every temporal edge ignores chronological order and
can invent impossible paths. Taking the intersection can discard valid journeys
whose edges occur at different compatible times.

## Dynamic graph versus temporal product state

The two viewpoints answer different query patterns:

- Dynamic BFS: after each update, what are shortest paths in the **current
  snapshot**?
- Temporal BFS: what chronologically valid journey exists through the whole
  sequence of edge availability?

The mixed-version example is erroneous for snapshot BFS but potentially valid
for temporal reachability. Correctness cannot be judged until this question is
settled.

## Parent and replay versions

A parent record should identify enough version/time information to replay its
edge. Otherwise a final chain may combine individually valid parents from
incompatible graph versions.

Snapshot output can store one graph version for the whole tree. Dynamic output
may need a new tree version or persistent parent history per update. Temporal
output needs edge timestamps and verifies their chronological order in addition
to endpoint adjacency.

Checking only current adjacency during replay can reject a historically valid
temporal journey or accept a newly inserted edge that was unavailable when the
claimed snapshot path was computed.

## Concurrency and multi-GPU version obligations

In distributed traversal, graph updates themselves must have ownership and
completion semantics:

- Which owner/version authorizes an adjacency change?
- Can one rank expand version `k` while another routes candidates from `k+1`?
- Are update messages ordered with respect to candidate messages?
- When is a snapshot epoch globally installed?
- Can an old candidate be recognized and replayed under its source version?
- What global event says all distance repairs for version `k` have converged?

A level barrier coordinates search work but does not by itself make adjacency
versions consistent. Conversely, version consistency does not prove the BFS
frontier is complete.

For GPU-resident adjacency, concurrent mutation also raises memory-safety and
visibility questions, but even perfectly race-free reads can combine versions
semantically if the algorithm permits them.

## Performance vocabulary without designing an updater

Meaningful dynamic measurements include:

```text
updates per batch
vertices whose distance decreases/increases
parent-only changes versus distance changes
newly reachable and newly unreachable states
edges rescanned during repair
version-install and quiescence time
stale candidate/update messages
full recomputation baseline
peak coexistence memory for graph versions.
```

Fast update throughput is not useful if queries observe an unnamed mixture of
versions. The first correctness comparison is always against fresh exact BFS on
the intended post-update snapshot at tractable scale.

## Counterexamples and rejected shortcuts

- **Every edge observed during traversal forms one valid graph path.** Mixed
  versions can combine edges that never coexisted.
- **Insertions leave old BFS exact.** They preserve old paths but may introduce
  shorter ones and new reachability.
- **Deletions only require removing the deleted edge.** Invalid distances and
  parents can cascade arbitrarily far.
- **One stored parent proves a deletion changed distance.** Another shortest
  parent may survive.
- **Changing Cayley generators is a small local update.** It changes a
  translated edge family across the whole group.
- **The union of temporal snapshots gives temporal reachability.** It ignores
  edge-time order.
- **A search level epoch is also a graph-version epoch.** They are independent
  consistency dimensions unless explicitly coupled.

## Audit checklist

1. Is the query about one snapshot, dynamic maintenance, or a temporal journey?
2. What exact graph/generator/legality version does each frontier use?
3. Can a returned parent chain contain edges from incompatible versions?
4. Are updates insertions, deletions, batches, or fully dynamic?
5. Which old labels are upper bounds, lower bounds, or invalid witnesses?
6. After deletion, are alternative shortest parents represented?
7. Does generator change preserve distance, labels, or neither?
8. What temporal objective is minimized: arrival, duration, hops, or cost?
9. How are update convergence and search termination distinguished globally?
10. Is every dynamic result checked against fresh BFS on the named snapshot?

## Sources

- Shimon Even and Yossi Shiloach, *An On-Line Edge-Deletion Problem*, Journal of
  the ACM 28(1), 1981, the classical decremental BFS-tree foundation; modern
  context and the ES-tree definition are summarized in
  [Bernstein and Chechik](https://aaronbernstein.cs.rutgers.edu/wp-content/uploads/sites/43/2018/12/STOC_2016.pdf).
- Ulrich Meyer, *On Dynamic Breadth-First Search in External-Memory*, STACS
  2008, [paper](https://arxiv.org/abs/0802.2847), for insertion-only versus
  deletion-only dynamic BFS under another memory model.
- Othon Michail, *An Introduction to Temporal Graphs: An Algorithmic
  Perspective*, Internet Mathematics 12(4),
  [author PDF](https://cgi.csc.liv.ac.uk/~michailo/Documents/Papers/Journals/im16.pdf),
  for temporal journeys, foremost reachability, and static time expansion.
- Notes 03, 11, 18, 20, and 21 supply static layer proof, alternative shortest
  parents, asynchronous repair/termination, product time state, and exhaustive
  certificate boundaries.

## Current synthesis

Static BFS is exact for one edge relation. Dynamic BFS maintains a sequence of
such exact fixed points. Temporal BFS searches paths whose edges are ordered in
time. Reading mutable adjacency without selecting one of these meanings can
produce a path belonging to no snapshot. In Cayley search, generator version is
part of the metric definition, so changing `S` invalidates more than a cache: it
changes the graph whose BFS theorem describes.
