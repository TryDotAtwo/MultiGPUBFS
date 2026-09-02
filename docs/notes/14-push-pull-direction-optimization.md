# Push and pull BFS: two evaluations of one next-layer predicate

At an exact level boundary,

```text
visited = B_d
frontier = F_d.
```

The next frontier is defined pointwise by

```text
v in F_(d+1)
iff v not in B_d
and exists u in F_d such that (u,v) is an edge.
```

Push and pull evaluate this same predicate from opposite sides of the edge.
They change enumeration work, not BFS direction or distance semantics.

## Push formulation

Push iterates the current frontier:

```text
for u in F_d:
    for v in out_neighbors(u):
        if v not in B_d:
            claim v for F_(d+1)
```

Its raw edge work is

```text
push_edges_d = sum_(u in F_d) out_degree(u).
```

Advantages:

- work is localized to the current frontier;
- it naturally fits implicit successor generation;
- a small frontier touches only a small part of the graph;
- parent/move information is present at generation time.

Costs:

- many frontier vertices can emit the same child;
- concurrent claims can contend;
- irregular explicit degrees create load imbalance;
- a large frontier can inspect many edges leading to already visited states.

## Pull formulation

Pull iterates potential destination vertices:

```text
for v not in B_d:
    for u in in_neighbors(v):
        if u in F_d:
            accept v into F_(d+1)
            break
```

It asks whether the current frontier intersects the predecessor set of each
unvisited vertex.

Unlike push, ordinary distance-only pull naturally produces at most one output
record per destination and can stop scanning `v` after the first frontier
parent.  This avoids a candidate occurrence bag and many same-child claims.

Its work is data- and order-dependent:

```text
pull_checks_d
= sum_(v not in B_d) predecessors inspected until first hit
```

or the full in-degree if no frontier predecessor exists.  There is no graph-
independent equality between this count and the number of accepted vertices.

## Exact equivalence theorem

Push and pull produce the same `F_(d+1)` if all of the following hold:

1. both use the same directed edge relation;
2. push enumerates all outgoing edges from `F_d`;
3. pull enumerates every currently unvisited vertex and all required incoming
   edges until a hit or exhaustion;
4. frontier membership is exact;
5. both modes use the same logical old ball `B_d`; a physical claim table may
   additionally accumulate a subset of `F_(d+1)` only if those claims remain in
   the final next frontier and are not expanded during this level;
6. no capacity loss or race drops an accepted vertex.

Proof is immediate from the defining existential predicate:

- push emits `v` exactly when it witnesses some edge from `F_d` to unvisited
  `v`;
- pull accepts `v` exactly when it finds the same kind of witness.

Deduplication differs physically, but the resulting vertex set is identical.

## Directed graphs require incoming adjacency

For an original edge `u -> v`, push reads `v` from `out_neighbors(u)`.  Pull,
while examining `v`, must find `u` in `in_neighbors(v)`.

Scanning `out_neighbors(v)` in pull mode tests a different predicate and can
silently traverse the wrong graph.  An explicit directed graph therefore needs
reverse adjacency or an equivalent predecessor index.

This use of incoming edges is not bidirectional search:

- pull still advances distance outward from the same source;
- backward bidirectional BFS advances a second distance field from the target
  in the reverse graph.

The words "direction" and "reverse" refer to different axes.

## Why early exit makes pull attractive

Suppose the frontier occupies a large, well-connected region.  Many unvisited
vertices may have several frontier predecessors.  Push inspects all outgoing
occurrences and resolves their duplicate child claims.  Pull can find one
predecessor early and stop.

The saving depends on predecessor order.  If a frontier parent appears first,
one check suffices; if it appears last, pull scans the whole list.  If no parent
exists, pull also scans the whole list.

Thus pull benefit depends on at least:

- number of unvisited vertices;
- their in-degrees;
- probability and position of a frontier predecessor;
- cost of frontier membership checks;
- cost of enumerating the unvisited universe.

Frontier cardinality alone is an incomplete switching signal.

## Three contrasting graph shapes

### Narrow path

On a long path, `|F_d|` is at most two.  Push inspects constant work near the
frontier.  Pull scans most unvisited vertices at every level.  Pull is
asymptotically wasteful.

### Dense graph near completion

When a large frontier touches most remaining vertices, push can generate many
duplicate occurrences.  Pull may test each remaining vertex and stop after one
or a few predecessor checks.  This is the regime motivating bottom-up BFS on
dense-frontier social/web graphs.

### High-degree frontier with a remote bottleneck

A large frontier can have many internal/back edges but reach only one new
vertex.  Push work is large.  Pull still scans every other unvisited vertex that
has no frontier predecessor, so it can also be large.  Neither frontier size nor
push edge count alone proves pull will win.

## Direction-optimizing BFS

Direction-optimizing BFS switches between top-down push and bottom-up pull while
preserving exact layers.  A typical shape is:

```text
small frontier       -> push
frontier becomes broad / push work high -> pull
remaining unvisited shrinks or frontier falls -> push again
```

The switching thresholds are cost heuristics.  They may use frontier edge
volume, unvisited edge volume, frontier size, or measured history.  Their
correctness obligation is only that either chosen mode computes the same next
frontier; their performance quality is graph- and hardware-dependent.

Beamer, Asanovic, and Patterson's direction-optimizing work demonstrates large
savings on tested explicit graph families.  It does not establish universal
thresholds or automatic applicability to implicit state spaces.

Switching has costs too:

- converting frontier list to bitmap/membership structure;
- maintaining or scanning unvisited sets;
- obtaining global work estimates;
- changing kernels/data access patterns;
- distributed frontier replication/communication.

A correct hybrid can be slower than fixed push if these costs exceed avoided
edge checks.

## Snapshot semantics matter

Pull's predicate uses `F_d`, not a frontier that grows while the same level is
being scanned.  If a newly accepted `F_(d+1)` vertex is immediately visible as
current frontier membership, pull can cascade through multiple hops in one
round and assign the wrong depth.

Similarly, visited must represent at least `B_d` before next-layer membership is
tested.  A clean implementation uses distinct current-frontier and next-
frontier state or epoch tags.

Asynchronous propagation can be correct under a different relaxation proof,
but it cannot inherit the one-level existential equivalence while mutating its
input predicate.

## Parents and all-shortest-path output

Distance-only pull stops at the first frontier predecessor.  That parent is a
valid shortest witness but depends on incoming-neighbor order.

For a canonical parent, pull must apply the declared tie rule rather than an
arbitrary break.  For the full shortest-path DAG or path counts, it must inspect
**all** frontier predecessors and retain/count their edge contributions.

This removes the central early-exit advantage for vertices with many shortest
parents.  Therefore push-versus-pull work depends on the requested output:

```text
distance / one arbitrary parent
!= deterministic parent
!= all shortest predecessors.
```

Pull can eliminate duplicate frontier membership while still needing all
parent-edge metadata.

## Multi-source labels add tie work

For scalar distance-to-set, any frontier predecessor proves the next distance.
For canonical nearest-source labels, the first predecessor may carry a larger
source ID than another frontier predecessor.  Pull must either scan all relevant
predecessors or use metadata that proves no better label remains.

Again, distance-only early exit is insufficient for richer Voronoi output.

## Why inverse generators are not enough for implicit pull

An implicit state graph usually provides

```text
given u, generate successors(u).
```

Inverse generators may additionally provide predecessors of a **known** state
`v`.  Conventional pull needs something else first:

```text
for every v not in B_d: ...
```

If the unvisited state universe cannot be cheaply enumerated, there is no outer
pull loop.  Knowing how to invert a move does not reveal which unknown states to
test.

This is the principal transfer boundary from explicit dense-ID BFS to puzzle/
Cayley BFS.

## Rankable finite state spaces

A finite implicit graph with a bijective rank can, in principle, enumerate
every rank and skip visited ones.  Pull then additionally needs:

- unranking or direct predecessor computation from rank;
- incoming generator actions;
- exact frontier membership by rank;
- a scan of much or all of the remaining universe each pull level.

Feasibility is not usefulness.  For `S_n`, a Lehmer rank makes every permutation
enumerable, but unranking every unvisited permutation merely to ask whether it
touches a modest frontier may cost far more than pushing generators from that
frontier.

A dense bitmap solves membership and enumeration addressing; it does not make
state reconstruction free.

## Cayley-specific geometry

For a finite symmetric generator set, each Cayley vertex has the same degree.
This makes push edge count predictable:

```text
push_edges_d = |S| * |F_d|.
```

It does not make pull attractive.  Pull work depends on the size of the whole
unvisited group region and on rank/unrank/predecessor costs.

Relations can give an unvisited vertex several frontier predecessors, making
pull early exit potentially useful near broad layers.  The same relations also
make push generate duplicates.  Which enumeration is cheaper remains a scoped
cost question; regularity alone answers neither.

For infinite finitely generated groups, the unvisited universe is infinite, so
a conventional full-universe pull pass is not executable.  Push remains locally
finite and solution-complete for finite-depth targets under note 09's
assumptions.

## GPU work signatures

Push and pull stress hardware differently:

| Push | Pull |
|---|---|
| frontier-sized outer parallelism | unvisited-universe outer parallelism |
| irregular outgoing edge expansion | irregular early-exit predecessor loops |
| duplicate child claims/atomics | mostly one destination writer |
| candidate/frontier compaction | unvisited filtering plus accepted compaction |
| adjacency reads near frontier | incoming adjacency reads across unvisited set |
| natural parent/move at emission | parent found during membership probe |

Pull may trade atomic contention for divergent loop lengths and extra memory
reads.  Push may expose too little parallelism on a tiny frontier; pull may
expose abundant but mostly useless scans.

Useful measurements include:

```text
push outgoing edges inspected
pull predecessor edges inspected
unvisited vertices tested
checks before first hit distribution
frontier membership bytes/cache behavior
candidate duplicate occurrences avoided
mode conversion and selection overhead
end-to-end level time.
```

No single count decides the mode across graph families or GPU generations.

## Multi-GPU pull obligations

In distributed push, candidate states naturally route to authoritative owners.
Distributed pull requires each owner of unvisited vertices to answer whether
their predecessors belong to the **global** current frontier.

Possible conceptual mechanisms include frontier replication, partition-aligned
membership exchange, or 2D adjacency decomposition.  Each adds questions:

- which ranks need which frontier membership bits?
- is incoming adjacency local?
- how are frontier conversions and broadcasts charged?
- can a rank stop predecessor scans from local information?
- when is the global `F_d` snapshot complete?

Beamer et al.'s distributed bottom-up work shows that communication structure
can be adapted for explicit partitioned graphs.  It does not remove the
enumerable-universe and predecessor-index requirements for implicit graphs.

## Counterexamples to common equivalences

### Pull is backward BFS

False.  Pull computes the next layer of the same forward distance field by
testing incoming witnesses.  Backward BFS has a different source and distance.

### Invertible moves imply pull support

False.  Inversion supplies predecessors of a known state, not enumeration of
all unknown/unvisited states.

### Pull eliminates duplicates, so it is always cheaper

It avoids duplicate destination claims but may scan the entire unvisited
universe and many failed predecessors.

### Dense rank implies efficient pull

Rank enables exact addressing.  Unranking, predecessor generation, and full-
universe scan cost remain.

### First predecessor is enough

Enough for distance and one arbitrary parent; insufficient for deterministic
ties, all shortest predecessors, path counts, or canonical multi-source labels.

### Mode choice changes correctness

Not when both modes satisfy the exact predicate and snapshot contract.  Mode
choice changes work.  Incorrect representations or partial snapshots—not the
word push/pull—break equivalence.

## Audit checklist

1. Is the intended predicate exactly `v notin B_d` and
   `exists u in F_d: u->v`?
2. Does pull use incoming edges of the original directed graph?
3. Is the entire unvisited universe enumerable and affordable to scan?
4. Is `F_d` immutable, is old-ball membership exact, and are any early
   next-frontier claims kept separate from current-level expansion?
5. What are actual push edge and pull predecessor-check counts?
6. Does pull stop at one parent, choose a canonical parent, or retain all?
7. What list/bitmap/frontier conversion cost is included?
8. For implicit states, how are ranks turned into expandable states?
9. For multi-GPU pull, where does global frontier membership reside?
10. Is switching evidence from the intended graph representation and complete
    traversal, rather than an isolated favorable level?

## Sources

- Scott Beamer, Krste Asanovic, and David Patterson,
  *Direction-Optimizing Breadth-First Search*, SC 2012,
  [paper](https://www.scottbeamer.net/pubs/beamer-sc2012.pdf),
  [doi:10.3233/SPR-130370](https://doi.org/10.3233/SPR-130370), for hybrid
  top-down/bottom-up BFS and edge-work switching on explicit graphs.
- Scott Beamer et al., *Distributed Memory Breadth-First Search Revisited:
  Enabling Bottom-Up Search*, 2013,
  [Berkeley technical report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-2.html),
  for distributed explicit-graph bottom-up communication structure.
- Notes 04, 05, 07, 10, 11, and 13 provide frontier/visited, variant, hardware,
  geometry, parent, and multi-source contracts used here.

## Current synthesis

Push and pull are dual evaluations of one existential next-frontier predicate.
Push enumerates evidence from the frontier; pull enumerates possible unvisited
destinations and searches for evidence.  Their exact frontier sets coincide
under complete edge enumeration and snapshot semantics, while their physical
work can differ radically.  Pull's defining prerequisite is not inverse edges
alone but an enumerable unvisited universe with exact incoming adjacency and
frontier membership—precisely the assumption many implicit/Cayley searches do
not have.
