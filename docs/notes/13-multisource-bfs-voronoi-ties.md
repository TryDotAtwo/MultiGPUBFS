# Multi-source BFS: distance to a set, Voronoi labels, and ties

"Multi-source BFS" is used for two different computations:

1. **one joint wavefront:** initialize a set `S` at distance zero and compute
   distance to the nearest source;
2. **many independent traversals:** compute a separate distance vector from
   every source, often sharing machine work but not merging semantics.

This note concerns the first.  It returns one scalar distance per vertex,

```text
d_S(v) = min_(s in S) dist(s,v),
```

not the full matrix `{dist(s,v) | s in S}`.  Merging independent traversals into
one visited set discards information and changes the problem.

## Metric balls from a source set

Initialize every distinct source at depth zero:

```text
B_S(0) = S
B_S(d+1) = B_S(d) union N_out(B_S(d))
F_S(d) = {v | d_S(v)=d}.
```

Equivalently,

```text
B_S(d) = union_(s in S) B_s(d).
```

For `d >= 1`, the corresponding statement for exact frontiers is **not** a
plain union:

```text
F_S(d) != union_(s in S) F_s(d)                 in general,

F_S(d) = (union_(s in S) F_s(d))
         minus (union_(s in S) B_s(d-1)).
```

The subtraction matters because a vertex can be exactly `d` steps from one
source but closer than `d` to another. On the path

```text
s -- x -- y -- t
```

with joint source set `{s,t}`, the independent depth-two frontiers are
`F_s(2)={y}` and `F_t(2)={x}`. Their union is `{x,y}`, but the joint wave has
already reached both vertices at depth one, so `F_{s,t}(2)` is empty. Balls
union directly because they express “within `d` of at least one source”;
frontiers must additionally exclude everything closer to any source.

The ordinary BFS induction applies unchanged: every vertex first discovered
from the complete depth-`d` wave has a path of length `d+1` from some source,
and no shorter source path remains unprocessed.

Sources can be disconnected from one another.  The joint traversal exhausts
the union of their reachable components.

## Virtual super-source equivalence

### A batch preserves source identity in visited

For the same path `s--x--y--t`, independent BFS rows and their joint minimum
are:

| Output | s | x | y | t |
|---|---:|---:|---:|---:|
| Distance from s | 0 | 1 | 2 | 3 |
| Distance from t | 3 | 2 | 1 | 0 |
| Joint distance from {s,t} | 0 | 1 | 1 | 0 |

At depth one, `x` is reached from `s`. At depth two, reaching `x` from `t`
is not redundant for the full matrix: it establishes a different entry,
`dist(t,x)=2`. One global visited bit for `x` would incorrectly suppress it.
Even retaining all *nearest* source labels does not help: `t` is not tied for
nearest at `x`, so the Voronoi output intentionally excludes it.

The independent-search semantic vertex is the pair `(source,vertex)`. Its
transition is

```text
(source,u) -> (source,v) whenever u -> v in the original graph.
```

This is a disjoint collection of source-tagged graph copies, initialized at
`(source,source)` for each requested source. Ordinary BFS on these pairs yields
the independent distance entries. Projecting away the source coordinate and
merging visited membership instead yields the joint nearest-source wave.

A bitset of sources stored at each vertex can encode pair membership without
materializing a tuple for each pair. But one new bit is a discovery for that
source; an old bit for another source does not reject it. A final membership
mask records which sources can reach the vertex, not when their bits first
arrived. Full distances require retaining that first-arrival depth per source
or an equivalent layer/history representation.

Thus shared adjacency reads or neighbor generation may reuse physical work
without merging the source-indexed novelty decisions. This is an output and
identity contract, not a batch implementation or a claim of GPU speedup. Eight
source-vertex distance entries in this example cannot be replaced by four
minima and nearest-source labels while preserving the requested matrix.

### Super-source construction for the joint minimum

Add a virtual vertex `q` with a zero-cost edge `q -> s` to every `s in S`.
Then shortest cost from `q` to `v` equals `d_S(v)`.

If the virtual edges are ordinary unit edges, all real distances become
`d_S(v)+1`.  Directly placing every source in `F_0` avoids this artificial
offset and preserves ordinary unit-edge BFS in the real graph.

The super-source is primarily a proof/modeling device.  It also makes clear
that duplicate source entries should not create duplicate vertices, unless
source labels/multiplicity are explicitly part of the requested path output.

## Distance is unique; nearest-source label may not be

Define the nearest-site set

```text
A(v) = argmin_(s in S) dist(s,v).
```

`d_S(v)` is one number.  `A(v)` can contain several sources.  A single owner or
color label

```text
label(v) in A(v)
```

requires a tie rule.

This produces a graph Voronoi interpretation:

- sources are sites;
- vertices uniquely closest to one site lie in its unambiguous cell;
- equidistant vertices form a tie/bisector region;
- assigning every tie to one site turns the set-valued diagram into a
  partition.

The partition is not determined by distances alone.

## Three legitimate tie contracts

### Arbitrary valid nearest source

Let first arrival win.  The distance and chosen label are valid, but labels can
change with source order, frontier order, adjacency order, thread scheduling,
GPU count, or message timing.

This is sufficient when only distance matters and labels are diagnostic.  It is
not deterministic output.

### Canonical source order

Choose the minimum source under a declared total order among all nearest
sources:

```text
key(v) = min_lex (distance_from_source, source_id).
```

For unit edges, a level-synchronous implementation can reduce all candidate
labels for a child before committing the next layer.  The source ID becomes
semantic metadata and must be stable across executions.

### Set-valued nearest sources

Retain the full `A(v)` rather than forcing a partition.  This preserves all
ties, but source-set metadata can become large.  It is the appropriate output
when every equidistant facility/source matters.

These contracts have identical scalar distances and different memory,
communication, and validation requirements.

## Equal-distance label improvement must propagate

Suppose a vertex is first assigned `(distance=5, source=9)` and later receives
`(distance=5, source=2)` under a minimum-source tie rule.  Its distance does not
change, but its semantic label does.

If descendants were already colored from source 9, merely updating this vertex
is insufficient: source 2 may also be the canonical winner for an entire
downstream tie region.  A correct strategy must either:

- resolve all source-label ties for a complete layer before expanding it; or
- reactivate/repropagate equal-distance label improvements.

This resembles relaxation even though hop distances themselves retain ordinary
BFS finality.  A visited boolean protects distance but cannot by itself enforce
canonical Voronoi labels.

## Source label and parent are separate choices

For a non-source vertex `v`, possible shortest predecessors are

```text
P(v) = {u | (u,v) in E and d_S(u)+1=d_S(v)}.
```

A parent can belong to a different nearest-source label if labels were assigned
inconsistently on ties.  To construct a forest rooted in the selected sources,
require

```text
label(parent(v)) = label(v)
```

in addition to the depth/edge condition.

Even after the source label is canonical, multiple same-label parents may
remain.  A deterministic forest needs a further parent/move tie-break.  One
possible global key is

```text
(distance, source_id, parent_identity, move_label).
```

Distance, source ownership, and parent choice should not be stored or validated
as though they were one field.

## Connectivity of Voronoi cells

If every labeled vertex retains a shortest parent with the same label, following
parents gives a path entirely inside its cell to the source.  Each nonempty cell
is therefore connected in the parent-forest sense (directed reachable from its
site under the original edge orientation).

An arbitrary independent tie assignment need not preserve this property.  It
can color an equidistant vertex for one source while coloring all of its usable
shortest predecessors for another.  The labels remain pointwise valid but do
not form a coherent rooted forest.

Thus "Voronoi partition" may mean merely nearest-site membership or the stronger
path-connected labeled decomposition.  The tie procedure determines which is
returned.

## Directed graphs: from sources versus to facilities

Forward multi-source BFS computes

```text
min_(s in S) dist(s,v).
```

This answers "which source can reach `v` most quickly?"

Many facility queries ask the opposite:

```text
min_(s in S) dist(v,s),
```

meaning "which facility can `v` reach?"  On a directed graph, seed the
facilities in the **reverse graph** to compute this quantity.  The two answers
coincide only under appropriate symmetry/undirectedness.

Using the wrong orientation can produce plausible colors with the wrong
meaning, exactly as in backward bidirectional BFS.

## Boundary edges and nearest-source adjacency

For an undirected graph with single-valued labels, an edge whose endpoints have
different labels crosses a discrete Voronoi boundary.  It can be used to derive
site-neighbor relationships or candidate inter-site paths:

```text
dist(s_i,u) + 1 + dist(v,s_j)
```

for a boundary edge `(u,v)` with labels `s_i` and `s_j`.

But the boundary graph depends on tie assignment.  Set-valued ties preserve
ambiguity; arbitrary coloring can move boundary edges inside the equidistant
region without changing any nearest distance.

Therefore downstream algorithms using cell adjacency must inherit the same tie
contract, rather than treating labels as incidental BFS parents.

## Multi-source versus multi-target search

To find the closest member of a target set `T` from one start `s`, ordinary
forward BFS from `s` can stop when it first discovers any target under the usual
contract.

Alternatively, reverse multi-source BFS seeded by `T` precomputes distance and
next-step guidance **to the nearest target** for every state that can reach one.
This is useful when many start queries share a fixed target set.

The two computations have different amortization and output:

- one forward query explores only what that start needs;
- reverse multi-source preprocessing constructs a reusable distance field;
- changing `T` changes the field;
- tie rules determine which equally close target a policy selects.

## Multi-source Cayley geometry

In a Cayley graph, a source set produces a union of translated metric balls.
For a group with an invariant word metric,

```text
dist(s,v) = word_length(s^-1 v)
```

under a consistent left/right convention.  The nearest-source field compares
these translated word lengths.

### Many source rows from one table: transport, not a minimum

In the full right Cayley graph with edges `x -> xg`, left translation
`L_(s^-1)(x)=s^-1 x` maps that edge to `s^-1 x -> (s^-1 x)g`. It is a
label-preserving graph automorphism taking `s` to identity `e`. Therefore

```text
dist(s,v) = dist(e,s^-1 v).
```

This direct path bijection holds also for directed, non-inverse-closed move
alphabets, using directed distance and allowing unreachable pairs. Computing
the group inverse `s^-1` for coordinates does not require that this inverse
itself be an allowed single move.

A complete identity-distance table can consequently answer every source row
by transforming the query endpoint. Source identity has not been dropped: it
is retained in the argument `s^-1 v`. This is different from joint multi-source
BFS, which replaces all rows by their pointwise minimum.

On the directed six-cycle `Z_6` with only move `+1`, the identity table is
`0,1,2,3,4,5`. The query from source `3` to vertex `1` becomes
`1-3 mod 6 = 4` and has distance four. Joint BFS from sources `{0,3}` instead
returns distance one at vertex `1`. Reuse of a table preserves the four;
merging the waves intentionally loses it.

The distinction qualifies the generic source-pair model above: it describes
the independent-query information, not a universal necessity to execute or
store each pair separately. With known automorphisms, rows can be represented
implicitly by one table and coordinate maps. Explicitly emitting every row
still has the requested matrix's output size.

Limits: a depth-limited identity table answers only covered relative elements;
absence outside that radius is not unreachability. A transitive Schreier action
does not automatically provide the required automorphisms of the graph for a
fixed generator alphabet (note 16). Also, the path bijection preserves labeled
words, but an implementation-specific first-parent tree or a state-ID tie rule
need not be preserved by a separate run with a different root. No new lookup
implementation or performance claim is made here.

### A table from identity is not a table to identity

For the same full right Cayley graph and the same allowed alphabet, define

```text
T_from(x) = dist(e,x)
T_to(x)   = dist(x,e) = T_from(x^-1).
```

The equality follows by left translation, not by assuming the graph is
undirected. On `Z_6` with only `+1`, `T_from(1)=1` but `T_to(1)=5`.
Therefore querying the outward table at `x` instead of `x^-1` can return the
wrong directed distance to the goal.

For a witness, an allowed word `w` from identity to `x^-1` can be replayed
unchanged from `x`: `x w = e`. In contrast, reversing the stored word from
identity to `x` and inverting its labels may introduce moves absent from the
allowed alphabet. The coordinate inverse is an algebraic operation; it is not
permission to execute inverse moves at unit cost.

Reverse-graph BFS from the goal is another way to build `T_to` directly at
key `x`, provided predecessor generation and forward replay labels are correct.
These are two distinct table conventions. If a table is radius-limited,
coverage must be checked at the key belonging to that convention. None of
these formulas assumes an arbitrary Schreier state has a well-defined group
inverse.

Applications include distance to a goal orbit or a collection of solved
representatives.  But replacing one fixed goal by an orbit/set changes the
query:

```text
distance to specific goal  !=  minimum distance to any symmetric representative
```

They coincide only if the problem itself identifies those representatives or a
lifting/orientation proof restores the requested solution.  Multi-source seeding
is not automatically a valid symmetry quotient.

For reversible moves, seeding goal states and expanding inverse moves creates a
lookup distance-to-goal table.  Parent/next-move orientation must still be
defined for forward replay from an arbitrary query state.

## Path counts with multiple sources

Several counting questions are possible:

- number of shortest paths from any source;
- number from the selected canonical nearest source;
- counts grouped by every tied nearest source;
- number of source labels rather than number of paths.

For total paths from a source set, initialize `sigma(s)=1` for each distinct
source and use the predecessor recurrence.  At a vertex tied between sources,
the count combines paths from all of them.

If only paths from the canonical source should count, predecessor accumulation
must be restricted by the selected label.  Distance correctness alone cannot
distinguish these outputs.

## Early stopping and partial Voronoi diagrams

Multi-source traversal may stop when:

- a particular query vertex obtains its nearest distance;
- waves from two specified source classes meet;
- a depth/radius limit is completed;
- every reachable component is exhausted.

A query vertex's distance can be final before its canonical label or full
nearest-source set is complete if equal-distance claims remain outstanding.
Likewise, a partial radius field is not a complete Voronoi partition.

The run result should therefore state which of these is final:

```text
distance only
one arbitrary nearest label
canonical nearest label
all nearest labels
parent forest / boundary graph.
```

## Parallel and GPU semantics

Distance-only joint BFS is friendly to ordinary exact visited claims:

- all sources are inserted at depth zero before expansion;
- same-layer waves claiming one child can choose any winner;
- one accepted child enters the next frontier.

Richer outputs add work:

- arbitrary label: winner carries source ID;
- canonical label: reduce equal-depth source IDs before expansion or propagate
  improvements;
- all labels: variable-size per-state tie sets;
- canonical parent: reduce parent/move keys within the selected label;
- cell boundaries: retain or reconstruct cross-label edges.

Atomic first-winner is exact for scalar distance and nondeterministic for label.
Atomic minimum over a packed `(distance,source)` key can express a canonical
order only if packing, comparison, and update propagation are exact and
overflow-safe.  This is a semantic observation, not a recommendation to build
such a backend.

## Multi-GPU: source label is not physical owner

Two unrelated notions of ownership coexist:

- **semantic source label:** which nearest site owns the Voronoi cell;
- **physical state owner:** which rank/GPU stores authoritative visited/metadata.

They should not be conflated.  Hashing a state to GPU 3 does not mean source 3
is nearest; assigning all of one Voronoi cell to one GPU may create severe and
evolving load imbalance.

At an authoritative state owner, candidates from several source ranks can have:

- different distances: keep the smaller distance;
- equal distance, different source labels: apply the declared tie contract;
- equal distance and label, different parents: apply parent/DAG contract.

For level-synchronous unit BFS, distance never decreases after the exact layer
is committed, but canonical labels may still require complete same-level
convergence before owner-local frontier expansion.

Distributed termination for distance-only BFS is ordinary global frontier
exhaustion.  Termination for canonical labels additionally needs all equal-depth
label messages that could improve committed metadata to be delivered or ruled
out.

## Bidirectional generalization

Bidirectional BFS can start from a source set `S` forward and a target set `T`
backward.  It computes

```text
D = min_(s in S, t in T) dist(s,t).
```

The same upper/lower-bound proof applies to the two multi-source balls.  Meeting
metadata now also chooses a source-target pair.  The distance can be unique
while several pairs and connector paths tie.

Stopping at the optimal scalar `D` does not necessarily enumerate every
minimizing `(s,t)` pair.  Pair labels and all-path output require completing the
relevant equality boundary, just as in notes 08 and 11.

## Counterexamples to common assumptions

### Source order is only cosmetic

It is cosmetic for scalar distance and observable for first-winner labels,
parents, cell boundaries, and returned paths.

### A vertex's final distance means its canonical label is final

An equal-distance claim from a smaller source ID can arrive later.  Without
layer reduction or repropagation, descendants can retain noncanonical labels.

### Multi-source BFS returns all source distances

It returns their pointwise minimum. The full matrix needs source-specific
information: independent source dimensions, repeated/batched traversals, or a
proved symmetry transport such as the full Cayley identity-table construction
above. The minimum field alone cannot reconstruct the lost rows.

### A Voronoi cell is automatically connected

Pointwise arbitrary tie choices can break a coherent same-label parent chain.
Connectivity follows from a tie/parent propagation rule, not merely from every
label being one of the nearest sites.

### Sources are GPU partitions

Semantic proximity regions and balanced physical ownership solve different
problems.  Equating them can preserve labels while destroying load balance.

### Seeding symmetry-equivalent goals preserves a fixed-goal distance

It computes minimum distance to the orbit.  That is a different metric query
unless the application quotient/lifting contract says otherwise.

## Audit checklist

1. Does "multi-source" mean nearest-source minimum or many independent distance
   vectors?
2. Is the required scalar `min_s dist(s,v)` or `min_s dist(v,s)` on a directed
   graph?
3. Are duplicate source states one site or multiple labels?
4. Is nearest-site output arbitrary, canonical, or set-valued?
5. Can equal-distance label improvements arrive after distance commitment, and
   how do they propagate?
6. Must parent chains stay inside the selected source cell?
7. What defines distinct paths and source multiplicity?
8. Are Voronoi boundaries/adjacencies downstream semantic output?
9. Is a goal set/orbit genuinely the requested target, or a changed problem?
10. Are semantic source labels separated from physical GPU/rank ownership?

## Sources

- Martin Erwig, *The Graph Voronoi Diagram with Applications*, Networks 36(3),
  2000,
  [paper](https://web.engr.oregonstate.edu/~erwig/papers/GraphVoronoi_Networks00.pdf),
  for shortest-path proximity regions and graph Voronoi structure.
- Pedro F. Felzenszwalb and Daniel P. Huttenlocher,
  *Distance Transforms of Sampled Functions*, Theory of Computing 8 (2012),
  [article](https://theoryofcomputing.org/articles/v008a019/), for the broader
  distance-transform viewpoint (with metrics/algorithms beyond graph hop BFS).
- Notes 05, 11, and 12 provide multi-source initialization, predecessor/path
  contracts, and zero-cost super-source semantics.

## Current synthesis

Joint multi-source BFS computes distance to a set.  That scalar field is
canonical; coloring it by a nearest source is extra structure.  Tie handling
can be arbitrary, canonical, or set-valued, and canonical equal-distance label
changes may need propagation even though BFS distances are already final.
Parent forests, path counts, Voronoi boundaries, reverse facility queries, and
physical GPU ownership each add separate semantics.  "Put all sources in the
queue" is correct only after the requested output is stated.
