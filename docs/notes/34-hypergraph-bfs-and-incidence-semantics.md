# Hypergraph BFS: one hyperedge step or two incidence steps?

"BFS on a hypergraph" is incomplete until a path and cost convention is named.
For an undirected hypergraph, moving between two vertices through one common
hyperedge can count as one hypergraph step, while the lossless incidence graph
represents it by two ordinary graph edges.  Directed hyperarcs introduce an
additional any-tail versus all-tail distinction that ordinary BFS cannot hide.

This note studies semantics and accounting only.  It does not design a
high-performance hypergraph traversal.

## Undirected Berge distance

Let a finite undirected hypergraph be

```text
H = (V,E),  E subset of nonempty subsets of V.
```

A Berge walk between original vertices alternates

```text
v_0, e_1, v_1, e_2, ..., e_k, v_k
```

with `v_(i-1),v_i in e_i`.  If each used hyperedge costs one, its length is
`k`.  The minimum such length defines the vertex-to-vertex Berge distance used
here.  Other hypergraph path definitions exist; they are not interchangeable.

For shortest distance, repeated removable sections can be eliminated, so a
minimum walk supplies the corresponding simple witness under the declared
Berge convention.

## Clique or two-section expansion

The clique expansion has vertex set `V` and connects distinct `u,v` whenever
some hyperedge contains both.  Then

```text
dist_clique(u,v) = dist_H(u,v)
```

for unweighted vertex endpoints: each hyperedge step projects to one clique
edge, and every clique edge names at least one witnessing hyperedge that lifts
back to one hypergraph step.

This scalar equality does not mean that the representation is lossless.  A
simple clique edge can forget:

- which hyperedge witnessed the transition;
- how many hyperedges contain the pair;
- the other vertices participating in the multiway relation;
- hyperedge labels, weights, or constraints.

If a returned path must name actual hyperedges, parent metadata needs a
witnessing hyperedge per step.

## Incidence or star expansion

The incidence graph is bipartite with nodes `V union E` and edges

```text
{v,e} iff v in e.
```

It preserves the complete incidence relation.  A one-hyperedge move
`u,e,v` becomes two ordinary graph edges:

```text
u -- e -- v.
```

Therefore, for original vertex endpoints,

```text
dist_incidence(u,v) = 2 * dist_H(u,v).
```

An ordinary incidence-graph BFS alternates:

```text
even graph depths: original vertices
odd graph depths:  hyperedge nodes.
```

Reporting its raw depth as the hypergraph distance silently doubles the answer.
One may instead group two incidence levels into one logical hypergraph level,
or define explicit asymmetric/weighted transition costs and use the algorithm
appropriate to those weights.  Merely renaming depth does not fix parent or
stopping logic unless the phase is recorded too.

## Minimal distance-scaling witness

For the single hyperedge

```text
e = {s,a,b,c},
```

all three non-source vertices have Berge distance one from `s`.  In the
incidence graph they are at graph distance two:

```text
s -> e -> a
s -> e -> b
s -> e -> c.
```

The clique expansion places them at graph distance one.  All three views are
internally correct; they answer different step-count questions.

## What a direct hypergraph frontier does

At logical depth `d`, a direct undirected traversal can be written as two
phases:

```text
active hyperedges = incident hyperedges of F_d not already settled
candidates        = all vertices in those hyperedges
F_(d+1)           = unique(candidates) minus B_d.
```

For distance-only reachability, an undirected hyperedge may be expanded when it
is first reached from a minimum-depth incident vertex.  Every other vertex in
that hyperedge then receives its best possible proposal through that hyperedge.
Re-expanding the same hyperedge from a later vertex cannot improve distances.

This shortcut has output qualifications.  If distinct hyperedges or distinct
incident predecessors define different shortest solutions, later same-depth
incidences may contribute:

- another shortest parent vertex;
- another parent hyperedge label;
- another shortest-path count contribution;
- a deterministic tie candidate.

"Hyperedge visited once" is therefore sufficient for one distance computation,
not automatically for every path-output contract.

## Overlap creates two duplicate dimensions

A candidate vertex may repeat because:

1. several frontier vertices enter the same hyperedge;
2. several different hyperedges contain the same source/candidate pair or lead
   to the same candidate.

Deduplicating hyperedge IDs and deduplicating vertex IDs solve different
problems.  A simple clique graph can collapse the second dimension before the
search begins, which preserves unweighted scalar distance but may destroy
labeled path multiplicity.

For example,

```text
e_1 = {s,t,a}
e_2 = {s,t,b}.
```

There are two distinct one-hyperedge witnesses from `s` to `t`.  An unweighted
simple clique expansion has one edge `{s,t}`.  Distance remains one; the count
of labeled hyperedge paths changes from two to one unless multiplicity metadata
is retained.

## Frontier and visited live in typed spaces

In incidence BFS there are two identity domains:

```text
visited_V subset V
visited_E subset E.
```

Combining them in one untyped key space risks a vertex/hyperedge ID collision.
It also obscures which empty frontier proves what:

- no active hyperedges after a completed vertex phase means no next vertex
  layer can be generated;
- an empty local vertex buffer before all active hyperedges are processed is
  not logical termination;
- target stopping for an original vertex occurs only after the phase that
  establishes its complete logical distance.

A checkpoint must record whether it is between `V->E`, between `E->V`, or at a
completed two-phase hypergraph boundary.

## Directed hypergraphs: OR and AND are different algorithms

A directed hyperarc may have a tail set `T(e)` and head set `H(e)`.  At least
two semantics occur:

- **OR-tail:** any reached tail vertex can traverse the hyperarc;
- **AND-tail:** every required tail vertex must be available before the arc
  activates.

OR-tail reachability can often be represented by directed incidence edges
`tail -> hyperedge -> head`, with the same phase/cost qualification as above.

AND-tail activation is not ordinary graph reachability.  For

```text
{a,b} -> {c},
```

source set `{a}` must not reach `c`, but an ordinary incidence path
`a -> e -> c` would claim that it does.  Correct AND semantics requires state
that records how many/which prerequisites have arrived, or a different
hyperpath formalism.  Calling the resulting process BFS does not make it the
ordinary shortest-path recurrence on vertices.

The cost model is also ambiguous: is activation time the maximum tail distance
plus one, the sum of acquisition costs, or a product-state schedule?  That must
be declared before distances can be compared.

## Multi-source semantics

With OR-tail undirected/Berge distance, multi-source BFS computes the minimum
number of hyperedges from any source, as usual.  If source labels or Voronoi
ties matter, a hyperedge reached simultaneously from several labels can spread
all of them to its incident vertices; early first-label stopping loses boundary
ties.

Under AND-tail semantics, multiple sources can cooperate to activate one
hyperarc.  The output is no longer necessarily the pointwise minimum of
independent single-source searches.  This is a fundamental change from ordinary
multi-source BFS, where seeding a union computes a minimum over sources.

## Cayley and puzzle boundary

A usual Cayley graph has binary transitions `g -> g*s`; a batch of generators
is not a hyperedge.  Treating all successors of `g` as one hyperedge would make
any two successors mutually adjacent in one logical step and change the word
metric.

Hypergraph semantics are appropriate only if the domain really contains a
multiway relation or synchronized prerequisite.  They should not be introduced
merely because GPU code processes a batch collectively.  Hardware batching
does not change graph arity.

## Work and storage accounting

For an explicit hypergraph, let

```text
M = sum_(e in E) |e|
```

be the number of incidences.  The incidence graph stores `M` bipartite edges
(or two directed entries).  A materialized clique expansion can require
roughly

```text
sum_e binomial(|e|,2)
```

pair occurrences before overlap deduplication.

These are representation sizes, not automatic runtime claims.  Direct
two-phase traversal may revisit incidences depending on hyperedge settlement,
output requirements, and frontier scheduling.  Useful counters separate:

- frontier-vertex to hyperedge incidences;
- unique newly active hyperedges;
- hyperedge to candidate-vertex incidences;
- unique candidate vertices;
- duplicate paths within one hyperedge versus across hyperedges;
- typed visited and frontier bytes for `V` and `E`.

## GPU and multi-GPU interpretation

The two phases have different degree distributions: vertex incidence degree
and hyperedge cardinality.  A single frontier size hides both.  Across devices,
ownership may be assigned separately to vertices and hyperedges, so one logical
hypergraph level can require two routing/reduction phases.

For correctness, the global boundary must ensure that every active hyperedge
and every emitted vertex candidate for the logical level has completed.  A
device-count change must migrate both typed visited domains and any partial
activation state, especially for AND-tail semantics.

These observations identify proof and measurement dimensions.  They do not
select a kernel, partition, or optimized representation.

## Rejected shortcuts

- **Incidence-graph BFS directly returns hyperedge distance.** Original vertex
  distances are doubled under unit incidence-edge costs.
- **Clique expansion is lossless because scalar distances agree.** It can erase
  hyperedge identity, multiplicity, and multiway membership.
- **One visited set is enough without typed IDs.** Vertices and hyperedges are
  distinct semantic domains.
- **Settling each hyperedge once preserves all outputs.** It preserves
  distance-only expansion under the stated undirected semantics, not all
  shortest parents or labeled counts.
- **Directed hyperarc traversal is ordinary incidence reachability.** AND-tail
  prerequisites give a direct counterexample.
- **A GPU batch of binary moves is a hyperedge.** Batching is physical;
  hyperedge arity is semantic.

## Sources

- The Electronic Journal of Combinatorics,
  [incidence graphs and Berge paths](https://www.combinatorics.org/ojs/index.php/eljc/article/download/v31i3p1/pdf/),
  states the correspondence between alternating hypergraph paths and paths in
  the bipartite incidence graph.
- *Mathematical Foundations of Hypergraph*,
  [clique and star expansions](https://link.springer.com/chapter/10.1007/978-981-99-0185-2_2),
  defines the two graph representations and their incidence construction.
- Notes 11, 13, 20, and 26 supply the output, multi-source, product-state, and
  phase-batching boundaries reused here.

## Current conclusion

For unweighted undirected Berge distance, clique expansion preserves
vertex-to-vertex distance while incidence expansion preserves structure and
doubles it.  Direct hypergraph BFS is naturally a typed two-phase traversal.
Once hyperedges are directed, labeled, weighted, or require all tails, neither
ordinary graph conversion nor the word BFS is safe without an explicit
semantic contract.
