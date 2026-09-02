# BFS, edge subdivision, topological minors, and integer weights

Subdividing an edge replaces one transition by a path of unit transitions.
Unlike contraction, this does not create a shortcut: it assigns a positive
integer length to the old transition. Between the original branch vertices,
ordinary BFS on the expanded graph is exactly a weighted shortest-path
calculation. The equivalence is semantic; it does not say that materializing
the expanded graph is a good implementation.

This note studies that equivalence and adds no implementation, optimizer,
benchmark, or GPU code.

## 1. Subdivision and its path map

Let every edge `e={u,v}` of an undirected graph `G` be replaced by an internally
vertex-disjoint `u-v` path `P_e` of positive integer length `ell(e)`. Call the
result `S(G,ell)`. The vertices inherited from `G` are branch vertices; all new
internal vertices have degree two.

Every `G` walk expands uniquely to an `S(G,ell)` walk by replacing each edge
with its path. Conversely, a walk between branch vertices can be shortened, if
necessary, to a simple path; whenever it enters the interior of `P_e`, degree
two forces it to leave through an endpoint of the same replacement path.
Suppressing those interiors recovers a `G` path.

Therefore, for branch vertices `u,v`,

```text
d_S(u,v) = min_(P:u->v in G) sum_(e in P) ell(e).
```

This is the core fact: unit BFS in the subdivision computes the positive
integer-weighted metric of the original graph, but only after the output is
interpreted on branch vertices.

## 2. Uniform subdivision scales branch distances

If every original edge is replaced by a path of the same length `k>=1`, then

```text
d_S(u,v) = k*d_G(u,v)                 for u,v in V(G).
```

For a branch source `s`, the branch part of every multiple-of-`k` sphere is
exactly the corresponding original sphere:

```text
S_(kj)^S(s) intersect V(G) = S_j^G(s).
```

The full expanded sphere is not merely an original sphere with a rescaled
index: it also contains transit vertices lying inside replacement paths. At a
transit vertex, the nearer endpoint of its old edge can depend on the source
and on competing routes through both endpoints.

## 3. Nonuniform subdivision changes the metric, not just its clock

With unequal `ell(e)`, an unweighted shortest path of `G` need not remain
shortest. A two-edge route whose lengths are `1+1` beats one old direct edge of
length `3`, even though the direct edge used fewer original hops.

Thus division of expanded BFS depth by one global constant is valid only for a
uniform subdivision. For nonuniform lengths, the result is the weighted metric
itself; there may be no integer "original BFS depth" to recover.

Positive integer lengths are essential for literal path expansion:

- zero-length edges would require identifying endpoints, not inserting a
  positive number of unit edges;
- negative lengths cannot be represented by unweighted path length;
- rational positive lengths can be scaled to integers only when a declared
  common denominator is acceptable, and that scaling can greatly enlarge the
  expanded graph.

## 4. Shortest paths and multiplicity

Because replacement paths are internally disjoint and unique per original
edge, weighted simple paths between branch vertices correspond to simple paths
in the subdivision. Consequently, the number of minimum-weight branch-to-branch
paths is preserved when paths are counted by original edge identity.

That statement changes if parallel edges are merged, replacement interiors are
shared, or path counting treats unit-edge transit sequences as distinct in some
other way. Edge identity and counting convention remain part of the contract.

Parents do not transfer one-for-one. An expanded BFS tree includes parents for
transit vertices, and suppressing them gives a weighted shortest-path tree only
on the branch vertices. Deterministic or shortlex label semantics additionally
require a declared mapping from each macro-edge to its original label.

## 5. Frontier profiles can change radically

Uniform subdivision preserves original sphere sizes as a subsequence of the
expanded branch-vertex layers, but it inserts new layers and many transit
vertices. It does not preserve frontier width, queue occupancy, edge work, or
diameter over the full expanded vertex set.

For example, subdivide every edge of `K_n` once, so each old edge has length
two. From one branch source:

- the first expanded layer has `n-1` transit vertices on incident edges;
- the second contains the other `n-1` branch vertices;
- the third contains the `binom(n-1,2)` transit vertices on edges between those
  branch vertices.

The original `K_n` had only one nonzero BFS layer of size `n-1`. Subdivision can
therefore turn a shallow dense graph into a deeper graph with a much larger
intermediate frontier even though all branch distances merely doubled.

## 6. Diameter and eccentricity need an endpoint convention

Under uniform length `k`, eccentricities and diameter restricted to branch
vertices scale by `k`. The diameter of the entire subdivision may involve
transit vertices, so it is a different maximum over a larger vertex set and
need not equal `k*diam(G)`.

The same distinction applies to radius, centers, distance sums, and
betweenness. "Preserved after subdivision" must say whether queries and demands
range over old branch vertices only or over every new vertex.

## 7. Topological minors do not preserve the host metric

A subdivision of `H` contained as a subgraph of `G` is a topological-minor model
of `H` in `G`. Its branch paths certify adjacency of the abstract pattern, but
their lengths may differ. Contracting each branch path recovers abstract
unit-length `H` and discards those lengths.

Hence three metrics must not be conflated:

1. unit distance in the abstract pattern `H`;
2. path length inside the selected subdivision model;
3. shortest distance in the whole host `G`, which may use edges outside the
   model and be still shorter.

Topological-minor containment alone supplies no equality among them. It is a
structural witness, not an isometric-embedding certificate.

## 8. Directed and labeled graphs

A positive-integer-weighted directed arc can be replaced by a directed chain.
Both directions of an undirected edge require the consistent undirected path,
whereas two directed arcs may have different lengths and separate interiors.

For labeled transition systems, internal unit edges are artificial phases of
one original move. Reporting them as original actions changes the language and
solution length. Replay must collapse the full chain back to the macro-edge and
reject incomplete chains.

## 9. Cayley and Schreier boundary

Subdividing Cayley edges usually destroys the original Cayley-state semantics:

- branch vertices still represent group elements or action states;
- transit vertices represent being partway through one generator application;
- branch and transit vertices generally have different degrees;
- uniform edge subdivision can change cycle parity and bipartiteness.

One can describe transit vertices with product/history state such as an active
generator and phase, but this is an expanded automaton, not the original group
or Schreier action. Exact deduplication must distinguish genuine group states
from phase states and must collapse completed chains before claiming an
original generator word.

## 10. Relation to weighted shortest-path algorithms

Explicit subdivision is a semantic reduction from positive integer weights to
unit edges. It can add

```text
sum_e (ell(e)-1)
```

vertices and `sum_e ell(e)` unit edges in place of `|E|` original edges.
Algorithms such as Dial's bucket method work with bounded nonnegative integer
weights without requiring this literal graph expansion. That is an algorithmic
alternative, not a different metric.

This note makes no claim about which representation is faster. It records the
objects that any later measurement must separate.

## 11. GPU and multi-GPU reporting boundary

For a subdivision-based probe, report separately:

- original branch vertices and weighted edges;
- expanded transit vertices and unit edges;
- logical weighted distance and physical expanded BFS depth;
- branch-only and all-vertex frontier profiles;
- macro-edge reconstruction and replay validation;
- owner placement of transit chains and cross-owner chain cuts;
- construction/materialization cost and traversal cost;
- weighted-graph and expanded-graph memory and throughput.

Long degree-two chains can add BFS rounds and owner crossings while doing no new
logical branching. Conversely, keeping them implicit changes the executed
algorithm from literal expanded-graph BFS. Both can be valid experiments, but
they are not the same workload.

## Sources

- R. Diestel,
  [*Graph Theory*, Chapter 1.7: Contraction and Minors](https://diestel-graph-theory.com/),
  for subdivisions, branch vertices, and topological-minor definitions.
- R. B. Dial,
  [*Algorithm 360: Shortest-Path Forest with Topological Ordering*](https://doi.org/10.1145/363269.363610),
  Communications of the ACM 12(11), 1969, for bounded integer-weight shortest
  paths without literal edge subdivision.
- Notes 16, 19, 20, 26, 29, 73, 87, 92, 99, and 124 provide this repository's
  action, label, product-state, graph-power, complexity, queue, history,
  reachability, geodesic-language, and contraction boundaries.

## Takeaway

Edge subdivision converts positive integer edge length into unit BFS depth.
Between original branch vertices this is an exact metric equivalence, and
uniform subdivision scales distances exactly. It does not preserve the full
vertex set, frontier profile, parent representation, Cayley state semantics, or
the host metric of a topological-minor model. Expanded BFS is therefore a clean
reasoning device only when branch/transit identity and macro-edge replay remain
explicit.
