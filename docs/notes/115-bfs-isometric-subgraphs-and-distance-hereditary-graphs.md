# BFS, isometric subgraphs, and distance-hereditary graphs

Deleting vertices cannot create a shorter route, but it can destroy every
shortest route and increase distances or disconnect the graph. Isometric and
distance-hereditary subgraphs describe the exceptional cases where retained
distances survive deletion exactly.

No recognition algorithm, pruning implementation, or optimized BFS is added.

## 1. What vertex deletion does to distance

Let `H` be a connected subgraph of `G`. Every path in `H` is also a path in `G`,
so for retained vertices

```text
d_H(u,v) >= d_G(u,v).
```

The inequality can be strict. In `C5`, delete one vertex and retain the induced
four-vertex path. Its endpoints have distance three in the path but distance two
in the original cycle through the deleted vertex.

If deletion disconnects `u` and `v`, their subgraph distance becomes infinite or
undefined under the selected convention. Thus "induced" means that all edges
between retained vertices are kept; it does not mean their original distances
are kept.

## 2. Isometric subgraphs

A subgraph `H` is isometric in `G` when

```text
d_H(u,v) = d_G(u,v)
```

for every pair of vertices of `H`. This is stronger than preserving distance
from one BFS root. A BFS tree is root-isometric but can have large stretch for
other pairs, as note 81 established.

One particular induced subgraph can be isometric even when the ambient graph has
many non-isometric induced subgraphs. Evidence for one chosen retained set does
not establish a hereditary graph class.

## 3. Distance-hereditary graphs

A connected graph is distance-hereditary when every connected induced subgraph
is isometric. Equivalently, every induced path in the graph is a shortest path
between its endpoints.

The path formulation is strong. An induced path has no chord among its own
vertices, but in a general graph its endpoints may have a shorter route through
vertices outside the path. Distance heredity forbids that hidden shortcut.

The `C5` example fails immediately: its induced four-vertex path is not
geodesic. Trees, complete graphs, and complete multipartite graphs are standard
distance-hereditary examples.

## 4. What happens to BFS layers

Let `H` be a connected induced subgraph of a distance-hereditary graph `G`, and
let root `s` belong to `H`. Then every retained vertex has the same distance from
`s` in both graphs. Therefore

```text
F_i(H,s) = F_i(G,s) intersect V(H).
```

The layer labels restrict exactly. Frontier cardinalities can shrink, and a
particular BFS parent edge can disappear, but at least one shortest retained
path remains for every retained vertex because `H` is connected and isometric.

This does not allow BFS to ignore deleted vertices before the retained set is
known. The theorem evaluates a declared induced subgraph; it does not identify
which vertices may safely be removed for an arbitrary target/output contract.

## 5. Pruning sequences

Every finite distance-hereditary graph can be built from one vertex by repeatedly
adding one of:

- a pendant vertex adjacent to exactly one existing vertex;
- a false twin with the same open neighborhood as an existing vertex and no
  edge to it;
- a true twin with the same closed neighborhood as an existing vertex and an
  edge to it.

Reversing these additions gives a pruning sequence. It is a structural
certificate when every step's pendant or twin relation is checked in the current
remaining graph, not merely in the original graph.

The sequence is not a BFS order. Pendant/twin removal follows neighborhood
structure, can mix root distances, and does not preserve a FIFO frontier
transaction step by step.

## 6. Forbidden induced obstructions

Distance-hereditary graphs are exactly the graphs with no induced:

- hole `C_k` for `k>=5`;
- house;
- gem, a four-vertex induced path plus a universal vertex;
- domino, two four-cycles sharing an edge.

A found obstruction is a compact negative certificate. Failure to find one in
a sample is not a positive certificate: the characterization quantifies over
all induced vertex sets, unless a complete recognition proof or pruning sequence
is supplied.

## 7. Distance-hereditary and chordal are different

The cycle `C4` is distance-hereditary but not chordal. Every connected induced
proper subgraph is a path of length at most three with no outside shortcut among
its retained vertices, while the full graph is an induced four-cycle.

The gem is chordal but not distance-hereditary. Its induced `P4` has endpoint
distance three internally, but the universal gem vertex gives a length-two path
in the full graph.

Graphs that are both chordal and distance-hereditary are called ptolemaic, but
neither property implies the other. Consequently, LexBFS/PEO evidence from note
114 cannot replace a distance-hereditary certificate.

## 8. Twins are not duplicate states

True or false twins have identical neighborhoods outside their pair, so their
successor rows may look redundant. They remain distinct graph vertices. Merging
them can change:

- whether both named targets are reported;
- path and predecessor multiplicities;
- the distance between the twins themselves, one for true twins and typically
  two for connected false twins;
- frontier cardinalities and visited-state counts;
- state-labeled path reconstruction.

A quotient may be valid for a narrower invariant with a proved lifting rule,
but twin structure alone does not authorize exact-state deduplication.

## 9. Cayley and Schreier examples

Vertex transitivity does not imply distance heredity. The standard undirected
cycle Cayley graph of `Z6` is not distance-hereditary because it is an induced
`C6`. With every nonidentity group element as a generator, the same group gives
`K6`, which is distance-hereditary. Thus the property depends on the generating
set, not only on the group.

The three-dimensional hypercube is a Cayley graph and is not
distance-hereditary: it contains an induced six-cycle. Small cube-like local
coordinates therefore do not provide the hereditary metric property.

A Schreier quotient changes which states become twins or pendants and requires
direct evidence. Directed positive alphabets need a separate asymmetric notion;
the standard theory here is undirected.

## 10. GPU and multi-GPU evidence boundary

A pruning sequence can encode repeated neighborhood structure compactly, but
ordinary exact BFS still has a distinct frontier/visited contract. Before using
twin structure in a GPU study, separate:

- validation of the pruning sequence against the exact current graph;
- storage or generation of pendant/twin neighborhood relations;
- any quotient and its output-specific lifting proof;
- exact BFS distance, parent, multiplicity, and state-identity validation;
- frontier expansion and distributed owner-routing throughput.

Sampling induced subgraphs can find a counterexample but cannot certify distance
heredity. In multi-GPU absence checks, every relevant vertex and edge owner must
participate before an obstruction or twin relation is declared absent. Structural
compression ratios and BFS throughput are separate measurements.

## Sources

- E. Howorka,
  [*A Characterization of Distance-Hereditary Graphs*](https://doi.org/10.1093/qmath/28.4.417),
  Quarterly Journal of Mathematics 28, 1977. Original metric and induced-path
  characterizations.
- H.-J. Bandelt and H. M. Mulder,
  [*Distance-Hereditary Graphs*](https://doi.org/10.1016/0095-8956(86)90043-2),
  Journal of Combinatorial Theory B 41, 1986. Pendant/twin construction and
  structural characterizations.
- H.-J. Bandelt and V. Chepoi,
  [*Metric Graph Theory and Geometry: A Survey*](https://doi.org/10.1090/conm/453/08795),
  Contemporary Mathematics 453, 2008. Isometric subgraphs and neighboring
  metric graph classes.

